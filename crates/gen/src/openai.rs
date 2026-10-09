//! OpenAI Images API (`gpt-image-*`): generations as JSON, edits and variations as multipart.

use base64::Engine as _;
use serde_json::{Value, json};

use crate::{Auth, GenError, HttpRequest, HttpTransport, ImageProvider, MAX_INPUT, MAX_RESPONSE, Result, UreqTransport, validate_prompt};

pub(crate) const OPENAI_BASE: &str = "https://api.openai.com/v1/images";
/// The model used when a command does not name one.
pub const OPENAI_DEFAULT_MODEL: &str = "gpt-image-1";
const PROVIDER: &str = "OpenAI";
const SIZES: [&str; 4] = ["1024x1024", "1024x1536", "1536x1024", "auto"];

/// OpenAI implementation. `key` remains process memory only; never serialize this type.
pub struct OpenAiProvider<T: HttpTransport = UreqTransport> {
    key: String,
    model: String,
    pub(crate) transport: T,
}

impl OpenAiProvider<UreqTransport> {
    /// Reads `OPENAI_API_KEY`.
    pub fn from_env(model: Option<&str>) -> Result<Self> {
        let key = crate::env_var(crate::OPENAI_KEY_ENV).ok_or(GenError::MissingProviderKey { provider: PROVIDER, variables: crate::OPENAI_KEY_ENV })?;
        Self::new(key, model.unwrap_or(OPENAI_DEFAULT_MODEL), UreqTransport)
    }
}

impl<T: HttpTransport> OpenAiProvider<T> {
    pub fn new(key: String, model: &str, transport: T) -> Result<Self> {
        if key.trim().is_empty() {
            return Err(GenError::MissingProviderKey { provider: PROVIDER, variables: crate::OPENAI_KEY_ENV });
        }
        if !model.starts_with("gpt-image") || model.len() > 80 || !model.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.') {
            return Err(GenError::Invalid("`model` must be a gpt-image model for OpenAI; set PHOTOCRAFT_GEN_PROVIDER=gemini to use Gemini"));
        }
        Ok(Self { key, model: model.to_owned(), transport })
    }

    fn request(&self, path: &str, content_type: &str, body: &[u8]) -> Result<Vec<u8>> {
        let url = format!("{OPENAI_BASE}{path}");
        let response = self.transport.post(&HttpRequest { url: &url, auth: Auth::Bearer(&self.key), content_type, body })?;
        if !(200..300).contains(&response.status) {
            return Err(GenError::Http { provider: PROVIDER, status: response.status, code: None });
        }
        if response.body.len() > MAX_RESPONSE {
            return Err(GenError::Service("response too large"));
        }
        let value: Value = serde_json::from_slice(&response.body).map_err(|_| GenError::Service("invalid JSON response"))?;
        let data = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|v| v.get("b64_json"))
            .and_then(Value::as_str)
            .ok_or(GenError::Service("response missing b64_json image"))?;
        if data.len() > MAX_RESPONSE {
            return Err(GenError::Service("image too large"));
        }
        base64::engine::general_purpose::STANDARD.decode(data).map_err(|_| GenError::Service("invalid base64 image"))
    }

    fn multipart(&self, image: &[u8], mask: Option<&[u8]>, prompt: &str, size: &str) -> Result<Vec<u8>> {
        validate(prompt, size)?;
        if image.is_empty() || image.len() > MAX_INPUT {
            return Err(GenError::Invalid("PNG image is empty or too large"));
        }
        if mask.is_some_and(|m| m.is_empty() || m.len() > MAX_INPUT) {
            return Err(GenError::Invalid("PNG mask is empty or too large"));
        }
        // A boundary absent from binary input. A constant is safe after checking both buffers.
        let mut boundary = "photocraft-boundary-1".to_owned();
        while prompt.contains(&boundary)
            || image.windows(boundary.len()).any(|w| w == boundary.as_bytes())
            || mask.is_some_and(|m| m.windows(boundary.len()).any(|w| w == boundary.as_bytes()))
        {
            boundary.push('x');
        }
        let mut body = Vec::new();
        field(&mut body, &boundary, "model", self.model.as_bytes());
        field(&mut body, &boundary, "prompt", prompt.as_bytes());
        field(&mut body, &boundary, "size", size.as_bytes());
        field(&mut body, &boundary, "output_format", b"png");
        file(&mut body, &boundary, "image", image);
        if let Some(m) = mask {
            file(&mut body, &boundary, "mask", m);
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        self.request("/edits", &format!("multipart/form-data; boundary={boundary}"), &body)
    }
}

impl<T: HttpTransport> ImageProvider for OpenAiProvider<T> {
    fn name(&self) -> &'static str {
        PROVIDER
    }
    fn size_for(&self, width: u32, height: u32) -> String {
        if width > height.saturating_mul(5) / 4 {
            "1536x1024"
        } else if height > width.saturating_mul(5) / 4 {
            "1024x1536"
        } else {
            "1024x1024"
        }
        .to_owned()
    }
    fn accepts(&self, size: &str, width: u32, height: u32) -> bool {
        match size {
            "1024x1024" => (width, height) == (1024, 1024),
            "1024x1536" => (width, height) == (1024, 1536),
            "1536x1024" => (width, height) == (1536, 1024),
            "auto" => width > 0 && height > 0,
            _ => false,
        }
    }
    fn generate(&self, prompt: &str, size: &str) -> Result<Vec<u8>> {
        validate(prompt, size)?;
        let body = json!({"model":self.model,"prompt":prompt,"size":size,"output_format":"png"});
        let bytes = serde_json::to_vec(&body).map_err(|_| GenError::Service("could not prepare request"))?;
        self.request("/generations", "application/json", &bytes)
    }
    fn edit(&self, image: &[u8], mask: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        self.multipart(image, Some(mask), prompt, size)
    }
    fn variations(&self, image: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        // GPT image models support prompt-guided variations through edits.
        self.multipart(image, None, prompt, size)
    }
}

fn validate(prompt: &str, size: &str) -> Result<()> {
    validate_prompt(prompt)?;
    if !SIZES.contains(&size) {
        return Err(GenError::Invalid("OpenAI sizes are 1024x1024, 1024x1536, 1536x1024 or auto"));
    }
    Ok(())
}

fn field(body: &mut Vec<u8>, boundary: &str, name: &str, value: &[u8]) {
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes());
    body.extend_from_slice(value);
    body.extend_from_slice(b"\r\n");
}

fn file(body: &mut Vec<u8>, boundary: &str, name: &str, value: &[u8]) {
    body.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"image.png\"\r\nContent-Type: image/png\r\n\r\n").as_bytes(),
    );
    body.extend_from_slice(value);
    body.extend_from_slice(b"\r\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Mock, contains, ok};

    #[test]
    fn requests_are_mockable_and_have_expected_shapes() {
        let p = OpenAiProvider::new("secret".into(), "gpt-image-1", Mock::new(Vec::new())).unwrap();
        assert_eq!(p.generate("cat", "1024x1024").unwrap(), [1, 2, 3]);
        p.edit(b"png", b"mask", "fill", "1024x1024").unwrap();
        p.variations(b"png", "similar", "1024x1024").unwrap();
        let r = p.transport.requests.lock().unwrap();
        assert_eq!(r.len(), 3);
        assert_eq!(r[0].url, "https://api.openai.com/v1/images/generations");
        assert_eq!(r[0].auth, ("Authorization".to_owned(), "Bearer secret".to_owned()));
        assert_eq!(r[0].content_type, "application/json");
        assert_eq!(
            serde_json::from_slice::<Value>(&r[0].body).unwrap(),
            json!({"model":"gpt-image-1","prompt":"cat","size":"1024x1024","output_format":"png"})
        );
        assert!(r[1].content_type.starts_with("multipart/form-data; boundary="));
        assert!(contains(&r[1].body, b"name=\"image\""));
        assert!(contains(&r[1].body, b"name=\"mask\""));
        assert!(contains(&r[1].body, b"1024x1024"));
        assert_eq!(r[2].url, "https://api.openai.com/v1/images/edits");
        assert!(!contains(&r[2].body, b"name=\"mask\""));
    }

    #[test]
    fn response_failures_and_credentials_are_redacted() {
        let secret = "sentinel-test-secret";
        for reply in [
            ok(401, br#"{"error":{"message":"Incorrect API key provided: sentinel-test-secret"}}"#),
            ok(429, b"{}"),
            ok(500, b""),
            ok(200, b"{"),
            ok(200, br#"{"data":[{"b64_json":"%%%"}]}"#),
            ok(200, br#"{"data":[]}"#),
            Err(GenError::Service("HTTPS connection failed")),
        ] {
            let p = OpenAiProvider::new(secret.into(), "gpt-image-1", Mock::new(vec![reply])).unwrap();
            let e = p.generate("cat", "1024x1024").unwrap_err();
            assert!(!format!("{e} {e:?}").contains(secret));
        }
        let p = OpenAiProvider::new(secret.into(), "gpt-image-1", Mock::new(vec![ok(401, b"{}")])).unwrap();
        let e = p.generate("cat", "1024x1024").unwrap_err();
        assert!(matches!(e, GenError::Http { provider: "OpenAI", status: 401, .. }));
        assert!(e.to_string().contains("API key"), "{e}");
        let missing = OpenAiProvider::new(" ".into(), "gpt-image-1", Mock::new(Vec::new())).err().unwrap();
        assert!(matches!(missing, GenError::MissingProviderKey { .. }));
        assert!(missing.to_string().starts_with("Generative AI is off"));
        let p = OpenAiProvider::new(secret.into(), "gpt-image-1", Mock::new(Vec::new())).unwrap();
        assert!(matches!(p.generate(" ", "1024x1024"), Err(GenError::Invalid(_))));
        assert!(matches!(p.generate("cat", "99x99"), Err(GenError::Invalid(_))));
        assert!(matches!(p.generate("cat", "16:9"), Err(GenError::Invalid(_))));
        assert!(p.transport.requests.lock().unwrap().is_empty());
        assert!(OpenAiProvider::new(secret.into(), "gemini-nano-banana-2.1", Mock::new(Vec::new())).is_err());
    }

    #[test]
    fn sizes_follow_the_area_shape_and_outputs_must_match_exactly() {
        let p = OpenAiProvider::new("k".into(), "gpt-image-1", Mock::new(Vec::new())).unwrap();
        assert_eq!(p.size_for(100, 100), "1024x1024");
        assert_eq!(p.size_for(300, 100), "1536x1024");
        assert_eq!(p.size_for(100, 300), "1024x1536");
        assert!(p.accepts("1024x1536", 1024, 1536));
        assert!(!p.accepts("1024x1536", 4, 6));
        assert!(p.accepts("auto", 1536, 1024));
        assert!(!p.accepts("auto", 0, 1024));
    }
}
