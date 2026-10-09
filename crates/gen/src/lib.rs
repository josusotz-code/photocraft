//! Optional image generation providers: Google Gemini and OpenAI.
//!
//! Nothing here runs unless a key is in the environment (`GEMINI_API_KEY` or `GOOGLE_API_KEY`,
//! `OPENAI_API_KEY`). Keys live in process memory only: they travel in one HTTPS header and
//! never appear in URLs, request bodies, errors, logs or saved documents. Errors never quote a
//! response body either, because a service's free-text message can echo the request.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod gemini;
mod openai;
mod pixels;

pub use gemini::{ASPECT_RATIOS, GEMINI_DEFAULT_MODEL, GeminiProvider};
pub use openai::{OPENAI_DEFAULT_MODEL, OpenAiProvider};

pub(crate) const MAX_RESPONSE: usize = 32 * 1024 * 1024;
pub(crate) const MAX_INPUT: usize = 20 * 1024 * 1024;

/// Google AI Studio key for Gemini (the name Google's docs and SDKs use).
pub const GEMINI_KEY_ENV: &str = "GEMINI_API_KEY";
/// Accepted as a fallback for [`GEMINI_KEY_ENV`]; Google's docs name both.
pub const GOOGLE_KEY_ENV: &str = "GOOGLE_API_KEY";
pub const OPENAI_KEY_ENV: &str = "OPENAI_API_KEY";
/// `gemini` or `openai`: picks the provider when both keys are set (Gemini wins otherwise).
pub const PROVIDER_ENV: &str = "PHOTOCRAFT_GEN_PROVIDER";

#[derive(Debug, thiserror::Error)]
pub enum GenError {
    #[error("Generative AI is off: set GEMINI_API_KEY (Google Gemini) or OPENAI_API_KEY (OpenAI) in the app's environment")]
    MissingKey,
    #[error("Generative AI is off: {provider} needs {variables} in the app's environment")]
    MissingProviderKey { provider: &'static str, variables: &'static str },
    #[error("PHOTOCRAFT_GEN_PROVIDER must be `gemini` or `openai`")]
    UnknownProvider,
    #[error("invalid image request: {0}")]
    Invalid(&'static str),
    #[error("image service unavailable: {0}")]
    Service(&'static str),
    #[error("{provider} declined the request ({reason}); change the prompt or image and try again")]
    Blocked { provider: &'static str, reason: &'static str },
    #[error("{provider} returned HTTP {status}{}: {}", code_suffix(.code), http_hint(.status))]
    Http { provider: &'static str, status: u16, code: Option<&'static str> },
}

fn code_suffix(code: &Option<&'static str>) -> String {
    code.map(|c| format!(" ({c})")).unwrap_or_default()
}

fn http_hint(status: &u16) -> &'static str {
    match status {
        400 => "the request was rejected; check the model, prompt, image and API key",
        401 => "the API key is missing, invalid or expired",
        402 => "the prepaid credit balance is used up",
        403 => "the API key has no permission for this model or project",
        404 => "the model was not found; check the `model` parameter",
        408 | 504 => "the request timed out; try again",
        429 => "rate limit or quota exceeded; wait a moment and try again",
        500..=599 => "the service had a problem; try again later",
        _ => "the request failed",
    }
}

pub type Result<T> = std::result::Result<T, GenError>;

/// An image service. Inputs are PNG bytes; a mask's transparent pixels mark the area to
/// repaint. Results are PNG or JPEG bytes.
pub trait ImageProvider: Send + Sync {
    /// Short display name for progress messages, such as `Gemini`.
    fn name(&self) -> &'static str;
    /// The `size` to request when the result will fill a `width` x `height` area.
    fn size_for(&self, width: u32, height: u32) -> String;
    /// Whether a returned `width` x `height` image answers a request for `size`.
    fn accepts(&self, size: &str, width: u32, height: u32) -> bool;
    fn generate(&self, prompt: &str, size: &str) -> Result<Vec<u8>>;
    fn edit(&self, image: &[u8], mask: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>>;
    fn variations(&self, image: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>>;
}

impl<P: ImageProvider + ?Sized> ImageProvider for Box<P> {
    fn name(&self) -> &'static str {
        (**self).name()
    }
    fn size_for(&self, width: u32, height: u32) -> String {
        (**self).size_for(width, height)
    }
    fn accepts(&self, size: &str, width: u32, height: u32) -> bool {
        (**self).accepts(size, width, height)
    }
    fn generate(&self, prompt: &str, size: &str) -> Result<Vec<u8>> {
        (**self).generate(prompt, size)
    }
    fn edit(&self, image: &[u8], mask: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        (**self).edit(image, mask, prompt, size)
    }
    fn variations(&self, image: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        (**self).variations(image, prompt, size)
    }
}

/// How a request proves who it is. `Debug` never prints the key.
#[derive(Clone, Copy)]
pub enum Auth<'a> {
    /// `Authorization: Bearer <key>` (OpenAI).
    Bearer(&'a str),
    /// `x-goog-api-key: <key>` (Gemini API).
    GoogleApiKey(&'a str),
}

impl Auth<'_> {
    /// The header name and value to send.
    pub fn header(&self) -> (&'static str, String) {
        match self {
            Auth::Bearer(key) => ("Authorization", format!("Bearer {key}")),
            Auth::GoogleApiKey(key) => ("x-goog-api-key", (*key).to_owned()),
        }
    }
}

impl std::fmt::Debug for Auth<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Auth::Bearer(_) => "Bearer(<redacted>)",
            Auth::GoogleApiKey(_) => "GoogleApiKey(<redacted>)",
        })
    }
}

/// One HTTPS POST. `Debug` shows the endpoint and sizes, never the key or the body.
#[derive(Clone, Copy)]
pub struct HttpRequest<'a> {
    pub url: &'a str,
    pub auth: Auth<'a>,
    pub content_type: &'a str,
    pub body: &'a [u8],
}

impl std::fmt::Debug for HttpRequest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpRequest")
            .field("url", &self.url)
            .field("auth", &self.auth)
            .field("content_type", &self.content_type)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// Any HTTP status comes back as a response; providers turn non-2xx into [`GenError::Http`].
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Injectable transport; tests can inspect a request without sending it.
pub trait HttpTransport: Send + Sync {
    fn post(&self, request: &HttpRequest<'_>) -> Result<HttpResponse>;
}

/// The only endpoints a request may go to.
#[cfg(not(target_arch = "wasm32"))]
const ALLOWED_ENDPOINTS: [&str; 2] = ["https://api.openai.com/v1/images/", gemini::GEMINI_BASE];

/// Blocking HTTPS transport, intended for an engine background job. Native only.
pub struct UreqTransport;

#[cfg(not(target_arch = "wasm32"))]
impl HttpTransport for UreqTransport {
    fn post(&self, request: &HttpRequest<'_>) -> Result<HttpResponse> {
        use std::io::Read as _;
        use std::time::Duration;
        if !ALLOWED_ENDPOINTS.iter().any(|base| request.url.starts_with(base)) {
            return Err(GenError::Invalid("endpoint not allowed"));
        }
        let config =
            ureq::Agent::config_builder().timeout_global(Some(Duration::from_secs(180))).https_only(true).max_redirects(0).http_status_as_error(false).build();
        let agent = config.new_agent();
        let (name, value) = request.auth.header();
        let mut response = match agent.post(request.url).header(name, value).header("Content-Type", request.content_type).send(request.body) {
            Ok(r) => r,
            Err(ureq::Error::Timeout(_)) => return Err(GenError::Service("the request timed out")),
            Err(_) => return Err(GenError::Service("HTTPS connection failed")),
        };
        let status = response.status().as_u16();
        let mut body = Vec::new();
        response.body_mut().as_reader().take((MAX_RESPONSE + 1) as u64).read_to_end(&mut body).map_err(|_| GenError::Service("could not read response"))?;
        if body.len() > MAX_RESPONSE {
            return Err(GenError::Service("response too large"));
        }
        Ok(HttpResponse { status, body })
    }
}

#[cfg(target_arch = "wasm32")]
impl HttpTransport for UreqTransport {
    fn post(&self, _: &HttpRequest<'_>) -> Result<HttpResponse> {
        Err(GenError::Service("image generation requires the desktop app"))
    }
}

/// Which service the image commands use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderKind {
    Gemini,
    OpenAi,
}

/// Reads an environment variable; unset, non-Unicode and blank values count as unset.
pub fn env_var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

pub(crate) fn gemini_key(env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let get = |name: &str| env(name).filter(|v| !v.trim().is_empty());
    get(GEMINI_KEY_ENV).or_else(|| get(GOOGLE_KEY_ENV))
}

/// Picks the provider from the environment (`env` looks variables up, so tests need not touch
/// the process environment): `PHOTOCRAFT_GEN_PROVIDER` decides when set; otherwise Gemini
/// when its key is set, then OpenAI; with no key the commands stay off.
pub fn choose_provider(env: &dyn Fn(&str) -> Option<String>) -> Result<ProviderKind> {
    let get = |name: &str| env(name).filter(|v| !v.trim().is_empty());
    let gemini = gemini_key(env).is_some();
    let openai = get(OPENAI_KEY_ENV).is_some();
    match get(PROVIDER_ENV).map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        Some("gemini") if gemini => Ok(ProviderKind::Gemini),
        Some("gemini") => Err(GenError::MissingProviderKey { provider: "Gemini", variables: "GEMINI_API_KEY (or GOOGLE_API_KEY)" }),
        Some("openai") if openai => Ok(ProviderKind::OpenAi),
        Some("openai") => Err(GenError::MissingProviderKey { provider: "OpenAI", variables: OPENAI_KEY_ENV }),
        Some(_) => Err(GenError::UnknownProvider),
        None if gemini => Ok(ProviderKind::Gemini),
        None if openai => Ok(ProviderKind::OpenAi),
        None => Err(GenError::MissingKey),
    }
}

/// The provider the environment selects, with `model` or that provider's default model.
pub fn provider_from_env(model: Option<&str>) -> Result<Box<dyn ImageProvider>> {
    build_provider(&env_var, model, UreqTransport)
}

/// [`provider_from_env`] with an injectable environment and transport.
pub fn build_provider<T: HttpTransport + 'static>(env: &dyn Fn(&str) -> Option<String>, model: Option<&str>, transport: T) -> Result<Box<dyn ImageProvider>> {
    match choose_provider(env)? {
        ProviderKind::Gemini => {
            let key = gemini_key(env).ok_or(GenError::MissingKey)?;
            Ok(Box::new(GeminiProvider::new(key, model.unwrap_or(GEMINI_DEFAULT_MODEL), transport)?))
        }
        ProviderKind::OpenAi => {
            let key = env(OPENAI_KEY_ENV).filter(|v| !v.trim().is_empty()).ok_or(GenError::MissingKey)?;
            Ok(Box::new(OpenAiProvider::new(key, model.unwrap_or(OPENAI_DEFAULT_MODEL), transport)?))
        }
    }
}

/// Whether `size` is one any provider understands: the OpenAI sizes, `auto`, or a Gemini
/// aspect ratio. The selected provider still rejects the ones it cannot serve.
pub fn is_known_size(size: &str) -> bool {
    gemini::aspect_ratio(size).is_ok()
}

pub(crate) fn validate_prompt(prompt: &str) -> Result<()> {
    if prompt.trim().is_empty() || prompt.len() > 32_000 {
        return Err(GenError::Invalid("prompt is empty or too long"));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use std::sync::Mutex;

    pub struct Recorded {
        pub url: String,
        pub auth: (String, String),
        pub content_type: String,
        pub body: Vec<u8>,
    }

    pub struct Mock {
        pub requests: Mutex<Vec<Recorded>>,
        replies: Mutex<Vec<Result<HttpResponse>>>,
    }

    impl Mock {
        pub fn new(replies: Vec<Result<HttpResponse>>) -> Self {
            Self { requests: Mutex::new(Vec::new()), replies: Mutex::new(replies) }
        }
    }

    impl HttpTransport for Mock {
        fn post(&self, r: &HttpRequest<'_>) -> Result<HttpResponse> {
            let (name, value) = r.auth.header();
            self.requests.lock().map_err(|_| GenError::Service("test lock"))?.push(Recorded {
                url: r.url.into(),
                auth: (name.into(), value),
                content_type: r.content_type.into(),
                body: r.body.to_vec(),
            });
            let mut replies = self.replies.lock().map_err(|_| GenError::Service("test lock"))?;
            if replies.is_empty() { ok(200, br#"{"data":[{"b64_json":"AQID"}]}"#) } else { replies.remove(0) }
        }
    }

    pub fn ok(status: u16, body: &[u8]) -> Result<HttpResponse> {
        Ok(HttpResponse { status, body: body.to_vec() })
    }

    pub fn contains(haystack: &[u8], needle: &[u8]) -> bool {
        haystack.windows(needle.len()).any(|w| w == needle)
    }

    pub fn png_rgba(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        let mut data = Vec::new();
        for y in 0..h {
            for x in 0..w {
                data.extend_from_slice(&f(x, y));
            }
        }
        let image = photocraft_codecs::Image::from_u8(w, h, photocraft_codecs::ChannelLayout::Rgba, data).unwrap();
        photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn provider_selection_prefers_gemini_and_honors_the_override() {
        use ProviderKind::*;
        let pick = |pairs: &[(&str, &str)]| choose_provider(&env(pairs));
        assert_eq!(pick(&[("GEMINI_API_KEY", "g")]).unwrap(), Gemini);
        assert_eq!(pick(&[("GOOGLE_API_KEY", "g")]).unwrap(), Gemini);
        assert_eq!(pick(&[("OPENAI_API_KEY", "o")]).unwrap(), OpenAi);
        assert_eq!(pick(&[("OPENAI_API_KEY", "o"), ("GEMINI_API_KEY", "g")]).unwrap(), Gemini);
        assert_eq!(pick(&[("OPENAI_API_KEY", "o"), ("GEMINI_API_KEY", "g"), ("PHOTOCRAFT_GEN_PROVIDER", "openai")]).unwrap(), OpenAi);
        assert_eq!(pick(&[("OPENAI_API_KEY", "o"), ("GEMINI_API_KEY", "g"), ("PHOTOCRAFT_GEN_PROVIDER", " Gemini ")]).unwrap(), Gemini);
        assert_eq!(pick(&[("OPENAI_API_KEY", "o"), ("GEMINI_API_KEY", "  "), ("PHOTOCRAFT_GEN_PROVIDER", "")]).unwrap(), OpenAi);
        let off = pick(&[]).unwrap_err().to_string();
        assert!(off.contains("GEMINI_API_KEY") && off.contains("OPENAI_API_KEY"), "{off}");
        let wrong = pick(&[("OPENAI_API_KEY", "o"), ("PHOTOCRAFT_GEN_PROVIDER", "gemini")]).unwrap_err().to_string();
        assert!(wrong.contains("Gemini needs GEMINI_API_KEY"), "{wrong}");
        let wrong = pick(&[("GEMINI_API_KEY", "g"), ("PHOTOCRAFT_GEN_PROVIDER", "openai")]).unwrap_err().to_string();
        assert!(wrong.contains("OpenAI needs OPENAI_API_KEY"), "{wrong}");
        assert!(matches!(pick(&[("GEMINI_API_KEY", "g"), ("PHOTOCRAFT_GEN_PROVIDER", "dall-e")]), Err(GenError::UnknownProvider)));
    }

    #[test]
    fn built_provider_uses_the_chosen_key_and_default_model() {
        use test_support::{Mock, ok};
        let png = test_support::png_rgba(1, 1, |_, _| [0, 0, 0, 255]);
        let reply = serde_json::json!({"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png)}}]}}]});
        let mock = std::sync::Arc::new(Mock::new(vec![ok(200, reply.to_string().as_bytes())]));
        struct Shared(std::sync::Arc<Mock>);
        impl HttpTransport for Shared {
            fn post(&self, r: &HttpRequest<'_>) -> Result<HttpResponse> {
                self.0.post(r)
            }
        }
        let both = env(&[("GEMINI_API_KEY", "gem"), ("GOOGLE_API_KEY", "goo"), ("OPENAI_API_KEY", "oai")]);
        let p = build_provider(&both, None, Shared(mock.clone())).unwrap();
        assert_eq!(p.name(), "Gemini");
        p.generate("x", "1:1").unwrap();
        let r = mock.requests.lock().unwrap();
        assert_eq!(r[0].auth, ("x-goog-api-key".to_owned(), "gem".to_owned()));
        assert!(r[0].url.contains("/models/gemini-nano-banana-2.1:generateContent"));
        drop(r);
        let google_only = env(&[("GOOGLE_API_KEY", "goo")]);
        assert_eq!(build_provider(&google_only, Some("gemini-3-pro-image"), Shared(mock.clone())).unwrap().name(), "Gemini");
        let openai = env(&[("OPENAI_API_KEY", "oai")]);
        assert_eq!(build_provider(&openai, None, Shared(mock.clone())).unwrap().name(), "OpenAI");
        // A model for the other provider is an actionable error, not a silent switch.
        let e = build_provider(&both, Some("gpt-image-1"), Shared(mock.clone())).err().unwrap().to_string();
        assert!(e.contains("PHOTOCRAFT_GEN_PROVIDER=openai"), "{e}");
        assert!(build_provider(&env(&[]), None, Shared(mock)).is_err());
    }

    #[test]
    fn transport_refuses_unlisted_endpoints() {
        let r = HttpRequest { url: "https://example.com/v1beta/models/x", auth: Auth::GoogleApiKey("k"), content_type: "application/json", body: b"{}" };
        assert!(matches!(UreqTransport.post(&r), Err(GenError::Invalid(_))));
        let r = HttpRequest { url: "http://generativelanguage.googleapis.com/v1beta/models/x", ..r };
        assert!(matches!(UreqTransport.post(&r), Err(GenError::Invalid(_))));
    }

    #[test]
    fn known_sizes_cover_both_providers() {
        for size in ["1024x1024", "1024x1536", "1536x1024", "auto", "16:9", "21:9"] {
            assert!(is_known_size(size), "{size}");
        }
        for size in ["", "1:4", "512x512", "16:9 "] {
            assert!(!is_known_size(size), "{size}");
        }
    }
}
