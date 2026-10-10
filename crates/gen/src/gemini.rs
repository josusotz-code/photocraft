//! Google Gemini image models through the Gemini API's `generateContent` method.
//!
//! Request and response shapes follow the official reference
//! (<https://ai.google.dev/api/generate-content>) and the image-generation guide
//! (<https://ai.google.dev/gemini-api/docs/image-generation>): parts carry text or
//! `inlineData {mimeType, data}` (base64), `generationConfig.responseModalities` asks for
//! `["TEXT","IMAGE"]`, and `generationConfig.imageConfig.aspectRatio` sets the shape. Images
//! come back as `inlineData` parts of `candidates[0].content.parts`; parts marked
//! `thought: true` are interim "thought images" and are skipped.
//!
//! Gemini has no mask input. Generative Fill follows the guide's "Inpainting (semantic
//! masking)" pattern: the region is sent with its context, a black-and-white mask goes along as
//! a second image, and the instruction says to change only the white area and keep everything
//! else. The engine then composites the result through the selection, so pixels outside the
//! selection can never change even if the model repaints more than asked.

use base64::Engine as _;
use serde_json::{Value, json};

use crate::{Auth, GenError, HttpRequest, HttpTransport, ImageProvider, MAX_INPUT, MAX_RESPONSE, Result, UreqTransport, pixels, validate_prompt};

pub(crate) const GEMINI_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models/";
/// The model used when a command does not name one: Nano Banana 2.1, the Gemini API's current
/// recommended image model (stable, October 2026).
pub const GEMINI_DEFAULT_MODEL: &str = "gemini-nano-banana-2.1";
const PROVIDER: &str = "Gemini";
const KEY_VARIABLES: &str = "GEMINI_API_KEY (or GOOGLE_API_KEY)";

/// Aspect ratios every current Gemini image model accepts in `imageConfig.aspectRatio`.
/// (1:4, 4:1, 1:8 and 8:1 exist on newer models only, so they are not offered.)
pub const ASPECT_RATIOS: [(&str, u32, u32); 10] = [
    ("1:1", 1, 1),
    ("2:3", 2, 3),
    ("3:2", 3, 2),
    ("3:4", 3, 4),
    ("4:3", 4, 3),
    ("4:5", 4, 5),
    ("5:4", 5, 4),
    ("9:16", 9, 16),
    ("16:9", 16, 9),
    ("21:9", 21, 9),
];

/// `|ln(returned ratio) − ln(requested ratio)|` up to which an image counts as that shape. The
/// documented 1K sizes differ from their nominal ratio by at most 0.011 (21:9 is 1584x672).
const RATIO_TOLERANCE: f64 = 0.03;

/// Gemini implementation. `key` remains process memory only; never serialize this type.
pub struct GeminiProvider<T: HttpTransport = UreqTransport> {
    key: String,
    model: String,
    pub(crate) transport: T,
}

impl GeminiProvider<UreqTransport> {
    /// Reads `GEMINI_API_KEY`, then `GOOGLE_API_KEY`.
    pub fn from_env(model: Option<&str>) -> Result<Self> {
        let key = crate::gemini_key(&crate::env_var).ok_or(GenError::MissingProviderKey { provider: PROVIDER, variables: KEY_VARIABLES })?;
        Self::new(key, model.unwrap_or(GEMINI_DEFAULT_MODEL), UreqTransport)
    }
}

impl<T: HttpTransport> GeminiProvider<T> {
    pub fn new(key: String, model: &str, transport: T) -> Result<Self> {
        if key.trim().is_empty() {
            return Err(GenError::MissingProviderKey { provider: PROVIDER, variables: KEY_VARIABLES });
        }
        // The id becomes a URL path segment: allow only the characters model ids use.
        if !model.starts_with("gemini-") || model.len() > 80 || !model.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.') {
            return Err(GenError::Invalid(
                "`model` must be a Gemini image model such as gemini-nano-banana-2.1; set PHOTOCRAFT_GEN_PROVIDER=openai to use a gpt-image model",
            ));
        }
        Ok(Self { key, model: model.to_owned(), transport })
    }

    fn request(&self, parts: Vec<Value>, size: &str) -> Result<Vec<u8>> {
        let mut config = json!({"responseModalities": ["TEXT", "IMAGE"]});
        if let Some(ratio) = aspect_ratio(size)? {
            config["imageConfig"] = json!({"aspectRatio": ratio});
        }
        let body = json!({"contents": [{"parts": parts}], "generationConfig": config});
        let bytes = serde_json::to_vec(&body).map_err(|_| GenError::Service("could not prepare request"))?;
        let url = format!("{GEMINI_BASE}{}:generateContent", self.model);
        let response = self.transport.post(&HttpRequest { url: &url, auth: Auth::GoogleApiKey(&self.key), content_type: "application/json", body: &bytes })?;
        if !(200..300).contains(&response.status) {
            return Err(GenError::Http { provider: PROVIDER, status: response.status, code: error_code(&response.body) });
        }
        parse_response(&response.body)
    }
}

impl<T: HttpTransport> ImageProvider for GeminiProvider<T> {
    fn name(&self) -> &'static str {
        PROVIDER
    }
    /// The supported aspect ratio closest to the area's shape.
    fn size_for(&self, width: u32, height: u32) -> String {
        let shape = ln_ratio(width, height);
        ASPECT_RATIOS.iter().min_by(|a, b| (ln_ratio(a.1, a.2) - shape).abs().total_cmp(&(ln_ratio(b.1, b.2) - shape).abs())).map_or("1:1", |r| r.0).to_owned()
    }
    /// Gemini picks the pixel size itself (1024x1024 at 1:1, 848x1264 at 2:3, ...), so a
    /// result is accepted when its shape matches the requested ratio.
    fn accepts(&self, size: &str, width: u32, height: u32) -> bool {
        if width == 0 || height == 0 {
            return false;
        }
        match aspect_ratio(size) {
            Ok(Some(ratio)) => {
                let wanted = ASPECT_RATIOS.iter().find(|r| r.0 == ratio).map_or(0.0, |r| ln_ratio(r.1, r.2));
                width.min(height) >= 256 && (ln_ratio(width, height) - wanted).abs() <= RATIO_TOLERANCE
            }
            Ok(None) => width.min(height) >= 64,
            Err(_) => false,
        }
    }
    fn generate(&self, prompt: &str, size: &str) -> Result<Vec<u8>> {
        validate_prompt(prompt)?;
        aspect_ratio(size)?;
        self.request(vec![json!({"text": prompt})], size)
    }
    fn edit(&self, image: &[u8], mask: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        validate_prompt(prompt)?;
        aspect_ratio(size)?;
        check_input(image, "PNG image is empty or too large")?;
        check_input(mask, "PNG mask is empty or too large")?;
        let prepared = pixels::prepare_edit(image, mask)?;
        let text = edit_instruction(prompt, prepared.bounds);
        self.request(vec![json!({"text": text}), inline_png(&prepared.image), inline_png(&prepared.mask)], size)
    }
    fn variations(&self, image: &[u8], prompt: &str, size: &str) -> Result<Vec<u8>> {
        validate_prompt(prompt)?;
        aspect_ratio(size)?;
        check_input(image, "PNG image is empty or too large")?;
        self.request(vec![json!({"text": prompt}), inline_png(image)], size)
    }
}

/// The instruction that stands in for a mask, after the guide's semantic-masking template.
fn edit_instruction(prompt: &str, [y0, x0, y1, x1]: [u32; 4]) -> String {
    format!(
        "Using the first image, change only the area that is white in the second image (a mask of the same size; \
         black means keep) to: {prompt}\n\
         Keep everything else in the image exactly the same, preserving the original style, lighting, and composition. \
         Mid-gray pixels inside the white area are empty canvas: fill them so the scene continues seamlessly. \
         The area to change lies within the box [ymin, xmin, ymax, xmax] = [{y0}, {x0}, {y1}, {x1}] on a 0-1000 scale. \
         Return the whole image with the same framing as the first image."
    )
}

fn inline_png(bytes: &[u8]) -> Value {
    json!({"inlineData": {"mimeType": "image/png", "data": base64::engine::general_purpose::STANDARD.encode(bytes)}})
}

fn check_input(bytes: &[u8], why: &'static str) -> Result<()> {
    if bytes.is_empty() || bytes.len() > MAX_INPUT { Err(GenError::Invalid(why)) } else { Ok(()) }
}

fn ln_ratio(width: u32, height: u32) -> f64 {
    (f64::from(width.max(1)) / f64::from(height.max(1))).ln()
}

/// Maps a command `size` to `imageConfig.aspectRatio`; `auto` lets the model choose (it follows
/// the reference image, or its default for text-only prompts).
pub(crate) fn aspect_ratio(size: &str) -> Result<Option<&'static str>> {
    match size {
        "auto" => Ok(None),
        "1024x1024" => Ok(Some("1:1")),
        "1024x1536" => Ok(Some("2:3")),
        "1536x1024" => Ok(Some("3:2")),
        other => ASPECT_RATIOS
            .iter()
            .find(|r| r.0 == other)
            .map(|r| Some(r.0))
            .ok_or(GenError::Invalid("Gemini sizes are 1024x1024, 1024x1536, 1536x1024, auto or an aspect ratio such as 16:9")),
    }
}

/// Finds the last final image in a `GenerateContentResponse`, or explains why there is none.
fn parse_response(body: &[u8]) -> Result<Vec<u8>> {
    if body.len() > MAX_RESPONSE {
        return Err(GenError::Service("response too large"));
    }
    let value: Value = serde_json::from_slice(body).map_err(|_| GenError::Service("invalid JSON response from Gemini"))?;
    if let Some(reason) = value.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(GenError::Blocked { provider: PROVIDER, reason: block_reason(reason) });
    }
    let candidate = value.get("candidates").and_then(Value::as_array).and_then(|c| c.first()).ok_or(GenError::Service("Gemini returned no image"))?;
    let parts = candidate.pointer("/content/parts").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    let image = parts.iter().rev().filter(|p| !p.get("thought").and_then(Value::as_bool).unwrap_or(false)).find_map(|p| {
        let blob = p.get("inlineData").or_else(|| p.get("inline_data"))?;
        let mime = blob.get("mimeType").or_else(|| blob.get("mime_type")).and_then(Value::as_str)?;
        if !mime.starts_with("image/") {
            return None;
        }
        blob.get("data").and_then(Value::as_str)
    });
    let Some(data) = image else {
        return Err(no_image(candidate.get("finishReason").and_then(Value::as_str).unwrap_or("")));
    };
    if data.len() > MAX_RESPONSE {
        return Err(GenError::Service("image too large"));
    }
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).map_err(|_| GenError::Service("invalid base64 image from Gemini"))?;
    if !(bytes.starts_with(b"\x89PNG\r\n\x1a\n") || bytes.starts_with(&[0xFF, 0xD8, 0xFF])) {
        return Err(GenError::Service("Gemini returned an image that is neither PNG nor JPEG"));
    }
    Ok(bytes)
}

fn block_reason(reason: &str) -> &'static str {
    match reason {
        "SAFETY" | "IMAGE_SAFETY" => "safety filters",
        "PROHIBITED_CONTENT" => "prohibited content",
        "BLOCKLIST" => "blocked terms",
        _ => "the prompt was blocked",
    }
}

fn no_image(finish_reason: &str) -> GenError {
    let blocked = |reason| GenError::Blocked { provider: PROVIDER, reason };
    match finish_reason {
        "SAFETY" | "IMAGE_SAFETY" => blocked("safety filters"),
        "PROHIBITED_CONTENT" | "IMAGE_PROHIBITED_CONTENT" => blocked("prohibited content"),
        "RECITATION" | "IMAGE_RECITATION" => blocked("too close to existing work"),
        "BLOCKLIST" => blocked("blocked terms"),
        "SPII" => blocked("personal information"),
        "NO_IMAGE" | "IMAGE_OTHER" => GenError::Service("Gemini did not produce an image; try rephrasing the prompt"),
        "MAX_TOKENS" => GenError::Service("Gemini stopped before producing an image"),
        _ => GenError::Service("Gemini returned no image"),
    }
}

/// The machine-readable status of a Google API error body, from a fixed list so the
/// free-text `message` (which can quote the request) is never shown.
fn error_code(body: &[u8]) -> Option<&'static str> {
    const CODES: [&str; 19] = [
        "INVALID_ARGUMENT",
        "FAILED_PRECONDITION",
        "OUT_OF_RANGE",
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
        "NOT_FOUND",
        "RESOURCE_EXHAUSTED",
        "CANCELLED",
        "INTERNAL",
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "invalid_request",
        "authentication",
        "payment_required",
        "permission_denied",
        "model_not_found",
        "rate_limit_exceeded",
        "quota_exceeded",
        "service_unavailable",
    ];
    let value: Value = serde_json::from_slice(body).ok()?;
    let error = value.get("error")?;
    [error.get("status"), error.get("code")].into_iter().flatten().filter_map(Value::as_str).find_map(|s| CODES.iter().find(|c| **c == s).copied())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{Mock, contains, ok, png_rgba};

    const MODEL_URL: &str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-nano-banana-2.1:generateContent";

    fn image_reply(png: &[u8]) -> Vec<u8> {
        let data = base64::engine::general_purpose::STANDARD.encode(png);
        serde_json::to_vec(&json!({
            "candidates": [{
                "content": {"role": "model", "parts": [
                    {"text": "Here is your image"},
                    {"inlineData": {"mimeType": "image/png", "data": data}}
                ]},
                "finishReason": "STOP"
            }]
        }))
        .unwrap()
    }
    fn provider(replies: Vec<Result<crate::HttpResponse>>) -> GeminiProvider<Mock> {
        GeminiProvider::new("secret".into(), GEMINI_DEFAULT_MODEL, Mock::new(replies)).unwrap()
    }
    fn body(p: &GeminiProvider<Mock>, i: usize) -> Value {
        serde_json::from_slice(&p.transport.requests.lock().unwrap()[i].body).unwrap()
    }
    fn inline_bytes(part: &Value) -> Vec<u8> {
        assert_eq!(part["inlineData"]["mimeType"], "image/png");
        base64::engine::general_purpose::STANDARD.decode(part["inlineData"]["data"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn generate_posts_documented_request_shape() {
        let png = png_rgba(4, 4, |_, _| [255, 0, 0, 255]);
        let p = provider(vec![ok(200, &image_reply(&png))]);
        assert_eq!(p.generate("a red square", "1536x1024").unwrap(), png);
        let r = p.transport.requests.lock().unwrap();
        assert_eq!(r[0].url, MODEL_URL);
        assert_eq!(r[0].auth, ("x-goog-api-key".to_owned(), "secret".to_owned()));
        assert_eq!(r[0].content_type, "application/json");
        drop(r);
        assert_eq!(
            body(&p, 0),
            json!({
                "contents": [{"parts": [{"text": "a red square"}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"], "imageConfig": {"aspectRatio": "3:2"}}
            })
        );
    }

    #[test]
    fn auto_size_omits_image_config_and_ratios_pass_through() {
        let png = png_rgba(2, 2, |_, _| [0, 0, 0, 255]);
        let p = provider(vec![ok(200, &image_reply(&png)), ok(200, &image_reply(&png))]);
        p.generate("x", "auto").unwrap();
        p.generate("x", "16:9").unwrap();
        assert!(body(&p, 0)["generationConfig"].get("imageConfig").is_none());
        assert_eq!(body(&p, 1)["generationConfig"]["imageConfig"]["aspectRatio"], "16:9");
    }

    #[test]
    fn edit_sends_flattened_image_black_and_white_mask_and_instruction() {
        // Left half transparent in the mask (repaint), right half opaque (keep); the image has
        // a transparent column that must arrive flattened onto mid-gray.
        let image = png_rgba(4, 2, |x, _| if x == 0 { [0, 0, 0, 0] } else { [10, 200, 30, 255] });
        let mask = png_rgba(4, 2, |x, _| [255, 255, 255, if x < 2 { 0 } else { 255 }]);
        let result = png_rgba(4, 2, |_, _| [1, 2, 3, 255]);
        let p = provider(vec![ok(200, &image_reply(&result))]);
        assert_eq!(p.edit(&image, &mask, "a blue door", "1:1").unwrap(), result);
        let b = body(&p, 0);
        let parts = b["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts.len(), 3);
        let text = parts[0]["text"].as_str().unwrap();
        assert!(text.contains("to: a blue door"));
        assert!(text.contains("Keep everything else in the image exactly the same"));
        assert!(text.contains("[0, 0, 1000, 500]"), "{text}");
        let sent =
            photocraft_codecs::decode(&inline_bytes(&parts[1])).unwrap().convert(photocraft_codecs::ChannelLayout::Rgb, photocraft_codecs::SampleType::U8);
        assert_eq!(&sent.data()[..3], &[128, 128, 128]);
        assert_eq!(&sent.data()[3..6], &[10, 200, 30]);
        let sent_mask =
            photocraft_codecs::decode(&inline_bytes(&parts[2])).unwrap().convert(photocraft_codecs::ChannelLayout::Gray, photocraft_codecs::SampleType::U8);
        assert_eq!(sent_mask.data(), &[255, 255, 0, 0, 255, 255, 0, 0]);
        assert_eq!(b["generationConfig"]["imageConfig"]["aspectRatio"], "1:1");
    }

    #[test]
    fn edit_rejects_unusable_inputs_before_sending() {
        let p = provider(Vec::new());
        let image = png_rgba(4, 4, |_, _| [0, 0, 0, 255]);
        let keep_all = png_rgba(4, 4, |_, _| [0, 0, 0, 255]);
        let other_size = png_rgba(2, 2, |_, _| [0, 0, 0, 0]);
        assert!(matches!(p.edit(&image, &keep_all, "x", "1:1"), Err(GenError::Invalid(_))));
        assert!(matches!(p.edit(&image, &other_size, "x", "1:1"), Err(GenError::Invalid(_))));
        assert!(matches!(p.edit(b"not a png", &keep_all, "x", "1:1"), Err(GenError::Invalid(_))));
        assert!(matches!(p.edit(&[], &keep_all, "x", "1:1"), Err(GenError::Invalid(_))));
        assert!(matches!(p.edit(&image, &other_size, "x", "7:3"), Err(GenError::Invalid(_))));
        assert!(matches!(p.variations(&image, " ", "1:1"), Err(GenError::Invalid(_))));
        assert!(p.transport.requests.lock().unwrap().is_empty());
    }

    #[test]
    fn variations_send_prompt_and_unchanged_image() {
        let image = png_rgba(3, 3, |_, _| [9, 9, 9, 255]);
        let p = provider(vec![ok(200, &image_reply(&image))]);
        p.variations(&image, "similar", "4:3").unwrap();
        let b = body(&p, 0);
        let parts = b["contents"][0]["parts"].as_array().unwrap();
        assert_eq!(parts[0], json!({"text": "similar"}));
        assert_eq!(inline_bytes(&parts[1]), image);
        assert_eq!(b["generationConfig"]["imageConfig"]["aspectRatio"], "4:3");
    }

    #[test]
    fn responses_skip_thought_images_and_accept_jpeg() {
        let thought = png_rgba(1, 1, |_, _| [0, 0, 0, 255]);
        let jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 0, 0];
        let b64 = |b: &[u8]| base64::engine::general_purpose::STANDARD.encode(b);
        let reply = json!({"candidates": [{"content": {"parts": [
            {"inlineData": {"mimeType": "image/jpeg", "data": b64(&jpeg)}},
            {"thought": true, "inlineData": {"mimeType": "image/png", "data": b64(&thought)}},
            {"text": "done"}
        ]}, "finishReason": "STOP"}]});
        assert_eq!(parse_response(&serde_json::to_vec(&reply).unwrap()).unwrap(), jpeg);
    }

    #[test]
    fn missing_images_safety_blocks_and_finish_reasons_are_actionable() {
        let cases: [(&[u8], &str); 9] = [
            (br#"{"promptFeedback":{"blockReason":"SAFETY"}}"#, "declined the request (safety filters)"),
            (br#"{"promptFeedback":{"blockReason":"PROHIBITED_CONTENT"}}"#, "prohibited content"),
            (br#"{"candidates":[{"content":{"parts":[{"text":"no"}]},"finishReason":"IMAGE_SAFETY"}]}"#, "safety filters"),
            (br#"{"candidates":[{"content":{"parts":[]},"finishReason":"IMAGE_RECITATION"}]}"#, "existing work"),
            (br#"{"candidates":[{"finishReason":"NO_IMAGE"}]}"#, "try rephrasing"),
            (br#"{"candidates":[{"content":{"parts":[{"text":"only text"}]},"finishReason":"STOP"}]}"#, "returned no image"),
            (br#"{"candidates":[]}"#, "returned no image"),
            (br#"{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"%%%"}}]}}]}"#, "invalid base64"),
            (br#"{"candidates":[{"content":{"parts":[{"inlineData":{"mimeType":"image/png","data":"AQID"}}]}}]}"#, "neither PNG nor JPEG"),
        ];
        for (reply, expected) in cases {
            let p = provider(vec![ok(200, reply)]);
            let e = p.generate("x", "1:1").unwrap_err();
            assert!(e.to_string().contains(expected), "{e} / {expected}");
        }
        let p = provider(vec![ok(200, b"{not json")]);
        assert!(p.generate("x", "1:1").unwrap_err().to_string().contains("invalid JSON"));
    }

    #[test]
    fn http_errors_name_status_and_hint_without_echoing_the_message() {
        let message = "message-that-must-not-appear";
        for (status, google, hint) in [
            (400, "INVALID_ARGUMENT", "check the model"),
            (401, "UNAUTHENTICATED", "API key is missing, invalid or expired"),
            (403, "PERMISSION_DENIED", "no permission"),
            (404, "NOT_FOUND", "model was not found"),
            (429, "RESOURCE_EXHAUSTED", "wait a moment"),
            (500, "INTERNAL", "try again later"),
            (503, "UNAVAILABLE", "try again later"),
        ] {
            let body = json!({"error": {"code": status, "message": message, "status": google}}).to_string();
            let p = provider(vec![ok(status, body.as_bytes())]);
            let e = p.generate("x", "1:1").unwrap_err();
            let text = e.to_string();
            assert!(text.starts_with(&format!("Gemini returned HTTP {status} ({google})")), "{text}");
            assert!(text.contains(hint), "{text}");
            assert!(!text.contains(message) && !format!("{e:?}").contains(message));
        }
        let p = provider(vec![ok(502, b"<html>bad gateway</html>")]);
        let e = p.generate("x", "1:1").unwrap_err();
        assert!(matches!(e, GenError::Http { status: 502, code: None, .. }));
        let p = provider(vec![ok(400, br#"{"error":{"status":"SOMETHING_NEW secret"}}"#)]);
        assert!(matches!(p.generate("x", "1:1").unwrap_err(), GenError::Http { code: None, .. }));
    }

    #[test]
    fn key_never_leaks_into_url_body_or_errors() {
        let secret = "AIza-sentinel-gemini-key";
        let image = png_rgba(2, 2, |_, _| [0, 0, 0, 255]);
        let mask = png_rgba(2, 2, |_, _| [0, 0, 0, 0]);
        let replies: Vec<Result<crate::HttpResponse>> = vec![
            ok(400, format!(r#"{{"error":{{"code":400,"message":"API key not valid: {secret}","status":"INVALID_ARGUMENT"}}}}"#).as_bytes()),
            ok(401, b"{}"),
            ok(403, b"{}"),
            ok(429, b"{}"),
            ok(500, b"{}"),
            ok(200, b"{"),
            ok(200, br#"{"promptFeedback":{"blockReason":"SAFETY"}}"#),
            Err(GenError::Service("HTTPS connection failed")),
        ];
        let n = replies.len();
        let p = GeminiProvider::new(secret.into(), GEMINI_DEFAULT_MODEL, Mock::new(replies)).unwrap();
        for i in 0..n {
            let e = match i % 3 {
                0 => p.generate("x", "1:1"),
                1 => p.edit(&image, &mask, "x", "1:1"),
                _ => p.variations(&image, "x", "1:1"),
            }
            .unwrap_err();
            assert!(!format!("{e} {e:?}").contains(secret), "{e}");
        }
        for r in p.transport.requests.lock().unwrap().iter() {
            assert!(!r.url.contains(secret) && !r.url.contains("key="));
            assert!(!contains(&r.body, secret.as_bytes()));
            assert_eq!(r.auth.0, "x-goog-api-key");
        }
        assert!(!format!("{:?}", Auth::GoogleApiKey(secret)).contains(secret));
        assert!(!format!("{:?}", HttpRequest { url: "u", auth: Auth::GoogleApiKey(secret), content_type: "c", body: b"" }).contains(secret));
    }

    #[test]
    fn model_ids_are_validated_and_sizes_map_to_ratios() {
        for bad in ["gpt-image-1", "gemini-x/../../v1/files", "gemini-a?key=1", ""] {
            assert!(GeminiProvider::new("k".into(), bad, Mock::new(Vec::new())).is_err(), "{bad}");
        }
        assert!(matches!(GeminiProvider::new(" ".into(), GEMINI_DEFAULT_MODEL, Mock::new(Vec::new())), Err(GenError::MissingProviderKey { .. })));
        let p = GeminiProvider::new("k".into(), "gemini-3-pro-image", Mock::new(Vec::new())).unwrap();
        assert_eq!(p.size_for(1000, 1000), "1:1");
        assert_eq!(p.size_for(1920, 1080), "16:9");
        assert_eq!(p.size_for(1000, 1500), "2:3");
        assert_eq!(p.size_for(5000, 100), "21:9");
        assert_eq!(p.size_for(0, 0), "1:1");
        // Documented 1K sizes for Nano Banana 2.1 pass; wrong shapes and tiny images do not.
        for (size, w, h) in
            [("1:1", 1024, 1024), ("1024x1536", 848, 1264), ("1536x1024", 1264, 848), ("16:9", 1376, 768), ("21:9", 1584, 672), ("4:5", 928, 1152)]
        {
            assert!(p.accepts(size, w, h), "{size} {w}x{h}");
        }
        assert!(!p.accepts("1:1", 1376, 768));
        assert!(!p.accepts("1:1", 4, 4));
        assert!(!p.accepts("1:1", 0, 1024));
        assert!(p.accepts("auto", 1376, 768));
        assert!(!p.accepts("99x99", 1024, 1024));
    }
}
