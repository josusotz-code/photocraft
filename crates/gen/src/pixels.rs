//! PNG preparation for providers without a mask input (Gemini).

use photocraft_codecs::{self as codecs, ChannelLayout, DecodeOptions, Format, Image, Limits, SampleType};

use crate::{GenError, Result};

/// Same caps as the engine's interchange images.
const MAX_SIDE: u32 = 4096;
const MAX_PIXELS: u64 = 16_000_000;
/// Fill for transparent pixels, so empty canvas reads as neutral gray rather than black.
const EMPTY: f32 = 128.0;

pub(crate) struct PreparedEdit {
    /// The image flattened onto mid-gray, RGB PNG.
    pub image: Vec<u8>,
    /// White where the engine's mask is transparent (repaint), black elsewhere; grayscale PNG.
    pub mask: Vec<u8>,
    /// `[ymin, xmin, ymax, xmax]` of the white area on a 0–1000 scale.
    pub bounds: [u32; 4],
}

fn decode(bytes: &[u8], why: &'static str) -> Result<Image> {
    let opts = DecodeOptions {
        limits: Limits { max_width: MAX_SIDE, max_height: MAX_SIDE, max_pixels: MAX_PIXELS, max_alloc: 128 * 1024 * 1024 },
        keep_orientation: false,
    };
    codecs::decode_as_with(Format::Png, bytes, &opts).map(|i| i.convert(ChannelLayout::Rgba, SampleType::U8)).map_err(|_| GenError::Invalid(why))
}

fn encode(width: u32, height: u32, layout: ChannelLayout, data: Vec<u8>) -> Result<Vec<u8>> {
    let image = Image::from_u8(width, height, layout, data).map_err(|_| GenError::Service("could not prepare image"))?;
    codecs::encode(&image, Format::Png, &Default::default()).map_err(|_| GenError::Service("could not prepare image"))
}

/// Turns the engine's edit pair (RGBA image, mask whose transparent pixels are repainted) into
/// what a mask-less model can follow: an opaque image and a black-and-white mask image.
pub(crate) fn prepare_edit(image: &[u8], mask: &[u8]) -> Result<PreparedEdit> {
    let image = decode(image, "PNG image could not be read")?;
    let mask = decode(mask, "PNG mask could not be read")?;
    let (w, h) = (image.width(), image.height());
    if (mask.width(), mask.height()) != (w, h) {
        return Err(GenError::Invalid("image and mask sizes differ"));
    }
    let mut rgb = Vec::with_capacity(image.data().len() / 4 * 3);
    for px in image.data().as_chunks::<4>().0 {
        let a = f32::from(px[3]) / 255.0;
        for &c in &px[..3] {
            rgb.push((f32::from(c) * a + EMPTY * (1.0 - a)).round().clamp(0.0, 255.0) as u8);
        }
    }
    let width = w as usize;
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    let mut gray = Vec::with_capacity(mask.data().len() / 4);
    for (i, px) in mask.data().as_chunks::<4>().0.iter().enumerate() {
        let repaint = px[3] < 128;
        gray.push(if repaint { 255 } else { 0 });
        if repaint && width > 0 {
            let (x, y) = (i % width, i / width);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    if x0 == usize::MAX {
        return Err(GenError::Invalid("the mask marks no area to change"));
    }
    let scale = |v: usize, size: u32| ((v as f64 / f64::from(size.max(1))) * 1000.0).round().clamp(0.0, 1000.0) as u32;
    Ok(PreparedEdit {
        image: encode(w, h, ChannelLayout::Rgb, rgb)?,
        mask: encode(w, h, ChannelLayout::Gray, gray)?,
        bounds: [scale(y0, h), scale(x0, w), scale(y1, h), scale(x1, w)],
    })
}
