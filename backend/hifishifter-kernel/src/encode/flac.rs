//! FLAC 编码器（rusty_flac 0.1.2，纯 Rust，Apache-2.0）。
//!
//! rusty_flac 的 `Encoder::finish(self)` 一次性返回完整 FLAC 流（暂无流式
//! 写盘 API），因此与 MP3 相同采用"内存缓冲、finish 后落盘"策略；内存峰值
//! ≈ 成品文件大小 + 编码器内部缓存的全部 i32 量化平面（约 4 B/样本/声道）。
//! FLAC 采样率为全表（≤ 2^20 Hz），无需 MP3 式的档位约束。
//! 量化经 [`super::quantize`]（可选 TPDF 抖动）手动完成后 `push_interleaved`，
//! 以便与 WAV 整数输出保持完全一致的量化语义。

use super::quantize;
use super::{
    DitherMode, EncodeError, EncodeSummary, FileAudioEncoder, FlacBitDepth, FlacEncodeOptions,
};
use crate::encode::is_cancelled;
use rusty_flac::Encoder as RustyFlacEncoder;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct FlacFileEncoder {
    /// `finish(self)` 消费编码器，用 Option 便于 take。
    encoder: Option<RustyFlacEncoder>,
    bits: u32,
    dither: DitherMode,
    dither_state: quantize::DitherState,
    /// 量化后的 i32 暂存，复用以避免每次 push 重新分配。
    scratch: Vec<i32>,
    path: PathBuf,
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl FlacFileEncoder {
    pub fn new(
        output_path: &Path,
        options: &FlacEncodeOptions,
        dither: DitherMode,
        channels: u16,
        sample_rate: u32,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Self, EncodeError> {
        let bits = match options.bit_depth {
            FlacBitDepth::I16 => 16u32,
            FlacBitDepth::I24 => 24u32,
        };
        let mut encoder = RustyFlacEncoder::new(sample_rate, u32::from(channels.max(1)), bits)
            .map_err(|e| EncodeError::invalid(format!("flac_encode_failed: {e}")))?;
        encoder.set_compression_level(u32::from(options.compression_level.min(8)));
        Ok(Self {
            encoder: Some(encoder),
            bits,
            dither,
            dither_state: quantize::DitherState::new(0x9E37_79B9_7F4A_7C15),
            scratch: Vec::new(),
            path: output_path.to_path_buf(),
            cancel_flag,
        })
    }
}

impl FileAudioEncoder for FlacFileEncoder {
    fn push(&mut self, interleaved: &[f32]) -> Result<(), EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        let encoder = self
            .encoder
            .as_mut()
            .ok_or_else(|| EncodeError::invalid("flac_encoder_closed"))?;
        self.scratch.clear();
        self.scratch.reserve(interleaved.len());
        for &sample in interleaved {
            self.scratch.push(quantize::quantize_sample(
                sample,
                self.bits,
                self.dither,
                &mut self.dither_state,
            ));
        }
        encoder
            .push_interleaved(&self.scratch)
            .map_err(|e| EncodeError::invalid(format!("flac_encode_failed: {e}")))
    }

    fn finish(mut self: Box<Self>) -> Result<EncodeSummary, EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        let encoder = self
            .encoder
            .take()
            .ok_or_else(|| EncodeError::invalid("flac_encoder_closed"))?;
        let data = encoder.finish();
        std::fs::write(&self.path, &data).map_err(EncodeError::Io)?;
        Ok(EncodeSummary {
            bytes_written: data.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::FlacBitDepth;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "hifishifter-flac-{name}-{}.flac",
            std::process::id()
        ))
    }

    /// 立体声交错正弦（左右同相）。
    fn tone(frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let value = (i as f32 * 0.05).sin() * 0.5;
                [value, value]
            })
            .collect()
    }

    /// 落盘后摘要与文件长度一致，且以 FLAC 魔数 `fLaC` 开头。
    #[test]
    fn encodes_pcm_to_a_flac_stream() {
        let path = temp_path("basic");
        let _ = std::fs::remove_file(&path);
        let options = FlacEncodeOptions {
            bit_depth: FlacBitDepth::I16,
            compression_level: 5,
        };
        let mut encoder =
            FlacFileEncoder::new(&path, &options, DitherMode::None, 2, 48_000, None).unwrap();
        encoder.push(&tone(48_000)).unwrap();
        let summary = Box::new(encoder).finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(summary.bytes_written, bytes.len() as u64);
        assert_eq!(&bytes[..4], b"fLaC");
    }

    /// 取消标志在 push 前生效时立即返回 `Cancelled`，不写出数据。
    #[test]
    fn a_cancelled_encoder_refuses_to_push() {
        let path = temp_path("cancel");
        let _ = std::fs::remove_file(&path);
        let cancel = Arc::new(AtomicBool::new(true));
        let mut encoder = FlacFileEncoder::new(
            &path,
            &FlacEncodeOptions::default(),
            DitherMode::None,
            2,
            48_000,
            Some(cancel),
        )
        .unwrap();
        assert!(matches!(
            encoder.push(&[0.5, 0.5]),
            Err(EncodeError::Cancelled)
        ));
        let _ = std::fs::remove_file(&path);
    }
}
