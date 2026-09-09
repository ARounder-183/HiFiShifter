//! WAV 编码器（hound）：16/24-bit 整数与 32-bit 浮点，增量写盘。
//!
//! 整数位深经 [`super::quantize`] 量化（可选 TPDF 抖动）；浮点直接写样本。
//! hound 在 `WavWriter::create` 时即创建 / 截断目标文件，随后逐块写入；
//! 取消或失败时半成品文件的删除由上层错误路径统一负责（与重构前行为一致）。

use super::quantize;
use super::{DitherMode, EncodeError, EncodeSummary, FileAudioEncoder, WavBitDepth, WavEncodeOptions};
use crate::encode::is_cancelled;
use hound::{SampleFormat, WavSpec, WavWriter};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// hound 错误 → 编码错误：I/O 原样保留，格式类错误转错误码。
fn map_hound_error(e: hound::Error) -> EncodeError {
    match e {
        hound::Error::IoError(io) => EncodeError::Io(io),
        other => EncodeError::invalid(format!("wav_encode_failed: {other}")),
    }
}

pub struct WavFileEncoder {
    path: PathBuf,
    spec: WavSpec,
    dither: DitherMode,
    /// `None` 表示已 finalize（或从未成功创建）。
    writer: Option<WavWriter<std::io::BufWriter<std::fs::File>>>,
    dither_state: quantize::DitherState,
    cancel_flag: Option<Arc<AtomicBool>>,
}

impl WavFileEncoder {
    pub fn new(
        output_path: &Path,
        options: &WavEncodeOptions,
        dither: DitherMode,
        channels: u16,
        sample_rate: u32,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Self, EncodeError> {
        let (bits_per_sample, sample_format) = match options.bit_depth {
            WavBitDepth::I16 => (16, SampleFormat::Int),
            WavBitDepth::I24 => (24, SampleFormat::Int),
            WavBitDepth::F32 => (32, SampleFormat::Float),
        };
        let spec = WavSpec {
            channels,
            sample_rate,
            bits_per_sample,
            sample_format,
        };
        let writer = WavWriter::create(output_path, spec).map_err(map_hound_error)?;
        Ok(Self {
            path: output_path.to_path_buf(),
            spec,
            dither,
            writer: Some(writer),
            dither_state: quantize::DitherState::new(0x9E37_79B9_7F4A_7C15),
            cancel_flag,
        })
    }
}

impl FileAudioEncoder for WavFileEncoder {
    fn push(&mut self, interleaved: &[f32]) -> Result<(), EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| EncodeError::invalid("wav_writer_closed"))?;
        match self.spec.sample_format {
            SampleFormat::Float => {
                for &sample in interleaved {
                    writer.write_sample(sample).map_err(map_hound_error)?;
                }
            }
            SampleFormat::Int => {
                let bits = u32::from(self.spec.bits_per_sample);
                for &sample in interleaved {
                    let quantized =
                        quantize::quantize_sample(sample, bits, self.dither, &mut self.dither_state);
                    writer.write_sample(quantized).map_err(map_hound_error)?;
                }
            }
        }
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<EncodeSummary, EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        let writer = self
            .writer
            .take()
            .ok_or_else(|| EncodeError::invalid("wav_writer_closed"))?;
        writer.finalize().map_err(map_hound_error)?;
        let bytes_written = std::fs::metadata(&self.path).map(|meta| meta.len()).unwrap_or(0);
        Ok(EncodeSummary { bytes_written })
    }
}
