//! WAV 编码器（hound）：16/24-bit 整数与 32-bit 浮点，增量写盘。
//!
//! 整数位深经 [`super::quantize`] 量化（可选 TPDF 抖动）；浮点直接写样本。
//! hound 在 `WavWriter::create` 时即创建 / 截断目标文件，随后逐块写入；
//! 取消或失败时半成品文件的删除由上层错误路径统一负责（与重构前行为一致）。

use super::quantize;
use super::{
    DitherMode, EncodeError, EncodeSummary, FileAudioEncoder, WavBitDepth, WavEncodeOptions,
};
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
                    let quantized = quantize::quantize_sample(
                        sample,
                        bits,
                        self.dither,
                        &mut self.dither_state,
                    );
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
        let bytes_written = std::fs::metadata(&self.path)
            .map(|meta| meta.len())
            .unwrap_or(0);
        Ok(EncodeSummary { bytes_written })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{DitherMode, WavBitDepth, WavEncodeOptions};

    /// 编码一组样本到临时文件再读回；调用后临时文件已删除。
    fn encode(
        name: &str,
        bit_depth: WavBitDepth,
        samples: &[f32],
    ) -> hound::WavReader<std::io::BufReader<std::fs::File>> {
        let path =
            std::env::temp_dir().join(format!("hifishifter-wav-{name}-{}.wav", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let options = WavEncodeOptions { bit_depth };
        let mut encoder =
            WavFileEncoder::new(&path, &options, DitherMode::None, 1, 48_000, None).unwrap();
        encoder.push(samples).unwrap();
        // `finish` 的接收者是 `Box<Self>`（trait 契约），不是 `&mut self`。
        Box::new(encoder).finish().unwrap();
        let reader = hound::WavReader::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        reader
    }

    /// 浮点位深逐样本原样写出，不做量化。
    #[test]
    fn float_output_round_trips_samples_verbatim() {
        let samples = [0.0_f32, 0.25, -0.5, 1.0];
        let reader = encode("f32", WavBitDepth::F32, &samples);
        assert_eq!(reader.spec().bits_per_sample, 32);
        assert_eq!(reader.spec().sample_format, hound::SampleFormat::Float);
        let read = reader
            .into_samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(read, samples);
    }

    /// 整型位深经量化：满量程映射到接近 i16 上限，零映射到零。
    #[test]
    fn integer_output_quantizes_to_the_bit_depth() {
        let reader = encode("i16", WavBitDepth::I16, &[0.0, 1.0, -1.0]);
        assert_eq!(reader.spec().bits_per_sample, 16);
        let read = reader
            .into_samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(read[0], 0);
        assert!(read[1] > 32_000, "{read:?}");
        assert!(read[2] < -32_000, "{read:?}");
    }

    /// 取消标志在 push 前生效时立即返回 `Cancelled`，不写出数据。
    #[test]
    fn a_cancelled_encoder_refuses_to_push() {
        let path =
            std::env::temp_dir().join(format!("hifishifter-wav-cancel-{}.wav", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let cancel = Arc::new(AtomicBool::new(true));
        let mut encoder = WavFileEncoder::new(
            &path,
            &WavEncodeOptions {
                bit_depth: WavBitDepth::F32,
            },
            DitherMode::None,
            1,
            48_000,
            Some(cancel),
        )
        .unwrap();
        assert!(matches!(encoder.push(&[0.5]), Err(EncodeError::Cancelled)));
        let _ = std::fs::remove_file(&path);
    }
}
