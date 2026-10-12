//! MP3 编码器（rusty_mp3 1.0.0，纯 Rust，Apache-2.0）。
//!
//! ⚠ 关键约束：rusty_mp3 的 Xing/Info 头在 `finish()` 时才以 `push_front`
//! 方式进入输出队列 —— 若在 push 阶段就把帧落盘，Xing 头将无法位于文件头
//! （上游源码注释明言该场景需要 two-pass）。因此本实现把全部 MP3 帧
//! 缓冲在内存中，`finish()` 后一次性写盘；内存峰值 ≈ 成品文件大小
//! 外加编码器内部的全量帧 PCM 缓冲（CBR 默认启用 3R1 水库时为全部帧的
//! f32 PCM；320 kbps 五分钟约 115 MB 成品 + ~100 MB PCM，人声典型工程远
//! 小于此）。
//!
//! 立体声 / 联合立体声由 rusty_mp3 逐帧自动决策；ID3v2 标签在音频帧之前
//! 写入（与 Xing/Info 头共存）。采样率合法性由 [`super::create_encoder`]
//! 硬校验（MPEG Layer III 表，见 `MP3_SAMPLE_RATES`）。

use super::{EncodeError, EncodeSummary, FileAudioEncoder, Mp3BitrateMode, Mp3EncodeOptions};
use crate::encode::is_cancelled;
use id3::TagLike;
use rusty_mp3::{Mp3Encoder, Mp3EncoderConfig};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct Mp3FileEncoder {
    encoder: Mp3Encoder,
    channels: u16,
    sample_rate: u32,
    /// 编码输出缓冲（含 finish 后前置的 Xing/Info 帧）。
    out: Vec<u8>,
    /// clamp 后的 PCM 暂存，复用以避免每次 push 重新分配。
    scratch: Vec<f32>,
    path: PathBuf,
    cancel_flag: Option<Arc<AtomicBool>>,
    options: Mp3EncodeOptions,
}

impl Mp3FileEncoder {
    pub fn new(
        output_path: &Path,
        options: &Mp3EncodeOptions,
        channels: u16,
        sample_rate: u32,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Result<Self, EncodeError> {
        let config = match options.mode {
            Mp3BitrateMode::Cbr { bitrate_kbps } => Mp3EncoderConfig {
                bitrate_kbps,
                vbr_quality: None,
            },
            Mp3BitrateMode::Vbr { quality_index } => Mp3EncoderConfig {
                bitrate_kbps: 0,
                vbr_quality: Some(rusty_mp3::vbr_quality_index(f32::from(quality_index))),
            },
        };
        Ok(Self {
            encoder: Mp3Encoder::new(config),
            channels: channels.max(1),
            sample_rate,
            out: Vec::new(),
            scratch: Vec::new(),
            path: output_path.to_path_buf(),
            cancel_flag,
            options: options.clone(),
        })
    }

    /// 排空就绪的编码帧。`Again` / `Eof` 都不是错误：push 阶段队列空返回
    /// `Again`，finish 后排空完毕返回 `Eof`，统一按"暂无更多输出"处理。
    fn drain(&mut self) {
        loop {
            match self.encoder.next_packet() {
                Ok(packet) => self.out.extend_from_slice(&packet),
                Err(rusty_mp3::error::Error::Again) | Err(rusty_mp3::error::Error::Eof) => break,
                Err(e) => {
                    // 非 drain 状态错误（理论上不会出现在 push/pull 循环里），
                    // 记录后中止排空，避免无限循环。
                    log::warn!("mp3 next_packet error: {e}");
                    break;
                }
            }
        }
    }
}

impl FileAudioEncoder for Mp3FileEncoder {
    fn push(&mut self, interleaved: &[f32]) -> Result<(), EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        self.scratch.clear();
        self.scratch.reserve(interleaved.len());
        // rusty_mp3 约定输入在 [-1, 1]；混音管线可能产生越界样本，钳制后再喂。
        for &sample in interleaved {
            self.scratch.push(sample.clamp(-1.0, 1.0));
        }
        self.encoder
            .push_pcm_f32(&self.scratch, self.channels, self.sample_rate)
            .map_err(|e| EncodeError::invalid(format!("mp3_encode_failed: {e}")))?;
        self.drain();
        Ok(())
    }

    fn finish(mut self: Box<Self>) -> Result<EncodeSummary, EncodeError> {
        if is_cancelled(&self.cancel_flag) {
            return Err(EncodeError::Cancelled);
        }
        self.encoder.finish();
        self.drain();

        // ID3v2 前置于音频帧；空标签直接跳过。
        let mut file_bytes = Vec::with_capacity(self.out.len() + 1024);
        if let Some(tags) = self.options.tags.normalized() {
            let mut tag = id3::Tag::default();
            if let Some(title) = &tags.title {
                tag.set_title(title);
            }
            if let Some(artist) = &tags.artist {
                tag.set_artist(artist);
            }
            if let Some(album) = &tags.album {
                tag.set_album(album);
            }
            if let Some(comment) = &tags.comment {
                tag.add_frame(id3::frame::Comment {
                    lang: "eng".to_string(),
                    description: String::new(),
                    text: comment.clone(),
                });
            }
            tag.write_to(&mut file_bytes, id3::Version::Id3v24)
                .map_err(|e| EncodeError::Io(std::io::Error::other(e.to_string())))?;
        }
        file_bytes.extend_from_slice(&self.out);

        std::fs::write(&self.path, &file_bytes).map_err(EncodeError::Io)?;
        Ok(EncodeSummary {
            bytes_written: file_bytes.len() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{Mp3BitrateMode, Mp3Tags};

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("hifishifter-mp3-{name}-{}.mp3", std::process::id()))
    }

    /// 立体声交错正弦（左右同相）；长度足够让 CBR 编码器产出多帧。
    fn tone(frames: usize) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| {
                let value = (i as f32 * 0.05).sin() * 0.5;
                [value, value]
            })
            .collect()
    }

    /// 落盘后摘要与文件长度一致，且文件以 MP3 帧同步字开头（无标签时无 ID3 前置）。
    #[test]
    fn encodes_pcm_to_a_playable_mp3() {
        let path = temp_path("basic");
        let _ = std::fs::remove_file(&path);
        let options = Mp3EncodeOptions {
            mode: Mp3BitrateMode::Cbr { bitrate_kbps: 128 },
            tags: Mp3Tags::default(),
        };
        let mut encoder = Mp3FileEncoder::new(&path, &options, 2, 48_000, None).unwrap();
        encoder.push(&tone(48_000)).unwrap();
        let summary = Box::new(encoder).finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(summary.bytes_written, bytes.len() as u64);
        // 11 位全 1 的帧同步：0xFF 后高 3 位为 111。
        assert!(bytes.len() > 2, "output too short: {} bytes", bytes.len());
        assert_eq!(bytes[0], 0xFF, "no frame sync: {:02X?}", &bytes[..4]);
        assert_eq!(bytes[1] & 0xE0, 0xE0, "no frame sync: {:02X?}", &bytes[..4]);
    }

    /// 非空标签写成 ID3v2 头，前置到文件开头。
    #[test]
    fn a_non_empty_tag_is_written_as_an_id3v2_header() {
        let path = temp_path("tagged");
        let _ = std::fs::remove_file(&path);
        let options = Mp3EncodeOptions {
            mode: Mp3BitrateMode::Vbr { quality_index: 2 },
            tags: Mp3Tags {
                title: Some("HiFi".to_string()),
                artist: Some("Shifter".to_string()),
                ..Default::default()
            },
        };
        let mut encoder = Mp3FileEncoder::new(&path, &options, 2, 44_100, None).unwrap();
        encoder.push(&tone(44_100)).unwrap();
        Box::new(encoder).finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(&bytes[..3], b"ID3");
    }

    /// 取消标志在 push 前生效时立即返回 `Cancelled`，不写出数据。
    #[test]
    fn a_cancelled_encoder_refuses_to_push() {
        let path = temp_path("cancel");
        let _ = std::fs::remove_file(&path);
        let cancel = Arc::new(AtomicBool::new(true));
        let mut encoder =
            Mp3FileEncoder::new(&path, &Mp3EncodeOptions::default(), 2, 44_100, Some(cancel))
                .unwrap();
        assert!(matches!(
            encoder.push(&[0.5, 0.5]),
            Err(EncodeError::Cancelled)
        ));
        let _ = std::fs::remove_file(&path);
    }
}
