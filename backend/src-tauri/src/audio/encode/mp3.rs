//! MP3 编码器（rusty_mp3 0.7.0，纯 Rust，Apache-2.0）。
//!
//! ⚠ 关键约束：rusty_mp3 的 Xing/Info 头在 `finish()` 时才以 `push_front`
//! 方式进入输出队列 —— 若在 push 阶段就把帧落盘，Xing 头将无法位于文件头
//! （上游源码注释明言该场景需要 two-pass）。因此本实现把全部 MP3 帧
//! 缓冲在内存中，`finish()` 后一次性写盘；内存峰值 ≈ 成品文件大小
//! （320 kbps 五分钟约 115 MB，人声典型工程远小于此）。
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
    tags: Mp3EncodeOptions,
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
            tags: options.clone(),
        })
    }

    /// 排空就绪的编码帧。`Again` / `Eof` 都不是错误：push 阶段队列空返回
    /// `Again`，finish 后排空完毕返回 `Eof`，统一按"暂无更多输出"处理。
    fn drain(&mut self) {
        loop {
            match self.encoder.next_packet() {
                Ok(packet) => self.out.extend_from_slice(&packet),
                Err(rusty_mp3::error::Error::Again)
                | Err(rusty_mp3::error::Error::Eof) => break,
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
        if let Some(tags) = self.tags.tags.normalized() {
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
