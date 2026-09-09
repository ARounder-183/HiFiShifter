//! 统一的导出编码层：WAV（hound）/ MP3（rusty_mp3）/ FLAC（rusty_flac）。
//!
//! `crate::mixdown` 完成交错 f32 混音后，由 [`create_encoder`] 按
//! [`OutputSpec`] 创建对应的 [`FileAudioEncoder`]；各实现负责量化
//! （整数位深可选 TPDF 抖动）、编码与落盘，并在分块推送之间响应取消标志。
//!
//! 落盘策略：WAV 走 hound 增量写盘；MP3 与 FLAC 因编码器约束
//! （rusty_mp3 的 Xing/Info 头、rusty_flac 的 `finish(self)` 全量输出）
//! 在内存缓冲后一次性写盘，内存峰值 ≈ 成品文件大小 + 编码器内部的全量
//! PCM/平面缓冲，详见各自模块注释。

pub mod flac;
pub mod mp3;
pub mod quantize;
pub mod wav;

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// ─── 输出描述 ─────────────────────────────────────────────────────────────────

/// 导出文件格式（serde 序列化为小写字符串）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Wav,
    Mp3,
    Flac,
}

impl OutputFormat {
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Wav => "wav",
            OutputFormat::Mp3 => "mp3",
            OutputFormat::Flac => "flac",
        }
    }

    pub fn as_name(self) -> &'static str {
        self.extension()
    }

    /// 宽松解析：用于持久化设置中的字符串字段，非法值由调用方回退默认。
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "wav" => Some(OutputFormat::Wav),
            "mp3" => Some(OutputFormat::Mp3),
            "flac" => Some(OutputFormat::Flac),
            _ => None,
        }
    }
}

/// WAV 位深：16 / 24-bit 整数与 32-bit 浮点。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WavBitDepth {
    I16,
    I24,
    #[default]
    F32,
}

/// FLAC 位深：rusty_flac 支持 8/16/24，导出面开放 16/24。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlacBitDepth {
    I16,
    #[default]
    I24,
}

/// 抖动模式：仅作用于整数位深输出（WAV i16/i24、FLAC）；浮点输出忽略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DitherMode {
    #[default]
    None,
    /// TPDF（三角概率密度函数）抖动，峰峰幅度 2 LSB。
    Tpdf,
}

impl DitherMode {
    pub fn as_name(self) -> &'static str {
        match self {
            DitherMode::None => "none",
            DitherMode::Tpdf => "tpdf",
        }
    }

    /// 宽松解析：非法值回退 `None`。
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "none" => Some(DitherMode::None),
            "tpdf" => Some(DitherMode::Tpdf),
            _ => None,
        }
    }
}

/// 导出声道模式：Mono = (L+R)×0.5 下混，在编码分叉点之前完成。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChannelMode {
    #[default]
    Stereo,
    Mono,
}

impl ChannelMode {
    pub fn as_name(self) -> &'static str {
        match self {
            ChannelMode::Stereo => "stereo",
            ChannelMode::Mono => "mono",
        }
    }

    /// 宽松解析：非法值回退 `Stereo`。
    pub fn from_name(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "stereo" => Some(ChannelMode::Stereo),
            "mono" => Some(ChannelMode::Mono),
            _ => None,
        }
    }
}

/// MP3 码率模式：CBR 固定码率（rusty_mp3 内部 `snap_bitrate` 吸附到 MPEG
/// 合法档位）；VBR 为 ffmpeg/LAME 式质量档（`-q:a`，0 = 最优 ~ 9 = 最小），
/// 经 `rusty_mp3::vbr_quality_index` 映射为目标平均码率。
///
/// 立体声 / 联合立体声由 rusty_mp3 逐帧自动决策，不作为用户参数暴露。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "mode",
    rename_all = "lowercase",
    rename_all_fields = "camelCase"
)]
pub enum Mp3BitrateMode {
    Cbr { bitrate_kbps: u32 },
    Vbr { quality_index: u8 },
}

impl Default for Mp3BitrateMode {
    fn default() -> Self {
        // VBR q2 ≈ 190 kbps 均值，LAME 社区公认的近似透明档。
        Mp3BitrateMode::Vbr { quality_index: 2 }
    }
}

/// MP3 ID3v2 标签；写入前会按 [`Mp3Tags::normalized`] 修剪并跳过全空标签。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Mp3Tags {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub artist: Option<String>,
    #[serde(default)]
    pub album: Option<String>,
    #[serde(default)]
    pub comment: Option<String>,
}

impl Mp3Tags {
    fn trim_opt(value: &Option<String>) -> Option<String> {
        value
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    }

    /// 修剪空白；全部字段为空 ⇒ `None`（调用方跳过 ID3 写入）。
    pub fn normalized(&self) -> Option<Mp3Tags> {
        let tags = Mp3Tags {
            title: Self::trim_opt(&self.title),
            artist: Self::trim_opt(&self.artist),
            album: Self::trim_opt(&self.album),
            comment: Self::trim_opt(&self.comment),
        };
        if tags.title.is_none()
            && tags.artist.is_none()
            && tags.album.is_none()
            && tags.comment.is_none()
        {
            None
        } else {
            Some(tags)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct WavEncodeOptions {
    pub bit_depth: WavBitDepth,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Mp3EncodeOptions {
    pub mode: Mp3BitrateMode,
    pub tags: Mp3Tags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct FlacEncodeOptions {
    pub bit_depth: FlacBitDepth,
    pub compression_level: u8,
}

impl Default for FlacEncodeOptions {
    fn default() -> Self {
        // 级别 5 = libFLAC/ffmpeg 默认，速度与压缩率的平衡点。
        FlacEncodeOptions {
            bit_depth: FlacBitDepth::I24,
            compression_level: 5,
        }
    }
}

/// 完整导出编码描述。采样率由 `MixdownOptions.sample_rate` 决定，不在此处。
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OutputSpec {
    pub format: OutputFormat,
    pub channel_mode: ChannelMode,
    pub dither: DitherMode,
    pub wav: WavEncodeOptions,
    pub mp3: Mp3EncodeOptions,
    pub flac: FlacEncodeOptions,
}

impl OutputSpec {
    /// 内部渲染（临时文件、胶合烘焙）：32-bit float WAV，不受用户设置影响。
    pub fn wav_32f() -> Self {
        Self::default()
    }

    /// 旧协议兼容：仅 bit_depth 驱动的 WAV 导出（16/24 → 整数，其余 → 32f）。
    pub fn wav_legacy(bit_depth: u32) -> Self {
        let mut spec = Self::default();
        spec.wav.bit_depth = match bit_depth {
            16 => WavBitDepth::I16,
            24 => WavBitDepth::I24,
            _ => WavBitDepth::F32,
        };
        spec
    }
}

// ─── 扩展名联动 ───────────────────────────────────────────────────────────────

/// 已知可替换的音频扩展名（ASCII 大小写不敏感）。
const KNOWN_AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "flac"];

/// 把文件名（或相对路径的末段）的扩展名替换 / 补全为目标格式扩展名。
///
/// 规则：末段已有 wav/mp3/flac 扩展名 → 原地替换；其余情况（无扩展名或
/// 其他扩展名）→ 追加。用于格式切换时输出路径的自动联动。
pub fn with_format_extension(file_name: &str, format: OutputFormat) -> String {
    let target = format.extension();
    let lower = file_name.to_ascii_lowercase();
    for ext in KNOWN_AUDIO_EXTENSIONS {
        let suffix = format!(".{ext}");
        // 严格长于后缀，避免把 ".wav" 这类裸扩展名误当词干。
        if lower.len() > suffix.len() && lower.ends_with(&suffix) {
            let stem = &file_name[..file_name.len() - suffix.len()];
            return format!("{stem}.{target}");
        }
    }
    format!("{file_name}.{target}")
}

// ─── 编码器接口 ───────────────────────────────────────────────────────────────

/// 导出编码错误；`Display` 直接产出命令层的错误码字符串。
#[derive(Debug)]
pub enum EncodeError {
    /// 用户取消（错误码 `export_cancelled`，与既有导出取消语义一致）。
    Cancelled,
    /// 参数或输入不合法；携带既有风格的错误码。
    Invalid(String),
    /// 文件 I/O 失败。
    Io(std::io::Error),
}

impl EncodeError {
    pub fn invalid(code: impl Into<String>) -> Self {
        EncodeError::Invalid(code.into())
    }
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::Cancelled => write!(f, "export_cancelled"),
            EncodeError::Invalid(code) => write!(f, "{code}"),
            EncodeError::Io(e) => write!(f, "encode_io_failed: {e}"),
        }
    }
}

impl From<std::io::Error> for EncodeError {
    fn from(e: std::io::Error) -> Self {
        EncodeError::Io(e)
    }
}

/// 编码完成摘要。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodeSummary {
    pub bytes_written: u64,
}

/// 文件音频编码器：接收 interleaved f32 `[-1, 1]` 分块直至
/// [`FileAudioEncoder::finish`]。
pub trait FileAudioEncoder {
    /// 推送一段交错样本（长度必须是声道数的整数倍）。
    /// 实现内部负责量化与编码；取消标志应在每个分块边界被检查。
    fn push(&mut self, interleaved: &[f32]) -> Result<(), EncodeError>;

    /// 结束输入并落盘，返回写盘摘要。取消或失败时由上层负责清理半成品。
    fn finish(self: Box<Self>) -> Result<EncodeSummary, EncodeError>;
}

/// MP3 支持的采样率（MPEG-2.5 / MPEG-2 / MPEG-1 Layer III 表）。
/// 88.2/96/176.4/192 kHz 不受支持 —— 由 UI 预先纠正，这里硬校验兜底。
pub const MP3_SAMPLE_RATES: &[u32] = &[
    8_000, 11_025, 12_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000,
];

/// 按 [`OutputSpec`] 创建对应格式的编码器。
pub fn create_encoder(
    output_path: &Path,
    spec: &OutputSpec,
    channels: u16,
    sample_rate: u32,
    cancel_flag: Option<Arc<AtomicBool>>,
) -> Result<Box<dyn FileAudioEncoder + Send>, EncodeError> {
    match spec.format {
        OutputFormat::Wav => Ok(Box::new(wav::WavFileEncoder::new(
            output_path,
            &spec.wav,
            spec.dither,
            channels,
            sample_rate,
            cancel_flag,
        )?)),
        OutputFormat::Mp3 => {
            if !MP3_SAMPLE_RATES.contains(&sample_rate) {
                return Err(EncodeError::invalid("mp3_unsupported_sample_rate"));
            }
            Ok(Box::new(mp3::Mp3FileEncoder::new(
                output_path,
                &spec.mp3,
                channels,
                sample_rate,
                cancel_flag,
            )?))
        }
        OutputFormat::Flac => Ok(Box::new(flac::FlacFileEncoder::new(
            output_path,
            &spec.flac,
            spec.dither,
            channels,
            sample_rate,
            cancel_flag,
        )?)),
    }
}

pub(crate) fn is_cancelled(flag: &Option<Arc<AtomicBool>>) -> bool {
    flag.as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_format_extension_replaces_known_extensions() {
        assert_eq!(
            with_format_extension("<ProjectName>.wav", OutputFormat::Mp3),
            "<ProjectName>.mp3"
        );
        assert_eq!(
            with_format_extension("song.MP3", OutputFormat::Flac),
            "song.flac"
        );
        assert_eq!(
            with_format_extension("a.b.Flac", OutputFormat::Wav),
            "a.b.wav"
        );
    }

    #[test]
    fn with_format_extension_appends_when_unknown_or_missing() {
        assert_eq!(
            with_format_extension("track", OutputFormat::Flac),
            "track.flac"
        );
        assert_eq!(
            with_format_extension("track.bak", OutputFormat::Mp3),
            "track.bak.mp3"
        );
        // 相对路径：仅末段参与替换，目录段不受影响。
        assert_eq!(
            with_format_extension("album.wav/track", OutputFormat::Mp3),
            "album.wav/track.mp3"
        );
        assert_eq!(
            with_format_extension("album/track.wav", OutputFormat::Mp3),
            "album/track.mp3"
        );
    }

    #[test]
    fn wav_legacy_maps_bit_depths() {
        assert_eq!(OutputSpec::wav_legacy(16).wav.bit_depth, WavBitDepth::I16);
        assert_eq!(OutputSpec::wav_legacy(24).wav.bit_depth, WavBitDepth::I24);
        assert_eq!(OutputSpec::wav_legacy(32).wav.bit_depth, WavBitDepth::F32);
        assert_eq!(
            OutputSpec::wav_legacy(8).wav.bit_depth,
            WavBitDepth::F32,
            "非法位深回退 32f"
        );
    }

    #[test]
    fn from_name_is_lenient() {
        assert_eq!(OutputFormat::from_name(" MP3 "), Some(OutputFormat::Mp3));
        assert_eq!(OutputFormat::from_name("ogg"), None);
        assert_eq!(DitherMode::from_name("tpdf"), Some(DitherMode::Tpdf));
        assert_eq!(DitherMode::from_name("noise"), None);
        assert_eq!(ChannelMode::from_name("MONO"), Some(ChannelMode::Mono));
        assert_eq!(ChannelMode::from_name("surround"), None);
    }

    #[test]
    fn output_spec_defaults_match_documented_values() {
        let spec = OutputSpec::default();
        assert_eq!(spec.format, OutputFormat::Wav);
        assert_eq!(spec.channel_mode, ChannelMode::Stereo);
        assert_eq!(spec.dither, DitherMode::None);
        assert_eq!(spec.wav.bit_depth, WavBitDepth::F32);
        assert_eq!(spec.mp3.mode, Mp3BitrateMode::Vbr { quality_index: 2 });
        assert_eq!(spec.flac.compression_level, 5);
        assert_eq!(spec.flac.bit_depth, FlacBitDepth::I24);
    }

    #[test]
    fn mp3_tags_normalized_trims_and_detects_empty() {
        let empty = Mp3Tags {
            title: Some("  ".to_string()),
            ..Default::default()
        };
        assert!(empty.normalized().is_none());
        let filled = Mp3Tags {
            artist: Some(" 秋浪 ".to_string()),
            ..Default::default()
        };
        assert_eq!(filled.normalized().unwrap().artist.as_deref(), Some("秋浪"));
    }

    #[test]
    fn mp3_bitrate_mode_serializes_tagged_camel_case() {
        let cbr = Mp3BitrateMode::Cbr { bitrate_kbps: 320 };
        let json = serde_json::to_string(&cbr).unwrap();
        assert_eq!(json, r#"{"mode":"cbr","bitrateKbps":320}"#);
        let vbr = Mp3BitrateMode::Vbr { quality_index: 2 };
        let json = serde_json::to_string(&vbr).unwrap();
        assert_eq!(json, r#"{"mode":"vbr","qualityIndex":2}"#);
        let round: Mp3BitrateMode = serde_json::from_str(&json).unwrap();
        assert_eq!(round, vbr);
    }
}
