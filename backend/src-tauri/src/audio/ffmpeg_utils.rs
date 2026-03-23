//! FFmpeg 外部转码工具模块。
//!
//! 当 symphonia/hound 无法解码的音频格式（如 WMA、APE、TAK 等）时，
//! 使用系统已安装的 ffmpeg 转码为临时 WAV 文件，解码完成后立即删除。

use std::path::{Path, PathBuf};
use std::process::Command;

/// ffmpeg 无法找到时返回的错误标识，前端据此弹出安装提示。
pub const FFMPEG_NOT_FOUND: &str = "FFMPEG_NOT_FOUND";

/// 原生支持的扩展名列表（hound + symphonia 可直接处理）
const NATIVE_EXTS: &[&str] = &["wav", "flac", "mp3", "ogg", "m4a", "aac", "aif", "aiff"];

/// 需要 ffmpeg fallback 的音频扩展名列表
const FFMPEG_AUDIO_EXTS: &[&str] = &["wma", "ape", "tak", "tta", "dff", "dsf", "opus", "mka", "webm"];

/// 支持提取音频轨道的视频扩展名列表（ffmpeg -vn 提取）
const VIDEO_EXTS: &[&str] = &[
    "mp4", "mkv", "avi", "mov", "wmv", "flv", "ts", "m4v",
    "mpg", "mpeg", "3gp", "vob", "rm", "rmvb",
];

/// 判断给定路径的扩展名是否为原生支持的音频格式
pub fn is_native_audio(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| NATIVE_EXTS.iter().any(|&x| x.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// 判断给定路径的扩展名是否需要 ffmpeg 转码（音频 fallback + 视频提取音轨）
pub fn needs_ffmpeg(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            FFMPEG_AUDIO_EXTS.iter().any(|&x| x.eq_ignore_ascii_case(e))
                || VIDEO_EXTS.iter().any(|&x| x.eq_ignore_ascii_case(e))
        })
        .unwrap_or(false)
}

/// 判断给定路径的扩展名是否为视频文件
pub fn is_video(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTS.iter().any(|&x| x.eq_ignore_ascii_case(e)))
        .unwrap_or(false)
}

/// 返回用于文件对话框的所有支持的音频格式扩展名（原生 + ffmpeg 音频 + 视频提取）
pub fn all_supported_extensions() -> Vec<&'static str> {
    let mut exts = Vec::with_capacity(NATIVE_EXTS.len() + FFMPEG_AUDIO_EXTS.len() + VIDEO_EXTS.len());
    exts.extend_from_slice(NATIVE_EXTS);
    exts.extend_from_slice(FFMPEG_AUDIO_EXTS);
    exts.extend_from_slice(VIDEO_EXTS);
    exts
}

/// 检测系统中是否安装了 ffmpeg。
///
/// 通过调用 `ffmpeg -version` 命令判断。
pub fn is_ffmpeg_available() -> bool {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .and_then(|mut child| child.wait())
        .map(|status| status.success())
        .unwrap_or(false)
}

/// 使用 ffmpeg 将源音频文件转码为临时 WAV（PCM 16-bit, 保持原始采样率）。
///
/// 返回生成的临时 WAV 文件路径。调用者负责在使用完毕后删除该文件。
///
/// # Errors
/// - 如果 ffmpeg 未安装，返回 `FFMPEG_NOT_FOUND` 错误标识。
/// - 如果转码过程失败，返回 ffmpeg 的 stderr 输出。
pub fn transcode_to_temp_wav(source: &Path) -> Result<PathBuf, String> {
    if !is_ffmpeg_available() {
        return Err(FFMPEG_NOT_FOUND.to_string());
    }

    let temp_dir = std::env::temp_dir().join("hifishifter");
    std::fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;

    let temp_wav = temp_dir.join(format!(
        "ffmpeg_{}_{}.wav",
        uuid::Uuid::new_v4().simple(),
        source
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio")
    ));

    let output = Command::new("ffmpeg")
        .args([
            "-y",                // 覆盖已有文件
            "-i",                // 输入
        ])
        .arg(source.as_os_str())
        .args([
            "-vn",               // 去除视频流
            "-acodec", "pcm_s16le", // 16-bit PCM
            "-f", "wav",         // 输出格式
        ])
        .arg(temp_wav.as_os_str())
        .output()
        .map_err(|e| format!("ffmpeg 执行失败: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // 清理可能生成的不完整文件
        let _ = std::fs::remove_file(&temp_wav);
        return Err(format!("ffmpeg 转码失败: {}", stderr.lines().last().unwrap_or("unknown error")));
    }

    if !temp_wav.exists() {
        return Err("ffmpeg 转码后未生成输出文件".to_string());
    }

    Ok(temp_wav)
}
