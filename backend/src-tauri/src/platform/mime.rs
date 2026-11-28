//! 扩展名 → MIME 类型。
//!
//! 【为什么需要这个模块】SAF 的 `Intent.EXTRA_MIME_TYPES` 只接受 **MIME 类型**
//! （`audio/x-wav`），而调用方手里只有扩展名（`wav`）。把扩展名直接塞进那个
//! extra，DocumentsUI 会认为"没有任何文件匹配"：选择器照常打开，但**每一个文件都是
//! 灰的、点不动**，而且没有任何提示 —— 用户看到的现象就是"导入音频点了没反应"。
//! 这是移动端导入全链路失效的真正原因（2026-11-24 在模拟器上实测确认）。
//!
//! 桌面的 `rfd` 吃的是扩展名，所以这层翻译在桌面路径上不存在，只在 Android 的
//! 原生桥之前发生。

/// 单个扩展名 → MIME 类型；不认识的返回 `None`。
///
/// 【为什么覆盖这么宽】它同时决定"选择器里能看见什么"。少写一个扩展名，那种文件
/// 在系统选择器里就是不可选的，而用户无从知道自己被过滤掉了。
pub fn for_extension(extension: &str) -> Option<&'static str> {
    let extension = extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    Some(match extension.as_str() {
        // ── 音频 ────────────────────────────────────────────────────────────
        "wav" => "audio/x-wav",
        "flac" => "audio/flac",
        "mp3" | "mp2" | "mpa" => "audio/mpeg",
        "ogg" | "oga" => "audio/ogg",
        "opus" => "audio/opus",
        "aac" | "m4a" => "audio/mp4",
        "aif" | "aiff" => "audio/aiff",
        "wma" => "audio/x-ms-wma",
        "ac3" | "eac3" => "audio/ac3",
        "ape" => "audio/x-ape",
        "wv" => "audio/x-wavpack",
        "dts" => "audio/vnd.dts",
        "amr" => "audio/3gpp",
        "mid" | "midi" => "audio/midi",
        // ── 视频 ────────────────────────────────────────────────────────────
        "mp4" | "m4v" => "video/mp4",
        "mov" => "video/quicktime",
        "mkv" => "video/x-matroska",
        "webm" => "video/webm",
        "avi" => "video/x-msvideo",
        "flv" => "video/x-flv",
        "wmv" => "video/x-ms-wmv",
        "ts" | "mts" | "m2ts" => "video/mp2t",
        "vob" | "mpg" | "mpeg" => "video/mpeg",
        "3gp" | "3g2" => "video/3gpp",
        "ogv" => "video/ogg",
        "rm" | "rmvb" => "application/vnd.rn-realmedia",
        // ── 图片（记事本插图 / 封面）────────────────────────────────────────
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "ico" => "image/x-icon",
        // ── 文档 / 工程 / 文本 ──────────────────────────────────────────────
        "zip" => "application/zip",
        "json" => "application/json",
        "md" => "text/markdown",
        "txt" => "text/plain",
        "csv" => "text/csv",
        "hshp" => "application/octet-stream",
        _ => return None,
    })
}

/// 一组扩展名 → 去重后的 MIME 列表，可直接放进 `EXTRA_MIME_TYPES`。
///
/// 【为什么"有一个不认识就整体不过滤"】`EXTRA_MIME_TYPES` 一旦给了，DocumentsUI
/// 只显示匹配的文件。漏掉一种扩展名，那种文件就会在选择器里变成灰的 —— 用户看不出
/// 原因，只觉得"我的文件不见了"。宁可不过滤（全部可选，选错了由后续的格式校验报错），
/// 也不要把文件藏起来。空列表 = 原生侧不下发这个 extra = 显示全部。
pub fn for_extensions(extensions: &[String]) -> Vec<String> {
    let mut mime_types: Vec<String> = Vec::new();
    for extension in extensions {
        match for_extension(extension) {
            Some(mime_type) => {
                if !mime_types.iter().any(|existing| existing == mime_type) {
                    mime_types.push(mime_type.to_string());
                }
            }
            None => return Vec::new(),
        }
    }
    mime_types
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_the_media_extensions_the_dialog_offers() {
        assert_eq!(for_extension("WAV"), Some("audio/x-wav"));
        assert_eq!(for_extension(".flac"), Some("audio/flac"));
        assert_eq!(for_extension("mp4"), Some("video/mp4"));
        assert_eq!(for_extension("exe"), None);
    }

    #[test]
    fn deduplicates_and_keeps_order() {
        let types = for_extensions(&["wav".into(), "aif".into(), "mp3".into()]);
        assert_eq!(types, vec!["audio/x-wav", "audio/aiff", "audio/mpeg"]);
    }

    #[test]
    fn an_unknown_extension_disables_filtering_entirely() {
        // 【为什么断言"空"】宁可显示全部文件，也不要把用户的文件藏起来。
        let types = for_extensions(&["wav".into(), "exe".into()]);
        assert!(types.is_empty());
    }
}
