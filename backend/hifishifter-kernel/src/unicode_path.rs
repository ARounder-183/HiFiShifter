//! 工程/归档文件名 Unicode 规范化。
//!
//! macOS 的 APFS/HFS+ 常把文件名以 NFD（分解形式）交给应用；Windows 的
//! 解压工具通常以 NFC（合成形式）落盘。两者肉眼相同、码点不同，
//! Path::exists() 会失败。所有写进工程/归档的相对路径与打开时的比较都
//! 必须经过这里。
//!
//! 边界：只做字符串规范化与已知路径的变体检查；不 read_dir、不递归、
//! 不建索引。缺失文件交给“重新捕获缺失媒体”手动兜底。

use unicode_normalization::UnicodeNormalization;

/// 显示/持久化用的规范形式：NFC。
pub fn normalize_text_nfc(value: &str) -> String {
    value.nfc().collect()
}

/// 兼容 macOS 实际文件名的形式：NFD。
pub fn normalize_text_nfd(value: &str) -> String {
    value.nfd().collect()
}

/// 相对路径：统一 / 分隔符 + 逐段 NFC。
pub fn normalize_relative_path(value: &str) -> String {
    value
        .replace('\\', "/")
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(normalize_text_nfc)
        .collect::<Vec<_>>()
        .join("/")
}

/// 相对路径：统一 / 分隔符 + 逐段 NFD。
pub fn normalize_relative_path_nfd(value: &str) -> String {
    value
        .replace('\\', "/")
        .split('/')
        .filter(|part| !part.is_empty() && *part != ".")
        .map(normalize_text_nfd)
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nfc_and_nfd_are_equivalent_for_dakuten() {
        assert_eq!(normalize_text_nfc("ド"), "ド");
        assert_eq!(normalize_text_nfd("ド"), "ド");
    }

    #[test]
    fn relative_paths_normalize_separators_and_segments() {
        assert_eq!(normalize_relative_path("A\\ド.wav"), "A/ド.wav");
        assert_eq!(
            normalize_relative_path_nfd("A/ド.wav"),
            "A/ド.wav"
        );
    }
}
