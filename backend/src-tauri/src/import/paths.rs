//! 导入器共用的路径判据。
//!
//! 两个导入器（REAPER 的 `.rpp`、VocalShifter 的 `.vshp/.vsp`）都要把工程里的
//! 素材路径解析成本机绝对路径。它们的路径来自**别的平台**是常态（工程在机器
//! 之间传递），而 `std::path::Path::is_absolute` 是**平台相关**的 —— 这正是
//! 本文件存在的唯一原因。

use std::path::Path;

/// 路径是否是绝对路径，**两种平台的写法都认**。
///
/// 【为什么不能只信 `Path::is_absolute`】Windows 上 `/Users/a/b.wav` 不是绝对
/// 路径（会被 join 成 `<base>\Users\a\b.wav`），POSIX 上 `C:\...` 同样不是。
/// 把对端平台的绝对路径当相对路径拼接，产出的必然是一个不存在的路径 ——
/// 用户看到的"素材缺失"还带着一条伪造的拼接路径，比直接报"路径来自其他平台"
/// 更难诊断。
///
/// 判据与 `Path::is_absolute` 在**本平台**上的结论一致，只额外补上对端平台的
/// 两种写法：POSIX 根（`/...`）与 Windows 盘符（`C:\...` / `C:/...`）。
/// UNC（`\\server\share`）已由本平台的 `is_absolute` 覆盖。
pub(crate) fn is_cross_platform_absolute(raw_path: &str) -> bool {
    if Path::new(raw_path).is_absolute() {
        return true;
    }
    if raw_path.starts_with('/') {
        return true;
    }
    let bytes = raw_path.as_bytes();
    bytes.len() >= 3
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && (bytes[2] == b'\\' || bytes[2] == b'/')
}

#[cfg(test)]
mod tests {
    use super::is_cross_platform_absolute;

    #[test]
    fn both_platform_absolute_forms_are_recognized() {
        // POSIX 绝对路径：在 Windows 上 Path::is_absolute 不认，这里必须认。
        assert!(is_cross_platform_absolute("/Users/me/audio.wav"));
        assert!(is_cross_platform_absolute("/home/me/audio.wav"));
        // Windows 盘符：在 POSIX 上 Path::is_absolute 不认，这里必须认。
        assert!(is_cross_platform_absolute(r"C:\Users\me\audio.wav"));
        assert!(is_cross_platform_absolute("C:/Users/me/audio.wav"));
        // 本平台的绝对路径仍由 Path::is_absolute 覆盖。
        assert!(is_cross_platform_absolute(
            &std::env::temp_dir().join("audio.wav").to_string_lossy()
        ));
        // 相对路径（含看起来像盘符的普通名字）不得被误判。
        assert!(!is_cross_platform_absolute("audio.wav"));
        assert!(!is_cross_platform_absolute("./sub/audio.wav"));
        assert!(!is_cross_platform_absolute("../audio.wav"));
        assert!(!is_cross_platform_absolute("C:audio.wav")); // 盘符相对：不是绝对
        assert!(!is_cross_platform_absolute("ab:cd/ef.wav")); // 第二位不是 ':'
        assert!(!is_cross_platform_absolute(""));
    }
}
