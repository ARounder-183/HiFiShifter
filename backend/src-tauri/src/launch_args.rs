// 启动参数解析模块。
//
// 目标：识别系统通过启动参数传入的可导入文件路径，
// 并在应用启动后触发对应打开/导入流程。

use std::ffi::OsString;
use std::path::Path;

fn normalize_candidate(arg: &str) -> String {
    let trimmed = arg.trim();
    // 某些启动器会保留外层引号，这里做一次轻量归一化。
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        return trimmed[1..trimmed.len() - 1].to_string();
    }
    trimmed.to_string()
}

pub fn is_project_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            ext.eq_ignore_ascii_case("hshp")
                || ext.eq_ignore_ascii_case("hsp")
                // 备份后缀：前端 dnd / 文件浏览器接受 hshp-bak、hsp-bak，
                // 启动参数必须用同一份白名单，否则从文件管理器双击备份工程
                // 会静默打开空白工程。
                || ext.eq_ignore_ascii_case("hshp-bak")
                || ext.eq_ignore_ascii_case("hsp-bak")
                || ext.eq_ignore_ascii_case("json")
        })
        .unwrap_or(false)
}

pub fn is_reaper_project_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("rpp") || ext.eq_ignore_ascii_case("rpp-bak"))
        .unwrap_or(false)
}

pub fn is_vocalshifter_project_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("vshp") || ext.eq_ignore_ascii_case("vsp"))
        .unwrap_or(false)
}

pub fn is_audio_file_path(path: &Path) -> bool {
    crate::media::is_media_extension(path)
}

pub fn is_supported_launch_file_path(path: &Path) -> bool {
    is_project_file_path(path)
        || is_reaper_project_file_path(path)
        || is_vocalshifter_project_file_path(path)
        || is_audio_file_path(path)
}

pub fn extract_project_path_from_args<I, S>(args: I) -> Option<String>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let mut iter = args.into_iter();
    // 跳过 argv[0]（可执行文件路径）。
    let _ = iter.next();

    for arg in iter {
        let raw = arg.into();
        let raw = raw.to_string_lossy();
        let candidate = normalize_candidate(&raw);
        if candidate.is_empty() {
            continue;
        }

        let path = Path::new(&candidate);
        if !is_supported_launch_file_path(path) {
            continue;
        }

        if path.exists() {
            return Some(candidate);
        }
    }

    None
}

/// 把 macOS LaunchServices 的 file:// URL 转成受支持的本地路径。
#[cfg(any(target_os = "macos", test))]
///
/// 只接受 file scheme，并复用启动参数白名单（工程 / Reaper / VocalShifter /
/// 媒体）。Apple Event 也可能送来 http 等 URL；不允许任意 scheme 进入打开流程。
pub fn path_from_open_url(url: &tauri::Url) -> Option<std::path::PathBuf> {
    if url.scheme() != "file" {
        return None;
    }
    let path = url.to_file_path().ok()?;
    is_supported_launch_file_path(&path).then_some(path)
}

/// 从一组 Opened URL 中取第一个受支持的本地路径。
#[cfg(any(target_os = "macos", test))]
///
/// 当前产品一次只打开一个工程；多个文件的情况由调用方记录 warn（不静默丢弃）。
pub fn pick_supported_open_url(urls: &[tauri::Url]) -> Option<std::path::PathBuf> {
    urls.iter().find_map(path_from_open_url)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project(name: &str) -> std::path::PathBuf {
        std::path::PathBuf::from(name)
    }

    #[test]
    fn project_suffixes_include_backups() {
        for name in ["a.hshp", "a.HSP", "a.hshp-bak", "a.hsp-bak", "a.json"] {
            assert!(is_project_file_path(&project(name)), "{name}");
        }
        assert!(!is_project_file_path(&project("a.txt")));
    }

    #[test]
    fn reaper_suffixes_include_backups() {
        assert!(is_reaper_project_file_path(&project("a.rpp")));
        assert!(is_reaper_project_file_path(&project("a.rpp-bak")));
        assert!(!is_reaper_project_file_path(&project("a.hshp")));
    }

    #[test]
    fn open_url_decodes_spaces_and_unicode() {
        // 用 from_file_path 构造 URL：Windows 的 file:// URL 需要盘符，
        // 手写 /tmp 会在 Windows 上被 url crate 拒绝。
        let original = std::env::temp_dir().join("My Project 测试.hshp");
        let url = tauri::Url::from_file_path(&original).unwrap();
        let path = path_from_open_url(&url).expect("supported file url");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            original.file_name().and_then(|name| name.to_str())
        );
    }

    #[test]
    fn open_url_rejects_non_file_and_unknown_suffix() {
        let http = tauri::Url::parse("https://example.com/a.hshp").unwrap();
        assert!(path_from_open_url(&http).is_none());
        let txt_path = std::env::temp_dir().join("a.txt");
        let txt = tauri::Url::from_file_path(&txt_path).unwrap();
        assert!(path_from_open_url(&txt).is_none());
    }

    #[test]
    fn pick_supported_open_url_returns_first_supported() {
        let dir = std::env::temp_dir();
        let urls = vec![
            tauri::Url::from_file_path(dir.join("a.txt")).unwrap(),
            tauri::Url::from_file_path(dir.join("b.hshp")).unwrap(),
            tauri::Url::from_file_path(dir.join("c.hsp")).unwrap(),
        ];
        assert_eq!(
            pick_supported_open_url(&urls)
                .and_then(|path| path.file_name().map(|name| name.to_owned())),
            Some(std::ffi::OsString::from("b.hshp"))
        );
    }
}
