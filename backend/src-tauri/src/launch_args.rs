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
        .map(|ext| ext.eq_ignore_ascii_case("hshp") || ext.eq_ignore_ascii_case("hsp"))
        .unwrap_or(false)
}

pub fn is_reaper_project_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("rpp"))
        .unwrap_or(false)
}

pub fn is_vocalshifter_project_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("vshp") || ext.eq_ignore_ascii_case("vsp"))
        .unwrap_or(false)
}

pub fn is_audio_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "wav"
                    | "flac"
                    | "mp3"
                    | "ogg"
                    | "m4a"
                    | "aac"
                    | "aif"
                    | "aiff"
                    | "wma"
                    | "opus"
            )
        })
        .unwrap_or(false)
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

#[cfg(test)]
mod tests {
    use super::extract_project_path_from_args;

    #[test]
    fn extract_path_accepts_hifishifter_project() {
        let temp = std::env::temp_dir().join("hifishifter_launch_args_test.hshp");
        std::fs::write(&temp, b"{}").expect("failed to create temp hshp file");

        let args = vec!["HiFiShifter.exe".to_string(), temp.display().to_string()];
        let found = extract_project_path_from_args(args);

        assert_eq!(found, Some(temp.display().to_string()));
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn extract_path_accepts_reaper_project() {
        let temp = std::env::temp_dir().join("hifishifter_launch_args_test.rpp");
        std::fs::write(&temp, b"reaper").expect("failed to create temp rpp file");

        let args = vec!["HiFiShifter.exe".to_string(), temp.display().to_string()];
        let found = extract_project_path_from_args(args);

        assert_eq!(found, Some(temp.display().to_string()));
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn extract_path_accepts_vocalshifter_project() {
        let temp = std::env::temp_dir().join("hifishifter_launch_args_test.vsp");
        std::fs::write(&temp, b"vsp").expect("failed to create temp vsp file");

        let args = vec!["HiFiShifter.exe".to_string(), temp.display().to_string()];
        let found = extract_project_path_from_args(args);

        assert_eq!(found, Some(temp.display().to_string()));
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn extract_path_accepts_audio_file() {
        let temp = std::env::temp_dir().join("hifishifter_launch_args_test.wav");
        std::fs::write(&temp, b"RIFF").expect("failed to create temp wav file");

        let args = vec!["HiFiShifter.exe".to_string(), temp.display().to_string()];
        let found = extract_project_path_from_args(args);

        assert_eq!(found, Some(temp.display().to_string()));
        let _ = std::fs::remove_file(&temp);
    }
}
