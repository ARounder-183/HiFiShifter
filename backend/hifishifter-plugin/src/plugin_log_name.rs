//! 宿主名 → 日志文件名（纯函数，可全量单测）。
//!
//! 【为什么要按宿主分文件】所有宿主共写一个 `plugin.log` 时，用户报障拿到的日志里
//! 可能混着 REAPER、Cubase 与独立验证脚本的会话，甚至互相轮转掉。宿主名本身是
//! `IHostApplication::getName` 给的（见 `editor::connection`），拿不到就归到 unknown。
//!
//! 【为什么过滤得这么狠】宿主名是**外部输入**，且会直接拼进文件名。任何 `/`、`\`、
//! `..` 漏过去就等于让宿主决定我们把文件写在哪 —— 那是安全问题，不是命名问题。
//! 这里只保留 ASCII 字母数字与 `._-`，其余一律折叠成 `-`；结果为空、全是分隔符、
//! 或命中 Windows 设备名时返回 `plugin-unknown.log`。

/// 宿主名词干的最大长度（字符）。够认出 `studio-one`、`reaper`、`cubase`，
/// 又不至于让路径超限。
const MAX_SLUG_CHARS: usize = 40;

/// Windows 设备名：即使带扩展名，`CON.log` 这类仍会被系统当成设备。
const RESERVED_NAMES: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// 宿主名 → 文件名词干；不可用时返回 `None`。
pub fn host_slug(name: &str) -> Option<String> {
    let mut slug = String::with_capacity(name.len().min(MAX_SLUG_CHARS));
    let mut last_was_separator = false;
    for character in name.chars() {
        if slug.chars().count() >= MAX_SLUG_CHARS {
            break;
        }
        let mapped = if character.is_ascii_alphanumeric() {
            character.to_ascii_lowercase()
        } else if character == '.' || character == '_' || character == '-' {
            character
        } else {
            '-'
        };
        // 折叠连续分隔符：`"A  B"` 不该变成 `a--b`。
        if mapped == '-' || mapped == '.' || mapped == '_' {
            if last_was_separator {
                continue;
            }
            last_was_separator = true;
        } else {
            last_was_separator = false;
        }
        slug.push(mapped);
    }
    let trimmed = slug.trim_matches(|c| c == '-' || c == '.' || c == '_');
    if trimmed.is_empty() {
        return None;
    }
    if RESERVED_NAMES.contains(&trimmed) {
        return None;
    }
    Some(trimmed.to_string())
}

/// 最终日志文件名：有宿主名 → `plugin-<slug>.log`，否则 `plugin-unknown.log`。
pub fn log_file_name(host: Option<&str>) -> String {
    match host.and_then(host_slug) {
        Some(slug) => format!("plugin-{slug}.log"),
        None => "plugin-unknown.log".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_host_name_becomes_a_lowercase_slug() {
        assert_eq!(log_file_name(Some("REAPER")), "plugin-reaper.log");
        assert_eq!(log_file_name(Some("Cubase 15")), "plugin-cubase-15.log");
        assert_eq!(log_file_name(Some("Studio One")), "plugin-studio-one.log");
    }

    /// 宿主名里的路径分隔符绝不能被带进文件名。
    #[test]
    fn path_separators_are_folded_away() {
        let name = log_file_name(Some("../../etc/passwd"));
        assert!(!name.contains('/'), "{name}");
        assert!(!name.contains('\\'), "{name}");
        assert!(!name.contains(".."), "{name}");
        assert!(name.starts_with("plugin-"), "{name}");
        assert!(name.ends_with(".log"), "{name}");
    }

    #[test]
    fn windows_device_names_fall_back_to_unknown() {
        for name in ["CON", "nul", "COM1", "lpt9"] {
            assert_eq!(log_file_name(Some(name)), "plugin-unknown.log", "{name}");
        }
    }

    #[test]
    fn empty_or_punctuation_only_names_fall_back_to_unknown() {
        assert_eq!(log_file_name(None), "plugin-unknown.log");
        assert_eq!(log_file_name(Some("")), "plugin-unknown.log");
        assert_eq!(log_file_name(Some("   ")), "plugin-unknown.log");
        assert_eq!(log_file_name(Some("///")), "plugin-unknown.log");
    }

    #[test]
    fn overlong_names_are_cut_without_splitting_characters() {
        let long = "a".repeat(200);
        let name = log_file_name(Some(&long));
        assert_eq!(name, format!("plugin-{}.log", "a".repeat(MAX_SLUG_CHARS)));
    }

    /// 同一宿主的大小写变体必须落到同一个文件，否则会散成一堆。
    #[test]
    fn case_variants_share_one_file() {
        assert_eq!(log_file_name(Some("Reaper")), log_file_name(Some("REAPER")));
    }
}
