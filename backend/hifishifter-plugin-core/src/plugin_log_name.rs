//! 插件类型 + 宿主名 → 日志文件名（纯函数，可全量单测）。
//!
//! 【为什么文件名里同时带类型和宿主】日志目录是**整个产品共用**的：VST3、CLAP 以及
//! 后续加入的格式都会写进去。只按宿主分文件（plugin-reaper.log）会让同一宿主里两种
//! 格式的会话挤在一个文件里、互相轮转掉，读者无法判断某一行来自哪份二进制；只按类型
//! 分文件（plugin-clap.log）则把 REAPER 与 Cubase 的会话混在一起。两个维度都进文件名，
//! 才是"这一行来自哪次运行"的最小充分描述。
//!
//! 形状固定为 plugin-<format>-<host>.log，例如 plugin-vst3-reaper.log /
//! plugin-clap-reaper.log。新增格式时只要在外壳入口登记一个新的 set_log_tag
//! 值，无需改动本模块 —— 这里对格式串只做与宿主名同样的安全过滤。
//!
//! 【为什么过滤得这么狠】格式与宿主名都是**外部输入**（宿主名来自
//! IHostApplication::getName / clap_host_t.name），且会直接拼进文件名。任何
//! /、\、.. 漏过去就等于让宿主决定我们把文件写在哪 —— 那是安全问题，
//! 不是命名问题。这里只保留 ASCII 字母数字与 ._-，其余一律折叠成 -；结果为空、
//! 全是分隔符、或命中 Windows 设备名时返回 None。

/// 单个词干的最大长度（字符）。够认出 studio-one、reaper、cubase，
/// 又不至于让路径超限。
const MAX_SLUG_CHARS: usize = 40;

/// 未知（或未登记）时的占位词干。
const UNKNOWN: &str = "unknown";

/// Windows 设备名：即使带扩展名，CON.log 这类仍会被系统当成设备。
const RESERVED_NAMES: &[&str] = &[
    "con", "prn", "aux", "nul", "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8",
    "com9", "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9",
];

/// 一段外部文本 → 文件名词干；不可用时返回 None。
///
/// 【为什么是公开函数】格式外壳可以拿它校验自己登记的标签是否合法，避免把
/// "日志文件名悄悄变成 plugin-unknown.log" 藏到运行期才发现。
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
        // 折叠连续分隔符："A  B" 不该变成 a--b。
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

/// 安全词干：拿不到就用 unknown，**绝不**让空串/非法串拼进文件名。
fn safe_slug(value: Option<&str>) -> String {
    value
        .and_then(host_slug)
        .unwrap_or_else(|| UNKNOWN.to_string())
}

/// 最终日志文件名：plugin-<format>-<host>.log。
///
/// format 是本份二进制的插件类型标签（vst3 / clap / 未来的 au …），
/// host 是宿主显示名；两者缺一都退化成 unknown，但**不会**因此少一段或改形，
/// 于是下游（诊断包、探针、轮转）对文件名的假设始终成立。
pub fn log_file_name(format: Option<&str>, host: Option<&str>) -> String {
    format!("plugin-{}-{}.log", safe_slug(format), safe_slug(host))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 命名契约：类型在宿主之前，两段都在，扩展名固定。
    #[test]
    fn the_name_carries_both_the_format_and_the_host() {
        assert_eq!(
            log_file_name(Some("vst3"), Some("REAPER")),
            "plugin-vst3-reaper.log"
        );
        assert_eq!(
            log_file_name(Some("clap"), Some("REAPER")),
            "plugin-clap-reaper.log"
        );
        assert_eq!(
            log_file_name(Some("clap"), Some("Cubase 15")),
            "plugin-clap-cubase-15.log"
        );
        // 未来新增格式不需要改这里：它只是另一段词干。
        assert_eq!(
            log_file_name(Some("au"), Some("Logic Pro")),
            "plugin-au-logic-pro.log"
        );
    }

    /// 同一宿主里两种格式必须落进**不同**文件，否则会话与轮转会互相踩。
    #[test]
    fn formats_never_share_a_log_file() {
        assert_ne!(
            log_file_name(Some("vst3"), Some("REAPER")),
            log_file_name(Some("clap"), Some("REAPER"))
        );
    }

    /// 缺格式 / 缺宿主都补 unknown，形状不变。
    #[test]
    fn missing_dimensions_fall_back_to_unknown_without_changing_the_shape() {
        assert_eq!(
            log_file_name(None, Some("REAPER")),
            "plugin-unknown-reaper.log"
        );
        assert_eq!(log_file_name(Some("clap"), None), "plugin-clap-unknown.log");
        assert_eq!(log_file_name(None, None), "plugin-unknown-unknown.log");
    }

    /// 路径分隔符绝不能被带进文件名（格式串与宿主串都要过这道闸）。
    #[test]
    fn path_separators_are_folded_away() {
        for value in ["../../etc/passwd", "..\\windows\\system32"] {
            let name = log_file_name(Some(value), Some(value));
            assert!(!name.contains('/'), "{name}");
            assert!(!name.contains('\\'), "{name}");
            assert!(!name.contains(".."), "{name}");
            assert!(name.starts_with("plugin-"), "{name}");
            assert!(name.ends_with(".log"), "{name}");
        }
    }

    #[test]
    fn windows_device_names_fall_back_to_unknown() {
        for name in ["CON", "nul", "COM1", "lpt9"] {
            assert_eq!(host_slug(name), None, "{name}");
            assert!(
                log_file_name(Some(name), Some(name)).contains("unknown"),
                "{name}"
            );
        }
    }

    #[test]
    fn empty_or_punctuation_only_names_fall_back_to_unknown() {
        assert_eq!(host_slug(""), None);
        assert_eq!(host_slug("   "), None);
        assert_eq!(host_slug("///"), None);
    }

    #[test]
    fn overlong_names_are_cut_without_splitting_characters() {
        let long = "a".repeat(200);
        assert_eq!(host_slug(&long), Some("a".repeat(MAX_SLUG_CHARS)));
    }

    /// 同一宿主的大小写变体必须落到同一个文件，否则会散成一堆。
    #[test]
    fn case_variants_share_one_file() {
        assert_eq!(
            log_file_name(Some("CLAP"), Some("Reaper")),
            log_file_name(Some("clap"), Some("REAPER"))
        );
    }
}
