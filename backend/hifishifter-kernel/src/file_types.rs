//! 「另存为 / 打开」对话框的**类型过滤词表** —— 全仓唯一一处定义。
//!
//! 【为什么下沉到内核】同一件事有四个实现：插件在 Windows 走 IFileSaveDialog、
//! Linux 走 GtkFileChooserNative、macOS 走 NSSavePanel，独立 App 走 rfd。
//! 此前每一处各自写字符串，于是实测出现三种互不相干的形态：
//!
//!   * Windows 的**另存为**根本没有调用 SetFileTypes —— 下拉框只剩「所有文件」，
//!     用户可以把 .zip 诊断包写成 .txt（同一个文件的**打开**对话框反而是对的）；
//!   * GTK 把过滤器的**名字**写成了 "*.mid"，下拉框显示的是通配符而不是类型名；
//!   * App 的附件另存为零过滤器，可以存成任意扩展名。
//!
//! 把这些名字与扩展名收进一张表之后，四个实现只回答一件事：「怎么把这张表交给
//! 系统」。新增一种导出就不再需要同时记住四套 API 的细节。
//!
//! 【为什么不放在插件 crate】独立 App 也要用它，而 App 不依赖插件 —— 放插件里就
//! 只能让 App 再抄一份，那正是本模块要消灭的东西。

use std::sync::LazyLock;

/// 一次对话框该带上的类型过滤。
///
/// 【为什么是 String/Vec 而不是 &'static str】附件另存为的扩展名来自**资产本身**
/// （用户贴进记事本的文件），编译期不存在；而固定的那几种又必须是静态可取的。
/// 两者共用一个类型，调用点才能只面对一种形状。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileType {
    /// 人类可读的类型名，进下拉框。
    name: String,
    /// 无点的扩展名（全小写），第一个是默认扩展名。
    extensions: Vec<String>,
}

impl FileType {
    /// 构造并**归一化**：去掉前导点、统一小写、丢空项、按首次出现去重。
    ///
    /// 【为什么归一化放在构造里】扩展名会同时出现在三个地方（COM 的模式串、
    /// GTK 的 pattern、macOS 的 allowedFileTypes）。任何一处拿到 "MID" 或 ".mid"
    /// 都会让过滤器悄悄失效，而三种平台的失效表现各不相同 —— 与其在三处各写一遍
    /// 防御，不如让构造出来的值本身就只有一个形态。
    pub fn new<I, S>(name: impl Into<String>, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut normalized: Vec<String> = Vec::new();
        for extension in extensions {
            let extension = extension.into();
            let extension = extension
                .trim()
                .trim_start_matches('.')
                .to_ascii_lowercase();
            if extension.is_empty() || normalized.contains(&extension) {
                continue;
            }
            normalized.push(extension);
        }
        Self {
            name: name.into(),
            extensions: normalized,
        }
    }

    /// 按**资产自身的扩展名**构造（附件另存为）。
    ///
    /// 名字由扩展名派生（"PNG File"）而不是写死一个「文件」—— 下拉框里出现
    /// 「PNG File」才是用户能判断的那一句。
    pub fn asset(extension: &str) -> Self {
        let normalized = extension
            .trim()
            .trim_start_matches('.')
            .to_ascii_lowercase();
        let name = if normalized.is_empty() {
            "File".to_string()
        } else {
            format!("{} File", normalized.to_ascii_uppercase())
        };
        Self::new(name, [normalized])
    }

    /// 下拉框里显示的类型名。
    pub fn name(&self) -> &str {
        &self.name
    }

    /// 全部扩展名（无点、全小写，第一个是默认扩展名）。
    pub fn extensions(&self) -> &[String] {
        &self.extensions
    }

    /// rfd 的 add_filter 要的是 &[&str]。
    pub fn extension_refs(&self) -> Vec<&str> {
        self.extensions.iter().map(String::as_str).collect()
    }

    /// 默认扩展名（不带点）。空表时返回空串 —— 调用点据此跳过 SetDefaultExtension。
    pub fn default_extension(&self) -> &str {
        self.extensions.first().map(String::as_str).unwrap_or("")
    }

    /// COM 的 COMDLG_FILTERSPEC 要**一条**模式串：`*.mid;*.midi`。
    pub fn com_pattern(&self) -> String {
        self.extensions
            .iter()
            .map(|extension| format!("*.{extension}"))
            .collect::<Vec<_>>()
            .join(";")
    }

    /// GTK 的 FileFilter 每个 pattern 单列一条（add_pattern 只接受一条模式）。
    pub fn patterns(&self) -> Vec<String> {
        self.extensions
            .iter()
            .map(|extension| format!("*.{extension}"))
            .collect()
    }

    /// 这份过滤是否真的能表达（至少一个扩展名、且有名字）。
    ///
    /// 【为什么需要】写一个空过滤器给系统，多数平台会**静默**退化成「所有文件」——
    /// 那正是本次要修的症状。宁可让调用点显式判断，也不要它悄悄退化。
    pub fn is_usable(&self) -> bool {
        !self.extensions.is_empty() && !self.name.trim().is_empty()
    }
}

/// 诊断包（Help → 导出诊断信息）。
pub static DIAGNOSTICS_ZIP: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("Diagnostics Archive", ["zip"]));

/// MIDI 文件（导入 / 导出）。
///
/// 【为什么要两个扩展名】App 侧一直认 mid + midi，插件侧只认 mid —— 同一个对话框
/// 在两处行为不同，用户按 App 的习惯起的 .midi 文件在插件里选不到。
pub static MIDI: LazyLock<FileType> = LazyLock::new(|| FileType::new("MIDI File", ["mid", "midi"]));

/// 纯数据导出（布局 / 主题 / 颤音预设）。
pub static HIFISHIFTER_JSON: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("HiFiShifter Data", ["json"]));

/// 记事本正文导出：Markdown。
pub static NOTEBOOK_MARKDOWN: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("Markdown Document", ["md"]));

/// 记事本正文导出：HTML。
pub static NOTEBOOK_HTML: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("HTML Document", ["html"]));

/// 记事本正文导出的类型表：按用户选的格式取一条。
///
/// 【为什么按扩展名而不是按格式枚举】前端只传 extension（导出对话框的选项就是
/// 它）。按枚举再加一层映射会让「前端加了一个格式、后端没跟上」变成静默的
/// 空过滤器；按扩展名至少会退化成 X File 这样的可读名字。
pub fn notebook_document(extension: &str) -> FileType {
    let normalized = extension
        .trim()
        .trim_start_matches('.')
        .to_ascii_lowercase();
    match normalized.as_str() {
        "md" => NOTEBOOK_MARKDOWN.clone(),
        "html" => NOTEBOOK_HTML.clone(),
        _ => FileType::asset(&normalized),
    }
}

/// 独立 App 的工程文件（另存为 / 打开）。
pub static HIFISHIFTER_PROJECT: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("HiFiShifter Project", ["hshp", "hsp"]));

/// 工程备份。
pub static HIFISHIFTER_PROJECT_BACKUP: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("HiFiShifter Backup Project", ["hshp-bak", "hsp-bak"]));

/// 纯 JSON 工程（旧格式 / 兼容导入）。
pub static JSON_PROJECT: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("JSON Project", ["json"]));

/// REAPER 工程（导入）。
pub static REAPER_PROJECT: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("Reaper Project", ["rpp"]));

/// REAPER 工程备份（导入）。
pub static REAPER_PROJECT_BACKUP: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("Reaper Backup Project", ["rpp-bak"]));

/// VocalShifter 工程（导入）。
pub static VOCALSHIFTER_PROJECT: LazyLock<FileType> =
    LazyLock::new(|| FileType::new("VocalShifter Project", ["vshp", "vsp"]));

/// 立体声 WAV（导出）。
pub static WAV_AUDIO: LazyLock<FileType> = LazyLock::new(|| FileType::new("WAV", ["wav"]));

#[cfg(test)]
mod tests {
    use super::*;

    /// 表里每一条都必须自洽：名字非空、扩展名无点且全小写、没有重复。
    ///
    /// 【为什么钉死】这张表是四个平台的**唯一**输入。一条写坏的条目在 Windows 上
    /// 表现为空下拉框、在 macOS 上表现为「类型不匹配」、在 Linux 上表现为选不到文件 ——
    /// 三种症状都离「扩展名多了一个点」很远。
    #[test]
    fn every_vocabulary_entry_is_self_consistent() {
        let table: [(&str, &LazyLock<FileType>); 10] = [
            ("DIAGNOSTICS_ZIP", &DIAGNOSTICS_ZIP),
            ("MIDI", &MIDI),
            ("HIFISHIFTER_JSON", &HIFISHIFTER_JSON),
            ("NOTEBOOK_MARKDOWN", &NOTEBOOK_MARKDOWN),
            ("NOTEBOOK_HTML", &NOTEBOOK_HTML),
            ("HIFISHIFTER_PROJECT", &HIFISHIFTER_PROJECT),
            ("HIFISHIFTER_PROJECT_BACKUP", &HIFISHIFTER_PROJECT_BACKUP),
            ("JSON_PROJECT", &JSON_PROJECT),
            ("REAPER_PROJECT", &REAPER_PROJECT),
            ("VOCALSHIFTER_PROJECT", &VOCALSHIFTER_PROJECT),
        ];
        for (label, file_type) in table {
            assert!(file_type.is_usable(), "{label} 没有可用的过滤器");
            assert!(!file_type.name().trim().is_empty(), "{label} 缺类型名");
            let mut seen = std::collections::BTreeSet::new();
            for extension in file_type.extensions() {
                assert!(!extension.is_empty(), "{label} 有空扩展名");
                assert!(
                    !extension.starts_with('.'),
                    "{label} 的扩展名带点：{extension}"
                );
                assert_eq!(
                    extension.to_ascii_lowercase(),
                    *extension,
                    "{label} 的扩展名不是小写：{extension}"
                );
                assert!(
                    seen.insert(extension.clone()),
                    "{label} 扩展名重复：{extension}"
                );
            }
            assert_eq!(
                file_type.default_extension(),
                file_type.extensions()[0],
                "{label} 的默认扩展名必须是第一条"
            );
        }
    }

    /// 构造即归一化：前导点、大小写、空项与重复项都必须被吃掉。
    #[test]
    fn construction_normalizes_extensions() {
        let file_type = FileType::new("Thing", [".MID", "midi", " ", ".mid"]);
        assert_eq!(file_type.extensions(), ["mid", "midi"]);
        assert_eq!(file_type.name(), "Thing");
    }

    /// COM 的模式串是一条分号串，GTK 的 pattern 是一组独立串 —— 两者必须同源。
    #[test]
    fn the_two_pattern_shapes_agree_with_the_extension_list() {
        assert_eq!(MIDI.com_pattern(), "*.mid;*.midi");
        assert_eq!(MIDI.patterns(), ["*.mid", "*.midi"]);
        assert_eq!(MIDI.extension_refs(), ["mid", "midi"]);
        assert_eq!(MIDI.default_extension(), "mid");
    }

    /// 附件另存为按资产扩展名构造，名字要能读。
    #[test]
    fn asset_filters_derive_a_readable_name() {
        let file_type = FileType::asset(".PNG");
        assert_eq!(file_type.name(), "PNG File");
        assert_eq!(file_type.extensions(), ["png"]);
    }

    /// 记事本两种格式各有自己的类型名，未知扩展名退化成资产名而不是空过滤器。
    #[test]
    fn notebook_documents_map_by_extension() {
        assert_eq!(notebook_document("md").name(), "Markdown Document");
        assert_eq!(notebook_document(".HTML").name(), "HTML Document");
        let fallback = notebook_document("rtf");
        assert_eq!(fallback.name(), "RTF File");
        assert!(fallback.is_usable());
    }
}
