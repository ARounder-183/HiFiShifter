//! 搜索：CJK 转写 + 分档匹配。
//!
//! 【为什么单独成模块】转写与匹配是**纯函数**，被两处消费：
//! - 后端目录遍历（`commands/file_browser.rs`）—— 文件搜索必须在这一侧，
//!   否则要把整棵目录树的文件名传到前端再过滤；
//! - 前端索引（通过 `transliterate` 命令批量取形态）—— 快捷键面板在 JS 侧
//!   完成匹配，但转写规则与本模块共用同一份实现。
//!
//! 若两处各写一份转写，「重」的多音字处理一旦漂移，用户搜 `chongzuo` 会在文件里
//! 命中、在快捷键里落空 —— 这种不一致极难排查，所以规则只能有一份。

pub mod matcher;
pub mod translit;

// 只重导出被模块外真正消费的项；其余（各档分值常量等）通过 `matcher::` 路径取，
// 免得留下没人用的重导出。
pub use matcher::{
    match_translit, MatchInfo, MatchKind, MatchOptions, Query, SearchMode, SCORE_LITERAL,
};
pub use translit::{translit, Translit, TranslitOptions};

/// 默认结果上限。与转写功能上线前 `search_files_recursive` 的硬编码值一致。
pub const DEFAULT_MAX_RESULTS: usize = 500;
/// 结果上限的允许区间上界（防止前端传入一个把内存打满的值）。
pub const MAX_MAX_RESULTS: usize = 5000;

/// 搜索命令的参数。全部字段可选：前端只在用户显式覆盖过时才传，
/// 缺省即取本模块的默认值（等价于「智能匹配、三项转写全开、上限 500」）。
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SearchOptions {
    pub mode: Option<SearchMode>,
    pub heteronym: Option<bool>,
    pub japanese_long_vowel: Option<bool>,
    pub korean_choseong: Option<bool>,
    pub max_results: Option<usize>,
}

impl SearchOptions {
    /// 解析成无 `Option` 的匹配参数。
    pub fn match_options(&self) -> MatchOptions {
        let default = MatchOptions::default();
        MatchOptions {
            mode: self.mode.unwrap_or(default.mode),
            translit: TranslitOptions {
                heteronym: self.heteronym.unwrap_or(default.translit.heteronym),
                japanese_long_vowel: self
                    .japanese_long_vowel
                    .unwrap_or(default.translit.japanese_long_vowel),
                korean_choseong: self
                    .korean_choseong
                    .unwrap_or(default.translit.korean_choseong),
            },
        }
    }

    /// 结果上限，收敛到 `[1, MAX_MAX_RESULTS]`。
    pub fn max_results(&self) -> usize {
        self.max_results
            .unwrap_or(DEFAULT_MAX_RESULTS)
            .clamp(1, MAX_MAX_RESULTS)
    }
}

/// `transliterate` 命令的返回项：前端建索引所需的全部形态。
///
/// 字段名与 [`Translit`] 对齐，避免两套叫法。前端只拿它做**字符串比较**，
/// 转写规则（多音字上限、长音省略、谚文分解）全在本模块里，前端不重复实现。
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslitResult {
    /// 原文折叠形态（NFKC + 去组合记号 + 小写）。
    pub latin: String,
    /// 全拼连写（去分隔符）。
    pub compact: String,
    /// 初声串。
    pub initials: String,
    /// 备用全拼形态（多音字 / 长音省略 / ü 的另一种写法）。
    pub variants: Vec<String>,
}

impl From<Translit> for TranslitResult {
    fn from(value: Translit) -> Self {
        Self {
            latin: value.latin,
            compact: value.compact,
            initials: value.initials,
            variants: value.variants,
        }
    }
}

/// 批量转写（供前端建索引）。
///
/// 【为什么是批量】快捷键面板要转写上百条动作名，逐个调用就是上百次 IPC 往返。
pub fn transliterate_batch(texts: &[String], options: &SearchOptions) -> Vec<TranslitResult> {
    let opts = options.match_options().translit;
    texts
        .iter()
        .map(|text| TranslitResult::from(translit(text, &opts)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_default_to_smart_and_all_translit_on() {
        let opts = SearchOptions::default().match_options();
        assert_eq!(opts.mode, SearchMode::Smart);
        assert!(opts.translit.heteronym);
        assert!(opts.translit.japanese_long_vowel);
        assert!(opts.translit.korean_choseong);
        assert_eq!(SearchOptions::default().max_results(), DEFAULT_MAX_RESULTS);
    }

    #[test]
    fn options_clamp_max_results() {
        let opts = SearchOptions {
            max_results: Some(0),
            ..SearchOptions::default()
        };
        assert_eq!(opts.max_results(), 1);
        let opts = SearchOptions {
            max_results: Some(usize::MAX),
            ..SearchOptions::default()
        };
        assert_eq!(opts.max_results(), MAX_MAX_RESULTS);
    }

    #[test]
    fn options_partial_override_keeps_other_defaults() {
        let opts = SearchOptions {
            mode: Some(SearchMode::Off),
            heteronym: Some(false),
            ..SearchOptions::default()
        };
        let resolved = opts.match_options();
        assert_eq!(resolved.mode, SearchMode::Off);
        assert!(!resolved.translit.heteronym);
        // 未显式覆盖的项仍取默认。
        assert!(resolved.translit.japanese_long_vowel);
    }

    #[test]
    fn batch_transliterate_keeps_input_order() {
        let texts = vec!["撤销".to_string(), "vocal01".to_string(), "ボーカル".to_string()];
        let out = transliterate_batch(&texts, &SearchOptions::default());
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].compact, "chexiao");
        assert_eq!(out[1].compact, "vocal01");
        assert_eq!(out[2].compact, "bookaru");
    }
}
