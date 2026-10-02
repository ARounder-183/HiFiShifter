//! 查询与被搜索文本的匹配判定与分档打分。
//!
//! 【为什么是分档而不是累加】沿用 `keybindingSearch.ts` 已确立的原则（那里有详注）：
//! 累加会让词条多的条目靠**数量**取胜 —— 一个在标签、id、分组名里都泛泛沾边的条目，
//! 能在每个 token 上叠出高分，压过真正像的那条。取最高一档之后，「这条有多像」由
//! 最强的那个信号决定，而不是由沾边的地方有多少决定。
//!
//! 【档位顺序的理由】
//! - 字面（`vocal` 命中 `vocal01.wav`）最高：转写是有损的，用户既然打了原文，
//!   说明他知道确切拼写，这条意涵强过任何转写猜测。
//! - 全拼前缀 > 全拼子串：`zhuge` 命中「主歌01」时，「主歌」开头的那条更该在前。
//! - 初声（`cx` → 「撤销」）在全拼之后：初声是**音节对齐**的缩写，歧义小；
//!   但它比全拼短，误命中的绝对数量仍然更多。
//! - 初声子序列 > 模糊子序列：前者只在音节边界上跳字符，后者可以在任意位置跳。
//! - 模糊子序列最低且**默认关闭**：它是误命中的主要来源。

use serde::{Deserialize, Serialize};

use super::translit::{fold_text, translit, Script, Translit, TranslitOptions};

/// 字面子串。用户打原文时命中。
pub const SCORE_LITERAL: u8 = 8;
/// 全拼前缀。
pub const SCORE_FULL_PREFIX: u8 = 7;
/// 全拼子串。
pub const SCORE_FULL_SUBSTRING: u8 = 6;
/// 初声精确。
pub const SCORE_INITIALS_EXACT: u8 = 5;
/// 初声前缀。
pub const SCORE_INITIALS_PREFIX: u8 = 4;
/// 初声子序列（`fg` → 「分割」）。
pub const SCORE_INITIALS_SUBSEQ: u8 = 3;
/// 全拼模糊子序列（需显式开启 fuzzy 模式）。
pub const SCORE_FUZZY: u8 = 2;

/// 转写类匹配（全拼 / 初声）要求查询串至少这么长。
///
/// 【为什么要有】没有它，单字符查询会把半个磁盘拉出来：`z` 会命中所有拼音里含 z
/// 的文件名。字面档不受此限 —— 单字符的字面子串与今天的 `contains` 行为一致。
const MIN_TRANSLIT_LEN: usize = 2;
/// 初声子序列匹配的最小查询长度。
const MIN_INITIALS_SUBSEQ_LEN: usize = 2;
/// 模糊子序列匹配的最小查询长度。两字符的子序列几乎能命中任何东西。
const MIN_FUZZY_LEN: usize = 3;

/// 匹配宽严。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SearchMode {
    /// 只有字面子串（与转写功能上线前的行为一致）。
    Off,
    /// 字面 + 全拼 + 初声。
    #[default]
    Smart,
    /// 再加全拼模糊子序列。
    Fuzzy,
}

/// 命中类型，用于给用户解释「为什么这条匹配上了」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MatchKind {
    Literal,
    Pinyin,
    Romaji,
    Choseong,
    Fuzzy,
}

/// 一条命中的说明。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchInfo {
    pub kind: MatchKind,
    /// 命中的档位分值，用于排序。见本文件顶部的档位表。
    pub score: u8,
    /// 命中的形态（`zhuge` / `cx`），供界面显示「匹配拼音 zhuge」。
    pub form: String,
}

/// 一次匹配使用的完整参数（已解析，无 `Option`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MatchOptions {
    pub mode: SearchMode,
    pub translit: TranslitOptions,
}

/// 查询串的预计算形态。一次查询会与成千上万条文本比较，形态只算一次。
#[derive(Debug, Clone)]
pub struct Query {
    /// 折叠后的原文（用于字面档）。
    literal: String,
    /// 全拼连写形态。
    compact: String,
    /// 初声形态。纯拉丁查询等于其 compact（这样 `cx` 这类缩写能直接对上汉字初声串）。
    initials: String,
    /// 全部全拼形态（主形态在前，多音字/长音/ü 变体在后）。
    compacts: Vec<String>,
}

impl Query {
    pub fn new(raw: &str, opts: &MatchOptions) -> Self {
        let literal = fold_text(raw);
        let forms = translit(raw, &opts.translit);
        // 纯拉丁查询没有「初声」可言，但用户输入 `cx` 时它本来就是缩写 —— 直接拿
        // compact 当作初声去对汉字文本的初声串，`cx` 才能命中「撤销」。
        let initials = if forms.has_cjk() {
            forms.initials.clone()
        } else {
            forms.compact.clone()
        };
        let mut compacts = Vec::with_capacity(1 + forms.variants.len());
        compacts.push(forms.compact.clone());
        compacts.extend(forms.variants.iter().cloned());
        Self {
            literal,
            compact: forms.compact,
            initials,
            compacts,
        }
    }

    /// 查询是否为空（空查询由调用方按「全部命中」处理，不走本模块）。
    pub fn is_empty(&self) -> bool {
        self.literal.is_empty()
    }

    fn compacts(&self) -> impl Iterator<Item = &str> {
        self.compacts.iter().map(String::as_str)
    }
}

/// 判定一条文本是否命中，返回最高档的命中说明。
///
/// 未命中返回 `None`。
pub fn match_translit(text: &Translit, query: &Query, mode: SearchMode) -> Option<MatchInfo> {
    if query.is_empty() {
        return None;
    }

    // A：字面子串。
    if text.latin.contains(query.literal.as_str()) {
        return Some(MatchInfo {
            kind: MatchKind::Literal,
            score: SCORE_LITERAL,
            form: query.literal.clone(),
        });
    }
    if mode == SearchMode::Off {
        return None;
    }

    let long_enough = query.compact.chars().count() >= MIN_TRANSLIT_LEN;

    // B/C：全拼前缀与子串。前缀优先于子串，故先扫前缀。
    if long_enough {
        for hay in text.compacts() {
            for needle in query.compacts() {
                if !needle.is_empty() && hay.starts_with(needle) {
                    return Some(MatchInfo {
                        kind: kind_for_full(text.script),
                        score: SCORE_FULL_PREFIX,
                        form: needle.to_string(),
                    });
                }
            }
        }
        for hay in text.compacts() {
            for needle in query.compacts() {
                if !needle.is_empty() && hay.contains(needle) {
                    return Some(MatchInfo {
                        kind: kind_for_full(text.script),
                        score: SCORE_FULL_SUBSTRING,
                        form: needle.to_string(),
                    });
                }
            }
        }
    }

    // D/E/F：初声。
    if !text.initials.is_empty() && query.initials.chars().count() >= MIN_TRANSLIT_LEN {
        let hay = text.initials.as_str();
        let needle = query.initials.as_str();
        if hay == needle {
            return Some(MatchInfo {
                kind: kind_for_initials(text.script),
                score: SCORE_INITIALS_EXACT,
                form: needle.to_string(),
            });
        }
        if hay.starts_with(needle) {
            return Some(MatchInfo {
                kind: kind_for_initials(text.script),
                score: SCORE_INITIALS_PREFIX,
                form: needle.to_string(),
            });
        }
        if needle.chars().count() >= MIN_INITIALS_SUBSEQ_LEN && is_subsequence(hay, needle) {
            return Some(MatchInfo {
                kind: kind_for_initials(text.script),
                score: SCORE_INITIALS_SUBSEQ,
                form: needle.to_string(),
            });
        }
    }

    // G：模糊子序列（仅 fuzzy 模式）。
    if mode == SearchMode::Fuzzy
        && query.compact.chars().count() >= MIN_FUZZY_LEN
        && is_subsequence(&text.compact, &query.compact)
    {
        return Some(MatchInfo {
            kind: MatchKind::Fuzzy,
            score: SCORE_FUZZY,
            form: query.compact.clone(),
        });
    }

    None
}

/// 全拼/字面命中的命名：拉丁文本走全拼档时说明它其实只是「忽略分隔符的字面命中」。
fn kind_for_full(script: Script) -> MatchKind {
    match script {
        Script::Latin => MatchKind::Literal,
        Script::Han => MatchKind::Pinyin,
        Script::Kana => MatchKind::Romaji,
        Script::Hangul => MatchKind::Romaji,
    }
}

fn kind_for_initials(script: Script) -> MatchKind {
    match script {
        Script::Latin => MatchKind::Literal,
        Script::Han => MatchKind::Pinyin,
        Script::Kana => MatchKind::Romaji,
        Script::Hangul => MatchKind::Choseong,
    }
}

/// `needle` 的字符是否按顺序出现在 `hay` 中。
fn is_subsequence(hay: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let mut hay_chars = hay.chars();
    needle.chars().all(|n| hay_chars.any(|h| h == n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(mode: SearchMode) -> MatchOptions {
        MatchOptions {
            mode,
            translit: TranslitOptions::default(),
        }
    }

    fn hit(text: &str, query: &str, mode: SearchMode) -> Option<MatchInfo> {
        let o = opts(mode);
        let q = Query::new(query, &o);
        let t = translit(text, &o.translit);
        match_translit(&t, &q, mode)
    }

    fn expect_score(text: &str, query: &str, mode: SearchMode, score: u8) -> MatchInfo {
        let info = hit(text, query, mode)
            .unwrap_or_else(|| panic!("{:?} 应命中 {:?}（mode={:?}）", query, text, mode));
        assert_eq!(
            info.score, score,
            "{:?} 命中 {:?} 的档位应为 {}，实际 {}（form={:?}）",
            query, text, score, info.score, info.form
        );
        info
    }

    #[test]
    fn literal_beats_everything() {
        let info = expect_score("vocal01.wav", "vocal", SearchMode::Smart, SCORE_LITERAL);
        assert_eq!(info.kind, MatchKind::Literal);
        assert_eq!(info.form, "vocal");
    }

    #[test]
    fn pinyin_full_hits_chinese_name() {
        let info = expect_score("主歌01.wav", "zhuge", SearchMode::Smart, SCORE_FULL_PREFIX);
        assert_eq!(info.kind, MatchKind::Pinyin);
        assert_eq!(info.form, "zhuge");

        expect_score(
            "主歌01.wav",
            "zhuge01",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
    }

    #[test]
    fn pinyin_substring_is_one_band_below_prefix() {
        expect_score(
            "翻唱主歌01.wav",
            "zhuge",
            SearchMode::Smart,
            SCORE_FULL_SUBSTRING,
        );
        expect_score("主歌01.wav", "zhuge", SearchMode::Smart, SCORE_FULL_PREFIX);
    }

    #[test]
    fn initials_exact_and_prefix() {
        let info = expect_score("撤销.wav", "cx", SearchMode::Smart, SCORE_INITIALS_EXACT);
        assert_eq!(info.kind, MatchKind::Pinyin);
        assert_eq!(info.form, "cx");

        expect_score("主歌01.wav", "zg", SearchMode::Smart, SCORE_INITIALS_EXACT);
    }

    #[test]
    fn initials_prefix_when_name_is_longer() {
        expect_score(
            "撤销并重做.wav",
            "cx",
            SearchMode::Smart,
            SCORE_INITIALS_PREFIX,
        );
    }

    #[test]
    fn initials_subsequence() {
        // `fy` 不是初声串 `fgypk` 的前缀，但按顺序出现 —— 这一档专门收留
        // 「记不全中间几个字」的缩写。
        let info = expect_score(
            "分割音频块.wav",
            "fy",
            SearchMode::Smart,
            SCORE_INITIALS_SUBSEQ,
        );
        assert_eq!(info.kind, MatchKind::Pinyin);
        assert_eq!(info.form, "fy");
        // 前缀命中仍然走更高的那一档。
        expect_score(
            "分割音频块.wav",
            "fg",
            SearchMode::Smart,
            SCORE_INITIALS_PREFIX,
        );
    }

    #[test]
    fn separators_are_ignored_by_pinyin_band() {
        expect_score(
            "主歌_01_take.wav",
            "zhuge01",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
        // 纯拉丁文本忽略分隔符时也走全拼档，但语义上仍是字面命中。
        let info = expect_score(
            "vocal_take_01.wav",
            "vocaltake",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
        assert_eq!(info.kind, MatchKind::Literal);
    }

    #[test]
    fn single_char_query_does_not_use_translit_bands() {
        // 目标：搜 `z` 不该把半个磁盘拉出来。
        assert!(hit("主歌01.wav", "z", SearchMode::Smart).is_none());
        // 但字面档不受影响。
        assert!(hit("zebra.wav", "z", SearchMode::Smart).is_some());
    }

    #[test]
    fn japanese_kana_matches_romaji() {
        let info = expect_score(
            "ボーカル.wav",
            "bokaru",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
        assert_eq!(info.kind, MatchKind::Romaji);

        let info = expect_score(
            "ボーカル.wav",
            "bkr",
            SearchMode::Smart,
            SCORE_INITIALS_EXACT,
        );
        assert_eq!(info.kind, MatchKind::Romaji);
        assert_eq!(info.form, "bkr");
    }

    #[test]
    fn japanese_kana_matches_long_vowel_form() {
        // 主形态 bookaru 不含 bokaru；长音变体让后者也能命中。
        expect_score(
            "ボーカル.wav",
            "bokaru",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
    }

    #[test]
    fn korean_choseong_matches() {
        let info = expect_score("한국어.wav", "hg", SearchMode::Smart, SCORE_INITIALS_EXACT);
        assert_eq!(info.kind, MatchKind::Choseong);
        assert_eq!(info.form, "hg");
    }

    #[test]
    fn korean_full_romaji_matches() {
        expect_score(
            "한국어.wav",
            "hangukeo",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
    }

    #[test]
    fn fuzzy_only_in_fuzzy_mode() {
        assert!(hit("主歌01.wav", "zge", SearchMode::Smart).is_none());
        let info = expect_score("主歌01.wav", "zge", SearchMode::Fuzzy, SCORE_FUZZY);
        assert_eq!(info.kind, MatchKind::Fuzzy);
    }

    #[test]
    fn fuzzy_requires_three_chars() {
        assert!(hit("主歌01.wav", "zg", SearchMode::Fuzzy).is_some());
        // 两字符在 fuzzy 模式下走的是初声档，不是模糊档。
        let info = hit("主歌01.wav", "zg", SearchMode::Fuzzy).unwrap();
        assert_ne!(info.kind, MatchKind::Fuzzy);
    }

    #[test]
    fn off_mode_is_literal_only() {
        assert!(hit("主歌01.wav", "zhuge", SearchMode::Off).is_none());
        assert!(hit("主歌01.wav", "主歌", SearchMode::Off).is_some());
        assert!(hit("vocal01.wav", "vocal", SearchMode::Off).is_some());
    }

    #[test]
    fn empty_query_never_matches() {
        assert!(hit("anything.wav", "", SearchMode::Smart).is_none());
        assert!(hit("anything.wav", "   ", SearchMode::Smart).is_none());
    }

    #[test]
    fn chinese_query_matches_chinese_text_literally() {
        let info = expect_score("主歌01.wav", "主歌", SearchMode::Smart, SCORE_LITERAL);
        assert_eq!(info.kind, MatchKind::Literal);
    }

    #[test]
    fn chinese_query_matches_latin_text_via_romanization() {
        // 查询侧转写成 romaji 后对拉丁文件名命中（跨语系素材包的常见情形）。
        expect_score(
            "bokaru_take.wav",
            "ボーカル",
            SearchMode::Smart,
            SCORE_FULL_PREFIX,
        );
    }

    #[test]
    fn heteronym_variant_matches_other_reading() {
        // 「重做」主形态是 zhongzuo，多音字变体让 chongzuo 也命中。
        expect_score("重做.wav", "chongzuo", SearchMode::Smart, SCORE_FULL_PREFIX);
    }

    #[test]
    fn score_ordering_is_strictly_descending() {
        let table = [
            SCORE_LITERAL,
            SCORE_FULL_PREFIX,
            SCORE_FULL_SUBSTRING,
            SCORE_INITIALS_EXACT,
            SCORE_INITIALS_PREFIX,
            SCORE_INITIALS_SUBSEQ,
            SCORE_FUZZY,
        ];
        for pair in table.windows(2) {
            assert!(pair[0] > pair[1], "档位必须严格递减: {:?}", table);
        }
    }

    #[test]
    fn is_subsequence_basics() {
        assert!(is_subsequence("fgypk", "fg"));
        assert!(is_subsequence("fgypk", "fk"));
        assert!(!is_subsequence("fgypk", "gf"));
        assert!(!is_subsequence("", "a"));
        assert!(is_subsequence("abc", ""));
    }
}
