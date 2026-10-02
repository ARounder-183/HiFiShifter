//! CJK → 拉丁转写：汉字→拼音、假名→罗马字、谚文→分解。
//!
//! 【解决什么问题】全站搜索此前只有「小写子串」一种手段（文件浏览器
//! `stem_lower.contains(query)`、快捷键面板 `includes`）。中文用户搜「主歌.wav」
//! 必须切到中文输入法敲 `主歌`，日文用户搜「ボーカル.wav」必须切到日文输入法。
//! 这里把被搜索文本与查询串都转写成**拉丁形态**，于是 `zhuge` / `bokaru` /
//! `chexiao` 都能直接命中。
//!
//! 【为什么单独成模块】本文件是纯函数、无 IO、无状态：`translit()` 的输入是
//! 字符串与开关，输出是 [`Translit`]。文件搜索（后端目录遍历）与快捷键搜索
//! （前端索引）共用同一份实现 —— 若两处各写一份，「重」的多音字处理一旦漂移，
//! 用户搜 `chongzuo` 会在文件里命中、在快捷键里落空，而这种不一致极难排查。
//!
//! 【为什么不用 deunicode】它一个 crate 覆盖全部语种，但实测（release 增量）：
//! deunicode +460KB / 5 万文件名 86ms，而 pinyin +240KB / 39ms；且它对汉字给出的是
//! `"Che Xiao"`（逐字映射 + 空格 + 首字母大写），用户实际打的是 `chexiao` —— 靠它
//! 支持连写还得自己做去空格与大小写归一。它唯一不可替代的能力（谚文罗马化）用
//! 19+21+28 三张常量表就能覆盖，见本文件 `HANGUL_*`。

use pinyin::{ToPinyin, ToPinyinMulti};
use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;
use wana_kana::ConvertJapanese;

/// 多音字展开的组合上限。
///
/// 【为什么必须有上限】组合数 = 各字读音数之积：`重做重做重做重做` 是 3×2×3×2×…，
/// 不加闸门会指数爆炸。8 覆盖真实文件名里出现的绝大多数情况（实测「和声」8 个
/// 组合、「分行」7 个），且把最坏情况锁死在 8 倍匹配成本。
pub const MAX_VARIANTS: usize = 8;

/// 转写开关。三项的成本与收益各自独立，因此分开而不是合成一个「宽松匹配」。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranslitOptions {
    /// 多音字：把「重」的 `zhong` / `chong` / `tong` 都登记为候选形态。
    pub heteronym: bool,
    /// 日文长音宽松：`bokaru` 也命中「ボーカル」（`ー` 转写为 `o`，用户常省略）。
    pub japanese_long_vowel: bool,
    /// 韩文初声：把谚文的声母串（`한국어` → `hg`）登记为可检索形态。
    pub korean_choseong: bool,
}

impl Default for TranslitOptions {
    fn default() -> Self {
        Self {
            heteronym: true,
            japanese_long_vowel: true,
            korean_choseong: true,
        }
    }
}

/// 一段文本的主导文字类型。只用于给命中结果命名（「匹配拼音」/「匹配罗马音」…），
/// 不参与匹配判定。
///
/// 优先级 Kana > Han > Hangul：日文文件名通常汉字与假名混排，而用户心里读的是
/// 假名音；把这类串报成「拼音」会让提示文案与用户预期不符。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Script {
    #[default]
    Latin,
    Han,
    Kana,
    Hangul,
}

/// 一段文本的可检索形态。
///
/// 每个字段都是「同一种内容的另一种写法」，而不是不同的匹配策略 —— 策略在
/// `matcher.rs` 里，且以**分档**而非累加的方式消费这些形态（见那里的注释）。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Translit {
    /// 原文折叠形态：NFKC + 去组合记号 + 小写，CJK 原样保留。
    /// 用于「字面子串」这一档 —— 用户打原文（`vocal`）时不该被转写猜测干扰。
    pub latin: String,
    /// 全拼连写、去掉一切非字母数字：`主歌01.wav` → `zhuge01wav`。
    /// 去掉分隔符是为了让 `zhuge01` 命中 `主歌_01.wav`（下划线不该成为障碍）。
    pub compact: String,
    /// 首字母：汉字取拼音首字母（`撤销` → `cx`），假名取辅音骨架
    /// （`ボーカル` → `bkr`），谚文取初声（`한국어` → `hg`）。非 CJK 字符不参与。
    pub initials: String,
    /// 备用全拼形态：多音字变体、`ü` 的 `u` 读法、日文省长音形态。
    /// **不含** `compact` 本身（消费端用 [`Translit::compacts`] 遍历）。
    pub variants: Vec<String>,
    /// 主导文字类型。
    pub script: Script,
}

impl Translit {
    /// 是否含 CJK/假名/谚文。纯 ASCII 文本走字面快路径，不做任何转写匹配。
    pub fn has_cjk(&self) -> bool {
        self.script != Script::Latin
    }

    /// 遍历全部全拼形态（主形态在前）。
    pub fn compacts(&self) -> impl Iterator<Item = &str> {
        std::iter::once(self.compact.as_str()).chain(self.variants.iter().map(String::as_str))
    }
}

fn is_han(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'   // CJK 扩展 A
        | '\u{4E00}'..='\u{9FFF}' // CJK 基本区
        | '\u{F900}'..='\u{FAFF}' // CJK 兼容表意文字
        | '\u{20000}'..='\u{2FA1F}' // 扩展 B 及以后
    )
}

fn is_kana(c: char) -> bool {
    matches!(c,
        '\u{3040}'..='\u{309F}'   // 平假名
        | '\u{30A0}'..='\u{30FF}' // 片假名（含长音符 ー）
        | '\u{31F0}'..='\u{31FF}' // 片假名语音扩展
        | '\u{FF66}'..='\u{FF9D}' // 半角片假名
    )
}

fn is_hangul(c: char) -> bool {
    matches!(c, '\u{AC00}'..='\u{D7A3}')
}

/// 谚文声母（19 个）。`ㅇ` 是零声母，故为空串。
const HANGUL_LEAD: [&str; 19] = [
    "g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "c", "k", "t", "p",
    "h",
];
/// 谚文韵母（21 个）。
const HANGUL_VOWEL: [&str; 21] = [
    "a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "weo", "we",
    "wi", "yu", "eu", "yi", "i",
];
/// 谚文韵尾（28 个）。`""` 表示无韵尾。
const HANGUL_TAIL: [&str; 28] = [
    "", "k", "kk", "ks", "n", "nj", "nh", "d", "l", "lg", "lm", "lb", "ls", "lt", "lp", "lh", "m",
    "b", "bs", "s", "ss", "ng", "j", "c", "k", "t", "p", "h",
];

/// 谚文音节 → 全拼（声母+韵母+韵尾）。
fn hangul_full(c: char) -> String {
    let base = c as u32 - 0xAC00;
    let lead = (base / (21 * 28)) as usize;
    let vowel = ((base % (21 * 28)) / 28) as usize;
    let tail = (base % 28) as usize;
    format!(
        "{}{}{}",
        HANGUL_LEAD[lead], HANGUL_VOWEL[vowel], HANGUL_TAIL[tail]
    )
}

/// 谚文音节 → 初声（声母）。
fn hangul_lead(c: char) -> &'static str {
    let base = c as u32 - 0xAC00;
    HANGUL_LEAD[(base / (21 * 28)) as usize]
}

/// `ü` 的两种输入法写法：拼音键盘上打 `v`，也有用户直接打 `u`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UmlautStyle {
    V,
    U,
}

/// 整串折叠：NFKC → 小写 → NFD → 去组合记号。
///
/// 实测：`ＶＯＣＡＬ．ｗａｖ` → `vocal.wav`、`Étude` → `etude`、`ﬁle` → `file`、
/// `①` → `1`。NFKC 负责全角与连字，NFD 去记号负责变音符号。
///
/// 注意副作用：`ü` 会被剥成 `u`。拼音侧因此**先**把 `ü` 换成 `v`/`u` 再走本函数
/// （见 [`romanize`]），否则「绿」的 `lü` 与用户的 `lv` 永远对不上。
pub fn fold_text(s: &str) -> String {
    s.nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
        .collect()
}

/// 单字符折叠，用于逐字组装 compact 形态。
fn fold_char(c: char) -> String {
    fold_text(&c.to_string())
}

/// 转写一段文本。
pub fn translit(source: &str, opts: &TranslitOptions) -> Translit {
    // 纯 ASCII 快路径：目录遍历里绝大多数文件名走这里（NFKC 对 ASCII 是恒等变换，
    // 逐字符折叠反而更慢）。这也是 `needs_translit` 判 false 时唯一会走的路径。
    if source.is_ascii() {
        let mut latin = String::with_capacity(source.len());
        let mut compact = String::with_capacity(source.len());
        for c in source.chars() {
            let lower = c.to_ascii_lowercase();
            latin.push(lower);
            if lower.is_ascii_alphanumeric() {
                compact.push(lower);
            }
        }
        return Translit {
            latin,
            compact,
            initials: String::new(),
            variants: Vec::new(),
            script: Script::Latin,
        };
    }

    let primary = romanize(source, opts, UmlautStyle::V);

    let mut variants: Vec<String> = Vec::new();
    let mut push = |candidate: String| {
        if candidate != primary.compact
            && !candidate.is_empty()
            && !variants.contains(&candidate)
            && variants.len() < MAX_VARIANTS
        {
            variants.push(candidate);
        }
    };

    // 「绿」：主形态是 lv（输入法习惯），再登记一个 lu（另一部分人的习惯）。
    if primary.has_umlaut {
        push(romanize(source, opts, UmlautStyle::U).compact);
    }

    // 日文长音：「ボーカル」转写为 `bookaru`，但用户常打 `bokaru`。
    // 做法是把 `ー` 摘掉再转写一次，而不是在结果里猜哪个元音该删。
    if opts.japanese_long_vowel && source.contains('\u{30FC}') {
        let stripped: String = source.chars().filter(|c| *c != '\u{30FC}').collect();
        push(romanize(&stripped, opts, UmlautStyle::V).compact);
    }

    if opts.heteronym {
        for candidate in heteronym_variants(source) {
            push(candidate);
        }
    }

    Translit {
        latin: fold_text(source),
        compact: primary.compact,
        initials: primary.initials,
        variants,
        script: primary.script,
    }
}

/// 逐字转写的中间结果。
struct Romanized {
    compact: String,
    initials: String,
    script: Script,
    has_umlaut: bool,
}

/// 逐字转写：汉字→拼音首音、假名→罗马字（整段连续假名一次转换）、谚文→分解、
/// 其余→折叠后保留字母数字。
fn romanize(source: &str, opts: &TranslitOptions, umlaut: UmlautStyle) -> Romanized {
    let chars: Vec<char> = source.chars().collect();
    let mut compact = String::with_capacity(source.len() * 2);
    let mut initials = String::new();
    let mut has_umlaut = false;
    let mut has_han = false;
    let mut has_kana = false;
    let mut has_hangul = false;

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if is_han(ch) {
            has_han = true;
            if let Some(p) = ch.to_pinyin() {
                let syllable = p.plain();
                push_syllable(&mut compact, syllable, umlaut, &mut has_umlaut);
                if let Some(first) = syllable.chars().next() {
                    initials.push(first);
                }
            }
            i += 1;
        } else if is_kana(ch) {
            has_kana = true;
            // 连续假名一次转换：wana_kana 的入口是 &str，逐字符调用会把
            // 分词与匹配开销按字符数放大（实测整串 6.3µs/名，逐字更差）。
            let mut run = String::new();
            while i < chars.len() && is_kana(chars[i]) {
                run.push(chars[i]);
                i += 1;
            }
            // 半角片假名（ﾎﾞｰｶﾙ）先归一到全角再转，否则会被整段透传。
            let normalized: String = run.nfkc().collect();
            let romaji = normalized.to_romaji();
            push_alnum(&mut compact, &romaji);
            // 假名的「首字母」= 辅音骨架（`bookaru` → `bkr`）。与韩文初声开关无关：
            // 这一项对含假名的文本恒成立，关掉它等于让日文缩写检索完全失效。
            push_consonant_skeleton(&mut initials, &romaji);
        } else if is_hangul(ch) {
            has_hangul = true;
            push_syllable(&mut compact, &hangul_full(ch), umlaut, &mut has_umlaut);
            if opts.korean_choseong {
                initials.push_str(hangul_lead(ch));
            }
            i += 1;
        } else {
            let folded = fold_char(ch);
            for c in folded.chars() {
                if c.is_alphanumeric() {
                    compact.push(c);
                }
            }
            i += 1;
        }
    }

    let script = if has_kana {
        Script::Kana
    } else if has_han {
        Script::Han
    } else if has_hangul {
        Script::Hangul
    } else {
        Script::Latin
    };

    Romanized {
        compact,
        initials,
        script,
        has_umlaut,
    }
}

/// 追加一个音节，按需处理 `ü`。
fn push_syllable(out: &mut String, syllable: &str, umlaut: UmlautStyle, has_umlaut: &mut bool) {
    if syllable.contains('ü') {
        *has_umlaut = true;
        let replacement = match umlaut {
            UmlautStyle::V => 'v',
            UmlautStyle::U => 'u',
        };
        for c in syllable.chars() {
            if c == 'ü' {
                out.push(replacement);
            } else if c.is_alphanumeric() {
                out.push(c);
            }
        }
    } else {
        push_alnum(out, syllable);
    }
}

/// 追加一段已是拉丁的文本，只做字母数字过滤（分隔符在 compact 形态里被丢弃）。
fn push_alnum(out: &mut String, text: &str) {
    for c in text.chars() {
        if c.is_alphanumeric() {
            out.push(c);
        }
    }
}

/// 日文辅音骨架：`bookaru` → `bkr`。
///
/// 【为什么不是「每个假名的首字母」】日文的音节是 CV 结构，首字母就是辅音本身；
/// 但长音 `ー` 与拗音会让「第几个字符是第几个音节」对不上。直接删元音得到辅音串，
/// 与日语输入法的「子音検索」习惯一致，也不必知道音节边界。
fn push_consonant_skeleton(out: &mut String, romaji: &str) {
    for c in romaji.chars() {
        if c.is_ascii_alphabetic() && !matches!(c, 'a' | 'e' | 'i' | 'o' | 'u') {
            out.push(c);
        }
    }
}

/// 多音字组合（已截断到 [`MAX_VARIANTS`]，首个组合恒为全首音形态）。
///
/// 返回的是 compact 形态（非字母数字已丢弃），因此可以直接与 `Translit::compact`
/// 比较去重。
fn heteronym_variants(source: &str) -> Vec<String> {
    let mut out: Vec<String> = vec![String::new()];
    for ch in source.chars() {
        let alts = roman_alternatives(ch);
        if alts.len() == 1 {
            let only = &alts[0];
            for s in out.iter_mut() {
                s.push_str(only);
            }
            continue;
        }
        let mut next: Vec<String> = Vec::with_capacity(out.len() * alts.len());
        'outer: for prefix in &out {
            for alt in &alts {
                let mut s = prefix.clone();
                s.push_str(alt);
                next.push(s);
                if next.len() >= MAX_VARIANTS {
                    break 'outer;
                }
            }
        }
        out = next;
    }
    out
}

/// 单字符的候选转写（compact 形态）。非 CJK 字符恒返回一个元素。
fn roman_alternatives(ch: char) -> Vec<String> {
    if is_han(ch) {
        let mut out: Vec<String> = Vec::new();
        if let Some(multi) = ch.to_pinyin_multi() {
            for p in multi {
                let plain = p.plain();
                if plain.contains('ü') {
                    // ü 的两种写法各自登记：lv / lu。
                    for style in [UmlautStyle::V, UmlautStyle::U] {
                        let mut s = String::new();
                        let mut dummy = false;
                        push_syllable(&mut s, plain, style, &mut dummy);
                        if !out.contains(&s) {
                            out.push(s);
                        }
                    }
                } else if !out.iter().any(|existing| existing == plain) {
                    out.push(plain.to_string());
                }
            }
        }
        if out.is_empty() {
            out.push(ch.to_string());
        }
        out.truncate(3);
        out
    } else if is_kana(ch) {
        let normalized: String = ch.to_string().nfkc().collect();
        vec![normalized
            .to_romaji()
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect()]
    } else if is_hangul(ch) {
        // 谚文没有多音字；初声是**独立的检索形态**（`initials`），不能混进全拼变体 ——
        // 混进去会让 `hg` 这类两字符缩写以「全拼前缀」的高档命中，与它实际的
        // 信息量（只是声母串）不符。
        vec![hangul_full(ch)]
    } else {
        let folded = fold_char(ch);
        vec![folded.chars().filter(|c| c.is_alphanumeric()).collect()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(source: &str) -> Translit {
        translit(source, &TranslitOptions::default())
    }

    fn no_heteronym() -> TranslitOptions {
        TranslitOptions {
            heteronym: false,
            ..TranslitOptions::default()
        }
    }

    #[test]
    fn pinyin_full_and_initials() {
        let out = t("分割音频块");
        assert_eq!(out.compact, "fengeyinpinkuai");
        assert_eq!(out.initials, "fgypk");
        assert_eq!(out.script, Script::Han);
        assert!(out.has_cjk());
    }

    #[test]
    fn pinyin_of_short_labels() {
        assert_eq!(t("撤销").compact, "chexiao");
        assert_eq!(t("撤销").initials, "cx");
        assert_eq!(t("打开文件").compact, "dakaiwenjian");
        assert_eq!(t("波形").compact, "boxing");
    }

    #[test]
    fn umlaut_gets_both_spellings() {
        let out = t("绿");
        assert_eq!(out.compact, "lv");
        assert!(
            out.variants.iter().any(|v| v == "lu"),
            "变体: {:?}",
            out.variants
        );
    }

    #[test]
    fn kana_romaji_and_consonant_skeleton() {
        let out = t("ひらがな");
        assert_eq!(out.compact, "hiragana");
        assert_eq!(out.initials, "hrgn");
        assert_eq!(out.script, Script::Kana);

        let out = t("カタカナ");
        assert_eq!(out.compact, "katakana");
    }

    #[test]
    fn kana_long_vowel_variant() {
        let out = t("ボーカル");
        assert_eq!(out.compact, "bookaru");
        assert!(
            out.variants.iter().any(|v| v == "bokaru"),
            "变体: {:?}",
            out.variants
        );
        assert_eq!(out.initials, "bkr");
    }

    #[test]
    fn japanese_kanji_falls_back_to_pinyin() {
        // wana_kana 不处理汉字，汉字部分走拼音兜底（中文用户搜日语工程时正是这个习惯）。
        let out = t("波形表示");
        assert_eq!(out.compact, "boxingbiaoshi");
        assert_eq!(out.script, Script::Han);
    }

    #[test]
    fn hangul_full_and_choseong() {
        let out = t("한국어");
        assert_eq!(out.compact, "hangukeo");
        assert_eq!(out.initials, "hg");
        assert_eq!(out.script, Script::Hangul);

        let out = t("볼륨");
        assert_eq!(out.compact, "bolryum");
        assert_eq!(out.initials, "br");
    }

    #[test]
    fn ascii_is_preserved_and_folded() {
        let out = t("Vocal_Take01.WAV");
        assert_eq!(out.compact, "vocaltake01wav");
        assert_eq!(out.initials, "");
        assert_eq!(out.script, Script::Latin);
        assert!(!out.has_cjk());
    }

    #[test]
    fn ascii_fast_path_matches_the_full_path() {
        // 快路径（`source.is_ascii()`）与完整路径必须给出同样的形态，
        // 否则「纯 ASCII 文件名」与「含全角字符的文件名」会有两套行为。
        let fast = t("Vocal_Take01.WAV");
        let slow = t("Vocal_Take01.WAＶ"); // 末字符为全角
        assert_eq!(fast.compact, "vocaltake01wav");
        assert_eq!(slow.compact, "vocaltake01wav");
        assert_eq!(fast.script, Script::Latin);
    }

    #[test]
    fn fullwidth_and_diacritics_fold() {
        assert_eq!(t("ＶＯＣＡＬ．ｗａｖ").compact, "vocalwav");
        assert_eq!(t("Étude").compact, "etude");
        assert_eq!(t("Ünïcödé").compact, "unicode");
        assert_eq!(t("ﬁle").compact, "file");
    }

    #[test]
    fn mixed_script_concatenates() {
        let out = t("主歌_Vocal_01");
        assert_eq!(out.compact, "zhugevocal01");
        assert_eq!(out.initials, "zg");
        assert_eq!(out.latin, "主歌_vocal_01");
    }

    #[test]
    fn heteronym_variants_are_capped() {
        let out = t("重做重做重做重做");
        assert!(out.variants.len() <= MAX_VARIANTS);
        let out = t("重做");
        assert!(
            out.variants.iter().any(|v| v == "chongzuo"),
            "变体: {:?}",
            out.variants
        );
    }

    #[test]
    fn heteronym_can_be_disabled() {
        let out = translit("重做", &no_heteronym());
        assert_eq!(out.compact, "zhongzuo");
        assert!(
            !out.variants.iter().any(|v| v == "chongzuo"),
            "变体: {:?}",
            out.variants
        );
    }

    #[test]
    fn empty_and_symbol_only_input() {
        let out = t("");
        assert_eq!(out.compact, "");
        assert_eq!(out.script, Script::Latin);
        let out = t("!!!");
        assert_eq!(out.compact, "");
    }
}
