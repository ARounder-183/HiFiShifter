/**
 * 前端侧的匹配判定：与后端 `search::matcher` 同规则、同档位。
 *
 * 【为什么这里要再写一份】文件搜索在后端（目录遍历没法搬到前端），快捷键面板在
 * 前端（上百条动作名，每次击键都走 IPC 会让输入变得粘滞）。**转写规则只有一份**
 * —— 由后端 `transliterate` 命令产出形态；前端只做「对这些形态做字符串比较」，
 * 也就是本文件。它不含任何语言知识（没有拼音表、没有假名表），因此不会漂移。
 *
 * 【为什么规则要照抄而不是「大致类似」】两处对同一个查询给出不同的命中集合，是
 * 最难排查的一类不一致（用户只会觉得「这个搜索时灵时不灵」）。因此本文件的档位
 * 常量、长度门槛、判定顺序都与 `matcher.rs` 逐条对应，测试用例也共用同一组向量
 * （见 `translit.test.ts` 与 `matcher.rs` 的 `tests` 模块）。
 */

import type { SearchMode } from "./searchSettings";

/** 字面子串。 */
export const SCORE_LITERAL = 8;
/** 全拼前缀。 */
export const SCORE_FULL_PREFIX = 7;
/** 全拼子串。 */
export const SCORE_FULL_SUBSTRING = 6;
/** 初声精确。 */
export const SCORE_INITIALS_EXACT = 5;
/** 初声前缀。 */
export const SCORE_INITIALS_PREFIX = 4;
/** 初声子序列。 */
export const SCORE_INITIALS_SUBSEQ = 3;
/** 全拼模糊子序列（仅 fuzzy 模式）。 */
export const SCORE_FUZZY = 2;

/** 转写类匹配要求查询串至少这么长（与后端 `MIN_TRANSLIT_LEN` 一致）。 */
const MIN_TRANSLIT_LEN = 2;
/** 初声子序列匹配的最小查询长度。 */
const MIN_INITIALS_SUBSEQ_LEN = 2;
/** 模糊子序列匹配的最小查询长度。 */
const MIN_FUZZY_LEN = 3;

export type MatchKind = "literal" | "pinyin" | "romaji" | "choseong" | "fuzzy";

export interface MatchInfo {
    kind: MatchKind;
    score: number;
    form: string;
}

/** 一段文本的可检索形态。字段与后端 `search::TranslitResult` 对齐。 */
export interface TranslitForms {
    latin: string;
    compact: string;
    initials: string;
    variants: string[];
}

/** 查询串的预计算形态。 */
export interface QueryForms {
    literal: string;
    compact: string;
    initials: string;
    compacts: string[];
}

/**
 * 整串折叠：NFKC → 小写 → NFD → 去组合记号。
 *
 * 与后端 `translit::fold_text` 的步骤顺序一致：NFKC 负责全角与连字（`ＶＯＣＡＬ` →
 * `vocal`、`ﬁle` → `file`），NFD 去记号负责变音符号（`Étude` → `etude`）。
 */
export function foldText(text: string): string {
    return text.normalize("NFKC").toLowerCase().normalize("NFD").replace(/\p{M}/gu, "");
}

/** 只保留字母与数字（与 Rust 的 `char::is_alphanumeric` 对齐）。 */
export function alnumOnly(text: string): string {
    return text.replace(/[^\p{L}\p{N}]/gu, "");
}

/**
 * 后端不可用时的降级形态（浏览器 dev 模式）。
 *
 * 只做折叠与去分隔符，没有拼音/罗马字 —— 匹配退化为「字面 + 忽略分隔符」，
 * 功能不中断，也不会因为拿不到转写而报错。
 */
export function fallbackTranslit(text: string): TranslitForms {
    const latin = foldText(text);
    return { latin, compact: alnumOnly(latin), initials: "", variants: [] };
}

/** 把后端返回的形态补齐成完整结构（缺字段时按降级形态处理）。 */
export function normalizeTranslitForms(input: unknown, source: string): TranslitForms {
    if (!input || typeof input !== "object") return fallbackTranslit(source);
    const raw = input as Partial<Record<keyof TranslitForms, unknown>>;
    const fallback = fallbackTranslit(source);
    return {
        latin: typeof raw.latin === "string" ? raw.latin : fallback.latin,
        compact: typeof raw.compact === "string" ? raw.compact : fallback.compact,
        initials: typeof raw.initials === "string" ? raw.initials : "",
        variants: Array.isArray(raw.variants)
            ? raw.variants.filter((v): v is string => typeof v === "string")
            : [],
    };
}

/**
 * 构造查询形态。
 *
 * 【为什么查询串不在前端做转写】转写需要拼音/假名表，前端没有也不该有。查询串
 * 绝大多数是拉丁（用户就是用拼音/罗马字在搜），此时「折叠 + 去分隔符」就够了；
 * 查询串本身含 CJK 时，字面档直接命中 CJK 文本，也不需要转写。
 */
export function buildQuery(raw: string): QueryForms {
    const forms = fallbackTranslit(raw);
    return {
        literal: forms.latin,
        compact: forms.compact,
        initials: forms.compact,
        compacts: [forms.compact],
    };
}

function compactsOf(text: TranslitForms): string[] {
    return [text.compact, ...text.variants];
}

/** `needle` 的字符是否按顺序出现在 `hay` 中。 */
export function isSubsequence(hay: string, needle: string): boolean {
    if (needle.length === 0) return true;
    let index = 0;
    for (const ch of hay) {
        if (ch === needle[index]) {
            index += 1;
            if (index === needle.length) return true;
        }
    }
    return false;
}

function kindForFull(script: "latin" | "cjk"): MatchKind {
    return script === "latin" ? "literal" : "pinyin";
}

/**
 * 判定一条文本是否命中，返回最高档的命中说明；未命中返回 `null`。
 *
 * `cjkScript` 用于命名命中类型：中文（pinyin）/ 日文（romaji）/ 韩文初声（choseong）。
 * 前端拿不到后端的 `Script` 判定，但快捷键面板的命中提示不区分这三者，因此这里
 * 只区分「拉丁」与「CJK」。
 */
export function matchTranslit(
    text: TranslitForms,
    query: QueryForms,
    mode: SearchMode,
    cjkScript: "pinyin" | "romaji" | "choseong" = "pinyin",
): MatchInfo | null {
    if (query.literal.length === 0) return null;

    // A：字面子串。
    if (text.latin.includes(query.literal)) {
        return { kind: "literal", score: SCORE_LITERAL, form: query.literal };
    }
    if (mode === "off") return null;

    const hasCjk = text.initials.length > 0;
    const longEnough = [...query.compact].length >= MIN_TRANSLIT_LEN;

    // B/C：全拼前缀与子串。
    if (longEnough && query.compact.length > 0) {
        for (const hay of compactsOf(text)) {
            for (const needle of query.compacts) {
                if (needle.length > 0 && hay.startsWith(needle)) {
                    return {
                        kind: kindForFull(hasCjk ? "cjk" : "latin"),
                        score: SCORE_FULL_PREFIX,
                        form: needle,
                    };
                }
            }
        }
        for (const hay of compactsOf(text)) {
            for (const needle of query.compacts) {
                if (needle.length > 0 && hay.includes(needle)) {
                    return {
                        kind: kindForFull(hasCjk ? "cjk" : "latin"),
                        score: SCORE_FULL_SUBSTRING,
                        form: needle,
                    };
                }
            }
        }
    }

    // D/E/F：初声。
    if (text.initials.length > 0 && [...query.initials].length >= MIN_TRANSLIT_LEN) {
        if (text.initials === query.initials) {
            return { kind: cjkScript, score: SCORE_INITIALS_EXACT, form: query.initials };
        }
        if (text.initials.startsWith(query.initials)) {
            return { kind: cjkScript, score: SCORE_INITIALS_PREFIX, form: query.initials };
        }
        if (
            [...query.initials].length >= MIN_INITIALS_SUBSEQ_LEN &&
            isSubsequence(text.initials, query.initials)
        ) {
            return { kind: cjkScript, score: SCORE_INITIALS_SUBSEQ, form: query.initials };
        }
    }

    // G：模糊子序列（仅 fuzzy 模式）。
    if (
        mode === "fuzzy" &&
        [...query.compact].length >= MIN_FUZZY_LEN &&
        isSubsequence(text.compact, query.compact)
    ) {
        return { kind: "fuzzy", score: SCORE_FUZZY, form: query.compact };
    }

    return null;
}
