/**
 * 命中原因的文案：把后端的 `matchInfo` 变成「· 匹配拼音 zhuge」这样一行说明。
 *
 * 【为什么必须显示】转写匹配最大的风险不是误命中，而是**不可解释** —— 用户打 `zge`
 * 冒出「主歌.wav」，他不知道这条为什么会在。字面命中不显示（理由本来就看得见），
 * 只有转写/模糊命中才补这一行；它同时也是功能的教学入口。
 */

import type { MessageKey } from "../../../i18n/messages";
import type { FileMatchInfo } from "../../../services/api/fileBrowser";

/**
 * 命中类型 → 词典键的显式映射。
 *
 * 【为什么不用 `search_matched_${kind}` 模板拼接】模板拼键名时，词典里少一个键
 * 不会报错 —— `t()` 拿到不存在的键会原样返回键名，界面上就出现
 * `search_matched_choseong` 这样的英文键。写成显式表后，两个方向都被类型检查守住。
 */
const MATCH_KIND_LABEL_KEY: Record<FileMatchInfo["kind"], MessageKey> = {
    literal: "search_matched_literal",
    pinyin: "search_matched_pinyin",
    romaji: "search_matched_romaji",
    choseong: "search_matched_choseong",
    fuzzy: "search_matched_fuzzy",
};

/** 命中类型是否值得向用户解释（字面命中的理由本来就看得见）。 */
export function shouldExplainMatch(kind: FileMatchInfo["kind"]): boolean {
    return kind !== "literal";
}

/**
 * 生成命中说明的词典键与插值参数。
 *
 * 返回 `null` 表示不需要说明（字面命中 / 没有命中信息 / 用户关掉了这个显示项）。
 */
export function matchReasonOf(
    matchInfo: FileMatchInfo | undefined,
    showMatchReason: boolean,
): { key: MessageKey; vars: { form: string } } | null {
    if (!matchInfo || !showMatchReason) return null;
    if (!shouldExplainMatch(matchInfo.kind)) return null;
    return { key: MATCH_KIND_LABEL_KEY[matchInfo.kind], vars: { form: matchInfo.form } };
}
