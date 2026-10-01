/**
 * 搜索匹配设置：类型、默认值与归一化。
 *
 * 【为什么要单独成文件】这套设置被四处消费 —— 文件浏览器、快速搜索、快捷键面板、
 * 外观设置的字体过滤 —— 且要持久化到 `app_config.json` 的 `ui.search`。类型与
 * 归一化集中在这里，四处才不会各自解释「缺字段时算什么」。
 *
 * 【为什么「总开关 + 模式」是两个字段】用户说「我想关掉」和「我想收紧」是两件事。
 * 合成一个三态枚举会让两者互相干扰：把宽严从「模糊」调回「智能」会顺带把功能打开，
 * 而用户上次明明是关掉的。分开之后，总开关负责「这个功能存不存在」，模式负责
 * 「匹配得多宽」；两者的合成只发生在下发命令的那一刻（见 `searchOptionsPayload`）。
 */

import type { MessageKey } from "../../i18n/messages";

/** 匹配宽严。与后端 `search::SearchMode` 的 serde 取值一一对应。 */
export type SearchMode = "off" | "smart" | "fuzzy";

export interface SearchSettings {
    /** 转写匹配总开关。关闭等价于 `mode = "off"`。 */
    translit: boolean;
    /** `off` / `smart`（默认）/ `fuzzy`。 */
    mode: SearchMode;
    /** 多音字展开：`chongzuo` 也能命中「重做」。 */
    heteronym: boolean;
    /** 日文长音宽松：`bokaru` 也能命中「ボーカル」。 */
    japaneseLongVowel: boolean;
    /** 韩文初声：`hg` 也能命中「한국어」。 */
    koreanChoseong: boolean;
    /** 结果里显示「为什么命中」。 */
    showMatchReason: boolean;
}

export const DEFAULT_SEARCH_SETTINGS: SearchSettings = {
    translit: true,
    mode: "smart",
    heteronym: true,
    japaneseLongVowel: true,
    koreanChoseong: true,
    showMatchReason: true,
};

const SEARCH_MODES: readonly SearchMode[] = ["off", "smart", "fuzzy"];

function asMode(value: unknown, fallback: SearchMode): SearchMode {
    return typeof value === "string" && (SEARCH_MODES as readonly string[]).includes(value)
        ? (value as SearchMode)
        : fallback;
}

/**
 * 归一化一份可能来自旧配置 / 手改文件的设置。
 *
 * 【为什么缺省一律「开」】转写匹配是**召回**功能：它的失败模式是「该找到的没找到」，
 * 而用户往往意识不到少打一个键就能找到，只会以为文件不在。它的代价（误命中）由
 * 分档排序兜住 —— 真正像的那条永远排第一。唯一例外是 `fuzzy`，它召回增益小而误命中
 * 代价大，因此默认停在 `smart`。
 */
export function normalizeSearchSettings(input: unknown): SearchSettings {
    if (!input || typeof input !== "object") return { ...DEFAULT_SEARCH_SETTINGS };
    const raw = input as Partial<Record<keyof SearchSettings, unknown>>;
    return {
        translit: typeof raw.translit === "boolean" ? raw.translit : true,
        mode: asMode(raw.mode, "smart"),
        heteronym: typeof raw.heteronym === "boolean" ? raw.heteronym : true,
        japaneseLongVowel:
            typeof raw.japaneseLongVowel === "boolean" ? raw.japaneseLongVowel : true,
        koreanChoseong: typeof raw.koreanChoseong === "boolean" ? raw.koreanChoseong : true,
        showMatchReason: typeof raw.showMatchReason === "boolean" ? raw.showMatchReason : true,
    };
}

/** 生效的匹配模式：总开关关掉时一律是 `off`，不再看 `mode` 存的是什么。 */
export function effectiveSearchMode(settings: SearchSettings): SearchMode {
    return settings.translit ? settings.mode : "off";
}

/** 下发给后端 `search_files_recursive` / `transliterate` 的参数。 */
export interface SearchOptionsPayload {
    mode: SearchMode;
    heteronym: boolean;
    japaneseLongVowel: boolean;
    koreanChoseong: boolean;
}

export function searchOptionsPayload(settings: SearchSettings): SearchOptionsPayload {
    return {
        mode: effectiveSearchMode(settings),
        heteronym: settings.heteronym,
        japaneseLongVowel: settings.japaneseLongVowel,
        koreanChoseong: settings.koreanChoseong,
    };
}

/**
 * 匹配模式 → 词典键的**显式**映射。
 *
 * 【为什么不用 `search_mode_${mode}` 模板拼接】模板拼键名意味着词典里少一个键也
 * 不会报错 —— `t()` 拿到不存在的键会原样返回键名，界面上就出现 `search_mode_fuzzy`
 * 这样的英文键。写成显式表后，两个方向都被类型检查守住。
 */
export const SEARCH_MODE_LABEL_KEY: Record<SearchMode, MessageKey> = {
    off: "search_mode_off",
    smart: "search_mode_smart",
    fuzzy: "search_mode_fuzzy",
};
