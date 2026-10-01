import { invoke } from "../invoke";
import type { SearchOptionsPayload } from "../../features/search/searchSettings";

/**
 * 后端转写产出的可检索形态（与 `search::TranslitResult` 对齐）。
 *
 * 前端只拿它做字符串比较，不重复实现任何语言知识（拼音表、假名表都不在前端）。
 */
export interface TranslitResult {
    latin: string;
    compact: string;
    initials: string;
    variants: string[];
}

export const searchApi = {
    /**
     * 批量转写文本。
     *
     * 【什么时候调用】建索引时一次（快捷键面板的动作名与分组名、字体名），
     * **不在每次击键的路径上** —— 击键仍然走前端纯 JS 的索引匹配。
     */
    transliterate: (texts: string[], options?: SearchOptionsPayload) =>
        invoke<TranslitResult[]>("transliterate", texts, options),
};
