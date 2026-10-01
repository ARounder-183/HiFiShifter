/**
 * 文本 → 转写形态的索引（异步、带降级）。
 *
 * 【为什么需要它】转写规则在后端（`transliterate` 命令），而快捷键面板的匹配必须
 * 在前端同步完成 —— 每次击键都走 IPC 会让输入变粘滞。折中是：**建索引时**批量取
 * 一次转写形态，之后击键只做纯字符串比较。
 *
 * 【为什么要「先降级、后替换」】后端返回之前不能让界面空着。降级形态在渲染期直接
 * 算出来（只做折叠，字面匹配立即可用），后端结果到达后覆盖它 —— 用户看不到加载态，
 * 转写能力只是「稍后生效」。后端不可用（浏览器 dev 模式）时停在降级形态，功能不中断。
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { searchApi } from "../../services/api/search";
import { effectiveSearchMode, searchOptionsPayload, type SearchSettings } from "./searchSettings";
import { fallbackTranslit, normalizeTranslitForms, type TranslitForms } from "./translit";

const EMPTY: Map<string, TranslitForms> = new Map();

function fallbackMap(texts: readonly string[]): Map<string, TranslitForms> {
    const map = new Map<string, TranslitForms>();
    for (const text of texts) map.set(text, fallbackTranslit(text));
    return map;
}

/**
 * 取一组文本的转写形态。
 *
 * @param texts 需要建索引的全部文案（去重由调用方负责，重复项在这里天然合并）。
 * @param settings 搜索设置；`mode = off`（或总开关关闭）时不做任何转写。
 */
export function useTranslitIndex(
    texts: readonly string[],
    settings: SearchSettings,
): Map<string, TranslitForms> {
    const enabled = effectiveSearchMode(settings) !== "off";
    // 文案数组每次渲染都可能是新引用；用内容做键，避免无谓的重建。
    const key = useMemo(() => (enabled ? texts.join("\u0000") : ""), [texts, enabled]);
    const payload = useMemo(() => searchOptionsPayload(settings), [settings]);

    /** 渲染期即可用的降级形态（只做折叠）。 */
    const fallback = useMemo(
        () => (enabled && texts.length > 0 ? fallbackMap(texts) : EMPTY),
        // 用 key 而不是 texts 作依赖：内容相同就不重算。
        // eslint-disable-next-line react-hooks/exhaustive-deps
        [key, enabled],
    );

    /*
     * 用 ref 持有最新数组，让下面的 effect 只依赖 `key`（字符串）而不是数组引用 ——
     * 数组引用每次渲染都可能变，直接进依赖会让 effect 反复重建索引。
     * 赋值放在 effect 里（而不是渲染期）：既满足「渲染期不写 ref」的规则，
     * 也保证它在下面的 effect 之前执行（同一提交内按声明顺序）。
     */
    const textsRef = useRef(texts);
    useEffect(() => {
        textsRef.current = texts;
    });

    /** 后端返回的形态。带 `key` 一起存，键一变就自动作废。 */
    const [remote, setRemote] = useState<{ key: string; map: Map<string, TranslitForms> } | null>(
        null,
    );

    useEffect(() => {
        if (!enabled || key === "") return;
        const list = textsRef.current;
        if (list.length === 0) return;
        let cancelled = false;
        void searchApi
            .transliterate([...list], payload)
            .then((results) => {
                if (cancelled) return;
                const map = new Map<string, TranslitForms>();
                list.forEach((text, index) => {
                    map.set(text, normalizeTranslitForms(results[index], text));
                });
                setRemote({ key, map });
            })
            .catch(() => {
                // 后端不可用：停在降级形态（字面匹配仍然工作）。
            });
        return () => {
            cancelled = true;
        };
    }, [key, enabled, payload]);

    return remote && remote.key === key ? remote.map : fallback;
}

/**
 * 把转写形态摊平成可检索的词条（全拼 + 多音字变体 + 初声）。
 *
 * 【为什么要过滤长度】单字符词条（`a`、`l`）会让任何含该字符的查询都命中，
 * 与后端 `MIN_TRANSLIT_LEN` 的用意一致。
 */
export function translitTermsOf(forms: TranslitForms | undefined): string[] {
    if (!forms) return [];
    const terms = [forms.compact, ...forms.variants, forms.initials];
    return terms.filter((term) => term.length >= 2);
}
