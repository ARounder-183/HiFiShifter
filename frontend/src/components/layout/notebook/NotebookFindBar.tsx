/*
 * 面板内查找/替换。
 *
 * 全局 Ctrl+F 被应用拦截（用于它自己的快速搜索），记事本因此自带一条查找条：
 * - 富文本视图：在 ProseMirror 文档里按文本节点定位，命中即选中并滚动到可见；
 * - 源码视图：直接在 textarea 的字符串上定位并设置选区。
 *
 * 两种视图共用同一套"匹配 → 上/下一处"的交互，切换视图不会丢掉查询词。
 */

import type { Editor } from "@tiptap/core";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { findMatchesInDoc, findMatchesInText, type TextMatch } from "./notebookFind";

export interface NotebookFindBarProps {
    editor: Editor | null;
    sourceMode: boolean;
    sourceValue: string;
    onClose: () => void;
}

export function NotebookFindBar({ editor, sourceMode, sourceValue, onClose }: NotebookFindBarProps) {
    const { t } = useI18n();
    const [query, setQuery] = useState("");
    const [cursor, setCursor] = useState(0);
    const textareaRef = useRef<HTMLTextAreaElement | null>(null);

    const richMatches = useMemo(
        () => (!sourceMode && editor && query ? findMatchesInDoc(editor, query) : []),
        [editor, query, sourceMode],
    );

    const sourceMatches = useMemo(
        () => (sourceMode && query ? findMatchesInText(sourceValue, query) : ([] as TextMatch[])),
        [query, sourceMode, sourceValue],
    );

    const matches = sourceMode ? sourceMatches : richMatches;

    const goTo = useCallback(
        (next: number) => {
            if (matches.length === 0) return;
            const wrapped = ((next % matches.length) + matches.length) % matches.length;
            setCursor(wrapped);
            const match = matches[wrapped];
            if (sourceMode) {
                const textarea = textareaRef.current;
                if (!textarea) return;
                textarea.focus();
                textarea.setSelectionRange(match.from, match.to);
                // 把命中行滚到可视区中部附近。
                const before = sourceValue.slice(0, match.from);
                const line = before.split("\n").length;
                const lineHeight = Number.parseFloat(getComputedStyle(textarea).lineHeight) || 18;
                textarea.scrollTop = Math.max(0, (line - 4) * lineHeight);
                return;
            }
            if (!editor) return;
            editor.chain().focus().setTextSelection({ from: match.from, to: match.to }).scrollIntoView().run();
        },
        [editor, matches, sourceMode, sourceValue],
    );

    // 查询变化时跳到第一处；不自动跳转会让用户以为"没找到"。
    useEffect(() => {
        setCursor(0);
        if (matches.length > 0) goTo(0);
        // 只依赖查询词与匹配数量：goTo 每次渲染身份都变，不能进依赖。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [query, matches.length]);

    return (
        <div className="hs-notebook-findbar">
            <input
                ref={textareaRef as unknown as React.RefObject<HTMLInputElement>}
                autoFocus
                value={query}
                placeholder={t("notebook_find_placeholder")}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                    event.stopPropagation();
                    if (event.key === "Enter") {
                        goTo(cursor + (event.shiftKey ? -1 : 1));
                    } else if (event.key === "Escape") {
                        onClose();
                    }
                }}
            />
            <span className="hs-notebook-findbar-count">
                {matches.length === 0 ? t("notebook_find_no_match") : `${cursor + 1}/${matches.length}`}
            </span>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                title={t("notebook_find_prev")}
                onClick={() => goTo(cursor - 1)}
            >
                ▲
            </button>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                title={t("notebook_find_next")}
                onClick={() => goTo(cursor + 1)}
            >
                ▼
            </button>
            <button type="button" className="hs-notebook-toolbar-btn" title={t("close")} onClick={onClose}>
                ✕
            </button>
        </div>
    );
}
