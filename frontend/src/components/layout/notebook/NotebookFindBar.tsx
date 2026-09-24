/*
 * 面板内查找。
 *
 * 全局 Ctrl+F 被应用拦截（用于它自己的快速搜索），记事本因此自带一条查找条：
 * - 富文本视图：在 ProseMirror 文档里按文本节点定位，命中即选中并滚动到可见；
 * - 源码视图：直接在源码 textarea 上设置选区。
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
    /** 取源码视图的 textarea（源码模式下定位选区用）。 */
    getSourceTextarea: () => HTMLTextAreaElement | null;
    onClose: () => void;
}

export function NotebookFindBar({
    editor,
    sourceMode,
    sourceValue,
    getSourceTextarea,
    onClose,
}: NotebookFindBarProps) {
    const { t } = useI18n();
    const [query, setQuery] = useState("");
    const [cursor, setCursor] = useState(0);

    const richMatches = useMemo(
        () =>
            !sourceMode && editor && !editor.isDestroyed && query
                ? findMatchesInDoc(editor, query)
                : [],
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
                const textarea = getSourceTextarea();
                if (!textarea) return;
                textarea.focus();
                textarea.setSelectionRange(match.from, match.to);
                // 把命中行滚到可视区中部附近。
                const line = sourceValue.slice(0, match.from).split("\n").length;
                const lineHeight = Number.parseFloat(getComputedStyle(textarea).lineHeight);
                textarea.scrollTop = Math.max(
                    0,
                    (line - 4) * (Number.isFinite(lineHeight) ? lineHeight : 18),
                );
                return;
            }
            if (!editor || editor.isDestroyed) return;
            editor
                .chain()
                .focus()
                .setTextSelection({ from: match.from, to: match.to })
                .scrollIntoView()
                .run();
        },
        [editor, getSourceTextarea, matches, sourceMode, sourceValue],
    );

    // goTo 每次渲染身份都会变，放进 ref 以免把下面的"查询变化即跳转"变成
    // 每次渲染都执行。
    const goToRef = useRef(goTo);
    useEffect(() => {
        goToRef.current = goTo;
    }, [goTo]);

    // 查询变化时跳到第一处；不自动跳转会让用户以为"没找到"。
    useEffect(() => {
        if (matches.length === 0) return;
        goToRef.current(0);
    }, [query, matches.length]);

    return (
        <div className="hs-notebook-findbar">
            <input
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
                {matches.length === 0
                    ? t("notebook_find_no_match")
                    : `${cursor + 1}/${matches.length}`}
            </span>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                data-tooltip={t("notebook_find_prev")}
                onClick={() => goTo(cursor - 1)}
            >
                ▲
            </button>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                data-tooltip={t("notebook_find_next")}
                onClick={() => goTo(cursor + 1)}
            >
                ▼
            </button>
            <button
                type="button"
                className="hs-notebook-toolbar-btn"
                data-tooltip={t("close")}
                onClick={onClose}
            >
                ✕
            </button>
        </div>
    );
}
