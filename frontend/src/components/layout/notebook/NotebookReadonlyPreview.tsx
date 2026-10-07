/*
 * 分栏模式右侧的只读渲染。
 *
 * 直接用同一个编辑器内核（同一套扩展、同一套 CSS）渲染，`editable: false`：
 * 这样"富文本视图"和"预览"永远长得一样 —— 用另一套渲染器（比如另写一个
 * markdown→HTML）必然会漂移，用户会看到两个不同版本的同一条笔记。
 */

import { EditorContent, useEditor } from "@tiptap/react";
import type { Editor } from "@tiptap/core";
import { useCallback, useEffect, useMemo, useRef } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";

import {
    prepareNotebookContext,
    type NotebookMenuFlags,
    type NotebookMenuTarget,
} from "./notebookContextTarget";
import { buildNotebookExtensions } from "./notebookExtensions";

/** 只读预览内容同步的去抖窗口（毫秒）。 */
const PREVIEW_SYNC_DEBOUNCE_MS = 250;

/** 右键落点解析的结果（由预览栏自己算，因为编辑器是它自己的）。 */
export interface NotebookPreviewContextRequest {
    x: number;
    y: number;
    target: NotebookMenuTarget;
    flags: NotebookMenuFlags;
    editor: Editor;
}

export interface NotebookReadonlyPreviewProps {
    markdown: string;
    /** 与左栏联动的滚动同步（比例同步）。 */
    scrollSyncSource?: HTMLElement | null;
    /**
     * 右键菜单。
     *
     * 【为什么把落点解析放在这里】本组件有**自己的**编辑器实例（同一套扩展、
     * `editable: false`）。落点几何与开关都得问它要 —— 拿左栏那份去解析会得到
     * 错位的落点，而且它的 `editable: true` 会让菜单错误地给出编辑项。
     * 菜单**内容**仍由面板统一决定（规则只有一份）。
     */
    onContextMenu?: (request: NotebookPreviewContextRequest) => void;
}

export function NotebookReadonlyPreview({
    markdown,
    scrollSyncSource,
    onContextMenu,
}: NotebookReadonlyPreviewProps) {
    const extensions = useMemo(
        () =>
            buildNotebookExtensions({
                markdownShortcuts: false,
                slashCommands: false,
            }),
        [],
    );

    const editor = useEditor({
        extensions,
        content: markdown,
        editable: false,
        editorProps: { attributes: { class: "hs-notebook-prose" } },
    });

    const containerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!editor || editor.isDestroyed) return;
        // 去抖：主编辑器每敲一键都会推新 markdown 进来，而一次同步要重跑
        // markdown-it 解析 + ProseMirror 整篇替换。大笔记下这是肉眼可见的
        // 打字卡顿；200-300ms 的延迟对滚动同步与内容一致性没有可感影响。
        const timer = window.setTimeout(() => {
            if (editor.isDestroyed) return;
            try {
                editor.commands.setContent(markdown, { emitUpdate: false });
            } catch {
                // 编辑器正在重建：内容会在下次渲染时重新同步。
            }
        }, PREVIEW_SYNC_DEBOUNCE_MS);
        return () => window.clearTimeout(timer);
    }, [editor, markdown]);

    // 比例滚动同步：两侧排版一致，但行高与图片加载会带来高度差，因此按
    // "滚动百分比"对齐而不是按像素。反向同步加锁避免互相触发形成回环。
    useEffect(() => {
        const target = containerRef.current;
        if (!scrollSyncSource || !target) return;
        let syncing = false;
        const source = scrollSyncSource;
        const onSourceScroll = () => {
            if (syncing) return;
            syncing = true;
            const sourceRange = source.scrollHeight - source.clientHeight;
            const targetRange = target.scrollHeight - target.clientHeight;
            if (sourceRange > 0 && targetRange > 0) {
                target.scrollTop = (source.scrollTop / sourceRange) * targetRange;
            }
            syncing = false;
        };
        source.addEventListener("scroll", onSourceScroll, { passive: true });
        return () => source.removeEventListener("scroll", onSourceScroll);
    }, [scrollSyncSource]);

    const handleContextMenu = useCallback(
        (event: ReactMouseEvent) => {
            if (!onContextMenu || !editor || editor.isDestroyed) return;
            event.preventDefault();
            // 只读编辑器：落点不会移动任何光标（`applyContextSelection` 在只读下
            // 也只是改选区，而只读下改选区是允许的 —— 用户正是在选他要复制的东西）。
            const { target, flags } = prepareNotebookContext(editor, event.clientX, event.clientY);
            onContextMenu({
                x: event.clientX,
                y: event.clientY,
                target,
                flags,
                editor,
            });
        },
        [editor, onContextMenu],
    );

    return (
        <div
            ref={containerRef}
            className="hs-scroll-gutter-flush h-full min-w-0 flex-1 overflow-auto border-l border-qt-border bg-qt-base px-3 py-3"
            onContextMenu={handleContextMenu}
        >
            <EditorContent editor={editor} />
        </div>
    );
}
