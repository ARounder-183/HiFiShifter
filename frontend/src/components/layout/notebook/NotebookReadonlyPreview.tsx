/*
 * 分栏模式右侧的只读渲染。
 *
 * 直接用同一个编辑器内核（同一套扩展、同一套 CSS）渲染，`editable: false`：
 * 这样"富文本视图"和"预览"永远长得一样 —— 用另一套渲染器（比如另写一个
 * markdown→HTML）必然会漂移，用户会看到两个不同版本的同一条笔记。
 */

import { EditorContent, useEditor } from "@tiptap/react";
import { useEffect, useMemo, useRef } from "react";

import { buildNotebookExtensions } from "./notebookExtensions";

/** 只读预览内容同步的去抖窗口（毫秒）。 */
const PREVIEW_SYNC_DEBOUNCE_MS = 250;

export interface NotebookReadonlyPreviewProps {
    markdown: string;
    /** 与左栏联动的滚动同步（比例同步）。 */
    scrollSyncSource?: HTMLElement | null;
}

export function NotebookReadonlyPreview({
    markdown,
    scrollSyncSource,
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

    return (
        <div
            ref={containerRef}
            className="hs-scroll-gutter-flush h-full min-w-0 flex-1 overflow-auto border-l border-qt-border bg-qt-base px-3 py-3"
        >
            <EditorContent editor={editor} />
        </div>
    );
}
