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

export interface NotebookReadonlyPreviewProps {
    markdown: string;
    placeholder: string;
    /** 与左栏联动的滚动同步（比例同步）。 */
    scrollSyncSource?: HTMLElement | null;
}

export function NotebookReadonlyPreview({
    markdown,
    placeholder,
    scrollSyncSource,
}: NotebookReadonlyPreviewProps) {
    const extensions = useMemo(
        () =>
            buildNotebookExtensions({
                placeholder,
                markdownShortcuts: false,
                slashCommands: false,
            }),
        [placeholder],
    );

    const editor = useEditor({
        extensions,
        content: markdown,
        editable: false,
        editorProps: { attributes: { class: "hs-notebook-prose" } },
    });

    const containerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        if (!editor) return;
        editor.commands.setContent(markdown, { emitUpdate: false });
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
            className="h-full min-w-0 flex-1 overflow-auto border-l border-qt-border bg-qt-base px-3 py-3"
        >
            <EditorContent editor={editor} />
        </div>
    );
}
