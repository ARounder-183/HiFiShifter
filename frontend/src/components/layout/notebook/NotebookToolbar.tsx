/*
 * 记事本格式化工具栏 + `/` 斜杠插入菜单。
 *
 * 工具栏刻意用**文字字形**（B / I / S / `<>` / H1 …）而不是图标包：Markdown
 * 的标记本身就是文字，字形与"这段会变成什么语法"一一对应，比抽象图标更
 * 直观，也不用为 20 个按钮去挑图标。
 *
 * 所有按钮都走 `editor.chain()` 命令，因此在源码视图（不挂编辑器）下工具栏
 * 会被整体隐藏 —— 面板负责这件事。
 */

import type { Editor } from "@tiptap/core";
import { useEffect, useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { insertTable, type ToolbarInsertHandlers } from "./notebookInsert";
import { useNotebookSlashMenu } from "./useNotebookSlashMenu";

export interface NotebookToolbarProps {
    editor: Editor;
    handlers: ToolbarInsertHandlers;
    /** 是否启用 `/` 斜杠菜单。 */
    slashCommands: boolean;
}

export function NotebookToolbar({ editor, handlers, slashCommands }: NotebookToolbarProps) {
    const { t } = useI18n();
    const slash = useNotebookSlashMenu(editor, slashCommands, handlers);
    /** 链接编辑浮层的草稿；null = 收起。替代 window.prompt：Tauri/WKWebView 下脚本对话框静默返回 null（见 ClipContextMenu 同款理由）。 */
    const [linkDraft, setLinkDraft] = useState<string | null>(null);

    // 选区变化即收起浮层：浮层编辑的是"当前选区"的链接，选区一旦移走，
    // 再确认就会把链接贴到错误的位置上。
    useEffect(() => {
        if (linkDraft === null) return;
        const close = () => setLinkDraft(null);
        editor.on("selectionUpdate", close);
        return () => {
            editor.off("selectionUpdate", close);
        };
    }, [editor, linkDraft]);

    /** 应用（Enter / OK）：与原 window.prompt 版本同一套命令链。 */
    const applyLink = () => {
        const href = (linkDraft ?? "").trim();
        setLinkDraft(null);
        if (!href) {
            editor.chain().focus().extendMarkRange("link").unsetLink().run();
            return;
        }
        editor.chain().focus().extendMarkRange("link").setLink({ href }).run();
    };

    return (
        <div className="hs-notebook-toolbar">
            <div className="hs-notebook-toolbar-group">
                <ToolbarButton
                    label="H1"
                    tooltip={t("notebook_toolbar_heading1")}
                    active={editor.isActive("heading", { level: 1 })}
                    onClick={() => editor.chain().focus().toggleHeading({ level: 1 }).run()}
                />
                <ToolbarButton
                    label="H2"
                    tooltip={t("notebook_toolbar_heading2")}
                    active={editor.isActive("heading", { level: 2 })}
                    onClick={() => editor.chain().focus().toggleHeading({ level: 2 }).run()}
                />
                <ToolbarButton
                    label="H3"
                    tooltip={t("notebook_toolbar_heading3")}
                    active={editor.isActive("heading", { level: 3 })}
                    onClick={() => editor.chain().focus().toggleHeading({ level: 3 }).run()}
                />
                <ToolbarButton
                    label="¶"
                    tooltip={t("notebook_toolbar_paragraph")}
                    active={editor.isActive("paragraph")}
                    onClick={() => editor.chain().focus().setParagraph().run()}
                />
            </div>

            <ToolbarSeparator />

            <div className="hs-notebook-toolbar-group">
                <ToolbarButton
                    label="B"
                    bold
                    tooltip={t("notebook_toolbar_bold")}
                    active={editor.isActive("bold")}
                    onClick={() => editor.chain().focus().toggleBold().run()}
                />
                <ToolbarButton
                    label="I"
                    italic
                    tooltip={t("notebook_toolbar_italic")}
                    active={editor.isActive("italic")}
                    onClick={() => editor.chain().focus().toggleItalic().run()}
                />
                <ToolbarButton
                    label="S"
                    strike
                    tooltip={t("notebook_toolbar_strike")}
                    active={editor.isActive("strike")}
                    onClick={() => editor.chain().focus().toggleStrike().run()}
                />
                <ToolbarButton
                    label="<>"
                    tooltip={t("notebook_toolbar_inline_code")}
                    active={editor.isActive("code")}
                    onClick={() => editor.chain().focus().toggleCode().run()}
                />
            </div>

            <ToolbarSeparator />

            <div className="hs-notebook-toolbar-group">
                <ToolbarButton
                    label="•"
                    tooltip={t("notebook_toolbar_bullet_list")}
                    active={editor.isActive("bulletList")}
                    onClick={() => editor.chain().focus().toggleBulletList().run()}
                />
                <ToolbarButton
                    label="1."
                    tooltip={t("notebook_toolbar_ordered_list")}
                    active={editor.isActive("orderedList")}
                    onClick={() => editor.chain().focus().toggleOrderedList().run()}
                />
                <ToolbarButton
                    label="☑"
                    tooltip={t("notebook_toolbar_task_list")}
                    active={editor.isActive("taskList")}
                    onClick={() => editor.chain().focus().toggleTaskList().run()}
                />
                <ToolbarButton
                    label="❝"
                    tooltip={t("notebook_toolbar_quote")}
                    active={editor.isActive("blockquote")}
                    onClick={() => editor.chain().focus().toggleBlockquote().run()}
                />
                <ToolbarButton
                    label="{ }"
                    tooltip={t("notebook_toolbar_code_block")}
                    active={editor.isActive("codeBlock")}
                    onClick={() => editor.chain().focus().toggleCodeBlock().run()}
                />
            </div>

            <ToolbarSeparator />

            <div className="hs-notebook-toolbar-group">
                <ToolbarButton
                    label="🔗"
                    tooltip={t("notebook_toolbar_link")}
                    active={editor.isActive("link") || linkDraft !== null}
                    onClick={() => setLinkDraft(String(editor.getAttributes("link").href ?? ""))}
                />
                <ToolbarButton
                    label="🖼"
                    tooltip={t("notebook_toolbar_image")}
                    onClick={handlers.insertImage}
                />
                <ToolbarButton
                    label="▦"
                    tooltip={t("notebook_toolbar_table")}
                    onClick={() => insertTable(editor)}
                />
                <ToolbarButton
                    label="―"
                    tooltip={t("notebook_toolbar_rule")}
                    onClick={() => editor.chain().focus().setHorizontalRule().run()}
                />
            </div>

            <ToolbarSeparator />

            <div className="hs-notebook-toolbar-group">
                <ToolbarButton
                    label="⏱"
                    tooltip={t("notebook_toolbar_timecode")}
                    onClick={handlers.insertTimecode}
                />
                <ToolbarButton
                    label="✂"
                    tooltip={t("notebook_toolbar_clip_ref")}
                    onClick={handlers.insertClipReference}
                />
                <ToolbarButton
                    label="📋"
                    tooltip={t("notebook_toolbar_stage_clipboard")}
                    onClick={handlers.stageClipboard}
                />
                <ToolbarButton
                    label="ℹ"
                    tooltip={t("notebook_toolbar_project_info")}
                    onClick={handlers.insertProjectInfo}
                />
            </div>

            {linkDraft !== null ? (
                <div className="hs-notebook-link-popover">
                    {/* 键盘可用：Enter 应用、Escape 取消；stopPropagation 挡掉
                        编辑器快捷键（输入框内不应触发 Ctrl+B 之类）。 */}
                    <input
                        autoFocus
                        value={linkDraft}
                        placeholder={t("notebook_link_prompt")}
                        aria-label={t("notebook_link_prompt")}
                        onChange={(event) => setLinkDraft(event.target.value)}
                        onKeyDown={(event) => {
                            event.stopPropagation();
                            if (event.key === "Enter") {
                                event.preventDefault();
                                applyLink();
                            } else if (event.key === "Escape") {
                                setLinkDraft(null);
                            }
                        }}
                    />
                    <button
                        type="button"
                        className="hs-notebook-toolbar-btn"
                        data-tooltip={t("ok")}
                        aria-label={t("ok")}
                        onPointerDown={(event) => event.preventDefault()}
                        onClick={applyLink}
                    >
                        ✓
                    </button>
                    <button
                        type="button"
                        className="hs-notebook-toolbar-btn"
                        data-tooltip={t("cancel")}
                        aria-label={t("cancel")}
                        onPointerDown={(event) => event.preventDefault()}
                        onClick={() => setLinkDraft(null)}
                    >
                        ✕
                    </button>
                </div>
            ) : null}

            {slash.menu ? (
                <div
                    className="hs-notebook-slash-menu"
                    style={{ left: slash.menu.x, top: slash.menu.y }}
                    onPointerDown={(event) => event.preventDefault()}
                >
                    {slash.items.map((item) => (
                        <button key={item.key} type="button" onClick={() => slash.run(item.key)}>
                            <span className="hs-notebook-slash-label">{item.label}</span>
                            <span className="hs-notebook-slash-hint">{item.hint}</span>
                        </button>
                    ))}
                </div>
            ) : null}
        </div>
    );
}

function ToolbarButton({
    label,
    tooltip,
    active,
    bold,
    italic,
    strike,
    onClick,
}: {
    label: string;
    /** 悬停提示文本；渲染为项目自定义 tooltip 的 `data-tooltip`。 */
    tooltip: string;
    active?: boolean;
    bold?: boolean;
    italic?: boolean;
    strike?: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            className="hs-notebook-toolbar-btn"
            data-active={active ? "true" : "false"}
            data-tooltip={tooltip}
            aria-label={tooltip}
            style={{
                fontWeight: bold ? 700 : undefined,
                fontStyle: italic ? "italic" : undefined,
                textDecoration: strike ? "line-through" : undefined,
            }}
            // 按下时不抢焦点：编辑器保持选区，命令才能作用于正确的范围。
            onPointerDown={(event) => event.preventDefault()}
            onClick={onClick}
        >
            {label}
        </button>
    );
}

function ToolbarSeparator() {
    return <span className="hs-notebook-toolbar-sep" />;
}
