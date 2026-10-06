/*
 * 记事本格式化工具栏 + `/` 斜杠插入菜单。
 *
 * 【两套符号，各有理由】Markdown 的标记型按钮刻意用**文字字形**
 * （B / I / S / `<>` / H1 / • / 1. / { } / ―）：标记本身就是文字，字形与
 * "这段会变成什么语法"一一对应。
 *
 * 而**图示型动作**（链接 / 图片 / 表格 / 时间码 / 片段引用 / 暂存到暂存区 /
 * 工程信息 / 任务清单 / 引用）此前用 emoji（🔗 🖼 ▦ ⏱ ✂ 📋 ℹ ☑ ❝）——
 * emoji 的渲染随平台与字体变化（Windows 上是彩色字形、macOS 上是另一套），
 * 同一排按钮里深浅与尺寸都不齐，和其余按钮的线性图标语言也冲突。它们没有
 * "语法自示"的价值，改用 `@radix-ui/react-icons`。
 *
 * 所有按钮都走 `editor.chain()` 命令，因此在源码视图（不挂编辑器）下工具栏
 * 会被整体隐藏 —— 面板负责这件事。
 */

import type { Editor } from "@tiptap/core";
import type { ReactNode } from "react";
import {
    CheckboxIcon,
    ClipboardIcon,
    ImageIcon,
    InfoCircledIcon,
    Link2Icon,
    QuoteIcon,
    ScissorsIcon,
    TableIcon,
    TimerIcon,
} from "@radix-ui/react-icons";

import { useI18n } from "../../../i18n/I18nProvider";
import { insertTable, type ToolbarInsertHandlers } from "./notebookInsert";
import { useNotebookSlashMenu } from "./useNotebookSlashMenu";

export interface NotebookToolbarProps {
    editor: Editor;
    handlers: ToolbarInsertHandlers;
    /** 是否启用 `/` 斜杠菜单。 */
    slashCommands: boolean;
    /**
     * 打开链接地址浮层。
     *
     * 浮层本身由面板渲染，不在这里 —— 工具栏可被收起，而 Ctrl/⌘+K 不该跟着
     * 失效（见 `NotebookLinkEditor` 的文件头）。
     */
    onEditLink: () => void;
    /** 链接浮层是否正开着（决定按钮的激活态）。 */
    linkEditorOpen: boolean;
}

export function NotebookToolbar({
    editor,
    handlers,
    slashCommands,
    onEditLink,
    linkEditorOpen,
}: NotebookToolbarProps) {
    const { t, shortcut } = useI18n();
    const slash = useNotebookSlashMenu(editor, slashCommands, handlers);

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
                    tooltip={shortcut("notebook_toolbar_bold")}
                    active={editor.isActive("bold")}
                    onClick={() => editor.chain().focus().toggleBold().run()}
                />
                <ToolbarButton
                    label="I"
                    italic
                    tooltip={shortcut("notebook_toolbar_italic")}
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
                    label={<CheckboxIcon />}
                    tooltip={t("notebook_toolbar_task_list")}
                    active={editor.isActive("taskList")}
                    onClick={() => editor.chain().focus().toggleTaskList().run()}
                />
                <ToolbarButton
                    label={<QuoteIcon />}
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
                    label={<Link2Icon />}
                    tooltip={shortcut("notebook_toolbar_link")}
                    active={editor.isActive("link") || linkEditorOpen}
                    onClick={onEditLink}
                />
                <ToolbarButton
                    label={<ImageIcon />}
                    tooltip={t("notebook_toolbar_image")}
                    onClick={handlers.insertImage}
                />
                <ToolbarButton
                    label={<TableIcon />}
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
                    label={<TimerIcon />}
                    tooltip={t("notebook_toolbar_timecode")}
                    onClick={handlers.insertTimecode}
                />
                <ToolbarButton
                    label={<ScissorsIcon />}
                    tooltip={t("notebook_toolbar_clip_ref")}
                    onClick={handlers.insertClipReference}
                />
                <ToolbarButton
                    label={<ClipboardIcon />}
                    tooltip={t("notebook_toolbar_stage_clipboard")}
                    onClick={handlers.stageClipboard}
                />
                <ToolbarButton
                    label={<InfoCircledIcon />}
                    tooltip={t("notebook_toolbar_project_info")}
                    onClick={handlers.insertProjectInfo}
                />
            </div>

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
    /** 文字字形或图标（图示型动作用图标，见文件头）。 */
    label: ReactNode;
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
