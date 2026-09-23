/*
 * `/` 唤出的插入菜单。
 *
 * 独立成模块：`NotebookToolbar.tsx` 只导出组件才能保住 React Fast Refresh；
 * 而且这个 hook 自己就是一块独立逻辑（触发判定 + 过滤 + 执行）。
 */

import type { Editor } from "@tiptap/core";
import { useEffect, useMemo, useState } from "react";

import { useI18n } from "../../../i18n/I18nProvider";
import { insertTable, type ToolbarInsertHandlers } from "./notebookInsert";

export type SlashActionKey =
    | "heading1"
    | "heading2"
    | "heading3"
    | "bulletList"
    | "orderedList"
    | "taskList"
    | "quote"
    | "codeBlock"
    | "table"
    | "image"
    | "timecode"
    | "clipRef"
    | "projectInfo"
    | "stageClipboard";

export interface SlashMenuItem {
    key: SlashActionKey;
    label: string;
    hint: string;
}

export interface SlashMenuState {
    x: number;
    y: number;
    /** `/` 所在位置，插入后要把它和查询串删掉。 */
    from: number;
    to: number;
}

export interface SlashMenuResult {
    menu: SlashMenuState | null;
    items: SlashMenuItem[];
    run: (key: SlashActionKey) => void;
}

/**
 * 触发条件刻意收窄：**光标所在段落只有 `/` 加若干字母**时才弹出。
 *
 * 这样在正文里写 "and/or"、"24/7" 不会弹出菜单 —— 误触比少一个入口更烦人。
 */
export function useNotebookSlashMenu(
    editor: Editor,
    enabled: boolean,
    handlers: ToolbarInsertHandlers,
): SlashMenuResult {
    const { t } = useI18n();
    const [menu, setMenu] = useState<SlashMenuState | null>(null);
    const [query, setQuery] = useState("");

    const allItems = useMemo<SlashMenuItem[]>(
        () => [
            { key: "heading1", label: t("notebook_toolbar_heading1"), hint: "#" },
            { key: "heading2", label: t("notebook_toolbar_heading2"), hint: "##" },
            { key: "heading3", label: t("notebook_toolbar_heading3"), hint: "###" },
            { key: "bulletList", label: t("notebook_toolbar_bullet_list"), hint: "-" },
            { key: "orderedList", label: t("notebook_toolbar_ordered_list"), hint: "1." },
            { key: "taskList", label: t("notebook_toolbar_task_list"), hint: "- [ ]" },
            { key: "quote", label: t("notebook_toolbar_quote"), hint: ">" },
            { key: "codeBlock", label: t("notebook_toolbar_code_block"), hint: "```" },
            { key: "table", label: t("notebook_toolbar_table"), hint: "| a |" },
            { key: "image", label: t("notebook_toolbar_image"), hint: "![]" },
            { key: "timecode", label: t("notebook_toolbar_timecode"), hint: "hifi://" },
            { key: "clipRef", label: t("notebook_toolbar_clip_ref"), hint: "hifi://" },
            { key: "projectInfo", label: t("notebook_toolbar_project_info"), hint: "" },
            { key: "stageClipboard", label: t("notebook_toolbar_stage_clipboard"), hint: "hifi-clip" },
        ],
        [t],
    );

    useEffect(() => {
        // 关闭时也照常注册监听：只在回调里判 `enabled` 并清掉菜单，
        // 避免在 effect 体里同步 setState（会触发级联渲染）。
        const sync = () => {
            if (!enabled) {
                setMenu(null);
                return;
            }
            const { state, view } = editor;
            const { $from, empty } = state.selection;
            if (!empty || $from.parent.type.name !== "paragraph") {
                setMenu(null);
                return;
            }
            const textBefore = $from.parent.textBetween(0, $from.parentOffset, undefined, "\uFFFC");
            const match = /^\/([A-Za-z]*)$/.exec(textBefore);
            if (!match) {
                setMenu(null);
                return;
            }
            const coords = view.coordsAtPos($from.pos);
            setQuery(match[1].toLowerCase());
            setMenu({
                x: coords.left,
                y: coords.bottom + 2,
                from: $from.start(),
                to: $from.pos,
            });
        };
        const onBlur = () => setMenu(null);
        editor.on("update", sync);
        editor.on("selectionUpdate", sync);
        editor.on("blur", onBlur);
        return () => {
            editor.off("update", sync);
            editor.off("selectionUpdate", sync);
            editor.off("blur", onBlur);
        };
    }, [editor, enabled]);

    const items = useMemo(() => {
        if (!query) return allItems;
        return allItems.filter((item) => matchesQuery(item, query));
    }, [allItems, query]);

    function run(key: SlashActionKey) {
        const current = menu;
        setMenu(null);
        if (current) {
            // 先删掉 `/query`，再执行插入 —— 否则菜单文字会留在正文里。
            editor.chain().focus().deleteRange({ from: current.from, to: current.to }).run();
        }
        switch (key) {
            case "heading1":
                editor.chain().focus().toggleHeading({ level: 1 }).run();
                return;
            case "heading2":
                editor.chain().focus().toggleHeading({ level: 2 }).run();
                return;
            case "heading3":
                editor.chain().focus().toggleHeading({ level: 3 }).run();
                return;
            case "bulletList":
                editor.chain().focus().toggleBulletList().run();
                return;
            case "orderedList":
                editor.chain().focus().toggleOrderedList().run();
                return;
            case "taskList":
                editor.chain().focus().toggleTaskList().run();
                return;
            case "quote":
                editor.chain().focus().toggleBlockquote().run();
                return;
            case "codeBlock":
                editor.chain().focus().toggleCodeBlock().run();
                return;
            case "table":
                insertTable(editor);
                return;
            case "image":
                handlers.insertImage();
                return;
            case "timecode":
                handlers.insertTimecode();
                return;
            case "clipRef":
                handlers.insertClipReference();
                return;
            case "projectInfo":
                handlers.insertProjectInfo();
                return;
            case "stageClipboard":
                handlers.stageClipboard();
                return;
            default:
                return;
        }
    }

    return { menu, items, run };
}

/**
 * 匹配规则：标签包含查询串，或命令键以查询串开头。
 *
 * 不做拼音匹配：那需要一份拼音表，收益（少打两个键）远不及它的体积与维护成本。
 */
function matchesQuery(item: SlashMenuItem, query: string): boolean {
    const label = item.label.toLowerCase();
    const hint = item.hint.toLowerCase();
    return label.includes(query) || hint.includes(query) || item.key.toLowerCase().startsWith(query);
}
