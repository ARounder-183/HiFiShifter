/*
 * 把记事本选区写进系统剪贴板（供右键菜单的"剪切 / 复制"使用）。
 *
 * ## 为什么不复用编辑器自己的 copy 事件
 *
 * 正文里 Ctrl+C 走的是浏览器原生复制：ProseMirror 写 `text/html` 与
 * `text/plain`，`installClipboardFlavorWriter` 再补 `text/markdown` 并按设置
 * 裁剪。那条路径的前提是**编辑器持有焦点且存在原生选区**。
 *
 * 右键菜单不满足这个前提：点菜单按钮会把焦点移到按钮上（Chromium 上按钮
 * 点击即聚焦），此时 `document.execCommand("copy")` 复制的是"当前文档选区"，
 * 而在菜单项上按下的那一刻它可能已经空了 —— 表现为"点了复制，粘出来什么都没有"。
 *
 * 所以这里**不依赖焦点、不依赖 DOM 选区**：直接读 ProseMirror 的 state
 * （选区存在 state 里，不受焦点影响），自己算出三种 flavor 再写出去。
 * 语义与 `installClipboardFlavorWriter` 完全一致 —— 同一个
 * `computeClipboardFlavors`，因此"复制格式"设置对两条路径同样生效。
 *
 * ## 为什么优先 `navigator.clipboard.write`
 *
 * 它是唯一能一次写入多个 flavor 的通道（`writeText` 只能写 `text/plain`，
 * 粘进 Word 会丢掉格式）。本仓已有先例证明这条路在 WebView2 下可用：
 * `NotebookImageNodeView` 用 `navigator.clipboard.write` 写 PNG。
 * 不可用时退回 `copyTextToClipboard`（`utils/copyText.ts` 的两级兜底），
 * 代价是只剩纯文本 —— 但"复制成功但没格式"远好于"复制失败"。
 */

import { getHTMLFromFragment, type Editor } from "@tiptap/core";

import { copyTextToClipboard } from "../../../utils/copyText";
import { computeClipboardFlavors } from "./notebookClipboard";

/** 复制内容的 flavor 来源：与编辑器内复制共用同一份设置契约。 */
export interface NotebookCopySettings {
    copyFormat: string;
    copyPlainTextAs: string;
}

/** 复制内容的 flavor 来源：覆盖 `copyFormat`，用于"复制为 Markdown / 纯文本"。 */
export type NotebookCopyFlavorOverride = "markdown" | "text";

/**
 * 把当前选区写进系统剪贴板。
 *
 * @param editor 编辑器实例（只读它的 state）。
 * @param settings 复制设置（决定写哪些 flavor）。
 * @param override 单次覆盖：菜单的"复制为 Markdown / 复制为纯文本"用它把
 *   `copyFormat` 顶掉一次，而**不改设置** —— 用户想这一次只要源码，不该顺手
 *   改掉他所有复制行为。
 * @returns 是否写入成功。空选区返回 `false`（调用方据此禁用菜单项）。
 */
export async function writeNotebookSelection(
    editor: Editor,
    settings: NotebookCopySettings,
    override?: NotebookCopyFlavorOverride,
): Promise<boolean> {
    if (editor.isDestroyed) return false;
    const { state } = editor;
    if (state.selection.empty) return false;

    const effective: NotebookCopySettings = override
        ? { copyFormat: override, copyPlainTextAs: override }
        : settings;
    const flavors = computeClipboardFlavors(editor, effective);

    const html = flavors.keepHtml ? selectionHtml(editor) : null;
    const wrote = await writeFlavors({
        plainText: flavors.plainText,
        html,
        markdown: flavors.markdown,
    });
    if (wrote) return true;

    // 多 flavor 通道不可用（旧 WebView / 非安全上下文）：退回纯文本。
    return copyTextToClipboard(flavors.plainText);
}

/** 取选区的 HTML。编辑器正在重建时返回 null（少写一个 flavor，不打断复制）。 */
function selectionHtml(editor: Editor): string | null {
    try {
        const { from, to } = editor.state.selection;
        const slice = editor.state.doc.cut(from, to);
        return getHTMLFromFragment(slice.content, editor.schema);
    } catch {
        return null;
    }
}

/** 一次写入多个 flavor。任一步不可用即返回 false，由调用方兜底。 */
async function writeFlavors(flavors: {
    plainText: string;
    html: string | null;
    markdown: string | null;
}): Promise<boolean> {
    if (typeof ClipboardItem === "undefined") return false;
    const clipboard = navigator.clipboard;
    if (!clipboard || typeof clipboard.write !== "function") return false;

    try {
        const entries: Record<string, Blob> = {
            // `text/plain` 必须有：没有它，只认纯文本的应用会粘贴出空内容。
            "text/plain": new Blob([flavors.plainText], { type: "text/plain" }),
        };
        if (flavors.html) {
            entries["text/html"] = new Blob([flavors.html], { type: "text/html" });
        }
        if (flavors.markdown) {
            // 非标准但无害：不认它的应用会忽略这个 flavor，认它的（Markdown
            // 编辑器、聊天窗口）能拿到未经 HTML 转换的源码。
            entries["text/markdown"] = new Blob([flavors.markdown], { type: "text/markdown" });
        }
        await clipboard.write([new ClipboardItem(entries)]);
        return true;
    } catch {
        return false;
    }
}
