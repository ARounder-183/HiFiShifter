// @vitest-environment jsdom
/*
 * 链接写入的回归测试（真实 TipTap 编辑器）。
 *
 * 【要钉死什么】`setLink` 在**空选区**上只写"存储标记"（stored marks）—— 它
 * 作用于下一个键入的字符，文档当场毫无变化。曾经的实现只有这一条路径，于是
 * 用户填完地址按回车看到的是"什么都没发生"，只能自己再打一遍文字。
 *
 * 【为什么必须用真实编辑器】"存储标记被写下"与"链接出现在文档里"是两件事，
 * 而只有真实编辑器能区分它们：前者用 `setLink` 就能做到、断言命令返回 true 也
 * 会通过；后者必须读 `doc` 才能看见。这里的断言全部落在**序列化后的正文**上。
 */

import { Editor } from "@tiptap/core";
import { test } from "vitest";

import { applyNotebookLink, clearNotebookLink, currentLinkHref } from "./notebookLinkEdit.ts";
import { buildNotebookExtensions } from "./notebookExtensions.ts";
import { documentMarkdown } from "./markdownCodec.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

/** 建一个真实编辑器（不需要 React：这里只验证文档，不渲染自定义 NodeView）。 */
function makeEditor(content: string): Editor {
    return new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({ markdownShortcuts: true, slashCommands: false }),
        // `content` 是 **Markdown**：`Markdown.configure({ html: false })` 之后
        // 传入的字符串按 Markdown 解析，写 `<p>x</p>` 会被当成一段字面文本。
        content,
    });
}

/** 把光标放到正文末尾（不选中任何文本）。 */
function collapseToEnd(editor: Editor): void {
    editor.commands.setTextSelection(editor.state.doc.content.size - 1);
}

/** 把光标放到文档里第一个链接的文字中间。 */
function collapseInsideLink(editor: Editor): void {
    let start: number | null = null;
    editor.state.doc.descendants((node, pos) => {
        if (start !== null) return false;
        if (!node.isText || !node.marks.some((mark) => mark.type.name === "link")) return true;
        start = pos;
        return false;
    });
    if (start === null) throw new Error("document has no link to place the cursor in");
    editor.commands.setTextSelection(start + 1);
}

test("components/layout/notebook/notebookLinkEdit.test.ts scripted checks", () => {
    // ── 未选中文本：地址本身成为链接文字（本次修复的核心） ──────────
    {
        const editor = makeEditor("看看这里");
        collapseToEnd(editor);
        applyNotebookLink(editor, "https://example.com/a");
        assertEqual(
            documentMarkdown(editor),
            "看看这里[https://example.com/a](https://example.com/a)",
            "empty selection inserts the address as linked text",
        );
        editor.destroy();
    }

    // ── 选中文字：链接加在选中的文字上，不插入地址 ──────────────────
    {
        const editor = makeEditor("点这里");
        // 选中「点这里」（段落内容从 1 开始）
        editor.commands.setTextSelection({ from: 1, to: 4 });
        applyNotebookLink(editor, "https://example.com/b");
        assertEqual(
            documentMarkdown(editor),
            "[点这里](https://example.com/b)",
            "selection gets the link, no extra text",
        );
        editor.destroy();
    }

    // ── 光标停在已有链接里：改写地址，文字不动 ─────────────────────
    {
        const editor = makeEditor("看看这里");
        collapseToEnd(editor);
        applyNotebookLink(editor, "https://old.example.com");
        collapseInsideLink(editor);
        assertEqual(currentLinkHref(editor), "https://old.example.com", "existing href read back");
        applyNotebookLink(editor, "https://new.example.com");
        assertEqual(
            documentMarkdown(editor),
            // 文字保持原样（它是当初插入的地址），换掉的只有目标地址。
            "看看这里[https://old.example.com](https://new.example.com)",
            "rewriting keeps the text and only swaps the address",
        );
        editor.destroy();
    }

    // ── 缺协议的地址：补成绝对地址，否则会被当成相对地址 ─────────────
    {
        const editor = makeEditor("站");
        collapseToEnd(editor);
        applyNotebookLink(editor, "www.bilibili.com");
        assertEqual(
            documentMarkdown(editor),
            "站[https://www.bilibili.com](https://www.bilibili.com)",
            "a bare host is stored absolute, not relative to the app origin",
        );
        editor.destroy();
    }

    // ── 手写的相对链接（不经插入入口）：渲染时补协议，源码不动 ────────
    {
        const editor = makeEditor("[x](www.bilibili.com)");
        const anchor = new DOMParser()
            .parseFromString(editor.getHTML(), "text/html")
            .querySelector("a");
        assertEqual(
            anchor?.getAttribute("href"),
            "https://www.bilibili.com",
            "hand-written relative href renders absolute",
        );
        assertEqual(
            documentMarkdown(editor),
            "[x](www.bilibili.com)",
            "the author's markdown source is not rewritten",
        );
        editor.destroy();
    }

    // ── 内部链接（hifi://）不受归一化影响 ───────────────────────────
    {
        const editor = makeEditor("跳");
        collapseToEnd(editor);
        applyNotebookLink(editor, "hifi://seek/12.5");
        assertEqual(
            documentMarkdown(editor),
            "跳[hifi://seek/12.5](hifi://seek/12.5)",
            "internal links keep their own protocol",
        );
        editor.destroy();
    }

    // ── 空地址：只摘链接标记，文字留下（unlink 而不是 delete） ───────
    {
        const editor = makeEditor("看看这里");
        collapseToEnd(editor);
        applyNotebookLink(editor, "https://example.com/c");
        collapseInsideLink(editor);
        applyNotebookLink(editor, "   ");
        assertEqual(
            documentMarkdown(editor),
            "看看这里https://example.com/c",
            "blank address unlinks but keeps the text",
        );
        assertEqual(currentLinkHref(editor), "", "no link left behind");
        editor.destroy();
    }

    // ── 地址白名单：`javascript:` 连文字都不该落进文档 ──────────────
    {
        const editor = makeEditor("x");
        collapseToEnd(editor);
        applyNotebookLink(editor, "javascript:alert(1)");
        assertEqual(documentMarkdown(editor), "x", "rejected protocol inserts nothing");
        editor.destroy();
    }

    // ── clearNotebookLink：光标在链接中间也整段摘掉标记 ──────────────
    {
        const editor = makeEditor("看看这里");
        collapseToEnd(editor);
        applyNotebookLink(editor, "https://example.com/d");
        collapseInsideLink(editor);
        clearNotebookLink(editor);
        assertEqual(
            documentMarkdown(editor),
            "看看这里https://example.com/d",
            "clear drops the mark from the whole link, text stays",
        );
        editor.destroy();
    }
});
