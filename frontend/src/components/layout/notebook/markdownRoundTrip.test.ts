// @vitest-environment jsdom
/*
 * Markdown ⇄ 编辑器文档的往返测试。
 *
 * 【为什么这个测试最重要】整个记事本设计的地基是"Markdown 是唯一真源"：
 * 正文存进后端的是序列化结果，重开工程时再解析回来。往返一旦有损，用户的
 * 笔记就会被自己悄悄改写。这里锁定的是**代表性文档**的稳定性，而不是
 * 每一种 Markdown 语法的完备性。
 *
 * 需要 jsdom：TipTap/ProseMirror 要真实 DOM 才能建编辑器。
 */

import { Editor } from "@tiptap/core";
import { test } from "vitest";

import { buildNotebookExtensions } from "./notebookExtensions";
import { documentMarkdown } from "./markdownCodec";

function createEditor(content: string): Editor {
    return new Editor({
        element: document.createElement("div"),
        extensions: buildNotebookExtensions({
            placeholder: "",
            markdownShortcuts: true,
            slashCommands: false,
        }),
        content,
    });
}

/** 一次往返：md → 编辑器 → md。 */
function roundTrip(markdown: string): string {
    const editor = createEditor(markdown);
    try {
        return documentMarkdown(editor);
    } finally {
        editor.destroy();
    }
}

function assertStable(markdown: string, label: string): string {
    const once = roundTrip(markdown);
    const twice = roundTrip(once);
    if (once !== twice) {
        throw new Error(
            `${label}: round trip is not idempotent\n--- first ---\n${once}\n--- second ---\n${twice}`,
        );
    }
    return once;
}

function assertIncludes(haystack: string, needle: string, label: string): void {
    if (!haystack.includes(needle)) {
        throw new Error(
            `${label}: expected to include ${JSON.stringify(needle)}\n--- got ---\n${haystack}`,
        );
    }
}

test("components/layout/notebook/markdownRoundTrip.test.ts scripted checks", async () => {
    // ── 基础块级语法 ─────────────────────────────────────────────
    const basic = assertStable(
        [
            "# 一级标题",
            "",
            "普通段落，含 **粗体**、*斜体*、~~删除线~~ 与 `行内代码`。",
            "",
            "## 二级标题",
            "",
            "- 甲",
            "- 乙",
            "",
            "1. 第一",
            "2. 第二",
            "",
            "> 引用",
            "",
            "```",
            "code block",
            "```",
            "",
            "---",
        ].join("\n"),
        "basic blocks",
    );
    assertIncludes(basic, "# 一级标题", "h1 kept");
    assertIncludes(basic, "**粗体**", "bold kept");
    assertIncludes(basic, "~~删除线~~", "strike kept");
    assertIncludes(basic, "> 引用", "quote kept");
    assertIncludes(basic, "---", "hr kept");

    // ── GFM 表格（自定义序列化器：必须永不退化成 HTML 占位符）──────
    const table = assertStable(
        ["| 参数 | 值 |", "| --- | --- |", "| pitch | +2 |", "| formant | -1 |"].join("\n"),
        "gfm table",
    );
    assertIncludes(table, "| 参数 | 值 |", "table header kept");
    assertIncludes(table, "| pitch | +2 |", "table body kept");
    if (table.includes("markdownHTMLNode")) {
        throw new Error(`table degraded to an HTML placeholder:\n${table}`);
    }

    // ── 任务列表 ─────────────────────────────────────────────────
    const tasks = assertStable("- [ ] 待办\n- [x] 已完成", "task list");
    assertIncludes(tasks, "[ ]", "unchecked task kept");
    assertIncludes(tasks, "[x]", "checked task kept");

    // ── 图片：宽度编码在 src 的 `#w=` 片段里 ─────────────────────
    const image = assertStable("![图注](hifi-asset://abc123.webp#w=640)", "image with width");
    assertIncludes(image, "hifi-asset://abc123.webp#w=640", "asset ref and width kept");
    assertIncludes(image, "![图注]", "alt kept");

    // 无宽度的图片不应被凭空加上 `#w=`。
    const plainImage = assertStable("![图](hifi-asset://def456.png)", "image without width");
    assertIncludes(plainImage, "hifi-asset://def456.png", "asset ref kept");
    if (plainImage.includes("#w=")) throw new Error(`unexpected width fragment:\n${plainImage}`);

    // 相对路径与 data URI 也要能原样往返（link / embed 两种存储模式）。
    const linked = assertStable(
        "![截图](%E7%B4%A0%E6%9D%90/%E6%88%AA%E5%9B%BE.png)",
        "linked image",
    );
    assertIncludes(linked, "%E7%B4%A0%E6%9D%90", "encoded relative path kept");

    // ── 剪贴板暂存块：自定义节点的往返（最脆的一环）───────────────
    const clipFence = [
        "```hifi-clip",
        "id: 7c1e9a4b2d",
        "kind: clips",
        "title: 副歌 A",
        "source: 我的歌",
        "encoding: fragment",
        "clips: 3",
        "tracks: 1",
        "duration: 4.820",
        "captured: 2026-09-23T10:04:11Z",
        "```",
    ].join("\n");
    const clip = assertStable(clipFence, "staged clip block");
    assertIncludes(clip, "```hifi-clip", "fence language kept");
    assertIncludes(clip, "id: 7c1e9a4b2d", "clip id kept");
    assertIncludes(clip, "title: 副歌 A", "clip title kept");
    assertIncludes(clip, "duration: 4.820", "clip duration kept");

    // 正文里的暂存块必须真的是**块级节点**，而不是被当成普通代码块：
    // 重新序列化后仍带 ```hifi-clip 就是证据（普通代码块只会写 ```）。
    const editor = createEditor(clipFence);
    try {
        let clipNodeCount = 0;
        editor.state.doc.descendants((node) => {
            if (node.type.name === "hifiClipBlock") clipNodeCount += 1;
            return true;
        });
        if (clipNodeCount !== 1) {
            throw new Error(`expected exactly one hifiClipBlock node, got ${clipNodeCount}`);
        }
    } finally {
        editor.destroy();
    }

    // ── 内部链接（跳播放头 / 引用 Clip）──────────────────────────
    const links = assertStable(
        "[1:23.456](hifi://seek/83.456) 与 [副歌 A](hifi://clip/abc-1)",
        "internal links",
    );
    assertIncludes(links, "hifi://seek/83.456", "seek link kept");
    assertIncludes(links, "hifi://clip/abc-1", "clip link kept");

    // ── 空文档与纯文本 ───────────────────────────────────────────
    if (roundTrip("").trim() !== "") throw new Error("empty document should stay empty");
    assertIncludes(
        assertStable("只有一行普通文字。", "plain text"),
        "只有一行普通文字。",
        "plain text kept",
    );
});
