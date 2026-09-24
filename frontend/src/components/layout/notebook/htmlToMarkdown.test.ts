// @vitest-environment jsdom
/*
 * 粘贴转换需要真实的 DOM：DOMPurify 在无 window 的环境里是空实现（直接返回
 * 空串），Turndown 也需要 DOMParser。因此这个文件单独跑在 jsdom 下 ——
 * 仓库其余测试保持 node 环境（更快）。
 */

import { test } from "vitest";

import { htmlToMarkdown, sanitizePastedHtml } from "./htmlToMarkdown.ts";

function assertIncludes(haystack: string, needle: string, label: string): void {
    if (!haystack.includes(needle)) {
        throw new Error(
            `${label}: expected to include ${JSON.stringify(needle)}, received ${JSON.stringify(haystack)}`,
        );
    }
}

function assertExcludes(haystack: string, needle: string, label: string): void {
    if (haystack.includes(needle)) {
        throw new Error(
            `${label}: expected NOT to include ${JSON.stringify(needle)}, received ${JSON.stringify(haystack)}`,
        );
    }
}

test("components/layout/notebook/htmlToMarkdown.test.ts scripted checks", async () => {
    // 标题 / 粗斜体 / 行内码 / 链接 —— 最常见的网页粘贴形态。
    const simple = htmlToMarkdown(
        "<h2>标题</h2><p>普通 <strong>粗</strong> 与 <em>斜</em> 与 <code>x=1</code></p>" +
            '<p><a href="https://example.com/a">链接</a></p>',
    );
    assertIncludes(simple, "## 标题", "heading");
    assertIncludes(simple, "**粗**", "bold");
    assertIncludes(simple, "*斜*", "italic");
    assertIncludes(simple, "`x=1`", "inline code");
    assertIncludes(simple, "[链接](https://example.com/a)", "link");

    // 列表与引用。
    const lists = htmlToMarkdown(
        "<ul><li>甲</li><li>乙</li></ul><blockquote><p>引用</p></blockquote>",
    );
    assertIncludes(lists, "- 甲", "bullet item");
    assertIncludes(lists, "> 引用", "blockquote");

    // GFM 表格（从 Word/网页粘贴参数表是高频场景）。
    const table = htmlToMarkdown(
        "<table><thead><tr><th>参数</th><th>值</th></tr></thead>" +
            "<tbody><tr><td>pitch</td><td>+2</td></tr></tbody></table>",
    );
    assertIncludes(table, "| 参数 | 值 |", "table header row");
    assertIncludes(table, "| --- | --- |", "table delimiter row");
    assertIncludes(table, "| pitch | +2 |", "table body row");

    // 危险内容必须被消毒掉：脚本、事件属性、javascript: 链接。
    const dirty =
        '<p onclick="steal()">安全</p><script>alert(1)</script>' +
        '<p><a href="javascript:alert(2)">坏链接</a></p>';
    const sanitized = sanitizePastedHtml(dirty);
    assertExcludes(sanitized, "<script", "script removed");
    assertExcludes(sanitized, "onclick", "event handler removed");
    const cleaned = htmlToMarkdown(dirty);
    assertIncludes(cleaned, "安全", "text content kept");
    assertExcludes(cleaned, "javascript:", "javascript url removed");

    // 图片保留 src/alt；尺寸类属性不进 Markdown（宽度由本 app 自己的 #w= 承载）。
    const withImage = htmlToMarkdown('<p><img src="https://x/y.png" alt="图" width="640"></p>');
    assertIncludes(withImage, "![图](https://x/y.png)", "image");

    // 空输入不炸。
    if (htmlToMarkdown("") !== "") throw new Error("empty html should yield empty markdown");
    if (htmlToMarkdown("   ") !== "") throw new Error("blank html should yield empty markdown");
});
