/*
 * 链接地址归一化的判定表。
 *
 * 【要钉死什么】只有"缺协议的裸主机名"才补协议；明确的相对写法、已有协议、
 * 邮箱都不能被改坏。最容易写错的两条：
 *   - `example.com:8080` 不能被当成协议（`example.com:` 形似协议名）；
 *   - `assets/img.png` 不能被当成主机名（第一段不含点）。
 */

import { test } from "vitest";

import { normalizeLinkHref } from "./notebookLinkUrl.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

test("components/layout/notebook/notebookLinkUrl.test.ts scripted checks", () => {
    // ── 本次修复的目标：裸主机名补 https ────────────────────────────
    assertEqual(normalizeLinkHref("www.bilibili.com"), "https://www.bilibili.com", "www host");
    assertEqual(normalizeLinkHref("bilibili.com"), "https://bilibili.com", "bare domain");
    assertEqual(
        normalizeLinkHref("www.bilibili.com/video/BV1xx?t=1#a"),
        "https://www.bilibili.com/video/BV1xx?t=1#a",
        "host with path, query and fragment",
    );
    assertEqual(
        normalizeLinkHref("  www.bilibili.com  "),
        "https://www.bilibili.com",
        "surrounding whitespace is trimmed",
    );
    assertEqual(
        normalizeLinkHref("example.com:8080/x"),
        "https://example.com:8080/x",
        "a port is not mistaken for a scheme",
    );

    // ── 已有协议：一个字符都不动 ────────────────────────────────────
    for (const href of [
        "http://example.com",
        "https://example.com",
        "hifi://seek/12.5",
        "hifi://clip/abc",
        "mailto:someone@example.com",
        "tel:+8613800000000",
        "ftp://files.example.com",
    ]) {
        assertEqual(normalizeLinkHref(href), href, `scheme kept: ${href}`);
    }

    // ── 明确的相对写法 / 锚点：用户的选择，不动 ─────────────────────
    for (const href of ["/assets/a.png", "#section", "?q=1", "./notes.md", "../up.md"]) {
        assertEqual(normalizeLinkHref(href), href, `relative kept: ${href}`);
    }

    // ── 邮箱：补 mailto 而不是 https ────────────────────────────────
    assertEqual(
        normalizeLinkHref("someone@example.com"),
        "mailto:someone@example.com",
        "bare address becomes mailto",
    );

    // ── 认不出来就不猜 ──────────────────────────────────────────────
    for (const href of ["foo", "assets/img.png", "localhost:3000"]) {
        assertEqual(normalizeLinkHref(href), href, `left alone: ${href}`);
    }

    // ── 空值 ────────────────────────────────────────────────────────
    assertEqual(normalizeLinkHref(""), "", "empty");
    assertEqual(normalizeLinkHref("   "), "", "blank");

    // ── 非法协议保持原样，交给 Link 扩展的白名单去拒 ─────────────────
    assertEqual(
        normalizeLinkHref("javascript:alert(1)"),
        "javascript:alert(1)",
        "rejected protocols are not rewritten, only refused later",
    );
});
