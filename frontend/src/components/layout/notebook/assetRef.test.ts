import { test } from "vitest";

import {
    assetIdFromSrc,
    assetWidthFromSrc,
    clipBlockIdsInMarkdown,
    formatAssetRef,
    isAssetRef,
    referencedAssetIds,
    scanAssetRefs,
} from "./assetRef.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

test("components/layout/notebook/assetRef.test.ts scripted checks", async () => {
    // ── 基本往返 ──────────────────────────────────────────────
    assertEqual(
        formatAssetRef("abc123", "webp", 640),
        "hifi-asset://abc123.webp#w=640",
        "full form",
    );
    assertEqual(formatAssetRef("abc123"), "hifi-asset://abc123", "id only");
    assertEqual(
        formatAssetRef("abc123", "png", 0),
        "hifi-asset://abc123.png",
        "zero width omitted",
    );

    // ── 扫描 ──────────────────────────────────────────────────
    const refs = scanAssetRefs("前 ![图](hifi-asset://a1.webp#w=320) 后 hifi-asset://b2.png 尾");
    assertEqual(refs.length, 2, "two refs");
    assertEqual(refs[0].id, "a1", "first id");
    assertEqual(refs[0].ext, "webp", "first ext");
    assertEqual(refs[0].width, 320, "first width");
    assertEqual(refs[1].id, "b2", "second id");
    assertEqual(refs[1].width, undefined, "second width absent");
    const src = "前 ![图](hifi-asset://a1.webp#w=320) 后 hifi-asset://b2.png 尾";
    assertEqual(
        src.slice(refs[0].start, refs[0].end),
        "hifi-asset://a1.webp#w=320",
        "range covers ref",
    );

    // ── 容错 ──────────────────────────────────────────────────
    assertEqual(scanAssetRefs("hifi-asset:// 没有 id").length, 0, "bare scheme ignored");
    assertEqual(scanAssetRefs("hifi-asset://../escape").length, 0, "traversal-ish id ignored");
    assertEqual(scanAssetRefs("hifi-asset://x.png#w=").length, 1, "dangling width still a ref");
    assertEqual(
        scanAssetRefs("hifi-asset://x.png#w=")[0].width,
        undefined,
        "dangling width absent",
    );

    // ── 单引用取值 ────────────────────────────────────────────
    assertEqual(assetIdFromSrc("hifi-asset://zz.webp#w=100"), "zz", "assetIdFromSrc");
    assertEqual(assetIdFromSrc("https://example.com/a.png"), null, "http not asset");
    assertEqual(assetIdFromSrc("./local.png"), null, "relative not asset");
    assertEqual(assetWidthFromSrc("hifi-asset://zz.webp#w=100"), 100, "assetWidthFromSrc");
    assertEqual(assetWidthFromSrc("./local.png"), null, "relative width absent");
    assertEqual(isAssetRef("hifi-asset://a"), true, "isAssetRef true");
    assertEqual(isAssetRef("data:image/png;base64,AAA"), false, "data uri not asset");

    // ── 正文引用集合（图片 + 剪贴板块）────────────────────────
    const markdown = [
        "# 标题",
        "",
        "![图](hifi-asset://img1.webp)",
        "",
        "```hifi-clip",
        "id: clip1",
        "kind: clips",
        "```",
        "",
        "重复引用同一张图：hifi-asset://img1.webp",
    ].join("\n");
    assertEqual(
        [...referencedAssetIds(markdown)].sort(),
        ["clip1", "img1"],
        "referenced ids deduped",
    );
    assertEqual(clipBlockIdsInMarkdown(markdown), ["clip1"], "clip block ids");
    // `id:` 出现在普通正文里（不在围栏内）也按引用处理 —— 与后端实现一致，
    // 宁可多留一条附件，也不要误删正在被引用的字节。
    assertEqual(clipBlockIdsInMarkdown("id: plain").length, 1, "bare id line counted");
});
