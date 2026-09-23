import { test } from "vitest";

import {
    baseName,
    decodePathFromMarkdown,
    dirName,
    encodePathForMarkdown,
    isAbsolutePath,
    relativePath,
    resolvePath,
    stemName,
    toPosix,
} from "./notebookPaths.ts";
import { buildClipLink, buildSeekLink, formatTimecode, parseInternalLink } from "./timecode.ts";
import { normalizeNotebookSettings, NOTEBOOK_PANEL_MAX_WIDTH } from "./notebookSettings.ts";
import { computeTargetSize, chooseEncodeFormat } from "./notebookImagePipeline.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

test("components/layout/notebook/notebookPaths.test.ts scripted checks", async () => {
    assertEqual(toPosix("a\\b\\c.png"), "a/b/c.png", "toPosix");
    assertEqual(dirName("a/b/c.png"), "a/b", "dirName");
    assertEqual(dirName("c.png"), "", "dirName without dir");
    assertEqual(baseName("a/b/c.png"), "c.png", "baseName");
    assertEqual(stemName("a/b/c.tar.gz"), "c.tar", "stemName");

    assertEqual(isAbsolutePath("/home/u/a.png"), true, "posix absolute");
    assertEqual(isAbsolutePath("C:\\work\\a.png"), true, "windows absolute");
    assertEqual(isAbsolutePath("素材/a.png"), false, "relative");

    // 相对路径：同盘可算，跨盘放弃（Windows 上 C: → D: 无法表达）。
    assertEqual(relativePath("C:/work/song", "C:/work/song/assets/a.png"), "assets/a.png", "same dir child");
    assertEqual(relativePath("C:/work/song", "C:/work/ref/a.png"), "../ref/a.png", "sibling");
    assertEqual(relativePath("C:/work/song", "D:/other/a.png"), null, "cross drive");
    assertEqual(relativePath("/a/b", "/a/b"), ".", "identical");

    assertEqual(resolvePath("C:/work/song", "assets/a.png"), "C:/work/song/assets/a.png", "resolve child");
    assertEqual(resolvePath("C:/work/song", "../ref/a.png"), "C:/work/ref/a.png", "resolve parent");
    assertEqual(resolvePath("/a/b", "/abs/a.png"), "/abs/a.png", "resolve absolute passthrough");

    // 空格与括号必须编码，否则 Markdown 链接语法会被截断。
    assertEqual(
        encodePathForMarkdown("素材/截图 (1).png"),
        "素材/截图%20(1).png".replace("(", "%28").replace(")", "%29"),
        "encode spaces and parens",
    );
    assertEqual(decodePathFromMarkdown("素材/%E6%88%AA%E5%9B%BE.png"), "素材/截图.png", "decode");
    assertEqual(
        decodePathFromMarkdown(encodePathForMarkdown("素材/截图 (1).png")),
        "素材/截图 (1).png",
        "encode/decode round trip",
    );
});

test("components/layout/notebook/timecode.test.ts scripted checks", async () => {
    assertEqual(formatTimecode(83.456), "1:23.456", "under an hour");
    assertEqual(formatTimecode(3723.5), "1:02:03.500", "over an hour");
    assertEqual(formatTimecode(-1), "0:00.000", "negative clamps");

    const link = buildSeekLink(83.456);
    assertEqual(link, "[1:23.456](hifi://seek/83.456)", "seek link");
    assertEqual(parseInternalLink("hifi://seek/83.456"), { type: "seek", seconds: 83.456 }, "parse seek");
    assertEqual(buildSeekLink(12, "这里要重唱"), "[这里要重唱](hifi://seek/12.000)", "custom label");

    assertEqual(parseInternalLink("hifi://clip/abc-1"), { type: "clip", clipId: "abc-1" }, "parse clip");
    // 方括号会截断链接文本，必须折叠掉。
    assertEqual(buildClipLink("c1", "副歌 [A]"), "[副歌 (A)](hifi://clip/c1)", "clip label sanitized");

    assertEqual(parseInternalLink("https://example.com"), null, "external link");
    assertEqual(parseInternalLink("hifi://seek/abc"), null, "bad seek payload");
    assertEqual(parseInternalLink("hifi://clip/"), null, "empty clip id");
});

test("components/layout/notebook/notebookSettings.test.ts scripted checks", async () => {
    const defaults = normalizeNotebookSettings(null);
    assertEqual(defaults.defaultMode, "rich", "default mode");
    assertEqual(defaults.imageStorage, "sidecar", "default image storage");
    assertEqual(defaults.smartPaste, true, "smart paste on by default");

    // 非法取值一律退回默认，避免配置被手改后炸在渲染路径上。
    const bogus = normalizeNotebookSettings({
        defaultMode: "wysiwyg" as never,
        imageStorage: "ftp" as never,
        copyFormat: "rtf" as never,
        panelWidth: 99999,
        imageMaxDimensionPx: -5,
        historySplitIdleMs: Number.NaN,
    });
    assertEqual(bogus.defaultMode, "rich", "bogus mode");
    assertEqual(bogus.imageStorage, "sidecar", "bogus storage");
    assertEqual(bogus.copyFormat, "markdown+html", "bogus copy format");
    assertEqual(bogus.panelWidth, NOTEBOOK_PANEL_MAX_WIDTH, "panel width clamped");
    assertEqual(bogus.imageMaxDimensionPx, 0, "negative dimension clamped to 0");
    assertEqual(bogus.historySplitIdleMs, 0, "NaN falls back to default");

    // 0 是 imageMaxDimensionPx 的合法值（= 不缩放），不能被当成"未设置"。
    assertEqual(normalizeNotebookSettings({ imageMaxDimensionPx: 0 }).imageMaxDimensionPx, 0, "zero is legal");

    const partial = normalizeNotebookSettings({ spellCheck: true });
    assertEqual(partial.spellCheck, true, "explicit true kept");
    assertEqual(partial.showToolbar, true, "unspecified keeps default");
});

test("components/layout/notebook/notebookImagePipeline.test.ts scripted checks", async () => {
    // 缩放：长边压到上限，短边等比；未超限则不动。
    assertEqual(computeTargetSize(4000, 2000, 2048), { width: 2048, height: 1024, scaled: true }, "downscale landscape");
    assertEqual(computeTargetSize(2000, 4000, 2048), { width: 1024, height: 2048, scaled: true }, "downscale portrait");
    assertEqual(computeTargetSize(800, 600, 2048), { width: 800, height: 600, scaled: false }, "no upscale");
    assertEqual(computeTargetSize(4000, 2000, 0), { width: 4000, height: 2000, scaled: false }, "zero means original");

    // auto：无损格式且无需缩放时保持原样；需要缩放时转 WebP。
    assertEqual(chooseEncodeFormat("image/png", "auto", false), { ext: "png", mime: "image/png", quality: 1 }, "keep png");
    assertEqual(chooseEncodeFormat("image/png", "auto", true), { ext: "webp", mime: "image/webp", quality: 0.85 }, "scaled png to webp");
    assertEqual(chooseEncodeFormat("image/jpeg", "auto", false), { ext: "webp", mime: "image/webp", quality: 0.85 }, "jpeg to webp");
    assertEqual(chooseEncodeFormat("image/png", "jpeg", false).ext, "jpg", "forced jpeg");
    assertEqual(chooseEncodeFormat("image/jpeg", "png", false).ext, "png", "forced png");
});
