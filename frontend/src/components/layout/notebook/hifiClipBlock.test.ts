import { test } from "vitest";

import {
    defaultClipTitle,
    formatClipDuration,
    parseHifiClipFenceBody,
    serializeHifiClipFence,
    serializeHifiClipFenceBody,
    type HifiClipBlockAttrs,
} from "./hifiClipBlock.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

const BASE: HifiClipBlockAttrs = {
    id: "7c1e9a4b2d",
    kind: "clips",
    title: "副歌 A",
    source: "我的歌",
    clipCount: 3,
    trackCount: 1,
    durationSec: 4.82,
    captured: "2026-09-23T10:04:11Z",
    encoding: "fragment",
};

test("components/layout/notebook/hifiClipBlock.test.ts scripted checks", async () => {
    // ── 往返：解析(序列化(x)) === x ───────────────────────────────
    const body = serializeHifiClipFenceBody(BASE);
    assertEqual(parseHifiClipFenceBody(body), BASE, "round trip");

    const fence = serializeHifiClipFence(BASE);
    assertEqual(fence.split("\n")[0], "```hifi-clip", "fence opening");
    assertEqual(fence.split("\n").at(-1), "```", "fence closing");

    // 多行值里的换行会破坏"一行一字段"结构，必须被折叠。
    const messy = parseHifiClipFenceBody(
        serializeHifiClipFenceBody({ ...BASE, title: "第一行\n第二行" }),
    );
    assertEqual(messy?.title, "第一行 第二行", "newlines folded");

    // ── 非法正文退化为"不是暂存块" ───────────────────────────────
    assertEqual(parseHifiClipFenceBody("kind: clips"), null, "missing id");
    assertEqual(parseHifiClipFenceBody("id: ../escape"), null, "traversal-ish id");
    assertEqual(parseHifiClipFenceBody(""), null, "empty body");

    // ── 缺省与容错 ───────────────────────────────────────────────
    const minimal = parseHifiClipFenceBody("id: abc\n");
    assertEqual(minimal?.kind, "clips", "default kind");
    assertEqual(minimal?.encoding, "fragment", "default encoding");
    assertEqual(minimal?.durationSec, 0, "missing duration is 0");

    // kind=param 时编码自动识别为 param，并读取参数名/帧数。
    const param = parseHifiClipFenceBody(
        serializeHifiClipFenceBody({
            ...BASE,
            kind: "param",
            encoding: "param",
            param: "pitch",
            frameCount: 1280,
        }),
    );
    assertEqual(param?.encoding, "param", "param encoding");
    assertEqual(param?.param, "pitch", "param name");
    assertEqual(param?.frameCount, 1280, "param frame count");
    // 参数线没有时长，序列化时不该写出 duration 行。
    assertEqual(serializeHifiClipFenceBody(param!).includes("duration:"), false, "no duration line");

    // 未知 kind 退化为 clips（未来版本写入的新类型不该让块消失）。
    assertEqual(parseHifiClipFenceBody("id: abc\nkind: future\n")?.kind, "clips", "unknown kind");

    // ── 展示辅助 ─────────────────────────────────────────────────
    assertEqual(formatClipDuration(4.82), "0:04.820", "duration format");
    assertEqual(formatClipDuration(83.456), "1:23.456", "duration format over a minute");
    assertEqual(formatClipDuration(0), "0:00.000", "zero duration");
    assertEqual(defaultClipTitle(BASE), "3 clips · 0:04.820", "default title");
    assertEqual(
        defaultClipTitle({ ...BASE, trackCount: 2 }),
        "3 clips · 2 tracks · 0:04.820",
        "multi-track title",
    );
});
