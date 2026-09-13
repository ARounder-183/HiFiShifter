import { test } from "vitest";

import {
    clipboardPreviewSpans,
    clipboardSpanFrames,
    clipboardTotalFrames,
    mapClipboardToTargetRanges,
    normalizeClipboardData,
    parseParamClipboardPayload,
    toParamClipboardPayload,
    type ParamClipboardData,
} from "./paramClipboardMapping.js";

/**
 * 这里锁住的是需求里那两个方向相反的映射场景 —— 断层的任何一侧都不得被
 * 填充、不得被压缩：
 *   A. 复制 0~1s、2~3s → 目标 0~3s：只写两段，1~2s 保持原值；
 *   B. 复制 0~3s → 目标 0~1s、2~3s：第一段取剪贴板开头，第二段取剪贴板
 *      对应偏移（即末尾），断层处既不预览也不粘贴。
 * 同时锁住单段 ↔ 单段退化为旧行为（对齐选区起点 + 超长截断）。
 */
test("components/layout/pianoRoll/paramClipboardMapping.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }
    function assertJson(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const b = JSON.stringify(expected);
        if (a !== b) {
            throw new Error(`${label}: expected ${b}, received ${a}`);
        }
    }

    /** 生成 [start, start+len) 的递增序列，便于肉眼核对切片位置。 */
    const ramp = (start: number, len: number): number[] =>
        Array.from({ length: len }, (_, i) => start + i);

    // ── 载荷解析 / 序列化 ───────────────────────────────────────────────
    {
        const legacy = parseParamClipboardPayload({
            version: 1,
            kind: "param",
            param: "pitch",
            framePeriodMs: 5,
            values: [1, 2, 3],
        });
        assertJson(
            legacy,
            { param: "pitch", framePeriodMs: 5, segments: [{ startFrame: 0, values: [1, 2, 3] }] },
            "v1 loads as single segment",
        );
    }
    {
        const data: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 0, values: [1, 2] },
                { startFrame: 20, values: [3] },
            ],
        };
        const payload = toParamClipboardPayload(data);
        assertEqual(payload.version, 2, "writes v2");
        assertJson(parseParamClipboardPayload(payload), data, "v2 round trip");
    }
    // 非法载荷一律视为「没有参数线数据」
    assertEqual(parseParamClipboardPayload(null), null, "null payload");
    assertEqual(parseParamClipboardPayload({ kind: "clip" }), null, "wrong kind");
    assertEqual(parseParamClipboardPayload({ kind: "param", version: 9 }), null, "unknown version");
    assertEqual(
        parseParamClipboardPayload({ kind: "param", version: 2, param: "pitch", segments: [] }),
        null,
        "empty segments",
    );
    // 值去非有限、段按偏移排序、空段丢弃
    {
        const parsed = parseParamClipboardPayload({
            kind: "param",
            version: 2,
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 30, values: [7] },
                { startFrame: 10, values: [] },
                { startFrame: 0, values: [1, NaN, 3] },
            ],
        });
        assertJson(
            parsed,
            {
                param: "pitch",
                framePeriodMs: 5,
                segments: [
                    { startFrame: 0, values: [1, 0, 3] },
                    { startFrame: 30, values: [7] },
                ],
            },
            "sanitize/sort/drop-empty",
        );
    }

    // ── 汇总 ────────────────────────────────────────────────────────────
    assertEqual(
        clipboardTotalFrames({
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 0, values: [0, 0, 0] },
                { startFrame: 200, values: [0] },
            ],
        }),
        4,
        "total frames excludes gap",
    );
    assertEqual(
        clipboardSpanFrames({
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 0, values: [0, 0, 0] },
                { startFrame: 200, values: [0] },
            ],
        }),
        201,
        "span reaches last segment end",
    );
    assertEqual(clipboardTotalFrames(null), 0, "total null");
    assertEqual(
        normalizeClipboardData({ param: "pitch", framePeriodMs: 0, segments: [] }),
        null,
        "normalize drops empty payload",
    );

    // ── 场景 A：多段剪贴板 → 单段目标（断层不得被压缩） ──────────────────
    {
        // 0~1s / 2~3s 的复制结果：偏移 0..99 与 200..299 帧
        const clipboard: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 0, values: ramp(1000, 100) },
                { startFrame: 200, values: ramp(2000, 100) },
            ],
        };
        const writes = mapClipboardToTargetRanges({
            targetRanges: [{ startFrame: 0, frameCount: 300 }],
            clipboard,
        });
        assertEqual(writes.length, 2, "A: two writes, gap untouched");
        assertJson(
            writes.map((w) => [w.startFrame, w.values.length]),
            [
                [0, 100],
                [200, 100],
            ],
            "A: write spans",
        );
        assertEqual(writes[0].values[0], 1000, "A: first segment values");
        assertEqual(writes[1].values[0], 2000, "A: second segment values");
    }
    // 目标整体偏移：写入随选区起点整体平移，段内相对偏移不变
    {
        const clipboard: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [
                { startFrame: 0, values: ramp(1, 100) },
                { startFrame: 200, values: ramp(2, 100) },
            ],
        };
        const writes = mapClipboardToTargetRanges({
            targetRanges: [{ startFrame: 5000, frameCount: 300 }],
            clipboard,
        });
        assertJson(
            writes.map((w) => [w.startFrame, w.values.length]),
            [
                [5000, 100],
                [5200, 100],
            ],
            "A': offsets shift with selection origin",
        );
    }

    // ── 场景 B：单段剪贴板 → 多段目标（按偏移求交，断层不写） ────────────
    {
        // 0~3s 的复制结果：偏移 0..299 帧
        const clipboard: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [{ startFrame: 0, values: ramp(0, 300) }],
        };
        const writes = mapClipboardToTargetRanges({
            targetRanges: [
                { startFrame: 0, frameCount: 100 },
                { startFrame: 200, frameCount: 100 },
            ],
            clipboard,
        });
        assertEqual(writes.length, 2, "B: two writes");
        // 第一段 = 剪贴板开头 1s
        assertJson(
            [writes[0].startFrame, writes[0].values[0], writes[0].values[99]],
            [0, 0, 99],
            "B: first range takes clipboard head",
        );
        // 第二段 = 剪贴板最后 1s（偏移 200 起）
        assertJson(
            [writes[1].startFrame, writes[1].values[0], writes[1].values[99]],
            [200, 200, 299],
            "B: second range takes clipboard tail",
        );
    }
    // 目标断层落在剪贴板范围内、但目标段超出剪贴板 → 只写交集
    {
        const clipboard: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [{ startFrame: 0, values: ramp(0, 300) }],
        };
        const writes = mapClipboardToTargetRanges({
            targetRanges: [
                { startFrame: 0, frameCount: 100 },
                { startFrame: 400, frameCount: 100 },
            ],
            clipboard,
        });
        assertEqual(writes.length, 1, "B': out-of-clipboard range writes nothing");
        assertEqual(writes[0].startFrame, 0, "B': only in-range write");
    }

    // ── 单段 ↔ 单段：退化为旧行为 ───────────────────────────────────────
    {
        const clipboard: ParamClipboardData = {
            param: "pitch",
            framePeriodMs: 5,
            segments: [{ startFrame: 0, values: ramp(0, 300) }],
        };
        // 剪贴板长于选区 → 截断到选区长度
        const truncated = mapClipboardToTargetRanges({
            targetRanges: [{ startFrame: 1000, frameCount: 100 }],
            clipboard,
        });
        assertJson(
            truncated.map((w) => [w.startFrame, w.values.length, w.values[0]]),
            [[1000, 100, 0]],
            "single: aligned to selection start and truncated",
        );
        // 剪贴板短于选区 → 只写到剪贴板结束，其余帧保持原值
        const shorter = mapClipboardToTargetRanges({
            targetRanges: [{ startFrame: 1000, frameCount: 500 }],
            clipboard,
        });
        assertJson(
            shorter.map((w) => [w.startFrame, w.values.length]),
            [[1000, 300]],
            "single: shorter clipboard leaves tail untouched",
        );
    }

    // ── 无写入的情形（预览与粘贴都应当跳过） ───────────────────────────
    assertEqual(
        mapClipboardToTargetRanges({ targetRanges: [], clipboard: null }).length,
        0,
        "no target no write",
    );
    assertEqual(
        mapClipboardToTargetRanges({
            targetRanges: [{ startFrame: 0, frameCount: 0 }],
            clipboard: {
                param: "pitch",
                framePeriodMs: 5,
                segments: [{ startFrame: 0, values: [1] }],
            },
        }).length,
        0,
        "zero-length target writes nothing",
    );

    // ── 预览片段：秒换算用目标帧周期（与落盘一致） ──────────────────────
    {
        const spans = clipboardPreviewSpans({
            targetRanges: [
                { startFrame: 0, frameCount: 100 },
                { startFrame: 200, frameCount: 100 },
            ],
            clipboard: {
                param: "pitch",
                framePeriodMs: 5,
                segments: [{ startFrame: 0, values: [1, 2, 3, 4] }],
            },
            targetFramePeriodMs: 10,
        });
        // 偏移 0..3 帧 @ fp=10ms → 起点 0s；偏移 200 起已超出剪贴板 → 无
        assertEqual(spans.length, 1, "preview spans count");
        assertJson(
            [spans[0].startSec, spans[0].framePeriodMs, spans[0].values.length],
            [0, 10, 4],
            "preview uses target frame period",
        );
    }
});
