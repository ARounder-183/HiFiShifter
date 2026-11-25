import { describe, expect, it } from "vitest";

import {
    createLiveEditOverride,
    createStrokeBuffer,
    extendCurveViewToViewport,
} from "./strokeBuffer";
import type { LiveEditOverride } from "./strokeBuffer";
import type { ParamViewSegment } from "./types";

/**
 * 绘制工具的**取样域**：视口 ∩ (窗口 ∪ 笔画域)。
 *
 * 【为什么单独成一个文件】这条判据决定"拖到窗口外还能不能看见自己画的那一段"。
 * 它是纯函数，可以在 node 里把三种情形（窗口内 / 笔画内 / 都没有）逐帧断言清楚 ——
 * 而现场那一次的表现是"参数线整段消失，再拖一会儿才出现"，靠肉眼复现很贵。
 */

function view(args: {
    startFrame: number;
    length: number;
    stride?: number;
    base?: (index: number) => number;
}): ParamViewSegment {
    const stride = args.stride ?? 1;
    const base = args.base ?? (() => 1);
    const edit: number[] = [];
    const orig: number[] = [];
    for (let index = 0; index < args.length; index += 1) {
        edit.push(base(index));
        orig.push(base(index));
    }
    return {
        key: `v2|track|pitch|${args.startFrame}|${args.length}|${stride}`,
        framePeriodMs: 5,
        startFrame: args.startFrame,
        stride,
        referenceKind: "source_curve",
        orig,
        edit,
    };
}

function live(buffer = createStrokeBuffer("v2|track|pitch")): LiveEditOverride {
    const override = createLiveEditOverride("v2|track|pitch", 1);
    return { ...override, buffer };
}

describe("extendCurveViewToViewport", () => {
    it("keeps the window values and fills the outside from the stroke", () => {
        const buffer = createStrokeBuffer("v2|track|pitch");
        for (let frame = 1000; frame <= 1200; frame += 1) buffer.write(frame, [frame]);
        const extended = extendCurveViewToViewport({
            view: view({ startFrame: 1000, length: 100 }),
            live: live(buffer),
            viewport: { startFrame: 1100, endFrame: 1200 },
        });
        expect(extended).not.toBeNull();
        const segment = extended as ParamViewSegment;
        expect(segment.startFrame).toBe(1000);
        expect(segment.edit).toHaveLength(201);
        // 窗口内：窗口值（这里 window 的 base 恒为 1）。
        expect(segment.edit[0]).toBe(1);
        expect(segment.edit[99]).toBe(1);
        // 窗口外、笔画内：缓冲值。
        expect(segment.edit[100]).toBe(1100);
        expect(segment.edit[200]).toBe(1200);
        // orig 只在窗口内有值，窗口外一律"没有数据"。
        expect(segment.orig[0]).toBe(1);
        expect(Number.isNaN(segment.orig[100])).toBe(true);
    });

    it("writes NaN where neither the window nor the stroke has data", () => {
        const extended = extendCurveViewToViewport({
            view: view({ startFrame: 1000, length: 100 }),
            live: live(),
            viewport: { startFrame: 1100, endFrame: 1150 },
        });
        const segment = extended as ParamViewSegment;
        expect(segment).not.toBeNull();
        // 【为什么是 NaN 而不是 0】NaN 让折线在缺口处断开；0 会画出一条贴着底边的假曲线。
        expect(Number.isNaN(segment.edit[100])).toBe(true);
        expect(segment.edit[0]).toBe(1);
    });

    it("returns null when the window already covers the viewport (identity preserved)", () => {
        const window = view({ startFrame: 1000, length: 400 });
        const result = extendCurveViewToViewport({
            view: window,
            live: live(),
            viewport: { startFrame: 1100, endFrame: 1200 },
        });
        expect(result).toBeNull();
    });

    it("caps the extension at half a viewport, so a long stroke cannot blow up the domain", () => {
        const buffer = createStrokeBuffer("v2|track|pitch");
        for (let frame = 0; frame < 2_000_000; frame += 1000) buffer.write(frame, [1]);
        const extended = extendCurveViewToViewport({
            view: view({ startFrame: 1_000_000, length: 10 }),
            live: live(buffer),
            viewport: { startFrame: 1_000_000, endFrame: 1_000_100 },
        });
        const segment = extended as ParamViewSegment;
        // 视口 100 帧 ⇒ 半屏 50 帧；起点不得早于 999 950，终点不得晚于 1 000 150。
        expect(segment.startFrame).toBeGreaterThanOrEqual(999_950);
        expect(segment.startFrame + (segment.edit.length - 1) * segment.stride).toBeLessThanOrEqual(
            1_000_150,
        );
    });

    it("gives up (null) instead of rendering an unbounded domain", () => {
        const buffer = createStrokeBuffer("v2|track|pitch");
        buffer.write(0, [1]);
        buffer.write(999_999, [1]);
        const result = extendCurveViewToViewport({
            view: view({ startFrame: 999_990, length: 10 }),
            live: live(buffer),
            viewport: { startFrame: 999_990, endFrame: 1_000_000 },
            maxFrames: 8,
        });
        expect(result).toBeNull();
    });

    it("keeps a stride > 1 on the same lattice as the window", () => {
        const buffer = createStrokeBuffer("v2|track|pitch");
        for (let frame = 1000; frame <= 1400; frame += 1) buffer.write(frame, [7]);
        const extended = extendCurveViewToViewport({
            view: view({ startFrame: 1000, length: 10, stride: 5 }),
            live: live(buffer),
            viewport: { startFrame: 1100, endFrame: 1200 },
        });
        const segment = extended as ParamViewSegment;
        expect(segment.stride).toBe(5);
        // 起点必须在窗口同一格上：1000 + k×5。
        expect((segment.startFrame - 1000) % 5).toBe(0);
        expect(segment.edit.every((value) => value === 1 || value === 7)).toBe(true);
    });
});
