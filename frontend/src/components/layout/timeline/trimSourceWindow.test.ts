/**
 * ★ 回归：裁切 / 延伸的**源窗口方向**（含倒放）。
 *
 * ## 缺陷形态
 *
 * 用户直觉的规则**与是否倒放无关**：拖**右**边 ⇒ **左**边内容固定；拖**左**边 ⇒
 * **右**边内容固定。而"时间轴左/右端分别播放哪个源位置"在倒放时是**镜像**的
 *（后端 `state.rs:121` `clip_playback_window_sec`：倒放 `win = [se − len·r, se)`，
 * 升序消费后整体翻转 ⇒ 左端播 `se`、右端播 `se − len·r`）。
 *
 * 修复前这里按"正放"写死了 `右缘 ⇒ 动 sourceEndSec`，于是倒放 Clip：
 *
 * - 拖右缘改的是 `sourceEndSec`（那是**左**端）⇒ 左端跟着动，固定成了右端；
 * - 拖左缘改的是 `sourceStartSec`（那是**右**端）⇒ 右端跟着动，固定成了左端。
 *
 * ## 本用例钉住什么
 *
 * 不只比字段，而是**换算回消费窗口**断言"被固定的那一侧源位置逐值不变" ——
 * 这才是用户看到的东西。字段层面的对错可以有两种等价写法，消费窗口只有一种。
 */
import { describe, expect, it } from "vitest";

import { resolveTrimSourceWindow } from "./trimSourceWindow";

/**
 * 时间轴某一端播放的源位置（后端 `clip_playback_window_sec` 的镜像）。
 *
 * 正放 `win = [ss, ss + len·r)`，倒放 `win = [se − len·r, se)` 且输出翻转。
 */
function timelineEdgeSourceSec(args: {
    side: "left" | "right";
    reversed: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    lengthSec: number;
    rate: number;
}): number {
    const span = args.lengthSec * args.rate;
    if (args.side === "left") {
        return args.reversed ? args.sourceEndSec : args.sourceStartSec;
    }
    return args.reversed ? args.sourceEndSec - span : args.sourceStartSec + span;
}

interface Case {
    edge: "left" | "right";
    reversed: boolean;
    deltaSec: number;
    sourceStartSec: number;
    sourceEndSec: number;
    lengthSec: number;
    rate: number;
}

function resolve(args: Case) {
    return resolveTrimSourceWindow({
        edge: args.edge,
        reversed: args.reversed,
        loopEnabled: false,
        deltaSec: args.deltaSec,
        rate: args.rate,
        sourceStartSec: args.sourceStartSec,
        sourceEndSec: args.sourceEndSec,
    });
}

/** 拖拽后的新长度（与 `resolveTrimEdge` 同口径）。 */
function nextLength(args: Case): number {
    return args.edge === "right" ? args.lengthSec + args.deltaSec : args.lengthSec - args.deltaSec;
}

describe("resolveTrimSourceWindow：固定端不变（正放）", () => {
    const base: Case = {
        edge: "right",
        reversed: false,
        deltaSec: 1,
        sourceStartSec: 6,
        sourceEndSec: 10,
        lengthSec: 4,
        rate: 1,
    };

    it("拖右缘 ⇒ 左端（sourceStartSec）固定", () => {
        const before = timelineEdgeSourceSec({ ...base, side: "left" });
        const out = resolve(base)!;
        expect(out.sourceStartSec).toBe(6);
        expect(out.sourceEndSec).toBe(11);
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "left",
            }),
        ).toBeCloseTo(before, 12);
    });

    it("拖左缘 ⇒ 右端（sourceEndSec）固定", () => {
        const c: Case = { ...base, edge: "left" };
        const before = timelineEdgeSourceSec({ ...c, side: "right" });
        const out = resolve(c)!;
        expect(out.sourceStartSec).toBe(7);
        expect(out.sourceEndSec).toBe(10);
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "right",
            }),
        ).toBeCloseTo(before, 12);
    });
});

describe("★ resolveTrimSourceWindow：固定端不变（倒放）", () => {
    // 倒放、se=10、len=4、r=1 ⇒ win=[6,10]：**左端播 10、右端播 6**。
    const base: Case = {
        edge: "right",
        reversed: true,
        deltaSec: 1,
        sourceStartSec: 6,
        sourceEndSec: 10,
        lengthSec: 4,
        rate: 1,
    };

    it("★ 拖右缘 ⇒ 左端固定：动的是 sourceStartSec，且方向为负", () => {
        const before = timelineEdgeSourceSec({ ...base, side: "left" });
        expect(before).toBe(10); // 左端播 se
        const out = resolve(base)!;
        // 修复前这里会写 sourceEndSec（= 左端）⇒ 左端跟着动。
        expect(out.sourceEndSec).toBe(10);
        expect(out.sourceStartSec).toBe(5); // 6 − 1·1
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "left",
            }),
        ).toBeCloseTo(10, 12);
        // 右端则按内容消费移动：新 win=[5,10]，右端播 5。
        expect(
            timelineEdgeSourceSec({
                ...base,
                ...out,
                lengthSec: nextLength(base),
                side: "right",
            }),
        ).toBeCloseTo(5, 12);
    });

    it("★ 拖左缘 ⇒ 右端固定：动的是 sourceEndSec，且方向为负", () => {
        const c: Case = { ...base, edge: "left" };
        const before = timelineEdgeSourceSec({ ...c, side: "right" });
        expect(before).toBe(6); // 右端播 se − len·r
        const out = resolve(c)!;
        // 修复前这里会写 sourceStartSec（= 右端）⇒ 右端跟着动。
        expect(out.sourceStartSec).toBe(6);
        expect(out.sourceEndSec).toBe(9); // 10 − 1·1
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "right",
            }),
        ).toBeCloseTo(6, 12);
    });

    it("缩短（deltaSec 为负）同样保持固定端", () => {
        const c: Case = { ...base, deltaSec: -1 };
        const before = timelineEdgeSourceSec({ ...c, side: "left" });
        const out = resolve(c)!;
        expect(out.sourceStartSec).toBe(7);
        expect(
            timelineEdgeSourceSec({
                ...c,
                ...out,
                lengthSec: nextLength(c),
                side: "left",
            }),
        ).toBeCloseTo(before, 12);
    });
});

describe("resolveTrimSourceWindow：Loop / 非法输入", () => {
    it("Loop ⇒ null（源字段是回绕锚点，裁切只改长度）", () => {
        expect(
            resolveTrimSourceWindow({
                edge: "right",
                reversed: false,
                loopEnabled: true,
                deltaSec: 1,
                rate: 1,
                sourceStartSec: 0.5,
                sourceEndSec: 4,
            }),
        ).toBeNull();
        expect(
            resolveTrimSourceWindow({
                edge: "left",
                reversed: true,
                loopEnabled: true,
                deltaSec: -1,
                rate: 2,
                sourceStartSec: 0.5,
                sourceEndSec: 4,
            }),
        ).toBeNull();
    });

    it("deltaSec 非有限 ⇒ null（调用方跳过该帧）", () => {
        expect(
            resolveTrimSourceWindow({
                edge: "right",
                reversed: false,
                loopEnabled: false,
                deltaSec: Number.NaN,
                rate: 1,
                sourceStartSec: 0,
                sourceEndSec: 1,
            }),
        ).toBeNull();
    });

    it("速率非法 ⇒ 按 1 处理（与后端 pr_valid 同口径）", () => {
        const out = resolveTrimSourceWindow({
            edge: "right",
            reversed: false,
            loopEnabled: false,
            deltaSec: 2,
            rate: Number.NaN,
            sourceStartSec: 0,
            sourceEndSec: 4,
        })!;
        expect(out.sourceEndSec).toBe(6);
    });
});

describe("resolveTrimSourceWindow：组合速率（take 速率 ≠ 1）", () => {
    it("源位移按组合速率折算（正放）", () => {
        const out = resolve({
            edge: "right",
            reversed: false,
            deltaSec: 1,
            sourceStartSec: 6,
            sourceEndSec: 10,
            lengthSec: 4,
            rate: 2, // clip 1× × take 2×
        })!;
        expect(out.sourceEndSec).toBe(12);
        expect(out.sourceStartSec).toBe(6);
    });

    it("★ 组合速率 + 倒放：方向仍为负、幅度按组合速率", () => {
        const out = resolve({
            edge: "right",
            reversed: true,
            deltaSec: 1,
            sourceStartSec: 6,
            sourceEndSec: 10,
            lengthSec: 4,
            rate: 2,
        })!;
        expect(out.sourceStartSec).toBe(4); // 6 − 1·2
        expect(out.sourceEndSec).toBe(10);
    });
});

describe("resolveTrimSourceWindow：消费窗口层面的方向不变式", () => {
    it("★ 四种组合 × 两个方向：被拖边的**对侧**源位置逐值不变", () => {
        for (const reversed of [false, true]) {
            for (const edge of ["left", "right"] as const) {
                for (const rate of [1, 0.5, 2]) {
                    for (const deltaSec of [-1.5, -0.25, 0.25, 1.5]) {
                        const c: Case = {
                            edge,
                            reversed,
                            deltaSec,
                            sourceStartSec: 6,
                            sourceEndSec: 10,
                            lengthSec: 4,
                            rate,
                        };
                        const fixedSide = edge === "right" ? "left" : "right";
                        const before = timelineEdgeSourceSec({ ...c, side: fixedSide });
                        const out = resolve(c)!;
                        const after = timelineEdgeSourceSec({
                            reversed,
                            sourceStartSec: out.sourceStartSec,
                            sourceEndSec: out.sourceEndSec,
                            lengthSec: nextLength(c),
                            rate,
                            side: fixedSide,
                        });
                        expect(after).toBeCloseTo(before, 9);
                    }
                }
            }
        }
    });
});
