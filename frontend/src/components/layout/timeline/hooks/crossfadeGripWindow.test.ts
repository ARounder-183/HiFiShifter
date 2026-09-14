/**
 * crossfadeGripWindow（交叉点抓手几何）行为自检。
 *
 * 【主要内容】
 * 1. 同向模式：两侧同向平移、重叠不变、A 用派生窗口；倒放 / Loop 的分支；
 * 2. 反向模式：两侧相向移动、重叠变化、淡变按比例缩放（auto → auto 字段）；
 * 3. 可行区间：两侧共享同一位移，任一顶到极限时整组停住；无交集返回 null。
 *
 * 【作用】这些断言是「抓手拖拽把源窗口写坏」这类缺陷的回归护栏：源窗口错了
 * 但长度是对的，画面上完全看不出来，只有波形与听感会错。
 *
 * 【与其他模块的关系】覆盖 `crossfadeGripWindow.ts`；不依赖 React / Redux。
 */

import { describe, expect, it } from "vitest";

import {
    computeCrossfadeGrip,
    type CrossfadeGripArgs,
    type CrossfadeGripClipBase,
} from "./crossfadeGripWindow";

function clip(
    id: string,
    startSec: number,
    lengthSec: number,
    extra: Partial<CrossfadeGripClipBase> = {},
): CrossfadeGripClipBase {
    return {
        id,
        startSec,
        lengthSec,
        sourceStartSec: 0,
        sourceEndSec: 4,
        playbackRate: 1,
        loopEnabled: false,
        reversed: false,
        mediaDurationSec: 10,
        ...extra,
    };
}

function args(over: Partial<CrossfadeGripArgs> = {}): CrossfadeGripArgs {
    return {
        // A: [2,6)（长 4，源 [0,4)），B: [4,8)（长 4）→ 重叠 2 秒。
        earlier: clip("a", 2, 4),
        later: clip("b", 4, 4),
        deltaSec: 0,
        opposite: false,
        baseOverlapSec: 2,
        earlierFadeOutSec: 2,
        laterFadeInSec: 2,
        earlierFadeOutAuto: false,
        laterFadeInAuto: false,
        ...over,
    };
}

describe("computeCrossfadeGrip · 同向模式", () => {
    it("两侧同向平移、重叠长度不变、A 走派生窗口", () => {
        const out = computeCrossfadeGrip(args({ deltaSec: 1 }));
        expect(out).not.toBeNull();
        // A 右缘 +1 → 长度 5，派生窗口终点 = 0 + 5×1 = 5。
        expect(out!.earlier).toEqual({
            clipId: "a",
            startSec: 2,
            lengthSec: 5,
            sourceEndSec: 5,
        });
        // B 左缘 +1 → 起点 5、长度 3，正放派生窗口起点 = 0 + 1×1 = 1。
        expect(out!.later).toEqual({
            clipId: "b",
            startSec: 5,
            lengthSec: 3,
            sourceStartSec: 1,
        });
        // 重叠不变 → 不缩放淡变。
        expect(out!.fades).toEqual([]);
    });

    it("位移被 B 的长度限住（B 不能裁成负长度）", () => {
        const out = computeCrossfadeGrip(args({ deltaSec: 99 }));
        expect(out!.deltaSec).toBe(4);
        expect(out!.later.lengthSec).toBe(0);
    });

    it("负位移被 A 的长度限住（A 不能裁成负长度）", () => {
        const out = computeCrossfadeGrip(args({ deltaSec: -99 }));
        expect(out!.deltaSec).toBe(-4);
        expect(out!.earlier.lengthSec).toBe(0);
    });

    it("Loop 的 A 只改长度、不动源窗口", () => {
        const out = computeCrossfadeGrip(
            args({ earlier: clip("a", 2, 4, { loopEnabled: true }), deltaSec: 1 }),
        );
        expect(out!.earlier.sourceEndSec).toBeUndefined();
        expect(out!.earlier.sourceStartSec).toBeUndefined();
        expect(out!.earlier.lengthSec).toBe(5);
    });

    it("倒放非 Loop 的 A 向右延伸时消费窗口起点，且不越出 sourceEnd", () => {
        const out = computeCrossfadeGrip(
            args({
                earlier: clip("a", 2, 4, {
                    reversed: true,
                    sourceStartSec: 3,
                    sourceEndSec: 8,
                }),
                deltaSec: 1,
            }),
        );
        // nextTrimStart = clamp(3 - 1×1, 0, 8) = 2
        expect(out!.earlier.sourceStartSec).toBe(2);
        expect(out!.earlier.lengthSec).toBe(5);
    });

    it("倒放非 Loop 的 A 窗口耗尽后停在 0（其余部分按静音渲染）", () => {
        const out = computeCrossfadeGrip(
            args({
                earlier: clip("a", 2, 4, {
                    reversed: true,
                    sourceStartSec: 0.5,
                    sourceEndSec: 8,
                }),
                deltaSec: 3,
            }),
        );
        expect(out!.earlier.sourceStartSec).toBe(0);
        expect(out!.earlier.lengthSec).toBe(7);
    });

    it("Loop 的 B 左缘延伸时锚点在媒体域内环绕", () => {
        const out = computeCrossfadeGrip(
            args({
                later: clip("b", 4, 4, {
                    loopEnabled: true,
                    sourceStartSec: 0.5,
                    mediaDurationSec: 10,
                }),
                deltaSec: -1,
            }),
        );
        // startDelta = -1 → sourceStart = 0.5 + (-1)×1 = -0.5 → 环绕为 9.5
        expect(out!.later.sourceStartSec).toBeCloseTo(9.5, 6);
    });

    it("非法位移返回 null", () => {
        expect(computeCrossfadeGrip(args({ deltaSec: Number.NaN }))).toBeNull();
    });
});

describe("computeCrossfadeGrip · 反向模式", () => {
    it("两侧相向移动，重叠按 2×delta 变化", () => {
        const out = computeCrossfadeGrip(args({ opposite: true, deltaSec: 0.5 }));
        // A 右缘 +0.5 → 长度 4.5；B 左缘 -0.5 → 起点 3.5、长度 4.5。
        expect(out!.earlier.lengthSec).toBeCloseTo(4.5, 6);
        expect(out!.later.startSec).toBeCloseTo(3.5, 6);
        expect(out!.later.lengthSec).toBeCloseTo(4.5, 6);
    });

    it("手动淡变按新重叠比例缩放（写手动字段）", () => {
        const out = computeCrossfadeGrip(args({ opposite: true, deltaSec: 0.5 }));
        // newOverlap = 2 + 2×0.5 = 3；ratio = 1.5；2 → 3。
        expect(out!.fades).toEqual([
            { clipId: "a", fadeOutSec: 3 },
            { clipId: "b", fadeInSec: 3 },
        ]);
    });

    it("自动交叉淡化按比例缩放（写 auto 字段，保持 auto == 重叠）", () => {
        const out = computeCrossfadeGrip(
            args({
                opposite: true,
                deltaSec: 0.5,
                earlierFadeOutAuto: true,
                laterFadeInAuto: true,
            }),
        );
        expect(out!.fades).toEqual([
            { clipId: "a", autoFadeOutSec: 3 },
            { clipId: "b", autoFadeInSec: 3 },
        ]);
    });

    it("向左拖时重叠收缩但不会低于最小重叠", () => {
        const out = computeCrossfadeGrip(args({ opposite: true, deltaSec: -99 }));
        // minDelta = max(-2, -4, (0.0002-2)/2 = -0.9999) = -0.9999
        expect(out!.deltaSec).toBeCloseTo(-0.9999, 6);
        const overlap = 2 + 2 * out!.deltaSec;
        expect(overlap).toBeCloseTo(0.0002, 6);
    });

    it("同向模式不产生淡变缩放", () => {
        expect(computeCrossfadeGrip(args({ deltaSec: 0.5 }))!.fades).toEqual([]);
    });
});
