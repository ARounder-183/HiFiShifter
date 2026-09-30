import { describe, expect, test } from "vitest";

import { sanitizeVibratoPreset } from "../../../features/vibrato/vibratoPresets";
import {
    applyPreviewGesture,
    cursorForZone,
    cycleWidthPxFor,
    handleLayoutFor,
    hitTestPreviewZone,
    wrapPhaseDeg,
    type PreviewGestureSnapshot,
} from "./vibratoPreviewGestures";

const snapshot = (overrides: Partial<PreviewGestureSnapshot> = {}): PreviewGestureSnapshot => ({
    attackMs: 100,
    releaseMs: 100,
    depthCents: 30,
    startPhaseDeg: 0,
    windowMs: 1600,
    widthPx: 400,
    cycleWidthPx: 40,
    centsPerPx: 0.5,
    ...overrides,
});

describe("hitTestPreviewZone", () => {
    const layout = { attackFrac: 0.1, releaseFrac: 0.9 };

    test("手柄附近优先命中（即使不在外缘带内）", () => {
        // 手柄被拖到画布中部时仍要抓得住。
        const mid = { attackFrac: 0.5, releaseFrac: 0.6 };
        expect(hitTestPreviewZone(200, 400, mid)).toEqual({ kind: "attack" });
        expect(hitTestPreviewZone(240, 400, mid)).toEqual({ kind: "release" });
    });

    test("手柄之外的左右外缘 12% 是后备抓手", () => {
        const offscreen = { attackFrac: 0, releaseFrac: 1 };
        expect(hitTestPreviewZone(10, 400, offscreen)).toEqual({ kind: "attack" });
        expect(hitTestPreviewZone(390, 400, offscreen)).toEqual({ kind: "release" });
    });

    test("中间区域是主体", () => {
        expect(hitTestPreviewZone(200, 400, layout)).toEqual({ kind: "body" });
    });

    test("宽度为 0 不除零（退化输入返回主体）", () => {
        expect(hitTestPreviewZone(0.5, 0, { attackFrac: 0.1, releaseFrac: 0.9 })).toEqual({
            kind: "body",
        });
    });
});

describe("cursorForZone", () => {
    test("手柄用左右缩放光标，主体用移动光标", () => {
        expect(cursorForZone({ kind: "attack" })).toBe("ew-resize");
        expect(cursorForZone({ kind: "release" })).toBe("ew-resize");
        expect(cursorForZone({ kind: "body" })).toBe("move");
    });
});

describe("wrapPhaseDeg", () => {
    test("取模到 [0,360)", () => {
        expect(wrapPhaseDeg(0)).toBe(0);
        expect(wrapPhaseDeg(370)).toBeCloseTo(10, 9);
        expect(wrapPhaseDeg(-10)).toBeCloseTo(350, 9);
        expect(wrapPhaseDeg(-720)).toBe(0);
    });
});

describe("applyPreviewGesture", () => {
    test("渐入：向右加长，钳在窗口一半", () => {
        // 400px 宽 / 1600ms → 4ms per px。
        const next = applyPreviewGesture({ kind: "attack" }, snapshot(), 25, 0);
        expect(next.attackMs).toBeCloseTo(200, 9);
        const long = applyPreviewGesture({ kind: "attack" }, snapshot(), 10_000, 0);
        expect(long.attackMs).toBe(800); // windowMs / 2
        const short = applyPreviewGesture({ kind: "attack" }, snapshot(), -10_000, 0);
        expect(short.attackMs).toBe(0);
    });

    test("渐出：向右加长（与渐入同向）", () => {
        const next = applyPreviewGesture({ kind: "release" }, snapshot(), 25, 0);
        expect(next.releaseMs).toBeCloseTo(200, 9);
        const long = applyPreviewGesture({ kind: "release" }, snapshot(), 10_000, 0);
        expect(long.releaseMs).toBe(800);
    });

    test("主体：水平位移改相位（一个可见周期 = 360°）", () => {
        // cycleWidthPx = 40 → 40px = 360°。
        const next = applyPreviewGesture({ kind: "body" }, snapshot(), 10, 0);
        expect(next.startPhaseDeg).toBeCloseTo(90, 9);
        // 反向并取模。
        const back = applyPreviewGesture({ kind: "body" }, snapshot(), -10, 0);
        expect(back.startPhaseDeg).toBeCloseTo(270, 9);
    });

    test("主体：向上拖动加深，向下拖动变浅且不小于 0", () => {
        // centsPerPx = 0.5 → 向上 20px = +40 cents。
        const deeper = applyPreviewGesture({ kind: "body" }, snapshot(), 0, -20);
        expect(deeper.depthCents).toBeCloseTo(70, 9);
        const shallow = applyPreviewGesture({ kind: "body" }, snapshot(), 0, 200);
        expect(shallow.depthCents).toBe(0);
    });

    test("退化几何不产生 NaN", () => {
        const degenerate = snapshot({ widthPx: 0, cycleWidthPx: 0, centsPerPx: 0 });
        const next = applyPreviewGesture({ kind: "body" }, degenerate, 10, -10);
        expect(Number.isFinite(next.startPhaseDeg as number)).toBe(true);
        expect(Number.isFinite(next.depthCents as number)).toBe(true);
    });
});

describe("cycleWidthPxFor", () => {
    test("hz 模式：宽度 /（Hz × 秒）", () => {
        const preset = sanitizeVibratoPreset({ id: "custom_a", rateMode: "hz", rateHz: 5 });
        // 400px / (5Hz × 1.6s) = 50px。
        expect(cycleWidthPxFor(preset, 400, 1600)).toBeCloseTo(50, 9);
    });

    test("cycles 模式：整段周期数摊到窗口时长", () => {
        const preset = sanitizeVibratoPreset({ id: "custom_a", rateMode: "cycles", cycles: 8 });
        // 8 个周期铺满 400px → 每周期 50px。
        expect(cycleWidthPxFor(preset, 400, 1600)).toBeCloseTo(50, 9);
    });

    test("速率为 0 时回退到整段宽度（不除零）", () => {
        const preset = { ...sanitizeVibratoPreset({ id: "custom_a" }), rateHz: 0 };
        expect(cycleWidthPxFor(preset, 400, 1600)).toBe(400);
    });
});

describe("handleLayoutFor", () => {
    test("渐入在 attackMs 处，渐出在 windowMs - releaseMs 处", () => {
        const layout = handleLayoutFor({ attackMs: 400, releaseMs: 800 }, 1600);
        expect(layout.attackFrac).toBeCloseTo(0.25, 9);
        expect(layout.releaseFrac).toBeCloseTo(0.5, 9);
    });

    test("超出窗口时钳到 [0,1]", () => {
        const layout = handleLayoutFor({ attackMs: 5000, releaseMs: 5000 }, 1600);
        expect(layout.attackFrac).toBe(1);
        expect(layout.releaseFrac).toBe(0);
    });
});
