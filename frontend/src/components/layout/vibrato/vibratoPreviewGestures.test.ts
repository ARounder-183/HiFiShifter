import { describe, expect, test } from "vitest";

import { sanitizeVibratoPreset, VIBRATO_LIMITS } from "../../../features/vibrato/vibratoPresets";
import {
    advancePreviewFineDrag,
    applyPreviewGesture,
    createPreviewFineDragState,
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

    /*
     * 渐入 / 渐出各自都能拖满整条线，因此两个手柄完全可能叠在一起（例如渐入
     * 拉满、渐出归零，两个都贴到右缘）。此时若仍按"渐入优先"，渐出手柄就永远
     * 抓不到了 —— 按距离取近的那个。
     */
    test("两个手柄都命中时取更近的那个", () => {
        // 两者完全重合：等距 → 退回渐入（确定性优先，不随机）。
        expect(hitTestPreviewZone(400, 400, { attackFrac: 1, releaseFrac: 1 })).toEqual({
            kind: "attack",
        });
        // 渐入贴右缘（x=400）、渐出在其左侧 8px（x=392）：两者都在 9px 命中半径内，
        // 此时必须按距离判定，否则渐出一侧永远抓不到。
        const split = { attackFrac: 1, releaseFrac: 0.98 };
        expect(hitTestPreviewZone(394, 400, split)).toEqual({ kind: "release" });
        expect(hitTestPreviewZone(398, 400, split)).toEqual({ kind: "attack" });
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
    test("渐入：向右加长，可铺满整条线", () => {
        // 400px 宽 / 1600ms → 4ms per px。
        const next = applyPreviewGesture({ kind: "attack" }, snapshot(), 25, 0);
        expect(next.attackMs).toBeCloseTo(200, 9);
        // 上限是**整段时长**（而不是一半）：渐入可以拉满整条颤音线。
        const long = applyPreviewGesture({ kind: "attack" }, snapshot(), 10_000, 0);
        expect(long.attackMs).toBe(1600);
        const half = applyPreviewGesture({ kind: "attack" }, snapshot(), 100, 0);
        expect(half.attackMs).toBeCloseTo(500, 9);
        const short = applyPreviewGesture({ kind: "attack" }, snapshot(), -10_000, 0);
        expect(short.attackMs).toBe(0);
    });

    test("渐出：手柄跟手 —— 向右拖缩短，向左拖加长", () => {
        // 手柄画在 windowMs - releaseMs 处，向右拖即让斜坡起点右移，渐出变短。
        const shorter = applyPreviewGesture({ kind: "release" }, snapshot(), 25, 0);
        expect(shorter.releaseMs).toBeCloseTo(0, 9); // 100 - 25*4 → 钳到 0
        const longer = applyPreviewGesture({ kind: "release" }, snapshot(), -25, 0);
        expect(longer.releaseMs).toBeCloseTo(200, 9);
        // 上限同样是整段时长。
        const capped = applyPreviewGesture({ kind: "release" }, snapshot(), -10_000, 0);
        expect(capped.releaseMs).toBe(1600);
    });

    /*
     * 手感回归：**手柄必须跟着指针走**。
     *
     * 【为什么单测这条】"拖拽逻辑与直觉相反"是一个只有上手才会发现的缺陷，而且
     * 恰好不会抛错 —— 渐出手柄画在 `windowMs - releaseMs`，映射一旦忘了取反，
     * 往右拖手柄却往左跑。这里把"位移方向 → 手柄绘制方向"这条不变量钉住，不依赖
     * 具体数值。
     */
    test("渐入手柄跟手：向右拖 → 手柄右移", () => {
        const before = handleLayoutFor({ attackMs: 100, releaseMs: 100 }, 1600);
        const next = applyPreviewGesture({ kind: "attack" }, snapshot(), 25, 0);
        const after = handleLayoutFor({ attackMs: next.attackMs as number, releaseMs: 100 }, 1600);
        expect(after.attackFrac).toBeGreaterThan(before.attackFrac);
    });

    test("渐出手柄跟手：向右拖 → 手柄右移（不是反向）", () => {
        const before = handleLayoutFor({ attackMs: 100, releaseMs: 400 }, 1600);
        const next = applyPreviewGesture({ kind: "release" }, snapshot({ releaseMs: 400 }), 25, 0);
        const after = handleLayoutFor({ attackMs: 100, releaseMs: next.releaseMs as number }, 1600);
        expect(after.releaseFrac).toBeGreaterThan(before.releaseFrac);
    });

    test("主体：水平位移改相位（一个可见周期 = 360°）", () => {
        // cycleWidthPx = 40 → 40px = 360°。
        const next = applyPreviewGesture({ kind: "body" }, snapshot(), 10, 0);
        expect(next.startPhaseDeg).toBeCloseTo(90, 9);
        // 反向并取模。
        const back = applyPreviewGesture({ kind: "body" }, snapshot(), -10, 0);
        expect(back.startPhaseDeg).toBeCloseTo(270, 9);
    });

    test("主体：向上拖动加深，向下拖动变浅（可为负，即反相）", () => {
        // centsPerPx = 0.5：像素位移**乘以**它才是 cents 位移。
        // 向上 20px → 30 + 20*0.5 = 40。
        const deeper = applyPreviewGesture({ kind: "body" }, snapshot(), 0, -20);
        expect(deeper.depthCents).toBeCloseTo(40, 9);
        // 30 - 200*0.5 = -70：负值合法（波形反相）。
        const shallow = applyPreviewGesture({ kind: "body" }, snapshot(), 0, 200);
        expect(shallow.depthCents).toBeCloseTo(-70, 9);
        // 越过下界才钳住。
        const floored = applyPreviewGesture({ kind: "body" }, snapshot(), 0, 100_000);
        expect(floored.depthCents).toBe(VIBRATO_LIMITS.depthCents.min);
    });

    /*
     * 回归：纵向必须**乘** centsPerPx，不能除。
     *
     * 【为什么单测这条】除反了会让灵敏度随深度反比变化 —— 深度越小，每像素走过的
     * cents 越多，拖一点点就把幅度拉飞（"起始深度很小却变化特别大"）。而且它不会
     * 抛错，只会让手感不可控。这里把"与画布 1:1"这条不变量钉住：把波峰从中线附近
     * 拖到中线，深度应当归零。
     */
    test("纵向与画布 1:1：拖动等于画布半高的距离即把深度清零", () => {
        // 模拟画布：reach = 54px、halfCents = 1.15 × 深度。
        const depth = 40;
        const reach = 54;
        const centsPerPx = (depth * 1.15) / reach;
        // 波峰画在 midY - depth/halfCents*reach 处，即距中线 0.87*reach 像素。
        const peakOffsetPx = (depth / (depth * 1.15)) * reach;
        const next = applyPreviewGesture(
            { kind: "body" },
            snapshot({ depthCents: depth, centsPerPx }),
            0,
            peakOffsetPx,
        );
        expect(Math.abs(next.depthCents as number)).toBeLessThan(0.5);
    });

    test("小深度：每像素走过的 cents 更少（灵敏度与深度同向，而非反比）", () => {
        const reach = 54;
        const shallow = applyPreviewGesture(
            { kind: "body" },
            snapshot({ depthCents: 5, centsPerPx: (5 * 1.15) / reach }),
            0,
            -10,
        );
        const deep = applyPreviewGesture(
            { kind: "body" },
            snapshot({ depthCents: 100, centsPerPx: (100 * 1.15) / reach }),
            0,
            -10,
        );
        const shallowDelta = (shallow.depthCents as number) - 5;
        const deepDelta = (deep.depthCents as number) - 100;
        expect(shallowDelta).toBeGreaterThan(0);
        expect(shallowDelta).toBeLessThan(deepDelta);
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

describe("advancePreviewFineDrag（精细调整）", () => {
    test("未按修饰键：累计位移原样通过", () => {
        const state = createPreviewFineDragState(false);
        expect(advancePreviewFineDrag(state, 10, -20, false)).toEqual({ deltaX: 10, deltaY: -20 });
    });

    test("按住修饰键：累计位移按比例缩小", () => {
        const state = createPreviewFineDragState(true);
        const first = advancePreviewFineDrag(state, 10, -20, true);
        expect(first.deltaX).toBeCloseTo(2, 9);
        expect(first.deltaY).toBeCloseTo(-4, 9);
    });

    /*
     * ★ 用户报告的缺陷：拖拽途中按下 / 松开「精细调整」会让预览"闪回"。
     *
     * 根因是每帧用"从起点算起的**总**位移 × 当前比例"重算 —— 一按 Ctrl，此前
     * 已经累计的位移被整体重新缩小，数值瞬间跳回去，正在进行的拖拽被打断。
     * 正确做法是按**增量**缩放：比例变化只影响此后每帧走多少。
     */
    test("中途按下修饰键：累计量连续，不闪回", () => {
        const state = createPreviewFineDragState(false);
        const before = advancePreviewFineDrag(state, 100, 0, false);
        expect(before.deltaX).toBe(100);
        // 按下 Ctrl 的那一刻：累计量必须还在 100 附近，绝不回退。
        const atToggle = advancePreviewFineDrag(state, 100, 0, true);
        expect(atToggle.deltaX).toBeCloseTo(before.deltaX, 9);
        // 此后每帧只走一小步（远小于不按修饰键时的 1:1）。
        const after = advancePreviewFineDrag(state, 110, 0, true);
        expect(after.deltaX - before.deltaX).toBeLessThan(10);
        expect(after.deltaX - before.deltaX).toBeGreaterThan(0);
    });

    test("中途松开修饰键：累计量连续，此后恢复 1:1", () => {
        const state = createPreviewFineDragState(true);
        const before = advancePreviewFineDrag(state, 100, 0, true);
        const atToggle = advancePreviewFineDrag(state, 100, 0, false);
        expect(atToggle.deltaX).toBeCloseTo(before.deltaX, 9);
        const after = advancePreviewFineDrag(state, 110, 0, false);
        expect(after.deltaX - before.deltaX).toBeCloseTo(10, 9);
    });

    test("起手就按住修饰键：首帧即走稳态比例（不按「刚按下」处理）", () => {
        const state = createPreviewFineDragState(true);
        const first = advancePreviewFineDrag(state, 100, 0, true);
        // 若误判成"刚按下"，首帧会走 0.65 而不是 0.2。
        expect(first.deltaX).toBeCloseTo(20, 9);
    });

    test("精细调整确实减小了深度变化量（拖同样的距离只走一小步）", () => {
        const snapshotValue = snapshot({ depthCents: 30, centsPerPx: 0.5 });
        const coarse = applyPreviewGesture({ kind: "body" }, snapshotValue, 0, -20);
        const state = createPreviewFineDragState(true);
        const fine = advancePreviewFineDrag(state, 0, -20, true);
        const fineResult = applyPreviewGesture(
            { kind: "body" },
            snapshotValue,
            fine.deltaX,
            fine.deltaY,
        );
        const coarseDelta = (coarse.depthCents as number) - 30;
        const fineDelta = (fineResult.depthCents as number) - 30;
        expect(fineDelta).toBeGreaterThan(0);
        expect(fineDelta).toBeLessThan(coarseDelta);
    });
});
