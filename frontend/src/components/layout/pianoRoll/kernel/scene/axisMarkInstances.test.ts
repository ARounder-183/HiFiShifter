/**
 * 参数编辑器内核 · 数值轴刻度实例构建（纯函数）单测。
 *
 * 【本测试守护什么】
 * 1. **步长选择**：候选步长表与 [5,12] 数量约束必须与 `render.ts` 一致，否则两条
 *    渲染路径的刻度密度会不同（视觉上"某种参数下刻度变密了"）；
 * 2. cents 的**退化修正**：跨度过大时回退到更粗的 nice step；
 * 3. 刻度线的 y 用 `valueToY + 0.5` 且线宽 strong=1.25 / weak=1（既有行为）；
 * 4. degrees 会**额外补一条只画标签的 0 刻度**（原实现的 `fillText` 没有配对的
 *    `stroke`）——补错会让 0 处多出一条线。
 */
import { describe, expect, it } from "vitest";

import {
    buildAxisMarkInstances,
    formatAxisMarkLabel,
    niceAxisStep,
    resolveAxisKind,
    resolveAxisStep,
    type AxisKind,
} from "./axisMarkInstances";

/** 值 → y 的线性投影桩（值 0 在中心，便于手算）。 */
const makeValueToY = () => (v: number, h: number) => h / 2 - (v / 100) * h;

const baseArgs = {
    heightPx: 400,
    axisWidthPx: 56,
    dpr: 2,
    valueToY: makeValueToY(),
};

describe("niceAxisStep", () => {
    it("按 1 / 2 / 5 / 10 档取整（与 render.ts 一致）", () => {
        expect(niceAxisStep(10, 4)).toBe(2); // rough 2.5 -> normalized 2.5 -> 2
        expect(niceAxisStep(100, 4)).toBe(20); // rough 25 -> 2 -> 20
        expect(niceAxisStep(3, 4)).toBe(1); // rough 0.75 -> mag .1, norm 7.5 -> 10*.1=1
    });
});

describe("resolveAxisKind", () => {
    it("按参数名识别种类", () => {
        expect(resolveAxisKind("child_pitch_offset_cents@t1")).toBe("cents");
        expect(resolveAxisKind("child_formant_offset_cents@t1")).toBe("formantCents");
        expect(resolveAxisKind("child_pitch_offset_degrees@t1")).toBe("degrees");
        expect(resolveAxisKind("something_else")).toBe("fallback");
    });
});

describe("resolveAxisStep", () => {
    it("cents / formant 的候选表与强刻度间隔", () => {
        // 候选表降序、取第一个使刻度数落在 [5,12] 的项：
        //   range=300 -> 100 给 4 个（不足 5）-> 50 给 7 个 ✓
        expect(resolveAxisStep("cents", 300)).toEqual({ step: 50, strongMod: 1200 });
        //   range=200 -> 100 给 3 个 -> 50 给 5 个 ✓
        expect(resolveAxisStep("formantCents", 200)).toEqual({
            step: 50,
            strongMod: 600,
        });
    });

    it("degrees 用更粗的候选表、强刻度每 7 个内部单位", () => {
        const r = resolveAxisStep("degrees", 7);
        expect(r.strongMod).toBe(7);
        expect([14, 7, 3, 1]).toContain(r.step);
    });

    it("cents 跨度过大时回退到更粗步长（退化修正）", () => {
        // 极大跨度 -> 候选表里最小的 1 会给出过多刻度 -> 回退
        const r = resolveAxisStep("cents", 100000);
        expect(r.step).toBeGreaterThan(1);
        // 回退后刻度数应显著下降
        expect(100000 / r.step).toBeLessThan(100000);
    });

    it("fallback 种返回有限步长（调用方不应依赖 strongMod）", () => {
        const r = resolveAxisStep("fallback", 300);
        expect(Number.isFinite(r.step)).toBe(true);
        expect(r.step).toBeGreaterThan(0);
    });
});

describe("buildAxisMarkInstances", () => {
    it("按步长产出刻度，数量落在合理范围", () => {
        const marks = buildAxisMarkInstances({
            ...baseArgs,
            kind: "cents",
            view: { center: 0, span: 1000 },
            paramName: "child_pitch_offset_cents@t1",
        });
        // 步长选择按"ceil(range/step)+1 ∈ [5,12]"筛选候选（range=1000 -> step=300），
        // 但**实际落进视口的刻度**可以少于该估计值（此处是 -300/0/300 共 3 个）。
        // 因此这里只断言"确实产出了刻度且不过密"，不硬编码数量。
        expect(marks.length).toBeGreaterThan(0);
        expect(marks.length).toBeLessThanOrEqual(14);
    });

    it("刻度线 y 用 valueToY + 0.5，线宽 strong 1.25 / weak 1", () => {
        const marks = buildAxisMarkInstances({
            ...baseArgs,
            kind: "cents",
            view: { center: 0, span: 1200 },
            paramName: "child_pitch_offset_cents@t1",
        });
        const strong = marks.filter((m) => m.isStrong);
        const weak = marks.filter((m) => !m.isStrong);
        expect(strong.length).toBeGreaterThan(0);
        expect(weak.length).toBeGreaterThan(0);
        for (const m of marks) {
            expect(m.line.h).toBeCloseTo(m.isStrong ? 1.25 : 1, 9);
            expect(m.line.w).toBe(56);
            expect(m.line.x).toBe(0);
            // 上缘 + 半宽 == valueToY + 0.5（stroke 中心语义）
            const expectedCenter = makeValueToY()(m.value, 400) + 0.5;
            expect(m.line.y + m.line.h / 2).toBeCloseTo(expectedCenter, 9);
        }
    });

    it("strong 判定按各类的间隔（cents 1200 / formant 600 / degrees 7）", () => {
        const cents = buildAxisMarkInstances({
            ...baseArgs,
            kind: "cents",
            view: { center: 0, span: 2400 },
            paramName: "child_pitch_offset_cents@t1",
        });
        for (const m of cents) {
            expect(m.isStrong).toBe(Math.round(m.value) % 1200 === 0);
        }
        const formant = buildAxisMarkInstances({
            ...baseArgs,
            kind: "formantCents",
            view: { center: 0, span: 1200 },
            paramName: "child_formant_offset_cents@t1",
        });
        for (const m of formant) {
            expect(m.isStrong).toBe(Math.round(m.value) % 600 === 0);
        }
    });

    it("degrees 无条件补一条只画标签的 0 刻度（复刻 legacy 的重复绘制）", () => {
        // 【这是一个刻意的"不一致"，必须复刻】render.ts:611-613 无条件 fillText(0)。
        // 由于 degrees 的步长是整数、刻度序列是步长的整数倍，**视口含 0 时 0 已是
        // 普通刻度**，此时原实现会在同一位置画两次标签，alpha 叠加后明显更深
        // （0.55 -> 0.7975）。因此构建器也必须产出两条 value=0 的实例。
        const withZero = buildAxisMarkInstances({
            ...baseArgs,
            kind: "degrees",
            view: { center: 3.5, span: 7 },
            paramName: "child_pitch_offset_degrees@t1",
        });
        const zeros = withZero.filter((m) => m.value === 0);
        expect(zeros.length).toBe(2);
        // 恰好多出来的那一次没有配对的分隔线（lineOnly）
        expect(zeros.filter((m) => m.lineOnly).length).toBe(1);
        expect(zeros.filter((m) => !m.lineOnly).length).toBe(1);
        // 无论视口是否含 0，都恰好补一条 lineOnly
        const offZero = buildAxisMarkInstances({
            ...baseArgs,
            kind: "degrees",
            view: { center: 8.5, span: 7 },
            paramName: "child_pitch_offset_degrees@t1",
        });
        expect(offZero.filter((m) => m.lineOnly).length).toBe(1);
        expect(offZero.filter((m) => m.lineOnly)[0].value).toBe(0);
    });

    it("非有限 span / 非法 dpr 返回空数组（防死循环）", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(
                buildAxisMarkInstances({
                    ...baseArgs,
                    kind: "cents",
                    view: { center: 0, span: bad },
                }),
            ).toEqual([]);
        }
        expect(
            buildAxisMarkInstances({
                ...baseArgs,
                kind: "cents",
                view: { center: 0, span: 100 },
                dpr: 0,
            }),
        ).toEqual([]);
    });
});

describe("formatAxisMarkLabel", () => {
    it("取 4 位有效数字并去掉尾随零", () => {
        expect(formatAxisMarkLabel(0)).toBe("0");
        expect(formatAxisMarkLabel(100)).toBe("100");
        expect(formatAxisMarkLabel(1 / 3)).toBe("0.3333");
    });

    it("极大 / 极小值会输出指数记法（字形图集必须覆盖 e/+/-）", () => {
        expect(formatAxisMarkLabel(1e-7)).toContain("e");
        expect(formatAxisMarkLabel(1e21)).toContain("e");
    });

    it("度数参数经显示换算", () => {
        const withParam = formatAxisMarkLabel(7, "child_pitch_offset_degrees@t1");
        const without = formatAxisMarkLabel(7);
        expect(withParam).not.toBe(without);
    });
});

/**
 * 与 Canvas2D 路径的**逐值等价**守护。
 *
 * 【为什么必须有这一组】上面的用例只证明"构建器符合我对 render.ts 的理解"。
 * 这里原地复刻 `render.ts:507-628` 的四个分支（不改动原文件），在
 * 种类 × center × span 的组合空间上逐刻度比对**值、标签、强判定与是否画线**。
 *
 * 这一组实际抓到过两个真实分歧（都已在实现里修正）：
 * 1. 回退分支必须用 `niceAxisStep(span, 4)`，而不是 cents 的候选表——后者在
 *    小跨度下差几个数量级（span=1e-6 时 1 个刻度 vs 5 个）；
 * 2. degrees 的 0 标签是**无条件重复绘制**的（原实现没有去重），不是"缺失时补"。
 */
describe("与 render.ts 刻度分支逐值等价", () => {
    function legacyMarks(kind: AxisKind, center: number, span: number) {
        const s = Math.max(1e-6, span);
        const vMin = center - s / 2;
        const vMax = center + s / 2;
        const rows: { v: number; strong: boolean; label: string; hasLine: boolean }[] = [];
        if (kind === "cents" || kind === "formantCents") {
            const range = vMax - vMin;
            const candidates = [1200, 600, 300, 200, 100, 50, 25, 10, 5, 1];
            let chosen = candidates[candidates.length - 1];
            for (const c of candidates) {
                const n = Math.ceil(range / c) + 1;
                if (n >= 5 && n <= 12) {
                    chosen = c;
                    break;
                }
            }
            if (kind === "cents") {
                const approx = range / chosen;
                if (approx > 12) {
                    const ns = niceAxisStep(range, 8);
                    if (ns > chosen) chosen = ns;
                    else {
                        const larger = candidates.find((c) => c > chosen);
                        if (larger !== undefined) chosen = larger;
                    }
                }
            }
            const strongMod = kind === "cents" ? 1200 : 600;
            for (
                let m = Math.ceil(vMin / chosen) * chosen;
                m <= vMax + chosen * 0.01;
                m += chosen
            ) {
                rows.push({
                    v: m,
                    strong: Math.round(m) % strongMod === 0,
                    label: formatAxisMarkLabel(m),
                    hasLine: true,
                });
            }
        } else if (kind === "degrees") {
            const candidates = [14, 7, 3, 1];
            let chosen = candidates[candidates.length - 1];
            for (const c of candidates) {
                const n = Math.ceil((vMax - vMin) / c) + 1;
                if (n >= 5 && n <= 12) {
                    chosen = c;
                    break;
                }
            }
            for (
                let m = Math.ceil(vMin / chosen) * chosen;
                m <= vMax + chosen * 0.01;
                m += chosen
            ) {
                rows.push({
                    v: m,
                    strong: Math.round(m) % 7 === 0,
                    label: formatAxisMarkLabel(m),
                    hasLine: true,
                });
            }
            // 原实现无条件补画 0 标签（不去重）
            rows.push({ v: 0, strong: false, label: formatAxisMarkLabel(0), hasLine: false });
        } else {
            const ns = niceAxisStep(s, 4);
            for (let m = Math.ceil(vMin / ns) * ns; m <= vMax + ns * 0.01; m += ns) {
                rows.push({ v: m, strong: false, label: formatAxisMarkLabel(m), hasLine: true });
            }
        }
        return rows;
    }

    it("四分支 × center × span 组合下值 / 标签 / 强判定 / 画线一致", () => {
        let compared = 0;
        for (const kind of ["cents", "formantCents", "degrees", "fallback"] as AxisKind[]) {
            for (const center of [0, 37.5, -1200, 600, 3.5, 8.5]) {
                for (const span of [1e-6, 100, 300, 1200, 2400, 10000]) {
                    const legacy = legacyMarks(kind, center, span);
                    const built = buildAxisMarkInstances({
                        kind,
                        view: { center, span },
                        heightPx: 400,
                        axisWidthPx: 56,
                        dpr: 2,
                        valueToY: makeValueToY(),
                    });
                    expect(built.length, `${kind} center=${center} span=${span}`).toBe(
                        legacy.length,
                    );
                    for (let i = 0; i < built.length; i += 1) {
                        expect(built[i].value).toBeCloseTo(legacy[i].v, 9);
                        expect(built[i].label).toBe(legacy[i].label);
                        expect(built[i].isStrong).toBe(legacy[i].strong);
                        expect(built[i].lineOnly).toBe(!legacy[i].hasLine);
                        compared += 1;
                    }
                }
            }
        }
        // 组合空间足够大，避免"看起来通过"其实是空循环
        expect(compared).toBeGreaterThan(10000);
    });
});
