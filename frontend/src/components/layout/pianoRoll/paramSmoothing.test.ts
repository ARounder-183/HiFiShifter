import { describe, expect, it } from "vitest";

import {
    applyEdgeBlend,
    drawSmoothSigmaMsFromStrength,
    edgeBlendWeight,
    edgeHalfSpanFramesForSelection,
    editablePitchValue,
    smoothCurveGaussian,
    smoothSigmaMsFromUnits,
    strengthToHalfSpanFrames,
} from "./paramSmoothing";

function flatBase(n: number, value = 60): number[] {
    return new Array<number>(n).fill(value);
}

function expectCloseTo(actual: number[], expected: number[], precision = 9): void {
    expect(actual.length).toBe(expected.length);
    for (let i = 0; i < expected.length; i += 1) {
        expect(actual[i]).toBeCloseTo(expected[i], precision);
    }
}

describe("edgeBlendWeight 权重剖面", () => {
    it("三种形状端点与中点一致（u=0/0.5/1）", () => {
        for (const shape of ["linear", "smoothstep", "quintic"] as const) {
            expect(edgeBlendWeight(shape, 0)).toBe(0);
            expect(edgeBlendWeight(shape, 0.5)).toBeCloseTo(0.5, 12);
            expect(edgeBlendWeight(shape, 1)).toBe(1);
        }
    });

    it("smoothstep 中段取值与单调性", () => {
        expect(edgeBlendWeight("smoothstep", 0.25)).toBeCloseTo(0.15625, 12);
        expect(edgeBlendWeight("smoothstep", 0.75)).toBeCloseTo(0.84375, 12);
        let prev = -1;
        for (let u = 0; u <= 1.001; u += 0.05) {
            const w = edgeBlendWeight("smoothstep", u);
            expect(w).toBeGreaterThanOrEqual(prev);
            prev = w;
        }
    });

    it("u 超界被钳制", () => {
        expect(edgeBlendWeight("smoothstep", -0.7)).toBe(0);
        expect(edgeBlendWeight("smoothstep", 1.9)).toBe(1);
    });
});

describe("strengthToHalfSpanFrames 强度→帧数映射（毫秒定标）", () => {
    it("fp=5ms 时的刻度", () => {
        expect(strengthToHalfSpanFrames(100, 5, 1000)).toBe(12);
        expect(strengthToHalfSpanFrames(50, 5, 1000)).toBe(6);
        expect(strengthToHalfSpanFrames(25, 5, 1000)).toBe(3);
    });

    it("毫秒不变性：fp 翻倍 → 帧数减半", () => {
        expect(strengthToHalfSpanFrames(100, 10, 1000)).toBe(6);
        expect(strengthToHalfSpanFrames(100, 20, 1000)).toBe(3);
    });

    it("上限钳制与零强度", () => {
        expect(strengthToHalfSpanFrames(100, 5, 5)).toBe(5);
        expect(strengthToHalfSpanFrames(100, 5, 0)).toBe(0);
        expect(strengthToHalfSpanFrames(0, 5, 1000)).toBe(0);
        expect(strengthToHalfSpanFrames(-5, 5, 1000)).toBe(0);
    });
});

describe("edgeHalfSpanFramesForSelection 选区半宽", () => {
    it("时间上限优先，选区长度上限兜底", () => {
        // 100 帧 @5ms：时间上限 12 < 长度上限 25 → 12
        expect(
            edgeHalfSpanFramesForSelection({
                strengthPercent: 100,
                framePeriodMs: 5,
                editedLen: 100,
            }),
        ).toBe(12);
        // 20 帧：长度上限 floor(20/4)=5 → 5
        expect(
            edgeHalfSpanFramesForSelection({
                strengthPercent: 100,
                framePeriodMs: 5,
                editedLen: 20,
            }),
        ).toBe(5);
        // 8 帧：长度上限 2 → 2（过渡带不越过选区中点）
        expect(
            edgeHalfSpanFramesForSelection({
                strengthPercent: 100,
                framePeriodMs: 5,
                editedLen: 8,
            }),
        ).toBe(2);
    });

    it("零强度 / 空选区 → 0", () => {
        expect(
            edgeHalfSpanFramesForSelection({
                strengthPercent: 0,
                framePeriodMs: 5,
                editedLen: 100,
            }),
        ).toBe(0);
        expect(
            edgeHalfSpanFramesForSelection({ strengthPercent: 50, framePeriodMs: 5, editedLen: 0 }),
        ).toBe(0);
    });
});

describe("applyEdgeBlend：delta 空间边缘交叉淡化", () => {
    // 平直曲线 60，选区 [10,30) 编辑为 62（delta=+2），half=5（floor(20/4)）
    const base = flatBase(50);
    const editedDense = base.slice();
    for (let i = 10; i < 30; i += 1) editedDense[i] = 62;
    const HALF = 5;

    function run(): number[] {
        const dense = editedDense.slice();
        applyEdgeBlend({ dense, base, editedStartIdx: 10, editedLen: 20, halfSpanFrames: HALF });
        return dense;
    }

    it("左边界带：单调过渡，边界帧折半，带外不动", () => {
        const out = run();
        expect(out[5]).toBe(60); // u=0 → 不动
        expect(out[6]).toBeCloseTo(60 + 0.028 * 2, 9);
        expect(out[7]).toBeCloseTo(60 + 0.104 * 2, 9);
        expect(out[8]).toBeCloseTo(60 + 0.216 * 2, 9);
        expect(out[9]).toBeCloseTo(60 + 0.352 * 2, 9);
        expect(out[10]).toBeCloseTo(61, 9); // 边界帧 w=0.5
        expect(out[11]).toBeCloseTo(60 + 0.648 * 2, 9);
        expect(out[14]).toBeCloseTo(60 + 0.972 * 2, 9);
        expect(out[15]).toBe(62); // u=1 → 完整编辑值
        expect(out[4]).toBe(60); // 带外原值
        expect(out[0]).toBe(60);
    });

    it("右边界带对称；带间内部保持完整编辑值", () => {
        const out = run();
        expect(out[29]).toBeCloseTo(61, 9);
        expect(out[30]).toBeCloseTo(60 + 0.352 * 2, 9);
        expect(out[31]).toBeCloseTo(60 + 0.216 * 2, 9);
        expect(out[33]).toBeCloseTo(60 + 0.028 * 2, 9);
        expect(out[34]).toBe(60);
        expect(out[35]).toBe(60);
        for (let f = 16; f <= 23; f += 1) {
            expect(out[f]).toBe(62);
        }
        for (let f = 25; f <= 28; f += 1) {
            expect(out[f]).toBeGreaterThan(60);
            expect(out[f]).toBeLessThan(62);
        }
        expect(out[24]).toBe(62); // 右带 u=1 → 完整编辑值
        expect(out[29]).toBeCloseTo(61, 9);
    });

    it("可组合性：三次 +2 与一次 +6 在全场完全一致", () => {
        let cur = base;
        for (let k = 0; k < 3; k += 1) {
            const dense = cur.slice();
            for (let i = 10; i < 30; i += 1) dense[i] += 2;
            applyEdgeBlend({
                dense,
                base: cur,
                editedStartIdx: 10,
                editedLen: 20,
                halfSpanFrames: HALF,
            });
            cur = dense;
        }
        const once = base.slice();
        for (let i = 10; i < 30; i += 1) once[i] += 6;
        applyEdgeBlend({
            dense: once,
            base,
            editedStartIdx: 10,
            editedLen: 20,
            halfSpanFrames: HALF,
        });
        expectCloseTo(cur, once);
    });

    it("P0：选区外未浊帧（pitch=0）绝不被写成非零", () => {
        const base0 = flatBase(50);
        for (let i = 5; i < 10; i += 1) base0[i] = 0; // 左边界外的未浊段
        for (let i = 30; i < 34; i += 1) base0[i] = 0; // 右边界外的未浊段
        const dense = base0.slice();
        for (let i = 10; i < 30; i += 1) dense[i] = 62;
        applyEdgeBlend({
            dense,
            base: base0,
            editedStartIdx: 10,
            editedLen: 20,
            halfSpanFrames: HALF,
            isEditable: editablePitchValue,
        });
        for (let i = 5; i < 10; i += 1) expect(dense[i]).toBe(0);
        for (let i = 30; i < 34; i += 1) expect(dense[i]).toBe(0);
        // 有声帧的淡化照常工作
        expect(dense[10]).toBeCloseTo(61, 9);
        expect(dense[4]).toBe(60);
    });

    it("斜坡素材：选区外侧保持原斜率，无 kink/凹陷", () => {
        const slope = new Array<number>(60);
        for (let f = 0; f < 60; f += 1) slope[f] = 55 + 0.5 * f;
        const dense = slope.slice();
        for (let i = 20; i < 40; i += 1) dense[i] += 2;
        applyEdgeBlend({
            dense,
            base: slope,
            editedStartIdx: 20,
            editedLen: 20,
            halfSpanFrames: 5,
        });
        // 选区外左侧（[15,20)）输出严格递增且不低于原值（delta>0 时）
        for (let f = 15; f < 20; f += 1) {
            expect(dense[f]).toBeGreaterThanOrEqual(slope[f] - 1e-9);
            expect(dense[f + 1]).toBeGreaterThan(dense[f]);
        }
        // 选区外右侧（[40,44]）：附加增量严格递减且非负（总值仍随斜坡上升）
        let prevDelta = Number.POSITIVE_INFINITY;
        for (let f = 40; f <= 44; f += 1) {
            const deltaAdd = dense[f] - slope[f];
            expect(deltaAdd).toBeGreaterThanOrEqual(-1e-9);
            expect(deltaAdd).toBeLessThan(prevDelta);
            prevDelta = deltaAdd;
        }
        // 带外完全不动
        expect(dense[10]).toBeCloseTo(slope[10], 9);
        expect(dense[50]).toBeCloseTo(slope[50], 9);
    });

    it("editedAt（setPitch 语义）：选区外向目标滑移，未浊帧保持 0", () => {
        const base0 = flatBase(30);
        for (let i = 5; i < 10; i += 1) base0[i] = 0;
        const dense = base0.slice();
        for (let i = 10; i < 20; i += 1) dense[i] = 65;
        applyEdgeBlend({
            dense,
            base: base0,
            editedStartIdx: 10,
            editedLen: 10,
            halfSpanFrames: 2, // floor(10/4)=2
            editedAt: () => 65,
            isEditable: editablePitchValue,
        });
        for (let i = 5; i < 10; i += 1) expect(dense[i]).toBe(0);
        // 左带 [8,12]：u=(f−8)/4
        expect(dense[10]).toBeCloseTo(60 + 0.5 * 5, 9); // u=0.5 → 62.5
        expect(dense[11]).toBeCloseTo(60 + edgeBlendWeight("smoothstep", 0.75) * 5, 9);
        expect(dense[12]).toBeCloseTo(65, 9); // u=1 → 完整目标值
        // 右带 [17,21]：u=(21−f)/4，选区外向目标渐出
        expect(dense[19]).toBeCloseTo(60 + 0.5 * 5, 9);
        expect(dense[20]).toBeCloseTo(60 + edgeBlendWeight("smoothstep", 0.25) * 5, 9);
        expect(dense[21]).toBeCloseTo(60, 9); // u=0 → 不动
    });

    it("halfSpan=0 时完全不动", () => {
        const dense = editedDense.slice();
        applyEdgeBlend({ dense, base, editedStartIdx: 10, editedLen: 20, halfSpanFrames: 0 });
        expect(dense).toEqual(editedDense);
    });
});

describe("smoothCurveGaussian 高斯平滑", () => {
    const FP = 5;

    it("常量曲线不变；σ 过小为恒等", () => {
        const c = flatBase(40, 60);
        expect(smoothCurveGaussian(c, { sigmaMs: 5, framePeriodMs: FP })).toEqual(c);
        expect(smoothCurveGaussian(c, { sigmaMs: 2, framePeriodMs: FP })).toEqual(c); // σf=0.4 < 0.5
    });

    it("阶跃：对称、单调、无过冲、远处不变", () => {
        const step = [...new Array<number>(20).fill(0), ...new Array<number>(20).fill(1)];
        const out = smoothCurveGaussian(step, {
            sigmaMs: 5,
            framePeriodMs: FP,
            medianPrepass: false,
        });
        for (const v of out) {
            expect(v).toBeGreaterThanOrEqual(-1e-12);
            expect(v).toBeLessThanOrEqual(1 + 1e-12);
        }
        expect(out[10]).toBeCloseTo(0, 9);
        expect(out[29]).toBeCloseTo(1, 9);
        expect(out[19]).toBeCloseTo(0.3004, 3);
        expect(out[20]).toBeCloseTo(0.6996, 3);
        // 关于 19/20 中点对称
        expect(out[19]).toBeCloseTo(1 - out[20], 9);
        expect(out[18]).toBeCloseTo(1 - out[21], 9);
    });

    it("孤立有效样本保持自身（重归一语义）", () => {
        const out = smoothCurveGaussian([0, 5, 0], {
            sigmaMs: 5,
            framePeriodMs: FP,
            valueFilter: (v) => v !== 0,
        });
        expect(out[1]).toBe(5);
        expect(out[0]).toBe(0);
        expect(out[2]).toBe(0);
    });

    it("哨兵帧不参与均值也不被改写", () => {
        const vals = [58, 62, 0, 58, 62];
        const out = smoothCurveGaussian(vals, {
            sigmaMs: 5,
            framePeriodMs: FP,
            valueFilter: (v) => v !== 0,
        });
        expect(out[2]).toBe(0);
        // 若 0 参与均值，out[1] 会被拖向 ~47；trend 延拓下保持在有效值邻域
        expect(out[1]).toBeGreaterThan(55);
        expect(out[1]).toBeLessThan(62);
        expect(Number.isFinite(out[1])).toBe(true);
    });

    it("短未浊缺口桥接，长缺口硬切分段", () => {
        // 短缺口（2 ≤ bridge=10）：out[4] 被右侧 65 拉高
        const shortGap = [...new Array<number>(5).fill(55), 0, 0, ...new Array<number>(5).fill(65)];
        const outShort = smoothCurveGaussian(shortGap, {
            sigmaMs: 5,
            framePeriodMs: FP,
            valueFilter: (v) => v !== 0,
        });
        expect(outShort[4]).toBeGreaterThan(55.01);
        expect(outShort[4]).toBeLessThan(65);
        // 长缺口（20 > bridge=10）：两侧各自精确保持
        const longGap = [
            ...new Array<number>(5).fill(55),
            ...new Array<number>(20).fill(0),
            ...new Array<number>(5).fill(65),
        ];
        const outLong = smoothCurveGaussian(longGap, {
            sigmaMs: 5,
            framePeriodMs: FP,
            valueFilter: (v) => v !== 0,
        });
        expect(outLong[4]).toBeCloseTo(55, 9);
        expect(outLong[25]).toBeCloseTo(65, 9);
        for (let i = 5; i < 25; i += 1) expect(outLong[i]).toBe(0);
    });

    it("中值预清去除孤立毛刺", () => {
        const spike = [60, 60, 90, 60, 60];
        const withPrepass = smoothCurveGaussian(spike, {
            sigmaMs: 5,
            framePeriodMs: FP,
            medianPrepass: true,
        });
        expect(withPrepass[2]).toBeCloseTo(60, 9);
        const noPrepass = smoothCurveGaussian(spike, {
            sigmaMs: 5,
            framePeriodMs: FP,
            medianPrepass: false,
        });
        expect(noPrepass[2]).toBeGreaterThan(65);
    });

    it("trend 边界：线性曲线全程精确保持（端点不内拉）", () => {
        const line = new Array<number>(10);
        for (let f = 0; f < 10; f += 1) line[f] = f;
        const out = smoothCurveGaussian(line, { sigmaMs: 5, framePeriodMs: FP });
        for (let f = 0; f < 10; f += 1) expect(out[f]).toBeCloseTo(f, 9);
    });

    it("hold 边界：端点被拉向内部（与 trend 对照）", () => {
        const line = new Array<number>(10);
        for (let f = 0; f < 10; f += 1) line[f] = f;
        const out = smoothCurveGaussian(line, { sigmaMs: 5, framePeriodMs: FP, boundary: "hold" });
        expect(out[0]).toBeGreaterThan(0.2);
        expect(out[0]).toBeLessThan(1);
    });

    it("真实上下文优先于 trend 延拓", () => {
        // 倍增序列 [1,2,4,8]，左侧真实延拓 0.5/0.25/0.125
        const exp2 = [1, 2, 4, 8];
        const withCtx = smoothCurveGaussian(exp2, {
            sigmaMs: 5,
            framePeriodMs: FP,
            leftContext: [0.125, 0.25, 0.5],
        });
        // 真实几何延拓参与均值 → 端点高于 trend（trend 恰为 1.0）
        const withTrend = smoothCurveGaussian(exp2, { sigmaMs: 5, framePeriodMs: FP });
        expect(withTrend[0]).toBeCloseTo(1, 9);
        expect(withCtx[0]).toBeCloseTo(1.2697, 3);
        expect(withCtx[0]).toBeGreaterThan(withTrend[0]);
    });
});

describe("强度 → σ 映射", () => {
    it("平滑化 op：u=0→0，u=1→60ms，单调", () => {
        expect(smoothSigmaMsFromUnits(0)).toBe(0);
        expect(smoothSigmaMsFromUnits(1)).toBeCloseTo(60, 9);
        const half = smoothSigmaMsFromUnits(0.5);
        expect(half).toBeGreaterThan(smoothSigmaMsFromUnits(0.3));
        expect(half).toBeLessThan(smoothSigmaMsFromUnits(0.7));
    });

    it("右键下拖 u>1：沿同一幂曲线继续加深（C¹，不回退到钳制）", () => {
        // 2^1.4 ≈ 2.639 → σ = 2 + 58·2.639 ≈ 155ms
        expect(smoothSigmaMsFromUnits(2)).toBeGreaterThan(smoothSigmaMsFromUnits(1));
        expect(smoothSigmaMsFromUnits(2)).toBeLessThan(160);
        expect(smoothSigmaMsFromUnits(3)).toBeGreaterThan(smoothSigmaMsFromUnits(2));
        expect(smoothSigmaMsFromUnits(4.5)).toBeGreaterThan(smoothSigmaMsFromUnits(3));
    });

    it("σ 硬上限兜底（数值/性能），百分比显示不受影响", () => {
        expect(smoothSigmaMsFromUnits(10)).toBe(500);
        expect(smoothSigmaMsFromUnits(1000)).toBe(500);
    });

    it("绘制后平滑：线性映射到 40ms", () => {
        expect(drawSmoothSigmaMsFromStrength(0)).toBe(0);
        expect(drawSmoothSigmaMsFromStrength(100)).toBeCloseTo(40, 9);
        expect(drawSmoothSigmaMsFromStrength(50)).toBeCloseTo(20, 9);
    });
});

describe("editablePitchValue 哨兵判定", () => {
    it("0 与 NaN 不可编辑，其余可编辑", () => {
        expect(editablePitchValue(0)).toBe(false);
        expect(editablePitchValue(Number.NaN)).toBe(false);
        expect(editablePitchValue(60)).toBe(true);
        expect(editablePitchValue(-1)).toBe(true);
    });
});
