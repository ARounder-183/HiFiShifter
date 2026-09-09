import { describe, expect, it } from "vitest";

import {
    averageSelectionValues,
    rightDragUpScale,
    scaleSelectionDeviation,
    smoothContextPadFrames,
    smoothSelectionValues,
    transformSelectionByRightDrag,
} from "./selectionTransforms";

describe("smoothSelectionValues（高斯化后的「平滑化」op）", () => {
    it("强度 0 → 恒等", () => {
        const v = [60, 61, 59, 62];
        expect(smoothSelectionValues(v, "pitch", 0)).toEqual(v);
    });

    it("强度越大抹得越平（单调）", () => {
        // 三角形抖动
        const wobble = [60, 62, 60, 58, 60, 62, 60, 58, 60];
        const weak = smoothSelectionValues(wobble, "loudness", 0.2, { framePeriodMs: 5 });
        const strong = smoothSelectionValues(wobble, "loudness", 0.9, { framePeriodMs: 5 });
        const peakDeviation = (arr: number[]) =>
            Math.max(...arr.map((v) => Math.abs(v - 60)));
        expect(peakDeviation(strong)).toBeLessThan(peakDeviation(weak));
        expect(peakDeviation(weak)).toBeLessThan(2);
    });

    it("pitch 哨兵帧（0）保持 0 且不参与均值", () => {
        const out = smoothSelectionValues([55, 65, 0, 65, 55], "pitch", 0.5);
        expect(out[2]).toBe(0);
        expect(Number.isFinite(out[1])).toBe(true);
        expect(out[1]).toBeGreaterThan(54);
        expect(out[1]).toBeLessThan(66);
    });

    it("提供上下文时选区边界无缝：线性曲线全程精确保持", () => {
        const line = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        // 强度取小值使卷积半径（5 帧）≤ 上下文(3) + trend 延拓覆盖，
        // 全窗都是精确的线性延拓 → 输出与输入逐帧一致
        const out = smoothSelectionValues(line, "loudness", 0.2, {
            framePeriodMs: 5,
            leftContext: [-3, -2, -1],
            rightContext: [10, 11, 12],
        });
        for (let i = 0; i < line.length; i += 1) {
            expect(out[i]).toBeCloseTo(line[i], 9);
        }
    });

    it("非 pitch 参数不做哨兵过滤", () => {
        const out = smoothSelectionValues([5, 0, 10, 0, 5], "loudness", 0.5);
        // 0 是合法值，参与均值 → 中点被摊低但不为 10
        expect(out[2]).toBeLessThan(10);
        expect(out[2]).toBeGreaterThan(0);
    });
});

describe("smoothContextPadFrames", () => {
    it("3σ 截断半径（毫秒定标）", () => {
        expect(smoothContextPadFrames(1, 5)).toBe(36); // ⌈3·60/5⌉
        expect(smoothContextPadFrames(0, 5)).toBe(0);
        expect(smoothContextPadFrames(1, 10)).toBe(18);
    });
});

describe("transformSelectionByRightDrag", () => {
    it("dy=0 恒等（逐帧精确相等）", () => {
        const v = [60, 59.7, 60.3, 60];
        expect(transformSelectionByRightDrag(v, "pitch", 0, { framePeriodMs: 5 })).toEqual(v);
    });

    it("rightDragUpScale 映射：+2%/px，无上限", () => {
        expect(rightDragUpScale(0)).toBe(1);
        expect(rightDragUpScale(25)).toBeCloseTo(1.5, 9);
        expect(rightDragUpScale(50)).toBe(2);
        expect(rightDragUpScale(100)).toBe(3);
    });

    it("上拖（音高）：颤音深度按 scale 加深，中心不变", () => {
        // ±0.3 semitone、周期 36 帧（180ms @5ms）、5 个周期
        const vib = new Array<number>(181);
        for (let i = 0; i < vib.length; i += 1) {
            vib[i] = 60 + 0.3 * Math.sin((2 * Math.PI * i) / 36);
        }
        const out = transformSelectionByRightDrag(vib, "pitch", 50, { framePeriodMs: 5 });
        const peak = Math.max(...out.map((x) => Math.abs(x - 60)));
        // σ_trend = 150ms 对 180ms 颤音泄漏 ≈ 0 → 残差完整放大 ×2
        expect(peak).toBeGreaterThan(0.55);
        expect(peak).toBeLessThan(0.65);
        const center = out.reduce((a, b) => a + b, 0) / out.length;
        expect(center).toBeCloseTo(60, 6);
    });

    it("上拖（音高）：纯滑音（趋势）完全保持 —— 均值中心旧法会斜率×2（P1 锁定）", () => {
        const glide = new Array<number>(301);
        for (let i = 0; i < glide.length; i += 1) {
            glide[i] = 40 + (8 * i) / (glide.length - 1);
        }
        const out = transformSelectionByRightDrag(glide, "pitch", 50, { framePeriodMs: 5 });
        for (let i = 0; i < glide.length; i += 1) {
            expect(out[i]).toBeCloseTo(glide[i], 6);
        }
        // 对照：旧均值中心法（scaleSelectionDeviation）会把斜率 ×2 —— 这正是
        // 换残差放大的原因（ characterizing 旧行为差异）。
        const legacy = scaleSelectionDeviation(glide, "pitch", 2);
        expect(legacy[0]).toBeCloseTo(36, 6);
        expect(legacy[legacy.length - 1]).toBeCloseTo(52, 6);
    });

    it("上拖（音高）：多音选区音程不变、平台保持、台阶过冲有界（P1 锁定）", () => {
        const step = [...new Array<number>(50).fill(60), ...new Array<number>(50).fill(64)];
        const out = transformSelectionByRightDrag(step, "pitch", 50, { framePeriodMs: 5 });
        // 远端平台保持原值（趋势精确、残差≈0）
        expect(out[5]).toBeCloseTo(60, 1);
        expect(out[95]).toBeCloseTo(64, 1);
        // 音程不放大：全场不越出原值域 ±0.6（旧均值中心法会变成 58↔66）
        expect(Math.min(...out)).toBeGreaterThan(59.4);
        expect(Math.max(...out)).toBeLessThan(64.6);
        // 对照：旧均值中心法把音程放大一倍
        const legacy = scaleSelectionDeviation(step, "pitch", 2);
        expect(Math.min(...legacy)).toBeCloseTo(58, 6);
        expect(Math.max(...legacy)).toBeCloseTo(66, 6);
    });

    it("上拖（音高）：极端 scale 下绝不写出 ≤0（哨兵防护）", () => {
        // ±20 半音的极端振荡，k=8：raw 残差放大远越 0 → 钉在哨兵下限
        const wild = new Array<number>(100);
        for (let i = 0; i < wild.length; i += 1) {
            wild[i] = i % 2 === 0 ? 60 : 20;
        }
        const out = transformSelectionByRightDrag(wild, "pitch", 350, { framePeriodMs: 5 });
        for (const v of out) {
            expect(v).toBeGreaterThan(0);
        }
        expect(Math.min(...out)).toBeLessThan(0.1);
    });

    it("上拖（音高）：scale 单调 → 细节能量单调不减", () => {
        const vib = new Array<number>(121);
        for (let i = 0; i < vib.length; i += 1) {
            vib[i] = 60 + 0.3 * Math.sin((2 * Math.PI * i) / 36);
        }
        const weak = transformSelectionByRightDrag(vib, "pitch", 25, { framePeriodMs: 5 });
        const strong = transformSelectionByRightDrag(vib, "pitch", 50, { framePeriodMs: 5 });
        const peak = (arr: number[]) => Math.max(...arr.map((x) => Math.abs(x - 60)));
        expect(peak(strong)).toBeGreaterThan(peak(weak));
    });

    it("上拖（非音高）：保持均值中心缩放（动态范围语义）", () => {
        // dy=25 → scale=1.5：mean=60 → 58→57, 62→63
        const out = transformSelectionByRightDrag([58, 60, 62], "loudness", 25);
        expect(out[0]).toBeCloseTo(57, 9);
        expect(out[1]).toBeCloseTo(60, 9);
        expect(out[2]).toBeCloseTo(63, 9);
    });

    it("下拖：平滑，且帧周期参与毫秒定标", () => {
        const wobble = [60, 62, 60, 58, 60];
        const out = transformSelectionByRightDrag(wobble, "pitch", -50, { framePeriodMs: 5 });
        const peakDeviation = Math.max(...out.map((x) => Math.abs(x - 60)));
        expect(peakDeviation).toBeLessThan(2);
        expect(Number.isFinite(peakDeviation)).toBe(true);
    });

    it("下拖超过 -50px 继续加深（不饱和），直至曲线展平", () => {
        // 周期 60 帧的正弦抖动（±2），3 个周期
        const wobble = new Array<number>(181);
        for (let i = 0; i < wobble.length; i += 1) {
            wobble[i] = 60 + 2 * Math.sin((2 * Math.PI * i) / 60);
        }
        const peak = (arr: number[]) => {
            let m = 0;
            for (let i = 30; i <= 150; i += 1) {
                m = Math.max(m, Math.abs(arr[i] - 60));
            }
            return m;
        };
        const at50 = transformSelectionByRightDrag(wobble, "pitch", -50, { framePeriodMs: 5 });
        const at75 = transformSelectionByRightDrag(wobble, "pitch", -75, { framePeriodMs: 5 });
        const at100 = transformSelectionByRightDrag(wobble, "pitch", -100, { framePeriodMs: 5 });
        // 严格单调加深：u=1 → σ=60ms 尚有残留纹波；u=2 → σ≈155ms 已基本展平
        expect(peak(at50)).toBeGreaterThan(0.5);
        expect(peak(at75)).toBeLessThan(peak(at50));
        expect(peak(at100)).toBeLessThan(peak(at75));
        expect(peak(at100)).toBeLessThan(0.2);
    });
});

describe("averageSelectionValues / scaleSelectionDeviation（既有行为锁定）", () => {
    it("均值拉拢受强度控制", () => {
        const out = averageSelectionValues([60, 62], "pitch", 100);
        expect(out[0]).toBeCloseTo(61, 9);
        expect(out[1]).toBeCloseTo(61, 9);
    });

    it("pitch 的 0 不参与均值也不被改动", () => {
        const out = averageSelectionValues([60, 0, 62], "pitch", 100);
        expect(out[1]).toBe(0);
        expect(out[0]).toBeCloseTo(61, 9);
        expect(out[2]).toBeCloseTo(61, 9);
    });

    it("偏差缩放围绕均值", () => {
        const out = scaleSelectionDeviation([58, 62], "pitch", 0.5);
        expect(out[0]).toBeCloseTo(59, 9);
        expect(out[1]).toBeCloseTo(61, 9);
    });
});
