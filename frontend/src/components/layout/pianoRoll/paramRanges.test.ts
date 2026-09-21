import { describe, expect, it } from "vitest";

import {
    computeDynGain,
    DYN_DEFAULT_VIEW,
    DYN_FOLLOW_ORIG,
    DYN_MAX_GAIN,
    DYN_VALUE_MAX,
    DYN_VALUE_MIN,
    DYN_SILENCE_FLOOR,
    dynMultiplicativeFactor,
    isDynParam,
    restoreDynSentinels,
    VOLUME_DEFAULT_VIEW,
} from "./paramRanges";

describe("isDynParam", () => {
    it("识别 dyn 与历史别名 dyn_edit", () => {
        expect(isDynParam("dyn")).toBe(true);
        expect(isDynParam("dyn_edit")).toBe(true);
        expect(isDynParam("volume")).toBe(false);
        expect(isDynParam("pitch")).toBe(false);
        expect(isDynParam(null)).toBe(false);
        expect(isDynParam(undefined)).toBe(false);
    });
});

describe("restoreDynSentinels", () => {
    it("把位图标记的未画帧恢复成哨兵（写回时不得物化基线）", () => {
        // 帧 0/2 是用户画的，帧 1/3 未画（载荷里已解析成基线值 0.7）。
        const values = [0.8, 0.7, 0.5, 0.7];
        restoreDynSentinels(values, [false, true, false, true]);
        expect(values[0]).toBe(0.8);
        expect(values[1]).toBe(DYN_FOLLOW_ORIG);
        expect(values[2]).toBe(0.5);
        expect(values[3]).toBe(DYN_FOLLOW_ORIG);
    });

    it("无位图时原样返回（非 dyn / 旧载荷兼容）", () => {
        const values = [0.8, 0.7];
        expect(restoreDynSentinels(values, undefined)).toBe(values);
        expect(restoreDynSentinels(values, [])).toBe(values);
    });

    it("位图长度短于值数组时只处理重叠区间", () => {
        const values = [0.8, 0.7, 0.6];
        restoreDynSentinels(values, [true]);
        expect(values[0]).toBe(DYN_FOLLOW_ORIG);
        expect(values[1]).toBe(0.7);
        expect(values[2]).toBe(0.6);
    });
});

describe("dyn 值域与默认视口", () => {
    it("值域 0..1（≤0 dBFS，>1 会削顶）；默认视口铺满 0..1", () => {
        expect(DYN_VALUE_MIN).toBe(0);
        expect(DYN_VALUE_MAX).toBe(1);
        expect(DYN_DEFAULT_VIEW).toEqual({ center: 0.5, span: 1.0 });
        // 默认视口恰好覆盖整个值域：不必缩放即可看到全部可编辑区间，
        // 顶部不再为"超过满量程"预留无效高度。
        const low = DYN_DEFAULT_VIEW.center - DYN_DEFAULT_VIEW.span / 2;
        const high = DYN_DEFAULT_VIEW.center + DYN_DEFAULT_VIEW.span / 2;
        expect(low).toBe(DYN_VALUE_MIN);
        expect(high).toBe(DYN_VALUE_MAX);
        // volume 保持 0..2 视口（>1 的提升是常态）。
        expect(VOLUME_DEFAULT_VIEW).toEqual({ center: 1.0, span: 2.0 });
    });
});

describe("dynMultiplicativeFactor（乘性拖拽核心）", () => {
    it("0.5 个值单位 = ×2（+6 dB）；负向 = ×0.5", () => {
        expect(dynMultiplicativeFactor(0.5)).toBeCloseTo(2, 9);
        expect(dynMultiplicativeFactor(-0.5)).toBeCloseTo(0.5, 9);
        expect(dynMultiplicativeFactor(0)).toBe(1);
    });

    it("★ 乘性语义：静音（0）× 任何系数仍为 0，0.2 → ×2 = 0.4", () => {
        // 用户核心诉求：拖拽不能把无声拖出响度。
        const factor = dynMultiplicativeFactor(0.5); // ×2
        const curve = [0, 0.2, 0.5];
        const dragged = curve.map((v) => v * factor);
        expect(dragged[0]).toBe(0);
        expect(dragged[1]).toBeCloseTo(0.4, 9);
        expect(dragged[2]).toBeCloseTo(1.0, 9);
    });

    it("非有限输入回退恒等", () => {
        expect(dynMultiplicativeFactor(Number.NaN)).toBe(1);
        expect(dynMultiplicativeFactor(Number.POSITIVE_INFINITY)).toBe(1);
    });
});

/**
 * 动态增益公式（前端唯一实现，与后端 `compute_dyn_gain` 逐分支同构）。
 *
 * 【为什么单独测】它同时被波形预览与 live 编辑路径使用；一旦与后端分叉，
 * 用户就会看到"波形能提升、实际播放不提升"。核心回归是**安静内容必须可提升**。
 */
describe("computeDynGain", () => {
    it("★ 真实素材的安静段必须精确兑现目标（门限曾是 −26 dBFS、上限曾仅 ×4）", () => {
        const target = 0.582; // −4.7 dBFS，用户实测的典型目标
        for (const db of [-20, -26, -34, -40, -45, -55, -58]) {
            const orig = Math.pow(10, db / 20);
            const gain = computeDynGain(target, orig);
            // 必须精确兑现，而不是"能提升一点就行"—— dyn 是绝对目标电平。
            expect(gain).toBeCloseTo(target / orig, 3);
        }
    });

    it("哨兵 / 画静音 / 无内容各分支", () => {
        expect(computeDynGain(DYN_FOLLOW_ORIG, 0.5)).toBe(1); // 哨兵：沿用原声
        expect(computeDynGain(0, 0)).toBe(0); // 真静音 + 画静音 = 静音
        expect(computeDynGain(0, 0.02)).toBe(0); // 画 0 必须真静音
        // 无内容帧（−80 dBFS）的提升：分母钳到下限 ⇒ 有界到上限（不无限放大），
        // 而不是旧的"拒绝放大返回 1"（那会造成门限处阶跃 → 伪影）。
        expect(computeDynGain(1, 0.0001)).toBe(DYN_MAX_GAIN);
        expect(computeDynGain(1, 0)).toBe(DYN_MAX_GAIN);
    });

    it("★ 增益关于原声连续（近零伪影的根因）", () => {
        // 跨下限密集采样：相邻增益的相对变化必须极小。旧实现（低于下限拒绝放大）
        // 会在下限处产生 1 → 上限 的阶跃，使近零段的波形列高随机跳变。
        let prev: number | null = null;
        let maxRatio = 0;
        for (let k = 0; k <= 2000; k += 1) {
            const orig = DYN_SILENCE_FLOOR * 0.5 * (1 + k / 1000);
            const gain = computeDynGain(1, orig);
            if (prev !== null && prev > 0 && gain > 0) {
                maxRatio = Math.max(maxRatio, Math.max(gain / prev, prev / gain));
            }
            prev = gain;
        }
        expect(maxRatio).toBeLessThan(1.002);
    });

    it("衰减照常生效；上限只是数值兜底", () => {
        expect(computeDynGain(0.05, 0.1)).toBeCloseTo(0.5, 9);
        // 下限之内衰减到一半。
        expect(computeDynGain(0.002, 0.004)).toBeCloseTo(0.5, 9);
        expect(computeDynGain(0, 0.0005)).toBe(0); // 画静音
        // 下限之下分母恒被钳到下限 ⇒ 增益只由目标决定（无台阶）。
        expect(computeDynGain(0.0005, 0.0005)).toBe(computeDynGain(0.0005, 0.0001));
        // 上限 = 从下限兑现到值域顶端；越界曲线才被它兜住。
        expect(DYN_MAX_GAIN).toBe(DYN_VALUE_MAX / DYN_SILENCE_FLOOR);
        expect(computeDynGain(1e9, DYN_SILENCE_FLOOR * 1.0001)).toBeLessThanOrEqual(DYN_MAX_GAIN);
    });

    it("非有限输入不产生 NaN", () => {
        expect(computeDynGain(Number.NaN, 1)).toBe(1);
        expect(computeDynGain(1, Number.POSITIVE_INFINITY)).toBe(1);
    });
});
