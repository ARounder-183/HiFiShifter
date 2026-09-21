import { describe, expect, it } from "vitest";

import {
    clampParamWriteValue,
    computeDynGain,
    dynDragScaleFactor,
    shiftDynValueForDrag,
    shiftValueForDrag,
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
        // 无内容帧（低于静音下限）：分母钳到下限 ⇒ 增益**有界**，并按"无内容"
        // 平滑淡出（不是旧的固定 ×1000 —— 那会把抖动噪声底抬成可闻嘶声，
        // 也不是更早的"拒绝放大返回 1" —— 那会造成门限处阶跃 → 伪影）。
        const dither = 10 ** (-80 / 20); // 1e-4
        const gDither = computeDynGain(1, dither);
        expect(gDither).toBeLessThan(DYN_MAX_GAIN);
        expect(gDither).toBeGreaterThan(0);
        expect(computeDynGain(1, 0)).toBe(0); // 真静音：无内容 → 增益 0
        expect(computeDynGain(1, DYN_SILENCE_FLOOR)).toBeCloseTo(DYN_MAX_GAIN, 6); // 门限处不淡出
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
        // 下限之下：增益有界**且随原声平滑淡出**（无内容越彻底越小、无台阶）。
        expect(computeDynGain(0.0005, 0.0001)).toBeLessThan(computeDynGain(0.0005, 0.0005));
        // 连续性：原声轴上一阶上采样，相邻取值之差有界。
        let prev = computeDynGain(0.001, 0);
        for (let step = 1; step <= 200; step += 1) {
            const cur = computeDynGain(0.001, (DYN_SILENCE_FLOOR * step) / 200);
            expect(Math.abs(cur - prev)).toBeLessThan(0.01);
            prev = cur;
        }
        // 上限 = 从下限兑现到值域顶端；越界曲线才被它兜住。
        expect(DYN_MAX_GAIN).toBe(DYN_VALUE_MAX / DYN_SILENCE_FLOOR);
        expect(computeDynGain(1e9, DYN_SILENCE_FLOOR * 1.0001)).toBeLessThanOrEqual(DYN_MAX_GAIN);
    });

    it("非有限输入不产生 NaN", () => {
        expect(computeDynGain(Number.NaN, 1)).toBe(1);
        expect(computeDynGain(1, Number.POSITIVE_INFINITY)).toBe(1);
    });
});

describe("clampParamWriteValue（与后端写入分支同构）", () => {
    it("音量：钳到 0..2（负值一并钳到 0）", () => {
        expect(clampParamWriteValue("volume", 1.5)).toBe(1.5);
        expect(clampParamWriteValue("volume", 4)).toBe(2);
        expect(clampParamWriteValue("volume", -0.5)).toBe(0);
        expect(clampParamWriteValue("hifigan_volume", 9)).toBe(2);
    });

    it("动态：上界 1；负值收敛到「沿用原声」哨兵（不是 0）", () => {
        expect(clampParamWriteValue("dyn", 0.4)).toBe(0.4);
        // 选区上拖的乘性变换（orig × 2^Δ）能轻易越过 1 —— 这正是"波形超出"的来源。
        expect(clampParamWriteValue("dyn", 0.5 * dynMultiplicativeFactor(1))).toBe(1);
        expect(clampParamWriteValue("dyn", -0.0001)).toBe(DYN_FOLLOW_ORIG);
        expect(clampParamWriteValue("dyn", 0)).toBe(0);
        expect(clampParamWriteValue("dyn_edit", 5)).toBe(1);
    });

    it("音高：0 是「未设置」哨兵，绝不能被钳进 1..127", () => {
        expect(clampParamWriteValue("pitch", 0)).toBe(0);
        expect(clampParamWriteValue("pitch", 60)).toBe(60);
        expect(clampParamWriteValue("pitch", 200)).toBe(127);
        expect(clampParamWriteValue("pitch", 0.4)).toBe(1);
    });

    it("张力：钳到 ±100", () => {
        expect(clampParamWriteValue("tension", 250)).toBe(100);
        expect(clampParamWriteValue("tension", -250)).toBe(-100);
    });

    it("子轨共振峰偏移：钳到 ±2400 音分", () => {
        expect(clampParamWriteValue("child_formant_offset_cents@t1", 9999)).toBe(2400);
        expect(clampParamWriteValue("child_formant_offset_cents@t1", -9999)).toBe(-2400);
    });

    it("不在本组内的参数原样返回（后端也是 `_ => {}`）", () => {
        expect(clampParamWriteValue("breathiness", 42)).toBe(42);
        expect(clampParamWriteValue("child_pitch_offset_cents@t1", 1e9)).toBe(1e9);
    });

    it("非有限值原样透传（后端有独立的兜底语义）", () => {
        expect(Number.isNaN(clampParamWriteValue("volume", Number.NaN))).toBe(true);
        expect(clampParamWriteValue("volume", Number.POSITIVE_INFINITY)).toBe(
            Number.POSITIVE_INFINITY,
        );
    });

    it("★ 拖拽越界后钳制是幂等的（预览反复写入同一处的结果稳定）", () => {
        for (const param of ["volume", "dyn", "pitch", "tension"]) {
            const once = clampParamWriteValue(param, 1e6);
            expect(clampParamWriteValue(param, once)).toBe(once);
        }
    });
});

describe("shiftValueForDrag（线性偏移：非比值参数 / 动态锚点贴地时的退路）", () => {
    it("偏移量与指针位移一一对应（未触边界时）", () => {
        for (const delta of [0.1, 0.25, 0.5, -0.1, -0.4]) {
            for (const orig of [0.02, 0.05, 0.2, 0.5, 0.9]) {
                const result = shiftValueForDrag("dyn", orig, delta);
                // 未饱和时位移必须与指针位移严格一致；饱和（触到值域边界）时
                // 停在边界上，这也正是"跟手"该有的表现。
                if (result > 0 && result < 1) {
                    expect(result - orig).toBeCloseTo(delta, 10);
                } else {
                    expect(result === 0 || result === 1).toBe(true);
                }
            }
        }
    });

    it("★ 动态小值不再被压缩：0.05 上拖 0.5 得到 0.55（旧乘性法则只有 0.1）", () => {
        expect(shiftValueForDrag("dyn", 0.05, 0.5)).toBeCloseTo(0.55, 10);
        // 旧法则的对照值，钉住"不再回退到乘性"。
        expect(0.05 * dynMultiplicativeFactor(0.5)).toBeCloseTo(0.1, 10);
    });

    it("非比值参数与动态的线性退路完全一致", () => {
        // 同一份"原始值 + 位移"，两个参数在未触边界时给出同样的增量。
        for (const [orig, delta] of [
            [0.2, 0.3],
            [0.6, -0.25],
        ] as const) {
            const dynMoved = shiftValueForDrag("dyn", orig, delta) - orig;
            const volMoved = shiftValueForDrag("volume", orig, delta) - orig;
            expect(dynMoved).toBeCloseTo(volMoved, 10);
        }
    });

    it("触边界时钳住（不越过存储值域）", () => {
        expect(shiftValueForDrag("dyn", 0.9, 0.5)).toBe(1);
        expect(shiftValueForDrag("dyn", 0.05, -0.5)).toBe(0);
        expect(shiftValueForDrag("volume", 1.8, 0.5)).toBe(2);
        expect(shiftValueForDrag("volume", 0.1, -0.5)).toBe(0);
        expect(shiftValueForDrag("tension", 90, 50)).toBe(100);
    });

    it("线性偏移下静音（0）会被抬起来（动态只在锚点贴地时才走这条路径）", () => {
        expect(shiftValueForDrag("dyn", 0, 0.3)).toBeCloseTo(0.3, 10);
        // 但不会越过值域（0 往下拖仍是 0）。
        expect(shiftValueForDrag("dyn", 0, -0.3)).toBe(0);
    });

    it("音高的 0 哨兵与负哨兵都不被破坏", () => {
        // pitch：0 = 未设置。线性偏移会把它推成非 0（与手绘一致，且后端写入口
        // 的 `if v != 0.0` 分支同样允许非 0），但不会把 0 变成负数。
        expect(shiftValueForDrag("pitch", 60, 5)).toBe(65);
        expect(shiftValueForDrag("pitch", 60, -80)).toBe(1);
    });

    it("非有限输入不产生 NaN（原样返回 / 只钳制）", () => {
        expect(shiftValueForDrag("dyn", 0.5, Number.NaN)).toBe(0.5);
        expect(Number.isNaN(shiftValueForDrag("dyn", Number.NaN, 0.5))).toBe(true);
    });
});

describe("dynDragScaleFactor（幅度系数由被抓住那条线的位置导出）", () => {
    it("★ 系数令锚点恰好跟手：A·k = A + Δ", () => {
        for (const A of [0.02, 0.05, 0.2, 0.5, 0.9]) {
            for (const delta of [0.05, 0.2, -0.01, -0.5]) {
                const k = dynDragScaleFactor(A, delta);
                expect(k).not.toBeNull();
                // 锚点位移 = 指针位移（这就是"跟手"），未被下界截断时成立。
                const moved = A * (k as number) - A;
                if ((k as number) > 0) expect(moved).toBeCloseTo(delta, 10);
            }
        }
    });

    it("★ 值越小系数越大 —— 这正是旧固定灵敏度缺失的那一项", () => {
        // 旧实现 k = 2^(Δ/0.5) 与位置无关；锚点位移 ∝ A，值越小动得越少。
        const delta = 0.1;
        const kLow = dynDragScaleFactor(0.05, delta) as number;
        const kHigh = dynDragScaleFactor(0.5, delta) as number;
        expect(kLow).toBeGreaterThan(kHigh);
        // 位置带来的放大恰好抵消"乘性缩放按比例施加"的衰减。
        expect(0.05 * kLow - 0.05).toBeCloseTo(0.1, 10);
        expect(0.5 * kHigh - 0.5).toBeCloseTo(0.1, 10);
    });

    it("锚点到值域底部时系数取 0（整段缩到静音），不为负（不上下翻转）", () => {
        expect(dynDragScaleFactor(0.2, -0.2)).toBe(0);
        expect(dynDragScaleFactor(0.2, -5)).toBe(0);
    });

    it("锚点贴地（≤ 下限）返回 null —— 乘性对 0 无解，交给调用方退回线性", () => {
        expect(dynDragScaleFactor(0, 0.3)).toBeNull();
        expect(dynDragScaleFactor(0.0001, 0.3)).toBeNull();
        expect(dynDragScaleFactor(Number.NaN, 0.3)).toBeNull();
    });

    it("位移非有限时退化为恒等（不炸）", () => {
        expect(dynDragScaleFactor(0.5, Number.NaN)).toBe(1);
    });
});

describe("shiftDynValueForDrag（动态拖拽的完整法则）", () => {
    it("★ 锚点跟手 + 其余点按比例缩放（倍率域相对关系保留）", () => {
        const anchor = 0.05;
        const delta = 0.05; // ×2
        expect(shiftDynValueForDrag(0.05, anchor, delta)).toBeCloseTo(0.1, 10); // 锚点跟手
        expect(shiftDynValueForDrag(0.1, anchor, delta)).toBeCloseTo(0.2, 10);
        // 相对关系保留：0.05 : 0.10 = 0.10 : 0.20
        const lo = shiftDynValueForDrag(0.05, anchor, delta);
        const hi = shiftDynValueForDrag(0.1, anchor, delta);
        expect(hi / lo).toBeCloseTo(2, 10);
    });

    it("★ 静音帧保持静音（0 × k = 0）—— 乘性语义不被破坏", () => {
        expect(shiftDynValueForDrag(0, 0.05, 0.5)).toBe(0);
        expect(shiftDynValueForDrag(0, 0.5, -0.2)).toBe(0);
    });

    it("与用户报告的故障对齐：0.05 处上拖半屏，线到 0.55（旧固定灵敏度只到 0.1）", () => {
        expect(shiftDynValueForDrag(0.05, 0.05, 0.5)).toBeCloseTo(0.55, 10);
        expect(0.05 * dynMultiplicativeFactor(0.5)).toBeCloseTo(0.1, 10); // 旧对照值
    });

    it("结果钳在动态值域内（放大到顶不越界）", () => {
        expect(shiftDynValueForDrag(0.9, 0.2, 0.5)).toBe(1);
        expect(shiftDynValueForDrag(1, 0.2, 0.5)).toBe(1);
    });

    it("锚点贴地时退回线性偏移（手势仍然可用，跟手）", () => {
        // 抓住的是静音：乘性无解，退回线性 —— 锚点跟着光标走。
        expect(shiftDynValueForDrag(0, 0, 0.3)).toBeCloseTo(0.3, 10);
        expect(shiftDynValueForDrag(0.05, 0, 0.3)).toBeCloseTo(0.35, 10);
        // 该路径下负值收敛到 0（静音），不套用后端的"沿用原声"哨兵。
        expect(shiftDynValueForDrag(0.05, 0, -0.5)).toBe(0);
    });

    it("锚点拖到底：整段缩到静音（与「把锚点拖到 0」一致）", () => {
        expect(shiftDynValueForDrag(0.4, 0.5, -0.5)).toBe(0);
        expect(shiftDynValueForDrag(0.05, 0.5, -0.5)).toBe(0);
    });
});

describe("无内容噪声底不再被放大（用户场景：原声 ≈ −90 dB 处拉高动态）", () => {
    const DITHER = 10 ** (-90 / 20); // ≈ 3.16e-5，16bit 抖动量级

    it("★ 输出电平必须低于 −60 dBFS（不可闻）", () => {
        for (const target of [0.25, 0.5, 1]) {
            const gain = computeDynGain(target, DITHER);
            const outDb = 20 * Math.log10(Math.max(DITHER * gain, 1e-12));
            expect(outDb).toBeLessThan(-60);
        }
    });

    it("有内容处不受影响：−40 dBFS 的轻声照常兑现目标", () => {
        const quiet = 10 ** (-40 / 20);
        expect(computeDynGain(0.5, quiet)).toBeCloseTo(0.5 / quiet, 6);
        // 下限之上未画帧恒为 1（未编辑区逐像素不变的前提）。
        for (const db of [-60, -50, -30, -6]) {
            const orig = 10 ** (db / 20);
            expect(computeDynGain(orig, orig)).toBeCloseTo(1, 9);
        }
    });
});
