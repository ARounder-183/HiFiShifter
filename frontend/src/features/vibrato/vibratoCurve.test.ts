import { describe, expect, test } from "vitest";

import { buildVibratoCurve, estimateCycles, type VibratoRenderInput } from "./vibratoCurve";
import { sanitizeVibratoPreset } from "./vibratoPresets";
import type { VibratoPreset, VibratoPresetInput } from "./vibratoTypes";

const FP = 5;

function preset(overrides: VibratoPresetInput): VibratoPreset {
    return sanitizeVibratoPreset({ id: "custom_test", ...overrides });
}

/** 一段完全"平"的预设：无渐入渐出、无渐强、无不规则度、恒定速率。 */
function steady(overrides: VibratoPresetInput = {}): VibratoPreset {
    return preset({
        attackMs: 0,
        releaseMs: 0,
        depthRamp: { start: 1, end: 1 },
        irregularity: 0,
        rateRampEnd: 1,
        ...overrides,
    });
}

/** 以 `seconds` 时长、恒定基线的条件渲染一段颤音。 */
function render(
    p: VibratoPreset,
    seconds: number,
    extra: Partial<VibratoRenderInput> = {},
): number[] {
    const frames = Math.round((seconds * 1000) / FP) + 1;
    return buildVibratoCurve({
        startFrame: 0,
        startValue: 60,
        endFrame: frames - 1,
        endValue: 60,
        preset: p,
        param: extra.param ?? "pitch",
        framePeriodMs: FP,
        ...extra,
    }).dense;
}

/**
 * 与 `render` 同一条件，但取深度包络（cents，恒非负）。
 *
 * 包络是"渐入 / 渐出把深度压到多少"的直接读数：断言它比断言波形本身清楚得多
 * （波形值还叠着当前相位，同一帧可能恰好落在过零点上）。
 */
function renderEnvelope(p: VibratoPreset, seconds: number): number[] {
    const frames = Math.round((seconds * 1000) / FP) + 1;
    return (
        buildVibratoCurve({
            startFrame: 0,
            startValue: 60,
            endFrame: frames - 1,
            endValue: 60,
            preset: p,
            param: "pitch",
            framePeriodMs: FP,
            collectEnvelope: true,
        }).envelope ?? []
    );
}

/**
 * 统计向上穿越基线的次数（≈ 周期数）。
 *
 * 用严格 `<` 作为"前值"条件：首帧的相位为 0，波形恰好等于基线，若用
 * `<=` 会把"首帧 → 次帧"误算成一次穿越，所有计数整体多 1。
 */
function countUpCrossings(values: readonly number[], base: number): number {
    let count = 0;
    for (let i = 1; i < values.length; i += 1) {
        if (values[i - 1] < base && values[i] >= base) count += 1;
    }
    return count;
}

describe("速率以 Hz 计，与选区长度无关", () => {
    /*
     * 【这条测试锁的是历史 bug】旧实现把 `t` 当作 0..1 的归一化进度，
     * `sin(2π·f·t)` 里的 `f` 因此是"整段几个周期"——同一段颤音拖得越长
     * 听感越慢。三个不同长度必须给出同一个 Hz。
     */
    test("同一速率在不同时长下解析出的 Hz 一致", () => {
        const p = steady({ rateHz: 5 });
        for (const seconds of [0.4, 1.0, 2.0]) {
            const values = render(p, seconds);
            const rate = countUpCrossings(values, 60) / seconds;
            expect(
                Math.abs(rate - 5),
                `${seconds}s 段解析出 ${rate.toFixed(2)} Hz，期望约 5 Hz`,
            ).toBeLessThan(0.75);
        }
    });

    test("周期个数随时长线性增长（而不是恒定）", () => {
        const p = steady({ rateHz: 4 });
        const short = countUpCrossings(render(p, 1.0), 60);
        const long = countUpCrossings(render(p, 2.0), 60);
        // 两倍时长 → 大致两倍周期数。
        expect(Math.abs(long - short * 2)).toBeLessThanOrEqual(1);
    });

    test("按周期数模式：整段恒为 N 个周期，与时长无关", () => {
        /*
         * 末帧恰好落在第 N 个过零点上，浮点累积误差决定它算不算"已跨越"，
         * 因此计数允许 ±1 的边界歧义。真正要锁的是：**周期数与时长无关**
         * ——这正是与频率模式的本质区别（后者随时长增长）。
         */
        const byCycle = steady({ rateMode: "cycles", cycles: 6 });
        for (const seconds of [0.5, 1.5]) {
            const count = countUpCrossings(render(byCycle, seconds), 60);
            expect(Math.abs(count - 6), `${seconds}s 段得到 ${count} 个周期`).toBeLessThanOrEqual(
                1,
            );
        }

        // 对照组：频率模式下周期数随时长增长。
        const byRate = steady({ rateHz: 4 });
        expect(countUpCrossings(render(byRate, 1.5), 60)).toBeGreaterThan(
            countUpCrossings(render(byRate, 0.5), 60),
        );
    });
});

describe("速率渐变按积分累积相位", () => {
    /*
     * 【为什么测周期数而不是测相位】若把变速率写成 `sin(2π·f(t)·t)`，
     * 相位会有跳变、周期数也对不上积分值。周期数等于速率的积分是最
     * 直接、也最难"蒙对"的判据。
     */
    test("末端 2× 速率时，周期数等于平均速率的积分", () => {
        const p = steady({ rateHz: 5, rateRampEnd: 2 });
        const seconds = 1;
        const values = render(p, seconds);
        // 平均速率 = 5 × (1 + 2) / 2 = 7.5 Hz。
        const count = countUpCrossings(values, 60);
        expect(Math.abs(count - 7.5)).toBeLessThanOrEqual(1);
    });

    test("相位连续：相邻帧的跳变不超过瞬时速率允许的上界", () => {
        const p = steady({ rateHz: 5, rateRampEnd: 2, depthCents: 100 });
        const values = render(p, 1);
        // 末端瞬时速率 10 Hz，一帧 5 ms 最多走 0.05 周期；正弦在零交叉处
        // 斜率最大，故单帧最大增量 ≈ 100 cents × 2π × 0.05 ≈ 31.4 cents。
        const maxAllowed = 100 * 2 * Math.PI * 0.05 * 1.05;
        let maxDelta = 0;
        for (let i = 1; i < values.length; i += 1) {
            maxDelta = Math.max(maxDelta, Math.abs(values[i] - values[i - 1]));
        }
        expect(maxDelta).toBeLessThan(maxAllowed);
    });
});

describe("收尾对齐整数周期", () => {
    test("末帧回到基线上", () => {
        // 5.3 Hz × 1 s = 5.3 个周期；对齐后应为 5 或 6 个整周期，
        // 末帧落在正弦的过零点上。
        const p = steady({ rateHz: 5.3, alignCycles: true });
        const values = render(p, 1);
        expect(Math.abs(values[values.length - 1] - 60)).toBeLessThan(1e-6);
    });

    test("不对齐时末帧一般不落在基线上（对照组）", () => {
        const p = steady({ rateHz: 5.3, alignCycles: false });
        const values = render(p, 1);
        // 0.3 个周期的余量 → 偏离基线明显。
        expect(Math.abs(values[values.length - 1] - 60)).toBeGreaterThan(0.05);
    });
});

describe("深度按参数族换算", () => {
    test("音高：加性，100 cents = 1 个半音", () => {
        const values = render(steady({ depthCents: 100 }), 1, { param: "pitch" });
        const max = Math.max(...values);
        const min = Math.min(...values);
        expect(max).toBeCloseTo(61, 2);
        expect(min).toBeCloseTo(59, 2);
    });

    test("音高：30 cents 的默认深度是 ±0.3 半音", () => {
        const values = render(steady({ depthCents: 30 }), 1, { param: "pitch" });
        expect(Math.max(...values) - 60).toBeCloseTo(0.3, 2);
    });

    test("乘性增益：静音帧保持静音（历史实现用加性，会把静音抬起来）", () => {
        const zeros = new Array<number>(201).fill(0);
        const values = render(steady({ depthCents: 60 }), 1, {
            baseline: "existing",
            param: "dyn",
            original: zeros,
        });
        for (const value of values) expect(value).toBe(0);
    });

    test("乘性增益：深度 100 = ±100% 倍率", () => {
        const ones = new Array<number>(201).fill(1);
        const values = render(steady({ depthCents: 100 }), 1, {
            baseline: "existing",
            param: "dyn",
            original: ones,
        });
        expect(Math.max(...values)).toBeCloseTo(2, 2);
        expect(Math.min(...values)).toBeCloseTo(0, 2);
    });

    test("乘性增益：负半周钳到 0，不产生负增益", () => {
        const ones = new Array<number>(201).fill(1);
        const values = render(steady({ depthCents: 200 }), 1, {
            baseline: "existing",
            param: "volume",
            original: ones,
        });
        for (const value of values) expect(value).toBeGreaterThanOrEqual(0);
    });

    test("cents 类参数 1:1（子轨音分偏移直接按分算）", () => {
        const values = render(steady({ depthCents: 50 }), 1, {
            param: "child_pitch_offset_cents@track-1",
        });
        // 峰值通常落在两帧之间，采样到的最大值略小于理论幅值。
        expect(Math.max(...values) - 60).toBeCloseTo(50, 1);
    });

    test("原始值域参数按半量程百分比换算（张力 ±100 → 深度 30 = ±30）", () => {
        const values = render(steady({ depthCents: 30 }), 1, { param: "tension" });
        expect(Math.max(...values) - 60).toBeCloseTo(30, 2);
    });
});

describe("基线模式", () => {
    test("existing：在已有曲线上叠加，保留原有起伏", () => {
        // 一条从 58 线性升到 62 的曲线。
        const original = Array.from({ length: 201 }, (_, i) => 58 + (4 * i) / 200);
        const values = buildVibratoCurve({
            startFrame: 0,
            startValue: 58,
            endFrame: 200,
            endValue: 62,
            preset: steady({ depthCents: 20 }),
            baseline: "existing",
            param: "pitch",
            framePeriodMs: FP,
            original,
        }).dense;
        // 每个点的偏移都在 ±0.2 半音内，且整体趋势不变。
        for (let i = 0; i < values.length; i += 1) {
            expect(Math.abs(values[i] - original[i])).toBeLessThanOrEqual(0.2 + 1e-9);
        }
        expect(values[0]).toBeLessThan(values[values.length - 1]);
    });

    test("average：整段围绕均值恒定摆动", () => {
        const values = buildVibratoCurve({
            startFrame: 0,
            startValue: 58,
            endFrame: 200,
            endValue: 62,
            preset: steady({ depthCents: 20 }),
            param: "pitch",
            framePeriodMs: FP,
            baseline: "average",
        }).dense;
        const mean = (Math.max(...values) + Math.min(...values)) / 2;
        expect(mean).toBeCloseTo(60, 2);
    });

    test("holdStart：基线不随终点漂移", () => {
        const values = buildVibratoCurve({
            startFrame: 0,
            startValue: 60,
            endFrame: 200,
            endValue: 72,
            preset: steady({ depthCents: 20 }),
            baseline: "holdStart",
            param: "pitch",
            framePeriodMs: FP,
        }).dense;
        expect((Math.max(...values) + Math.min(...values)) / 2).toBeCloseTo(60, 2);
    });

    test("line：基线在两端点之间线性插值", () => {
        const values = buildVibratoCurve({
            startFrame: 0,
            startValue: 60,
            endFrame: 200,
            endValue: 72,
            preset: steady({ depthCents: 20 }),
            baseline: "line",
            param: "pitch",
            framePeriodMs: FP,
        }).dense;
        // 整段的中心线是两端点的中点（60 与 72 之中 → 66）。
        expect((Math.max(...values) + Math.min(...values)) / 2).toBeCloseTo(66, 1);
        // 末帧附近围绕 72 摆动。
        const tail = values.slice(-3);
        expect(tail.reduce((a, b) => a + b, 0) / tail.length).toBeCloseTo(72, 1);
    });
});

describe("深度包络", () => {
    test("渐入：首帧无偏移，渐入结束后达到满幅", () => {
        const values = render(steady({ depthCents: 100, attackMs: 200 }), 1);
        expect(Math.abs(values[0] - 60)).toBeLessThan(1e-9);
        // 第 100 帧（500 ms）已在渐入之后。
        expect(Math.abs(values[100] - 60)).toBeGreaterThan(0.5);
    });

    test("渐出：末帧回到基线", () => {
        const values = render(steady({ depthCents: 100, releaseMs: 200 }), 1);
        expect(Math.abs(values[values.length - 1] - 60)).toBeLessThan(1e-9);
    });

    test("渐强：起点倍率决定首帧深度", () => {
        const values = render(steady({ depthCents: 100, depthRamp: { start: 0.5, end: 1 } }), 1);
        // 首帧相位为 0，正弦本身也是 0，因此这里比较的是斜率而非绝对值：
        // 前几帧的包络应约为满幅的一半。
        const early = Math.abs(values[1] - 60);
        const late = Math.abs(values[100] - 60);
        expect(early).toBeGreaterThan(0);
        expect(late).toBeGreaterThan(early);
    });

    /*
     * 【这条测试锁的是"渐入 / 渐出只能到中线"的旧限制】上限曾经是整段时长的
     * **一半**（怕短选区上两者互相吃掉），于是用户做不出"整条线由弱到强"的颤音：
     * 渐入拉到一半就到顶了。现在上限是整段时长本身。
     */
    test("渐入可以铺满整条线（上限是整段时长，而不是一半）", () => {
        // 100 ms 的选区，却请求 500 ms 的渐入：整条线都在渐入，末帧才到满幅。
        const env = renderEnvelope(steady({ depthCents: 100, attackMs: 500 }), 0.1);
        expect(env[0]).toBeCloseTo(0, 9);
        expect(env[env.length - 1]).toBeCloseTo(100, 6);
        // 旧实现的中点已经满幅了；现在中点应当只有一半左右。
        const mid = env[Math.floor(env.length / 2)];
        expect(mid).toBeGreaterThan(20);
        expect(mid).toBeLessThan(80);
    });

    test("渐出可以铺满整条线", () => {
        const env = renderEnvelope(steady({ depthCents: 100, releaseMs: 500 }), 0.1);
        expect(env[0]).toBeCloseTo(100, 6);
        expect(env[env.length - 1]).toBeCloseTo(0, 9);
    });

    /*
     * 两者重叠时按**乘积**合成（与音频里两级推子串联同理）：两端归零、中间是一个
     * 连续凹下去的拱形，而不是互相截断。保持乘法而不是"按比例压到刚好相接"，
     * 是因为手柄位置必须始终等于斜坡的起点 / 终点 —— 归一化会让另一个手柄在用户
     * 拖这一个时自己动起来。
     */
    test("渐入与渐出重叠时相乘：连续凹形，两端归零", () => {
        const env = renderEnvelope(steady({ depthCents: 100, attackMs: 500, releaseMs: 500 }), 0.1);
        expect(env[0]).toBeCloseTo(0, 9);
        expect(env[env.length - 1]).toBeCloseTo(0, 9);
        const mid = env[Math.floor(env.length / 2)];
        expect(mid).toBeGreaterThan(0);
        expect(mid).toBeLessThan(50);
        // 相邻帧连续：重叠不会制造台阶。
        let maxJump = 0;
        for (let i = 1; i < env.length; i += 1) {
            maxJump = Math.max(maxJump, Math.abs(env[i] - env[i - 1]));
        }
        expect(maxJump).toBeLessThan(20);
    });
});

describe("不规则度", () => {
    test("同一 seed 逐值可复现（预览与提交必须一致）", () => {
        const p = steady({ depthCents: 40, irregularity: 60 });
        const a = render(p, 1, { seed: 42 });
        const b = render(p, 1, { seed: 42 });
        expect(a).toEqual(b);
    });

    test("不同 seed 得到不同波形", () => {
        const p = steady({ depthCents: 40, irregularity: 60 });
        const a = render(p, 1, { seed: 1 });
        const b = render(p, 1, { seed: 2 });
        expect(a).not.toEqual(b);
    });

    test("不规则度为 0 时与 seed 无关", () => {
        const p = steady({ depthCents: 40, irregularity: 0 });
        expect(render(p, 1, { seed: 1 })).toEqual(render(p, 1, { seed: 999 }));
    });

    /*
     * ★ 回归：省略 `seed` 时必须以**预设自己的 `seed` 字段**为准。
     *
     * 故障形态：这里只在 `input.seed` 里找种子并兜底为 0，于是除了显式传种子的
     * 拖拽路径以外，管理器预览 / 试听 / 套用到选区都恒定用 0 渲染 —— 骰子按钮
     * 明明改了 `preset.seed`，画面上却什么都不会变（"按钮完全不起作用"）。
     */
    test("省略 seed 时用预设自己的 seed 字段（骰子按钮改的就是它）", () => {
        const one = render(steady({ depthCents: 40, irregularity: 60, seed: 1 }), 1);
        const two = render(steady({ depthCents: 40, irregularity: 60, seed: 2 }), 1);
        expect(one).not.toEqual(two);
        // 同一预设仍然逐帧一致（预览与提交不能跳）。
        expect(render(steady({ depthCents: 40, irregularity: 60, seed: 1 }), 1)).toEqual(one);
    });

    test("显式传入的 seed 优先于预设字段（拖拽路径用本次手势的工作种子）", () => {
        const p = steady({ depthCents: 40, irregularity: 60, seed: 1 });
        expect(render(p, 1, { seed: 7 })).not.toEqual(render(p, 1));
    });
});

describe("逐帧吸附按最终值作用（既有的量化画线行为）", () => {
    /*
     * 【这是刻意保留的设计】开着吸附时整条曲线被量化到半音 / 音阶格，
     * 用户因此可以快速画出一条**量化的参数线**。吸附必须作用在合成后的
     * 值上，而不是只作用在基线上 —— 改动这条语义会破坏该功能。
     */
    test("吸附回调作用于合成后的值", () => {
        const values = render(steady({ depthCents: 37 }), 0.5, {
            snapFinalValue: (value) => Math.round(value),
        });
        for (const value of values) expect(Number.isInteger(value)).toBe(true);
    });

    test("吸附回调拿到的是帧号，便于按位置查音阶", () => {
        const frames: number[] = [];
        render(steady({ depthCents: 10 }), 0.1, {
            snapFinalValue: (value, frame) => {
                frames.push(frame);
                return value;
            },
        });
        expect(frames[0]).toBe(0);
        expect(frames[frames.length - 1]).toBe(20);
    });

    test("不传吸附回调时不做任何量化", () => {
        const values = render(steady({ depthCents: 37 }), 0.5);
        expect(values.some((value) => !Number.isInteger(value))).toBe(true);
    });
});

describe("输出形状与边界", () => {
    test("dense 长度与 minF/maxF 对齐，且支持反向拖拽", () => {
        const result = buildVibratoCurve({
            startFrame: 300,
            startValue: 60,
            endFrame: 200,
            endValue: 62,
            preset: steady({ depthCents: 20 }),
            param: "pitch",
            framePeriodMs: FP,
        });
        expect(result.minF).toBe(200);
        expect(result.maxF).toBe(300);
        expect(result.dense).toHaveLength(101);
        for (const value of result.dense) expect(Number.isFinite(value)).toBe(true);
    });

    test("零长度拖拽不抛异常", () => {
        const result = buildVibratoCurve({
            startFrame: 100,
            startValue: 60,
            endFrame: 100,
            endValue: 60,
            preset: steady({ depthCents: 50 }),
            param: "pitch",
            framePeriodMs: FP,
        });
        expect(result.dense).toHaveLength(1);
        expect(Number.isFinite(result.dense[0])).toBe(true);
    });

    test("深度 0 时输出恒等于基线（直线预设）", () => {
        const values = render(steady({ depthCents: 0 }), 1);
        for (const value of values) expect(value).toBe(60);
    });

    test("非有限帧周期回退到默认值而不是产生 NaN", () => {
        const values = render(steady({ depthCents: 30 }), 0.5, { framePeriodMs: Number.NaN });
        for (const value of values) expect(Number.isFinite(value)).toBe(true);
    });

    test("所有形状都能产出有限值", () => {
        for (const shape of [
            "sine",
            "triangle",
            "sawUp",
            "sawDown",
            "square",
            "trapezoid",
            "trill",
        ] as const) {
            const values = render(
                steady({ depthCents: 40, cycle: { kind: "shape", shape, skew: 0.5 } }),
                0.5,
            );
            for (const value of values) expect(Number.isFinite(value)).toBe(true);
        }
    });

    test("采样式波形（手绘表）同样可用", () => {
        const table = [-1, 0, 1, 0, -1, 0, 1, 0];
        const values = render(steady({ depthCents: 40, cycle: { kind: "table", table } }), 0.5);
        for (const value of values) expect(Number.isFinite(value)).toBe(true);
        expect(Math.max(...values)).toBeGreaterThan(60);
        expect(Math.min(...values)).toBeLessThan(60);
    });
});

describe("estimateCycles", () => {
    test("按周期数模式时返回设定值", () => {
        const p = steady({ rateMode: "cycles", cycles: 7 });
        expect(estimateCycles(p, 201, FP)).toBeCloseTo(7, 6);
    });

    test("按频率模式时跟随时长", () => {
        const p = steady({ rateHz: 5 });
        expect(estimateCycles(p, 201, FP)).toBeCloseTo(5, 6);
        expect(estimateCycles(p, 401, FP)).toBeCloseTo(10, 6);
    });

    test("对齐整数周期时取整", () => {
        const p = steady({ rateHz: 5.3, alignCycles: true });
        expect(estimateCycles(p, 201, FP)).toBe(5);
    });
});

/*
 * 逐帧基线（预览的纵轴定标要用它）。
 *
 * 【为什么值得测】预览画的是"相对基线的偏移"。基线模式（`line` / `holdStart` /
 * `holdEnd` / `average` / `existing`）的判定只在内核里，调用方重算必然与渲染分叉 ——
 * 所以这个输出必须与 `dense` 出自同一次计算，且能被独立核对。
 */
describe("collectBaseline", () => {
    test("缺省不返回；开启后与 dense 同长", () => {
        const p = steady({ depthCents: 40 });
        const base = {
            startFrame: 0,
            startValue: 60,
            endFrame: 99,
            endValue: 60,
            preset: p,
            param: "pitch",
            framePeriodMs: FP,
        };
        expect(buildVibratoCurve(base).baseline).toBeUndefined();
        const withBaseline = buildVibratoCurve({ ...base, collectBaseline: true });
        expect(withBaseline.baseline?.length).toBe(withBaseline.dense.length);
    });

    test("existing 基线逐帧等于原曲线", () => {
        const original = Array.from({ length: 100 }, (_, i) => 60 + i * 0.1);
        const result = buildVibratoCurve({
            startFrame: 0,
            startValue: original[0],
            endFrame: original.length - 1,
            endValue: original[original.length - 1],
            original,
            preset: steady({ depthCents: 40 }),
            baseline: "existing",
            param: "pitch",
            framePeriodMs: FP,
            collectBaseline: true,
        });
        for (let i = 0; i < original.length; i += 1) {
            expect(result.baseline![i]).toBeCloseTo(original[i], 9);
        }
    });

    test("line 基线就是端点之间的直线（与渲染输出互相印证）", () => {
        // 深度 0 时 `dense` 即基线本身：用它来核对 `baseline`，不是另写一份公式。
        const result = buildVibratoCurve({
            startFrame: 0,
            startValue: 58,
            endFrame: 99,
            endValue: 62,
            preset: steady({ depthCents: 0 }),
            baseline: "line",
            param: "pitch",
            framePeriodMs: FP,
            collectBaseline: true,
        });
        expect(result.baseline![0]).toBeCloseTo(58, 9);
        expect(result.baseline![result.dense.length - 1]).toBeCloseTo(62, 9);
        for (let i = 0; i < result.dense.length; i += 1) {
            expect(result.baseline![i]).toBeCloseTo(result.dense[i], 9);
        }
    });
});
