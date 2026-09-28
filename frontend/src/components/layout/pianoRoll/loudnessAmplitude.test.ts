import { describe, expect, it } from "vitest";

import {
    makeLoudnessAmplitudeMap,
    type LoudnessAutomationSource,
    type LoudnessLiveCurve,
} from "./PianoRollWaveformSurface";
import { readAmplitudeRevision, type WaveformAmplitudeMap } from "../../../waveform/geometry";
import { DYN_MAX_GAIN, DYN_SILENCE_FLOOR, DYN_VALUE_MAX } from "./paramRanges";

/**
 * 构造快照：`volume` / `dynTarget` / `dynBaseline` 逐帧给定，帧周期 10ms
 * （便于心算）。首帧对应时间轴帧 0。
 */
function source(args: {
    volume?: number[];
    dynTarget?: number[];
    dynBaseline?: number[];
    stride?: number;
    framePeriodMs?: number;
    startFrame?: number;
}): LoudnessAutomationSource {
    const {
        volume = [1, 1],
        dynTarget = [1, 1],
        dynBaseline = [],
        stride = 1,
        framePeriodMs = 10,
        startFrame = 0,
    } = args;
    return { startFrame, framePeriodMs, stride, volume, dynTarget, dynBaseline };
}

function noLive(): LoudnessLiveCurve | null {
    return null;
}

describe("makeLoudnessAmplitudeMap", () => {
    it("scales the waveform by volume × dyn gain at each moment", () => {
        // 帧 0：volume 0.5、目标 1.0 / 原声 0.5 → dyn ×2 → 合计 ×1；
        // 帧 1：volume 1.0、目标 0.5 / 原声 1.0 → dyn ×0.5 → 合计 ×0.5。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [0.5, 1], dynTarget: [1, 0.5], dynBaseline: [0.5, 1] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(0.4, 1, 0)).toBeCloseTo(0.4 * 0.5 * 2, 10);
        expect(map(0.4, 1, 0.01)).toBeCloseTo(0.4 * 1.0 * 0.5, 10);
    });

    it("multiplies clip gain (second argument) into the result", () => {
        // 几何层传入的 gain = clip 增益 × 淡化；映射必须乘在 volume/dyn 之外。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [2], dynTarget: [1], dynBaseline: [1] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(0.3, 0.5, 0)).toBeCloseTo(0.3 * 0.5 * 2 * 1, 10);
    });

    it("★ gain follows the live curve without rebuilding the map", () => {
        // 这一条钉住"编辑动态时波形实时跟随"：映射对象保持同一引用，
        // 而内部数据（live 覆盖）变化后必须反映到输出上。
        let live = [1, 1];
        const map = makeLoudnessAmplitudeMap(
            source({ dynBaseline: [1, 1] }),
            {
                volume: noLive,
                dyn: () => ({ startFrame: 0, stride: 1, values: live }),
            },
            () => 0,
        );
        const before = map(1, 1, 0);
        live = [4, 1]; // 用户把目标电平抬到 4（上限）
        const after = map(1, 1, 0);
        expect(after).toBeCloseTo(before * 4, 10);
    });

    it("★ live volume wins inside its window, snapshot outside", () => {
        // 编辑音量时：live 覆盖窗口内读 live，窗口外回退快照。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [1, 1, 1], dynBaseline: [] }),
            {
                volume: () => ({ startFrame: 0, stride: 1, values: [0.5, 0.5] }),
                dyn: noLive,
            },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(0.5, 10); // live 窗口内
        expect(map(1, 1, 0.02)).toBeCloseTo(1.0, 10); // live 窗口外 → 快照 1.0
    });

    it("applies unity gain where the target follows the original", () => {
        // 目标 == 原声 → 增益 1。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [0.5], dynBaseline: [0.5] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(1, 10);
    });

    it("★ boosts quiet content at any realistic level (was blocked below −26 dBFS)", () => {
        // 回归：门限曾定在 −26 dBFS（0.05），导致轻声/气声/尾音整体提不上去。
        // 这些电平必须都能提升（下限已降到 −60 dBFS）。
        for (const db of [-20, -26, -34, -40, -45, -55]) {
            const base = Math.pow(10, db / 20);
            const map = makeLoudnessAmplitudeMap(
                source({ dynTarget: [0.582], dynBaseline: [base] }),
                { volume: noLive, dyn: noLive },
                () => 0,
            );
            // 所需倍数很大 → 钳到上限，但**必须大于 1**（旧实现恒为 1 = 提不动）。
            expect(map(1, 1, 0)).toBeGreaterThan(1);
        }
    });

    it("bounds no-content frames and fades them out (continuity)", () => {
        // 无内容帧（−80 dBFS）：分母钳到下限 ⇒ 增益**有界**，并按"无内容"平滑淡出。
        // 两个历史实现都被否掉了：固定 ×1000 会把抖动噪声底抬成可闻嘶声（−90 dB
        // 原声 ×500 = −36 dBFS）；而更早的"低于门限就拒绝放大返回 1"会在下限处
        // 产生阶跃（近零伪影的根因）。淡出曲线（smoothstep）在下限处**导数也连续**。
        const baseline = 0.0001;
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [1], dynBaseline: [baseline] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        const gain = map(1, 1, 0);
        // 过渡带 = [下限×0.5, 下限]（−66…−60 dBFS）：−80 dBFS 远在其下 ⇒ 完全不放大。
        expect(gain).toBeGreaterThanOrEqual(0);
        expect(gain).toBeLessThan(DYN_MAX_GAIN);
        // 真正要保证的是**输出电平**（原声 × 增益）：远低于旧的"按目标电平放大"。
        expect(20 * Math.log10(baseline * gain)).toBeLessThan(-50);
        expect(20 * Math.log10(baseline * DYN_MAX_GAIN)).toBeCloseTo(-20, 0); // 旧行为对照
    });

    it("★ drawn silence silences the noise floor (protection never blocks attenuation)", () => {
        // 回归：原声在门限之下但非零（0 < base < 0.05）时，画 0 必须真的静音 ——
        // 旧实现把静音保护无条件作用于所有请求，"全曲线拉到 0 后噪声底仍发声"。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [0], dynBaseline: [0.02] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(0, 10);
        // 真静音基线（0）：画 0 = 静音。
        const zeroBase = makeLoudnessAmplitudeMap(
            source({ dynTarget: [0], dynBaseline: [0] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(zeroBase(1, 1, 0)).toBeCloseTo(0, 10);
    });

    it("★ honors the target exactly; the clamp is only a numeric backstop", () => {
        // 回归：上限曾仅 ×4，使 −26 dBFS（0.05）画满量程目标时只得 ×4
        //（"画了目标却达不到"）。现在上限 = 值域顶端/下限，正常输入不可达。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [1], dynBaseline: [0.05] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(20, 9); // 1 / 0.05，精确兑现

        // 上限只在越界曲线造成荒谬增益时兜底：它远高于任何合法目标所需的倍数。
        expect(DYN_MAX_GAIN).toBe(DYN_VALUE_MAX / DYN_SILENCE_FLOOR);
    });

    it("does not extend the last value past the curve (no tail pollution)", () => {
        // 曲线只有 1 帧：其后所有时刻都必须回到"不施加增益"，
        // 而不是把末值一直 hold 下去（历史共振峰 bug 的同源教训）。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [1], dynBaseline: [0.25] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(4, 10);
        // 越界（帧 1 及以后）→ 不施加动态增益。
        expect(map(1, 1, 0.01)).toBeCloseTo(1, 10);
        expect(map(1, 1, 5)).toBeCloseTo(1, 10);
    });

    it("treats out-of-range volume as unity (holds no automation)", () => {
        // 音量曲线只有 1 帧：其后回退 1.0（越界持有末值对音量是合理的，
        // 但快照覆盖全工程，实际只在快照边界外发生）。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [0.25] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(0.25, 10);
        expect(map(1, 1, 1)).toBeCloseTo(1, 10);
    });

    it("respects stride when mapping time to curve index", () => {
        // stride 2：曲线每 2 个参数帧一个采样 → 帧号跨度翻倍（20ms）。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [0.5, 0.25], dynBaseline: [1, 1], stride: 2 }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(0.5, 10); // 帧 0 → ×0.5
        expect(map(1, 1, 0.02)).toBeCloseTo(0.25, 10); // 帧 2（stride 2）→ ×0.25
    });

    it("blends target and baseline linearly in time", () => {
        // 帧 0→1 之间取中点（5ms）：目标 1→0.5、原声 1→1 → 中点目标 0.75 → 增益 0.75。
        const map = makeLoudnessAmplitudeMap(
            source({ dynTarget: [1, 0.5], dynBaseline: [1, 1] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0.005)).toBeCloseTo(0.75, 10);
    });

    it("dyn gain stays unity when the baseline is empty (analysis pending)", () => {
        // 分析未就绪：动态不生效，volume 仍生效 —— 绝不返回全零映射。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [0.5], dynTarget: [1], dynBaseline: [] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(1, 1, 0)).toBeCloseTo(0.5, 10);
    });

    it("degenerates to linear when time is unavailable", () => {
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [0.1], dynTarget: [1], dynBaseline: [0.25] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(0.5, 2, null)).toBeCloseTo(1.0, 10);
        expect(map(0.5, 2, Number.NaN)).toBeCloseTo(1.0, 10);
    });

    it("exposes the revision reader for the geometry cache", () => {
        // 修订号通过函数惰性读取，映射对象本身不必重建 —— 否则每次
        // pointermove 都会重建整份几何。
        let rev = 0;
        const map = makeLoudnessAmplitudeMap(
            source({ dynBaseline: [1] }),
            { volume: noLive, dyn: noLive },
            () => rev,
        );
        const first = map;
        rev += 1;
        expect(map).toBe(first);
        expect(readAmplitudeRevision(map)).toBe(1);
    });

    it("ignores non-finite peaks", () => {
        const map = makeLoudnessAmplitudeMap(
            source({ dynBaseline: [1] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        expect(map(Number.NaN, 1, 0)).toBe(0);
        expect(map(1, Number.POSITIVE_INFINITY, 0)).toBe(0);
    });
});

/**
 * 时域因子视图（`factorAt`）。
 *
 * 【为什么单独测量】几何层在映射声明了 `factorAt` 时走快路径（每切片求值一次
 * 因子、min/max 各一次乘法），不再逐值调用映射 —— 这是拖动音量/动态时不卡顿的
 * 关键（一次重建最多数万次调用，逐值路径要在那里重复取样曲线与解析 live 覆盖）。
 * 快路径必须与逐值路径**逐值等价**：同一份数据不能因为走了哪条路径而画出不同
 * 的波形。因此这里逐点比对两者。
 */
describe("makeLoudnessAmplitudeMap · factor view", () => {
    /** 取映射上挂载的因子视图（类型断言收在这里）。 */
    function factorOf(map: WaveformAmplitudeMap): (t: number | null) => number | null {
        const fn = (map as unknown as { factorAt?: (t: number | null) => number | null }).factorAt;
        if (typeof fn !== "function") throw new Error("factorAt must be declared");
        return fn;
    }

    it("★ is equivalent to the per-value map at every sampled time", () => {
        // 音量 + 动态同时生效，并带一段 live 覆盖：两条路径必须逐点相等。
        const live = { startFrame: 0, stride: 1, values: [0.5, 0.5, 0.25, 0.25] };
        const map = makeLoudnessAmplitudeMap(
            source({
                volume: [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                dynTarget: [1, 0.9, 0.5, 0.25, 1, 1],
                dynBaseline: [0.5, 1, 1, 0.5, 1, 1],
            }),
            { volume: () => live, dyn: noLive },
            () => 0,
        );
        const factor = factorOf(map);

        // 覆盖 timeSec 的各类取值：live 窗口内 / 窗口外、曲线端点、越界。
        const times = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 1, 60];
        for (const t of times) {
            const perValue = map(0.4, 0.75, t);
            const viaFactor = 0.4 * 0.75 * (factor(t) ?? Number.NaN);
            expect(viaFactor).toBeCloseTo(perValue, 12);
        }
    });

    it("★ matches the per-value map for non-finite times", () => {
        // timeSec 未知（null / NaN）→ 两边都必须退化为"不施加响度自动化"。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [0.1], dynTarget: [1], dynBaseline: [0.25] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        const factor = factorOf(map);

        expect(factor(null)).toBe(1);
        expect(factor(Number.NaN)).toBe(1);
        expect(map(0.5, 2, null)).toBeCloseTo(1.0, 10);
        expect(map(0.5, 2, Number.NaN)).toBeCloseTo(1.0, 10);
    });

    it("★ reflects live edits without rebuilding the map", () => {
        // 拖动契约：映射对象引用不变，因子视图读到的必须是**最新**的 live 值。
        let liveValues = [1, 1];
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [1, 1], dynBaseline: [] }),
            { volume: () => ({ startFrame: 0, stride: 1, values: liveValues }), dyn: noLive },
            () => 0,
        );
        const factor = factorOf(map);

        expect(factor(0)).toBeCloseTo(1, 10);
        liveValues = [0.25, 1];
        expect(factor(0)).toBeCloseTo(0.25, 10);
        // 逐值路径同步跟随（同一实现）。
        expect(map(1, 1, 0)).toBeCloseTo(0.25, 10);
    });

    it("returns a finite factor where the per-value map is finite", () => {
        // 动态静音（基线 0）等退化分支下因子仍须有限，否则几何层会跳过整列。
        const map = makeLoudnessAmplitudeMap(
            source({ volume: [1, 1], dynTarget: [0, 1], dynBaseline: [0, 0.0001] }),
            { volume: noLive, dyn: noLive },
            () => 0,
        );
        const factor = factorOf(map);

        expect(factor(0)).toBe(0); // 真静音：画了静音 = 静音
        // 无内容帧：有界 + 淡出（详见上一条用例的说明）。关键是**有限** ——
        // 非有限值会让几何层跳过整列。
        const gain = factor(0.01);
        expect(Number.isFinite(gain)).toBe(true);
        expect(gain).toBeGreaterThanOrEqual(0);
        expect(gain).toBeLessThan(DYN_MAX_GAIN);
    });

    it("stays attached to the same map object as revision()", () => {
        const map = makeLoudnessAmplitudeMap(
            source({ dynBaseline: [1] }),
            { volume: noLive, dyn: noLive },
            () => 7,
        );
        expect(typeof factorOf(map)).toBe("function");
        expect(readAmplitudeRevision(map)).toBe(7);
    });
});
