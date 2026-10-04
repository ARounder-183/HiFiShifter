/**
 * 「拖拽 clip 期间参数编辑器波形错位」的**行为回归**。
 *
 * ## 被钉住的缺陷
 *
 * 波形乘数是 `volume(t) × dyn增益(t)`，而 `dyn增益 = 目标电平 / 原声电平基线`。
 * 基线是 **clip 几何的函数**：clip 从 0s 移到 3s，同一段素材的基线就从帧 `[0,200)`
 * 挪到 `[600,800)`。时间轴拖拽期间几何只在 Redux 里乐观更新，响度快照要等提交才
 * 重取 —— 于是整个手势期间「旧位置的基线 × 新位置的峰值」：
 *
 * - 把 clip 拖到**静音区**：新位置的旧基线接近 0，"无内容淡出"把波形压成一条平线；
 * - 拖到**响区**：目标/极小基线被放大成满高平台。
 *
 * 松手后快照重取，波形才恢复 —— 这正是用户报告的「拖拽过程中波形有很大问题」。
 *
 * 本文件的每一条用例都对应一个必须成立的语义，而不是实现细节。
 */
import { describe, expect, it } from "vitest";

import { makeLoudnessAmplitudeMap, type LoudnessAutomationSource } from "./PianoRollWaveformSurface";
import {
    createLoudnessGeometryWarp,
    type LoudnessGeometryWarp,
    type WarpClipGeometry,
} from "./loudnessGeometryWarp";
import type { WaveformAmplitudeFactors } from "../../../waveform/geometry";

/** 帧周期 10ms：帧号 ×10 = 毫秒，便于心算。 */
const FP = 10;

function source(args: {
    volume: number[];
    dynTarget: number[];
    dynBaseline: number[];
}): LoudnessAutomationSource {
    return {
        startFrame: 0,
        stride: 1,
        framePeriodMs: FP,
        volume: args.volume,
        dynTarget: args.dynTarget,
        dynBaseline: args.dynBaseline,
    };
}

/** 建映射实例（`warp` 惰性读取，便于在同一实例上换映射后重建查表）。 */
function amplitudeMap(
    src: LoudnessAutomationSource,
    warp: () => LoudnessGeometryWarp | null,
): WaveformAmplitudeFactors {
    const map = makeLoudnessAmplitudeMap(
        src,
        { volume: () => null, dyn: () => null },
        () => 0,
        warp,
    );
    return map as unknown as WaveformAmplitudeFactors;
}

interface GeoOpts {
    gain?: number;
    sourceStartSec?: number;
    sourceEndSec?: number;
    playbackRate?: number;
    reversed?: boolean;
    loopEnabled?: boolean;
}

/** 一份 clip 几何（源窗口默认为 [0, 长度×速率)）。 */
function geo(
    id: string,
    startSec: number,
    lengthSec: number,
    opts: GeoOpts = {},
): WarpClipGeometry {
    const rate = opts.playbackRate ?? 1;
    const sourceStartSec = opts.sourceStartSec ?? 0;
    return {
        id,
        startSec,
        lengthSec,
        gain: opts.gain ?? 1,
        sourceStartSec,
        sourceEndSec: opts.sourceEndSec ?? sourceStartSec + lengthSec * rate,
        playbackRate: rate,
        reversed: opts.reversed ?? false,
        loopEnabled: opts.loopEnabled ?? false,
    };
}

function warpOf(
    origin: readonly WarpClipGeometry[],
    clips: readonly WarpClipGeometry[],
    lockParamLines = false,
): LoudnessGeometryWarp | null {
    return createLoudnessGeometryWarp({
        origin,
        clips,
        framePeriodMs: FP,
        lockParamLines,
    });
}

/** 帧号 → 秒。 */
const at = (frame: number): number => (frame * FP) / 1000;

describe("拖拽期间基线跟随几何", () => {
    /** 前 16 帧响（0.5），之后是远低于下限的静音（0.0001）。 */
    function loudThenSilent(): LoudnessAutomationSource {
        const loud = new Array<number>(16).fill(0.5);
        const silent = new Array<number>(32).fill(0.0001);
        const baseline = [...loud, ...silent];
        return source({
            volume: new Array<number>(48).fill(1),
            // 目标与基线逐位相同 = 「未画」（后端出口把哨兵物化成了基线值）。
            dynTarget: [...baseline],
            dynBaseline: baseline,
        });
    }

    it("★ 把 clip 拖进静音区后波形不再被压平（未画帧的目标跟随原声）", () => {
        const src = loudThenSilent();
        // clip 从帧 [0,16) 移到 [16,32)：查询帧 24 的素材原先在帧 8。
        const origin = [geo("c1", 0, 0.16)];
        const moved = [geo("c1", 0.16, 0.16)];

        // 无映射（= 修复前的行为）：新位置的旧基线是静音 ⇒ 淡出把波形压成 0。
        const before = amplitudeMap(src, () => null);
        expect(before.factorAt?.(at(24))).toBe(0);

        // 有映射：基线跟着素材搬到帧 8（响段）⇒ 未画帧增益 1，波形保持原样。
        const after = amplitudeMap(src, () => warpOf(origin, moved));
        expect(after.factorAt?.(at(24))).toBeCloseTo(1, 6);
    });

    it("★ 平移的「形状不变」性质：新位置的因子等于旧位置的因子", () => {
        // 带起伏的曲线，避免"恒等"掩盖错误。
        const baseline = Array.from({ length: 48 }, (_, f) => 0.2 + 0.5 * Math.abs(Math.sin(f * 0.4)));
        const target = baseline.map((b, f) => b * (0.3 + 1.5 * Math.abs(Math.cos(f * 0.31))));
        const src = source({
            volume: Array.from({ length: 48 }, (_, f) => 0.6 + 0.4 * Math.sin(f * 0.23)),
            dynTarget: target,
            dynBaseline: baseline,
        });
        const delta = 12;
        const warp = warpOf(
            [geo("c1", 0, 0.24)],
            [geo("c1", at(delta), 0.24)],
            true,
        );
        const mapped = amplitudeMap(src, () => warp);
        const plain = amplitudeMap(src, () => null);
        for (let k = 0; k < 24; k += 1) {
            expect(mapped.factorAt?.(at(delta + k))).toBeCloseTo(
                plain.factorAt?.(at(k)) as number,
                10,
            );
        }
    });

    it("无映射时映射路径与既有行为逐值相同（稳态零影响）", () => {
        const baseline = Array.from({ length: 32 }, (_, f) => 0.1 + 0.4 * Math.abs(Math.sin(f * 0.5)));
        const src = source({
            volume: new Array<number>(32).fill(1),
            dynTarget: baseline.map((b) => b * 0.7),
            dynBaseline: baseline,
        });
        // 一侧挂映射 provider（恒返回 null），一侧完全不传 —— 稳态下必须逐值相同。
        const withProvider = amplitudeMap(src, () => null);
        const withoutProvider = makeLoudnessAmplitudeMap(
            src,
            { volume: () => null, dyn: () => null },
            () => 0,
        ) as unknown as WaveformAmplitudeFactors;
        for (let f = 0; f < 32; f += 1) {
            expect(withProvider.factorAt?.(at(f))).toBe(withoutProvider.factorAt?.(at(f)));
        }
        // 带查表也一样（映射恒为 null 时 LUT 的键与既有路径一致）。
        withProvider.beginWindow?.(at(0), at(31));
        for (let f = 0; f < 32; f += 1) {
            expect(withProvider.factorAt?.(at(f))).toBe(withoutProvider.factorAt?.(at(f)));
        }
    });
});

describe("用户曲线是否随几何搬移（镜像后端的锁定参数线开关）", () => {
    // 目标曲线在帧 0 与帧 4 处明显不同，用来区分"搬移"与"不搬移"。
    const src = source({
        volume: new Array<number>(8).fill(1),
        dynTarget: [0.8, 0.8, 0.5, 0.5, 0.4, 0.4, 0.4, 0.4],
        dynBaseline: new Array<number>(8).fill(0.5),
    });
    // 帧 [0,2) → [4,6)：纯移动。
    const origin = [geo("c1", 0, 0.02)];
    const moved = [geo("c1", at(4), 0.02)];

    it("锁定参数线关闭：目标曲线留在绝对时间（后端不会搬移它）", () => {
        const map = amplitudeMap(src, () => warpOf(origin, moved, false));
        // 帧 4 的目标仍是 0.4（原位置的值），基线取帧 0 的 0.5 → 0.8。
        expect(map.factorAt?.(at(4))).toBeCloseTo(0.4 / 0.5, 6);
    });

    it("锁定参数线开启：目标曲线跟着 clip 搬移（与后端同一语义）", () => {
        const map = amplitudeMap(src, () => warpOf(origin, moved, true));
        // 帧 4 的目标来自帧 0 的 0.8，基线同样来自帧 0 的 0.5 → 1.6。
        expect(map.factorAt?.(at(4))).toBeCloseTo(0.8 / 0.5, 6);
    });

    it("锁定参数线开启：被搬走的旧范围恢复为 pad（音量 1.0）", () => {
        const volumeSrc = source({
            volume: [2, 2, 2, 2, 1, 1, 1, 1],
            dynTarget: [],
            dynBaseline: [],
        });
        const map = amplitudeMap(volumeSrc, () => warpOf(origin, moved, true));
        // 新范围 [4,6)：音量取自帧 0..1（= 2）。
        expect(map.factorAt?.(at(4))).toBeCloseTo(2, 6);
        // 旧范围 [0,2) 未被新范围覆盖 → pad → 音量归 1。
        expect(map.factorAt?.(at(0))).toBeCloseTo(1, 6);
        // 既不在旧范围也不在新范围 → 恒等。
        expect(map.factorAt?.(at(6))).toBeCloseTo(1, 6);
    });

    it("★ Slip：曲线**不**跟着搬（后端对 Slip 不做参数线重映射）", () => {
        // 单调目标曲线：曲线若被搬移，取值会明显不同（可判别）。
        const slipSrc = source({
            volume: new Array<number>(8).fill(1),
            dynTarget: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            dynBaseline: new Array<number>(8).fill(0.5),
        });
        // 同一时间轴区间，源窗口平移 0.01 s（1 帧）⇒ 内容整体提前 1 帧。
        const base = [geo("c1", 0, 0.08, { sourceStartSec: 0, sourceEndSec: 0.08 })];
        const slipped = [geo("c1", 0, 0.08, { sourceStartSec: 0.01, sourceEndSec: 0.09 })];
        for (const lock of [false, true]) {
            const map = amplitudeMap(slipSrc, () => warpOf(base, slipped, lock));
            // 帧 4：基线来自帧 5（= 0.5，内容搬了）；目标仍是该帧画的值 0.6（曲线没搬）。
            expect(map.factorAt?.(at(4))).toBeCloseTo(0.6 / 0.5, 6);
        }
    });
});

describe("增益变化（基线含静态 clip 增益）", () => {
    const baseline = new Array<number>(8).fill(0.5);

    it("未画帧：增益变化后仍为「不改变」（目标跟随缩放后的原声）", () => {
        const src = source({
            volume: new Array<number>(8).fill(1),
            dynTarget: [...baseline],
            dynBaseline: baseline,
        });
        const map = amplitudeMap(src, () =>
            warpOf([geo("c1", 0, 0.08, { gain: 1 })], [geo("c1", 0, 0.08, { gain: 2 })]),
        );
        // 基线 ×2 → 1.0；未画帧的目标也取 1.0 ⇒ 增益 1（后端语义：未画 = 不改变）。
        expect(map.factorAt?.(at(2))).toBeCloseTo(1, 6);
    });

    it("画过的帧：目标不随增益缩放（它是绝对电平），增益按新基线兑现", () => {
        const src = source({
            volume: new Array<number>(8).fill(1),
            dynTarget: new Array<number>(8).fill(0.8),
            dynBaseline: baseline,
        });
        const map = amplitudeMap(src, () =>
            warpOf([geo("c1", 0, 0.08, { gain: 1 })], [geo("c1", 0, 0.08, { gain: 2 })]),
        );
        // 目标 0.8（绝对） / 新基线 1.0 = 0.8。
        expect(map.factorAt?.(at(2))).toBeCloseTo(0.8, 6);
    });
});

describe("手势几何 → 映射的整链", () => {
    it("★ 从「按下时 / 当前」几何构造出的映射，能让波形保持形状", () => {
        const loud = new Array<number>(16).fill(0.5);
        const silent = new Array<number>(32).fill(0.0001);
        const baseline = [...loud, ...silent];
        const src = source({
            volume: new Array<number>(48).fill(1),
            dynTarget: [...baseline],
            dynBaseline: baseline,
        });

        const warp = warpOf([geo("c1", 0, 0.16)], [geo("c1", 0.16, 0.16)]);
        expect(warp).not.toBeNull();
        const map = amplitudeMap(src, () => warp);
        expect(map.factorAt?.(at(24))).toBeCloseTo(1, 6);
        // 未受影响区域（新范围之外）保持恒等：那里没有素材被搬动。
        const plain = amplitudeMap(src, () => null);
        expect(map.factorAt?.(at(40))).toBe(plain.factorAt?.(at(40)));
    });

    it("★ 延伸：新露出的帧不再被旧快照的残留基线压平（宣告未知 ⇒ 不施加动态增益）", () => {
        const loud = new Array<number>(16).fill(0.5);
        const silent = new Array<number>(32).fill(0.0001);
        const baseline = [...loud, ...silent];
        const src = source({
            volume: new Array<number>(48).fill(1),
            dynTarget: [...baseline],
            dynBaseline: baseline,
        });
        // 右边延伸：旧 [0,16) → 新 [0,32)。
        const warp = warpOf([geo("c1", 0, 0.16)], [geo("c1", 0, 0.32)]);
        const map = amplitudeMap(src, () => warp);
        // 保留区（帧 8）恒等 ⇒ 未画帧增益 1。
        expect(map.factorAt?.(at(8))).toBeCloseTo(1, 6);
        // 新露出区（帧 24）：旧快照那里是静音；未知语义下按"无动态增益"处理 ⇒ 1，
        // 而不是被"无内容淡出"压成 0。
        expect(map.factorAt?.(at(24))).toBeCloseTo(1, 6);
    });
});
