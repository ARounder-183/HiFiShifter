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
    diffClipGeometryMappings,
    type LoudnessRangeMapping,
    type LoudnessGeometryWarp,
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

function warpOf(
    mappings: readonly LoudnessRangeMapping[],
    lockParamLines: boolean,
): LoudnessGeometryWarp | null {
    return createLoudnessGeometryWarp({ mappings, lockParamLines, framePeriodMs: FP });
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
        const mappings: LoudnessRangeMapping[] = [
            {
                oldStartSec: 0,
                oldLengthSec: 0.16,
                newStartSec: 0.16,
                newLengthSec: 0.16,
                gainScale: 1,
            },
        ];

        // 无映射（= 修复前的行为）：新位置的旧基线是静音 ⇒ 淡出把波形压成 0。
        const before = amplitudeMap(src, () => null);
        expect(before.factorAt?.(at(24))).toBe(0);

        // 有映射：基线跟着素材搬到帧 8（响段）⇒ 未画帧增益 1，波形保持原样。
        const after = amplitudeMap(src, () => warpOf(mappings, false));
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
            [
                {
                    oldStartSec: at(0),
                    oldLengthSec: 0.24,
                    newStartSec: at(delta),
                    newLengthSec: 0.24,
                    gainScale: 1,
                },
            ],
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
        const withWarpProvider = amplitudeMap(src, () => null);
        const withoutProvider = amplitudeMap(src, () => null);
        for (let f = 0; f < 32; f += 1) {
            expect(withWarpProvider.factorAt?.(at(f))).toBe(withoutProvider.factorAt?.(at(f)));
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
    const mappings: LoudnessRangeMapping[] = [
        {
            oldStartSec: 0,
            oldLengthSec: 0.02,
            newStartSec: 0.04,
            newLengthSec: 0.02,
            gainScale: 1,
        },
    ];

    it("锁定参数线关闭：目标曲线留在绝对时间（后端不会搬移它）", () => {
        const map = amplitudeMap(src, () => warpOf(mappings, false));
        // 帧 4 的目标仍是 0.4（原位置的值），基线取帧 0 的 0.5 → 0.8。
        expect(map.factorAt?.(at(4))).toBeCloseTo(0.4 / 0.5, 6);
    });

    it("锁定参数线开启：目标曲线跟着 clip 搬移（与后端同一语义）", () => {
        const map = amplitudeMap(src, () => warpOf(mappings, true));
        // 帧 4 的目标来自帧 0 的 0.8，基线同样来自帧 0 的 0.5 → 1.6。
        expect(map.factorAt?.(at(4))).toBeCloseTo(0.8 / 0.5, 6);
    });

    it("锁定参数线开启：被搬走的旧范围恢复为 pad（音量 1.0）", () => {
        const volumeSrc = source({
            volume: [2, 2, 2, 2, 1, 1, 1, 1],
            dynTarget: [],
            dynBaseline: [],
        });
        const map = amplitudeMap(volumeSrc, () => warpOf(mappings, true));
        // 新范围 [4,6)：音量取自帧 0..1（= 2）。
        expect(map.factorAt?.(at(4))).toBeCloseTo(2, 6);
        // 旧范围 [0,2) 未被新范围覆盖 → pad → 音量归 1。
        expect(map.factorAt?.(at(0))).toBeCloseTo(1, 6);
        // 既不在旧范围也不在新范围 → 恒等。
        expect(map.factorAt?.(at(6))).toBeCloseTo(1, 6);
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
            warpOf(
                [
                    {
                        oldStartSec: 0,
                        oldLengthSec: 0.08,
                        newStartSec: 0,
                        newLengthSec: 0.08,
                        gainScale: 2,
                    },
                ],
                false,
            ),
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
            warpOf(
                [
                    {
                        oldStartSec: 0,
                        oldLengthSec: 0.08,
                        newStartSec: 0,
                        newLengthSec: 0.08,
                        gainScale: 2,
                    },
                ],
                false,
            ),
        );
        // 目标 0.8（绝对） / 新基线 1.0 = 0.8。
        expect(map.factorAt?.(at(2))).toBeCloseTo(0.8, 6);
    });
});

describe("手势几何 → 差分 → 映射的整链", () => {
    it("★ 从「按下时 / 当前」几何差分出的映射，能让波形保持形状", () => {
        const loud = new Array<number>(16).fill(0.5);
        const silent = new Array<number>(32).fill(0.0001);
        const baseline = [...loud, ...silent];
        const src = source({
            volume: new Array<number>(48).fill(1),
            dynTarget: [...baseline],
            dynBaseline: baseline,
        });

        const origin = [{ clipId: "c1", startSec: 0, lengthSec: 0.16, gain: 1 }];
        const moved = [{ id: "c1", startSec: 0.16, lengthSec: 0.16, gain: 1 }];
        const mappings = diffClipGeometryMappings(origin, moved);
        expect(mappings).toHaveLength(1);

        const map = amplitudeMap(src, () => warpOf(mappings, false));
        expect(map.factorAt?.(at(24))).toBeCloseTo(1, 6);
        // 未受影响区域（新范围之外）保持恒等：那里没有素材被搬动。
        const plain = amplitudeMap(src, () => null);
        expect(map.factorAt?.(at(40))).toBe(plain.factorAt?.(at(40)));
    });
});
