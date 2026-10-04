/**
 * 幅度映射「查询窗口查表 vs 逐值求值」等价性回归。
 *
 * ## 为什么必须有这份测试
 *
 * `makeLoudnessAmplitudeMap` 是**纯乘性**的时域因子视图，几何层在一次重建里
 * 按「像素列 × 列内切片」向它查询数万次（`factorAt` 与 `levelCeilingOverWindow`）。
 * 为把这些查询从"每次重活"变成"每次查表"，实现里加了一条**并行路径**：
 *
 * - 逐值路径（`factorAtDirect` / 逐帧枚举）：把曲线在每个查询点上采样；
 * - 查表路径（`beginWindow` + 整数帧表 + 帧内线性插值）。
 *
 * 两条路径**必须逐值等价** —— 只要有一处偏差，拖动音量/动态时波形的高度就会
 * 与曲线对不上，且随缩放随机漂移。本测试用同一份数据同时构造两个映射实例，
 * 一个调用 `beginWindow`（走查表）、一个不调用（走逐值），在覆盖各种边界形状的
 * 查询点上逐值对拍：
 *
 * 1. live 覆盖窗口内部（查表走 live 表）；
 * 2. live 覆盖窗口外部（查表走整工程快照表）；
 * 3. **跨 live / 快照边界的小数帧**（一格的两端点来自不同数据路 —— 查表必须
 *    识别出"这一格不可插值"并回退逐值，否则会把两路数据混着插值）；
 * 4. 曲线数组范围之外（两侧都取不到 → 回退语义）；
 * 5. 查询点落在查表窗口之外（必须回退逐值，不得返回 null 让几何层丢列）。
 *
 * ## 还断言"查表真的被走了"
 *
 * 等价性测试若两条路径都实际走了逐值，会"空转通过"。因此额外统计 live 覆盖
 * provider 的调用次数：查表路径把"解析 live 覆盖"从**每个查询点**收敛为
 * **每次重建一次**（`beginWindow` 时），因此建表后查询期间 provider 必须**零调用**。
 * 这既是等价性测试的"有牙"证据，也正是本优化的核心机制。
 */
import { describe, expect, test } from "vitest";

import { createLiveOverrideReader } from "./liveLoudnessOverride";
import { makeLoudnessAmplitudeMap } from "./PianoRollWaveformSurface";
import { createLoudnessGeometryWarp, type LoudnessGeometryWarp, type WarpClipGeometry } from "./loudnessGeometryWarp";
import type { WaveformAmplitudeFactors, WaveformAmplitudeMap } from "../../../waveform/geometry";

const FRAME_MS = 5;
/** 整工程快照：帧 0..999（0..5s）。 */
const SNAPSHOT_FRAMES = 1000;
/** live 覆盖：帧 100..399（0.5..2.0s）——特意只覆盖窗口的一段。 */
const LIVE_START_FRAME = 100;
const LIVE_FRAME_COUNT = 300;

function makeCurves(stride = 1): {
    volume: number[];
    dynTarget: number[];
    dynBaseline: number[];
} {
    const count = Math.floor((SNAPSHOT_FRAMES - 1) / stride) + 1;
    const volume: number[] = new Array(count);
    const dynTarget: number[] = new Array(count);
    const dynBaseline: number[] = new Array(count);
    for (let i = 0; i < count; i += 1) {
        const f = i * stride;
        volume[i] = 0.5 + 0.5 * Math.sin(f * 0.031);
        // 【为什么基线要有一段持续处于淡出带】`dynContentFade` 的 smoothstep 过渡带
        // 是 [0.0005, 0.001]，但 `contentBaselineAt` 先做约 ±50ms 的平滑，**短于
        // 平滑窗**的下探会被邻近的大基线拉回带外、淡出恒为 1。旧夹具最小 0.02，
        // 因此"查表对淡出线性插值还是按就近采样帧阶跃"这一分歧永远不会被对拍发现。
        // 这里让基线在 [140, 220] 帧（远宽于平滑窗）内处于带内并缓慢变化，使淡出
        // 在 0..1 之间真实起伏。
        const inBand = f >= 140 && f <= 220;
        dynBaseline[i] = inBand
            ? 0.0005 + 0.0005 * (Math.abs(f - 180) / 40)
            : 0.02 + 0.4 * Math.abs(Math.sin(f * 0.017));
        dynTarget[i] = dynBaseline[i] * (0.2 + 1.8 * Math.abs(Math.cos(f * 0.011)));
    }
    return { volume, dynTarget, dynBaseline };
}

/** 一个映射实例 + 它的 provider 调用计数。 */
function makeInstrumentedMap(args: {
    snapshot: {
        volume: number[];
        dynTarget: number[];
        dynBaseline: number[];
    };
    liveEdit: number[] | null;
    /** 快照步长（默认 1；>1 时曲线数组按该步长解释绝对帧）。 */
    stride?: number;
    /** 拖拽期间的几何时域映射（缺省 = 无映射）。 */
    warp?: () => LoudnessGeometryWarp | null;
}): { factors: WaveformAmplitudeFactors; calls: () => number } {
    const stride = args.stride ?? 1;
    const reader = createLiveOverrideReader();
    const live =
        args.liveEdit === null
            ? null
            : {
                  key: `v2|track-1|volume|${LIVE_START_FRAME}|${LIVE_FRAME_COUNT}|${stride}`,
                  edit: args.liveEdit,
              };
    let calls = 0;
    const map = makeLoudnessAmplitudeMap(
        {
            startFrame: 0,
            stride,
            framePeriodMs: FRAME_MS,
            volume: args.snapshot.volume,
            dynTarget: args.snapshot.dynTarget,
            dynBaseline: args.snapshot.dynBaseline,
        },
        {
            volume: () => {
                calls += 1;
                return reader.read("volume", live);
            },
            dyn: () => {
                calls += 1;
                return reader.read("dyn", live);
            },
        },
        () => 0,
        args.warp,
    );
    return {
        factors: map as unknown as WaveformAmplitudeFactors,
        calls: () => calls,
    };
}

/** 覆盖各种边界形状的查询时刻（秒）——**故意含大量小数帧**。 */
function queryTimes(): number[] {
    const out: number[] = [];
    // 0.4s..2.1s 以 0.37 帧为步长扫过（跨 live 边界的每一格都会命中）。
    for (let f = 80; f <= 420; f += 0.37) out.push((f * FRAME_MS) / 1000);
    // 边界附近逐格细扫（含正好落在整数帧上）。
    for (let f = 96; f <= 104; f += 0.25) out.push((f * FRAME_MS) / 1000);
    for (let f = 396; f <= 404; f += 0.25) out.push((f * FRAME_MS) / 1000);
    // 曲线范围之外（快照 0..999 帧之外）与负数。
    out.push(0, -1, 5.0, 5.5, 6.0, 12.0);
    return out;
}

/** 上界钳制的查询窗口：宽度取一个 L0 桶（≈0.36ms）到数十 ms 不等。 */
function ceilingWindows(): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    for (let f = 80; f <= 420; f += 1.7) {
        const lo = (f * FRAME_MS) / 1000;
        out.push([lo, lo + 0.0004]);
        out.push([lo, lo + 0.009]);
        out.push([lo, lo + 0.05]);
    }
    out.push([0.6, 0.4]);
    return out;
}

describe("查表路径与逐值路径逐值等价", () => {
    test("factorAt / levelCeilingOverWindow / 逐值映射 三者全等", () => {
        const snapshot = makeCurves();
        const liveEdit: number[] = new Array(LIVE_FRAME_COUNT);
        for (let i = 0; i < LIVE_FRAME_COUNT; i += 1) {
            // live 覆盖里既有"用户画过的"帧，也有与快照不一致的帧，
            // 这样"跨边界插值"若发生就会立刻被对拍发现。
            liveEdit[i] = 0.3 + 1.2 * Math.abs(Math.sin(i * 0.07));
        }

        const lutMap = makeInstrumentedMap({ snapshot, liveEdit });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit });

        // 只给查表实例声明窗口：另一个实例因此完全走逐值路径。
        const times = queryTimes();
        let lo = Number.POSITIVE_INFINITY;
        let hi = Number.NEGATIVE_INFINITY;
        for (const t of times) {
            if (t < lo) lo = t;
            if (t > hi) hi = t;
        }
        lutMap.factors.beginWindow?.(lo, hi);

        const lutCallsAfterBuild = lutMap.calls();
        const directCallsBefore = directMap.calls();
        expect(lutCallsAfterBuild).toBeGreaterThan(0);

        for (const t of times) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
        // 逐值映射（几何层在因子为 null 时回退到的那条路径）同样必须等价。
        for (const t of times) {
            expect((lutMap.factors as unknown as WaveformAmplitudeMap)(0.5, 2, t)).toBe(
                (directMap.factors as unknown as WaveformAmplitudeMap)(0.5, 2, t),
            );
        }
        for (const [a, b] of ceilingWindows()) {
            expect(lutMap.factors.levelCeilingOverWindow?.(a, b)).toBe(
                directMap.factors.levelCeilingOverWindow?.(a, b),
            );
        }

        // ★ 有牙断言：查表路径的 provider 调用**不随查询次数增长** —— 这正是优化
        // 机制（把每个查询点的 live 覆盖解析收敛为每次重建一次）。
        //
        // 残余的少量调用是**必须**回退的查询点，来自两类边界（精确性要求它们
        // 走逐值路径，见 LUT_SRC_* 的说明）：
        //   ① live / 快照两路交界的那两格（帧 99 与 399 各一格）；
        //   ② 落在声明窗口之外的探测点（t = -1s）。
        // 因此这里断言的是"增长量远小于查询数"，而不是恒等于建表时的计数 ——
        // 后者会把上面这两类合法回退误判为缺陷。
        const lutDelta = lutMap.calls() - lutCallsAfterBuild;
        const directDelta = directMap.calls() - directCallsBefore;
        expect(times.length).toBeGreaterThan(900);
        expect(lutDelta).toBeLessThan(200);
        expect(directDelta).toBeGreaterThan(2000);
        expect(lutDelta * 10).toBeLessThan(directDelta);
    });

    test("无 live 覆盖（纯快照）时同样全等", () => {
        const snapshot = makeCurves();
        const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const times = queryTimes();
        lutMap.factors.beginWindow?.(0.4, 2.1);
        for (const t of times) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
        for (const [a, b] of ceilingWindows()) {
            expect(lutMap.factors.levelCeilingOverWindow?.(a, b)).toBe(
                directMap.factors.levelCeilingOverWindow?.(a, b),
            );
        }
    });

    /** 一份 clip 几何（源窗口默认 = [0, 长度×速率)，即"内容跨度守恒"）。 */
    interface ClipSpec {
        startSec: number;
        lengthSec: number;
        playbackRate?: number;
        gain?: number;
    }

    function clipOf(spec: ClipSpec, id: string): WarpClipGeometry {
        const rate = spec.playbackRate ?? 1;
        return {
            id,
            startSec: spec.startSec,
            lengthSec: spec.lengthSec,
            gain: spec.gain ?? 1,
            sourceStartSec: 0,
            sourceEndSec: spec.lengthSec * rate,
            playbackRate: rate,
            reversed: false,
            loopEnabled: false,
        };
    }

    /** 由若干「旧几何 → 新几何」的 clip 对构造映射（时间按 FRAME_MS 帧栅格）。 */
    function warpFrom(
        pairs: ReadonlyArray<readonly [ClipSpec, ClipSpec]>,
        lockParamLines: boolean,
    ): LoudnessGeometryWarp | null {
        return createLoudnessGeometryWarp({
            origin: pairs.map(([old], i) => clipOf(old, `c${i}`)),
            clips: pairs.map(([, next], i) => clipOf(next, `c${i}`)),
            framePeriodMs: FRAME_MS,
            lockParamLines,
        });
    }

    /**
     * 拖拽期间的几何时域映射夹具：一平移、一"压缩 + 增益"（速率 ×2、增益 ×1.25），
     * 且平移段的旧范围部分未被覆盖（形成空档）—— 覆盖映射的全部代码路径。
     *
     * 帧 ↔ 秒：帧号 × FRAME_MS。故帧 100 = 0.5 s、帧 600 = 3.0 s。
     */
    function warpFixture(lockParamLines: boolean): LoudnessGeometryWarp | null {
        return warpFrom(
            [
                // 平移：帧 100..400 → 帧 500..800（纯移动，源窗口不变）。
                [{ startSec: 0.5, lengthSec: 1.5 }, { startSec: 2.5, lengthSec: 1.5 }],
                // 压缩 + 增益：帧 600..1000 → 帧 200..400（速率 1→2，内容跨度守恒）。
                [
                    { startSec: 3.0, lengthSec: 2.0, gain: 1 },
                    { startSec: 1.0, lengthSec: 1.0, playbackRate: 2, gain: 1.25 },
                ],
            ],
            lockParamLines,
        );
    }

    /**
     * 只含**刚性平移**与**纯增益**的映射 —— 这两种都能保留查表的整数帧线性插值
     *（见 `interpolableAt`），因此可以用来断言"映射下查表仍真的被走了"。
     */
    function translationWarpFixture(lockParamLines: boolean): LoudnessGeometryWarp | null {
        return warpFrom(
            [
                // 平移：帧 100..400 → 帧 500..800。
                [{ startSec: 0.5, lengthSec: 1.5 }, { startSec: 2.5, lengthSec: 1.5 }],
                // 纯增益（时域不变）：帧 600..800，增益 ×1.25。
                [
                    { startSec: 3.0, lengthSec: 1.0, gain: 1 },
                    { startSec: 3.0, lengthSec: 1.0, gain: 1.25 },
                ],
            ],
            lockParamLines,
        );
    }

    /** 映射生效时的查询时刻：覆盖两段的新/旧范围、段边界与空档。 */
    function warpQueryTimes(): number[] {
        const out: number[] = [];
        for (let f = 80; f <= 1010; f += 0.53) out.push((f * FRAME_MS) / 1000);
        // 段边界逐格细扫（含正好落在边界帧上）—— 不可插值的格都在这里。
        for (const edge of [100, 200, 400, 500, 600, 800, 1000]) {
            for (let f = edge - 2; f <= edge + 2; f += 0.25) {
                out.push((f * FRAME_MS) / 1000);
            }
        }
        out.push(0, -1, 5.0, 5.5, 6.0);
        return out;
    }

    test("★ 几何映射（含裁切 / 拉伸的仿射重采样）下查表与逐值仍逐值等价", () => {
        for (const lockParamLines of [false, true]) {
            const snapshot = makeCurves();
            const lutMap = makeInstrumentedMap({
                snapshot,
                liveEdit: null,
                warp: () => warpFixture(lockParamLines),
            });
            const directMap = makeInstrumentedMap({
                snapshot,
                liveEdit: null,
                warp: () => warpFixture(lockParamLines),
            });
            const times = warpQueryTimes();
            lutMap.factors.beginWindow?.(0, 5.5);

            for (const t of times) {
                expect(
                    lutMap.factors.factorAt(t),
                    `lock=${lockParamLines} t=${t}`,
                ).toBeCloseTo(directMap.factors.factorAt(t) as number, 12);
            }
            for (const [a, b] of ceilingWindows()) {
                expect(
                    lutMap.factors.levelCeilingOverWindow?.(a, b),
                    `lock=${lockParamLines} window=[${a},${b}]`,
                ).toBeCloseTo(directMap.factors.levelCeilingOverWindow?.(a, b) as number, 12);
            }
        }
    });

    test("★ 刚性平移下查表仍真的被走了（只有段边界附近回退）", () => {
        // 移动是最常见的拖拽手势，必须仍然走查表 —— 否则等于把拖拽期间的单帧成本
        // 抬回"逐值求值"的量级（见 factorAt 的说明）。
        const snapshot = makeCurves();
        const lutMap = makeInstrumentedMap({
            snapshot,
            liveEdit: null,
            warp: () => translationWarpFixture(false),
        });
        const directMap = makeInstrumentedMap({
            snapshot,
            liveEdit: null,
            warp: () => translationWarpFixture(false),
        });
        const times = warpQueryTimes();
        lutMap.factors.beginWindow?.(0, 5.5);
        const lutCallsAfterBuild = lutMap.calls();
        const directCallsBefore = directMap.calls();

        for (const t of times) {
            expect(lutMap.factors.factorAt(t)).toBeCloseTo(
                directMap.factors.factorAt(t) as number,
                12,
            );
        }

        const lutDelta = lutMap.calls() - lutCallsAfterBuild;
        const directDelta = directMap.calls() - directCallsBefore;
        expect(times.length).toBeGreaterThan(1500);
        expect(lutDelta).toBeLessThan(200);
        expect(lutDelta * 5).toBeLessThan(directDelta);
    });

    test("★ 映射对象更换必须让查表重建（同一快照、同一修订号）", () => {
        // 拖拽期间几何逐帧变化、映射对象随之逐帧更换，而快照引用与修订号都不变。
        // 查表若只看后两者，就会一直复用拖拽第一帧的表 —— 波形停在那一刻。
        const snapshot = makeCurves();
        let warp = warpFixture(false);
        const map = makeInstrumentedMap({
            snapshot,
            liveEdit: null,
            warp: () => warp,
        });
        map.factors.beginWindow?.(0, 5.5);
        const before = map.factors.factorAt(1.05) as number;

        // 换一份映射（把平移目标挪到别处），重建后取值必须随之改变。
        warp = warpFrom(
            [[{ startSec: 0.5, lengthSec: 1.5 }, { startSec: 1.0, lengthSec: 1.5 }]],
            false,
        );
        map.factors.beginWindow?.(0, 5.5);
        const after = map.factors.factorAt(1.05) as number;
        expect(after).not.toBeCloseTo(before, 6);
    });

    test("★ 快照降采样（stride > 1）下同样全等 —— 淡出按就近采样帧阶跃", () => {
        // 长工程快照会按 stride > 1 降采样。此时逐值路径的淡出仍是"就近取整到
        // 采样帧再套 smoothstep"的阶跃函数，且阶跃点不落在格中点（如 stride=4
        // 时落在格边界），只按 `frac < 0.5` 的简化实现会与逐值路径分叉。
        for (const stride of [2, 3, 4]) {
            const snapshot = makeCurves(stride);
            const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null, stride });
            const directMap = makeInstrumentedMap({ snapshot, liveEdit: null, stride });
            const times = queryTimes();
            lutMap.factors.beginWindow?.(0.4, 2.1);
            for (const t of times) {
                // stride > 1 时查表在整数帧上再插值一次，与逐值的一次插值数学等价、
                // 但浮点运算顺序不同（末位 ULP 级差异），故这里按精度对拍；淡出的
                // 阶跃语义若写错（如简化为 `frac < 0.5`），差异远超该精度会立刻失败。
                expect(lutMap.factors.factorAt(t), `stride=${stride} t=${t}`).toBeCloseTo(
                    directMap.factors.factorAt(t) as number,
                    12,
                );
            }
            for (const [a, b] of ceilingWindows()) {
                expect(
                    lutMap.factors.levelCeilingOverWindow?.(a, b),
                    `stride=${stride} window=[${a},${b}]`,
                ).toBe(directMap.factors.levelCeilingOverWindow?.(a, b));
            }
        }
    });

    test("无动态基线（分析未就绪）时不钳制且因子只含音量", () => {
        const snapshot = makeCurves();
        snapshot.dynBaseline = [];
        snapshot.dynTarget = [];
        const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        lutMap.factors.beginWindow?.(0.4, 2.1);
        for (const t of queryTimes()) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
        for (const [a, b] of ceilingWindows()) {
            expect(lutMap.factors.levelCeilingOverWindow?.(a, b)).toBe(null);
            expect(directMap.factors.levelCeilingOverWindow?.(a, b)).toBe(null);
        }
    });

    test("查询点全部落在声明窗口之外时逐值回退仍然正确", () => {
        const snapshot = makeCurves();
        const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        // 声明一个与查询范围完全不相交的窗口 ⇒ 查表必然未命中。
        lutMap.factors.beginWindow?.(4.9, 5.0);
        for (const t of queryTimes()) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
    });

    test("窗口声明顺序颠倒（end < start）不影响结果", () => {
        const snapshot = makeCurves();
        const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        lutMap.factors.beginWindow?.(2.1, 0.4);
        for (const t of queryTimes()) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
        for (const [a, b] of ceilingWindows()) {
            expect(lutMap.factors.levelCeilingOverWindow?.(a, b)).toBe(
                directMap.factors.levelCeilingOverWindow?.(a, b),
            );
        }
    });

    test("重复 beginWindow（含窗口扩张）后仍然全等", () => {
        const snapshot = makeCurves();
        const lutMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        const directMap = makeInstrumentedMap({ snapshot, liveEdit: null });
        lutMap.factors.beginWindow?.(0.45, 1.0);
        lutMap.factors.beginWindow?.(0.42, 1.5);
        lutMap.factors.beginWindow?.(0.4, 2.1);
        for (const t of queryTimes()) {
            expect(lutMap.factors.factorAt(t)).toBe(directMap.factors.factorAt(t));
        }
    });
});
