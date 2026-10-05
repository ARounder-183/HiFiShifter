/**
 * 拖拽期间响度时域映射的契约回归。
 *
 * 【为什么单测它】这是"拖拽中波形错位"修复的**唯一数学落点**。失败模式只在真实
 * 拖拽里才看得见，且都表现为"波形看起来怪"而不是报错：
 *
 * - 映射算错 → 波形在拖拽中整段错位（本 bug 本身）；
 * - 该恒等的地方不恒等 → 没动的区域也跟着变形（延伸/截短被当成拉伸）；
 * - 该搬的地方没搬 → Slip 期间仍是「旧内容的基线 × 新内容的峰值」；
 * - 用户曲线的搬移策略与后端不一致 → 松手后曲线"跳回去"。
 *
 * 这里逐条钉住四种手势的解析解（见 `loudnessGeometryWarp` 文件头的推导），
 * 以及"未知基线"语义、增益比、pad、锁定参数线门控。
 */
import { describe, expect, it } from "vitest";

import {
    createLoudnessGeometryWarp,
    resolveClipConsumption,
    WARP_BASELINE_UNKNOWN,
    WARP_CURVE_PAD,
    WARP_SEGMENT_IDENTITY,
    type WarpClipGeometry,
} from "./loudnessGeometryWarp";
import { makeLoudnessAmplitudeMap } from "./PianoRollWaveformSurface";
import type { WaveformAmplitudeFactors } from "../../../waveform/geometry";

/** 帧周期：与工程默认一致（5 ms）→ 每秒 200 帧。 */
const FP = 5;
const FPS = 1000 / FP;

interface GeoOpts {
    gain?: number;
    sourceStartSec?: number;
    sourceEndSec?: number;
    playbackRate?: number;
    reversed?: boolean;
    loopEnabled?: boolean;
    /** 媒体总时长（Loop 回绕周期 D）。给了它才走"锚点回绕"路径。 */
    mediaDurationSec?: number;
}

/** 构造一份 clip 几何（默认正放、非 Loop、源窗口 = [0, 长度×速率)）。 */
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
        ...(opts.mediaDurationSec === undefined
            ? {}
            : { sourcePath: "/media.wav", durationSec: opts.mediaDurationSec }),
    };
}

/**
 * 后端 Loop 消费的**逐帧参考实现**。
 *
 * 同一公式出现在三处且必须逐帧一致：`pitch_clip::trim_and_resample_midi` 的 loop
 * 分支、`pitch_analysis/schedule.rs`、`audio/mixdown.rs` 的 `floor_mod` 锚点回绕：
 *
 *   正放 idx(i) = rem_euclid(round(source_start·fps) + round(i·rate), n)
 *   倒放 idx(i) = rem_euclid(round(min(source_end, D)·fps) − 1 − round(i·rate), n)
 *
 * `i = frame − round(start·fps)`（clip 内时间线帧下标）、`n = round(D·fps)`。
 * 返回**未回绕**的源位置（帧域）：对拍时对两边施加同一取模即可。
 */
function backendLoopSourcePos(
    g: WarpClipGeometry,
    frameF: number,
    fps: number,
    mediaDurationSec: number,
): number {
    const startF = Math.round(g.startSec * fps);
    const rate = g.playbackRate ?? 1;
    const i = frameF - startF;
    if (g.reversed === true) {
        const anchorR = Math.round(Math.min(g.sourceEndSec ?? 0, mediaDurationSec) * fps) - 1;
        return anchorR - i * rate;
    }
    const anchorF = Math.round((g.sourceStartSec ?? 0) * fps);
    return anchorF + i * rate;
}

/** 两个源位置在周期 `period` 下的最小环形距离（帧）。 */
function circularDistance(a: number, b: number, period: number): number {
    const raw = (((a - b) % period) + period) % period;
    return Math.min(raw, period - raw);
}

function warpOf(
    origin: readonly WarpClipGeometry[],
    clips: readonly WarpClipGeometry[],
    lockParamLines = false,
) {
    return createLoudnessGeometryWarp({
        origin,
        clips,
        framePeriodMs: FP,
        lockParamLines,
    });
}

describe("resolveClipConsumption", () => {
    it("正放：锚点在 sourceStart，速率为正", () => {
        const c = resolveClipConsumption(geo("c", 1, 2, { sourceStartSec: 3 }), FPS)!;
        expect(c.startF).toBe(200);
        expect(c.lenF).toBe(400);
        expect(c.rate).toBe(1);
        expect(c.anchorF).toBeCloseTo(600, 9);
        expect(c.exact).toBe(true);
    });

    it("倒放：消费窗口重定向到 sourceEnd 侧，速率为负（与后端 clip_pitch_trim_window_sec 同构）", () => {
        const c = resolveClipConsumption(
            geo("c", 0, 2, { sourceStartSec: 2, sourceEndSec: 4, reversed: true }),
            FPS,
        )!;
        // 窗口 [se−len·rate, se] = [2, 4] 升序消费后整体翻转 ⇒ 第 0 帧消费源 4s。
        expect(c.rate).toBe(-1);
        expect(c.anchorF).toBeCloseTo(799, 9);
        expect(c.exact).toBe(true);
    });

    it("Loop + 媒体时长未知：无法表达回绕，退化为正放仿射", () => {
        const c = resolveClipConsumption(
            geo("c", 1, 2, { sourceStartSec: 0.5, loopEnabled: true }),
            FPS,
        )!;
        expect(c.exact).toBe(false);
        expect(c.rate).toBe(1);
    });

    it("速率非正 / 帧周期非正时不可解析", () => {
        expect(resolveClipConsumption(geo("c", 0, 1, { playbackRate: 0 }), FPS)).toBeNull();
        expect(resolveClipConsumption(geo("c", 0, 1), 0)).toBeNull();
    });
});

describe("移动（纯平移）", () => {
    // 0..2 s → 2..4 s（帧 0..400 → 400..800）。
    const originGeo = geo("c1", 0, 2);
    const moved = geo("c1", 2, 2);
    const warp = warpOf([originGeo], [moved])!;

    it("新范围内的帧映射回旧位置（逐帧精确平移）", () => {
        for (let k = 0; k < 400; k += 1) {
            expect(warp.baselineFrame(400 + k)).toBe(k);
        }
    });

    it("新范围外的帧保持恒等（未受影响区域不变形）", () => {
        expect(warp.baselineFrame(0)).toBe(0);
        expect(warp.baselineFrame(399)).toBe(399);
        expect(warp.baselineFrame(800)).toBe(800);
        expect(warp.baselineFrame(1234)).toBe(1234);
    });

    it("映射段编号只在新区间内出现", () => {
        expect(warp.baselineSegment(399)).toBe(WARP_SEGMENT_IDENTITY);
        expect(warp.baselineSegment(400)).toBe(0);
        expect(warp.baselineSegment(799)).toBe(0);
        expect(warp.baselineSegment(800)).toBe(WARP_SEGMENT_IDENTITY);
    });

    it("平移可走查表的整数帧线性插值", () => {
        expect(warp.interpolableAt(500)).toBe(true);
        expect(warp.interpolableAt(0)).toBe(true);
    });

    it("锁定参数线关闭时用户曲线恒等（后端不会搬移它们）", () => {
        for (const f of [0, 100, 400, 600, 900]) {
            expect(warp.curveFrame(f)).toBe(f);
            expect(warp.curveSegment(f)).toBe(WARP_SEGMENT_IDENTITY);
        }
    });

    it("锁定参数线开启时曲线与基线同一条平移（后端 move_clips 携带参数线）", () => {
        const locked = warpOf([originGeo], [moved], true)!;
        for (let k = 0; k < 400; k += 1) {
            expect(locked.curveFrame(400 + k)).toBe(k);
        }
        expect(locked.curveSegment(400)).toBe(0);
    });

    it("锁定参数线开启时旧范围中未被覆盖的帧恢复为 pad", () => {
        const locked = warpOf([originGeo], [moved], true)!;
        expect(locked.curveFrame(0)).toBe(WARP_CURVE_PAD);
        expect(locked.curveFrame(399)).toBe(WARP_CURVE_PAD);
        expect(locked.curveSegment(200)).toBe(WARP_CURVE_PAD);
        // 既不在旧范围也不在新范围：恒等。
        expect(locked.curveFrame(800)).toBe(800);
        expect(locked.curveSegment(900)).toBe(WARP_SEGMENT_IDENTITY);
    });
});

describe("Slip（源窗口平移，时间轴与长度不变）", () => {
    // 源窗口右移 0.5 s（= 100 帧）⇒ 内容整体提前 100 帧。
    const originGeo = geo("c1", 1, 2, { sourceStartSec: 0, sourceEndSec: 2 });
    const slipped = geo("c1", 1, 2, { sourceStartSec: 0.5, sourceEndSec: 2.5 });
    const warp = warpOf([originGeo], [slipped])!;

    it("基线整体平移 Δ源/速率（旧内容跟着搬）", () => {
        expect(warp.baselineFrame(200)).toBe(300);
        expect(warp.baselineFrame(300)).toBe(400);
        // 平移量恒为 100 帧。
        for (const f of [200, 250, 300, 400, 499]) {
            expect(warp.baselineFrame(f)).toBe(f + 100);
        }
    });

    it("平移出旧覆盖范围的帧宣告「基线未知」（不能拿别处的值顶替）", () => {
        // 旧范围 [200, 600)：映射后 ≥ 600 的帧不可用。
        expect(warp.baselineFrame(499)).toBe(599);
        expect(warp.baselineFrame(500)).toBe(WARP_BASELINE_UNKNOWN);
        expect(warp.baselineFrame(599)).toBe(WARP_BASELINE_UNKNOWN);
        // 未知帧的增益比也退化为 1。
        expect(warp.baselineScale(500)).toBe(1);
    });

    it("★ 用户曲线**不**跟着搬 —— 后端对 Slip 不做任何参数线重映射", () => {
        for (const lock of [false, true]) {
            const w = warpOf([originGeo], [slipped], lock)!;
            for (const f of [200, 300, 450, 599]) {
                expect(w.curveFrame(f)).toBe(f);
                expect(w.curveSegment(f)).toBe(WARP_SEGMENT_IDENTITY);
            }
        }
    });

    it("平移可走查表（整数帧位移）", () => {
        expect(warp.interpolableAt(300)).toBe(true);
    });
});

describe("拉伸/缩短（Alt + 边缘，速率变、内容跨度不变）", () => {
    // 2 s 正放 → 4 s 半速：内容跨度 len·rate 恒为 2 s。
    const originGeo = geo("c1", 0, 2, { playbackRate: 1, sourceEndSec: 2 });
    const stretched = geo("c1", 0, 4, { playbackRate: 0.5, sourceEndSec: 2 });
    const warp = warpOf([originGeo], [stretched])!;

    it("基线按**内容**仿射缩放（斜率 = 速率比 = 1/2，而非曲线重采样的 (len−1) 比）", () => {
        // 新帧 f 播放的源内容 = 旧帧 0.5f 播放的那段。
        expect(warp.baselineFrame(0)).toBe(0);
        expect(warp.baselineFrame(400)).toBe(200);
        expect(warp.baselineFrame(798)).toBeCloseTo(399, 9);
        // 端点必须留在旧覆盖范围内（内容跨度不变 ⇒ 全程有定义）。
        expect(warp.baselineFrame(799)).toBeLessThan(400);
        expect(warp.baselineFrame(799)).not.toBe(WARP_BASELINE_UNKNOWN);
    });

    it("拉伸全程无「未知」帧（内容跨度守恒）", () => {
        for (const f of [0, 1, 100, 399, 400, 798, 799]) {
            expect(warp.baselineFrame(f)).not.toBe(WARP_BASELINE_UNKNOWN);
        }
    });

    it("不能走查表的整数帧插值（折点会被搬进格内）", () => {
        expect(warp.interpolableAt(400)).toBe(false);
    });

    it("锁定参数线开启时曲线按后端 resample_curve 的 (旧长−1)/(新长−1) 重采样", () => {
        const locked = warpOf([originGeo], [stretched], true)!;
        const ratio = (400 - 1) / (800 - 1);
        for (const f of [0, 100, 399, 799]) {
            expect(locked.curveFrame(f)).toBeCloseTo(f * ratio, 9);
        }
        // 变长方向：新范围完整覆盖旧范围 ⇒ 没有 pad 区。
        for (const f of [0, 400, 799]) {
            expect(locked.curveFrame(f)).not.toBe(WARP_CURVE_PAD);
        }
    });

    it("缩短方向：旧范围尾部不再被覆盖 → pad（后端 restore 阶段写回 pad 值）", () => {
        // 4 s 正放 → 2 s 双速：内容跨度仍为 4 s。
        const longGeo = geo("c1", 0, 4, { playbackRate: 1, sourceEndSec: 4 });
        const shortened = geo("c1", 0, 2, { playbackRate: 2, sourceEndSec: 4 });
        const locked = warpOf([longGeo], [shortened], true)!;
        const ratio = (800 - 1) / (400 - 1);
        expect(locked.curveFrame(0)).toBeCloseTo(0, 9);
        expect(locked.curveFrame(399)).toBeCloseTo(399 * ratio, 9);
        // 旧范围 [0,800) 中 400..799 未被任何新范围覆盖。
        expect(locked.curveFrame(400)).toBe(WARP_CURVE_PAD);
        expect(locked.curveFrame(799)).toBe(WARP_CURVE_PAD);
        // 基线仍然全程有定义（内容跨度守恒）。
        expect(locked.baselineFrame(399)).toBeLessThan(800);
        expect(locked.baselineFrame(399)).not.toBe(WARP_BASELINE_UNKNOWN);
    });
});

describe("延伸/截短（速率不变，源窗口随边缘同步移动）—— 真相是「恒等」", () => {
    it("★ 右边延伸：旧覆盖范围内逐帧恒等（不是拉伸）", () => {
        const originGeo = geo("c1", 1, 2, { sourceStartSec: 0, sourceEndSec: 2 });
        const extended = geo("c1", 1, 4, { sourceStartSec: 0, sourceEndSec: 4 });
        const warp = warpOf([originGeo], [extended])!;
        // 内容位置根本没动：每帧仍播放原来源的那一帧。
        for (const f of [200, 201, 300, 450, 599]) {
            expect(warp.baselineFrame(f)).toBe(f);
        }
        // 新露出的部分在旧快照里没有电平 ⇒ 未知（不是拿旧值顶替）。
        expect(warp.baselineFrame(600)).toBe(WARP_BASELINE_UNKNOWN);
        expect(warp.baselineFrame(900)).toBe(WARP_BASELINE_UNKNOWN);
        // 恒等 ⇒ 可走查表。
        expect(warp.interpolableAt(300)).toBe(true);
    });

    it("★ 左边延伸：起点与源窗口同步前移 ⇒ 保留部分同样恒等，新露出部分未知", () => {
        const originGeo = geo("c1", 2, 2, { sourceStartSec: 0, sourceEndSec: 2 });
        const extended = geo("c1", 1, 3, { sourceStartSec: -1, sourceEndSec: 2 });
        const warp = warpOf([originGeo], [extended])!;
        for (const f of [400, 500, 700, 799]) {
            expect(warp.baselineFrame(f)).toBe(f);
        }
        expect(warp.baselineFrame(200)).toBe(WARP_BASELINE_UNKNOWN);
        expect(warp.baselineFrame(399)).toBe(WARP_BASELINE_UNKNOWN);
    });

    it("★ 截短：保留部分恒等（波形不缩放）", () => {
        const originGeo = geo("c1", 1, 2, { sourceStartSec: 0, sourceEndSec: 2 });
        const truncated = geo("c1", 1, 1, { sourceStartSec: 0, sourceEndSec: 1 });
        const warp = warpOf([originGeo], [truncated])!;
        for (const f of [200, 250, 399]) {
            expect(warp.baselineFrame(f)).toBe(f);
        }
        // 更长的旧范围不再被查询到（新范围之外 ⇒ 恒等，由调用方按未覆盖处理）。
        expect(warp.baselineFrame(400)).toBe(400);
    });

    it("★ 用户曲线**不**跟着搬（后端对延伸/截短不做参数线重映射）", () => {
        const originGeo = geo("c1", 1, 2, { sourceStartSec: 0, sourceEndSec: 2 });
        const extended = geo("c1", 1, 4, { sourceStartSec: 0, sourceEndSec: 4 });
        for (const lock of [false, true]) {
            const w = warpOf([originGeo], [extended], lock)!;
            for (const f of [200, 400, 700, 950]) {
                expect(w.curveFrame(f)).toBe(f);
                expect(w.curveSegment(f)).toBe(WARP_SEGMENT_IDENTITY);
            }
        }
    });
});

describe("增益（基线含静态 clip 增益）", () => {
    const originGeo = geo("c1", 0, 2, { gain: 0.5 });
    const louder = geo("c1", 0, 2, { gain: 1 });
    const warp = warpOf([originGeo], [louder], true)!;

    it("基线按增益比缩放，且时域不变", () => {
        expect(warp.baselineScale(100)).toBe(2);
        expect(warp.baselineFrame(100)).toBe(100);
    });

    it("范围外增益比为 1", () => {
        expect(warp.baselineScale(500)).toBe(1);
    });

    it("纯增益变化不搬移用户曲线（后端不会调用时域映射）", () => {
        expect(warp.curveFrame(100)).toBe(100);
        expect(warp.curveSegment(100)).toBe(WARP_SEGMENT_IDENTITY);
    });
});

describe("边界与退化", () => {
    it("无可用映射 / 非法帧周期时返回 null（调用方完全跳过该路径）", () => {
        expect(warpOf([geo("c1", 0, 2)], [])).toBeNull();
        expect(warpOf([], [geo("c1", 0, 2)])).toBeNull();
        expect(
            createLoudnessGeometryWarp({
                origin: [geo("c1", 0, 2)],
                clips: [geo("c1", 1, 2)],
                framePeriodMs: 0,
                lockParamLines: true,
            }),
        ).toBeNull();
    });

    it("手势期间新增 / 删除的 clip 不产出映射", () => {
        const warp = warpOf(
            [geo("gone", 0, 1), geo("kept", 0, 1)],
            [geo("kept", 1, 1), geo("fresh", 0, 1)],
        )!;
        expect(warp.baselineFrame(200)).toBe(0);
        // 新增 clip 的区间不受映射影响（恒等）。
        expect(warp.baselineFrame(100)).toBe(100);
    });

    it("零长度的新范围被丢弃", () => {
        expect(warpOf([geo("c1", 0, 2)], [geo("c1", 0, 0)])).toBeNull();
    });

    it("多段各自独立映射（编组联动 / 波纹跟随）", () => {
        const warp = warpOf(
            [geo("a", 0, 1), geo("b", 2, 1)],
            [geo("a", 1, 1), geo("b", 4, 1)],
        )!;
        expect(warp.baselineFrame(200)).toBe(0);
        expect(warp.baselineFrame(800)).toBe(400);
        expect(warp.baselineFrame(600)).toBe(600);
    });

    it("新范围重叠时后者胜（复刻后端逐段写入的覆盖顺序）", () => {
        const warp = warpOf(
            [geo("a", 0, 2), geo("b", 4, 2)],
            [geo("a", 2, 2), geo("b", 2, 2)],
        )!;
        // 帧 400..800 被两段同时覆盖，后一段（旧起点 4 s = 帧 800）胜出。
        expect(warp.baselineFrame(400)).toBe(800);
        expect(warp.baselineFrame(600)).toBe(1000);
    });

    it("Loop clip 在移动下仍精确平移（回绕与平移可交换）", () => {
        const warp = warpOf(
            [geo("c1", 0, 2, { loopEnabled: true })],
            [geo("c1", 1, 2, { loopEnabled: true })],
        )!;
        expect(warp.baselineFrame(200)).toBe(0);
        expect(warp.baselineFrame(399)).toBe(199);
    });

    it("倒放 clip 也能被搬运（锚点在 sourceEnd 侧）", () => {
        const originGeo = geo("c1", 0, 2, {
            sourceStartSec: 2,
            sourceEndSec: 4,
            reversed: true,
        });
        const moved = geo("c1", 1, 2, {
            sourceStartSec: 2,
            sourceEndSec: 4,
            reversed: true,
        });
        const warp = warpOf([originGeo], [moved])!;
        // 纯平移：新帧 k 对应旧帧 k−200。
        expect(warp.baselineFrame(200)).toBe(0);
        expect(warp.baselineFrame(399)).toBe(199);
    });
});

/**
 * ★ Loop 的锚点回绕必须与后端**同向**，倒放尤其。
 *
 * 后端三处（`pitch_clip::trim_and_resample_midi` 的 loop 分支、`schedule.rs`、
 * `mixdown.rs` 的 `floor_mod`）逐帧同一公式：
 *
 *   正放 idx(i) = rem_euclid(round(source_start·fps) + round(i·rate), n)
 *   倒放 idx(i) = rem_euclid(round(min(source_end, D)·fps) − 1 − round(i·rate), n)
 *
 * 修复前这里一律返回"正放仿射"（锚点 `source_start`、速率 `+rate`），**忽略
 * `reversed`**。于是倒放 + Loop 的 clip 在**锚点会变**的手势（Slip / 裁切）下，
 * 基线被搬到**相反方向**，松手后才被权威快照纠正。
 */
describe("Loop：锚点回绕与后端同向（含倒放）", () => {
    /** 媒体 3s ⇒ 回绕周期 600 帧。 */
    const D = 3;
    const N = Math.round(D * FPS);

    it("正放：锚点 = round(source_start·fps)，速率为正", () => {
        const c = resolveClipConsumption(
            geo("c", 0, 1, { sourceStartSec: 0.5, loopEnabled: true, mediaDurationSec: D }),
            FPS,
        )!;
        expect(c.exact).toBe(true);
        expect(c.rate).toBe(1);
        expect(c.anchorF).toBe(Math.round(0.5 * FPS));
    });

    it("★ 倒放：锚点在 min(source_end, D) 侧、速率为负", () => {
        const c = resolveClipConsumption(
            geo("c", 0, 1, {
                sourceStartSec: 0.5,
                sourceEndSec: 2,
                reversed: true,
                loopEnabled: true,
                mediaDurationSec: D,
            }),
            FPS,
        )!;
        expect(c.exact).toBe(true);
        // 倒放消费 s(i) = anchor_r − 1 − i·rate（后端首帧消费 anchor_r − 1）。
        expect(c.rate).toBe(-1);
        expect(c.anchorF).toBe(Math.round(2 * FPS) - 1);
    });

    it("★ 倒放 + source_end 越过媒体末端：锚点被 min(source_end, D) 钳到 D", () => {
        const c = resolveClipConsumption(
            geo("c", 0, 1, {
                sourceStartSec: 0.5,
                sourceEndSec: D + 1,
                reversed: true,
                loopEnabled: true,
                mediaDurationSec: D,
            }),
            FPS,
        )!;
        expect(c.anchorF).toBe(Math.round(D * FPS) - 1);
    });

    it("★ 倒放 + Loop 的 Slip：基线按后端同向平移（修复前方向相反）", () => {
        const originGeo = geo("c1", 0, 1, {
            sourceStartSec: 0.5,
            sourceEndSec: 2,
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: D,
        });
        const slipped = geo("c1", 0, 1, {
            sourceStartSec: 0.7,
            sourceEndSec: 2.2,
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: D,
        });
        const warp = warpOf([originGeo], [slipped])!;
        // Δanchor_r = (2.2 − 2.0)·fps = 40 帧，rate_old = −1 ⇒ g(f) = f − 40。
        // （修复前 rate_old = +1 且锚点在 source_start 侧 ⇒ g(f) = f + 40，方向相反。）
        for (let f = 40; f < 200; f += 1) {
            expect(warp.baselineFrame(f)).toBe(f - 40);
        }
    });

    it("★ 与后端 floor_mod 参考实现对拍（倒放 Slip / 倒放 Alt 拉伸 / 正放 Slip）", () => {
        const loopReversed = {
            reversed: true,
            loopEnabled: true,
            mediaDurationSec: D,
        } as const;
        const cases: [WarpClipGeometry, WarpClipGeometry][] = [
            // 倒放 + Loop + Slip（锚点变化）。
            [
                geo("r", 0, 1, { sourceStartSec: 0.5, sourceEndSec: 2, ...loopReversed }),
                geo("r", 0, 1, { sourceStartSec: 0.7, sourceEndSec: 2.2, ...loopReversed }),
            ],
            // 倒放 + Loop + Alt 拉伸（速率变化）。
            [
                geo("r", 0, 1, { sourceStartSec: 0.5, sourceEndSec: 2, ...loopReversed }),
                geo("r", 0, 2, {
                    sourceStartSec: 0.5,
                    sourceEndSec: 2,
                    playbackRate: 0.5,
                    ...loopReversed,
                }),
            ],
            // 正放 + Loop + Slip（锚点在 source_start 侧）。
            [
                geo("f", 0, 1, { sourceStartSec: 0.5, loopEnabled: true, mediaDurationSec: D }),
                geo("f", 0, 1, { sourceStartSec: 0.9, loopEnabled: true, mediaDurationSec: D }),
            ],
        ];
        for (const [before, after] of cases) {
            const warp = warpOf([before], [after])!;
            expect(warp).not.toBeNull();
            const startF = Math.round(after.startSec * FPS);
            const endF = startF + Math.round(after.lengthSec * FPS);
            let checked = 0;
            for (let f = startF; f < endF; f += 1) {
                const g = warp.baselineFrame(f);
                if (g === WARP_BASELINE_UNKNOWN) continue;
                // 仿射解出的 g 使**未回绕**源位置相等；取模对两边施加同一映射。
                const expected = backendLoopSourcePos(after, f, FPS, D);
                const got = backendLoopSourcePos(before, g, FPS, D);
                expect(circularDistance(expected, got, N)).toBeLessThan(1e-6);
                checked += 1;
            }
            // 至少要有一批帧真的被对拍过（避免"全 UNKNOWN ⇒ 空循环"式的假绿）。
            expect(checked).toBeGreaterThan(50);
        }
    });

    it("回绕周期暴露为时间线帧数（源域周期 ÷ |rate|）", () => {
        const fwd = resolveClipConsumption(
            geo("c", 0, 1, { loopEnabled: true, mediaDurationSec: D }),
            FPS,
        )!;
        expect(fwd.wrapFrames).toBeCloseTo(N, 9);
        // rate 0.5：时间线走一帧只消费 0.5 源帧 ⇒ 时间线周期翻倍。
        const slow = resolveClipConsumption(
            geo("c", 0, 1, { loopEnabled: true, mediaDurationSec: D, playbackRate: 0.5 }),
            FPS,
        )!;
        expect(slow.wrapFrames).toBeCloseTo(N / 0.5, 9);
        // 倒放：周期取 |rate|，仍为正。
        const rev = resolveClipConsumption(
            geo("c", 0, 1, {
                reversed: true,
                sourceEndSec: 1,
                loopEnabled: true,
                mediaDurationSec: D,
            }),
            FPS,
        )!;
        expect(rev.wrapFrames).toBeCloseTo(N, 9);
    });

    it("非周期情形 wrapFrames 为 0（非 Loop / Loop 但媒体时长未知）", () => {
        expect(resolveClipConsumption(geo("c", 0, 1), FPS)!.wrapFrames).toBe(0);
        expect(
            resolveClipConsumption(geo("c", 0, 1, { loopEnabled: true }), FPS)!.wrapFrames,
        ).toBe(0);
    });

    it("★ 周期段不得做整数帧插值（折返跳变会让线性插值造出不存在的中间值）", () => {
        // 3.1 → 0.15：锚点相位移动 0.05s（跨过媒体端点），段非恒等 ⇒ 一定建段。
        const warp = warpOf(
            [geo("c", 0, 1, { sourceStartSec: 3.1, loopEnabled: true, mediaDurationSec: D })],
            [geo("c", 0, 1, { sourceStartSec: 0.15, loopEnabled: true, mediaDurationSec: D })],
            true,
        )!;
        expect(warp).not.toBeNull();
        for (let f = 0; f < 200; f += 7) {
            expect(warp.interpolableAt(f)).toBe(false);
        }
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// ★ Loop 锚点跨过媒体端点：基线不得整段被判为「未知」
//
// ## 缺陷形态（用户报告："Slip 时动态参数干扰波形，且与 Loop 位置有关"）
//
// `computeSlipWindow` 把 `sourceStartSec` 取模进 `[0, D)`（`slipWindow.ts:150-151`），
// 于是**逐帧连续**的 Slip 会在某一帧让它从 `3.98` 跳到 `0.03`。前端若直接相减，
// 锚点差是 **−795 帧**，而真实相位差只有 **+5 帧** —— 偏出的整整一个周期会把
// **每一帧**都送出旧范围，`baselineFrame` 对全部帧返回 `WARP_BASELINE_UNKNOWN`，
// 动态增益整段退化为 1，松手拿到权威基线才恢复。
//
// 同理，`clip.lengthSec > D` 的 clip 跨多个周期，偏移累积也会让边缘帧越界。
//
// ## 本组用例钉住什么
//
// 用**独立**的后端同款组装（per-source-frame 电平 → 时间线）算出"松手后的权威基线"，
// 与拖拽期的映射逐帧比对 —— 但**只统计旧几何确实消费过的源帧**：真正新揭示的素材
// 前端原理上无法预测（反目标），不属于本缺陷。
// ─────────────────────────────────────────────────────────────────────────────
describe("Loop：锚点跨界时的基线映射（周期折返）", () => {
    const D = 2;
    const MEDIA_FRAMES = Math.round(D * FPS);
    /** 覆盖最长的用例（5s clip ⇒ 1000 帧）并留出余量。 */
    const TOTAL_FRAMES = 1400;

    /** 源域"真实电平"曲线（后端 per-source-frame 分析结果的替身）。 */
    function materialLevel(srcFrame: number): number {
        const t = srcFrame / FPS;
        return 0.2 + 0.15 * Math.sin(t * 5.5) + 0.05 * Math.sin(t * 41);
    }

    /** 时间线帧 → 源帧（后端同款：Loop 回绕 / 倒放翻转 / 窗口）；-1 = 静音。 */
    function srcFrameAt(g: WarpClipGeometry, frameF: number): number {
        const startF = Math.round(g.startSec * FPS);
        const lenF = Math.round(g.lengthSec * FPS);
        const i = frameF - startF;
        if (i < 0 || i >= lenF) return -1;
        const rate = g.playbackRate ?? 1;
        if (g.loopEnabled === true) {
            const consumed = Math.round(i * rate);
            const idx =
                g.reversed === true
                    ? Math.round(Math.min(g.sourceEndSec ?? 0, D) * FPS) - 1 - consumed
                    : Math.round((g.sourceStartSec ?? 0) * FPS) + consumed;
            return ((idx % MEDIA_FRAMES) + MEDIA_FRAMES) % MEDIA_FRAMES;
        }
        const winStart =
            g.reversed === true
                ? (g.sourceEndSec ?? 0) - g.lengthSec * rate
                : (g.sourceStartSec ?? 0);
        const k = g.reversed === true ? lenF - 1 - i : i;
        const srcF = Math.round(winStart * FPS + k * rate);
        return srcF >= 0 && srcF < MEDIA_FRAMES ? srcF : -1;
    }

    /** 组装整工程原声基线（单 clip；后端 `assemble_dyn_orig_from_cache` 的替身）。 */
    function assembleBaseline(g: WarpClipGeometry): number[] {
        const out = new Array<number>(TOTAL_FRAMES).fill(0);
        for (let f = 0; f < TOTAL_FRAMES; f += 1) {
            const idx = srcFrameAt(g, f);
            if (idx >= 0) out[f] = materialLevel(idx) * (g.gain ?? 1);
        }
        return out;
    }

    function coveredSourceFrames(g: WarpClipGeometry): Set<number> {
        const out = new Set<number>();
        const startF = Math.round(g.startSec * FPS);
        const endF = startF + Math.round(g.lengthSec * FPS);
        for (let f = startF; f < endF; f += 1) {
            const idx = srcFrameAt(g, f);
            if (idx >= 0) out.add(idx);
        }
        return out;
    }

    const VOL = new Array<number>(TOTAL_FRAMES).fill(1);
    /** 用户画了恒定目标 ⇒ 动态增益真正参与显示（否则未画帧恒为 1，掩盖错误）。 */
    const TARGET = new Array<number>(TOTAL_FRAMES).fill(0.6);

    function amplitudeMap(baseline: number[], warp: unknown): WaveformAmplitudeFactors {
        return makeLoudnessAmplitudeMap(
            {
                startFrame: 0,
                stride: 1,
                framePeriodMs: FP,
                volume: VOL,
                dynTarget: TARGET,
                dynBaseline: baseline,
            },
            { volume: () => null, dyn: () => null },
            () => 0,
            (() => warp) as never,
        ) as unknown as WaveformAmplitudeFactors;
    }

    /** 拖拽期 vs 松手后（只统计旧覆盖帧）。 */
    function dragVsRelease(before: WarpClipGeometry, after: WarpClipGeometry) {
        const warp = warpOf([before], [after], true);
        const drag = amplitudeMap(assembleBaseline(before), warp);
        const release = amplitudeMap(assembleBaseline(after), null);
        const hi = ((TOTAL_FRAMES - 1) * FP) / 1000;
        drag.beginWindow?.(0, hi);
        release.beginWindow?.(0, hi);
        const covered = coveredSourceFrames(before);
        const startF = Math.round(after.startSec * FPS);
        const endF = startF + Math.round(after.lengthSec * FPS);
        let unknown = 0;
        let unknownAmongCovered = 0;
        let maxDiff = 0;
        let checked = 0;
        for (let f = startF; f < endF; f += 1) {
            const isUnknown = warp !== null && warp.baselineFrame(f) === WARP_BASELINE_UNKNOWN;
            if (isUnknown) unknown += 1;
            const idx = srcFrameAt(after, f);
            if (idx < 0 || !covered.has(idx)) continue;
            checked += 1;
            if (isUnknown) unknownAmongCovered += 1;
            const a = drag.factorAt?.((f * FP) / 1000) ?? Number.NaN;
            const b = release.factorAt?.((f * FP) / 1000) ?? Number.NaN;
            const d = Math.abs(a - b);
            if (d > maxDiff) maxDiff = d;
        }
        return { unknown, unknownAmongCovered, maxDiff, checked, frames: endF - startF };
    }

    /** Loop clip（默认起点 0、长度 1s、媒体时长 D）。 */
    const loopClip = (o: GeoOpts & { lengthSec?: number } = {}): WarpClipGeometry =>
        geo("c", 0, o.lengthSec ?? 1, { loopEnabled: true, mediaDurationSec: D, ...o });

    it("★ 跨界 Slip（3.98→0.03）：不得整段判未知，且与松手后逐值一致", () => {
        const r = dragVsRelease(
            loopClip({ sourceStartSec: 3.98, sourceEndSec: 4.98 }),
            loopClip({ sourceStartSec: 0.03, sourceEndSec: 1.03 }),
        );
        // 修复前：unknown = 200/200（整段）、maxDiff ≈ 468（DYN_MAX_GAIN 量级）。
        expect(r.unknownAmongCovered).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
        // 真的对拍过一批帧（避免"全被过滤 ⇒ 空循环"式假绿）。
        expect(r.checked).toBeGreaterThan(150);
    });

    it("★ 逆方向跨界 Slip（0.03→3.98）：最小剩余取负号的那一支同样正确", () => {
        const r = dragVsRelease(
            loopClip({ sourceStartSec: 0.03, sourceEndSec: 1.03 }),
            loopClip({ sourceStartSec: 3.98, sourceEndSec: 4.98 }),
        );
        expect(r.unknownAmongCovered).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
        expect(r.checked).toBeGreaterThan(150);
    });

    it("★ clip 长于一个周期（5s > 2s）：Slip 后无任何未知帧", () => {
        const r = dragVsRelease(
            loopClip({ lengthSec: 5, sourceStartSec: 3.1, sourceEndSec: 8.1 }),
            loopClip({ lengthSec: 5, sourceStartSec: 0.1, sourceEndSec: 5.1 }),
        );
        // 整段都在旧覆盖内（5s 覆盖了整整 2.5 个周期）⇒ 一个未知帧都不该有。
        expect(r.unknown).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
        expect(r.checked).toBeGreaterThan(900);
    });

    it("★ clip 长于一个周期：小幅 Slip 也不得让边缘帧越界", () => {
        const r = dragVsRelease(
            loopClip({ lengthSec: 5, sourceStartSec: 1, sourceEndSec: 6 }),
            loopClip({ lengthSec: 5, sourceStartSec: 1.05, sourceEndSec: 6.05 }),
        );
        expect(r.unknown).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
    });

    it("★ 倒放 + Loop + 跨界 Slip（锚点在 min(se,D) 侧、方向为负）", () => {
        const r = dragVsRelease(
            loopClip({ reversed: true, sourceStartSec: 0, sourceEndSec: 0.03 }),
            loopClip({ reversed: true, sourceStartSec: 0, sourceEndSec: 3.98 }),
        );
        expect(r.unknownAmongCovered).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
        expect(r.checked).toBeGreaterThan(150);
    });

    it("★ Slip 整一个周期 ⇒ 内容逐帧不变（映射退化为恒等）", () => {
        const before = loopClip({ sourceStartSec: 1.3, sourceEndSec: 2.3 });
        const after = loopClip({ sourceStartSec: 3.3, sourceEndSec: 4.3 });
        // 一个周期 = D = 2s；相位完全相同 ⇒ 没有需要搬移的东西。
        expect(warpOf([before], [after], true)).toBeNull();
        const r = dragVsRelease(before, after);
        expect(r.unknown).toBe(0);
        expect(r.maxDiff).toBeLessThan(1e-6);
    });

    it("对照组：非 Loop 的 move / slip / trim 逐值不变（修复不外溢）", () => {
        const plain = (o: GeoOpts = {}) => geo("c", 0, 1, o);
        const move = dragVsRelease(plain(), geo("c", 0.05, 1, {}));
        expect(move.unknown).toBe(0);
        expect(move.maxDiff).toBeLessThan(1e-6);
        const slip = dragVsRelease(
            plain(),
            plain({ sourceStartSec: 0.05, sourceEndSec: 1.05 }),
        );
        expect(slip.maxDiff).toBeLessThan(1e-6);
        const trim = dragVsRelease(plain(), geo("c", 0, 1.05, {}));
        expect(trim.maxDiff).toBeLessThan(1e-6);
    });

    it("对照组：Loop 的 move / trim（锚点相位不变）逐值不变", () => {
        const move = dragVsRelease(loopClip(), geo("c", 0.05, 1, {
            loopEnabled: true,
            mediaDurationSec: D,
        }));
        expect(move.unknown).toBe(0);
        expect(move.maxDiff).toBeLessThan(1e-6);
        const trim = dragVsRelease(
            loopClip(),
            geo("c", 0, 1.05, { loopEnabled: true, mediaDurationSec: D }),
        );
        expect(trim.maxDiff).toBeLessThan(1e-6);
    });

    it("退化：Loop 但媒体时长未知 ⇒ 与修复前一致（wrapFrames = 0）", () => {
        const before = geo("c", 0, 1, { loopEnabled: true, sourceStartSec: 0.5 });
        const after = geo("c", 0, 1, { loopEnabled: true, sourceStartSec: 0.9 });
        expect(resolveClipConsumption(before, FPS)!.wrapFrames).toBe(0);
        const warp = warpOf([before], [after], true)!;
        // 无周期可折返：越界即未知（保持既有退化语义，不引入伪基线）。
        expect(warp.interpolableAt(0)).toBe(true);
    });
});
