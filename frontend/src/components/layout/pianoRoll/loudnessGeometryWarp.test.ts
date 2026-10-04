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
    };
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

    it("Loop：回绕非仿射，标记为不精确（仍给出正放仿射近似）", () => {
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
