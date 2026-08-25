/**
 * Loop（循环源）/ 内容边界 吸附工具。
 *
 * ── 循环节的统一定义 ─────────────────────────────────────────────
 * 引擎对 Clip 内容的映射（与渲染/混音/引擎一致）：
 *   正放 s(t) = floor_mod(source_start + t·rate, D)      （Loop）
 *   倒放 s(t) = floor_mod(min(source_end, D) − t·rate, D)（Loop）
 *   非 Loop：s(t) = source_start ± t·r（不回绕；s ∉ [0, D) 为静音，
 *           包括右缘延伸产生的"尾部静音"与 REAPER 左延伸的前导静音）
 * 其中 D = 源媒体总时长。
 *
 * "循环节位置"= 媒体边界（s=0 与 s=D）在时间线上投影出的相位点：
 * - Loop：每过一个周期 D/rate 就重复一次 → 无穷等差候选族；
 * - 非 Loop：只有一次穿越 → 有限候选。特别地，未循环 Clip 的
 *   "循环节"就是**原始媒体内容在 Clip 内的终止位置**（s=D 的投影）。
 *
 * 两类操作共用同一套候选（offset 均为相对 Clip 基准起点的有符号秒）：
 * - edge（trim_left / trim_right 拖边）：移动边缘恰好落在媒体边界上；
 * - slip（内容平移）：媒体边界恰好对齐 Clip 起点（T=0）或终点（T=len）。
 */

/** Clip 源媒体总时长（秒）：优先 durationFrames/sourceSampleRate，回退 durationSec。 */
export function clipMediaDurationSec(clip: {
    durationFrames?: number | null;
    sourceSampleRate?: number | null;
    durationSec?: number | null;
}): number | null {
    const frames = Number(clip.durationFrames);
    const sampleRate = Number(clip.sourceSampleRate);
    if (Number.isFinite(frames) && frames > 0 && Number.isFinite(sampleRate) && sampleRate > 0) {
        return frames / sampleRate;
    }
    const durationSec = Number(clip.durationSec);
    if (Number.isFinite(durationSec) && durationSec > 0) return durationSec;
    return null;
}

function floorMod(value: number, period: number): number {
    return value - period * Math.floor(value / period);
}

export type BoundarySnapMode = "edge" | "slip";

export interface BoundarySnapClip {
    loopEnabled: boolean;
    reversed: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    playbackRate: number;
    /** Clip 长度（秒）：slip 模式的"对齐到 Clip 终点"候选需要它。 */
    lengthSec: number;
    durationFrames?: number | null;
    sourceSampleRate?: number | null;
    durationSec?: number | null;
    /**
     * 内容时长（秒）覆盖值：音高参考块等无源媒体 Clip 由调用方用
     * resolveClipContentDurationSec 预解析后传入；优先于下方元数据链。
     */
    contentDurationSec?: number | null;
}

/**
 * 返回距 rawOffsetSec 最近的吸附偏移（时间线秒，相对 Clip 基准起点 /
 * Slip 偏移量）；无可用候选时返回 null。调用方自行比较与"吸附距离"
 * 阈值决定是否采纳 —— 本函数不做阈值判断。
 */
export function nearestBoundarySnapOffsetSec(
    clip: BoundarySnapClip,
    mode: BoundarySnapMode,
    rawOffsetSec: number,
): number | null {
    const rate =
        Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6 ? clip.playbackRate : 1;
    const raw = Number(rawOffsetSec);
    if (!Number.isFinite(raw)) return null;
    const ss = Number(clip.sourceStartSec) || 0;
    const se = Number(clip.sourceEndSec) || 0;
    const len = Math.max(0, Number(clip.lengthSec) || 0);

    // ── Loop：mod-D 相位族 ─────────────────────────────────────
    // 边界 b∈{0,D} 的 mod-D 族完全相同（相差恰一个周期），取单族即可：
    //   正放 edge/slip：o·rate ≡ −source_start        (mod D)
    //   倒放 edge     ：o·rate ≡ +source_end          (mod D)
    //   倒放 slip     ：o·rate ≡ −source_end          (mod D)
    // slip 另有"媒体边界对齐到 Clip 终点"的第二相位族（再偏移 ±len）。
    if (clip.loopEnabled) {
        const d =
            clip.contentDurationSec != null && clip.contentDurationSec > 1e-9
                ? clip.contentDurationSec
                : clipMediaDurationSec(clip);
        if (d == null || !(d > 1e-9)) return null;
        // 倒放锚点与引擎/渲染同约定：clamp 到媒体时长 min(se, D) ——
        // 超界的 se（split 环绕窗口/历史数据可达）若不 clamp，相位族会
        // 与波形/音频的实际回绕位置错开 se mod D。
        const seEff = Math.min(se, d);
        let phis: number[];
        if (!clip.reversed) {
            const p0 = floorMod(-ss, d);
            // 终点对齐：δr ≡ −ss − len·r (mod D)
            phis = mode === "slip" ? [p0, floorMod(p0 - len * rate, d)] : [p0];
        } else if (mode === "edge") {
            phis = [floorMod(seEff, d)];
        } else {
            const p0 = floorMod(-seEff, d);
            // 终点对齐：δr ≡ −se + len·r (mod D)
            phis = [p0, floorMod(p0 + len * rate, d)];
        }
        let best: number | null = null;
        let bestDist = Number.POSITIVE_INFINITY;
        for (const phi of phis) {
            const kNearest = Math.round((raw * rate - phi) / d);
            for (let k = kNearest - 2; k <= kNearest + 2; k += 1) {
                const offset = (phi + k * d) / rate;
                const dist = Math.abs(offset - raw);
                if (dist < bestDist - 1e-12) {
                    best = offset;
                    bestDist = dist;
                }
            }
        }
        return best;
    }

    // ── 非 Loop：有限候选（媒体边界的一次性穿越）───────────────
    // 向左/向右延伸均已放开（对称无界），所有候选都可达。
    const bounds: number[] = [0];
    const d =
        clip.contentDurationSec != null && clip.contentDurationSec > 1e-9
            ? clip.contentDurationSec
            : clipMediaDurationSec(clip);
    if (d != null && d > 1e-9) bounds.push(d);
    const cands: number[] = [];
    for (const b of bounds) {
        if (mode === "edge") {
            // 边缘落在媒体边界上：
            cands.push(clip.reversed ? (se - b) / rate : (b - ss) / rate);
        } else {
            // Slip：边界对齐 Clip 起点（T=0）或终点（T=len）。
            const base = clip.reversed ? (b - se) / rate : (b - ss) / rate;
            cands.push(base);
            cands.push(clip.reversed ? base + len : base - len);
        }
    }
    let best: number | null = null;
    let bestDist = Number.POSITIVE_INFINITY;
    for (const offset of cands) {
        if (!Number.isFinite(offset)) continue;
        const dist = Math.abs(offset - raw);
        if (dist < bestDist - 1e-12) {
            best = offset;
            bestDist = dist;
        }
    }
    return best;
}

/** 循环/边界吸附阈值（秒）：由吸附设置的"吸附距离"（像素）换算。 */
export function loopSnapThresholdSec(snapDistancePx: number, pxPerSec: number): number {
    return Math.max(0, Number(snapDistancePx) || 0) / Math.max(1e-9, pxPerSec);
}

/** Slip 平移量下媒体边界与 Clip 边缘的对齐情况。 */
export interface SlipAlignedSides {
    /** 某媒体边界恰好落在 Clip 起点（T=0）。 */
    start: boolean;
    /** 某媒体边界恰好落在 Clip 终点（T=len）。 */
    end: boolean;
}

/**
 * 判定 Slip 窗口平移量 `windowShiftSec`（与 nearestBoundarySnapOffsetSec
 * 的 rawOffsetSec 同一 X 域：指针位移 × dir）下，媒体边界与 Clip 的哪些
 * 边缘重合。
 *
 * 与 nearestBoundarySnapOffsetSec("slip") 的候选族**同一套**相位/等式：
 * - Loop：正放起点 φ=floor_mod(−ss,D)、终点 φ=floor_mod(−ss−len·r,D)；
 *   倒放起点 φ=floor_mod(−se_eff,D)、终点 φ=floor_mod(−se_eff+len·r,D)
 *   （se_eff = min(se, D)，与引擎锚点约定一致）；
 * - 非 Loop：有限等式候选，b ∈ {0, D}。
 *
 * 用于吸附竖线高亮只标记真正对齐的一侧 —— 两侧同时对齐（如 len·r 恰为
 * 整周期）才两侧都标。
 */
export function slipBoundaryAlignedSides(
    clip: BoundarySnapClip,
    windowShiftSec: number,
    eps = 1e-6,
): SlipAlignedSides {
    const rate =
        Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6 ? clip.playbackRate : 1;
    const raw = Number(windowShiftSec);
    if (!Number.isFinite(raw)) return { start: false, end: false };
    const ss = Number(clip.sourceStartSec) || 0;
    const se = Number(clip.sourceEndSec) || 0;
    const len = Math.max(0, Number(clip.lengthSec) || 0);

    const d =
        clip.contentDurationSec != null && clip.contentDurationSec > 1e-9
            ? clip.contentDurationSec
            : clipMediaDurationSec(clip);

    // ── Loop：mod-D 相位族（环绕距离比较）─────────────────────
    if (clip.loopEnabled) {
        if (d == null || !(d > 1e-9)) return { start: false, end: false };
        const seEff = Math.min(se, d);
        let startPhi: number;
        let endPhi: number;
        if (!clip.reversed) {
            const p0 = floorMod(-ss, d);
            startPhi = p0;
            endPhi = floorMod(p0 - len * rate, d);
        } else {
            const p0 = floorMod(-seEff, d);
            startPhi = p0;
            endPhi = floorMod(p0 + len * rate, d);
        }
        const phase = raw * rate;
        const tol = eps * rate;
        // 环绕距离：|floor_mod(phase − phi + d/2, d) − d/2|
        const near = (phi: number) =>
            Math.abs(floorMod(phase - phi + d / 2, d) - d / 2) <= tol;
        return { start: near(startPhi), end: near(endPhi) };
    }

    // ── 非 Loop：有限等式候选（b ∈ {0, D}）──────────────────
    const bounds = d != null && d > 1e-9 ? [0, d] : [0];
    let start = false;
    let end = false;
    for (const b of bounds) {
        const base = clip.reversed ? (b - se) / rate : (b - ss) / rate;
        if (Math.abs(raw - base) <= eps) start = true;
        const endCand = clip.reversed ? base + len : base - len;
        if (Math.abs(raw - endCand) <= eps) end = true;
    }
    return { start, end };
}
