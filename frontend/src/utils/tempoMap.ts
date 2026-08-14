/**
 * Tempo Map 工具模块。
 *
 * HiFiShifter 的时间轴以“秒”为绝对坐标（clip 位置、播放头、参数帧都是秒/帧），
 * 因此 Tempo Map 采用“时间锚定”模型：
 * - 每个变化点（TempoPoint）锚定在一个绝对秒位置上；
 * - 点携带 BPM、拍号（分子/分母）与可选的音阶（key signature）数据；
 * - 从该秒位置起生效，直到下一个变化点；
 * - 音乐时间（拍 / 小节.拍）完全由 Tempo Map 推导，编辑变化点不会移动任何音频。
 *
 * 拍号为“每小节拍数”的概念：一小节的拍数 = numerator * 4 / denominator。
 * 变化点处的音阶为 null 时表示“跟随工程音阶”。
 */

import type { ScaleLike } from "./musicalScales";
import { isScaleKey, normalizeCustomScaleNotes } from "./musicalScales";

// ────────────────────────────────────────────────────────────────────────────
// 类型
// ────────────────────────────────────────────────────────────────────────────

/** 变化点携带的音阶数据；null 表示跟随工程音阶。 */
export interface TempoMapScaleData {
    /** 内置音阶键名（如 "C"、"Db"）。 */
    key?: string;
    /** 自定义音阶名称。 */
    name?: string;
    /** 自定义音阶音级集合（0-11）。 */
    notes?: number[];
}

export interface TempoPoint {
    id: string;
    /** 绝对秒位置（时间锚定）。 */
    positionSec: number;
    /** BPM。 */
    bpm: number;
    /** 拍号分子（如 3/4 的 3）。 */
    numerator: number;
    /** 拍号分母（如 3/4 的 4）。 */
    denominator: number;
    /** 音阶覆盖；null = 跟随工程音阶。 */
    scale: TempoMapScaleData | null;
}

export interface TempoMap {
    /** 按 positionSec 升序排列；第一个点必须位于 0。 */
    points: TempoPoint[];
}

/** 拍.小节分解结果（bar/beat 均为 1 起始）。 */
export interface BarBeat {
    /** 小节号，1 起始。 */
    bar: number;
    /** 小节内拍号，1 起始。 */
    beat: number;
    /** 拍内余量（0..1，以拍为单位）。 */
    sub: number;
}

export interface TempoAtSec {
    bpm: number;
    numerator: number;
    denominator: number;
}

export interface TempoGridLine {
    sec: number;
    isBar: boolean;
}

// ────────────────────────────────────────────────────────────────────────────
// 常量与基础工具
// ────────────────────────────────────────────────────────────────────────────

export const TEMPO_BPM_MIN = 10;
export const TEMPO_BPM_MAX = 960;
export const TEMPO_NUMERATOR_MAX = 32;
export const TEMPO_DENOMINATORS = [1, 2, 4, 8, 16, 32] as const;

let tempoPointIdCounter = 0;

export function createTempoPointId(): string {
    tempoPointIdCounter += 1;
    return `tp_${Date.now().toString(36)}_${tempoPointIdCounter}`;
}

export function clampBpm(bpm: number): number {
    if (!Number.isFinite(bpm)) return 120;
    return Math.min(TEMPO_BPM_MAX, Math.max(TEMPO_BPM_MIN, bpm));
}

export function clampNumerator(numerator: number): number {
    if (!Number.isFinite(numerator)) return 4;
    return Math.min(TEMPO_NUMERATOR_MAX, Math.max(1, Math.round(numerator)));
}

export function clampDenominator(denominator: number): number {
    const rounded = Math.round(denominator);
    return (TEMPO_DENOMINATORS as readonly number[]).includes(rounded) ? rounded : 4;
}

export function isTempoMapEmpty(map: TempoMap | null | undefined): boolean {
    return !map || map.points.length === 0;
}

/** 每小节拍数（拍号为“分子个 4/分母 音符”，一小节 = numerator * 4 / denominator 拍）。 */
export function beatsPerBarOf(point: Pick<TempoPoint, "numerator" | "denominator">): number {
    const denominator = clampDenominator(point.denominator);
    return (clampNumerator(point.numerator) * 4) / denominator;
}

/** 格式化 BPM 显示（整数则无小数）。 */
export function formatTempoBpm(bpm: number): string {
    const rounded = Math.round(bpm * 1000) / 1000;
    return Number.isInteger(rounded) ? String(rounded) : String(rounded);
}

/** 拍号显示文本，如 "4/4"、"3/4"。 */
export function formatTimeSignature(point: Pick<TempoPoint, "numerator" | "denominator">): string {
    return `${clampNumerator(point.numerator)}/${clampDenominator(point.denominator)}`;
}

// ────────────────────────────────────────────────────────────────────────────
// 规范化 / 序列化
// ────────────────────────────────────────────────────────────────────────────

/**
 * 规范化 TempoMap：
 * - 按位置排序；位置过近（< 1e-6s）的重复点只保留第一个；
 * - 第一个点必须位于 0（不足则用 fallback 值补一个，并携带工程音阶 —— 初始点即工程基准记录）；
 * - 无点返回 null。
 */
export function normalizeTempoMap(
    map: TempoMap | null | undefined,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
    opts?: { projectScale?: ScaleLike; projectScaleName?: string },
): TempoMap | null {
    if (!map || !Array.isArray(map.points)) return null;

    const fallback = clampNumerator(fallbackBeatsPerBar || 4);
    const points: TempoPoint[] = [];
    for (const raw of map.points) {
        if (!raw || typeof raw.id !== "string" || !raw.id) continue;
        const positionSec = Math.max(0, Number(raw.positionSec) || 0);
        if (points.length > 0 && Math.abs(points[points.length - 1].positionSec - positionSec) < 1e-6) {
            continue;
        }
        points.push({
            id: raw.id,
            positionSec,
            bpm: clampBpm(Number(raw.bpm)),
            numerator: clampNumerator(Number(raw.numerator)),
            denominator: clampDenominator(Number(raw.denominator)),
            scale: normalizeScaleData(raw.scale),
        });
    }
    points.sort((a, b) => a.positionSec - b.positionSec);
    if (points.length === 0) return null;

    if (points[0].positionSec > 1e-9) {
        points.unshift({
            id: createTempoPointId(),
            positionSec: 0,
            bpm: clampBpm(fallbackBpm),
            numerator: fallback,
            denominator: 4,
            // 初始点即工程基准记录：携带工程音阶（键或自定义音级）。
            scale: scaleLikeToScaleData(opts?.projectScale, opts?.projectScaleName),
        });
    }
    points[0].positionSec = 0;
    return { points };
}

export function normalizeScaleData(scale: TempoMapScaleData | null | undefined): TempoMapScaleData | null {
    if (!scale) return null;
    const key = typeof scale.key === "string" && isScaleKey(scale.key) ? scale.key : undefined;
    const notes = Array.isArray(scale.notes) ? normalizeCustomScaleNotes(scale.notes) : undefined;
    if (!key && (!notes || notes.length === 0)) return null;
    return {
        key,
        name: typeof scale.name === "string" ? scale.name : undefined,
        notes,
    };
}

/**
 * 后端（camelCase）→ 前端 TempoMap。
 *
 * 后端 `TimelineStatePayload.tempo_map` 是变化点的“裸数组”
 * （`Option<Vec<TempoPointPayload>>` 直接序列化），不是 `{ points: [...] }` 包装对象；
 * 同时容忍旧的包装形状，保证任何后端版本都不被误判为“无 Tempo Map”。
 */
export function fromBackendTempoMap(
    data: unknown,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
    opts?: { projectScale?: ScaleLike; projectScaleName?: string },
): TempoMap | null {
    const maybeWrapped = data as { points?: unknown } | null;
    const rawPoints: unknown = Array.isArray(data) ? data : maybeWrapped?.points;
    if (!Array.isArray(rawPoints)) return null;
    return normalizeTempoMap(
        {
            points: rawPoints.map((p) => {
                const item = p as {
                    id?: string;
                    positionSec?: number;
                    bpm?: number;
                    numerator?: number;
                    denominator?: number;
                    scale?: TempoMapScaleData | null;
                };
                return {
                    id: String(item.id ?? createTempoPointId()),
                    positionSec: Number(item.positionSec ?? 0),
                    bpm: Number(item.bpm ?? fallbackBpm),
                    numerator: Number(item.numerator ?? fallbackBeatsPerBar),
                    denominator: Number(item.denominator ?? 4),
                    scale: item.scale ?? null,
                };
            }),
        },
        fallbackBpm,
        fallbackBeatsPerBar,
        opts,
    );
}

/**
 * 前端 TempoMap → 后端（camelCase）载荷：变化点“裸数组”（null = 清除）。
 * 与后端 `set_timeline_tempo_map(tempoMap: Option<Vec<TempoPointPayload>>)` 参数一致。
 */
export function toBackendTempoMap(
    map: TempoMap | null,
): Array<{
    id: string;
    positionSec: number;
    bpm: number;
    numerator: number;
    denominator: number;
    scale: {
        key: string | null;
        name: string | null;
        notes: number[] | null;
    } | null;
}> | null {
    if (!map) return null;
    return map.points.map((p) => ({
        id: p.id,
        positionSec: p.positionSec,
        bpm: p.bpm,
        numerator: p.numerator,
        denominator: p.denominator,
        scale: p.scale
            ? {
                  key: p.scale.key ?? null,
                  name: p.scale.name ?? null,
                  notes: p.scale.notes ? [...p.scale.notes] : null,
              }
            : null,
    }));
}

// ────────────────────────────────────────────────────────────────────────────
// 查询
// ────────────────────────────────────────────────────────────────────────────

/** 返回 governing `sec` 的变化点下标（最后一个 positionSec <= sec 的点）。 */
export function pointIndexAtSec(map: TempoMap, sec: number): number {
    const { points } = map;
    const target = Math.max(0, sec);
    let lo = 0;
    let hi = points.length - 1;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (points[mid].positionSec <= target + 1e-9) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

export function tempoAtSec(
    map: TempoMap | null,
    sec: number,
    fallback: { bpm: number; beatsPerBar: number },
): TempoAtSec {
    if (!map || map.points.length === 0) {
        const beatsPerBar = clampNumerator(fallback.beatsPerBar || 4);
        return { bpm: clampBpm(fallback.bpm), numerator: beatsPerBar, denominator: 4 };
    }
    const point = map.points[pointIndexAtSec(map, sec)];
    return {
        bpm: point.bpm,
        numerator: point.numerator,
        denominator: point.denominator,
    };
}

export function pointAtSec(map: TempoMap, sec: number): TempoPoint {
    return map.points[pointIndexAtSec(map, sec)];
}

/** 将变化点的音阶数据转换为 ScaleLike；无法解析返回 null。 */
export function scaleDataToScaleLike(data: TempoMapScaleData | null | undefined): ScaleLike | null {
    if (!data) return null;
    if (typeof data.key === "string" && isScaleKey(data.key)) return data.key;
    if (Array.isArray(data.notes) && data.notes.length > 0) {
        return normalizeCustomScaleNotes(data.notes);
    }
    return null;
}

/** 将 ScaleLike（含自定义音阶名）转换为变化点音阶数据；无法解析返回 null。 */
export function scaleLikeToScaleData(
    scale: ScaleLike | null | undefined,
    name?: string,
): TempoMapScaleData | null {
    if (!scale) return null;
    if (Array.isArray(scale)) {
        return { name: name ?? "", notes: [...scale] };
    }
    return { key: scale as string };
}

/**
 * 查询 sec 位置生效的音阶：
 * - 从该位置向前找最近一个携带音阶的变化点；
 * - 找不到则返回工程音阶。
 */
export function effectiveScaleAtSec(
    map: TempoMap | null,
    sec: number,
    projectScale: ScaleLike | undefined,
): ScaleLike | undefined {
    if (!map || map.points.length === 0) return projectScale;
    const target = Math.max(0, sec);
    for (let i = pointIndexAtSec(map, target); i >= 0; i -= 1) {
        const scale = scaleDataToScaleLike(map.points[i].scale);
        if (scale) return scale;
    }
    return projectScale;
}

/**
 * 查询某个变化点“之前”生效的音阶（用于“跟随之前的音阶”文案）：
 * 0 位置初始点之前的音阶即工程音阶；其它位置返回其前一个点的生效音阶。
 */
export function previousScaleAtSec(
    map: TempoMap | null,
    positionSec: number,
    projectScale: ScaleLike | undefined,
): ScaleLike | undefined {
    if (positionSec <= 1e-9) return projectScale;
    return effectiveScaleAtSec(map, positionSec - 1e-6, projectScale);
}

/**
 * 查询 [startSec, endSec] 范围内生效过的音阶变化（位置 + 音阶），
 * 用于判断“工程音阶”选项是否受 Tempo Map 影响。
 */
export function scaleChangesInRange(
    map: TempoMap | null,
    startSec: number,
    endSec: number,
): Array<{ positionSec: number; scale: ScaleLike }> {
    if (!map || endSec < startSec) return [];
    const out: Array<{ positionSec: number; scale: ScaleLike }> = [];
    for (const point of map.points) {
        if (point.positionSec > endSec + 1e-9) break;
        if (point.positionSec < startSec - 1e-9 && point.positionSec > 0) continue;
        const scale = scaleDataToScaleLike(point.scale);
        if (scale) out.push({ positionSec: point.positionSec, scale });
    }
    return out;
}

/**
 * 音阶签名（用于缓存失效比较）：工程音阶 + 各音阶变化点的 (位置, 音阶)。
 */
export function tempoMapScaleSignature(
    map: TempoMap | null,
    projectScale: ScaleLike | undefined,
): string {
    const parts: string[] = [];
    parts.push(`proj:${scaleSignaturePart(projectScale)}`);
    if (map) {
        for (const point of map.points) {
            const scale = scaleDataToScaleLike(point.scale);
            if (scale) {
                parts.push(`${point.positionSec.toFixed(6)}:${scaleSignaturePart(scale)}`);
            }
        }
    }
    return parts.join("|");
}

function scaleSignaturePart(scale: ScaleLike | undefined): string {
    if (!scale) return "-";
    if (Array.isArray(scale)) return normalizeCustomScaleNotes(scale).join(",");
    return scale as string;
}

// ────────────────────────────────────────────────────────────────────────────
// 秒 ↔ 拍 双向转换
// ────────────────────────────────────────────────────────────────────────────

/** 每个变化点的全局拍位置缓存（由 sec 积分得到）。 */
export interface TempoMapBeatCache {
    map: TempoMap;
    fallbackBpm: number;
    /** 与 points 等长；第 i 个变化点位置对应的全局拍。 */
    pointBeats: number[];
    /** 每段拍速（拍/秒）。 */
    beatRates: number[];
}

export function buildBeatCache(map: TempoMap, fallbackBpm: number): TempoMapBeatCache {
    const safeBpm = Math.max(1, clampBpm(fallbackBpm));
    const pointBeats: number[] = [0];
    const beatRates: number[] = [];
    for (let i = 0; i < map.points.length; i += 1) {
        const point = map.points[i];
        const rate = point.bpm / 60;
        beatRates.push(rate);
        if (i + 1 < map.points.length) {
            const next = map.points[i + 1];
            const spanSec = Math.max(0, next.positionSec - point.positionSec);
            pointBeats.push(pointBeats[i] + spanSec * rate);
        }
    }
    return { map, fallbackBpm: safeBpm, pointBeats, beatRates };
}

/** 秒 → 全局拍。无 Tempo Map 时退化为恒定 BPM。 */
export function secToBeat(map: TempoMap | null, sec: number, fallbackBpm: number): number {
    const safeSec = Math.max(0, sec);
    if (!map || map.points.length === 0) {
        return (safeSec * Math.max(1, fallbackBpm || 120)) / 60;
    }
    const cache = buildBeatCache(map, fallbackBpm);
    const idx = pointIndexAtSec(map, safeSec);
    const point = map.points[idx];
    return cache.pointBeats[idx] + (safeSec - point.positionSec) * cache.beatRates[idx];
}

/** 全局拍 → 秒。无 Tempo Map 时退化为恒定 BPM。 */
export function beatToSec(map: TempoMap | null, beat: number, fallbackBpm: number): number {
    const safeBeat = Math.max(0, beat);
    if (!map || map.points.length === 0) {
        return (safeBeat * 60) / Math.max(1, fallbackBpm || 120);
    }
    const cache = buildBeatCache(map, fallbackBpm);
    const { points } = map;
    for (let i = points.length - 1; i >= 0; i -= 1) {
        if (safeBeat >= cache.pointBeats[i] - 1e-9) {
            return points[i].positionSec + (safeBeat - cache.pointBeats[i]) / cache.beatRates[i];
        }
    }
    return points[0].positionSec + safeBeat / cache.beatRates[0];
}

// ────────────────────────────────────────────────────────────────────────────
// 小节.拍 分解
// ────────────────────────────────────────────────────────────────────────────

/**
 * 全局拍 → 小节.拍.余量（bar/beat 1 起始）。
 *
 * 小节对齐规则（与 REAPER 的拍号变化语义一致）：
 * - 每段内部按该段拍号均匀分小节；
 * - 段末尾不足一小节的余拍计为“不完整小节”，小节号继续累加；
 * - 拍号变化点处重新开始小节对齐。
 */
export function beatToBarBeat(
    map: TempoMap | null,
    beat: number,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
): BarBeat {
    const safeBeat = Math.max(0, beat);
    if (!map || map.points.length === 0) {
        const bpb = Math.max(1, beatsPerBarOf({ numerator: fallbackBeatsPerBar || 4, denominator: 4 }));
        const barIndex = Math.floor(safeBeat / bpb);
        const inBar = safeBeat - barIndex * bpb;
        const beatIndex = Math.floor(inBar);
        return { bar: barIndex + 1, beat: beatIndex + 1, sub: inBar - beatIndex };
    }

    const cache = buildBeatCache(map, fallbackBpm);
    const { points } = map;
    let bar = 1;

    for (let i = 0; i < points.length; i += 1) {
        const segStartBeat = cache.pointBeats[i];
        const segEndBeat = i + 1 < points.length ? cache.pointBeats[i + 1] : Infinity;
        const bpb = Math.max(1, beatsPerBarOf(points[i]));

        if (safeBeat < segEndBeat || !Number.isFinite(segEndBeat)) {
            const rel = safeBeat - segStartBeat;
            const fullBars = Math.floor(rel / bpb);
            const inBar = rel - fullBars * bpb;
            const beatIndex = Math.floor(inBar);
            return {
                bar: bar + fullBars,
                beat: beatIndex + 1,
                sub: inBar - beatIndex,
            };
        }

        const segLen = segEndBeat - segStartBeat;
        const fullBars = Math.floor(segLen / bpb);
        const leftover = segLen - fullBars * bpb;
        bar += fullBars + (leftover > 1e-9 ? 1 : 0);
    }

    // 超出工程末尾：按最后一段外推。
    const lastIndex = points.length - 1;
    const last = points[lastIndex];
    const bpb = Math.max(1, beatsPerBarOf(last));
    const rel = safeBeat - cache.pointBeats[lastIndex];
    const fullBars = Math.floor(rel / bpb);
    const inBar = rel - fullBars * bpb;
    const beatIndex = Math.floor(inBar);
    return { bar: bar + fullBars, beat: beatIndex + 1, sub: inBar - beatIndex };
}

/** 秒 → 小节.拍.余量。 */
export function barBeatAtSec(
    map: TempoMap | null,
    sec: number,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
): BarBeat {
    return beatToBarBeat(map, secToBeat(map, sec, fallbackBpm), fallbackBpm, fallbackBeatsPerBar);
}

// ────────────────────────────────────────────────────────────────────────────
// 网格与吸附
// ────────────────────────────────────────────────────────────────────────────

/**
 * 网格吸附：把秒位置吸附到最近的网格线。
 *
 * 有 Tempo Map 时，网格线按“每段内局部对齐”生成（每个变化点处重新对齐小节/节拍，
 * 与时间标尺和背景网格完全一致），因此吸附也必须在当前生效段内做局部取整：
 * 以段起点为原点，把段内拍数吸附到 stepBeats 的整数倍。
 */
export function snapSecToTempoGrid(
    sec: number,
    map: TempoMap | null,
    stepBeats: number,
    fallbackBpm: number,
): number {
    const safeStep = Math.max(1e-9, stepBeats);
    const safeSec = Math.max(0, sec);
    if (!map || map.points.length === 0) {
        const stepSec = (safeStep * 60) / Math.max(1, fallbackBpm || 120);
        return Math.max(0, Math.round(safeSec / stepSec) * stepSec);
    }
    const idx = pointIndexAtSec(map, safeSec);
    const point = map.points[idx];
    const bpm = Math.max(1, point.bpm);
    const localBeat = (safeSec - point.positionSec) * (bpm / 60);
    const snappedLocal = Math.max(0, Math.round(localBeat / safeStep) * safeStep);
    return point.positionSec + (snappedLocal * 60) / bpm;
}

/**
 * 生成 [startSec, endSec] 范围内的网格线（弱网格 + 小节强网格）。
 *
 * 有 Tempo Map 时，每个变化点处重新对齐小节/节拍（与标尺标签、barBeatAtSec 一致）：
 * - 弱网格线：每段内以段起点为原点、按 stepBeats 等距生成；
 * - 强网格线：每段内按该段拍号（beatsPerBar）取小节边界，段起点本身也是小节对齐点。
 */
export function buildTempoGridLines(args: {
    startSec: number;
    endSec: number;
    map: TempoMap | null;
    stepBeats: number;
    fallbackBpm: number;
    fallbackBeatsPerBar: number;
}): TempoGridLine[] {
    const { startSec, endSec, map, stepBeats, fallbackBpm, fallbackBeatsPerBar } = args;
    if (endSec < startSec) return [];
    const safeStep = Math.max(1e-9, stepBeats);
    const lines: TempoGridLine[] = [];

    const add = (sec: number, isBar: boolean) => {
        if (!Number.isFinite(sec) || sec < startSec - 1e-9 || sec > endSec + 1e-9) return;
        lines.push({ sec, isBar });
    };

    if (!map || map.points.length === 0) {
        // 无 Tempo Map：均匀网格（全局拍坐标）。
        const startBeat = secToBeat(null, startSec, fallbackBpm);
        const endBeat = secToBeat(null, endSec, fallbackBpm);
        const firstIndex = Math.floor(startBeat / safeStep - 1e-9);
        const lastIndex = Math.ceil(endBeat / safeStep + 1e-9);
        for (let index = firstIndex; index <= lastIndex; index += 1) {
            const beat = index * safeStep;
            if (beat < 0) continue;
            add(beatToSec(null, beat, fallbackBpm), false);
        }
        const bpb = Math.max(1, beatsPerBarOf({ numerator: fallbackBeatsPerBar || 4, denominator: 4 }));
        const firstBarIndex = Math.floor(startBeat / bpb + 1e-9);
        const lastBarIndex = Math.ceil(endBeat / bpb - 1e-9);
        for (let k = firstBarIndex; k <= lastBarIndex; k += 1) {
            add(beatToSec(null, k * bpb, fallbackBpm), true);
        }
        lines.sort((a, b) => a.sec - b.sec);
        return lines;
    }

    // ── Tempo Map 路径：逐段局部对齐 ──
    const segments = tempoMapSegments(map, Math.max(endSec, map.points[map.points.length - 1].positionSec));
    for (let i = 0; i < segments.length; i += 1) {
        const segment = segments[i];
        const segBpm = Math.max(1, segment.point.bpm);
        const segSecPerBeat = 60 / segBpm;
        const segBpb = Math.max(1, beatsPerBarOf(segment.point));
        const clampedStart = Math.max(segment.startSec, startSec);
        const clampedEnd = Math.min(segment.endSec, endSec);
        if (clampedEnd < clampedStart - 1e-9) continue;

        const localStartBeat = (clampedStart - segment.startSec) / segSecPerBeat;
        const localEndBeat = (clampedEnd - segment.startSec) / segSecPerBeat;

        // 弱网格线（段内 stepBeats 的整数倍）。
        const firstWeak = Math.ceil(localStartBeat / safeStep - 1e-9);
        const lastWeak = Math.floor(localEndBeat / safeStep + 1e-9);
        for (let k = firstWeak; k <= lastWeak; k += 1) {
            if (k < 0) continue;
            add(segment.startSec + k * safeStep * segSecPerBeat, false);
        }

        // 强网格线（段内小节边界；段起点本身也是对齐点）。
        const firstBar = Math.ceil(localStartBeat / segBpb - 1e-9);
        const lastBar = Math.floor(localEndBeat / segBpb + 1e-9);
        for (let k = firstBar; k <= lastBar; k += 1) {
            if (k < 0) continue;
            add(segment.startSec + k * segBpb * segSecPerBeat, true);
        }
    }

    lines.sort((a, b) => a.sec - b.sec);
    return lines;
}

// ────────────────────────────────────────────────────────────────────────────
// 编辑辅助
// ────────────────────────────────────────────────────────────────────────────

/** 插入变化点（自动排序、去除与已有点过近的点）。 */
export function insertTempoPoint(map: TempoMap, point: TempoPoint): TempoMap {
    const points = [...map.points];
    const idx = points.findIndex((p) => p.positionSec > point.positionSec + 1e-6);
    if (idx < 0) points.push(point);
    else points.splice(idx, 0, point);
    return { points };
}

export function updateTempoPoint(map: TempoMap, id: string, patch: Partial<Omit<TempoPoint, "id">>): TempoMap {
    return {
        points: map.points.map((p) => {
            if (p.id !== id) return p;
            return {
                ...p,
                ...patch,
                positionSec: Math.max(0, Number(patch.positionSec ?? p.positionSec) || 0),
                bpm: clampBpm(Number(patch.bpm ?? p.bpm)),
                numerator: clampNumerator(Number(patch.numerator ?? p.numerator)),
                denominator: clampDenominator(Number(patch.denominator ?? p.denominator)),
                scale: patch.scale === undefined ? p.scale : normalizeScaleData(patch.scale),
            };
        }),
    };
}

/** 删除变化点；删除最后一个点或第一个点（map 只剩它）时返回 null。 */
export function removeTempoPoint(map: TempoMap, id: string): TempoMap | null {
    const points = map.points.filter((p) => p.id !== id);
    if (points.length === 0) return null;
    // 第一个点必须位于 0：若删除了 0 位置的点，把下一个点钉到 0。
    if (points[0].positionSec > 1e-9) {
        points[0] = { ...points[0], positionSec: 0 };
    }
    return { points };
}

/** 创建位于 sec 的新变化点（继承该位置当前生效的 BPM/拍号；音阶跟随之前的音阶）。
 * 首次创建（map 为 null）时，0 位置初始点即工程基准记录：携带工程 BPM/拍号/音阶。 */
export function createTempoPointAt(
    map: TempoMap | null,
    sec: number,
    fallback: { bpm: number; beatsPerBar: number },
    opts?: { projectScale?: ScaleLike; projectScaleName?: string },
): { map: TempoMap; point: TempoPoint } {
    const at = tempoAtSec(map, sec, fallback);
    const point: TempoPoint = {
        id: createTempoPointId(),
        positionSec: Math.max(0, sec),
        bpm: at.bpm,
        numerator: at.numerator,
        denominator: at.denominator,
        scale: null,
    };
    if (!map) {
        const first: TempoPoint = {
            id: createTempoPointId(),
            positionSec: 0,
            bpm: clampBpm(fallback.bpm),
            numerator: clampNumerator(fallback.beatsPerBar || 4),
            denominator: 4,
            // 初始点即工程基准记录：携带工程音阶（键或自定义音级）。
            scale: scaleLikeToScaleData(opts?.projectScale, opts?.projectScaleName),
        };
        if (point.positionSec <= 1e-9) {
            return { map: { points: [point] }, point };
        }
        return { map: { points: [first, point] }, point };
    }
    return { map: insertTempoPoint(map, point), point };
}

/**
 * 时间轴上产生 Tempo Map 数据时同步的“工程基准值”：
 * 始终取 0 位置点的 BPM / 每小节拍数，保证删除 Tempo Map 后工程回退一致。
 */
export function tempoMapFallbackValues(map: TempoMap | null, fallback: { bpm: number; beatsPerBar: number }) {
    if (map && map.points.length > 0) {
        const first = map.points[0];
        return { bpm: first.bpm, beatsPerBar: first.numerator };
    }
    return fallback;
}

/** 将 map 中 0 位置点的 BPM/拍号替换为给定工程值（保持其他点不变）。 */
export function withProjectFallbackValues(
    map: TempoMap | null,
    fallback: { bpm: number; beatsPerBar: number },
): TempoMap | null {
    if (!map || map.points.length === 0) return map;
    const first = map.points[0];
    return {
        points: [
            {
                ...first,
                bpm: clampBpm(fallback.bpm),
                numerator: clampNumerator(fallback.beatsPerBar || 4),
            },
            ...map.points.slice(1),
        ],
    };
}

// ────────────────────────────────────────────────────────────────────────────
// 标尺行渲染辅助
// ────────────────────────────────────────────────────────────────────────────

/**
 * 生成当前可见范围（带缓冲）内的显式网格线 x 像素坐标（内容坐标系），
 * 供 BackgroundGrid 直接绘制。无 Tempo Map 时返回 null（走均匀网格路径）。
 */
export function buildTempoGridLineXsForViewport(args: {
    tempoMap: TempoMap | null;
    scrollLeft: number;
    viewportWidth: number;
    pxPerSec: number;
    projectSec: number;
    stepBeats: number;
    fallbackBpm: number;
    fallbackBeatsPerBar: number;
}): { weak: number[]; strong: number[] } | null {
    const {
        tempoMap,
        scrollLeft,
        viewportWidth,
        pxPerSec,
        projectSec,
        stepBeats,
        fallbackBpm,
        fallbackBeatsPerBar,
    } = args;
    if (!tempoMap || tempoMap.points.length === 0) return null;
    const bufferPx = Math.max(240, (Number.isFinite(viewportWidth) ? viewportWidth : 0) * 0.5);
    const startSec = Math.max(0, (scrollLeft - bufferPx) / Math.max(1e-9, pxPerSec));
    const endSec = Math.min(
        projectSec,
        (scrollLeft + viewportWidth + bufferPx) / Math.max(1e-9, pxPerSec),
    );

    // 密度上限与均匀网格一致：弱网格 ~160 条、强网格 ~48 条。
    const MAX_WEAK = 160;
    const MAX_STRONG = 48;
    let weakStep = Math.max(1e-9, stepBeats);
    let weakLines = buildTempoGridLines({
        startSec,
        endSec,
        map: tempoMap,
        stepBeats: weakStep,
        fallbackBpm,
        fallbackBeatsPerBar,
    });
    while (weakLines.filter((l) => !l.isBar).length > MAX_WEAK && weakStep < 256) {
        weakStep *= 2;
        weakLines = buildTempoGridLines({
            startSec,
            endSec,
            map: tempoMap,
            stepBeats: weakStep,
            fallbackBpm,
            fallbackBeatsPerBar,
        });
    }
    const weak = dedupeSorted(
        weakLines.filter((l) => !l.isBar).map((l) => l.sec * pxPerSec),
    );
    let strongLines = weakLines.filter((l) => l.isBar);
    if (strongLines.length > MAX_STRONG) {
        const stride = Math.ceil(strongLines.length / MAX_STRONG);
        strongLines = strongLines.filter((_, index) => index % stride === 0);
    }
    const strong = dedupeSorted(strongLines.map((l) => l.sec * pxPerSec));
    return { weak, strong };
}

/** 去重（保留升序），消除相邻段边界处重复生成的网格线。 */
function dedupeSorted(values: number[]): number[] {
    const out: number[] = [];
    for (const v of values) {
        if (out.length === 0 || Math.abs(out[out.length - 1] - v) > 0.5) {
            out.push(v);
        }
    }
    return out;
}

export interface TempoSegment {
    startSec: number;
    endSec: number;
    point: TempoPoint;
    /** 该段生效音阶（null = 跟随工程音阶）。 */
    scale: ScaleLike | null;
}

/** 把 TempoMap 展开为显示段（截断到 [0, projectSec]）。 */
export function tempoMapSegments(map: TempoMap | null, projectSec: number): TempoSegment[] {
    const endSec = Math.max(0, projectSec);
    if (!map || map.points.length === 0) {
        return [];
    }
    const segments: TempoSegment[] = [];
    for (let i = 0; i < map.points.length; i += 1) {
        const startSec = map.points[i].positionSec;
        const nextSec = i + 1 < map.points.length ? map.points[i + 1].positionSec : endSec;
        if (startSec > endSec) break;
        segments.push({
            startSec,
            endSec: Math.max(startSec, Math.min(nextSec, endSec)),
            point: map.points[i],
            // 仅当该变化点显式携带音阶覆盖时才非 null（null = 跟随工程音阶）。
            scale: scaleDataToScaleLike(map.points[i].scale),
        });
    }
    return segments;
}

/**
 * 计算 [startSec, endSec] 内逐段生效的音阶（合并相邻同音阶段）。
 * 返回 null 表示无 Tempo Map 音阶数据（调用方可退回单音阶渲染路径）。
 * 段内 scale 为 null 表示该段跟随工程音阶（不高亮）。
 */
export function buildScaleSegments(
    map: TempoMap | null,
    projectScale: ScaleLike | undefined,
    startSec: number,
    endSec: number,
): Array<{ startSec: number; endSec: number; scale: ScaleLike | null }> | null {
    if (!map || map.points.length === 0) return null;
    const segments = tempoMapSegments(map, endSec);
    if (segments.length === 0) return null;

    const out: Array<{ startSec: number; endSec: number; scale: ScaleLike | null }> = [];
    let current: ScaleLike | null | undefined = undefined;
    for (const seg of segments) {
        if (seg.endSec <= startSec + 1e-9) continue;
        if (seg.startSec >= endSec - 1e-9) continue;
        const explicit = scaleDataToScaleLike(seg.point.scale);
        if (explicit) current = explicit;
        else if (current === undefined) current = projectScale ?? null;
        const clampedStart = Math.max(seg.startSec, startSec);
        const clampedEnd = Math.min(seg.endSec, endSec);
        const effective = current ?? null;
        const last = out[out.length - 1];
        if (
            last &&
            Math.abs(last.endSec - clampedStart) < 1e-6 &&
            scaleLikeEquals(last.scale, effective)
        ) {
            last.endSec = clampedEnd;
            continue;
        }
        out.push({ startSec: clampedStart, endSec: clampedEnd, scale: effective });
    }
    return out;
}

function scaleLikeEquals(a: ScaleLike | null, b: ScaleLike | null): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    if (Array.isArray(a) && Array.isArray(b)) {
        const na = normalizeCustomScaleNotes(a);
        const nb = normalizeCustomScaleNotes(b);
        return na.length === nb.length && na.every((v, i) => v === nb[i]);
    }
    return !Array.isArray(a) && !Array.isArray(b) && a === b;
}
