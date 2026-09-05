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
 * 变化点处的音阶为 null 时表示“跟随之前的音阶”（透明：继续使用之前最近的
 * 显式音阶，全无显式音阶时使用工程音阶）；
 * 变化点处的拍号为 null 时表示“跟随之前的拍号”（0 位置初始点必须显式携带拍号）。
 */

import type { ScaleLike } from "./musicalScales.ts";
import {
    SCALE_KEYS,
    SCALE_LABELS,
    isScaleKey,
    normalizeCustomScaleNotes,
} from "./musicalScales.ts";

// ────────────────────────────────────────────────────────────────────────────
// 类型
// ────────────────────────────────────────────────────────────────────────────

/** 变化点携带的音阶数据；null 表示跟随之前的音阶（透明，见模块注释）。 */
export interface TempoMapScaleData {
    /** 内置音阶键名（如 "C"、"Db"）。 */
    key?: string;
    /** 自定义音阶名称。 */
    name?: string;
    /** 自定义音阶音级集合（0-11）。 */
    notes?: number[];
}

/** 拍号（如 3/4 的分子 3、分母 4）。 */
export interface TempoTimeSignature {
    numerator: number;
    denominator: number;
}

export interface TempoPoint {
    id: string;
    /** 绝对秒位置（时间锚定）。 */
    positionSec: number;
    /** BPM。 */
    bpm: number;
    /** 拍号；null = 跟随之前的拍号（0 位置初始点必须显式）。 */
    timeSignature: TempoTimeSignature | null;
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
export function beatsPerBarOf(sig: TempoTimeSignature): number {
    const denominator = clampDenominator(sig.denominator);
    return (clampNumerator(sig.numerator) * 4) / denominator;
}

/** 构造已钳制的拍号。 */
export function makeTimeSignature(numerator: number, denominator: number): TempoTimeSignature {
    return {
        numerator: clampNumerator(numerator),
        denominator: clampDenominator(denominator),
    };
}

/** 从不可信数据构造拍号；分子/分母任一缺失或非法时返回 null。 */
export function normalizeTimeSignature(
    value:
        | {
              numerator?: number | null;
              denominator?: number | null;
          }
        | null
        | undefined,
): TempoTimeSignature | null {
    if (!value) return null;
    const numerator = Number(value.numerator);
    const denominator = Number(value.denominator);
    if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) return null;
    return makeTimeSignature(numerator, denominator);
}

/** 格式化 BPM 显示（整数则无小数；其余保留最多 3 位小数，去除多余尾零）。 */
export function formatTempoBpm(bpm: number): string {
    const rounded = Math.round(bpm * 1000) / 1000;
    return String(rounded);
}

/** 拍号显示文本，如 "4/4"、"3/4"。 */
export function formatTimeSignature(sig: TempoTimeSignature): string {
    return `${clampNumerator(sig.numerator)}/${clampDenominator(sig.denominator)}`;
}

// ────────────────────────────────────────────────────────────────────────────
// 规范化 / 序列化
// ────────────────────────────────────────────────────────────────────────────

/**
 * 规范化 TempoMap：
 * - 按位置排序；位置过近（< 1e-6s）的重复点只保留第一个；
 * - 第一个点必须位于 0（不足则用 fallback 值补一个，并携带工程音阶 —— 初始点即工程基准记录）；
 * - 第一个点必须显式携带拍号（它是工程基准记录，不存在“之前”可跟随）；
 *   其余点拍号为 null 时表示“跟随之前的拍号”。
 * - 无点返回 null。
 */
export function normalizeTempoMap(
    map: TempoMap | null | undefined,
    fallbackBpm: number,
    fallbackBeatsPerBar: number,
    opts?: {
        projectScale?: ScaleLike;
        projectScaleName?: string;
        /** 工程基准拍号分母（无 Tempo Map 时的工程记录值；默认 4）。 */
        projectDenominator?: number;
    },
): TempoMap | null {
    if (!map || !Array.isArray(map.points)) return null;

    const fallback = clampNumerator(fallbackBeatsPerBar || 4);
    const fallbackDenominator = clampDenominator(opts?.projectDenominator ?? 4);
    const raw: TempoPoint[] = [];
    for (const rawPoint of map.points) {
        if (!rawPoint || typeof rawPoint.id !== "string" || !rawPoint.id) continue;
        raw.push({
            id: rawPoint.id,
            positionSec: Math.max(0, Number(rawPoint.positionSec) || 0),
            bpm: clampBpm(Number(rawPoint.bpm)),
            timeSignature: normalizeTimeSignature(rawPoint.timeSignature),
            scale: normalizeScaleData(rawPoint.scale),
        });
    }
    // 先排序再去重：输入乱序时，相邻的重复位置（如 [5, 0, 0.0000005]）必须
    // 在排序后才能可靠合并，否则会同时保留，破坏“位置严格递增”的契约。
    raw.sort((a, b) => a.positionSec - b.positionSec);
    const points: TempoPoint[] = [];
    for (const p of raw) {
        if (
            points.length > 0 &&
            Math.abs(points[points.length - 1].positionSec - p.positionSec) < 1e-6
        ) {
            continue;
        }
        points.push(p);
    }
    if (points.length === 0) return null;

    if (points[0].positionSec > 1e-9) {
        points.unshift({
            id: createTempoPointId(),
            positionSec: 0,
            bpm: clampBpm(fallbackBpm),
            timeSignature: { numerator: fallback, denominator: fallbackDenominator },
            // 初始点即工程基准记录：携带工程音阶（键或自定义音级）。
            scale: scaleLikeToScaleData(opts?.projectScale, opts?.projectScaleName),
        });
    }
    points[0].positionSec = 0;
    // 初始点必须显式携带拍号：缺失（非法输入）时用工程基准值物化。
    if (!points[0].timeSignature) {
        points[0] = {
            ...points[0],
            timeSignature: { numerator: fallback, denominator: fallbackDenominator },
        };
    }
    return { points };
}

export function normalizeScaleData(
    scale: TempoMapScaleData | null | undefined,
): TempoMapScaleData | null {
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
    opts?: {
        projectScale?: ScaleLike;
        projectScaleName?: string;
        projectDenominator?: number;
    },
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
                    numerator?: number | null;
                    denominator?: number | null;
                    scale?: TempoMapScaleData | null;
                };
                const timeSignature =
                    item.numerator != null && item.denominator != null
                        ? makeTimeSignature(Number(item.numerator), Number(item.denominator))
                        : null;
                return {
                    id: String(item.id ?? createTempoPointId()),
                    positionSec: Number(item.positionSec ?? 0),
                    bpm: Number(item.bpm ?? fallbackBpm),
                    timeSignature,
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
 * 拍号跟随之前的拍号时 numerator/denominator 序列化为 null。
 */
export function toBackendTempoMap(map: TempoMap | null): Array<{
    id: string;
    positionSec: number;
    bpm: number;
    numerator: number | null;
    denominator: number | null;
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
        numerator: p.timeSignature?.numerator ?? null,
        denominator: p.timeSignature?.denominator ?? null,
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

/** 工程基准拍号（无 Tempo Map 时退化为该值；防御性默认 4/4）。 */
export const DEFAULT_TIME_SIGNATURE: TempoTimeSignature = { numerator: 4, denominator: 4 };

/**
 * 计算每个变化点“生效拍号”的一趟扫描结果（与 points 等长）：
 * 从前往后携带最近一个显式拍号；0 位置点必须显式（规范化保证），
 * 因此任何位置都有确定的值。
 */
export function effectiveTimeSignatures(map: TempoMap): TempoTimeSignature[] {
    let carry: TempoTimeSignature | null = null;
    return map.points.map((point) => {
        if (point.timeSignature) carry = point.timeSignature;
        return carry ?? DEFAULT_TIME_SIGNATURE;
    });
}

/** 下标处变化点的生效拍号（跟随之前的拍号时解析为实际值）。 */
export function effectiveTimeSignatureAt(map: TempoMap, index: number): TempoTimeSignature {
    const safeIndex = Math.min(Math.max(0, index), map.points.length - 1);
    return effectiveTimeSignatures(map)[safeIndex];
}

/** 某绝对秒位置生效的拍号；无 Tempo Map 时退化为工程基准值。 */
export function timeSignatureAtSec(
    map: TempoMap | null,
    sec: number,
    fallback: { beatsPerBar: number; denominator?: number },
): TempoTimeSignature {
    if (!map || map.points.length === 0) {
        return {
            numerator: clampNumerator(fallback.beatsPerBar || 4),
            denominator: clampDenominator(fallback.denominator ?? 4),
        };
    }
    return effectiveTimeSignatureAt(map, pointIndexAtSec(map, sec));
}

/**
 * 查询某个变化点“之前”生效的拍号（用于“跟随之前的拍号”文案与解析）：
 * 0 位置初始点之前的拍号即工程基准记录；其它位置返回其前一个点的生效拍号。
 */
export function previousTimeSignatureAtSec(
    map: TempoMap | null,
    positionSec: number,
    fallback: { beatsPerBar: number; denominator?: number },
): TempoTimeSignature {
    if (positionSec <= 1e-9) {
        return {
            numerator: clampNumerator(fallback.beatsPerBar || 4),
            denominator: clampDenominator(fallback.denominator ?? 4),
        };
    }
    return timeSignatureAtSec(map, positionSec - 1e-6, fallback);
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
    const sig = effectiveTimeSignatureAt(map, pointIndexAtSec(map, sec));
    return {
        bpm: point.bpm,
        numerator: sig.numerator,
        denominator: sig.denominator,
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
 * - 从该位置向前找最近一个携带音阶的变化点（音阶为 null 的变化点是透明的，
 *   表示“跟随之前的音阶”，不重置为工程音阶）；
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
 * 查询 [startSec, endSec] 范围内生效过的音阶变化（位置 + 音阶）。
 *
 * 除范围内的显式变化点外，还包含“管辖范围起点”的变化点（起点之前最近一个
 * 显式音阶变化）—— 它虽然位于范围之外，但决定了范围起点的生效音阶，
 * 用于判断“工程音阶”选项是否受 Tempo Map 影响时不能遗漏。
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
        const scale = scaleDataToScaleLike(point.scale);
        if (!scale) continue;
        if (point.positionSec < startSec - 1e-9) {
            // 管辖范围起点的变化点：始终保留（放在结果首位）。
            if (out.length === 0) {
                out.push({ positionSec: point.positionSec, scale });
            } else {
                out[0] = { positionSec: point.positionSec, scale };
            }
            continue;
        }
        out.push({ positionSec: point.positionSec, scale });
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
        // 与 snapSecToTempoGrid / buildTempoGridLines 一致：BPM 必须钳制，
        // 否则 0/负 BPM 会产生 0/负拍速（除零 → NaN/Infinity，负值 → 拍数递减）。
        const rate = Math.max(1, clampBpm(point.bpm)) / 60;
        beatRates.push(rate);
        if (i + 1 < map.points.length) {
            const next = map.points[i + 1];
            const spanSec = Math.max(0, next.positionSec - point.positionSec);
            pointBeats.push(pointBeats[i] + spanSec * rate);
        }
    }
    return { map, fallbackBpm: safeBpm, pointBeats, beatRates };
}

// 进程级 memo：secToBeat / beatToSec / beatToBarBeat 会被标尺刻度格式化
// 每帧调用上百次，直接调 buildBeatCache 意味着每帧 O(N)×数百次的数组
// 分配。TempoMap 在编辑时走不可变更新（新建 map/points），因此按
// map + points 引用 + fallbackBpm 缓存是安全的。
const beatCacheMemo = new WeakMap<TempoMap, TempoMapBeatCache>();

function getBeatCache(map: TempoMap, fallbackBpm: number): TempoMapBeatCache {
    const cached = beatCacheMemo.get(map);
    if (cached && cached.fallbackBpm === fallbackBpm && cached.map.points === map.points) {
        return cached;
    }
    const built = buildBeatCache(map, fallbackBpm);
    beatCacheMemo.set(map, built);
    return built;
}

/** 秒 → 全局拍。无 Tempo Map 时退化为恒定 BPM。 */
export function secToBeat(map: TempoMap | null, sec: number, fallbackBpm: number): number {
    const safeSec = Math.max(0, sec);
    if (!map || map.points.length === 0) {
        return (safeSec * Math.max(1, fallbackBpm || 120)) / 60;
    }
    const cache = getBeatCache(map, fallbackBpm);
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
    const cache = getBeatCache(map, fallbackBpm);
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
        const bpb = Math.max(
            1,
            beatsPerBarOf({ numerator: fallbackBeatsPerBar || 4, denominator: 4 }),
        );
        const barIndex = Math.floor(safeBeat / bpb);
        const inBar = safeBeat - barIndex * bpb;
        const beatIndex = Math.floor(inBar);
        return { bar: barIndex + 1, beat: beatIndex + 1, sub: inBar - beatIndex };
    }

    const cache = getBeatCache(map, fallbackBpm);
    const { points } = map;
    // 每段生效拍号（跟随之前的拍号时解析为实际值）。
    const segmentSigs = effectiveTimeSignatures(map);
    let bar = 1;

    for (let i = 0; i < points.length; i += 1) {
        const segStartBeat = cache.pointBeats[i];
        const segEndBeat = i + 1 < points.length ? cache.pointBeats[i + 1] : Infinity;
        const bpb = Math.max(1, beatsPerBarOf(segmentSigs[i]));

        if (safeBeat < segEndBeat || !Number.isFinite(segEndBeat)) {
            const rel = safeBeat - segStartBeat;
            let fullBars = Math.floor(rel / bpb);
            const inBar = rel - fullBars * bpb;
            let beatIndex = Math.floor(inBar);
            let sub = inBar - beatIndex;
            // 消除浮点误差：拍内余量无限接近 1 时进位到下一拍
            // （否则标尺会出现 "4.2.1000" 这类不存在的格式）。
            if (sub > 1 - 1e-9) {
                beatIndex += 1;
                sub = 0;
                if (beatIndex >= bpb) {
                    fullBars += 1;
                    beatIndex = 0;
                }
            }
            return {
                bar: bar + fullBars,
                beat: beatIndex + 1,
                sub,
            };
        }

        const segLen = segEndBeat - segStartBeat;
        const fullBars = Math.floor(segLen / bpb);
        const leftover = segLen - fullBars * bpb;
        bar += fullBars + (leftover > 1e-9 ? 1 : 0);
    }
    // 末段的 segEndBeat 为 Infinity，循环必然在末段内 return ——
    // 此处不可达（旧版在这里有一段"超出工程末尾外推"的死代码）。
    return { bar, beat: 1, sub: 0 };
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
 * 吸附到网格：把秒位置吸附到最近的网格线。
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
    /** Swing 强度（0-100）：弱网格奇数格向后偏移半步的最大百分比。 */
    swingPercent?: number;
    /** 强网格（小节线）抽取步长：仅绘制 k % strongStride === 0 的小节线（密度上限优化）。 */
    strongStride?: number;
}): TempoGridLine[] {
    const { startSec, endSec, map, stepBeats, fallbackBpm, fallbackBeatsPerBar } = args;
    if (endSec < startSec) return [];
    const safeStep = Math.max(1e-9, stepBeats);
    const swingPercent = Math.max(0, Math.min(100, args.swingPercent ?? 0));
    const strongStride = Math.max(1, Math.floor(args.strongStride ?? 1));
    const lines: TempoGridLine[] = [];

    const add = (sec: number, isBar: boolean) => {
        if (!Number.isFinite(sec) || sec < startSec - 1e-9 || sec > endSec + 1e-9) return;
        lines.push({ sec, isBar });
    };
    const swingAt = (segBpm: number, index: number) => {
        if (swingPercent <= 0 || index % 2 === 0) return 0;
        return (swingPercent / 100) * 0.5 * safeStep * (60 / Math.max(1, segBpm));
    };

    if (!map || map.points.length === 0) {
        // 无 Tempo Map：均匀网格（全局拍坐标）。
        const startBeat = secToBeat(null, startSec, fallbackBpm);
        const endBeat = secToBeat(null, endSec, fallbackBpm);
        const firstIndex = Math.floor(startBeat / safeStep - 1e-9);
        const lastIndex = Math.ceil(endBeat / safeStep + 1e-9);
        for (let index = firstIndex; index <= lastIndex; index += 1) {
            if (index < 0) continue;
            const beat = index * safeStep;
            add(beatToSec(null, beat, fallbackBpm) + swingAt(fallbackBpm, index), false);
        }
        const bpb = Math.max(
            1,
            beatsPerBarOf({ numerator: fallbackBeatsPerBar || 4, denominator: 4 }),
        );
        const firstBarIndex = Math.floor(startBeat / bpb + 1e-9);
        const lastBarIndex = Math.ceil(endBeat / bpb - 1e-9);
        for (let k = firstBarIndex; k <= lastBarIndex; k += 1) {
            add(beatToSec(null, k * bpb, fallbackBpm), true);
        }
        lines.sort((a, b) => a.sec - b.sec);
        return lines;
    }

    // ── Tempo Map 路径：逐段局部对齐 ──
    const segments = tempoMapSegments(
        map,
        Math.max(endSec, map.points[map.points.length - 1].positionSec),
    );
    for (let i = 0; i < segments.length; i += 1) {
        const segment = segments[i];
        const segBpm = Math.max(1, segment.point.bpm);
        const segSecPerBeat = 60 / segBpm;
        const segBpb = Math.max(1, segment.beatsPerBar);
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
            add(segment.startSec + k * safeStep * segSecPerBeat + swingAt(segBpm, k), false);
        }

        // 强网格线（段内小节边界；段起点本身也是对齐点）。
        const firstBar = Math.ceil(localStartBeat / segBpb - 1e-9);
        const lastBar = Math.floor(localEndBeat / segBpb + 1e-9);
        for (let k = firstBar; k <= lastBar; k += 1) {
            if (k < 0) continue;
            if (k % strongStride !== 0) continue;
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
    // 与已有点过近（< 1e-6s）时不插入（规范化契约：不允许重复位置）。
    if (points.some((p) => Math.abs(p.positionSec - point.positionSec) < 1e-6)) {
        return { points };
    }
    const idx = points.findIndex((p) => p.positionSec > point.positionSec + 1e-6);
    if (idx < 0) points.push(point);
    else points.splice(idx, 0, point);
    return { points };
}

/**
 * 解析补丁后的位置秒数。
 *
 * 非有限值（NaN / ±Infinity）时**保留原值**，而不是折叠成 0。旧实现写作
 * `Number(x) || 0`，一旦上游吸附算出 NaN，标记会被静默瞬移到工程起点；接着它
 * 与 0 位置的工程基准点重合，被 `normalizeTempoMap` 的相邻去重直接删除，表现
 * 为"拖动一下标记就飞到最左边，然后消失、再也拖不动"。
 */
function resolvePatchedPositionSec(patched: number | undefined, current: number): number {
    const candidate = Number(patched ?? current);
    const base = Number.isFinite(candidate) ? candidate : Number(current);
    return Math.max(0, Number.isFinite(base) ? base : 0);
}

export function updateTempoPoint(
    map: TempoMap,
    id: string,
    patch: Partial<Omit<TempoPoint, "id">>,
): TempoMap {
    const positionPatched = patch.positionSec !== undefined;
    const points = map.points.map((p) => {
        if (p.id !== id) return p;
        return {
            ...p,
            ...patch,
            positionSec: resolvePatchedPositionSec(patch.positionSec, p.positionSec),
            bpm: clampBpm(Number(patch.bpm ?? p.bpm)),
            timeSignature:
                patch.timeSignature === undefined
                    ? p.timeSignature
                    : patch.timeSignature == null
                      ? null
                      : makeTimeSignature(
                            patch.timeSignature.numerator,
                            patch.timeSignature.denominator,
                        ),
            scale: patch.scale === undefined ? p.scale : normalizeScaleData(patch.scale),
        };
    });
    if (positionPatched) {
        // 位置变化可能破坏“按 positionSec 升序”的不变量（拖拽快速跨越相邻点等），
        // 重新排序并钉住 0 位置点 —— 下游 pointIndexAtSec（二分查找）、
        // buildBeatCache（积分）等都依赖严格升序。
        points.sort((a, b) => a.positionSec - b.positionSec);
        if (points.length > 0 && points[0].positionSec > 1e-9) {
            points[0] = { ...points[0], positionSec: 0 };
        }
    }
    return { points };
}

/**
 * 把变化点的目标位置钳制到相邻点之间（拖拽用）。
 *
 * 【为什么不简单写成 `min(max(prev, sec), max(next, prev))`】
 * 旧式写法在 `next <= prev`（相邻点过近，或末点的工程上限低于前一个点）时会把
 * 上界退化成 `prev`，于是点被**永久钉死**在 `prev` 上：此后无论怎么拖都只能得
 * 到同一个值。这正是"拖一下就飞到最左边、之后再也拖不动"的成因。
 *
 * 这里保证：下界恒为 `prevSec`；上界仅在其严格大于下界时生效，否则视为无上界。
 * 因此点永远不会进入"上下界重合、动弹不得"的退化状态。
 *
 * @param maxSec 末点的位置上限（工程/视口末尾）。仅当大于下限时生效；
 *               传 `Infinity` 表示不设上限。
 */
export function clampTempoPointSec(args: {
    points: readonly TempoPoint[];
    pointId: string;
    desiredSec: number;
    /** 相邻点最小间距（秒），避免两点重合后被规范化去重。 */
    minGapSec: number;
    maxSec: number;
}): number {
    const { points, pointId, desiredSec, minGapSec, maxSec } = args;
    const index = points.findIndex((point) => point.id === pointId);
    if (index < 0) return desiredSec;

    const gap = Number.isFinite(minGapSec) ? Math.max(0, minGapSec) : 0;
    // 0 位置点是工程基准记录（调用方已禁止拖动），这里仍给出安全下界。
    const prevSec = index > 0 ? points[index - 1].positionSec + gap : 0;
    const hasNext = index + 1 < points.length;
    const rawUpperSec = hasNext ? points[index + 1].positionSec - gap : maxSec;

    const upperSec =
        Number.isFinite(rawUpperSec) && rawUpperSec > prevSec
            ? rawUpperSec
            : Number.POSITIVE_INFINITY;

    const candidate = Number.isFinite(desiredSec) ? desiredSec : prevSec;
    const clamped = Math.min(Math.max(prevSec, candidate), upperSec);
    // 数值兜底：任何异常都退回到下界，绝不返回 NaN（会瞬移到 0）。
    return Number.isFinite(clamped) ? Math.max(prevSec, clamped) : prevSec;
}

/** 删除变化点；删除最后一个点或第一个点（map 只剩它）时返回 null。 */
export function removeTempoPoint(map: TempoMap, id: string): TempoMap | null {
    const points = map.points.filter((p) => p.id !== id);
    if (points.length === 0) return null;
    // 第一个点必须位于 0：若删除了 0 位置的点，把下一个点钉到 0。
    if (points[0].positionSec > 1e-9) {
        points[0] = { ...points[0], positionSec: 0 };
        // 钉到 0 的点成为工程基准记录：必须显式携带拍号。
        if (!points[0].timeSignature) {
            const sig = effectiveTimeSignatureAt(
                map,
                map.points.findIndex((p) => p.id === points[0].id),
            );
            points[0] = { ...points[0], timeSignature: sig };
        }
    }
    return { points };
}

/**
 * 创建位于 sec 的新变化点：
 * - BPM 继承该位置当前生效值；拍号默认“跟随之前的拍号”（timeSignature: null）；
 * - 音阶默认“跟随工程音阶”（scale: null）。
 * 首次创建（map 为 null）时，0 位置初始点即工程基准记录：携带工程 BPM/拍号/音阶。
 *
 * 返回 `created` 表示是否真的插入了新点：当目标位置与既有过近点
 * （< 1e-6s）冲突时，`point` 为该**既有点**且 `created` 为 false ——
 * 调用方应选中/编辑返回的 point，而不是拿一个未插入的“幽灵点”进入
 * 选中/内联编辑状态（旧实现会让 UI 呈现一个不存在的新点）。
 */
export function createTempoPointAt(
    map: TempoMap | null,
    sec: number,
    fallback: { bpm: number; beatsPerBar: number; denominator?: number },
    opts?: { projectScale?: ScaleLike; projectScaleName?: string },
): { map: TempoMap; point: TempoPoint; created: boolean } {
    const at = tempoAtSec(map, sec, fallback);
    const point: TempoPoint = {
        id: createTempoPointId(),
        positionSec: Math.max(0, sec),
        bpm: at.bpm,
        // 新添加的变化点默认“跟随之前的拍号”。
        timeSignature: null,
        scale: null,
    };
    if (!map) {
        const first: TempoPoint = {
            id: createTempoPointId(),
            positionSec: 0,
            bpm: clampBpm(fallback.bpm),
            // 初始点即工程基准记录：显式携带工程拍号。
            timeSignature: {
                numerator: clampNumerator(fallback.beatsPerBar || 4),
                denominator: clampDenominator(fallback.denominator ?? 4),
            },
            // 初始点即工程基准记录：携带工程音阶（键或自定义音级）。
            scale: scaleLikeToScaleData(opts?.projectScale, opts?.projectScaleName),
        };
        if (point.positionSec <= 1e-9) {
            // 新建位置即 0：直接以工程基准点作为初始点（必须显式携带拍号/音阶，
            // 否则后续序列化会把 numerator/denominator 写成 null，后端按 4/4 物化，
            // 与工程基准（如 3/4）不一致）。
            return { map: { points: [first] }, point: first, created: true };
        }
        return { map: { points: [first, point] }, point, created: true };
    }
    const nextMap = insertTempoPoint(map, point);
    if (nextMap.points.includes(point)) {
        return { map: nextMap, point, created: true };
    }
    // 插入被拒（与既有过近点冲突）：返回既有点本身。
    const existing =
        map.points.find((p) => Math.abs(p.positionSec - point.positionSec) < 1e-6) ?? point;
    return { map: nextMap, point: existing, created: false };
}

/**
 * 时间轴上产生 Tempo Map 数据时同步的“工程基准值”：
 * 始终取 0 位置点的 BPM / 每小节拍数 / 拍号分母，保证删除 Tempo Map 后工程回退一致。
 */
export function tempoMapFallbackValues(
    map: TempoMap | null,
    fallback: { bpm: number; beatsPerBar: number; denominator?: number },
): { bpm: number; beatsPerBar: number; denominator: number } {
    if (map && map.points.length > 0) {
        const first = map.points[0];
        const sig = first.timeSignature ?? { numerator: 4, denominator: 4 };
        return {
            bpm: first.bpm,
            beatsPerBar: sig.numerator,
            denominator: sig.denominator,
        };
    }
    return {
        bpm: fallback.bpm,
        beatsPerBar: fallback.beatsPerBar,
        denominator: clampDenominator(fallback.denominator ?? 4),
    };
}

/** 将 map 中 0 位置点的 BPM/拍号替换为给定工程值（保持其他点不变）。 */
export function withProjectFallbackValues(
    map: TempoMap | null,
    fallback: { bpm: number; beatsPerBar: number; denominator?: number },
): TempoMap | null {
    if (!map || map.points.length === 0) return map;
    const first = map.points[0];
    return {
        points: [
            {
                ...first,
                bpm: clampBpm(fallback.bpm),
                timeSignature: {
                    numerator: clampNumerator(fallback.beatsPerBar || 4),
                    denominator: clampDenominator(fallback.denominator ?? 4),
                },
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
    /** Swing 强度（0-100）。 */
    swingPercent?: number;
    /** 用户配置的最小网格线像素间距（弱线）。 */
    minSpacingPx?: number;
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
        swingPercent = 0,
        minSpacingPx = 8,
    } = args;
    if (!tempoMap || tempoMap.points.length === 0) return null;
    // 工程末尾之后继续填充：右边界取可见范围，不再钳到 projectSec。
    void projectSec;
    const bufferPx = Math.max(240, (Number.isFinite(viewportWidth) ? viewportWidth : 0) * 0.5);
    const startSec = Math.max(0, (scrollLeft - bufferPx) / Math.max(1e-9, pxPerSec));
    const endSec = Math.max(
        startSec,
        (scrollLeft + viewportWidth + bufferPx) / Math.max(1e-9, pxPerSec),
    );

    // 密度上限与均匀网格一致：弱网格 ~160 条、强网格 ~48 条，
    // 用户自定义最小像素间距时按其换算（仍保留绝对上限防卡死）。
    const maxWeak = Math.max(
        1,
        Math.min(
            160,
            Math.floor(
                (Number.isFinite(viewportWidth) ? Math.max(0, viewportWidth) : 0) /
                    Math.max(1, minSpacingPx),
            ) || 160,
        ),
    );
    const MAX_WEAK = maxWeak;
    const MAX_STRONG = Math.max(1, Math.ceil(maxWeak / 3));
    // ★ 先用“视口跨度”估算所需步长，再开始生成 —— 若以最细网格全量生成，
    // 长工程 + 细网格（如 2 小时 @960BPM、1/64）会先分配数百万条网格线，
    // 与“消除卡死”的目标背道而驰。
    // 工程末尾之后仍继续填充网格，因此跨度不能取“生成范围”（会随右侧空白
    // 区无限变长、把工程内的网格越估越粗）；以真实视口跨度估步长，样式才能
    // 与工程长度内的网格保持一致。
    const viewportSpanSec = Math.max(
        1e-9,
        (Number.isFinite(viewportWidth) ? Math.max(0, viewportWidth) : 0) /
            Math.max(1e-9, pxPerSec),
    );
    const spanBeats = Math.max(
        1e-9,
        secToBeat(tempoMap, startSec + viewportSpanSec, fallbackBpm) -
            secToBeat(tempoMap, startSec, fallbackBpm),
    );
    let weakStep = Math.max(1e-9, stepBeats);
    while (spanBeats / weakStep > MAX_WEAK) {
        weakStep *= 2;
    }
    // 强网格（小节线）抽取步长：按工程拍号近似估算每小节拍数。
    const bpbEst = Math.max(
        1,
        beatsPerBarOf({
            numerator: fallbackBeatsPerBar || 4,
            denominator: 4,
        }),
    );
    let strongStride = Math.max(1, Math.ceil(spanBeats / bpbEst / MAX_STRONG));

    const buildLines = () =>
        buildTempoGridLines({
            startSec,
            endSec,
            map: tempoMap,
            stepBeats: weakStep,
            fallbackBpm,
            fallbackBeatsPerBar,
            swingPercent,
            strongStride,
        });

    let weakLines = buildLines();
    // 兜底：分段对齐可能让估算偏少，仍保留按实际条数加倍的逻辑。
    while (weakLines.filter((l) => !l.isBar).length > MAX_WEAK) {
        weakStep *= 2;
        weakLines = buildLines();
    }
    // 不同段拍号/速度会让强线估算失准，同样按实际条数加粗抽取步长。
    while (weakLines.filter((l) => l.isBar).length > MAX_STRONG) {
        strongStride *= 2;
        weakLines = buildLines();
    }
    const weak = dedupeSorted(weakLines.filter((l) => !l.isBar).map((l) => l.sec * pxPerSec));
    const strong = dedupeSorted(weakLines.filter((l) => l.isBar).map((l) => l.sec * pxPerSec));
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

// ────────────────────────────────────────────────────────────────────────────
// 变化点旗帜文本与“悬浮标签”（viewport 左侧的黏性参数标签）
// ────────────────────────────────────────────────────────────────────────────

/**
 * 变化点旗帜的展示文本：
 * - 拍号与音阶都跟随：`120`；
 * - 仅拍号显式：`120 4/4`；
 * - 仅音阶显式：`120 - C / Am`；
 * - 都显式：`120 4/4 - C / Am`。
 * 标尺行旗帜与视口左侧悬浮标签共用，保证内容一致。
 */
export function tempoPointFlagLabel(point: TempoPoint): string {
    let label = formatTempoBpm(point.bpm);
    if (point.timeSignature) {
        label += ` ${formatTimeSignature(point.timeSignature)}`;
    }
    const scaleLabel = tempoPointScaleShortLabel(point.scale);
    if (scaleLabel) {
        label += ` - ${scaleLabel}`;
    }
    return label;
}

/** 变化点音阶的短标签（无音阶数据时返回 null）。 */
export function tempoPointScaleShortLabel(
    scale: TempoMapScaleData | null | undefined,
): string | null {
    if (!scale) return null;
    if (scale.key) {
        return SCALE_LABELS[scale.key as keyof typeof SCALE_LABELS] ?? scale.key;
    }
    if (scale.name) return scale.name;
    return "…";
}

/** 旗帜文本宽度估算（px，9px 字号等宽近似：每字符 6px + 内边距 16px）。 */
export function tempoFlagLabelWidthPx(label: string): number {
    return label.length * 6 + 16;
}

export interface TempoFloatingLabelState {
    /** 悬浮标签应展示的文本（画面最左侧可见位置所在段的参数）。 */
    label: string;
    /** 管辖画面最左侧的旗帜是否已完全滚出画面左侧（此时才需要悬浮标签）。 */
    governingOffscreen: boolean;
    /** 是否有任何旗帜与悬浮标签显示区域重叠（重叠时隐藏悬浮标签）。 */
    blocked: boolean;
}
/**
 * 计算视口左侧“悬浮标签”的显示状态。
 *
 * 时间轴是连续的画布，而画面只是其中一个小窗口：当管辖画面最左侧的
 * 变化点旗帜（蓝色标签）滚出画面左侧后，用户无从得知该段的参数，
 * 因此在标尺行最左侧显示一个悬浮标签。规则：
 * - 管辖旗帜完全在画面左侧之外时才显示；
 * - 任何旗帜（尤其是从左侧进入画面的下一个旗帜）与悬浮标签区域
 *   重叠时隐藏，避免互相遮挡。
 */
export function computeTempoFloatingLabelState(args: {
    tempoMap: TempoMap;
    scrollLeft: number;
    pxPerSec: number;
    chipExtraWidthPx?: number;
    marginPx?: number;
}): TempoFloatingLabelState {
    const { tempoMap, scrollLeft, pxPerSec, chipExtraWidthPx = 14, marginPx = 10 } = args;
    const safePx = Math.max(1e-9, pxPerSec);
    const leftSec = Math.max(0, scrollLeft / safePx);
    // 空 map 防御：无变化点时不存在管辖旗帜（调用方通常已过滤，但这里不能越界）。
    if (tempoMap.points.length === 0) {
        return { label: "", governingOffscreen: false, blocked: false };
    }
    const idx = pointIndexAtSec(tempoMap, leftSec);
    const governing = tempoMap.points[idx];
    const label = tempoPointFlagLabel(governing);
    const chipWidthEst = tempoFlagLabelWidthPx(label) + chipExtraWidthPx;
    const gx = governing.positionSec * safePx - scrollLeft;
    const governingOffscreen = gx + tempoFlagLabelWidthPx(label) < -2;
    let blocked = false;
    for (const p of tempoMap.points) {
        const x = p.positionSec * safePx - scrollLeft;
        if (x < chipWidthEst + marginPx && x + tempoFlagLabelWidthPx(tempoPointFlagLabel(p)) > -2) {
            blocked = true;
            break;
        }
    }
    return { label, governingOffscreen, blocked };
}

/**
 * 判断某个内容坐标点击（秒）是否命中一个变化点旗帜的可视范围
 * （从点的位置向右延伸整个旗帜文本的宽度）。返回命中点的下标，未命中返回 null。
 */
export function tempoPointHitTest(
    map: TempoMap,
    clickedSec: number,
    pxPerSec: number,
): number | null {
    const clickPx = Math.max(0, clickedSec) * Math.max(1e-9, pxPerSec);
    let bestIndex: number | null = null;
    let bestDist = Infinity;
    for (let i = 0; i < map.points.length; i += 1) {
        const p = map.points[i];
        const left = p.positionSec * Math.max(1e-9, pxPerSec);
        const width = tempoFlagLabelWidthPx(tempoPointFlagLabel(p));
        const hitRight = left + width + 8;
        if (clickPx < left - 4 || clickPx > hitRight) continue;
        // 命中的点：与旗帜边缘的最小距离（用于多个旗帜重叠时选择最近者）。
        const dist = Math.min(Math.abs(clickPx - left), Math.abs(clickPx - hitRight));
        if (dist < bestDist) {
            bestDist = dist;
            bestIndex = i;
        }
    }
    return bestIndex;
}

/**
 * 解析变化点旗帜文本，供旗帜内联编辑使用。解析失败返回 null（调用方静默放弃、不应用）。
 *
 * 支持的格式与旗帜展示完全一致：
 * - `120`                    → 仅 BPM（拍号、音阶都跟随之前）；
 * - `120 4/4`                → BPM + 显式拍号（音阶跟随）；
 * - `120 - C / Am`           → BPM + 显式音阶（拍号跟随）；
 * - `120 4/4 - C / Am`       → BPM + 拍号 + 音阶。
 *
 * 规则：
 * - BPM：浮点数（钳制 10-960）；
 * - 拍号：`分子/分母`，分子 1-32、分母限 {1,2,4,8,16,32}；
 * - 音阶（可选，` - ` 之后）：匹配内置音阶显示标签（如 "C / Am"）、
 *   裸键名（如 "C"）或自定义音阶预设名；无法识别时整体解析失败；
 * - 无音阶部分 → scale = null（跟随工程音阶）；无拍号部分 → timeSignature = null（跟随之前的拍号）。
 */
export function parseTempoPointText(
    text: string,
    customPresets: ReadonlyArray<{ id: string | number; name: string; notes: number[] }>,
): {
    bpm: number;
    timeSignature: TempoTimeSignature | null;
    scale: TempoMapScaleData | null;
} | null {
    const raw = text.trim();
    if (!raw) return null;

    // `BPM [拍号][ - 音阶]`（BPM 可能带小数；音阶部分用第一个 ` - ` 分隔，
    // 音阶名自身可以包含 `-`；拍号部分仅在以数字开头时识别，避免误吞
    // `120 - C / Am` 这种“拍号跟随、音阶显式”的输入）。
    const headMatch = /^(\d+(?:\.\d+)?)(?:\s+(\d+)\s*\/\s*(\d+))?(?:\s*-\s*(.*))?$/.exec(raw);
    if (!headMatch) return null;
    const bpm = clampBpm(Number(headMatch[1]));
    const sigText = headMatch[2];
    const denText = headMatch[3];
    const scaleText = (headMatch[4] ?? "").trim();

    let timeSignature: TempoTimeSignature | null = null;
    if (sigText !== undefined && denText !== undefined) {
        const numerator = Number(sigText);
        const denominator = Number(denText);
        if (!Number.isFinite(numerator) || !Number.isFinite(denominator)) return null;
        if (numerator < 1 || numerator > TEMPO_NUMERATOR_MAX) return null;
        if (!(TEMPO_DENOMINATORS as readonly number[]).includes(denominator)) return null;
        timeSignature = {
            numerator: clampNumerator(numerator),
            denominator: clampDenominator(denominator),
        };
    }

    if (!scaleText) {
        return { bpm, timeSignature, scale: null };
    }

    // 内置键（直接键名）。
    if ((SCALE_KEYS as readonly string[]).includes(scaleText)) {
        return { bpm, timeSignature, scale: { key: scaleText } };
    }
    // 内置音阶显示标签（"C / Am" 等，反向查表）。
    for (const key of SCALE_KEYS) {
        if (SCALE_LABELS[key] === scaleText) {
            return { bpm, timeSignature, scale: { key } };
        }
    }
    // 自定义音阶预设名。
    const preset = customPresets.find((p) => p.name === scaleText);
    if (preset) {
        return {
            bpm,
            timeSignature,
            scale: { name: preset.name, notes: [...preset.notes] },
        };
    }
    return null;
}

export interface TempoSegment {
    startSec: number;
    endSec: number;
    point: TempoPoint;
    /** 该段生效音阶（null = 跟随工程音阶）。 */
    scale: ScaleLike | null;
    /** 该段生效拍号（跟随之前的拍号时已解析为实际值）。 */
    timeSignature: TempoTimeSignature;
    /** 该段生效每小节拍数。 */
    beatsPerBar: number;
}

/** 把 TempoMap 展开为显示段（截断到 [0, projectSec]）。 */
export function tempoMapSegments(map: TempoMap | null, projectSec: number): TempoSegment[] {
    const endSec = Math.max(0, projectSec);
    if (!map || map.points.length === 0) {
        return [];
    }
    const segmentSigs = effectiveTimeSignatures(map);
    const segments: TempoSegment[] = [];
    for (let i = 0; i < map.points.length; i += 1) {
        const startSec = map.points[i].positionSec;
        const nextSec = i + 1 < map.points.length ? map.points[i + 1].positionSec : endSec;
        if (startSec > endSec) break;
        const timeSignature = segmentSigs[i];
        segments.push({
            startSec,
            endSec: Math.max(startSec, Math.min(nextSec, endSec)),
            point: map.points[i],
            // 仅当该变化点显式携带音阶覆盖时才非 null（null = 跟随工程音阶）。
            scale: scaleDataToScaleLike(map.points[i].scale),
            timeSignature,
            beatsPerBar: Math.max(1, beatsPerBarOf(timeSignature)),
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

/** 两个音阶是否等价（内置键按名比较；自定义音阶按规范化音级集合比较）。 */
export function scaleLikeEquals(
    a: ScaleLike | null | undefined,
    b: ScaleLike | null | undefined,
): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    if (Array.isArray(a) && Array.isArray(b)) {
        const na = normalizeCustomScaleNotes(a);
        const nb = normalizeCustomScaleNotes(b);
        return na.length === nb.length && na.every((v, i) => v === nb[i]);
    }
    return !Array.isArray(a) && !Array.isArray(b) && a === b;
}
