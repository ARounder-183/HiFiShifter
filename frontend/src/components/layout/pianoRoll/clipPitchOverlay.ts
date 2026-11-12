import { strokeColorForClip } from "../timeline/runtime/midiPitchCurve";
import { toFrame } from "./paramBlockProjection";

/**
 * 参数编辑器 · Clip 源音高参考线（纯派生模型）。
 *
 * 数据源是后端 clip_pitch_data 推送并存入 session.clipPitchCurves 的
 * per-clip 曲线。它与根轨道合成的 pv.orig 不是一回事：
 *
 * - pv.orig 是单值合成线：重叠 Clip、音高参考块都会让不同来源在同一帧竞争，
 *   最终只留下一个值；
 * - 本模型保留每个 Clip 自己的曲线，用于重合、音高参考块覆盖和原始 F0 对照。
 *
 * 【帧基准：一律是工程帧，不是连续秒】
 * 后端 emit 时曲线第 i 个采样对应工程帧 `round(clip.startSec * 1000 / fp) + i`
 * （与 assemble_pitch_orig_from_cache 写入 pv.orig 的位置、与参数块投影
 * toFrame() 完全同源）。本模型因此只保存 `startFrame`，绝不保存
 * `curveStartSec` —— 按连续秒落笔会让每条 Clip 各偏一个亚帧相位（≤2.5 ms），
 * 在放大到 0.5 s 视口时是 8~9 px 的可见错位，用户会以为是自己操作错了。
 *
 * 本文件不碰 React / DOM / GL；只负责把 Clip 几何、曲线、选中态和重合关系
 * 变成可绘制的分段描述符，供 sceneBuilders / render.ts 共用。
 */

const EPS = 1e-9;

/** ClipInfo 中本模块需要的字段（保持结构类型，便于测试）。 */
export interface ClipPitchOverlayClip {
    readonly id: string;
    readonly trackId: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly muted?: boolean;
    readonly color?: string;
    readonly sourcePath?: string;
    readonly midiNoteData?: readonly unknown[];
}

export interface ClipPitchCurveData {
    readonly curveStartSec: number;
    readonly midiCurve: readonly number[];
    readonly framePeriodMs: number;
}

export interface ClipPitchOverlaySegment {
    /** 曲线数组中的起始下标（含）。 */
    readonly startIndex: number;
    /** 曲线数组中的结束下标（不含）。 */
    readonly endIndexExclusive: number;
    /** 该段同期的 Clip 重合深度；1 = 不重合，2 = 两条 Clip 重合。 */
    readonly overlapDepth: number;
}

export interface ClipPitchOverlay {
    readonly clipId: string;
    /** 曲线第 0 个采样所在的工程帧（= round(clip.startSec * 1000 / fp)）。 */
    readonly startFrame: number;
    /** 该 Clip 在工程帧网格上的帧数（= round(clip.lengthSec * 1000 / fp)）。 */
    readonly frameCount: number;
    readonly framePeriodMs: number;
    readonly midiCurve: readonly number[];
    readonly kind: "sourceF0" | "midiReference";
    readonly strokeColor: string;
    readonly selected: boolean;
    readonly overlapDepth: number;
    readonly zOrder: number;
    readonly segments: readonly ClipPitchOverlaySegment[];
}

export interface ClipPitchOverlayVisual {
    readonly alpha: number;
    readonly lineWidthPx: number;
    readonly dash: readonly [number, number] | null;
}

interface OverlapBand {
    readonly startSec: number;
    readonly endSec: number;
    readonly depth: number;
}

function clipStart(clip: ClipPitchOverlayClip): number {
    return Math.max(0, Number.isFinite(clip.startSec) ? clip.startSec : 0);
}

function clipLength(clip: ClipPitchOverlayClip): number {
    return Number.isFinite(clip.lengthSec) ? Math.max(0, clip.lengthSec) : 0;
}

function clipEnd(clip: ClipPitchOverlayClip): number {
    return clipStart(clip) + clipLength(clip);
}

/**
 * 计算一个 Clip 区间内所有「至少两个 Clip 同时存在」的区段，以及最大重合深度。
 *
 * 只使用 Clip 的工程区间；参数块是否被上层块覆盖与音频是否真的重合无关。
 * 重合只是「提高 alpha」的判据，不是几何来源，因此继续用秒（与 Clip 字段同单位），
 * 不引入第二次取整。
 */
function overlapBandsForClip(
    clip: ClipPitchOverlayClip,
    clips: readonly ClipPitchOverlayClip[],
): { bands: OverlapBand[]; maxDepth: number } {
    const start = clipStart(clip);
    const end = clipEnd(clip);
    if (!(end > start + EPS)) return { bands: [], maxDepth: 1 };

    const boundaries = new Set<number>([start, end]);
    for (const other of clips) {
        if (other.id === clip.id) continue;
        const otherStart = clipStart(other);
        const otherEnd = clipEnd(other);
        const overlapStart = Math.max(start, otherStart);
        const overlapEnd = Math.min(end, otherEnd);
        if (overlapEnd > overlapStart + EPS) {
            boundaries.add(overlapStart);
            boundaries.add(overlapEnd);
        }
    }
    const points = [...boundaries].sort((a, b) => a - b);
    const bands: OverlapBand[] = [];
    let maxDepth = 1;
    for (let i = 0; i + 1 < points.length; i += 1) {
        const from = points[i]!;
        const to = points[i + 1]!;
        if (!(to > from + EPS)) continue;
        const mid = (from + to) / 2;
        let depth = 1;
        for (const other of clips) {
            if (other.id === clip.id) continue;
            if (clipStart(other) <= mid && clipEnd(other) > mid) depth += 1;
        }
        maxDepth = Math.max(maxDepth, depth);
        if (depth >= 2) {
            const last = bands[bands.length - 1];
            if (last !== undefined && Math.abs(last.endSec - from) <= EPS) {
                bands[bands.length - 1] = {
                    startSec: last.startSec,
                    endSec: to,
                    depth: Math.max(last.depth, depth),
                };
            } else {
                bands.push({ startSec: from, endSec: to, depth });
            }
        }
    }
    return { bands, maxDepth };
}

function depthAtSec(bands: readonly OverlapBand[], sec: number): number {
    let depth = 1;
    for (const band of bands) {
        if (sec >= band.startSec - EPS && sec < band.endSec - EPS) {
            depth = Math.max(depth, band.depth);
        }
    }
    return depth;
}

/**
 * ClipPitchCurves → per-clip 参考线。
 *
 * 过滤规则与后端 emit 契约对齐（全部按**工程帧**判断，与绘制位置同一把尺子）：
 * - `round(curveStartSec·1000/fp)` 必须等于 `round(clip.startSec·1000/fp)`；
 *   移动不足半帧时网格没变，曲线仍然有效；超过半帧立即作废；
 * - 曲线帧数必须覆盖 Clip 的帧数，否则整条丢弃（不拉伸、不补零）；
 * - 有声段按 midi <= 0 / 非有限值断开；重合深度变化也断开，便于分段着色。
 */
export function buildClipPitchOverlays(args: {
    readonly clips: readonly ClipPitchOverlayClip[];
    readonly curves: Readonly<Record<string, ClipPitchCurveData>>;
    readonly groupTrackIds: ReadonlySet<string>;
    readonly selectedClipIds: ReadonlySet<string>;
    /**
     * 工程的帧周期（毫秒），来自参数视图。已知时必须与曲线自带的帧周期一致，
     * 否则曲线会整体缩放。缺省（参数视图还没取到）时不做这项校验。
     */
    readonly projectFramePeriodMs?: number;
}): ClipPitchOverlay[] {
    const { clips, curves, groupTrackIds, selectedClipIds, projectFramePeriodMs } = args;
    const overlays: ClipPitchOverlay[] = [];
    clips.forEach((clip, zOrder) => {
        if (!groupTrackIds.has(clip.trackId) || clip.muted === true) return;
        const curve = curves[clip.id];
        if (curve === undefined || !Number.isFinite(curve.curveStartSec)) return;
        const framePeriodMs = curve.framePeriodMs;
        if (!Number.isFinite(framePeriodMs) || framePeriodMs <= 0) return;
        // 曲线与工程不在同一个帧周期上 ⇒ 画出来必然整体缩放，宁可整条不画。
        if (
            projectFramePeriodMs != null &&
            Number.isFinite(projectFramePeriodMs) &&
            Math.abs(framePeriodMs - projectFramePeriodMs) > 0.1
        ) {
            return;
        }
        const frameSec = framePeriodMs / 1000;

        const startFrame = toFrame(clipStart(clip), framePeriodMs);
        const frameCount = toFrame(clipLength(clip), framePeriodMs);
        if (frameCount < 2) return;
        // 旧缓存属于别的几何：起点落在另一个工程帧上就整条丢弃。
        if (toFrame(curve.curveStartSec, framePeriodMs) !== startFrame) return;
        // 曲线必须覆盖整个 Clip；短了说明是分割 / 拉伸前的旧数据。
        if (curve.midiCurve.length < frameCount) return;

        const { bands, maxDepth } = overlapBandsForClip(clip, clips);
        const segments: ClipPitchOverlaySegment[] = [];
        let segmentStart = -1;
        let segmentDepth = 0;
        const flush = (endExclusive: number) => {
            if (segmentStart >= 0 && endExclusive - segmentStart >= 2) {
                segments.push({
                    startIndex: segmentStart,
                    endIndexExclusive: endExclusive,
                    overlapDepth: segmentDepth,
                });
            }
            segmentStart = -1;
            segmentDepth = 0;
        };
        for (let index = 0; index < frameCount; index += 1) {
            const value = curve.midiCurve[index];
            const voiced = value != null && Number.isFinite(value) && value > 0;
            if (!voiced) {
                flush(index);
                continue;
            }
            const depth = depthAtSec(bands, (startFrame + index) * frameSec);
            if (segmentStart < 0) {
                segmentStart = index;
                segmentDepth = depth;
            } else if (depth !== segmentDepth) {
                flush(index);
                segmentStart = index;
                segmentDepth = depth;
            }
        }
        flush(frameCount);
        if (segments.length === 0) return;

        overlays.push({
            clipId: clip.id,
            startFrame,
            frameCount,
            framePeriodMs,
            midiCurve: curve.midiCurve,
            kind:
                clip.midiNoteData !== undefined && clip.midiNoteData.length > 0
                    ? "midiReference"
                    : "sourceF0",
            strokeColor: strokeColorForClip({ color: clip.color ?? "" }),
            selected: selectedClipIds.has(clip.id),
            overlapDepth: maxDepth,
            zOrder,
            segments,
        });
    });

    // 低层先画、高层后画、选中 Clip 最后画；否则选中线会被相邻 Clip 盖住。
    overlays.sort(
        (a, b) =>
            Number(a.selected) - Number(b.selected) ||
            a.zOrder - b.zOrder ||
            a.clipId.localeCompare(b.clipId),
    );
    return overlays;
}

/** 一条 Clip 参考线图层的视觉参数（颜色由调用方从 strokeColor 解析）。 */
export function resolveClipPitchOverlayVisual(
    overlay: ClipPitchOverlay,
    overlapDepth: number,
): ClipPitchOverlayVisual {
    const reference = overlay.kind === "midiReference";
    const baseAlpha = reference ? 0.22 : 0.3;
    const overlapAlpha = reference ? 0.4 : 0.52;
    const alpha = overlay.selected
        ? reference
            ? 0.55
            : 0.82
        : overlapDepth >= 2
          ? overlapAlpha
          : baseAlpha;
    return {
        alpha,
        lineWidthPx: overlay.selected ? (reference ? 1.4 : 1.8) : reference ? 1.0 : 1.25,
        dash: reference ? [2, 3] : null,
    };
}
