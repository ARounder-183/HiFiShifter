/**
 * Pitch Reference Clip 原始音高线渲染器（Canvas2D，内容绝对坐标系）。
 *
 * 【主要内容】把轨道行内所有 MIDI / 音高参考 clip 的音高曲线绘制进 sticky
 * 时间线的一块共享画布。绘制在**内容绝对坐标**进行（与 clip 体画布同法），
 * 由调用方先 `translate(-scrollLeftPx, -scrollTopPx)` 完成视口平移。
 *
 * 【与旧实现的关系】逻辑逐行取自原 TrackLane 内挂载的 MidiPitchTrackCanvas
 * （每轨一块 canvas + 滚动事件内 JS 改写 translate3d 补偿）。旧实现是 sticky
 * 图层迁移完成后唯一残留的「内容层 canvas + 主线程 transform 补偿」路径：
 * 原生滚动由合成器线程先行提交，滚动事件里的补偿/重绘在主线程上必然晚到达，
 * 表现为音高线在滚动/缩放时滞后于波形与 Clip 体、随后才跳回。现随 sticky 层
 * + 统一帧提交器（timelineViewportBus.register）同帧同序重绘，元素位置由
 * 浏览器合成器保证与其它 sticky 图层一致。
 *
 * 【数据来源】与旧实现一致：
 *   - 优先从 clip.midiNoteData 即时生成音高曲线（拖拽/拉伸/Slip 时实时更新）；
 *   - 回退到 Redux clipPitchCurves（后端 clip_pitch_data 事件推送）。
 *
 * 【与其他模块的关系】
 * - 消费方：`TimelinePitchLineSurface`（sticky 波形层之上的音高线图层）。
 * - 依赖：`loopRender.ts`（回绕/派生窗口数学，与音频波形画布共用）、
 *   `timelineAxis.ts`（时间↔像素唯一换算入口）。
 */

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../constants.js";
import type { ClipInfo, MidiNoteEvent } from "../../../../features/session/sessionTypes";
import {
    drawLoopMarkers,
    modEuclid,
    resolveClipContentDurationSec,
    resolveSourceEndSec,
} from "../../../../utils/loopRender.js";
import {
    secToContentPx,
    viewportEndSec,
    viewportStartSec,
    type TimelineAxis,
} from "./timelineAxis.js";

// ========================================
// 常量
// ========================================

export const FRAME_PERIOD_MS = 5;

const CLIP_COLOR_TO_STROKE: Record<string, string> = {
    blue: "rgba(96, 165, 250, 0.85)",
    violet: "rgba(167, 139, 250, 0.85)",
    emerald: "rgba(52, 211, 153, 0.85)",
    amber: "rgba(251, 191, 36, 0.85)",
    cyan: "rgba(34, 211, 238, 0.85)",
};

function strokeColorForClip(clip: { color: string }): string {
    return CLIP_COLOR_TO_STROKE[clip.color] ?? "rgba(34, 211, 238, 0.78)";
}

// ========================================
// Loop（循环源）回绕描述
// ========================================

/**
 * Loop（循环源）回绕描述 —— 与后端 clip_loop_cycle_span_sec /
 * place_note_occurrence_frames 的锚点数学逐帧一致。
 */
interface LoopCycleDescriptor {
    /** 回绕周期 D（源域秒）：音频 clip = 整个媒体文件时长；纯 MIDI clip = 窗口跨度。 */
    cycleSec: number;
    /** 正放锚点（原始 sourceStartSec，可为负，floor_mod 环绕）。 */
    fwdAnchorSec: number;
    /**
     * 倒放锚点末端：周期来自媒体时长时 clamp 到 D；周期退化为窗口跨度时
     * 保持原始 sourceEndSec（与后端 place_note_occurrence_in_loop 一致 ——
     * 否则 slip 窗口 [2,7] 的跨度 clamp 会把倒放相位错误平移 ss=2s）。
     */
    revAnchorEndSec: number;
    /**
     * 周期是否来自真实媒体时长：
     * - true：坐标是**文件域**，首个回绕点在 headDur = 进入段耗尽处；
     * - false（纯 MIDI 窗口跨度）：坐标是**窗口相对域**，入口即窗口起点，
     *   首个回绕点在一个完整窗口周期之后。
     */
    cycleFromMedia: boolean;
}

/**
 * 解析 Loop 回绕描述。
 *
 * 关键语义：所有可发声内容（带源媒体的 clip、以及纯音高参考块）的实际
 * 声音/音高按**整个内容** floor_mod 回绕（与音频波形渲染 / 后端引擎一致），
 * 音高曲线与回绕标记必须使用同一周期，否则相位错位；
 * contentDurationSec 为 null（连音符内容都无法确定）时退化为窗口跨度。
 */
function resolveLoopCycleDescriptor(args: {
    loopEnabled: boolean;
    /** 内容时长（秒）：resolveClipContentDurationSec 的结果；null = 退化。 */
    contentDurationSec: number | null;
    sourceStartSec: number;
    sourceEndSec: number;
}): LoopCycleDescriptor | null {
    if (!args.loopEnabled) return null;
    const mediaDur = args.contentDurationSec ?? 0;
    const windowSpan = Math.abs(Number(args.sourceEndSec ?? 0) - Number(args.sourceStartSec ?? 0));
    const cycleSec = mediaDur > 1e-9 ? mediaDur : windowSpan;
    if (!Number.isFinite(cycleSec) || cycleSec <= 1e-9) return null;
    const srcStart = Number(args.sourceStartSec ?? 0);
    const srcEnd = Number(args.sourceEndSec ?? 0);
    return {
        cycleSec,
        fwdAnchorSec: srcStart,
        // 与后端 place_note_occurrence_in_loop / trim_and_resample_midi 同约定：
        // 倒放锚点只 clamp 到媒体时长上界、**不做 max(0)** —— 负 source_end
        // （slip/左延伸可达）由消费端的 modEuclid（floor_mod）统一环绕；
        // 此处若钳到 0 会让曲线/标记与音频出现恒定相位差。
        revAnchorEndSec: mediaDur > 1e-9 ? Math.min(srcEnd, mediaDur) : srcEnd,
        cycleFromMedia: mediaDur > 1e-9,
    };
}

// ========================================
// 音高曲线生成与缓存
// ========================================

/**
 * 从 MIDI note data 即时生成音高曲线。
 * 逻辑与后端 emit_clip_pitch_data_for_clip 的 MIDI 分支一致，
 * 支持 source range trim、playbackRate 拉伸、reversed 倒放，
 * 以及 Loop（循环源）：按媒体时长的锚点回绕重复铺满
 * （`loopCycle` 为 null 时走非循环路径）。
 */
function generateMidiCurveFromNotes(
    notes: ReadonlyArray<Pick<MidiNoteEvent, "startSec" | "endSec" | "note">>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    const fp = Math.max(FRAME_PERIOD_MS, 0.1);
    const targetFrames = Math.max(1, Math.round((clipLengthSec * 1000) / fp));
    const curve = new Array<number>(targetFrames).fill(0);

    const pr = Number.isFinite(playbackRate) && playbackRate > 0 ? playbackRate : 1;
    const srcTotalLen = sourceEndSec - sourceStartSec;

    for (const note of notes) {
        // Loop（循环源）：按媒体时长 D 的锚点回绕放置（与音频渲染的
        // floor_mod 映射一致）。不能用窗口比较过滤可见性 —— split 产生的
        // "环绕窗口"（start > end）会把所有音符误判为越界而全部丢弃。
        if (loopCycle && note.endSec - note.startSec > 1e-9) {
            const { cycleSec, fwdAnchorSec, revAnchorEndSec } = loopCycle;
            const u0 = reversed
                ? modEuclid(revAnchorEndSec - note.endSec, cycleSec)
                : modEuclid(note.startSec - fwdAnchorSec, cycleSec);
            const firstStartFrame = Math.round(((u0 / pr) * 1000) / fp);
            const lenFrames = Math.max(
                1,
                Math.round((((note.endSec - note.startSec) / pr) * 1000) / fp),
            );
            const cycleFrames = Math.max(1, Math.round(((cycleSec / pr) * 1000) / fp));
            const noteValue = note.note;
            for (let cycleOffset = 0; cycleOffset < targetFrames; cycleOffset += cycleFrames) {
                const writeStart = cycleOffset + firstStartFrame;
                const writeEnd = Math.min(cycleOffset + firstStartFrame + lenFrames, targetFrames);
                if (writeStart >= writeEnd) break;
                for (let frame = writeStart; frame < writeEnd; frame++) {
                    if (noteValue > curve[frame] || curve[frame] <= 0) {
                        curve[frame] = noteValue;
                    }
                }
            }
            continue;
        }

        if (note.endSec <= sourceStartSec || note.startSec >= sourceEndSec) continue;
        const relStart = Math.max(0, note.startSec - sourceStartSec);
        const relEnd = Math.min(srcTotalLen, note.endSec - sourceStartSec);
        if (relEnd <= relStart) continue;

        const [effStart, effEnd] = reversed
            ? [Math.max(0, srcTotalLen - relEnd), Math.min(srcTotalLen, srcTotalLen - relStart)]
            : [relStart, relEnd];
        if (effEnd <= effStart) continue;

        const noteStartFrame = Math.round(((effStart / pr) * 1000) / fp);
        const noteEndFrame = Math.round(((effEnd / pr) * 1000) / fp);
        const noteValue = note.note;

        // 非 Loop：单次写入（Loop 已在上方 placement 分支处理）。
        const writeEnd = Math.min(noteEndFrame, targetFrames);
        for (let frame = noteStartFrame; frame < writeEnd; frame++) {
            if (noteValue > curve[frame] || curve[frame] <= 0) {
                curve[frame] = noteValue;
            }
        }
    }

    // 填补音符之间的空隙（与后端 fill_gaps_in_pitch_edit 逻辑一致）
    if (fillGaps && curve.length > 0) {
        let first = -1;
        for (let i = 0; i < curve.length; i++) {
            if (curve[i] > 0) {
                first = i;
                break;
            }
        }
        let last = -1;
        for (let i = curve.length - 1; i >= 0; i--) {
            if (curve[i] > 0) {
                last = i;
                break;
            }
        }
        if (first >= 0 && last > first) {
            let lastPitch = 0;
            for (let i = first; i <= last; i++) {
                if (curve[i] > 0) {
                    lastPitch = curve[i];
                } else if (lastPitch > 0) {
                    curve[i] = lastPitch;
                }
            }
        }
    }

    return curve;
}

/**
 * 曲线生成缓存（F6 性能修复，承自 MidiPitchTrackCanvas）：
 * 每个绘制帧对每个可见 clip 全量重算会分配 lengthSec×200 的数组
 * （10000s clip ⇒ 2M 元素），且 Loop 分支按重复周期放大写入量。绘制帧之间
 * 同一 clip 的（notes 引用, 几何参数）几乎总是不变，直接复用上次结果即可。
 *
 * 缓存键：notes 数组**引用**（WeakMap 随 clip 释放，无泄漏）+ 几何参数串。
 * 每个 notes 引用最多保留 4 份几何变体（拖拽编辑时的中间态）。
 */
const midiCurveCache = new WeakMap<
    ReadonlyArray<Pick<MidiNoteEvent, "startSec" | "endSec" | "note">>,
    Map<string, number[]>
>();
const MIDI_CURVE_CACHE_MAX_PER_NOTES = 4;

function getCachedMidiCurve(
    notes: ReadonlyArray<Pick<MidiNoteEvent, "startSec" | "endSec" | "note">>,
    clipLengthSec: number,
    sourceStartSec: number,
    sourceEndSec: number,
    playbackRate: number,
    reversed: boolean,
    fillGaps: boolean,
    loopCycle: LoopCycleDescriptor | null,
): number[] {
    let inner = midiCurveCache.get(notes);
    if (!inner) {
        inner = new Map();
        midiCurveCache.set(notes, inner);
    }
    const key =
        `${clipLengthSec}|${sourceStartSec}|${sourceEndSec}|${playbackRate}|` +
        `${reversed ? 1 : 0}|${fillGaps ? 1 : 0}|` +
        (loopCycle
            ? `${loopCycle.cycleSec}|${loopCycle.fwdAnchorSec}|${loopCycle.revAnchorEndSec}|${loopCycle.cycleFromMedia ? 1 : 0}`
            : "null");
    const hit = inner.get(key);
    if (hit) return hit;
    const curve = generateMidiCurveFromNotes(
        notes,
        clipLengthSec,
        sourceStartSec,
        sourceEndSec,
        playbackRate,
        reversed,
        fillGaps,
        loopCycle,
    );
    if (inner.size >= MIDI_CURVE_CACHE_MAX_PER_NOTES) {
        const oldest = inner.keys().next().value;
        if (oldest !== undefined) inner.delete(oldest);
    }
    inner.set(key, curve);
    return curve;
}

/**
 * clip 内容时长的对象级缓存：`resolveClipContentDurationSec` 在无元数据时
 * 需遍历全部音符，旧实现每个绘制帧每个 clip 都重算一次（曲线与回绕标记共用
 * 一处调用也救不了滚动热路径）。滚动/缩放帧 clip 对象引用不变，按对象缓存
 * 即为 O(1)；编辑后 Redux 产生新 clip 对象，旧条目随 WeakMap 释放。
 * 命中判定同时校验全部输入字段引用/值，避免任何原地变更造成陈旧值。
 */
interface ContentDurationCacheEntry {
    sourcePath: string | null;
    durationFrames: number | null;
    sourceSampleRate: number | null;
    durationSec: number | null;
    notes: ReadonlyArray<Pick<MidiNoteEvent, "endSec">> | null;
    value: number | null;
}
const contentDurationCache = new WeakMap<ClipInfo, ContentDurationCacheEntry>();

function getCachedClipContentDurationSec(clip: ClipInfo): number | null {
    const notes = clip.midiNoteData ?? null;
    const hit = contentDurationCache.get(clip);
    if (
        hit &&
        hit.sourcePath === (clip.sourcePath ?? null) &&
        hit.durationFrames === (clip.durationFrames ?? null) &&
        hit.sourceSampleRate === (clip.sourceSampleRate ?? null) &&
        hit.durationSec === (clip.durationSec ?? null) &&
        hit.notes === notes
    ) {
        return hit.value;
    }
    const value = resolveClipContentDurationSec({
        sourcePath: clip.sourcePath,
        midiNoteData: notes,
        durationFrames: clip.durationFrames,
        sourceSampleRate: clip.sourceSampleRate,
        durationSec: clip.durationSec,
    });
    contentDurationCache.set(clip, {
        sourcePath: clip.sourcePath ?? null,
        durationFrames: clip.durationFrames ?? null,
        sourceSampleRate: clip.sourceSampleRate ?? null,
        durationSec: clip.durationSec ?? null,
        notes,
        value,
    });
    return value;
}

// ========================================
// 行级绘制入口
// ========================================

export interface TrackPitchLineDrawArgs {
    ctx: CanvasRenderingContext2D;
    /** 统一坐标投影：时间↔像素与视口裁剪的唯一来源。 */
    axis: TimelineAxis;
    /** 该行（轨道）上的完整 clip 列表。 */
    clips: readonly ClipInfo[];
    /** 行顶部（内容绝对 y，CSS 像素）。 */
    rowTopPx: number;
    /** 行高（CSS 像素）。 */
    rowHeight: number;
    /** 后端推送的 per-clip 音高曲线（无 midiNoteData 的 clip 的回退数据源）。 */
    clipPitchCurves: Record<
        string,
        { curveStartSec: number; midiCurve: number[]; framePeriodMs: number }
    >;
    /** 后端推送的 per-clip 音高范围（缺省 0..127）。 */
    clipPitchRanges: Record<string, { min: number; max: number }>;
    /** 描边宽度（CSS 像素）。 */
    strokeWidthPx?: number;
}

/**
 * 绘制一行（轨道）内全部音高 clip 的原始音高线与回绕/边界标记。
 *
 * 坐标系：内容绝对坐标（调用方已 translate 掉 scrollLeftPx / scrollTopPx）。
 * 曲线被裁剪进 clip 可见范围 ∩ 行波形区（头部条带以下、行底 padding 以上），
 * 与旧每轨 canvas 的裁剪语义逐像素一致。
 */
export function drawTrackPitchLines(args: TrackPitchLineDrawArgs): void {
    const {
        ctx,
        axis,
        clips,
        rowTopPx,
        rowHeight,
        clipPitchCurves,
        clipPitchRanges,
        strokeWidthPx = 1.5,
    } = args;

    // 行波形区（与 TimelineWaveformSurface 的行几何同一套常量）。
    const areaTopPx = rowTopPx + CLIP_HEADER_HEIGHT;
    const areaHeightPx = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);

    const viewportLeftPx = axis.scrollLeftPx;
    const viewportRightPx = axis.scrollLeftPx + axis.viewportWidthPx;
    const vpStartSec = viewportStartSec(axis);
    const vpEndSec = viewportEndSec(axis);

    for (const clip of clips) {
        if (clip.midiNoteCount == null) continue;
        if (!clip.lengthSec || clip.lengthSec <= 0) continue;

        const clipStartSec = clip.startSec;
        const clipEndSec = clipStartSec + clip.lengthSec;
        const clipStartPx = secToContentPx(axis, clipStartSec);
        const clipEndPx = secToContentPx(axis, clipEndSec);

        // clip 与视口的水平交集（像素域，先裁剪再生成曲线 —— 与旧实现的
        // 秒域交集顺序等价，避免为屏外 clip 构建曲线缓存）。
        const visLeftPx = Math.max(viewportLeftPx, clipStartPx);
        const visRightPx = Math.min(viewportRightPx, clipEndPx);
        if (visRightPx <= visLeftPx) continue;

        // ── 即时生成或回退读取音高曲线 ──
        let midiCurve: number[] | undefined;
        let curveStartSec: number;
        let framePeriodMs: number;

        // 内容时长（循环周期 D）：有源媒体 → 媒体总时长；纯音高参考块 →
        // 音符内容最大结束时间。对象级缓存（见上方注释）。
        const contentDurSec = getCachedClipContentDurationSec(clip);

        if (clip.midiNoteData && clip.midiNoteData.length > 0) {
            // 曲线消费窗口：非 Loop 正放 = 起点+长度×速率（派生），
            // 倒放 = [se−len·r, se]（锚定 se，sourceStart 不参与）；
            // 与音频渲染的窗口模型一致 —— 否则延伸过的倒放 Clip 曲线
            // 整体错位（该有声处显示为空）。
            // se 仅在**缺失/非法**时回退音符范围估计 —— 合法的 0/负值
            // （倒放静音段锚点）不得被改写，否则派生链整体错位。
            const seRaw = Number.isFinite(clip.sourceEndSec)
                ? clip.sourceEndSec
                : clip.midiNoteData.reduce((max, n) => Math.max(max, n.endSec), 0);
            const srcEnd = resolveSourceEndSec({
                loopEnabled: Boolean(clip.loopEnabled),
                reversed: Boolean(clip.reversed),
                sourceStartSec: Number(clip.sourceStartSec) || 0,
                playbackRate: Math.abs(Number(clip.playbackRate) || 1),
                lengthSec: clip.lengthSec,
                sourceEndSec: seRaw,
            });
            const curveWinStart =
                !clip.loopEnabled && clip.reversed
                    ? srcEnd -
                      Math.max(0, clip.lengthSec) * Math.abs(Number(clip.playbackRate) || 1)
                    : Number(clip.sourceStartSec) || 0;
            const loopCycle = resolveLoopCycleDescriptor({
                loopEnabled: Boolean(clip.loopEnabled),
                contentDurationSec: contentDurSec,
                sourceStartSec: clip.sourceStartSec,
                sourceEndSec: srcEnd,
            });
            midiCurve = getCachedMidiCurve(
                clip.midiNoteData,
                clip.lengthSec,
                curveWinStart,
                srcEnd,
                clip.playbackRate,
                clip.reversed,
                clip.midiFillGaps ?? false,
                loopCycle,
            );
            curveStartSec = clipStartSec;
            framePeriodMs = FRAME_PERIOD_MS;
        } else {
            const pitchData = clipPitchCurves[clip.id];
            if (!pitchData || !pitchData.midiCurve || pitchData.midiCurve.length < 2) continue;
            midiCurve = pitchData.midiCurve;
            curveStartSec = pitchData.curveStartSec ?? clipStartSec;
            framePeriodMs = pitchData.framePeriodMs || FRAME_PERIOD_MS;
        }

        if (!midiCurve || midiCurve.length < 2) continue;

        // 计算音高范围
        const pitchRange = clipPitchRanges[clip.id];
        const minNote = pitchRange?.min ?? 0;
        const maxNote = pitchRange?.max ?? 127;
        const noteSpan = Math.max(1, maxNote - minNote);

        // 曲线的时间跨度
        const curveDurationSec = (midiCurve.length * framePeriodMs) / 1000;
        const curveEndSec = curveStartSec + curveDurationSec;

        // 曲线与可见区域的交集（在时间线上；同时受 clip 自身范围约束）
        const overlapStartSec = Math.max(vpStartSec, clipStartSec, curveStartSec);
        const overlapEndSec = Math.min(vpEndSec, clipEndSec, curveEndSec);
        if (overlapEndSec <= overlapStartSec) continue;

        // 映射到帧索引范围
        const frameStartFrac = ((overlapStartSec - curveStartSec) * 1000) / framePeriodMs;
        const frameEndFrac = ((overlapEndSec - curveStartSec) * 1000) / framePeriodMs;
        const frameStart = Math.max(0, Math.floor(frameStartFrac));
        const frameEnd = Math.min(midiCurve.length - 1, Math.ceil(frameEndFrac));
        if (frameEnd <= frameStart) continue;

        // 每帧对应的画布像素步长
        const frameToPx = (framePeriodMs / 1000) * axis.pxPerSec;

        // 绘制连续折线（内容绝对坐标，水平+竖直同时裁剪进行波形区）
        ctx.save();
        ctx.beginPath();
        ctx.rect(visLeftPx, areaTopPx, visRightPx - visLeftPx, areaHeightPx);
        ctx.clip();

        const clipColor = strokeColorForClip(clip);
        ctx.strokeStyle = clipColor;
        ctx.lineWidth = strokeWidthPx;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";

        const alpha = clip.muted ? 0.4 : 0.85;
        ctx.globalAlpha = alpha;

        const padding = areaHeightPx * 0.1;
        let pathStarted = false;
        const minFrameStep = Math.max(1, Math.floor(0.5 / Math.max(0.01, frameToPx)));

        for (let fi = frameStart; fi <= frameEnd; fi += minFrameStep) {
            const midiValue = midiCurve[fi];
            if (midiValue <= 0) {
                pathStarted = false;
                continue;
            }

            const frameTimeSec = curveStartSec + (fi * framePeriodMs) / 1000;
            const x = secToContentPx(axis, frameTimeSec);

            const normalized = (midiValue - minNote) / noteSpan;
            const y = areaHeightPx - padding - normalized * (areaHeightPx - 2 * padding);
            const clampedY = Math.max(padding, Math.min(areaHeightPx - padding, y));

            if (!pathStarted) {
                ctx.moveTo(x, areaTopPx + clampedY);
                pathStarted = true;
            } else {
                ctx.lineTo(x, areaTopPx + clampedY);
            }
        }

        ctx.stroke();
        ctx.restore();

        // ── 循环节点倒三角标记 ──
        // 周期与曲线平铺一致：内容时长 D（对象级缓存，见上方）。
        const markerCycle = resolveLoopCycleDescriptor({
            loopEnabled: Boolean(clip.loopEnabled),
            contentDurationSec: contentDurSec,
            sourceStartSec: clip.sourceStartSec,
            sourceEndSec: Number(clip.sourceEndSec) || 0,
        });
        const markerRate =
            Math.abs(Number(clip.playbackRate ?? 1) || 1) < 1e-6
                ? 1
                : Math.abs(Number(clip.playbackRate ?? 1) || 1);
        const markerBodyDur = markerCycle ? markerCycle.cycleSec / markerRate : 0;
        if (markerBodyDur > 0 && clip.lengthSec > markerBodyDur + 1e-6) {
            // 标记必须锚定在**实际回绕点**：与音频波形的分段边界一致 ——
            // - 周期来自媒体：头部进入段耗尽处（headDur）及此后每个
            //   整文件周期边界；
            // - 纯 MIDI 窗口跨度（窗口相对域）：入口即窗口起点，
            //   首个回绕点在一个完整周期之后。
            // 不能一律用 k·周期：窗口不从文件原点进入时（trim/split），
            // 标记会与音频/曲线相位错开。
            const desc = markerCycle;
            const headDur = !desc
                ? 0
                : desc.cycleFromMedia
                  ? (clip.reversed
                        ? desc.revAnchorEndSec
                        : desc.cycleSec - modEuclid(desc.fwdAnchorSec, desc.cycleSec)) / markerRate
                  : markerBodyDur;
            const markers: number[] = [];
            {
                // 直接跳到可视范围内的第一个回绕点：既避免从 clip 入口
                // 逐周期空转数千次，也修复"深入长循环 clip 后标记消失"。
                const visLocalStart = vpStartSec - clipStartSec;
                const k0 = Math.max(0, Math.ceil((visLocalStart - headDur - 1e-6) / markerBodyDur));
                for (
                    let markerT = headDur + k0 * markerBodyDur;
                    markerT < clip.lengthSec - 1e-6 && markers.length < 4096;
                    markerT += markerBodyDur
                ) {
                    const mx = secToContentPx(axis, clipStartSec + markerT);
                    if (mx > viewportRightPx + 8) break;
                    // 恰好在 clip 起点/终点的回绕点不绘制（loopRender 约定；
                    // 倒放整文件 Loop 的 revAnchor=D 时 headDur=0 会命中）。
                    if (markerT <= 1e-6) continue;
                    if (mx < viewportLeftPx - 8) continue;
                    markers.push(Math.round(mx * 2) / 2);
                }
            }
            if (markers.length > 0) {
                ctx.save();
                ctx.translate(0, areaTopPx);
                drawLoopMarkers(ctx, markers, areaHeightPx, clipColor);
                ctx.restore();
            }
        } else if (contentDurSec != null) {
            // ── 非 Loop：媒体/内容边界标记 ──
            // 循环节 = 源媒体（或音符内容）在该 Clip 内的真实起始/
            // 终止位置（音频与静音的分界线），落在 Clip 内部时绘制。
            // 投影按**消费方向**：正放 t=(b−ss)/r；倒放 t=(se−b)/r
            // （倒放锚定窗口终点，与音频波形渲染一致）。
            const mediaDur = contentDurSec;
            {
                const srcEndResolved = Number(clip.sourceEndSec) || 0;
                const markers: number[] = [];
                for (const b of [0, mediaDur]) {
                    const tLocal = clip.reversed
                        ? (srcEndResolved - b) / markerRate
                        : (b - (Number(clip.sourceStartSec) || 0)) / markerRate;
                    if (tLocal <= 1e-6 || tLocal >= clip.lengthSec - 1e-6) continue;
                    const mx = secToContentPx(axis, clipStartSec + tLocal);
                    if (mx < viewportLeftPx - 8 || mx > viewportRightPx + 8) continue;
                    markers.push(Math.round(mx * 2) / 2);
                }
                if (markers.length > 0) {
                    ctx.save();
                    ctx.translate(0, areaTopPx);
                    drawLoopMarkers(ctx, markers, areaHeightPx, clipColor);
                    ctx.restore();
                }
            }
        }
    }
}
