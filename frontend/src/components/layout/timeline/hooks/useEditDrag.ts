import { useRef } from "react";
import { batch } from "react-redux";
import { store, type AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import {
    bumpParamsEpoch,
    checkpointHistory,
    moveClipStart,
    setClipAutoFades,
    setClipFades,
    setClipGain,
    setClipLength,
    setClipPlaybackRate,
    setClipSnapOffset,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipSourceRange,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import {
    applyAutoCrossfade,
    applyDetachedAutoCrossfadeClears,
    computeInitialCrossfadeSides,
    previewAutoCrossfade,
} from "./autoCrossfade";
import { clamp } from "../math";
import { advanceFineAxisDrag, type FineAxisDragState } from "../fineAxisDrag";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import type { TimelineSnapSettings } from "../../../../features/session/sessionTypes";
import { resolveClipContentDurationSec } from "../../../../utils/loopRender";
import {
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
} from "../../../../utils/loopSnap";
import {
    beginSnapGesture,
    computeEffectiveSnap,
    endSnapGesture,
} from "../../../../utils/timelineSnapping";
import {
    SNAP_HIGHLIGHT_GROUP,
    buildLoopBoundaryHighlightEntry,
    clearSnapHighlights,
    publishSnapHighlights,
} from "../../../../utils/snapHighlight";
import type { SnapObjectKind, SnapResult } from "../../../../utils/timelineSnapping";
import type { SnapTimelineOpts } from "./useTimelineState";
import { paramsApi } from "../../../../services/api";
import { webApi } from "../../../../services/webviewApi";
import {
    buildStretchGroupState,
    computeStretchGroupUpdate,
    scaleClipFadesForStretch,
    type StretchGroupState,
} from "./stretchGroup";
import { applyBulkFadeValue, applyBulkGainDeltaDb, getBulkEditableClipIds } from "./bulkClipEdit";
import { expandClipIdsWithGroups } from "./useGroupExpansion";
import { buildBulkClipStateUpdates } from "./bulkClipRemotePayloads";
import {
    applyRippleFollowerShift,
    buildRippleFollowers,
    type RippleFollowerMap,
    type RippleMode,
} from "../../../../features/session/ripplePreview";

const CLIP_GAIN_DRAG_DB_PER_PX = 0.25;

/**
 * 拉伸同步 SnapOffset：偏移点标记 Clip 内容中的位置，随长度按**总比例**
 *（nextLen / baseLen）线性缩放，并钳制到新长度内。必须使用拖拽起始的
 * 基准偏移 —— 逐帧读实时值再乘本帧比例会跨帧复合，呈超线性增长。
 * offset=0 时保持 0。
 */
function scaleSnapOffsetForStretch(
    baseOffsetSec: number | undefined,
    totalRatio: number,
    nextLengthSec: number,
): number {
    const base = Number(baseOffsetSec) || 0;
    if (!(base > 0)) return 0;
    const safeRatio = Number.isFinite(totalRatio) && totalRatio > 0 ? totalRatio : 1;
    return clamp(base * safeRatio, 0, Math.max(0, nextLengthSec));
}

/**
 * Loop（循环源）：把源域数值归一化到 [0, 媒体时长)。
 *
 * 裁短/延伸左缘会推进或回退进入锚点；锚点越过文件边界时按
 * 整个媒体文件取模环绕（floor_mod），与渲染/引擎的回绕映射一致。
 * 媒体时长未知时原样返回。
 */
function wrapIntoMediaDomain(
    valueSec: number,
    base: {
        durationFrames: number | null;
        sourceSampleRate: number | null;
        durationSec: number | null;
    },
): number {
    const d = (() => {
        if (base.durationFrames && base.sourceSampleRate && base.sourceSampleRate > 0) {
            return base.durationFrames / base.sourceSampleRate;
        }
        return base.durationSec || 0;
    })();
    if (!(d > 1e-9)) return valueSec;
    let v = valueSec % d;
    if (v < 0) v += d;
    return v;
}

export function resolveStretchParamTypes(
    pitchEditUserModified: boolean | null | undefined,
): Array<"pitch" | "tension"> {
    // 未手动编辑的 pitch 曲线由后端根据 clip 几何自动重建，
    // 前端若再次映射会造成“二次拉伸”。
    if (pitchEditUserModified === false) {
        return ["tension"];
    }
    return ["pitch", "tension"];
}

/**
 * 拉伸后对参数线进行时域映射（拉伸或压缩）。
 * 将旧范围 [oldStartSec, oldStartSec+oldLengthSec] 内的参数值，
 * 线性重映射到新范围 [newStartSec, newStartSec+newLengthSec]，
 * 并将不再被音频块覆盖的旧帧恢复为原始值。
 */
async function stretchLinkedParams(
    trackId: string,
    oldStartSec: number,
    oldLengthSec: number,
    newStartSec: number,
    newLengthSec: number,
): Promise<void> {
    if (
        Math.abs(oldLengthSec - newLengthSec) < 1e-6 &&
        Math.abs(oldStartSec - newStartSec) < 1e-6
    ) {
        return;
    }

    // 获取帧周期（通过最小量探针请求）。
    // 同时读取 pitch_edit_user_modified 以决定是否应手动映射 pitch。
    const probe = await paramsApi.getParamFrames(trackId, "pitch", 0, 1, 1);
    if (!probe?.ok) return;
    const fp = Math.max(1, Number(probe.frame_period_ms) || 5);
    const stretchParams = resolveStretchParamTypes(probe.pitch_edit_user_modified);

    const oldStartFrame = Math.round((oldStartSec * 1000) / fp);
    const oldEndFrame = Math.round(((oldStartSec + oldLengthSec) * 1000) / fp);
    const oldFrameCount = Math.max(1, oldEndFrame - oldStartFrame);

    const newStartFrame = Math.round((newStartSec * 1000) / fp);
    const newEndFrame = Math.round(((newStartSec + newLengthSec) * 1000) / fp);
    const newFrameCount = Math.max(1, newEndFrame - newStartFrame);

    for (const paramType of stretchParams) {
        const res = await paramsApi.getParamFrames(
            trackId,
            paramType,
            oldStartFrame,
            oldFrameCount,
            1,
        );
        if (!res?.ok) continue;
        const oldValues = (res.edit ?? []).map((v) => Number(v) || 0);
        if (oldValues.length === 0) continue;

        // 线性插值时域映射：用旧帧值填充新帧
        const newValues = new Array<number>(newFrameCount);
        const oldMaxIdx = oldValues.length - 1;
        const newMaxIdx = newFrameCount > 1 ? newFrameCount - 1 : 1;
        const ratio = oldMaxIdx / newMaxIdx;

        for (let i = 0; i < newFrameCount; i++) {
            const oldIdxF = i * ratio;
            const lo = oldIdxF | 0;
            const hi = lo < oldMaxIdx ? lo + 1 : oldMaxIdx;
            const frac = oldIdxF - lo;
            const loVal = oldValues[lo] ?? 0;
            const hiVal = oldValues[hi] ?? 0;
            if (paramType === "pitch") {
                // pitch=0 表示无效（无声）帧，保留 0
                if (loVal === 0 && hiVal === 0) {
                    newValues[i] = 0;
                } else if (loVal === 0) {
                    newValues[i] = 0;
                } else if (hiVal === 0) {
                    newValues[i] = frac < 0.5 ? loVal : 0;
                } else {
                    newValues[i] = loVal + (hiVal - loVal) * frac;
                }
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        }

        // 将重映射后的值写入新范围
        await paramsApi.setParamFrames(trackId, paramType, newStartFrame, newValues, false);

        // 恢复旧范围中不再被新音频块覆盖的帧（还原到原始值）
        const newRangeMax = newStartFrame + newFrameCount - 1;
        const oldRangeMax = oldStartFrame + oldFrameCount - 1;

        if (oldStartFrame < newStartFrame) {
            const clearLen = newStartFrame - oldStartFrame;
            void paramsApi.restoreParamFrames(trackId, paramType, oldStartFrame, clearLen, false);
        }
        if (oldRangeMax > newRangeMax) {
            const clearFrom = newRangeMax + 1;
            const clearLen = oldRangeMax - newRangeMax;
            void paramsApi.restoreParamFrames(trackId, paramType, clearFrom, clearLen, false);
        }
    }
}

type StretchRangeMapping = {
    oldStartSec: number;
    oldLengthSec: number;
    newStartSec: number;
    newLengthSec: number;
};

function buildMappedParamValues(
    oldValues: number[],
    paramType: "pitch" | "tension",
    newFrameCount: number,
): number[] {
    const newValues = new Array<number>(newFrameCount);
    const oldMaxIdx = oldValues.length - 1;
    const newMaxIdx = newFrameCount > 1 ? newFrameCount - 1 : 1;
    const ratio = oldMaxIdx / newMaxIdx;

    for (let i = 0; i < newFrameCount; i++) {
        const oldIdxF = i * ratio;
        const lo = oldIdxF | 0;
        const hi = lo < oldMaxIdx ? lo + 1 : oldMaxIdx;
        const frac = oldIdxF - lo;
        const loVal = oldValues[lo] ?? 0;
        const hiVal = oldValues[hi] ?? 0;
        if (paramType === "pitch") {
            if (loVal === 0 && hiVal === 0) {
                newValues[i] = 0;
            } else if (loVal === 0) {
                newValues[i] = 0;
            } else if (hiVal === 0) {
                newValues[i] = frac < 0.5 ? loVal : 0;
            } else {
                newValues[i] = loVal + (hiVal - loVal) * frac;
            }
        } else {
            newValues[i] = loVal + (hiVal - loVal) * frac;
        }
    }

    return newValues;
}

function subtractIntervals(
    range: { start: number; end: number },
    excluded: Array<{ start: number; end: number }>,
): Array<{ start: number; end: number }> {
    const sorted = excluded
        .filter((item) => item.end >= range.start && item.start <= range.end)
        .sort((a, b) => a.start - b.start);
    const result: Array<{ start: number; end: number }> = [];
    let cursor = range.start;

    for (const item of sorted) {
        if (cursor < item.start) {
            result.push({
                start: cursor,
                end: Math.min(item.start - 1, range.end),
            });
        }
        cursor = Math.max(cursor, item.end + 1);
        if (cursor > range.end) break;
    }

    if (cursor <= range.end) {
        result.push({ start: cursor, end: range.end });
    }

    return result;
}

/**
 * Stretch parameter lines for several clips on the same track as one batch.
 *
 * Per-clip independent writes were racing with per-clip restores: the old
 * range of one clip can overlap the new range of a neighbouring clip, so a
 * restore could erase freshly written mapped values. This version writes all
 * new ranges first, then only restores old-range parts that are not covered
 * by any new range on that track.
 */
async function stretchTrackLinkedParams(
    trackId: string,
    mappings: StretchRangeMapping[],
): Promise<void> {
    if (mappings.length === 0) return;

    const probe = await paramsApi.getParamFrames(trackId, "pitch", 0, 1, 1);
    if (!probe?.ok) return;
    const fp = Math.max(1, Number(probe.frame_period_ms) || 5);
    const stretchParams = resolveStretchParamTypes(probe.pitch_edit_user_modified);

    const frameMappings = mappings.map((mapping) => {
        const oldStartFrame = Math.round((mapping.oldStartSec * 1000) / fp);
        const oldEndFrame = Math.round(((mapping.oldStartSec + mapping.oldLengthSec) * 1000) / fp);
        const oldFrameCount = Math.max(1, oldEndFrame - oldStartFrame);

        const newStartFrame = Math.round((mapping.newStartSec * 1000) / fp);
        const newEndFrame = Math.round(((mapping.newStartSec + mapping.newLengthSec) * 1000) / fp);
        const newFrameCount = Math.max(1, newEndFrame - newStartFrame);

        return { oldStartFrame, oldFrameCount, newStartFrame, newFrameCount };
    });

    const newRanges = frameMappings.map((mapping) => ({
        start: mapping.newStartFrame,
        end: mapping.newStartFrame + mapping.newFrameCount - 1,
    }));

    for (const paramType of stretchParams) {
        const fetched = await Promise.all(
            frameMappings.map(async (mapping) => {
                const res = await paramsApi.getParamFrames(
                    trackId,
                    paramType,
                    mapping.oldStartFrame,
                    mapping.oldFrameCount,
                    1,
                );
                if (!res?.ok) return null;
                return {
                    mapping,
                    values: (res.edit ?? []).map((value) => Number(value) || 0),
                };
            }),
        );

        // Write every new range first so no restore can clobber a fresh write.
        for (const item of fetched) {
            if (!item || item.values.length === 0) continue;
            const newValues = buildMappedParamValues(
                item.values,
                paramType,
                item.mapping.newFrameCount,
            );
            await paramsApi.setParamFrames(
                trackId,
                paramType,
                item.mapping.newStartFrame,
                newValues,
                false,
            );
        }

        // Restore only old-range parts not covered by any new range.
        for (const mapping of frameMappings) {
            const oldEndFrame = mapping.oldStartFrame + mapping.oldFrameCount - 1;
            const restoreSegments = subtractIntervals(
                { start: mapping.oldStartFrame, end: oldEndFrame },
                newRanges,
            );
            for (const segment of restoreSegments) {
                await paramsApi.restoreParamFrames(
                    trackId,
                    paramType,
                    segment.start,
                    segment.end - segment.start + 1,
                    false,
                );
            }
        }
    }
}

export type EditDragType =
    | "trim_left"
    | "trim_right"
    | "stretch_left"
    | "stretch_right"
    | "fade_in"
    | "fade_out"
    | "gain"
    | "crossfade_edges";

export type EditDragState = {
    type: EditDragType;
    pointerId: number;
    clipId: string;
    basestartSec: number;
    baselengthSec: number;
    basePlaybackRate: number;
    baseSourceStartSec: number;
    baseSourceEndSec: number;
    basefadeInSec: number;
    basefadeOutSec: number;
    /** 淡入淡出相对拖拽的指针锚点（秒）：用于“相对偏移”而不是“边缘线对齐指针”。 */
    basePointerSec: number;
    baseGain: number;
    sourceBeats: number | null;
    rightEdgeBeat: number;
    baseReversed: boolean;
    baseDurationFrames: number | null;
    baseSourceSampleRate: number | null;
    baseDurationSec: number | null;
    stretchGroup: StretchGroupState | null;
    selectedClipIds: string[];
    baseGainById: Record<string, number>;
    /** 拖拽开始时读取的波纹模式（与后端提交时读到的设置保持一致）。 */
    rippleMode: RippleMode;
    /** 波纹跟随集：clipId → 初始起点（仅波纹开启且有跟随对象时非空）。 */
    rippleFollowers: RippleFollowerMap;
    /**
     * 自动交叉淡化：编辑前每侧重叠关系（被编辑剪辑 + 编辑前直接重叠邻居）。
     * 用于编辑导致“分开”时，只清掉自动交叉淡化、保留手动 fade（预览/提交一致）。
     */
    initialCrossfadeSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
    /** 自动交叉淡化真正受影响的 clip（被编辑 + 波纹随动 follower）。 */
    crossfadeClipIds: string[];
    /** 被编辑 clip 中本次编辑真正触碰的侧（trim/stretch 只影响对应边缘）。 */
    editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
    /** 交叉点拖拽：后一个 clip（较晚开始）的 id；仅类型为 "crossfade_edges" 时有效。 */
    crossfadePartnerClipId: string | null;
    /** 交叉点拖拽基础数据（仅 "crossfade_edges" 使用）。 */
    crossfadeBaseOverlapSec: number;
    crossfadeBaseFadeOutAuto: boolean;
    crossfadePartnerFadeInSec: number;
    crossfadePartnerFadeInAuto: boolean;
    /** Per-clip base state for multi-clip trim operations */
    baseByClipId: Record<
        string,
        {
            startSec: number;
            lengthSec: number;
            playbackRate: number;
            sourceStartSec: number;
            sourceEndSec: number;
            reversed: boolean;
            /** Loop（循环源）：延伸/裁短不受源媒体长度限制。 */
            loopEnabled: boolean;
            /** 拖拽起始基准 SnapOffset：拉伸按"总比例×基准"线性缩放。 */
            snapOffsetSec: number;
            durationFrames: number | null;
            sourceSampleRate: number | null;
            durationSec: number | null;
            /** 内容时长（媒体总时长 / 音符内容范围），用于循环节吸附。 */
            contentDurationSec: number | null;
        }
    >;
};

/**
 * 计算被编辑剪辑区域“当前最右缘 − 初始最右缘”的净位移（**带符号**）。
 *
 * 与后端区域化波纹一致：平移量 = 区域右缘的实际位移（含吸附、素材长度限制等
 * 约束后的真实值），确保“预览 → 提交”不跳变。
 *
 * ⚠️ 必须是带符号：拖右缘向左（缩短/截短）时位移为负，跟随剪辑要向左收拢。
 * 不能用“对 0 取 max”或“对各成员取最大正位移”的方式，否则负位移会被吞掉、
 * 向右正常而向左无实时波纹（曾为此引入 bug）。
 */
function computeRegionRightEdgeDelta(
    drag: EditDragState,
    clips: SessionState["clips"],
): number {
    let maxOldRight = Number.NEGATIVE_INFINITY;
    let maxNewRight = Number.NEGATIVE_INFINITY;
    for (const id of drag.selectedClipIds) {
        const base = drag.baseByClipId[id];
        const now = clips.find((c) => c.id === id);
        if (!base || !now) continue;
        maxOldRight = Math.max(maxOldRight, base.startSec + base.lengthSec);
        maxNewRight = Math.max(maxNewRight, Number(now.startSec) + Number(now.lengthSec));
    }
    if (!Number.isFinite(maxOldRight) || !Number.isFinite(maxNewRight)) {
        return 0;
    }
    return maxNewRight - maxOldRight;
}

export function useEditDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    /** 完整吸附结果入口（返回 SnapResult；负责发布吸附竖线高亮）。 */
    snapTimelineDetailed: (
        sec: number,
        object: SnapObjectKind,
        opts?: SnapTimelineOpts,
    ) => SnapResult;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 吸附全局开关 */
    snapEnabled: boolean;
    /** 完整吸附设置：循环节吸附距离读自 snapDistancePx（无论 enabled 与否都生效）。 */
    timelineSnap: TimelineSnapSettings;
    /** 当前缩放（像素/秒）：用于把吸附距离换算成秒。 */
    pxPerSec: number;
    /** 忽略编组 */
    ignoreGrouping: boolean;
    /** modifier.paramFineAdjust 绑定 */
    paramFineAdjustKb: Keybinding;
    /** modifier.clipCrossfadeGrip 绑定（交叉点手柄拖拽反向/缩放模式） */
    crossfadeGripKb: Keybinding;
}) {
    const {
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        snapTimelineDetailed,
        beatFromClientX,
        noSnapKb,
        snapEnabled,
        timelineSnap,
        pxPerSec,
        ignoreGrouping,
        paramFineAdjustKb,
        crossfadeGripKb,
    } = deps;

    const editDragRef = useRef<EditDragState | null>(null);
    // 用于节流向后端发送 clip 状态更新，避免拖动时频繁覆盖与后端同步引起闪烁
    const lastRemoteSentRef = useRef<Record<string, number>>({});

    function startEditDrag(e: React.PointerEvent, clipId: string, type: EditDragType) {
        if (e.button !== 0) return;
        const clip = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!clip) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const rightEdgeBeat = clip.startSec + clip.lengthSec;
        // 淡入淡出的相对拖拽锚点：以“鼠标按下位置”（deferred 起点通过 dragStartClientX
        // 传入）作为零偏移，而不是“实时把边缘线对齐到指针位置”。这样从包络线中间
        // 开始拖拽也不会发生跳变。
        const dragStartClientX =
            (e as unknown as { dragStartClientX?: number }).dragStartClientX ?? e.clientX;
        const basePointerSec = beatFromClientX(
            dragStartClientX,
            scroller.getBoundingClientRect(),
            scroller.scrollLeft,
        );

        // Resolve which clips to operate on.
        // Trim / stretch / slip expand to all selected + their group members.
        // Gain / fades only apply to multi-selected clips (no group expansion).
        const initialIds = getBulkEditableClipIds({
            activeClipId: clipId,
            multiSelectedClipIds,
            multiSelectedSet,
        });
        const supportsGroupExpansion =
            !ignoreGrouping && type !== "fade_in" && type !== "fade_out" && type !== "gain";
        let selectedClipIds = supportsGroupExpansion
            ? expandClipIdsWithGroups(
                  initialIds,
                  sessionRef.current.clips,
                  false,
                  sessionRef.current.disabledGroupIds,
              )
            : initialIds;
        // 交叉点拖拽：只作用于参与交叉淡化的两个 clip（不可扩展编组）。
        const crossfadePartnerClipId =
            type === "crossfade_edges"
                ? ((e as unknown as { crossfadePartnerClipId?: string }).crossfadePartnerClipId ??
                  null)
                : null;
        if (type === "crossfade_edges") {
            if (!crossfadePartnerClipId) return;
            selectedClipIds = [clipId, crossfadePartnerClipId];
        }
        const baseGainById = Object.fromEntries(
            selectedClipIds.map((id) => {
                const selectedClip = sessionRef.current.clips.find((entry) => entry.id === id);
                return [id, Number(selectedClip?.gain ?? 1) || 1];
            }),
        ) as Record<string, number>;
        const stretchGroup =
            type === "stretch_left" || type === "stretch_right"
                ? buildStretchGroupState({
                      clips: sessionRef.current.clips,
                      selectedClipIds,
                      anchorClipId: clipId,
                      edge: type,
                  })
                : null;

        dispatch(checkpointHistory());
        dispatch(beginInteraction());
        // 参与吸附的编辑类型登记吸附手势（stretch 此前漏登记，
        // 会导致工具栏吸附状态与竖线高亮在拉伸拖拽中不生效）。
        const beginsSnapGesture =
            type === "trim_left" ||
            type === "trim_right" ||
            type === "stretch_left" ||
            type === "stretch_right" ||
            type === "crossfade_edges";
        if (beginsSnapGesture) {
            beginSnapGesture();
        }

        // Gain drag sends throttled backend preview updates while dragging. Open the
        // backend undo group up front so the single checkpoint is the pre-drag value;
        // otherwise the final bulk write would checkpoint after the previews already
        // changed the backend, and undo would bounce back to the post-drag value.
        const gainUndoGroupPromise = type === "gain" ? webApi.beginUndoGroup() : null;

        // Fade drag（fade_in / fade_out）同理：拖动过程中节流写入多次后端，
        // 若各自独立生成 undo entry 会导致“撤销一次只撤销一半、需要按多次”。
        // 用一个 undo group 把整个 fade 拖拽包成单次撤销步。
        const fadeUndoGroupPromise =
            type === "fade_in" || type === "fade_out" ? webApi.beginUndoGroup() : null;

        // 交叉点拖拽同样把两个 clip 的修改并入同一个撤销步。
        const crossfadeUndoGroupPromise =
            type === "crossfade_edges" ? webApi.beginUndoGroup() : null;

        // 波纹（自动跟进）实时预览：拖拽开始时快照“后续跟随剪辑”的初始位置。
        // 区域语义与后端一致：原点 = 被编辑剪辑的最早起点；作用域轨道 = 全部被编辑
        // 剪辑所在轨道。只对右缘类编辑（trim_right / stretch_right）产生非零右缘位移。
        const rippleMode = sessionRef.current.rippleMode;
        let rippleOrigin = clip.startSec;
        const rippleTracks = new Set<string>([String(clip.trackId)]);
        for (const id of selectedClipIds) {
            const editedClip = id === clipId ? clip : sessionRef.current.clips.find((x) => x.id === id);
            if (!editedClip) continue;
            rippleOrigin = Math.min(rippleOrigin, editedClip.startSec);
            rippleTracks.add(String(editedClip.trackId));
        }
        const rippleFollowers = buildRippleFollowers(
            sessionRef.current.clips,
            new Set(selectedClipIds),
            rippleOrigin,
            rippleMode,
            rippleTracks,
        );

        // 自动交叉淡化：真正受影响的 clip = 被编辑剪辑 + 波纹随动 follower。
        // 只快照“编辑前直接重叠”的邻居，同轨但无关的 clip 不会被波及。
        const crossfadeClipIds = Array.from(
            new Set<string>([...selectedClipIds, ...Object.keys(rippleFollowers)]),
        );
        const initialCrossfadeSides = computeInitialCrossfadeSides(
            sessionRef.current.clips,
            crossfadeClipIds,
        );

        // 本次编辑实际触碰的侧：trim_left/stretch_left/fade_in 只影响左缘（fadeIn），
        // trim_right/stretch_right/fade_out 只影响右缘（fadeOut）。
        // 自动交叉淡化只能碰这些侧；其它侧（如 trim_left 时右缘与邻居的淡出重叠）
        // 绝不能自动调整。
        const editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }> = {};
        for (const id of selectedClipIds) {
            if (type === "trim_left" || type === "stretch_left" || type === "fade_in") {
                editSides[id] = { fadeIn: true, fadeOut: false };
            } else if (
                type === "trim_right" ||
                type === "stretch_right" ||
                type === "fade_out"
            ) {
                editSides[id] = { fadeIn: false, fadeOut: true };
            } else {
                editSides[id] = { fadeIn: false, fadeOut: false };
            }
        }

        // 淡入淡出相对拖拽的“视觉/有效”起点：自动交叉淡化生效时用自动长度，
        // 否则用手动长度。这样从“自动交叉淡化”直接拖成“手动淡入淡出”时，
        // 以用户当前看到的长度作为起点，拖拽过程不会从自动值跳变到隐藏的手动值。
        const effectiveFadeInSec =
            Number(clip.autoFadeInSec ?? 0) > 0 ? Number(clip.autoFadeInSec) : Number(clip.fadeInSec);
        const effectiveFadeOutSec =
            Number(clip.autoFadeOutSec ?? 0) > 0
                ? Number(clip.autoFadeOutSec)
                : Number(clip.fadeOutSec);
        // 交叉点手柄基础数据：重叠长度、两侧是否是自动淡化、以及右侧 clip 的淡入有效长度。
        const crossfadePartnerClip =
            crossfadePartnerClipId != null
                ? sessionRef.current.clips.find((c) => c.id === crossfadePartnerClipId)
                : undefined;
        const crossfadeBaseOverlapSec = crossfadePartnerClip
            ? Math.max(0, clip.startSec + clip.lengthSec - crossfadePartnerClip.startSec)
            : 0;
        const crossfadeBaseFadeOutAuto = Number(clip.autoFadeOutSec ?? 0) > 0;
        const crossfadePartnerFadeInSec = crossfadePartnerClip
            ? Number(crossfadePartnerClip.autoFadeInSec ?? 0) > 0
                ? Number(crossfadePartnerClip.autoFadeInSec)
                : Number(crossfadePartnerClip.fadeInSec)
            : 0;
        const crossfadePartnerFadeInAuto = Boolean(
            crossfadePartnerClip && Number(crossfadePartnerClip.autoFadeInSec ?? 0) > 0,
        );
        editDragRef.current = {
            type,
            pointerId: e.pointerId,
            clipId,
            basestartSec: clip.startSec,
            baselengthSec: clip.lengthSec,
            basePlaybackRate: Number(clip.playbackRate ?? 1) || 1,
            baseSourceStartSec: clip.sourceStartSec,
            baseSourceEndSec: clip.sourceEndSec,
            basefadeInSec: effectiveFadeInSec,
            basefadeOutSec: effectiveFadeOutSec,
            basePointerSec,
            baseGain: clip.gain,
            sourceBeats: null,
            rightEdgeBeat,
            baseReversed: !!clip.reversed,
            baseDurationFrames: clip.durationFrames ?? null,
            baseSourceSampleRate: clip.sourceSampleRate ?? null,
            baseDurationSec: clip.durationSec ?? null,
            stretchGroup,
            selectedClipIds,
            baseGainById,
            rippleMode,
            rippleFollowers,
            initialCrossfadeSides,
            crossfadeClipIds,
            editSides,
            crossfadePartnerClipId,
            crossfadeBaseOverlapSec,
            crossfadeBaseFadeOutAuto,
            crossfadePartnerFadeInSec,
            crossfadePartnerFadeInAuto,
            baseByClipId: Object.fromEntries(
                selectedClipIds.map((id) => {
                    const c =
                        id === clipId ? clip : sessionRef.current.clips.find((x) => x.id === id);
                    return [
                        id,
                        {
                            startSec: c?.startSec ?? 0,
                            lengthSec: c?.lengthSec ?? 0,
                            playbackRate: Number(c?.playbackRate ?? 1) || 1,
                            sourceStartSec: c?.sourceStartSec ?? 0,
                            sourceEndSec: c?.sourceEndSec ?? 0,
                            reversed: !!c?.reversed,
                            loopEnabled: !!c?.loopEnabled,
                            /** 拖拽起始基准 SnapOffset：拉伸按 总比例×基准 缩放，
                             *  禁止逐帧读实时值复合放大。 */
                            snapOffsetSec: Math.max(0, Number(c?.snapOffsetSec) || 0),
                            durationFrames: c?.durationFrames ?? null,
                            sourceSampleRate: c?.sourceSampleRate ?? null,
                            durationSec: c?.durationSec ?? null,
                            contentDurationSec: c
                                ? resolveClipContentDurationSec({
                                      sourcePath: c.sourcePath,
                                      midiNoteData: c.midiNoteData ?? null,
                                      durationFrames: c.durationFrames,
                                      sourceSampleRate: c.sourceSampleRate,
                                      durationSec: c.durationSec,
                                  })
                                : null,
                        },
                    ];
                }),
            ),
        };

        (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);

        let ticking = false;
        let latestEvent: PointerEvent | null = null;
        let accumulatedGainDeltaDb = 0;
        let gainFineAxisState: FineAxisDragState | null = null;
        let gainDragStartClientY: number | null = null;
        let remotePreviewChain: Promise<unknown> = Promise.resolve();

        const finishGainUndoGroup = async () => {
            if (!gainUndoGroupPromise) return;
            try {
                await remotePreviewChain;
                await gainUndoGroupPromise;
            } finally {
                await webApi.endUndoGroup();
            }
        };

        const finishFadeUndoGroup = async () => {
            if (!fadeUndoGroupPromise) return;
            // 由调用方在 persistPromise 完成后再调用（见 end() 的 finally）。
            // 这里只释放 undo group，使整个 fade 拖拽成为单个撤销步。
            await webApi.endUndoGroup();
        };

        const finishCrossfadeUndoGroup = async () => {
            if (!crossfadeUndoGroupPromise) return;
            await webApi.endUndoGroup();
        };

        function onMove(ev: PointerEvent) {
            latestEvent = ev;
            if (ticking) return;
            ticking = true;

            requestAnimationFrame(() => {
                ticking = false;
                if (!latestEvent) return;
                const currentEv = latestEvent;

                const drag = editDragRef.current;
                const el = scrollRef.current;
                if (!drag || drag.pointerId !== e.pointerId || !el) return;
                const b = el.getBoundingClientRect();
                let beat = beatFromClientX(currentEv.clientX, b, el.scrollLeft);
                const shouldSnap =
                    drag.type === "trim_left" ||
                    drag.type === "trim_right" ||
                    drag.type === "stretch_left" ||
                    drag.type === "stretch_right";
                const noSnapActive = isModifierActive(noSnapKb, currentEv);
                // "拖动时切换吸附"：修饰键把吸附总开关临时取反（开→关 / 关→开）。
                const effectiveSnap = computeEffectiveSnap(snapEnabled, noSnapActive);
                /** 被编辑 clip 的轨道（编辑类拖拽不跨轨，行固定）。 */
                const anchorTrackId =
                    sessionRef.current.clips.find((c) => c.id === drag.clipId)?.trackId ?? null;
                if (shouldSnap && effectiveSnap) {
                    // 吸附 + 竖线高亮发布：
                    // - trim_left / stretch_left → 被吸附对象是 Clip 的前缘；
                    // - trim_right / stretch_right → 后缘。
                    // 目标侧（网格线/对方 Clip 边缘等）与被吸附边标记都由
                    // snapTimelineDetailed 的 highlight 通道统一发布。
                    const leftEdge =
                        drag.type === "trim_right" || drag.type === "stretch_right";
                    beat = snapTimelineDetailed(beat, "clip", {
                        originSec: leftEdge ? drag.rightEdgeBeat : drag.basestartSec,
                        anchorTrackId,
                        excludeClipIds: new Set(drag.selectedClipIds),
                        highlight: {
                            sources: [
                                {
                                    trackId: anchorTrackId,
                                    clipId: drag.clipId,
                                },
                            ],
                        },
                    }).sec;
                } else if (shouldSnap) {
                    clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                }
                // ── 循环节 / 内容边界吸附 ───────────────────────────────
                // 属于常规吸附体系：受"吸附"总开关与"拖动时切换吸附"修饰键
                // （XOR）控制，且需在吸附设置中启用"Clip 边缘吸附到源素材
                // 首尾"。目标：媒体边界（s=0 与 s=D）在时间线上的投影位置
                // —— Loop Clip 呈 mod-D 等差族；未循环 Clip 的"循环节"即
                // 原始媒体内容在 Clip 内的终止/起始位置。命中时优先于普通
                // 网格吸附（语义更具体）。trim_left / trim_right 的边界
                // 同余式都只依赖"移动边缘相对 Clip 基准起点的时间线偏移"，
                // 故统一处理。
                if (
                    shouldSnap &&
                    effectiveSnap &&
                    (drag.type === "trim_left" || drag.type === "trim_right") &&
                    timelineSnap.snapDistancePx > 0 &&
                    timelineSnap.snapClipsToSourceMedia
                ) {
                    const anchorBase = drag.baseByClipId[drag.clipId];
                    if (anchorBase) {
                        const rawOffset = beat - anchorBase.startSec;
                        const snappedOffset = nearestBoundarySnapOffsetSec(
                            {
                                loopEnabled: anchorBase.loopEnabled,
                                reversed: anchorBase.reversed,
                                sourceStartSec: anchorBase.sourceStartSec,
                                sourceEndSec: anchorBase.sourceEndSec,
                                playbackRate: anchorBase.playbackRate,
                                lengthSec: anchorBase.lengthSec,
                                durationFrames: anchorBase.durationFrames,
                                sourceSampleRate: anchorBase.sourceSampleRate,
                                durationSec: anchorBase.durationSec,
                                contentDurationSec: anchorBase.contentDurationSec,
                            },
                            "edge",
                            rawOffset,
                        );
                        if (
                            snappedOffset != null &&
                            Math.abs(snappedOffset - rawOffset) <=
                                loopSnapThresholdSec(timelineSnap.snapDistancePx, pxPerSec) + 1e-12
                        ) {
                            beat = anchorBase.startSec + snappedOffset;
                            // 循环节命中：以“循环节”专用高亮覆盖常规吸附高亮
                            //（同组发布即整组替换）。目标（源媒体边界投影）与
                            // 被吸附边重合于同一 x，行内双亮条强调。
                            publishSnapHighlights(SNAP_HIGHLIGHT_GROUP, [
                                buildLoopBoundaryHighlightEntry({
                                    secs: [beat],
                                    trackId: anchorTrackId,
                                    clipId: drag.clipId,
                                }),
                            ]);
                        }
                    }
                }

                const minLen = 0.0;
                if (drag.type === "crossfade_edges") {
                    const partner = drag.crossfadePartnerClipId;
                    if (!partner) return;
                    // 修饰键（默认 Ctrl/Cmd）→ 反向模式：A 右缘与 B 左缘向相反方向移动，
                    // 重叠长度改变，淡化长度按比例缩放（auto 保持 == 重叠长度）。
                    const opposite = isModifierActive(crossfadeGripKb, currentEv);
                    const rawDelta = beat - drag.basePointerSec;
                    const aBase = drag.baseByClipId[drag.clipId];
                    const bBase = drag.baseByClipId[partner];
                    if (!aBase || !bBase) return;

                    // 计算两个 clip 允许的 delta 范围（A 右缘 / B 左缘），取交集 → 任一达到限制，
                    // 两者同时停止（共享同一 delta）。
                    // A 右缘延伸不受源媒体长度限制：Loop 由回绕内容填充，
                    // 非 Loop 超出源窗口/媒体末尾的部分按静音渲染（派生窗口）。
                    const aRate = aBase.playbackRate > 0 ? aBase.playbackRate : 1;
                    let minDelta = -aBase.lengthSec;
                    let maxDelta = Number.POSITIVE_INFINITY;

                    const bRate = bBase.playbackRate > 0 ? bBase.playbackRate : 1;
                    const bStartSign = opposite ? -1 : 1; // B.start = baseStart + bStartSign * delta
                    if (opposite) {
                        // B.start = baseStart - delta：向左拖（delta<0）裁短 B 起始向右，
                        // 向右拖（delta>0）延长 B 起始向左。
                        minDelta = Math.max(minDelta, -bBase.lengthSec);
                        maxDelta = Math.min(maxDelta, bBase.startSec);
                        // 反向/缩放模式：向左裁短时必须保持两个 clip 仍然重叠，
                        // 重叠长度最小为 MIN_CROSSFADE_OVERLAP_SEC。
                        // newOverlap = baseOverlap + 2*delta >= MIN → delta >= (MIN - baseOverlap)/2。
                        const minOverlap = 0.0002;
                        minDelta = Math.max(
                            minDelta,
                            (minOverlap - drag.crossfadeBaseOverlapSec) / 2,
                        );
                        // B 左缘延伸已放开：非 Loop 的锚点可越过媒体边界
                        //（越出部分为前导/尾部静音），不再按源媒体钳制。
                    } else {
                        // B.start = baseStart + delta（同方向，保持重叠长度不变）。
                        minDelta = Math.max(minDelta, -bBase.startSec);
                        maxDelta = Math.min(maxDelta, bBase.lengthSec);
                    }

                    if (minDelta > maxDelta) return;
                    const delta = clamp(rawDelta, minDelta, maxDelta);
                    const updates: Array<{
                        clipId: string;
                        startSec: number;
                        lengthSec: number;
                        sourceStartSec?: number;
                        sourceEndSec?: number;
                    }> = [];

                    // A：右缘（结束位置）移动 delta（同 trim_right）。
                    {
                        const base = aBase;
                        const rate = aRate;
                        if (base.loopEnabled) {
                            // Loop：只改长度，不修改源窗口（内容按周期回绕）。
                            updates.push({
                                clipId: drag.clipId,
                                startSec: base.startSec,
                                lengthSec: Math.max(0, base.lengthSec + delta),
                            });
                        } else if (base.reversed) {
                            // 倒放非 Loop：右缘延伸向下消费窗口起点（→0），
                            // 耗尽后继续增长的部分为静音尾巴，窗口保持不变。
                            let nextTrimStart = Math.max(
                                0,
                                base.sourceStartSec - delta * rate,
                            );
                            nextTrimStart = Math.min(nextTrimStart, base.sourceEndSec);
                            updates.push({
                                clipId: drag.clipId,
                                startSec: base.startSec,
                                lengthSec: Math.max(0, base.lengthSec + delta),
                                sourceStartSec: nextTrimStart,
                            });
                        } else {
                            // 正放非 Loop：派生窗口 —— source_end = 起点 + 长度×速率。
                            // 超出媒体末尾的部分由渲染管线填充静音。
                            const nextLen = Math.max(0, base.lengthSec + delta);
                            updates.push({
                                clipId: drag.clipId,
                                startSec: base.startSec,
                                lengthSec: nextLen,
                                sourceEndSec: base.sourceStartSec + nextLen * rate,
                            });
                        }
                    }

                    // B：左缘（起始位置）移动 bStartSign*delta（同 trim_left）。
                    {
                        const base = bBase;
                        const rate = bRate;
                        const startDelta = bStartSign * delta;
                        // Loop（循环原始文件）：锚点语义 —— 左缘右移（裁短）
                        // 推进锚点并环绕；左缘左移（延伸）回退锚点并环绕，
                        // 内容保持锚定。非 Loop 维持既有窗口推进逻辑。
                        const shortensFromLeft = startDelta > 0;
                        if (base.loopEnabled && !shortensFromLeft) {
                            const nextStart = Math.max(0, base.startSec + startDelta);
                            const patch: {
                                clipId: string;
                                startSec: number;
                                lengthSec: number;
                                sourceStartSec?: number;
                                sourceEndSec?: number;
                            } = {
                                clipId: partner,
                                startSec: nextStart,
                                lengthSec: Math.max(0, base.lengthSec - startDelta),
                            };
                            if (base.reversed) {
                                // 倒放锚点(source_end)随左缘延伸向上回退并环绕。
                                patch.sourceEndSec = wrapIntoMediaDomain(
                                    base.sourceEndSec - startDelta * rate,
                                    base,
                                );
                            } else {
                                patch.sourceStartSec = wrapIntoMediaDomain(
                                    base.sourceStartSec + startDelta * rate,
                                    base,
                                );
                            }
                            updates.push(patch);
                        } else if (base.loopEnabled && base.reversed) {
                            // Loop + 倒放 + 左缘右移（裁短）：锚点(source_end)向下环绕推进。
                            const nextTrimEnd = wrapIntoMediaDomain(
                                base.sourceEndSec - startDelta * rate,
                                base,
                            );
                            updates.push({
                                clipId: partner,
                                startSec: base.startSec + startDelta,
                                lengthSec: Math.max(
                                    0,
                                    base.lengthSec - startDelta,
                                ),
                                sourceEndSec: nextTrimEnd,
                            });
                        } else if (base.loopEnabled) {
                            // Loop + 正放 + 左缘右移（裁短）：锚点(source_start)向上环绕推进。
                            const nextTrimStart = wrapIntoMediaDomain(
                                base.sourceStartSec + startDelta * rate,
                                base,
                            );
                            updates.push({
                                clipId: partner,
                                startSec: base.startSec + startDelta,
                                lengthSec: Math.max(0, base.lengthSec - startDelta),
                                sourceStartSec: nextTrimStart,
                            });
                        } else if (base.reversed) {
                            // 倒放非 Loop：左缘对应窗口终点(source_end)。
                            // 向左延伸使其越过媒体时长 → 前导静音，不再钳制。
                            const nextTrimEnd = base.sourceEndSec - startDelta * rate;
                            const actualDeltaTrim = base.sourceEndSec - nextTrimEnd;
                            const actualDeltaTimeline = actualDeltaTrim / rate;
                            updates.push({
                                clipId: partner,
                                startSec: base.startSec + actualDeltaTimeline,
                                lengthSec: clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                ),
                                sourceEndSec: nextTrimEnd,
                            });
                        } else {
                            // 正放非 Loop：左缘对应窗口起点(source_start)。
                            // 向左延伸使其越过媒体起点 → 前导静音，不再钳制到 0。
                            const nextTrimStart = base.sourceStartSec + startDelta * rate;
                            const actualDeltaTrim = nextTrimStart - base.sourceStartSec;
                            const actualDeltaTimeline = actualDeltaTrim / rate;
                            updates.push({
                                clipId: partner,
                                startSec: base.startSec + actualDeltaTimeline,
                                lengthSec: clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                ),
                                sourceStartSec: nextTrimStart,
                            });
                        }
                    }

                    // 反向模式：按“新重叠 / 原重叠”比例缩放淡化长度。
                    // 自动淡化 → 写入 auto 字段（因此 auto 始终 == 重叠长度）；
                    // 手动淡化 → 写入手动字段（保持原有比例）。
                    const fadeUpdates: Array<{
                        clipId: string;
                        fadeInSec?: number;
                        fadeOutSec?: number;
                        autoFadeInSec?: number;
                        autoFadeOutSec?: number;
                    }> = [];
                    if (opposite && drag.crossfadeBaseOverlapSec > 0.001) {
                        const newOverlap = Math.max(0.0002, drag.crossfadeBaseOverlapSec + 2 * delta);
                        const ratio = newOverlap / drag.crossfadeBaseOverlapSec;
                        const aFade = drag.basefadeOutSec * ratio;
                        const bFade = drag.crossfadePartnerFadeInSec * ratio;
                        if (drag.crossfadeBaseFadeOutAuto) {
                            fadeUpdates.push({ clipId: drag.clipId, autoFadeOutSec: aFade });
                        } else {
                            fadeUpdates.push({ clipId: drag.clipId, fadeOutSec: aFade });
                        }
                        if (drag.crossfadePartnerFadeInAuto) {
                            fadeUpdates.push({ clipId: partner, autoFadeInSec: bFade });
                        } else {
                            fadeUpdates.push({ clipId: partner, fadeInSec: bFade });
                        }
                    }

                    batch(() => {
                        for (const u of updates) {
                            dispatch(moveClipStart({ clipId: u.clipId, startSec: u.startSec }));
                            dispatch(setClipLength({ clipId: u.clipId, lengthSec: u.lengthSec }));
                            if (u.sourceStartSec !== undefined) {
                                dispatch(
                                    setClipSourceRange({
                                        clipId: u.clipId,
                                        sourceStartSec: u.sourceStartSec,
                                    }),
                                );
                            }
                            if (u.sourceEndSec !== undefined) {
                                dispatch(
                                    setClipSourceRange({
                                        clipId: u.clipId,
                                        sourceEndSec: u.sourceEndSec,
                                    }),
                                );
                            }
                        }
                        for (const f of fadeUpdates) {
                            if (f.fadeInSec !== undefined || f.fadeOutSec !== undefined) {
                                dispatch(
                                    setClipFades({
                                        clipId: f.clipId,
                                        fadeInSec: f.fadeInSec,
                                        fadeOutSec: f.fadeOutSec,
                                    }),
                                );
                            }
                            if (f.autoFadeInSec !== undefined || f.autoFadeOutSec !== undefined) {
                                dispatch(
                                    setClipAutoFades({
                                        clipId: f.clipId,
                                        autoFadeInSec: f.autoFadeInSec,
                                        autoFadeOutSec: f.autoFadeOutSec,
                                    }),
                                );
                            }
                        }
                    });
                    return;
                }

                if (drag.type === "fade_in") {
                    // 相对拖拽：记录起点偏移，新长度 = 基础长度 + 指针位移。
                    // 不再“让边缘线实时对齐指针”，从包络线任意位置拖动都不会跳变。
                    const delta = beat - drag.basePointerSec;
                    const next = clamp(
                        drag.basefadeInSec + delta,
                        0,
                        Math.max(0, drag.baselengthSec),
                    );
                    const fadeUpdates = applyBulkFadeValue({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            sessionRef.current.clips.map((clip) => [clip.id, clip] as const),
                        ),
                        target: "fadeInSec",
                        nextValue: next,
                    });
                    batch(() => {
                        for (const update of fadeUpdates) {
                            dispatch(setClipFades(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                // 手动拖拽淡入 = 用户手动 fade，且清除该侧自动交叉淡化。
                                dispatch(setClipAutoFades({ clipId: drag.clipId, autoFadeInSec: 0 }));
                                // 直接 webApi 持久化（不走 thunk）：其 fulfilled 不会 force-apply
                                // 整份时间线覆盖本地乐观值，避免拖拽中淡入淡出包络闪烁。
                                void webApi.setClipState({
                                    clipId: drag.clipId,
                                    fadeInSec: next,
                                    autoFadeInSec: 0,
                                    checkpoint: false,
                                });
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }
                if (drag.type === "fade_out") {
                    // 相对拖拽：新长度 = 基础长度 − 指针位移（向右缩短、向左增长），
                    // 从包络线中间开始拖也不再跳变。
                    const delta = beat - drag.basePointerSec;
                    const next = clamp(
                        drag.basefadeOutSec - delta,
                        0,
                        Math.max(0, drag.baselengthSec),
                    );
                    const fadeUpdates = applyBulkFadeValue({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            sessionRef.current.clips.map((clip) => [clip.id, clip] as const),
                        ),
                        target: "fadeOutSec",
                        nextValue: next,
                    });
                    batch(() => {
                        for (const update of fadeUpdates) {
                            dispatch(setClipFades(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                // 手动拖拽淡出 = 手动 fade，且清除该侧自动交叉淡化。
                                dispatch(setClipAutoFades({ clipId: drag.clipId, autoFadeOutSec: 0 }));
                                // 直接 webApi 持久化（不走 thunk）：避免 force-apply 覆盖本地
                                // 乐观 fade 导致拖拽中淡入淡出包络闪烁。
                                void webApi.setClipState({
                                    clipId: drag.clipId,
                                    fadeOutSec: next,
                                    autoFadeOutSec: 0,
                                    checkpoint: false,
                                });
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }
                if (drag.type === "gain") {
                    if (gainDragStartClientY == null || !gainFineAxisState) {
                        gainDragStartClientY = currentEv.clientY;
                        gainFineAxisState = {
                            raw: currentEv.clientY,
                            adjusted: currentEv.clientY,
                            fineActive: isModifierActive(paramFineAdjustKb, currentEv),
                        };
                    }
                    const adjustedY = advanceFineAxisDrag(
                        gainFineAxisState,
                        currentEv.clientY,
                        isModifierActive(paramFineAdjustKb, currentEv),
                    );
                    const deltaY = gainDragStartClientY - adjustedY;
                    accumulatedGainDeltaDb = deltaY * CLIP_GAIN_DRAG_DB_PER_PX;
                    const gainUpdates = applyBulkGainDeltaDb({
                        clipIds: drag.selectedClipIds,
                        clipsById: new Map(
                            drag.selectedClipIds.map((id) => [
                                id,
                                { gain: drag.baseGainById[id] ?? 1 },
                            ]),
                        ),
                        deltaDb: accumulatedGainDeltaDb,
                        minDb: -12,
                        maxDb: 12,
                    });
                    batch(() => {
                        for (const update of gainUpdates) {
                            dispatch(setClipGain(update));
                        }
                    });
                    try {
                        if (drag.selectedClipIds.length === 1) {
                            const now = Date.now();
                            const last = lastRemoteSentRef.current[drag.clipId] || 0;
                            const nextGain = gainUpdates[0]?.gain;
                            if (nextGain != null && now - last > 200) {
                                lastRemoteSentRef.current[drag.clipId] = now;
                                const remoteUpdate = () =>
                                    webApi.setClipState({
                                        clipId: drag.clipId,
                                        gain: nextGain,
                                        checkpoint: false,
                                    });
                                const nextRemotePreview = gainUndoGroupPromise
                                    ? gainUndoGroupPromise.then(remoteUpdate)
                                    : Promise.resolve().then(remoteUpdate);
                                remotePreviewChain = remotePreviewChain
                                    .then(() => nextRemotePreview)
                                    .catch(() => undefined);
                            }
                        }
                    } catch {
                        // Best-effort remote preview update; ignore transient failures.
                    }
                    return;
                }

                // 自动交叉淡化实时预览：在位置/尺寸变化的每个分支，按当前乐观状态
                // 计算重叠并实时更新自动 fade 包络；松手时由 applyAutoCrossfade 持久化权威结果。
                // affectedSides = 编辑前的每侧重叠关系（分开时仅清自动交叉淡化、保留手动 fade）。
                const previewAutoCrossfadeNow = () => {
                    if (!sessionRef.current.autoCrossfadeEnabled) return;
                    // 用同步新鲜的 store.getState().session，避免 batch 内 sessionRef 滞后一帧，
                    // 导致“拖开瞬间”预览滞留最后一帧自动交叉淡化长度（松手后才跳变）。
                    previewAutoCrossfade(
                        store.getState().session,
                        drag.crossfadeClipIds,
                        dispatch,
                        drag.initialCrossfadeSides,
                        drag.editSides,
                    );
                };

                if (
                    drag.stretchGroup &&
                    (drag.type === "stretch_left" || drag.type === "stretch_right")
                ) {
                    const stretchGroup = drag.stretchGroup;
                    const update = computeStretchGroupUpdate({
                        group: stretchGroup,
                        edge: drag.type,
                        pointerSec: beat,
                    });
                    batch(() => {
                        for (const clipId of stretchGroup.clipIds) {
                            const next = update.byId[clipId];
                            if (!next) continue;
                            dispatch(
                                moveClipStart({
                                    clipId,
                                    startSec: next.startSec,
                                }),
                            );
                            dispatch(
                                setClipLength({
                                    clipId,
                                    lengthSec: next.lengthSec,
                                }),
                            );
                            // SnapOffset 随编组拉伸按"总比例×基准偏移"线性缩放
                            //（基准 = 拖拽起始快照，禁止逐帧复合）。
                            {
                                const base = drag.baseByClipId[clipId];
                                if (base) {
                                    const ratio =
                                        next.lengthSec / Math.max(1e-6, base.lengthSec);
                                    dispatch(
                                        setClipSnapOffset({
                                            clipId,
                                            snapOffsetSec: scaleSnapOffsetForStretch(
                                                base.snapOffsetSec,
                                                ratio,
                                                next.lengthSec,
                                            ),
                                        }),
                                    );
                                }
                            }
                            dispatch(
                                setClipPlaybackRate({
                                    clipId,
                                    playbackRate: next.playbackRate,
                                }),
                            );
                            dispatch(
                                setClipFades({
                                    clipId,
                                    fadeInSec: next.fadeInSec,
                                    fadeOutSec: next.fadeOutSec,
                                }),
                            );
                        }
                    });
                    // 波纹（自动跟进）实时预览：编组拉伸同样按“区域右缘净位移”实时波纹。
                    if (drag.rippleMode !== "off") {
                        const rippleRightDelta = computeRegionRightEdgeDelta(
                            drag,
                            sessionRef.current.clips,
                        );
                        if (Math.abs(rippleRightDelta) > 1e-9) {
                            applyRippleFollowerShift(
                                dispatch,
                                drag.rippleFollowers,
                                rippleRightDelta,
                            );
                        }
                    }
                    // 自动交叉淡化实时预览（编组拉伸）。
                    previewAutoCrossfadeNow();
                    return;
                }

                if (drag.type === "trim_left") {
                    const minLen = 0.0;
                    const anchorBase = drag.baseByClipId[drag.clipId];
                    if (!anchorBase) return;
                    const anchorRight = anchorBase.startSec + anchorBase.lengthSec;
                    const desiredStart = clamp(beat, 0, anchorRight - minLen);
                    const desiredDelta = desiredStart - anchorBase.startSec;

                    // Find the most constrained delta across all group members
                    // 向左延伸已放开（对称无界）：非 Loop 的锚点可越过媒体边界，
                    // 越出部分为前导静音（正放 source_start < 0 / 倒放
                    // source_end > D）。仅保留"裁短不得超过整窗/clip 长度"的上限。
                    let limitedDelta = desiredDelta;
                    for (const id of drag.selectedClipIds) {
                        const base = drag.baseByClipId[id];
                        if (!base) continue;
                        const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                        if (base.loopEnabled) {
                            // Loop（循环原始文件）：锚点可环绕，向左延伸不受
                            // 源长度限制；向右裁短仅受 clip 长度约束
                            //（锚点经 floor_mod 归一化，无需窗口跨度上限）。
                            limitedDelta = Math.min(limitedDelta, base.lengthSec - minLen);
                        } else if (base.reversed) {
                            // 倒放：裁短上限 = 整个窗口跨度。
                            const maxDelta = (base.sourceEndSec - base.sourceStartSec) / rate;
                            limitedDelta = Math.min(limitedDelta, maxDelta);
                        } else {
                            // 正放：裁短上限 = clip 长度。
                            limitedDelta = Math.min(limitedDelta, base.lengthSec - minLen);
                        }
                    }

                    batch(() => {
                        for (const id of drag.selectedClipIds) {
                            const base = drag.baseByClipId[id];
                            if (!base) continue;
                            const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                            if (base.loopEnabled && limitedDelta < 0) {
                                // Loop + 向左延伸：内容保持锚定 —— 锚点沿遍历方向
                                // 回退 |δ|·rate 并对整个媒体时长取模环绕
                                //（正放减 source_start，倒放加 source_end）。
                                dispatch(moveClipStart({ clipId: id, startSec: base.startSec + limitedDelta }));
                                dispatch(
                                    setClipLength({
                                        clipId: id,
                                        lengthSec: clamp(base.lengthSec - limitedDelta, minLen, 10_000),
                                    }),
                                );
                                if (base.reversed) {
                                    // 倒放锚点(source_end)向上回退并环绕。
                                    dispatch(
                                        setClipSourceRange({
                                            clipId: id,
                                            sourceEndSec: wrapIntoMediaDomain(
                                                base.sourceEndSec - limitedDelta * rate,
                                                base,
                                            ),
                                        }),
                                    );
                                } else {
                                    // 正放锚点(source_start)向下回退并环绕。
                                    dispatch(
                                        setClipSourceRange({
                                            clipId: id,
                                            sourceStartSec: wrapIntoMediaDomain(
                                                base.sourceStartSec + limitedDelta * rate,
                                                base,
                                            ),
                                        }),
                                    );
                                }
                            } else if (base.loopEnabled && base.reversed) {
                                // Loop + 倒放 + 裁短：锚点(source_end)向下推进并环绕。
                                let nextTrimEnd =
                                    base.sourceEndSec - limitedDelta * rate;
                                nextTrimEnd = wrapIntoMediaDomain(
                                    nextTrimEnd,
                                    base,
                                );
                                const actualDeltaTimeline = limitedDelta;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({ clipId: id, sourceEndSec: nextTrimEnd }),
                                );
                            } else if (base.loopEnabled) {
                                // Loop + 正放 + 裁短：锚点(source_start)向上推进并环绕。
                                let nextTrimStart =
                                    base.sourceStartSec + limitedDelta * rate;
                                nextTrimStart = wrapIntoMediaDomain(
                                    nextTrimStart,
                                    base,
                                );
                                const actualDeltaTimeline = limitedDelta;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({
                                        clipId: id,
                                        sourceStartSec: nextTrimStart,
                                    }),
                                );
                            } else if (base.reversed) {
                                // 倒放非 Loop：左缘对应窗口终点(source_end)。
                                // 向左延伸（delta<0）使其越过媒体时长 → 前导
                                // 静音；裁短受约束循环上限保护，无需在此钳制。
                                const nextTrimEnd = base.sourceEndSec - limitedDelta * rate;
                                const actualDeltaTrim = base.sourceEndSec - nextTrimEnd;
                                const actualDeltaTimeline = actualDeltaTrim / rate;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({ clipId: id, sourceEndSec: nextTrimEnd }),
                                );
                            } else {
                                // 正放非 Loop：左缘对应窗口起点(source_start)。
                                // 向左延伸（delta<0）使其越过媒体起点 → 前导
                                // 静音；不再钳制到 0。
                                const nextTrimStart = base.sourceStartSec + limitedDelta * rate;
                                const actualDeltaTrim = nextTrimStart - base.sourceStartSec;
                                const actualDeltaTimeline = actualDeltaTrim / rate;
                                const nextStart = base.startSec + actualDeltaTimeline;
                                const nextLen = clamp(
                                    base.lengthSec - actualDeltaTimeline,
                                    minLen,
                                    10_000,
                                );
                                dispatch(moveClipStart({ clipId: id, startSec: nextStart }));
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                dispatch(
                                    setClipSourceRange({
                                        clipId: id,
                                        sourceStartSec: nextTrimStart,
                                    }),
                                );
                            }
                        }
                    });
                    // 自动交叉淡化实时预览（左缘截短：重叠变化也会改变 fade）。
                    previewAutoCrossfadeNow();
                    return;
                }

                if (drag.type === "stretch_left") {
                    const desiredStart = clamp(beat, 0, drag.rightEdgeBeat - minLen);
                    const rawLen = clamp(drag.rightEdgeBeat - desiredStart, minLen, 10_000);
                    const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                    const baseRate =
                        drag.basePlaybackRate > 0 && Number.isFinite(drag.basePlaybackRate)
                            ? drag.basePlaybackRate
                            : 1;
                    const nextRate = clamp((baseRate * baseLen) / Math.max(1e-6, rawLen), 0.1, 10);
                    const correctedLen = (baseRate * baseLen) / nextRate;
                    const nextStart = drag.rightEdgeBeat - correctedLen;
                    const scaledFades = scaleClipFadesForStretch({
                        baseFadeInSec: drag.basefadeInSec,
                        baseFadeOutSec: drag.basefadeOutSec,
                        baseLengthSec: drag.baselengthSec,
                        nextLengthSec: correctedLen,
                    });
                    dispatch(moveClipStart({ clipId: drag.clipId, startSec: nextStart }));
                    dispatch(setClipLength({ clipId: drag.clipId, lengthSec: correctedLen }));
                    dispatch(setClipPlaybackRate({ clipId: drag.clipId, playbackRate: nextRate }));
                    // SnapOffset 随拉伸同步缩放：总比例 × 拖拽起始基准偏移。
                    dispatch(
                        setClipSnapOffset({
                            clipId: drag.clipId,
                            snapOffsetSec: scaleSnapOffsetForStretch(
                                drag.baseByClipId[drag.clipId]?.snapOffsetSec,
                                correctedLen / baseLen,
                                correctedLen,
                            ),
                        }),
                    );
                    dispatch(
                        setClipFades({
                            clipId: drag.clipId,
                            fadeInSec: scaledFades.fadeInSec,
                            fadeOutSec: scaledFades.fadeOutSec,
                        }),
                    );
                    // 自动交叉淡化实时预览（左缘拉伸）。
                    previewAutoCrossfadeNow();
                    return;
                }

                if (drag.type === "trim_right") {
                    const minLen = 0.0;
                    const anchorBase = drag.baseByClipId[drag.clipId];
                    if (!anchorBase) return;

                    const desiredRight = clamp(beat, anchorBase.startSec + minLen, 10_000);
                    const desiredLen = desiredRight - anchorBase.startSec;
                    const nextLen = clamp(desiredLen, minLen, 10_000);
                    const desiredDeltaTimeline = nextLen - anchorBase.lengthSec;

                    // Find the most constrained delta across all group members
                    // 右缘延伸（delta>0）不再受源媒体长度限制：非 Loop 的 Clip
                    // 超出源窗口/媒体末尾的部分按静音渲染（REAPER 同语义）。
                    // Loop 则由回绕内容填充。两者都只受全局长度上限约束。
                    let limitedDelta = desiredDeltaTimeline;
                    for (const id of drag.selectedClipIds) {
                        const base = drag.baseByClipId[id];
                        if (!base) continue;
                        limitedDelta = Math.min(limitedDelta, 10_000 - base.lengthSec);
                        limitedDelta = Math.max(limitedDelta, -base.lengthSec + minLen);
                    }

                    batch(() => {
                        for (const id of drag.selectedClipIds) {
                            const base = drag.baseByClipId[id];
                            if (!base) continue;
                            if (base.loopEnabled) {
                                // Loop：只改长度，源窗口保持不变 ——
                                // 延伸部分由循环回绕内容填充，裁短仅隐藏尾部
                                //（再次延伸时先恢复被隐藏的窗口内容）。
                                dispatch(
                                    setClipLength({
                                        clipId: id,
                                        lengthSec: clamp(base.lengthSec + limitedDelta, 0, 10_000),
                                    }),
                                );
                                continue;
                            }
                            const rate = base.playbackRate > 0 ? base.playbackRate : 1;
                            const nextLen = clamp(base.lengthSec + limitedDelta, 0, 10_000);
                            // 源窗口当前覆盖的可听时间线跨度（倒放分支使用）。
                            const spanTimeline =
                                Math.max(0, base.sourceEndSec - base.sourceStartSec) / rate;
                            if (base.reversed) {
                                // 倒放：右缘对应窗口起点(source_start)。
                                let nextTrimStart = base.sourceStartSec;
                                if (nextLen <= spanTimeline + 1e-9) {
                                    // 窗口内截短：起点随右缘上移（不越过窗口终点）。
                                    nextTrimStart = Math.min(
                                        base.sourceEndSec,
                                        base.sourceEndSec - nextLen * rate,
                                    );
                                } else if (base.sourceStartSec > 1e-9) {
                                    // 延伸：先向下消费窗口下方余量（→0），
                                    // 耗尽后继续增长的部分为静音尾巴，窗口保持不变。
                                    nextTrimStart = Math.max(
                                        0,
                                        base.sourceEndSec - nextLen * rate,
                                    );
                                    nextTrimStart = Math.min(nextTrimStart, base.sourceStartSec);
                                }
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                if (Math.abs(nextTrimStart - base.sourceStartSec) > 1e-12) {
                                    dispatch(
                                        setClipSourceRange({
                                            clipId: id,
                                            sourceStartSec: nextTrimStart,
                                        }),
                                    );
                                }
                            } else {
                                // 正放非 Loop：**派生窗口模型** —— 消费区间为
                                // [source_start, source_start + len·rate)，
                                // 落在媒体之外的部分渲染为静音。右缘移动只改
                                // 长度，source_end 随之派生（自愈历史数据中
                                // length 与窗口跨度不一致的状态；向右延伸不再
                                // 受源媒体长度限制）。
                                const derivedEnd = base.sourceStartSec + nextLen * rate;
                                dispatch(setClipLength({ clipId: id, lengthSec: nextLen }));
                                if (Math.abs(derivedEnd - base.sourceEndSec) > 1e-12) {
                                    dispatch(
                                        setClipSourceRange({
                                            clipId: id,
                                            sourceEndSec: derivedEnd,
                                        }),
                                    );
                                }
                            }
                        }
                    });
                    // 波纹（自动跟进）实时预览：以编辑区域“右缘净位移”为准
                    // （与后端区域化波纹一致，包含吸附与素材长度限制后的实际值）。
                    if (drag.rippleMode !== "off") {
                        const rippleRightDelta = computeRegionRightEdgeDelta(
                            drag,
                            sessionRef.current.clips,
                        );
                        if (Math.abs(rippleRightDelta) > 1e-9) {
                            applyRippleFollowerShift(
                                dispatch,
                                drag.rippleFollowers,
                                rippleRightDelta,
                            );
                        }
                    }
                    // 自动交叉淡化实时预览（右缘截短/延伸）。
                    previewAutoCrossfadeNow();
                    return;
                }

                if (drag.type === "stretch_right") {
                    const desiredRight = clamp(beat, drag.basestartSec + minLen, 10_000);
                    const rawLen = clamp(desiredRight - drag.basestartSec, minLen, 10_000);
                    const baseLen = Math.max(1e-6, Number(drag.baselengthSec) || 0);
                    const baseRate =
                        drag.basePlaybackRate > 0 && Number.isFinite(drag.basePlaybackRate)
                            ? drag.basePlaybackRate
                            : 1;
                    const nextRate = clamp((baseRate * baseLen) / Math.max(1e-6, rawLen), 0.1, 10);
                    const correctedLen = (baseRate * baseLen) / nextRate;
                    const scaledFades = scaleClipFadesForStretch({
                        baseFadeInSec: drag.basefadeInSec,
                        baseFadeOutSec: drag.basefadeOutSec,
                        baseLengthSec: drag.baselengthSec,
                        nextLengthSec: correctedLen,
                    });
                    dispatch(setClipLength({ clipId: drag.clipId, lengthSec: correctedLen }));
                    dispatch(setClipPlaybackRate({ clipId: drag.clipId, playbackRate: nextRate }));
                    // SnapOffset 随拉伸同步缩放：总比例 × 拖拽起始基准偏移。
                    dispatch(
                        setClipSnapOffset({
                            clipId: drag.clipId,
                            snapOffsetSec: scaleSnapOffsetForStretch(
                                drag.baseByClipId[drag.clipId]?.snapOffsetSec,
                                correctedLen / baseLen,
                                correctedLen,
                            ),
                        }),
                    );
                    dispatch(
                        setClipFades({
                            clipId: drag.clipId,
                            fadeInSec: scaledFades.fadeInSec,
                            fadeOutSec: scaledFades.fadeOutSec,
                        }),
                    );
                    // 波纹（自动跟进）实时预览：以编辑区域“右缘净位移”为准
                    // （与后端区域化波纹一致，包含吸附与素材长度限制后的实际值）。
                    if (drag.rippleMode !== "off") {
                        const rippleRightDelta = computeRegionRightEdgeDelta(
                            drag,
                            sessionRef.current.clips,
                        );
                        if (Math.abs(rippleRightDelta) > 1e-9) {
                            applyRippleFollowerShift(
                                dispatch,
                                drag.rippleFollowers,
                                rippleRightDelta,
                            );
                        }
                    }
                    // 自动交叉淡化实时预览（右缘拉伸）。
                    previewAutoCrossfadeNow();
                }
            });
        }

        function end() {
            const drag = editDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            editDragRef.current = null;
            if (
                drag.type === "trim_left" ||
                drag.type === "trim_right" ||
                drag.type === "stretch_left" ||
                drag.type === "stretch_right" ||
                drag.type === "crossfade_edges"
            ) {
                endSnapGesture();
                // 拖拽结束即时清除吸附竖线（endSnapGesture 深度归零也会兜底清理）。
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            }

            const isGroupStretch =
                drag.stretchGroup != null &&
                (drag.type === "stretch_left" || drag.type === "stretch_right");

            const clipNow = sessionRef.current.clips.find((c) => c.id === drag.clipId);
            if (!isGroupStretch && !clipNow) {
                if (gainUndoGroupPromise) {
                    void finishGainUndoGroup().catch(() => undefined);
                }
                dispatch(endInteraction());
                return;
            }
            const singleClipNow = clipNow ?? null;

            // 保存拉伸后的播放速率，persist 后重新应用（两阶段更新策略）
            let reapplyRates: Array<{ clipId: string; rate: number }> | null = null;

            const autoCrossfadeClipIds = drag.crossfadeClipIds;
            const shouldApplyAutoCrossfade =
                sessionRef.current.autoCrossfadeEnabled &&
                (drag.type === "trim_left" ||
                    drag.type === "trim_right" ||
                    drag.type === "stretch_left" ||
                    drag.type === "stretch_right");

            const runInsideUndoGroup = async (task: () => Promise<void>): Promise<void> => {
                await webApi.beginUndoGroup();
                try {
                    await task();
                } finally {
                    await webApi.endUndoGroup();
                }
            };

            const runWithOptionalAutoCrossfade = async (
                task: () => Promise<void>,
            ): Promise<void> => {
                if (!shouldApplyAutoCrossfade) {
                    await task();
                    // 开关关闭时也要清理“已脱离重叠”的自动交叉淡化，
                    // 保证 REAPER 导入等历史 auto 值不会盖住应恢复的手动 fade。
                    await applyDetachedAutoCrossfadeClears(
                        sessionRef.current,
                        autoCrossfadeClipIds,
                        dispatch,
                        drag.initialCrossfadeSides,
                        drag.editSides,
                    );
                    return;
                }

                await runInsideUndoGroup(async () => {
                    await task();
                    await applyAutoCrossfade(sessionRef.current, autoCrossfadeClipIds, dispatch, {
                        affectedSides: drag.initialCrossfadeSides,
                        editSides: drag.editSides,
                    });
                });
            };

            // 交互锁在最终持久化请求完成后才释放，
            // 避免 endInteraction() 到 fulfilled 之间的窗口内，
            // 其他 in-flight thunk 的旧快照覆盖前端乐观更新导致闪烁。

            let persistPromise: Promise<unknown> | null = null;
            if (isGroupStretch && drag.stretchGroup) {
                const stretchPatches = drag.stretchGroup.clipIds
                    .map((id) => {
                        const now = sessionRef.current.clips.find((c) => c.id === id);
                        if (!now) return null;
                        return {
                            clipId: id,
                            startSec: now.startSec,
                            lengthSec: now.lengthSec,
                            playbackRate: now.playbackRate,
                            snapOffsetSec: now.snapOffsetSec,
                            fadeInSec: now.fadeInSec,
                            fadeOutSec: now.fadeOutSec,
                        };
                    })
                    .filter(
                        (
                            patch,
                        ): patch is {
                            clipId: string;
                            startSec: number;
                            lengthSec: number;
                            playbackRate: number;
                            snapOffsetSec: number;
                            fadeInSec: number;
                            fadeOutSec: number;
                        } => patch != null,
                    );

                if (stretchPatches.length > 0) {
                    reapplyRates = stretchPatches
                        .filter((p) => p.playbackRate !== 1)
                        .map((p) => ({ clipId: p.clipId, rate: p.playbackRate }));
                    persistPromise = runInsideUndoGroup(async () => {
                        const stretchPersistPromises = stretchPatches.map((patch) =>
                            dispatch(
                                setClipStateRemote({
                                    clipId: patch.clipId,
                                    startSec: patch.startSec,
                                    lengthSec: patch.lengthSec,
                                    playbackRate: patch.playbackRate,
                                    snapOffsetSec: patch.snapOffsetSec,
                                    fadeInSec: patch.fadeInSec,
                                    fadeOutSec: patch.fadeOutSec,
                                    checkpoint: false,
                                }),
                            ).unwrap(),
                        );
                        await Promise.allSettled(stretchPersistPromises);

                        if (shouldApplyAutoCrossfade) {
                            await applyAutoCrossfade(
                                sessionRef.current,
                                autoCrossfadeClipIds,
                                dispatch,
                                {
                                    affectedSides: drag.initialCrossfadeSides,
                                    editSides: drag.editSides,
                                },
                            );
                        }
                    });
                }
            } else if (drag.type === "trim_left" && singleClipNow) {
                if (drag.selectedClipIds.length > 1) {
                    const trimPatches = drag.selectedClipIds
                        .map((id) => {
                            const now = sessionRef.current.clips.find((c) => c.id === id);
                            if (!now) return null;
                            return {
                                clipId: id,
                                startSec: now.startSec,
                                lengthSec: now.lengthSec,
                                reversed: now.reversed,
                                sourceStartSec: now.sourceStartSec,
                                sourceEndSec: now.sourceEndSec,
                            };
                        })
                        .filter((p) => p != null);
                    if (trimPatches.length > 0) {
                        persistPromise = runInsideUndoGroup(async () => {
                            const promises = trimPatches.map((patch) => {
                                const src = patch.reversed
                                    ? { sourceEndSec: patch.sourceEndSec }
                                    : { sourceStartSec: patch.sourceStartSec };
                                return dispatch(
                                    setClipStateRemote({
                                        clipId: patch.clipId,
                                        startSec: patch.startSec,
                                        lengthSec: patch.lengthSec,
                                        ...src,
                                        checkpoint: false,
                                    }),
                                ).unwrap();
                            });
                            await Promise.allSettled(promises);
                            if (shouldApplyAutoCrossfade) {
                                await applyAutoCrossfade(
                                    sessionRef.current,
                                    autoCrossfadeClipIds,
                                    dispatch,
                                    {
                                        affectedSides: drag.initialCrossfadeSides,
                                        editSides: drag.editSides,
                                    },
                                );
                            }
                        });
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceEndSec: singleClipNow.sourceEndSec }
                        : { sourceStartSec: singleClipNow.sourceStartSec };
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = runWithOptionalAutoCrossfade(async () => {
                            await dispatch(
                                setClipStateRemote({
                                    clipId: drag.clipId,
                                    startSec: singleClipNow.startSec,
                                    lengthSec: singleClipNow.lengthSec,
                                    ...sourceRangePatch,
                                    checkpoint: false,
                                }),
                            ).unwrap();
                        });
                    } else {
                        persistPromise = dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                startSec: singleClipNow.startSec,
                                lengthSec: singleClipNow.lengthSec,
                                ...sourceRangePatch,
                            }),
                        ).unwrap();
                    }
                }
            } else if (drag.type === "trim_right" && singleClipNow) {
                if (drag.selectedClipIds.length > 1) {
                    const trimPatches = drag.selectedClipIds
                        .map((id) => {
                            const now = sessionRef.current.clips.find((c) => c.id === id);
                            if (!now) return null;
                            return {
                                clipId: id,
                                lengthSec: now.lengthSec,
                                reversed: now.reversed,
                                sourceStartSec: now.sourceStartSec,
                                sourceEndSec: now.sourceEndSec,
                            };
                        })
                        .filter((p) => p != null);
                    if (trimPatches.length > 0) {
                        persistPromise = runInsideUndoGroup(async () => {
                            const promises = trimPatches.map((patch) => {
                                const src = patch.reversed
                                    ? { sourceStartSec: patch.sourceStartSec }
                                    : { sourceEndSec: patch.sourceEndSec };
                                return dispatch(
                                    setClipStateRemote({
                                        clipId: patch.clipId,
                                        lengthSec: patch.lengthSec,
                                        ...src,
                                        checkpoint: false,
                                    }),
                                ).unwrap();
                            });
                            await Promise.allSettled(promises);
                            if (shouldApplyAutoCrossfade) {
                                await applyAutoCrossfade(
                                    sessionRef.current,
                                    autoCrossfadeClipIds,
                                    dispatch,
                                    {
                                        affectedSides: drag.initialCrossfadeSides,
                                        editSides: drag.editSides,
                                    },
                                );
                            }
                        });
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceStartSec: singleClipNow.sourceStartSec }
                        : { sourceEndSec: singleClipNow.sourceEndSec };
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = runWithOptionalAutoCrossfade(async () => {
                            await dispatch(
                                setClipStateRemote({
                                    clipId: drag.clipId,
                                    lengthSec: singleClipNow.lengthSec,
                                    ...sourceRangePatch,
                                    checkpoint: false,
                                }),
                            ).unwrap();
                        });
                    } else {
                        persistPromise = dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                lengthSec: singleClipNow.lengthSec,
                                ...sourceRangePatch,
                            }),
                        ).unwrap();
                    }
                }
            } else if (drag.type === "stretch_left" && singleClipNow) {
                if (shouldApplyAutoCrossfade) {
                    persistPromise = runWithOptionalAutoCrossfade(async () => {
                        await dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                startSec: singleClipNow.startSec,
                                lengthSec: singleClipNow.lengthSec,
                                playbackRate: singleClipNow.playbackRate,
                                snapOffsetSec: singleClipNow.snapOffsetSec,
                                fadeInSec: singleClipNow.fadeInSec,
                                fadeOutSec: singleClipNow.fadeOutSec,
                                checkpoint: false,
                            }),
                        ).unwrap();
                    });
                } else {
                    persistPromise = dispatch(
                        setClipStateRemote({
                            clipId: drag.clipId,
                            startSec: singleClipNow.startSec,
                            lengthSec: singleClipNow.lengthSec,
                            playbackRate: singleClipNow.playbackRate,
                            snapOffsetSec: singleClipNow.snapOffsetSec,
                            fadeInSec: singleClipNow.fadeInSec,
                            fadeOutSec: singleClipNow.fadeOutSec,
                        }),
                    ).unwrap();
                }
                if (singleClipNow.playbackRate !== 1) {
                    reapplyRates = [{ clipId: drag.clipId, rate: singleClipNow.playbackRate }];
                }
            } else if (drag.type === "stretch_right" && singleClipNow) {
                if (shouldApplyAutoCrossfade) {
                    persistPromise = runWithOptionalAutoCrossfade(async () => {
                        await dispatch(
                            setClipStateRemote({
                                clipId: drag.clipId,
                                lengthSec: singleClipNow.lengthSec,
                                playbackRate: singleClipNow.playbackRate,
                                snapOffsetSec: singleClipNow.snapOffsetSec,
                                fadeInSec: singleClipNow.fadeInSec,
                                fadeOutSec: singleClipNow.fadeOutSec,
                                checkpoint: false,
                            }),
                        ).unwrap();
                    });
                } else {
                    persistPromise = dispatch(
                        setClipStateRemote({
                            clipId: drag.clipId,
                            lengthSec: singleClipNow.lengthSec,
                            playbackRate: singleClipNow.playbackRate,
                            snapOffsetSec: singleClipNow.snapOffsetSec,
                            fadeInSec: singleClipNow.fadeInSec,
                            fadeOutSec: singleClipNow.fadeOutSec,
                        }),
                    ).unwrap();
                }
                if (singleClipNow.playbackRate !== 1) {
                    reapplyRates = [{ clipId: drag.clipId, rate: singleClipNow.playbackRate }];
                }
            } else if (drag.type === "crossfade_edges") {
                const patches = drag.selectedClipIds
                    .map((id) => {
                        const now = sessionRef.current.clips.find((c) => c.id === id);
                        if (!now) return null;
                        return {
                            clipId: id,
                            startSec: now.startSec,
                            lengthSec: now.lengthSec,
                            sourceStartSec: now.sourceStartSec,
                            sourceEndSec: now.sourceEndSec,
                            fadeInSec: now.fadeInSec,
                            fadeOutSec: now.fadeOutSec,
                            autoFadeInSec: now.autoFadeInSec ?? 0,
                            autoFadeOutSec: now.autoFadeOutSec ?? 0,
                        };
                    })
                    .filter(
                        (
                            patch,
                        ): patch is {
                            clipId: string;
                            startSec: number;
                            lengthSec: number;
                            sourceStartSec: number;
                            sourceEndSec: number;
                            fadeInSec: number;
                            fadeOutSec: number;
                            autoFadeInSec: number;
                            autoFadeOutSec: number;
                        } => patch != null,
                    );
                if (patches.length > 0) {
                    const commitPatches = async () => {
                        const promises = patches.map((patch) =>
                            dispatch(
                                setClipStateRemote({
                                    clipId: patch.clipId,
                                    startSec: patch.startSec,
                                    lengthSec: patch.lengthSec,
                                    sourceStartSec: patch.sourceStartSec,
                                    sourceEndSec: patch.sourceEndSec,
                                    fadeInSec: patch.fadeInSec,
                                    fadeOutSec: patch.fadeOutSec,
                                    autoFadeInSec: patch.autoFadeInSec,
                                    autoFadeOutSec: patch.autoFadeOutSec,
                                    checkpoint: false,
                                }),
                            ).unwrap(),
                        );
                        await Promise.allSettled(promises);
                    };
                    persistPromise = crossfadeUndoGroupPromise
                        ? crossfadeUndoGroupPromise.then(commitPatches)
                        : commitPatches();
                }
            } else if (drag.type === "fade_in" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { fadeInSec: nextClip?.fadeInSec ?? 0 }] as const;
                    }),
                );
                persistPromise = dispatch(
                    setClipsStateBulkRemote({
                        updates: buildBulkClipStateUpdates({
                            clipIds: drag.selectedClipIds,
                            changesById,
                        }),
                        // 在 fade undo group 内：最终写入不产生独立撤销步，
                        // 整个 fade 拖拽 = 单个撤销步。
                        checkpoint: false,
                    }),
                )
                    .unwrap()
                    .then(() => {
                        // 用户手动拖拽淡入 → 手动 fade 生效，清除该侧自动交叉淡化。
                        dispatch(setClipAutoFades({ clipId: drag.clipId, autoFadeInSec: 0 }));
                        return webApi.setClipState({
                            clipId: drag.clipId,
                            autoFadeInSec: 0,
                            checkpoint: false,
                        });
                    });
            } else if (drag.type === "fade_out" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { fadeOutSec: nextClip?.fadeOutSec ?? 0 }] as const;
                    }),
                );
                persistPromise = dispatch(
                    setClipsStateBulkRemote({
                        updates: buildBulkClipStateUpdates({
                            clipIds: drag.selectedClipIds,
                            changesById,
                        }),
                        // 在 fade undo group 内：最终写入不产生独立撤销步。
                        checkpoint: false,
                    }),
                )
                    .unwrap()
                    .then(() => {
                        // 用户手动拖拽淡出 → 手动 fade 生效，清除该侧自动交叉淡化。
                        dispatch(setClipAutoFades({ clipId: drag.clipId, autoFadeOutSec: 0 }));
                        return webApi.setClipState({
                            clipId: drag.clipId,
                            autoFadeOutSec: 0,
                            checkpoint: false,
                        });
                    });
            } else if (drag.type === "gain" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        return [clipId, { gain: nextClip?.gain ?? 1 }] as const;
                    }),
                );
                const persistBulkGain = () =>
                    remotePreviewChain.then(() =>
                        dispatch(
                            setClipsStateBulkRemote({
                                updates: buildBulkClipStateUpdates({
                                    clipIds: drag.selectedClipIds,
                                    changesById,
                                }),
                            }),
                        ).unwrap(),
                    );
                persistPromise = gainUndoGroupPromise
                    ? gainUndoGroupPromise.then(persistBulkGain)
                    : persistBulkGain();
            }

            // 两阶段播放速率更新：后端响应后重新应用前端计算的值
            if (reapplyRates && reapplyRates.length > 0 && persistPromise) {
                void persistPromise.then(() => {
                    for (const { clipId, rate } of reapplyRates!) {
                        dispatch(setClipPlaybackRate({ clipId, playbackRate: rate }));
                    }
                });
            }

            // 拉伸后同步参数线：当"锁定参数线"启用时，将旧范围内的参数值时域映射到新范围
            const isStretch = drag.type === "stretch_left" || drag.type === "stretch_right";
            // The clip persistence already bumps paramsEpoch, but at that point
            // stretchLinkedParams has not written the new curves yet. Bump again
            // after the mapping finishes so the parameter editor fetches fresh data.
            if (isStretch && sessionRef.current.lockParamLinesEnabled && drag.stretchGroup) {
                const mappingsByRootTrack = new Map<string, StretchRangeMapping[]>();
                for (const id of drag.stretchGroup.clipIds) {
                    const initial = drag.stretchGroup?.initialById[id];
                    const now = sessionRef.current.clips.find((c) => c.id === id);
                    if (!initial || !now?.trackId) continue;
                    const rootTrackId = resolveRootTrackId(sessionRef.current.tracks, now.trackId);
                    if (!rootTrackId) continue;
                    const trackMappings = mappingsByRootTrack.get(rootTrackId) ?? [];
                    trackMappings.push({
                        oldStartSec: initial.startSec,
                        oldLengthSec: initial.lengthSec,
                        newStartSec: now.startSec,
                        newLengthSec: now.lengthSec,
                    });
                    mappingsByRootTrack.set(rootTrackId, trackMappings);
                }
                const stretchTasks = Array.from(mappingsByRootTrack, ([trackId, mappings]) =>
                    stretchTrackLinkedParams(trackId, mappings),
                );
                void Promise.resolve(persistPromise)
                    .then(() => Promise.allSettled(stretchTasks))
                    .finally(() => dispatch(bumpParamsEpoch()));
            } else if (
                isStretch &&
                sessionRef.current.lockParamLinesEnabled &&
                singleClipNow?.trackId
            ) {
                const stretchTrackId = singleClipNow.trackId;
                const oldStartSec = drag.basestartSec;
                const oldLengthSec = drag.baselengthSec;
                const newStartSec = singleClipNow.startSec;
                const newLengthSec = singleClipNow.lengthSec;
                void Promise.resolve(persistPromise)
                    .then(() =>
                        stretchLinkedParams(
                            stretchTrackId,
                            oldStartSec,
                            oldLengthSec,
                            newStartSec,
                            newLengthSec,
                        ),
                    )
                    .finally(() => dispatch(bumpParamsEpoch()));
            }

            // 在所有持久化请求完成后释放交互锁
            void Promise.resolve(persistPromise).finally(async () => {
                if (gainUndoGroupPromise) {
                    try {
                        await finishGainUndoGroup();
                    } catch {
                        // Best-effort undo-group cleanup.
                    }
                }
                if (fadeUndoGroupPromise) {
                    try {
                        await finishFadeUndoGroup();
                    } catch {
                        // Best-effort undo-group cleanup.
                    }
                }
                if (crossfadeUndoGroupPromise) {
                    try {
                        await finishCrossfadeUndoGroup();
                    } catch {
                        // Best-effort undo-group cleanup.
                    }
                }
                dispatch(endInteraction());
            });

            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { editDragRef, startEditDrag };
}
