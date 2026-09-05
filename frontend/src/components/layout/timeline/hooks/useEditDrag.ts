import { useRef } from "react";
import { batch } from "react-redux";
import { registerDragAbort } from "../gestureFocusGuard";
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
import { resolveCurvatureEditBase, solveNearestCurveDir } from "../reaperFade";
import { modifierWatcher } from "./modifierWatcher";

/**
 * 拖拽发起方经"延迟起手"后带给 startEditDrag 的类型化私有通道。
 * 曾经这些字段被直接塞到伪造的事件对象上（不可类型的字符串走私），
 * 现改为显式可选参数；事件对象只承担 React PointerEvent 本身的职责。
 */
export type EditDragChannelOpts = {
    /** 延迟起手时的按下 X（相对拖拽锚点用）。 */
    dragStartClientX?: number;
    /** 交叉点拖拽的后一个 clip id。 */
    crossfadePartnerClipId?: string | null;
    /**
     * Alt 曲率拖拽环境：包络 gain=1 基准线的客户 Y 与 body 高度，
     * 由发起组件在按下瞬间快照。缺失时曲率分支自动回退为长度模式。
     */
    fadePointerEnv?: { envTopClientY: number; bodyHeightPx: number } | null;
};

/** 曲率拖拽的指针→增益映射环境（由发起组件在按下瞬间快照）。 */
type FadeCurvePointerEnv = NonNullable<EditDragChannelOpts["fadePointerEnv"]>;

function sanitizeFadeEnv(raw: EditDragChannelOpts["fadePointerEnv"]): FadeCurvePointerEnv | null {
    if (!raw) return null;
    const top = Number(raw.envTopClientY);
    const height = Number(raw.bodyHeightPx);
    if (!Number.isFinite(top) || !Number.isFinite(height) || height <= 0) return null;
    return { envTopClientY: top, bodyHeightPx: height };
}
import type { Keybinding } from "../../../../features/keybindings/types";
import type { TimelineSnapSettings } from "../../../../features/session/sessionTypes";
import { resolveClipContentDurationSec } from "../../../../utils/loopRender";
import { loopSnapThresholdSec, nearestBoundarySnapOffsetSec } from "../../../../utils/loopSnap";
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

type StretchRangeMapping = {
    oldStartSec: number;
    oldLengthSec: number;
    newStartSec: number;
    newLengthSec: number;
};

/**
 * 拉伸后对参数线进行时域映射（拉伸或压缩）。
 *
 * 映射由后端 `stretch_track_linked_params` 一次性完成：pitch（用户编辑过时）、
 * tension 以及该根轨道上所有已存在的自动化曲线（volume/气声/子轨道偏移等，
 * 无论参数是否在 UI 中激活或有数据）。旧的前端实现只映射 pitch+tension，
 * 导致其余参数线在拉伸后遗留在旧位置，剪辑新范围内表现为被初始化。
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
    await stretchTrackLinkedParams(trackId, [
        { oldStartSec, oldLengthSec, newStartSec, newLengthSec },
    ]);
}

/**
 * Stretch parameter lines for several clips on the same root track as one
 * batch. The backend writes all new ranges first, then restores old-range
 * parts not covered by any new range, so neighbouring clips cannot erase
 * each other's freshly written values.
 */
async function stretchTrackLinkedParams(
    trackId: string,
    mappings: StretchRangeMapping[],
): Promise<void> {
    if (mappings.length === 0) return;
    await paramsApi.stretchTrackLinkedParams(trackId, mappings, false);
}

/**
 * 曲率拖拽：把指针时间/客户 Y 映射到曲线归一化坐标。
 * t 夹紧到 (0.001,0.999) 避免端点处 dir 反解退化；gain 由包络基准线换算。
 */
function resolveCurvePointer(
    env: FadeCurvePointerEnv,
    side: { leftSec: number; widthSec: number },
    pointerSec: number,
    clientY: number,
): { t: number; gain: number } | null {
    if (!(side.widthSec > 1e-9)) return null;
    const t = Math.min(0.999, Math.max(0.001, (pointerSec - side.leftSec) / side.widthSec));
    const gain = Math.min(
        1,
        Math.max(0, 1 - (clientY - env.envTopClientY) / Math.max(1, env.bodyHeightPx)),
    );
    return { t, gain };
}

/**
 * 淡化拖拽不再自带浮标：悬停信息 ToolTips 由 FadeHitLayer/OverlapEditLayer
 * 通过 data-tooltip 展示（类型/长度/曲率），拖拽时 AppTooltip 持续跟随指针，
 * 已覆盖实时反馈需求。
 */
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
    baseClipPlaybackRate: number;
    baseSourceStartSec: number;
    baseSourceEndSec: number;
    basefadeInSec: number;
    basefadeOutSec: number;
    /**
     * 每个被编辑 clip **手动**淡入淡出长度（fadeInSec/fadeOutSec 原值，
     * 与"有效长度"（basefadeInSec = auto>0 ? auto : manual）区分）。
     * 松手判定"本次手势是否真的改了手动长度"：纯曲率拖拽不改长度，
     * 必须**保留自动交叉淡化**（只写形状/曲率）；只有长度被改过的手势
     * 才落盘手动长度并清除该侧自动交叉淡化（自动 → 手动转换）。
     */
    baseManualFadeSecByClipId: Record<string, { in: number; out: number }>;
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
    /**
     * Alt 曲率拖拽：按下瞬间的指针→增益映射环境。
     * 缺失（角落手柄创建淡化等场景）时修饰键分支自动回退到长度模式。
     */
    fadeCurveEnv: FadeCurvePointerEnv | null;
    /** 单侧曲率快照：冻结的区域（秒，拖拽中长度不变）、形状与基准 dir。
     *  promoteFromLinear：源形状是线性（曲率对其无可见效果），首次实际
     *  编辑时必须同时提交提升后的形状（见 reaperFade.resolveCurvatureEditBase）。 */
    fadeCurveSide: {
        leftSec: number;
        widthSec: number;
        shape: number;
        baseDir: number;
        promoteFromLinear: boolean;
    } | null;
    /** 交叉点曲率快照：前 clip 淡出 / 后 clip 淡入两侧各自独立求解。 */
    crossfadeCurveSides: {
        a: {
            clipId: string;
            leftSec: number;
            widthSec: number;
            shape: number;
            baseDir: number;
            promoteFromLinear: boolean;
        };
        b: {
            clipId: string;
            leftSec: number;
            widthSec: number;
            shape: number;
            baseDir: number;
            promoteFromLinear: boolean;
        };
    } | null;
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
function computeRegionRightEdgeDelta(drag: EditDragState, clips: SessionState["clips"]): number {
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
    /** modifier.fadeCurvatureDrag 绑定（按住并拖动淡化包络线 = 调整曲率） */
    fadeCurvatureKb: Keybinding;
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
        fadeCurvatureKb,
    } = deps;

    const editDragRef = useRef<EditDragState | null>(null);
    // 用于节流向后端发送 clip 状态更新，避免拖动时频繁覆盖与后端同步引起闪烁
    const lastRemoteSentRef = useRef<Record<string, number>>({});

    function startEditDrag(
        e: React.PointerEvent,
        clipId: string,
        type: EditDragType,
        channel: EditDragChannelOpts = {},
    ) {
        if (e.button !== 0) return;
        const clip = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!clip) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const rightEdgeBeat = clip.startSec + clip.lengthSec;
        // 淡入淡出的相对拖拽锚点：以“鼠标按下位置”（deferred 起点通过 dragStartClientX
        // 传入）作为零偏移，而不是“实时把边缘线对齐到指针位置”。这样从包络线中间
        // 开始拖拽也不会发生跳变。
        const dragStartClientX = channel.dragStartClientX ?? e.clientX;
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
            type === "crossfade_edges" ? (channel.crossfadePartnerClipId ?? null) : null;
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
            const editedClip =
                id === clipId ? clip : sessionRef.current.clips.find((x) => x.id === id);
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
            } else if (type === "trim_right" || type === "stretch_right" || type === "fade_out") {
                editSides[id] = { fadeIn: false, fadeOut: true };
            } else {
                editSides[id] = { fadeIn: false, fadeOut: false };
            }
        }

        // 淡入淡出相对拖拽的“视觉/有效”起点：自动交叉淡化生效时用自动长度，
        // 否则用手动长度。这样从“自动交叉淡化”直接拖成“手动淡入淡出”时，
        // 以用户当前看到的长度作为起点，拖拽过程不会从自动值跳变到隐藏的手动值。
        const effectiveFadeInSec =
            Number(clip.autoFadeInSec ?? 0) > 0
                ? Number(clip.autoFadeInSec)
                : Number(clip.fadeInSec);
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

        // Alt 曲率拖拽快照：区域以秒冻结（拖拽中不改长度），形状沿用该侧当前
        // 值；求解器输出新 dir。交叉点模式同时快照两侧。
        const fadePointerEnv = sanitizeFadeEnv(channel.fadePointerEnv);
        let fadeCurveSide: EditDragState["fadeCurveSide"] = null;
        if (type === "fade_in" || type === "fade_out") {
            const widthSec = Math.max(
                0,
                Math.min(
                    type === "fade_in" ? effectiveFadeInSec : effectiveFadeOutSec,
                    clip.lengthSec,
                ),
            );
            const rawShape =
                (type === "fade_in" ? Number(clip.fadeInShape) : Number(clip.fadeOutShape)) || 0;
            const curvatureBase = resolveCurvatureEditBase(rawShape);
            fadeCurveSide = {
                leftSec:
                    type === "fade_in" ? clip.startSec : clip.startSec + clip.lengthSec - widthSec,
                widthSec,
                shape: curvatureBase.shape,
                baseDir:
                    (type === "fade_in" ? Number(clip.fadeInDir) : Number(clip.fadeOutDir)) || 0,
                promoteFromLinear: curvatureBase.promotedFromLinear,
            };
        }
        let crossfadeCurveSides: EditDragState["crossfadeCurveSides"] = null;
        if (type === "crossfade_edges" && crossfadePartnerClip) {
            const widthA = Math.max(0, Math.min(effectiveFadeOutSec, clip.lengthSec));
            const widthB = Math.max(
                0,
                Math.min(crossfadePartnerFadeInSec, crossfadePartnerClip.lengthSec),
            );
            const baseA = resolveCurvatureEditBase(Number(clip.fadeOutShape) || 0);
            const baseB = resolveCurvatureEditBase(Number(crossfadePartnerClip.fadeInShape) || 0);
            crossfadeCurveSides = {
                a: {
                    clipId: clip.id,
                    leftSec: clip.startSec + clip.lengthSec - widthA,
                    widthSec: widthA,
                    shape: baseA.shape,
                    baseDir: Number(clip.fadeOutDir) || 0,
                    promoteFromLinear: baseA.promotedFromLinear,
                },
                b: {
                    clipId: crossfadePartnerClip.id,
                    leftSec: crossfadePartnerClip.startSec,
                    widthSec: widthB,
                    shape: baseB.shape,
                    baseDir: Number(crossfadePartnerClip.fadeInDir) || 0,
                    promoteFromLinear: baseB.promotedFromLinear,
                },
            };
        }
        // 多选曲率拖拽（曾按 per-clip 环境广播）已撤回：曲率拖拽的指针 Y 必须
        // 映射到命中 clip 自己的"增益=1"基线，跨 clip 共用同一 Y 会得到错误
        // 的曲率（对齐 REAPER 只改当前 item），故只保留 anchor 的 fadeCurveEnv。
        editDragRef.current = {
            type,
            pointerId: e.pointerId,
            clipId,
            basestartSec: clip.startSec,
            baselengthSec: clip.lengthSec,
            basePlaybackRate: Number(clip.playbackRate ?? 1) || 1,
            baseClipPlaybackRate: Number(clip.clipPlaybackRate ?? 1) || 1,
            baseSourceStartSec: clip.sourceStartSec,
            baseSourceEndSec: clip.sourceEndSec,
            basefadeInSec: effectiveFadeInSec,
            basefadeOutSec: effectiveFadeOutSec,
            // 手动长度基准（逐 clip）：决定松手时是否把"自动交叉淡化"转成
            // 手动 fade —— 只有长度被本手势改动的 clip 才转换（见类型注释）。
            baseManualFadeSecByClipId: Object.fromEntries(
                selectedClipIds.map((id) => {
                    const c =
                        id === clipId ? clip : sessionRef.current.clips.find((x) => x.id === id);
                    return [
                        id,
                        {
                            in: Number(c?.fadeInSec ?? 0) || 0,
                            out: Number(c?.fadeOutSec ?? 0) || 0,
                        },
                    ];
                }),
            ),
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
            fadeCurveEnv: fadePointerEnv,
            fadeCurveSide,
            crossfadeCurveSides,
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

        // 失焦取消：切屏期间 pointerup/pointercancel 不会送达本窗口，拖拽
        // 会永久卡死（交互锁/undo group/吸附高亮全部悬置）。注册事件无关的
        // end()，由 gestureFocusGuard 在窗口 blur 时统一收尾 —— 走的就是
        // pointerup/pointercancel 的同一条 end()，undo group 在 finally 中
        // 必然关闭，后端撤销栈不会被冻结。
        const unregisterAbort = registerDragAbort(end);

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
                // 自愈：持续从真实原生事件刷新全局修饰键快照，覆盖 keydown
                // 被其他层拦截、窗口失焦恢复等导致的按键状态漂移。
                modifierWatcher.refreshFromEvent(currentEv);

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
                    const leftEdge = drag.type === "trim_right" || drag.type === "stretch_right";
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
                    // ── Alt 曲率模式：两条包络线分别求解“经过指针点”的新曲率，
                    // 视觉上等价于把交叉点拖到光标位置（对齐 REAPER 在交叉上下文
                    // 的 Alt 行为）。边缘位置与淡化长度完全不动。
                    if (
                        modifierWatcher.isKeybindingActive(fadeCurvatureKb, currentEv) &&
                        drag.fadeCurveEnv &&
                        drag.crossfadeCurveSides
                    ) {
                        const sides = drag.crossfadeCurveSides;
                        const pa = resolveCurvePointer(
                            drag.fadeCurveEnv,
                            sides.a,
                            beat,
                            currentEv.clientY,
                        );
                        const pb = resolveCurvePointer(
                            drag.fadeCurveEnv,
                            sides.b,
                            beat,
                            currentEv.clientY,
                        );
                        if (!pa || !pb) return;
                        // 最近点投影求解：两侧各自独立，平坦带平滑无瞬变。
                        const dirA = solveNearestCurveDir({
                            shape: sides.a.shape,
                            dir: sides.a.baseDir,
                            mode: "out",
                            pointerX01: pa.t,
                            pointerY01: pa.gain,
                            aspectYOverX:
                                drag.fadeCurveEnv.bodyHeightPx /
                                Math.max(1, sides.a.widthSec * pxPerSec),
                        }).dir;
                        const dirB = solveNearestCurveDir({
                            shape: sides.b.shape,
                            dir: sides.b.baseDir,
                            mode: "in",
                            pointerX01: pb.t,
                            pointerY01: pb.gain,
                            aspectYOverX:
                                drag.fadeCurveEnv.bodyHeightPx /
                                Math.max(1, sides.b.widthSec * pxPerSec),
                        }).dir;
                        sides.a.baseDir = dirA;
                        sides.b.baseDir = dirB;
                        batch(() => {
                            dispatch(setClipFades({ clipId: sides.a.clipId, fadeOutDir: dirA }));
                            dispatch(setClipFades({ clipId: sides.b.clipId, fadeInDir: dirB }));
                        });
                        try {
                            const now = Date.now();
                            const key = `${sides.a.clipId}:${sides.b.clipId}:curve-cross`;
                            const last = lastRemoteSentRef.current[key] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[key] = now;
                                void webApi.setClipState({
                                    clipId: sides.a.clipId,
                                    fadeOutDir: dirA,
                                    checkpoint: false,
                                });
                                void webApi.setClipState({
                                    clipId: sides.b.clipId,
                                    fadeInDir: dirB,
                                    checkpoint: false,
                                });
                            }
                        } catch {
                            // Best-effort remote preview update.
                        }
                        return;
                    }
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
                            let nextTrimStart = Math.max(0, base.sourceStartSec - delta * rate);
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
                                lengthSec: Math.max(0, base.lengthSec - startDelta),
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
                        const newOverlap = Math.max(
                            0.0002,
                            drag.crossfadeBaseOverlapSec + 2 * delta,
                        );
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
                    // ── Alt 曲率模式（每帧实时求值，可与长度模式无缝互切）──
                    if (
                        modifierWatcher.isKeybindingActive(fadeCurvatureKb, currentEv) &&
                        drag.fadeCurveEnv &&
                        drag.fadeCurveSide
                    ) {
                        const pt = resolveCurvePointer(
                            drag.fadeCurveEnv,
                            drag.fadeCurveSide,
                            beat,
                            currentEv.clientY,
                        );
                        if (!pt) return;
                        const nextDir = solveNearestCurveDir({
                            shape: drag.fadeCurveSide.shape,
                            dir: drag.fadeCurveSide.baseDir,
                            mode: "in",
                            pointerX01: pt.t,
                            pointerY01: pt.gain,
                            aspectYOverX:
                                drag.fadeCurveEnv.bodyHeightPx /
                                Math.max(1, drag.fadeCurveSide.widthSec * pxPerSec),
                        }).dir;
                        const side = drag.fadeCurveSide;
                        side.baseDir = nextDir;
                        // 曲率只作用于当前 clip 的该侧：指针 Y 必须映射到该 clip
                        // 自己的"增益=1"基线，而各行/各 clip 的 body 几何不同，
                        // 无法跨 clip 共用同一指针 Y（对齐 REAPER，只改当前 item）。
                        dispatch(setClipFades({ clipId: drag.clipId, fadeInDir: nextDir }));
                        try {
                            const now = Date.now();
                            const key = `${drag.clipId}:curve-in`;
                            const last = lastRemoteSentRef.current[key] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[key] = now;
                                void webApi.setClipState({
                                    clipId: drag.clipId,
                                    fadeInDir: nextDir,
                                    checkpoint: false,
                                });
                            }
                        } catch {
                            // Best-effort remote preview update.
                        }
                        return;
                    }
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
                        // 多选：逐个选中 clip 发送节流预览 + 清除该侧自动交叉淡化。
                        const now = Date.now();
                        const nextByClipId = new Map(fadeUpdates.map((u) => [u.clipId, u]));
                        for (const id of drag.selectedClipIds) {
                            const perClipNext =
                                (nextByClipId.get(id) as { fadeInSec?: number } | undefined)
                                    ?.fadeInSec ?? next;
                            const key = `${id}:fade-in`;
                            const last = lastRemoteSentRef.current[key] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[key] = now;
                                // 手动拖拽淡入 = 用户手动 fade，且清除该侧自动交叉淡化。
                                dispatch(setClipAutoFades({ clipId: id, autoFadeInSec: 0 }));
                                // 直接 webApi 持久化（不走 thunk）：其 fulfilled 不会 force-apply
                                // 整份时间线覆盖本地乐观值，避免拖拽中淡入淡出包络闪烁。
                                void webApi.setClipState({
                                    clipId: id,
                                    fadeInSec: perClipNext,
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
                    // ── Alt 曲率模式 ──
                    if (
                        modifierWatcher.isKeybindingActive(fadeCurvatureKb, currentEv) &&
                        drag.fadeCurveEnv &&
                        drag.fadeCurveSide
                    ) {
                        // 曲率只作用于当前 clip 的该侧（理由见 fade_in 分支注释）。
                        const pt = resolveCurvePointer(
                            drag.fadeCurveEnv,
                            drag.fadeCurveSide,
                            beat,
                            currentEv.clientY,
                        );
                        if (!pt) return;
                        const nextDir = solveNearestCurveDir({
                            shape: drag.fadeCurveSide.shape,
                            dir: drag.fadeCurveSide.baseDir,
                            mode: "out",
                            pointerX01: pt.t,
                            pointerY01: pt.gain,
                            aspectYOverX:
                                drag.fadeCurveEnv.bodyHeightPx /
                                Math.max(1, drag.fadeCurveSide.widthSec * pxPerSec),
                        }).dir;
                        const side = drag.fadeCurveSide;
                        side.baseDir = nextDir;
                        dispatch(setClipFades({ clipId: drag.clipId, fadeOutDir: nextDir }));
                        try {
                            const now = Date.now();
                            const key = `${drag.clipId}:curve-out`;
                            const last = lastRemoteSentRef.current[key] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[key] = now;
                                void webApi.setClipState({
                                    clipId: drag.clipId,
                                    fadeOutDir: nextDir,
                                    checkpoint: false,
                                });
                            }
                        } catch {
                            // Best-effort remote preview update.
                        }
                        return;
                    }
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
                        // 多选：逐个选中 clip 发送节流预览 + 清除该侧自动交叉淡化。
                        const now = Date.now();
                        const nextByClipId = new Map(fadeUpdates.map((u) => [u.clipId, u]));
                        for (const id of drag.selectedClipIds) {
                            const perClipNext =
                                (nextByClipId.get(id) as { fadeOutSec?: number } | undefined)
                                    ?.fadeOutSec ?? next;
                            const key = `${id}:fade-out`;
                            const last = lastRemoteSentRef.current[key] || 0;
                            if (now - last > 200) {
                                lastRemoteSentRef.current[key] = now;
                                // 手动拖拽淡出 = 手动 fade，且清除该侧自动交叉淡化。
                                dispatch(setClipAutoFades({ clipId: id, autoFadeOutSec: 0 }));
                                // 直接 webApi 持久化（不走 thunk）：避免 force-apply 覆盖本地
                                // 乐观 fade 导致拖拽中淡入淡出包络闪烁。
                                void webApi.setClipState({
                                    clipId: id,
                                    fadeOutSec: perClipNext,
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
                                    const ratio = next.lengthSec / Math.max(1e-6, base.lengthSec);
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
                                    clipPlaybackRate: next.clipPlaybackRate,
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
                                dispatch(
                                    moveClipStart({
                                        clipId: id,
                                        startSec: base.startSec + limitedDelta,
                                    }),
                                );
                                dispatch(
                                    setClipLength({
                                        clipId: id,
                                        lengthSec: clamp(
                                            base.lengthSec - limitedDelta,
                                            minLen,
                                            10_000,
                                        ),
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
                                let nextTrimEnd = base.sourceEndSec - limitedDelta * rate;
                                nextTrimEnd = wrapIntoMediaDomain(nextTrimEnd, base);
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
                                let nextTrimStart = base.sourceStartSec + limitedDelta * rate;
                                nextTrimStart = wrapIntoMediaDomain(nextTrimStart, base);
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
                        drag.baseClipPlaybackRate > 0 && Number.isFinite(drag.baseClipPlaybackRate)
                            ? drag.baseClipPlaybackRate
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
                    dispatch(
                        setClipPlaybackRate({ clipId: drag.clipId, clipPlaybackRate: nextRate }),
                    );
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
                                    nextTrimStart = Math.max(0, base.sourceEndSec - nextLen * rate);
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
                        drag.baseClipPlaybackRate > 0 && Number.isFinite(drag.baseClipPlaybackRate)
                            ? drag.baseClipPlaybackRate
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
                    dispatch(
                        setClipPlaybackRate({ clipId: drag.clipId, clipPlaybackRate: nextRate }),
                    );
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
            if (!drag || drag.pointerId !== e.pointerId) {
                // 本手势已被另一指针覆盖（dragRef 指向别的手势）。本手势在
                // window 上的监听器与已开的 undo 组仍必须收尾，否则监听器
                // 永久悬挂、suppress_checkpoints 卡死（撤销栈冻结）。
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", end);
                window.removeEventListener("pointercancel", end);
                if (gainUndoGroupPromise) {
                    void finishGainUndoGroup().catch(() => undefined);
                }
                if (fadeUndoGroupPromise) {
                    void finishFadeUndoGroup().catch(() => undefined);
                }
                if (crossfadeUndoGroupPromise) {
                    void finishCrossfadeUndoGroup().catch(() => undefined);
                }
                dispatch(endInteraction());
                return;
            }
            editDragRef.current = null;
            // 收尾第一步注销失焦守卫（幂等防双触发；blur 与 pointerup 竞态安全）。
            unregisterAbort();
            // 先解绑再走后续逻辑：任何早退分支（如目标 clip 已被删除）
            // 都不能把 window 上的监听器泄漏成永久悬挂。
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
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
                // 目标 clip 已被删除：本次拖拽无法落盘。
                // ★ 必须关闭**全部**已开的 undo group（fade/crossfade 组在拖拽
                // 开始时就已打开）——否则后端 suppress_checkpoints 永久置位，
                // 此后工程内一切撤销点都被吞掉、撤销栈冻结到重启。
                if (gainUndoGroupPromise) {
                    void finishGainUndoGroup().catch(() => undefined);
                }
                if (fadeUndoGroupPromise) {
                    void finishFadeUndoGroup().catch(() => undefined);
                }
                if (crossfadeUndoGroupPromise) {
                    void finishCrossfadeUndoGroup().catch(() => undefined);
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
                            clipPlaybackRate: now.clipPlaybackRate ?? 1,
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
                            clipPlaybackRate: number;
                            snapOffsetSec: number;
                            fadeInSec: number;
                            fadeOutSec: number;
                        } => patch != null,
                    );

                if (stretchPatches.length > 0) {
                    reapplyRates = stretchPatches
                        .filter((p) => p.clipPlaybackRate !== 1)
                        .map((p) => ({ clipId: p.clipId, rate: p.clipPlaybackRate }));
                    // 无操作守卫（B7）：所有成员都回到拖拽前状态时不再落盘，
                    // 也不开 undo group —— 不产生"撤销后什么都没变"的死撤销步。
                    const noChange = stretchPatches.every((patch) => {
                        const base = drag.baseByClipId[patch.clipId];
                        return (
                            base &&
                            Math.abs(patch.startSec - base.startSec) < 1e-9 &&
                            Math.abs(patch.lengthSec - base.lengthSec) < 1e-9
                        );
                    });
                    if (noChange) {
                        persistPromise = Promise.resolve();
                    } else {
                        persistPromise = runInsideUndoGroup(async () => {
                            // 原子批量：单请求 = 单响应 = 无中间部分快照
                            // （多选松手不再"先闪回原状再弹回编辑后"）。
                            await dispatch(
                                setClipsStateBulkRemote({
                                    updates: buildBulkClipStateUpdates({
                                        clipIds: stretchPatches.map((p) => p.clipId),
                                        changesById: new Map(
                                            stretchPatches.map((patch) => [
                                                patch.clipId,
                                                {
                                                    startSec: patch.startSec,
                                                    lengthSec: patch.lengthSec,
                                                    clipPlaybackRate: patch.clipPlaybackRate,
                                                    snapOffsetSec: patch.snapOffsetSec,
                                                    fadeInSec: patch.fadeInSec,
                                                    fadeOutSec: patch.fadeOutSec,
                                                },
                                            ]),
                                        ),
                                    }),
                                    checkpoint: false,
                                }),
                            ).unwrap();

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
                        // 无操作守卫：全部回到拖拽前 → 不落盘、不开组。
                        const noChange = trimPatches.every((patch) => {
                            const base = drag.baseByClipId[patch.clipId];
                            return (
                                base &&
                                Math.abs(patch.startSec - base.startSec) < 1e-9 &&
                                Math.abs(patch.lengthSec - base.lengthSec) < 1e-9
                            );
                        });
                        if (noChange) {
                            persistPromise = Promise.resolve();
                        } else {
                            persistPromise = runInsideUndoGroup(async () => {
                                // 原子批量：多选裁短单请求 = 单响应 = 无中间快照。
                                const changesById = new Map(
                                    trimPatches.map((patch) => [
                                        patch.clipId,
                                        {
                                            startSec: patch.startSec,
                                            lengthSec: patch.lengthSec,
                                            ...(patch.reversed
                                                ? { sourceEndSec: patch.sourceEndSec }
                                                : { sourceStartSec: patch.sourceStartSec }),
                                        },
                                    ]),
                                );
                                await dispatch(
                                    setClipsStateBulkRemote({
                                        updates: buildBulkClipStateUpdates({
                                            clipIds: trimPatches.map((p) => p.clipId),
                                            changesById,
                                        }),
                                        checkpoint: false,
                                    }),
                                ).unwrap();
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
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceEndSec: singleClipNow.sourceEndSec }
                        : { sourceStartSec: singleClipNow.sourceStartSec };
                    // 无操作守卫：拖回起点 → 不落盘、不产生死撤销步。
                    const noChange =
                        Math.abs(singleClipNow.startSec - drag.basestartSec) < 1e-9 &&
                        Math.abs(singleClipNow.lengthSec - drag.baselengthSec) < 1e-9;
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = noChange
                            ? Promise.resolve()
                            : runWithOptionalAutoCrossfade(async () => {
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
                        persistPromise = noChange
                            ? Promise.resolve()
                            : dispatch(
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
                        // 无操作守卫：全部回到拖拽前 → 不落盘、不开组。
                        const noChange = trimPatches.every((patch) => {
                            const base = drag.baseByClipId[patch.clipId];
                            return base && Math.abs(patch.lengthSec - base.lengthSec) < 1e-9;
                        });
                        if (noChange) {
                            persistPromise = Promise.resolve();
                        } else {
                            persistPromise = runInsideUndoGroup(async () => {
                                // 原子批量：多选裁短单请求 = 单响应 = 无中间快照。
                                const changesById = new Map(
                                    trimPatches.map((patch) => [
                                        patch.clipId,
                                        {
                                            lengthSec: patch.lengthSec,
                                            ...(patch.reversed
                                                ? { sourceStartSec: patch.sourceStartSec }
                                                : { sourceEndSec: patch.sourceEndSec }),
                                        },
                                    ]),
                                );
                                await dispatch(
                                    setClipsStateBulkRemote({
                                        updates: buildBulkClipStateUpdates({
                                            clipIds: trimPatches.map((p) => p.clipId),
                                            changesById,
                                        }),
                                        checkpoint: false,
                                    }),
                                ).unwrap();
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
                    }
                } else {
                    const sourceRangePatch = singleClipNow.reversed
                        ? { sourceStartSec: singleClipNow.sourceStartSec }
                        : { sourceEndSec: singleClipNow.sourceEndSec };
                    // 无操作守卫：拖回起点 → 不落盘、不产生死撤销步。
                    const noChange = Math.abs(singleClipNow.lengthSec - drag.baselengthSec) < 1e-9;
                    if (shouldApplyAutoCrossfade) {
                        persistPromise = noChange
                            ? Promise.resolve()
                            : runWithOptionalAutoCrossfade(async () => {
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
                        persistPromise = noChange
                            ? Promise.resolve()
                            : dispatch(
                                  setClipStateRemote({
                                      clipId: drag.clipId,
                                      lengthSec: singleClipNow.lengthSec,
                                      ...sourceRangePatch,
                                  }),
                              ).unwrap();
                    }
                }
            } else if (drag.type === "stretch_left" && singleClipNow) {
                // 无操作守卫：拖回起点 → 不落盘、不产生死撤销步。
                const stretchNoChange =
                    Math.abs(singleClipNow.startSec - drag.basestartSec) < 1e-9 &&
                    Math.abs(singleClipNow.lengthSec - drag.baselengthSec) < 1e-9 &&
                    Math.abs((singleClipNow.clipPlaybackRate ?? 1) - drag.baseClipPlaybackRate) <
                        1e-6;
                if (shouldApplyAutoCrossfade) {
                    persistPromise = stretchNoChange
                        ? Promise.resolve()
                        : runWithOptionalAutoCrossfade(async () => {
                              await dispatch(
                                  setClipStateRemote({
                                      clipId: drag.clipId,
                                      startSec: singleClipNow.startSec,
                                      lengthSec: singleClipNow.lengthSec,
                                      clipPlaybackRate: singleClipNow.clipPlaybackRate ?? 1,
                                      snapOffsetSec: singleClipNow.snapOffsetSec,
                                      fadeInSec: singleClipNow.fadeInSec,
                                      fadeOutSec: singleClipNow.fadeOutSec,
                                      checkpoint: false,
                                  }),
                              ).unwrap();
                          });
                } else {
                    persistPromise = stretchNoChange
                        ? Promise.resolve()
                        : dispatch(
                              setClipStateRemote({
                                  clipId: drag.clipId,
                                  startSec: singleClipNow.startSec,
                                  lengthSec: singleClipNow.lengthSec,
                                  // 与上方 auto-crossfade 分支同口径：拉伸修改的是
                                  // Clip 级倍率；发平铺 playbackRate 会在后端被当作
                                  // 组合有效速率写坏 Take 自身速率。
                                  clipPlaybackRate: singleClipNow.clipPlaybackRate ?? 1,
                                  snapOffsetSec: singleClipNow.snapOffsetSec,
                                  fadeInSec: singleClipNow.fadeInSec,
                                  fadeOutSec: singleClipNow.fadeOutSec,
                              }),
                          ).unwrap();
                }
                if ((singleClipNow.clipPlaybackRate ?? 1) !== 1) {
                    reapplyRates = [
                        { clipId: drag.clipId, rate: singleClipNow.clipPlaybackRate ?? 1 },
                    ];
                }
            } else if (drag.type === "stretch_right" && singleClipNow) {
                // 无操作守卫：拖回起点 → 不落盘、不产生死撤销步。
                const stretchNoChange2 =
                    Math.abs(singleClipNow.lengthSec - drag.baselengthSec) < 1e-9 &&
                    Math.abs((singleClipNow.clipPlaybackRate ?? 1) - drag.baseClipPlaybackRate) <
                        1e-6;
                if (shouldApplyAutoCrossfade) {
                    persistPromise = stretchNoChange2
                        ? Promise.resolve()
                        : runWithOptionalAutoCrossfade(async () => {
                              await dispatch(
                                  setClipStateRemote({
                                      clipId: drag.clipId,
                                      lengthSec: singleClipNow.lengthSec,
                                      clipPlaybackRate: singleClipNow.clipPlaybackRate ?? 1,
                                      snapOffsetSec: singleClipNow.snapOffsetSec,
                                      fadeInSec: singleClipNow.fadeInSec,
                                      fadeOutSec: singleClipNow.fadeOutSec,
                                      checkpoint: false,
                                  }),
                              ).unwrap();
                          });
                } else {
                    persistPromise = stretchNoChange2
                        ? Promise.resolve()
                        : dispatch(
                              setClipStateRemote({
                                  clipId: drag.clipId,
                                  lengthSec: singleClipNow.lengthSec,
                                  // 与上方 auto-crossfade 分支同口径：发 Clip 级倍率
                                  // 而不是组合有效速率（理由见 stretch_left 分支）。
                                  clipPlaybackRate: singleClipNow.clipPlaybackRate ?? 1,
                                  snapOffsetSec: singleClipNow.snapOffsetSec,
                                  fadeInSec: singleClipNow.fadeInSec,
                                  fadeOutSec: singleClipNow.fadeOutSec,
                              }),
                          ).unwrap();
                }
                if ((singleClipNow.clipPlaybackRate ?? 1) !== 1) {
                    reapplyRates = [
                        { clipId: drag.clipId, rate: singleClipNow.clipPlaybackRate ?? 1 },
                    ];
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
                            // 交叉点曲率拖拽的最终落盘：随整份 patch 一并提交，
                            // 避免 bulk/远端回灌丢掉拖拽期方向。
                            fadeInShape: now.fadeInShape,
                            fadeInDir: now.fadeInDir,
                            fadeOutShape: now.fadeOutShape,
                            fadeOutDir: now.fadeOutDir,
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
                            fadeInShape: number;
                            fadeInDir: number;
                            fadeOutShape: number;
                            fadeOutDir: number;
                        } => patch != null,
                    );
                if (patches.length > 0) {
                    // 无操作守卫：两侧都回到拖拽前 → 不落盘、不开组。
                    const noChange = patches.every((patch) => {
                        const base = drag.baseByClipId[patch.clipId];
                        if (!base) return false;
                        return (
                            Math.abs(patch.startSec - base.startSec) < 1e-9 &&
                            Math.abs(patch.lengthSec - base.lengthSec) < 1e-9
                        );
                    });
                    const commitPatches = async () => {
                        // 原子批量：交叉点两个 clip 一次提交 = 单响应 = 无中间快照。
                        const changesById = new Map(
                            patches.map((patch) => [
                                patch.clipId,
                                {
                                    startSec: patch.startSec,
                                    lengthSec: patch.lengthSec,
                                    sourceStartSec: patch.sourceStartSec,
                                    sourceEndSec: patch.sourceEndSec,
                                    fadeInSec: patch.fadeInSec,
                                    fadeOutSec: patch.fadeOutSec,
                                    autoFadeInSec: patch.autoFadeInSec,
                                    autoFadeOutSec: patch.autoFadeOutSec,
                                    // 交叉点曲率拖拽的最终落盘：随整份 patch 一并提交，
                                    // 避免 bulk/远端回灌丢掉拖拽期方向。
                                    fadeInShape: patch.fadeInShape,
                                    fadeInDir: patch.fadeInDir,
                                    fadeOutShape: patch.fadeOutShape,
                                    fadeOutDir: patch.fadeOutDir,
                                },
                            ]),
                        );
                        await dispatch(
                            setClipsStateBulkRemote({
                                updates: buildBulkClipStateUpdates({
                                    clipIds: patches.map((p) => p.clipId),
                                    changesById,
                                }),
                                checkpoint: false,
                            }),
                        ).unwrap();
                    };
                    persistPromise = noChange
                        ? Promise.resolve()
                        : crossfadeUndoGroupPromise
                          ? crossfadeUndoGroupPromise.then(commitPatches)
                          : commitPatches();
                }
            } else if (drag.type === "fade_in" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        if (!nextClip) return [clipId, {}] as const;
                        // 手动长度是否被本手势改动：长度拖拽（含多选）会改
                        // fadeInSec；纯曲率拖拽只改 dir——此时**必须保留
                        // 自动交叉淡化的 fade**（不写手动长度、不清 auto）。
                        const baseManual = drag.baseManualFadeSecByClipId[clipId]?.in ?? 0;
                        const lengthEdited =
                            Math.abs((Number(nextClip.fadeInSec) || 0) - baseManual) > 1e-9;
                        return [
                            clipId,
                            {
                                ...(lengthEdited ? { fadeInSec: nextClip.fadeInSec ?? 0 } : {}),
                                // 曲率/形状拖拽的最终落盘：缺了它们，
                                // bulk fulfilled 的整份回灌会丢掉拖拽期修改。
                                fadeInDir: nextClip?.fadeInDir ?? 0,
                                fadeInShape: nextClip?.fadeInShape ?? 0,
                            },
                        ] as const;
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
                        // 只清除"本手势确实改了手动长度"的 clip 的自动交叉淡化
                        //（自动 fade → 手动 fade 转换）。纯曲率拖拽保持
                        // 自动交叉淡化的 fade 关系，只是曲率被更新。
                        const clears = drag.selectedClipIds
                            .filter(
                                (clipId) =>
                                    Math.abs(
                                        (sessionRef.current.clips.find((c) => c.id === clipId)
                                            ?.fadeInSec ?? 0) -
                                            (drag.baseManualFadeSecByClipId[clipId]?.in ?? 0),
                                    ) > 1e-9,
                            )
                            .map((clipId) => {
                                dispatch(setClipAutoFades({ clipId, autoFadeInSec: 0 }));
                                return webApi.setClipState({
                                    clipId,
                                    autoFadeInSec: 0,
                                    checkpoint: false,
                                });
                            });
                        return Promise.allSettled(clears).then(() => undefined);
                    });
            } else if (drag.type === "fade_out" && singleClipNow) {
                const changesById = new Map(
                    drag.selectedClipIds.map((clipId) => {
                        const nextClip = sessionRef.current.clips.find((c) => c.id === clipId);
                        if (!nextClip) return [clipId, {}] as const;
                        const baseManual = drag.baseManualFadeSecByClipId[clipId]?.out ?? 0;
                        const lengthEdited =
                            Math.abs((Number(nextClip.fadeOutSec) || 0) - baseManual) > 1e-9;
                        return [
                            clipId,
                            {
                                ...(lengthEdited ? { fadeOutSec: nextClip.fadeOutSec ?? 0 } : {}),
                                fadeOutDir: nextClip?.fadeOutDir ?? 0,
                                fadeOutShape: nextClip?.fadeOutShape ?? 0,
                            },
                        ] as const;
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
                        // 同 fade_in：只对"长度被本手势改动"的 clip 清除自动交叉
                        // 淡化；纯曲率拖拽保留自动交叉淡化的 fade。
                        const clears = drag.selectedClipIds
                            .filter(
                                (clipId) =>
                                    Math.abs(
                                        (sessionRef.current.clips.find((c) => c.id === clipId)
                                            ?.fadeOutSec ?? 0) -
                                            (drag.baseManualFadeSecByClipId[clipId]?.out ?? 0),
                                    ) > 1e-9,
                            )
                            .map((clipId) => {
                                dispatch(setClipAutoFades({ clipId, autoFadeOutSec: 0 }));
                                return webApi.setClipState({
                                    clipId,
                                    autoFadeOutSec: 0,
                                    checkpoint: false,
                                });
                            });
                        return Promise.allSettled(clears).then(() => undefined);
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
                        dispatch(setClipPlaybackRate({ clipId, clipPlaybackRate: rate }));
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
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { editDragRef, startEditDrag };
}
