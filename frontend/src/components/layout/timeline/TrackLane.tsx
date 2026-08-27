/**
 * TrackLane - 时间轴单轨道视图，负责布局轨道波形、剪辑项与拖拽中的 ghost 预览。
 */
import React from "react";

import { isPrimaryModifierDown } from "../../../utils/platform";
import type { ClipFormantMorph, ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import type { Keybinding } from "../../../features/keybindings/types";
import { useI18n } from "../../../i18n/I18nProvider";
import type { FadeLengthFormatContext } from "./fadeTooltipText";
import type { GhostDragInfo } from "./hooks/useClipDrag";
import type { ClipRenameClickCandidate } from "./clip/ClipHeader";
import { ClipItem } from "./ClipItem";
import { OverlapEditLayer } from "./OverlapEditLayer";
import { CLIP_HEADER_HEIGHT, CLIP_BODY_PADDING_Y } from "./constants";
import { buildTimelineHitTestIndex, hitTestTimeline } from "./runtime/timelineHitTest";
import { MidiPitchTrackCanvas } from "../../waveform/MidiPitchTrackCanvas";

function compareClipRenderOrder(a: ClipInfo, b: ClipInfo): number {
    const d = (a.startSec ?? 0) - (b.startSec ?? 0);
    if (Math.abs(d) > 1e-9) return d;
    return String(a.id).localeCompare(String(b.id));
}

function sameStringArray(a: string[] | undefined, b: string[] | undefined): boolean {
    if (a === b) return true;
    if (!a || !b) return !a && !b;
    if (a.length !== b.length) return false;
    for (let index = 0; index < a.length; index += 1) {
        if (a[index] !== b[index]) return false;
    }
    return true;
}

/**
 * 计算每个 clip 在“自身左侧前导区”的重叠时长（秒）。
 *
 * 该前导重叠区对应“该 clip 在当前渲染顺序中位于上层”的区域，
 * 用于在重叠区做等权可视化混合，避免后绘制 clip 完全盖住前一个 clip。
 */
export function computeLeadingOverlapSecByClipId(clips: ClipInfo[]): Record<string, number> {
    const sorted = [...clips].sort(compareClipRenderOrder);
    const leadingOverlapSecByClipId: Record<string, number> = {};

    for (let i = 0; i < sorted.length; i += 1) {
        const clip = sorted[i];
        const clipStart = clip.startSec;
        const clipEnd = clip.startSec + clip.lengthSec;
        let leadingOverlapEnd = clipStart;

        for (let j = 0; j < i; j += 1) {
            const other = sorted[j];
            const otherEnd = other.startSec + other.lengthSec;
            const overlapEnd = Math.min(clipEnd, otherEnd);
            if (overlapEnd <= clipStart + 1e-9) continue;
            if (overlapEnd > leadingOverlapEnd) {
                leadingOverlapEnd = overlapEnd;
            }
        }

        leadingOverlapSecByClipId[clip.id] = Math.max(0, leadingOverlapEnd - clipStart);
    }

    return leadingOverlapSecByClipId;
}

type TrackLaneProps = {
    track: TrackInfo;
    allTracks: TrackInfo[];
    trackClips: ClipInfo[];

    rowHeight: number;
    pxPerSec: number;
    bpm: number;
    viewportWidthPx: number;
    viewportStartSec: number;
    viewportEndSec: number;
    overlayClipIds?: string[];

    altPressed: boolean;

    selectedClipId: string | null;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;

    /** 轨道主题色，用于 Clip 背景色和选中边框�?*/
    trackColor?: string;

    ensureSelected: (clipId: string) => void;
    selectClipRemote: (clipId: string) => void;
    openContextMenu: (clipId: string, clientX: number, clientY: number) => void;

    seekFromClientX: (clientX: number, commit: boolean) => void;
    startClipDrag: (
        e: React.PointerEvent<HTMLDivElement>,
        clipId: string,
        clipstartSec: number,
        altPressedHint?: boolean,
    ) => void;
    startEditDrag: (
        e: React.PointerEvent,
        clipId: string,
        type:
            | "trim_left"
            | "trim_right"
            | "stretch_left"
            | "stretch_right"
            | "fade_in"
            | "fade_out"
            | "gain"
            | "crossfade_edges",
        /** 延迟起手的类型化私有通道（与 useEditDrag.EditDragChannelOpts 一致）。 */
        channel?:
            | {
                  dragStartClientX?: number;
                  crossfadePartnerClipId?: string | null;
                  fadePointerEnv?: {
                      envTopClientY: number;
                      bodyHeightPx: number;
                  } | null;
              }
            | undefined,
    ) => void;
    /** SnapOffset 三角手柄拖拽（左下角；拖动调整吸附偏移）。 */
    startSnapOffsetDrag?: (e: React.PointerEvent, clipId: string) => void;
    toggleClipMuted: (clipId: string, nextMuted: boolean) => void;
    /** Ctrl+左键选择切换（会更新主选中 clip） */
    onCtrlToggleSelect: (clipId: string) => void;
    /** Ctrl+左键多选切换 */
    toggleMultiSelect: (clipId: string) => void;
    /** Shift+点击范围选择；targetClientX 用于基于鼠标位置构建矩形 */
    onShiftRangeSelect: (
        clipId: string,
        anchorClipIdOverride?: string | null,
        targetClientX?: number,
    ) => void;
    /** Shift 范围选择锚点（点击前快照） */
    rangeSelectAnchorClipId: string | null;
    /** 记录最近的点击 clientX，用于 Shift 范围选择的锚点位置 */
    recordLastClickPosition?: (clientX: number) => void;

    clearContextMenu: () => void;

    /** 当前正在重命名的 clipId（来自右键菜单或双击名称） */
    renamingClipId?: string | null;
    onRenameStart?: (clipId: string) => void;
    onRenameCommit?: (clipId: string, newName: string) => void;
    onRenameDone?: () => void;
    onRenameClickCandidate?: (candidate: ClipRenameClickCandidate | null) => void;
    onGainCommit?: (clipId: string, db: number) => void;
    onFormantMorphCommit?: (clipId: string, value: ClipFormantMorph, checkpoint: boolean) => void;
    activeGroupIds?: Set<string>;
    disabledGroupIds?: string[];
    onToggleGroupDisabled?: (groupId: string) => void;

    /** 复制拖动时的 ghost 预览信息 */
    ghostDrag?: GhostDragInfo | null;
    /** 当前拖拽处于纯竖直换轨锁定时，高亮的目标轨道 */
    verticalTrackLockTrackId?: string | null;
    /** 所有 clip 数据（用于跨轨道 ghost 查找） */
    allClips?: ClipInfo[];
    /** 在空间足够时显示全部 Take 波形。 */
    showAllTakes?: boolean;
    /** 点击 inactive take 波形时切换 active take。 */
    onActivateTake?: (clipId: string, takeId: string) => void;
    /** 形状循环键绑定（modifier.fadeShapeCycleClick），透传给 ClipItem。 */
    fadeShapeCycleKb?: Keybinding | null;
    /** 淡化长度 ToolTips 的相对时长时间上下文。 */
    fadeLengthFormatCtx: FadeLengthFormatContext;
    /** 修饰键下左键点击包络线 → 循环切换该侧曲线类型。 */
    onFadeShapeCycleClick?: (clipId: string, side: "in" | "out") => void;
    /** 抓手上的循环点击：同时切换交叉点两侧。 */
    onCrossfadeCycleClick?: (sides: Array<{ clipId: string; isOut: boolean }>) => void;
    /** 单条包络线上的循环点击（Ctrl+点击，非抓手）。 */
    onFadeShapeSingleCycle?: (side: { clipId: string; isOut: boolean }) => void;
};

export const TrackLane = React.memo(
    function TrackLane(props: TrackLaneProps) {
        const {
            track,
            allTracks,
            trackClips,
            rowHeight,
            pxPerSec,
            viewportWidthPx,
            viewportStartSec,
            viewportEndSec,
            overlayClipIds = [],
            altPressed,
            fadeShapeCycleKb = null,
            fadeLengthFormatCtx,
            onFadeShapeCycleClick,
            onCrossfadeCycleClick,
            selectedClipId,
            multiSelectedClipIds,
            multiSelectedSet,
            trackColor,
            ensureSelected,
            selectClipRemote,
            openContextMenu,
            seekFromClientX,
            startClipDrag,
            startEditDrag,
            startSnapOffsetDrag,
            toggleClipMuted,
            onCtrlToggleSelect,
            toggleMultiSelect,
            onShiftRangeSelect,
            rangeSelectAnchorClipId,
            recordLastClickPosition,
            clearContextMenu,
            renamingClipId,
            onRenameStart,
            onRenameCommit,
            onRenameDone,
            onRenameClickCandidate,
            onGainCommit,
            onFormantMorphCommit,
            activeGroupIds,
            disabledGroupIds,
            onToggleGroupDisabled,
            ghostDrag,
            verticalTrackLockTrackId,
            allClips,
            showAllTakes = true,
            onActivateTake,
        } = props;

// 淡变信息浮标与子层 i18n 文案。
        const { t } = useI18n();
        // 波形区域高度计算（与 ClipItem 一致）
        const waveformHeight = Math.max(1, rowHeight - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT);
        const [hoveredClipId, setHoveredClipId] = React.useState<string | null>(null);
        const showVerticalTrackLock = verticalTrackLockTrackId === track.id;

        // 计算当前轨道上需要渲染的 ghost clip 列表
        const ghostClips = React.useMemo(() => {
            if (!ghostDrag) return [];
            const result: { clip: ClipInfo; ghostStartSec: number }[] = [];
            const orderedTrackIds = allTracks.map((t) => t.id);
            const trackIndexById = Object.fromEntries(
                orderedTrackIds.map((id, idx) => [id, idx]),
            ) as Record<string, number>;
            const clipById = new Map((allClips ?? []).map((clip) => [clip.id, clip] as const));
            for (const clipId of ghostDrag.clipIds) {
                const initial = ghostDrag.initialById[clipId];
                if (!initial) continue;
                // 判断 ghost 是否应出现在当前轨道上
                let ghostTrackId = initial.trackId;
                if (ghostDrag.allowTrackMove) {
                    if (ghostDrag.targetTrackId == null) {
                        continue;
                    } else {
                        const sourceIndex = trackIndexById[initial.trackId];
                        const targetIndex = sourceIndex + ghostDrag.targetTrackOffset;
                        ghostTrackId = orderedTrackIds[targetIndex] ?? initial.trackId;
                    }
                }
                if (ghostTrackId !== track.id) continue;
                const clip = clipById.get(clipId);
                if (!clip) continue;
                result.push({
                    clip,
                    ghostStartSec: Math.max(0, initial.startSec + ghostDrag.deltaSec),
                });
            }
            return result;
        }, [ghostDrag, track.id, trackClips, allClips, allTracks]);

        const leadingOverlapSecByClipId = React.useMemo(
            () => computeLeadingOverlapSecByClipId(trackClips),
            [trackClips],
        );
        const laneHitTestIndex = React.useMemo(
            () =>
                buildTimelineHitTestIndex({
                    rowHeight,
                    pxPerSec,
                    visibleTracks: [{ id: track.id, topPx: 0 }],
                    visibleClips: trackClips.map((clip) => ({
                        id: clip.id,
                        trackId: clip.trackId,
                        startSec: clip.startSec,
                        lengthSec: clip.lengthSec,
                        snapOffsetSec: clip.snapOffsetSec,
                    })),
                }),
            [pxPerSec, rowHeight, track.id, trackClips],
        );
        const overlayClipIdSet = React.useMemo(() => {
            const next = new Set(overlayClipIds);
            if (hoveredClipId) {
                next.add(hoveredClipId);
            }
            return next;
        }, [hoveredClipId, overlayClipIds]);
        const overlayTrackClips = React.useMemo(
            () => trackClips.filter((clip) => overlayClipIdSet.has(clip.id)),
            [overlayClipIdSet, trackClips],
        );
        const hitTestLane = React.useCallback(
            (clientX: number, clientY: number, currentTarget: HTMLDivElement) => {
                const bounds = currentTarget.getBoundingClientRect();
                return hitTestTimeline(
                    {
                        screenX: clientX - bounds.left,
                        screenY: clientY - bounds.top,
                        scrollLeftPx: 0,
                        scrollTopPx: 0,
                    },
                    laneHitTestIndex,
                );
            },
            [laneHitTestIndex],
        );
        const isClipItemTarget = React.useCallback((target: EventTarget | null) => {
            return (target as HTMLElement | null)?.closest?.("[data-hs-clip-item='1']") != null;
        }, []);
        const isOverlapLayerTarget = React.useCallback((target: EventTarget | null) => {
            return (target as HTMLElement | null)?.closest?.("[data-hs-overlap-layer='1']") != null;
        }, []);
        const primeSelection = React.useCallback(
            (clipId: string, shouldPrimeSelection: boolean, clientX?: number) => {
                if (!shouldPrimeSelection) {
                    return;
                }
                if (!multiSelectedSet.has(clipId) || multiSelectedClipIds.length > 1) {
                    ensureSelected(clipId);
                }
                selectClipRemote(clipId);
                if (clientX != null) {
                    recordLastClickPosition?.(clientX);
                }
            },
            [
                ensureSelected,
                multiSelectedClipIds.length,
                multiSelectedSet,
                selectClipRemote,
                recordLastClickPosition,
            ],
        );
        const beginBodyInteraction = React.useCallback(
            (event: React.PointerEvent<HTMLDivElement>, clip: ClipInfo) => {
                // altPressed tracks the stretch modifier (configurable).
                // For click-selection bypass, only check physical Alt key to avoid
                // breaking Ctrl/Shift selection when those keys are stretch modifiers.
                const altKeyDown = Boolean(
                    event.altKey || event.nativeEvent.getModifierState?.("Alt"),
                );
                const primaryModifierDown = isPrimaryModifierDown(event);
                const doShiftRangeSelect = event.shiftKey && !altKeyDown && !primaryModifierDown;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = primaryModifierDown && !event.shiftKey && !altKeyDown;
                const allowSeek = !altKeyDown && !primaryModifierDown && !event.shiftKey;
                const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;
                const clipIsSelected =
                    multiSelectedClipIds.length > 0
                        ? multiSelectedSet.has(clip.id)
                        : selectedClipId === clip.id;
                const primedSelection = shouldPrimeSelection && !clipIsSelected;
                const startX = event.clientX;
                const startY = event.clientY;
                let moved = false;

                event.preventDefault();
                event.stopPropagation();
                clearContextMenu();

                if (primedSelection) {
                    primeSelection(clip.id, true, event.clientX);
                }

                const onMove = (ev: PointerEvent) => {
                    if (ev.pointerId !== event.pointerId) return;
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    if (dx * dx + dy * dy >= 9) moved = true;
                };

                const onUp = (ev: PointerEvent) => {
                    if (ev.pointerId !== event.pointerId) return;
                    window.removeEventListener("pointermove", onMove, true);
                    window.removeEventListener("pointerup", onUp, true);
                    window.removeEventListener("pointercancel", onUp, true);
                    if (!moved) {
                        if (doShiftRangeSelect) {
                            onShiftRangeSelect(clip.id, shiftRangeAnchorClipId, startX);
                        } else if (shouldPrimeSelection && !primedSelection) {
                            primeSelection(clip.id, true, event.clientX);
                        }
                        if (allowSeek) {
                            seekFromClientX(ev.clientX, true);
                        }
                    }
                };

                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onUp, true);
                window.addEventListener("pointercancel", onUp, true);

                startClipDrag(event, clip.id, clip.startSec, false);
            },
            [
                altPressed,
                clearContextMenu,
                selectedClipId,
                multiSelectedClipIds,
                multiSelectedSet,
                onShiftRangeSelect,
                primeSelection,
                rangeSelectAnchorClipId,
                seekFromClientX,
                startClipDrag,
            ],
        );
        /**
         * 轨道空白区（无任何 clip 编辑目标）的播放头手势：交互等级最低的
         * 播放头拖拽入口。按下立即提交一次 seek，拖动中视觉跟随（commit=false），
         * 松开提交。与标尺行为一致；Clip / 淡变 / 边缘等更优目标会先行命中，
         * 因此本手势永远不会与它们抢交互。
         */
        const beginBackgroundSeekInteraction = React.useCallback(
            (event: React.PointerEvent<HTMLDivElement>) => {
                if (event.button !== 0) return;
                event.preventDefault();
                event.stopPropagation();
                const pointerId = event.pointerId;
                seekFromClientX(event.clientX, true);

                const onMove = (ev: PointerEvent) => {
                    if (ev.pointerId !== pointerId) return;
                    seekFromClientX(ev.clientX, false);
                };
                const onEnd = (ev: PointerEvent) => {
                    if (ev.pointerId !== pointerId) return;
                    window.removeEventListener("pointermove", onMove, true);
                    window.removeEventListener("pointerup", onEnd, true);
                    window.removeEventListener("pointercancel", onEnd, true);
                    seekFromClientX(ev.clientX, true);
                };
                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onEnd, true);
                window.addEventListener("pointercancel", onEnd, true);
            },
            [seekFromClientX],
        );

        const beginEdgeInteraction = React.useCallback(
            (
                event: React.PointerEvent<HTMLDivElement>,
                clipId: string,
                edge: "trim_left" | "trim_right",
            ) => {
                if (event.button !== 0) return;

                // altPressed tracks the stretch modifier (configurable) — use it
                // for edit-mode selection (stretch vs trim). For click-selection
                // bypass, only check the physical Alt key.
                const stretchActive = altPressed;
                const altKeyDown = Boolean(
                    event.altKey || event.nativeEvent.getModifierState?.("Alt"),
                );
                const primaryModifierDown = isPrimaryModifierDown(event);
                const doShiftRangeSelect = event.shiftKey && !altKeyDown && !primaryModifierDown;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = primaryModifierDown && !event.shiftKey && !altKeyDown;
                const shouldPrimeSelection = !doCtrlToggleOnly && !doShiftRangeSelect;
                const clipIsSelected =
                    multiSelectedClipIds.length > 0
                        ? multiSelectedSet.has(clipId)
                        : selectedClipId === clipId;
                const primedSelection = shouldPrimeSelection && !clipIsSelected;
                const mode =
                    edge === "trim_left"
                        ? stretchActive
                            ? "stretch_left"
                            : "trim_left"
                        : stretchActive
                          ? "stretch_right"
                          : "trim_right";
                const startX = event.clientX;
                const startY = event.clientY;
                const pointerId = event.pointerId;
                const laneEl = event.currentTarget as HTMLElement;
                let dragStarted = false;

                event.preventDefault();
                event.stopPropagation();
                clearContextMenu();

                if (primedSelection) {
                    primeSelection(clipId, true, event.clientX);
                }

                const onMove = (ev: PointerEvent) => {
                    if (ev.pointerId !== pointerId || dragStarted) return;
                    const dx = ev.clientX - startX;
                    const dy = ev.clientY - startY;
                    if (dx * dx + dy * dy < 9) return;
                    dragStarted = true;
                    startEditDrag(event, clipId, mode);
                };

                const onEnd = (ev: PointerEvent) => {
                    if (ev.pointerId !== pointerId) return;
                    window.removeEventListener("pointermove", onMove, true);
                    window.removeEventListener("pointerup", onEnd, true);
                    window.removeEventListener("pointercancel", onEnd, true);
                    if (!dragStarted) {
                        if (doCtrlToggleOnly) {
                            onCtrlToggleSelect(clipId);
                            return;
                        }
                        if (doShiftRangeSelect) {
                            onShiftRangeSelect(clipId, shiftRangeAnchorClipId, startX);
                            return;
                        }
                        if (shouldPrimeSelection && !primedSelection) {
                            primeSelection(clipId, true, event.clientX);
                        }
                        // 单击 Clip 边缘（未拖动）→ 播放头跳到该边缘的准确位置。
                        // lane 容器左缘即时间轴 0 秒的客户坐标；clip 左/右缘在
                        // lane 内 = startSec（左缘）或 startSec+lengthSec（右缘）。
                        const edgeClip = trackClips.find((entry) => entry.id === clipId);
                        const laneRect = laneEl.getBoundingClientRect();
                        const edgeSec =
                            edge === "trim_left"
                                ? edgeClip?.startSec ?? 0
                                : (edgeClip?.startSec ?? 0) + (edgeClip?.lengthSec ?? 0);
                        seekFromClientX(laneRect.left + edgeSec * pxPerSec, true);
                    }
                };

                window.addEventListener("pointermove", onMove, true);
                window.addEventListener("pointerup", onEnd, true);
                window.addEventListener("pointercancel", onEnd, true);
            },
            [
                altPressed,
                clearContextMenu,
                selectedClipId,
                multiSelectedClipIds,
                multiSelectedSet,
                onCtrlToggleSelect,
                onShiftRangeSelect,
                primeSelection,
                rangeSelectAnchorClipId,
                seekFromClientX,
                startEditDrag,
            ],
        );
        // SnapOffset（吸附偏移）角部手势：左下角 ◣ 处按下即进入偏移拖拽
        // （内部走完整吸附引擎与竖线高亮）。选择预备语义与边缘交互一致。
        const beginSnapOffsetInteraction = React.useCallback(
            (event: React.PointerEvent<HTMLDivElement>, clip: ClipInfo) => {
                if (event.button !== 0) return;
                const altKeyDown = Boolean(
                    event.altKey || event.nativeEvent.getModifierState?.("Alt"),
                );
                const primaryModifierDown = isPrimaryModifierDown(event);
                const doShiftRangeSelect = event.shiftKey && !altKeyDown && !primaryModifierDown;
                const doCtrlToggleOnly = primaryModifierDown && !event.shiftKey && !altKeyDown;

                event.preventDefault();
                event.stopPropagation();
                clearContextMenu();

                if (!doShiftRangeSelect && !doCtrlToggleOnly) {
                    primeSelection(clip.id, true, event.clientX);
                }
                startSnapOffsetDrag?.(event, clip.id);
            },
            [clearContextMenu, primeSelection, startSnapOffsetDrag],
        );

        return (
            <div
                key={track.id}
                className="border-b border-qt-border relative"
                style={{
                    height: rowHeight,
                    backgroundColor: showVerticalTrackLock
                        ? "rgba(112, 192, 255, 0.08)"
                        : undefined,
                    boxShadow: showVerticalTrackLock
                        ? "inset 0 0 0 1px rgba(112, 192, 255, 0.72), inset 0 0 0 9999px rgba(112, 192, 255, 0.04)"
                        : undefined,
                }}
                onPointerMoveCapture={(event) => {
                    const hit = hitTestLane(event.clientX, event.clientY, event.currentTarget);
                    setHoveredClipId((previous) =>
                        previous === hit.clipId ? previous : hit.clipId,
                    );
                }}
                onContextMenuCapture={(event) => {
                    if (isClipItemTarget(event.target) || isOverlapLayerTarget(event.target)) {
                        return;
                    }
                    const hit = hitTestLane(event.clientX, event.clientY, event.currentTarget);
                    if (!hit.clipId) {
                        return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    if (multiSelectedClipIds.length <= 1) {
                        ensureSelected(hit.clipId);
                        selectClipRemote(hit.clipId);
                    }
                    openContextMenu(hit.clipId, event.clientX, event.clientY);
                }}
                onPointerDownCapture={(event) => {
                    if (isClipItemTarget(event.target) || isOverlapLayerTarget(event.target)) {
                        return;
                    }
                    if (event.button !== 0) {
                        return;
                    }
                    const hit = hitTestLane(event.clientX, event.clientY, event.currentTarget);
                    if (!hit.clipId) {
                        // 空白区 = 最低优先级播放头手势（播放头自身 pointer-events-none，
                        // 拖拽/单击落点都在这里响应）。
                        beginBackgroundSeekInteraction(event);
                        return;
                    }
                    const clip = trackClips.find((candidate) => candidate.id === hit.clipId);
                    if (!clip) {
                        return;
                    }
                    if (hit.zone === "snap_offset") {
                        beginSnapOffsetInteraction(event, clip);
                        return;
                    }
                    if (hit.zone === "trim_left" || hit.zone === "trim_right") {
                        beginEdgeInteraction(event, clip.id, hit.zone);
                        return;
                    }
                    beginBodyInteraction(event, clip);
                }}
                onPointerLeave={() => {
                    setHoveredClipId(null);
                }}
            >
                {showVerticalTrackLock ? (
                    <div className="absolute right-2 top-1 pointer-events-none z-20">
                        <div
                            className="rounded px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.08em]"
                            style={{
                                color: "rgba(235, 246, 255, 0.96)",
                                backgroundColor: "rgba(41, 117, 173, 0.88)",
                                boxShadow: "0 0 0 1px rgba(164, 217, 255, 0.38)",
                            }}
                        >
                            Vertical Lock
                        </div>
                    </div>
                ) : null}
                {/* MIDI 音高预览 Canvas：绘制 MIDI clip 的音高线 */}
                <MidiPitchTrackCanvas
                    clips={trackClips}
                    trackHeight={rowHeight}
                    waveformTop={CLIP_HEADER_HEIGHT}
                    waveformHeight={waveformHeight}
                    pxPerSec={pxPerSec}
                    viewportWidthPx={viewportWidthPx}
                    viewportStartSec={viewportStartSec}
                    viewportEndSec={viewportEndSec}
                    strokeWidth={1.5}
                />
                {overlayTrackClips.map((clip) => {
                    const selected =
                        multiSelectedClipIds.length > 0
                            ? multiSelectedSet.has(clip.id)
                            : selectedClipId === clip.id;

                    return (
                        <ClipItem
                            key={clip.id}
                            clip={clip}
                            rowHeight={rowHeight}
                            pxPerSec={pxPerSec}
                            altPressed={altPressed}
                            selected={selected}
                            leadingOverlapSec={leadingOverlapSecByClipId[clip.id] ?? 0}
                            isInMultiSelectedSet={multiSelectedSet.has(clip.id)}
                            multiSelectedCount={multiSelectedClipIds.length}
                            trackColor={trackColor}
                            ensureSelected={ensureSelected}
                            selectClipRemote={selectClipRemote}
                            openContextMenu={openContextMenu}
                            seekFromClientX={seekFromClientX}
                            startClipDrag={startClipDrag}
                            startEditDrag={startEditDrag}
                            startSnapOffsetDrag={startSnapOffsetDrag}
                            toggleClipMuted={toggleClipMuted}
                            onCtrlToggleSelect={onCtrlToggleSelect}
                            toggleMultiSelect={toggleMultiSelect}
                            onShiftRangeSelect={onShiftRangeSelect}
                            rangeSelectAnchorClipId={rangeSelectAnchorClipId}
                            recordLastClickPosition={recordLastClickPosition}
                            clearContextMenu={clearContextMenu}
                            triggerRename={renamingClipId === clip.id}
                            onRenameStart={onRenameStart}
                            onRenameCommit={onRenameCommit}
                            onRenameDone={onRenameDone}
                            onRenameClickCandidate={onRenameClickCandidate}
                            onGainCommit={onGainCommit}
                            onFormantMorphCommit={onFormantMorphCommit}
                            activeGroupIds={activeGroupIds}
                            disabledGroupIds={disabledGroupIds}
                            onToggleGroupDisabled={onToggleGroupDisabled}
                            hovered={hoveredClipId === clip.id}
                            showAllTakes={showAllTakes}
                            onActivateTake={onActivateTake}
                            fadeShapeCycleKb={fadeShapeCycleKb}
                            onFadeShapeCycleClick={onFadeShapeCycleClick}
                            fadeLengthFormatCtx={fadeLengthFormatCtx}
                        />
                    );
                })}
                {/* 交叉（重叠）区确定性编辑层：双方边缘/淡入淡出都可在合适位置编辑 */}
                <OverlapEditLayer
                    trackClips={trackClips}
                    pxPerSec={pxPerSec}
                    rowHeight={rowHeight}
                    altPressed={altPressed}
                    selectedClipId={selectedClipId}
                    multiSelectedClipIds={multiSelectedClipIds}
                    multiSelectedSet={multiSelectedSet}
                    ensureSelected={ensureSelected}
                    selectClipRemote={selectClipRemote}
                    recordLastClickPosition={recordLastClickPosition}
startEditDrag={startEditDrag}
                    startSnapOffsetDrag={startSnapOffsetDrag}
                    seekFromClientX={seekFromClientX}
                    fadeLengthFormatCtx={fadeLengthFormatCtx}
                    shapeCycleKb={fadeShapeCycleKb}
                    onCrossfadeCycleClick={onCrossfadeCycleClick}
                    onFadeShapeSingleCycle={({ clipId, isOut }) =>
                        onFadeShapeCycleClick?.(clipId, isOut ? "out" : "in")
                    }
                    t={(key) => t(key as Parameters<typeof t>[0])}
                />
                {/* Ghost clip 预览：复制拖动时显示半透明副本 */}
                {ghostClips.map(({ clip, ghostStartSec }) => {
                    const ghostLeft = Math.max(0, ghostStartSec * pxPerSec);
                    const ghostWidth = Math.max(1, clip.lengthSec * pxPerSec);
                    return (
                        <div
                            key={`ghost-${clip.id}`}
                            className="absolute pointer-events-none opacity-50"
                            style={{
                                left: ghostLeft,
                                width: ghostWidth,
                                top: 0,
                                height: rowHeight - CLIP_BODY_PADDING_Y,
                            }}
                        >
                            {/* Ghost header 条 */}
                            <div
                                className="absolute left-0 right-0 top-0"
                                style={{
                                    height: CLIP_HEADER_HEIGHT,
                                    backgroundColor: trackColor
                                        ? `color-mix(in oklab, var(--qt-clip-bg) 56%, ${trackColor} 44%)`
                                        : "var(--qt-clip-bg)",
                                }}
                            />
                            {/* Ghost body 区域 */}
                            <div
                                className="absolute left-0 right-0 bottom-0 border border-dashed border-white/60"
                                style={{
                                    top: CLIP_HEADER_HEIGHT,
                                    backgroundColor: trackColor
                                        ? `color-mix(in oklab, var(--qt-clip-bg) 60%, ${trackColor} 40%)`
                                        : "var(--qt-clip-bg)",
                                }}
                            />
                        </div>
                    );
                })}
            </div>
        );
    },
    (prev, next) => {
        return (
            prev.track === next.track &&
            prev.allTracks === next.allTracks &&
            prev.trackClips === next.trackClips &&
            prev.rowHeight === next.rowHeight &&
            prev.pxPerSec === next.pxPerSec &&
            prev.bpm === next.bpm &&
            prev.viewportWidthPx === next.viewportWidthPx &&
            prev.altPressed === next.altPressed &&
            prev.selectedClipId === next.selectedClipId &&
            prev.multiSelectedClipIds === next.multiSelectedClipIds &&
            prev.multiSelectedSet === next.multiSelectedSet &&
            prev.trackColor === next.trackColor &&
            prev.ensureSelected === next.ensureSelected &&
            prev.selectClipRemote === next.selectClipRemote &&
            prev.openContextMenu === next.openContextMenu &&
            prev.seekFromClientX === next.seekFromClientX &&
            prev.startClipDrag === next.startClipDrag &&
            prev.startEditDrag === next.startEditDrag &&
            prev.startSnapOffsetDrag === next.startSnapOffsetDrag &&
            prev.toggleClipMuted === next.toggleClipMuted &&
            prev.onCtrlToggleSelect === next.onCtrlToggleSelect &&
            prev.toggleMultiSelect === next.toggleMultiSelect &&
            prev.onShiftRangeSelect === next.onShiftRangeSelect &&
            prev.rangeSelectAnchorClipId === next.rangeSelectAnchorClipId &&
            prev.recordLastClickPosition === next.recordLastClickPosition &&
            prev.clearContextMenu === next.clearContextMenu &&
            prev.renamingClipId === next.renamingClipId &&
            prev.onRenameStart === next.onRenameStart &&
            prev.onRenameCommit === next.onRenameCommit &&
            prev.onRenameDone === next.onRenameDone &&
            prev.onRenameClickCandidate === next.onRenameClickCandidate &&
            prev.onGainCommit === next.onGainCommit &&
            prev.onFormantMorphCommit === next.onFormantMorphCommit &&
            prev.ghostDrag === next.ghostDrag &&
            prev.verticalTrackLockTrackId === next.verticalTrackLockTrackId &&
            prev.allClips === next.allClips &&
            prev.showAllTakes === next.showAllTakes &&
            prev.onActivateTake === next.onActivateTake &&
            sameStringArray(prev.overlayClipIds, next.overlayClipIds) &&
            prev.activeGroupIds === next.activeGroupIds &&
            prev.disabledGroupIds === next.disabledGroupIds &&
            prev.onToggleGroupDisabled === next.onToggleGroupDisabled
            // viewportStartSec / viewportEndSec are consumed by WaveformTrackCanvas via the viewport bus
            // after mount, so pure horizontal scroll should not force a TrackLane rerender.
        );
    },
);
