/**
 * TrackLane - 时间轴单轨道视图，负责布局轨道波形、剪辑项与拖拽中的 ghost 预览。
 */
import React from "react";

import type { ClipFormantMorph, ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import type { GhostDragInfo } from "./hooks/useClipDrag";
import type { ClipRenameClickCandidate } from "./clip/ClipHeader";
import { ClipItem } from "./ClipItem";
import { OverlapEditLayer } from "./OverlapEditLayer";
import { CLIP_HEADER_HEIGHT, CLIP_BODY_PADDING_Y } from "./constants";
import { buildTimelineHitTestIndex, hitTestTimeline } from "./runtime/timelineHitTest";
import { WaveformTrackCanvas } from "../../waveform/WaveformTrackCanvas";
import { MidiPitchTrackCanvas } from "../../waveform/MidiPitchTrackCanvas";
import { useAppTheme } from "../../../theme/AppThemeProvider";
import { getWaveformColors } from "../../../theme/waveformColors";

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
            | "gain",
    ) => void;
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

    /** Ctrl+拖动复制时的 ghost 预览信息 */
    ghostDrag?: GhostDragInfo | null;
    /** 当前拖拽处于纯竖直换轨锁定时，高亮的目标轨道 */
    verticalTrackLockTrackId?: string | null;
    /** 所有 clip 数据（用于跨轨道 ghost 查找） */
    allClips?: ClipInfo[];
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
        } = props;

        // 获取波形颜色配置
        const { mode: themeMode } = useAppTheme();
        const waveformColors = React.useMemo(
            () => getWaveformColors(themeMode, "timeline"),
            [themeMode],
        );

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
            return (
                (target as HTMLElement | null)?.closest?.("[data-hs-overlap-layer='1']") != null
            );
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
                const ctrlOrMeta = event.ctrlKey || event.metaKey;
                const doShiftRangeSelect = event.shiftKey && !altKeyDown && !ctrlOrMeta;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = ctrlOrMeta && !event.shiftKey && !altKeyDown;
                const allowSeek = !altKeyDown && !ctrlOrMeta && !event.shiftKey;
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
                const ctrlOrMeta = event.ctrlKey || event.metaKey;
                const doShiftRangeSelect = event.shiftKey && !altKeyDown && !ctrlOrMeta;
                const shiftRangeAnchorClipId = doShiftRangeSelect ? rangeSelectAnchorClipId : null;
                const doCtrlToggleOnly = ctrlOrMeta && !event.shiftKey && !altKeyDown;
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
                        seekFromClientX(ev.clientX, true);
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
                    if (
                        isClipItemTarget(event.target) ||
                        isOverlapLayerTarget(event.target)
                    ) {
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
                    if (
                        isClipItemTarget(event.target) ||
                        isOverlapLayerTarget(event.target)
                    ) {
                        return;
                    }
                    if (event.button !== 0) {
                        return;
                    }
                    const hit = hitTestLane(event.clientX, event.clientY, event.currentTarget);
                    if (!hit.clipId) {
                        return;
                    }
                    const clip = trackClips.find((candidate) => candidate.id === hit.clipId);
                    if (!clip) {
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
                {/* 轨道级波形 Canvas：一个 Canvas 绘制该轨道所有可见 clip 的波形 */}
                <WaveformTrackCanvas
                    clips={trackClips}
                    leadingOverlapSecByClipId={leadingOverlapSecByClipId}
                    trackHeight={rowHeight}
                    waveformTop={CLIP_HEADER_HEIGHT}
                    waveformHeight={waveformHeight}
                    pxPerSec={pxPerSec}
                    viewportWidthPx={viewportWidthPx}
                    viewportStartSec={viewportStartSec}
                    viewportEndSec={viewportEndSec}
                    strokeColor={waveformColors.stroke}
                    strokeWidth={1}
                />
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
                    startEditDrag={startEditDrag as (
                        e: React.PointerEvent,
                        clipId: string,
                        type: Parameters<typeof startEditDrag>[2],
                    ) => void}
                />
                {/* Ghost clip 预览：Ctrl+拖动复制时显示半透明副本 */}
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
            sameStringArray(prev.overlayClipIds, next.overlayClipIds) &&
            prev.activeGroupIds === next.activeGroupIds &&
            prev.disabledGroupIds === next.disabledGroupIds &&
            prev.onToggleGroupDisabled === next.onToggleGroupDisabled
            // viewportStartSec / viewportEndSec are consumed by WaveformTrackCanvas via the viewport bus
            // after mount, so pure horizontal scroll should not force a TrackLane rerender.
        );
    },
);
