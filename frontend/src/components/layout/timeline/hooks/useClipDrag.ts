import { useRef, useState } from "react";
import { batch } from "react-redux";
import { store, type AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    addTrackRemote,
    checkpointHistory,
    duplicateClipsBulkRemote,
    moveClipRemote,
    moveClipsRemote,
    moveClipStart,
    moveClipTrack,
    selectClipRemote,
    setClipAutoFades,
    selectTrackRemote,
    seekPlayhead,
    setplayheadSec,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import {
    applyAutoCrossfade,
    applyDetachedAutoCrossfadeClears,
    computeAutoCrossfadeFromPayload,
    computeInitialCrossfadeSides,
    previewAutoCrossfade,
} from "./autoCrossfade";
import { expandClipIdsWithGroups } from "./useGroupExpansion";
import { buildDuplicateClipsBulkPayload } from "./bulkClipRemotePayloads";
import {
    buildDropToNewTrackMoves,
    computeSelectedTrackSpan,
    computeTrackMoveBounds,
} from "./clipDropMoveUtils";
import {
    computeTimelineTrackDragLock,
    computeTimelineTrackDragLockThresholdPx,
} from "../runtime/timelineTrackDragLock";
import { webApi } from "../../../../services/webviewApi";
import { resolveClipDragCopyMode } from "./clipDragCopyMode";
import {
    applyRippleFollowerShift,
    buildRippleFollowers,
    type RippleFollowerMap,
    type RippleMode,
} from "../../../../features/session/ripplePreview";

export const NEW_TRACK_SENTINEL = "__hs_new_track__";

/** 把自动交叉淡化实时预览改动过的**自动** fade 恢复为拖拽初始值（取消/复制时调用）。 */
export function restoreInitialAutoFades(
    dispatch: AppDispatch,
    initialAutoFadeById: Record<string, { autoFadeInSec: number; autoFadeOutSec: number }>,
): void {
    for (const [clipId, fades] of Object.entries(initialAutoFadeById)) {
        dispatch(
            setClipAutoFades({
                clipId,
                autoFadeInSec: fades.autoFadeInSec,
                autoFadeOutSec: fades.autoFadeOutSec,
            }),
        );
    }
}

/** copyMode 拖动时的 ghost 预览信息 */
export type GhostDragInfo = {
    /** 参与复制拖动的 clip id 列表 */
    clipIds: string[];
    /** 每个 clip 的初始位置（秒）和 trackId */
    initialById: Record<string, { startSec: number; trackId: string }>;
    /** 相对于初始位置的偏移量（秒） */
    deltaSec: number;
    /** 目标 trackId（null 表示新轨道） */
    targetTrackId: string | null;
    /** 相对锚点轨道的偏移（用于跨轨道多选保持相对关系） */
    targetTrackOffset: number;
    /** 是否允许跨轨道移动 */
    allowTrackMove: boolean;
};

export type ClipDragState = {
    pointerId: number;
    anchorClipId: string;
    clipIds: string[];
    offsetBeat: number;
    initialById: Record<string, { startSec: number; trackId: string }>;
    minstartSec: number;
    allowTrackMove: boolean;
    initialAnchorstartSec: number;
    initialAnchorTrackId: string;
    initialTrackOrder: string[];
    initialTrackIndexById: Record<string, number>;
    minTrackOffset: number;
    maxTrackOffset: number;
    allowDropToNewTrack: boolean;
    hasMixedTrackSelection: boolean;
    lastTrackOffset: number;
    lastTrackId: string | null;
    lastDeltaBeat: number;
    copyMode: boolean;
    ctrlSelectionToggle: boolean;
    startClientX: number;
    startClientY: number;
    hasMoved: boolean;
    /** 拖拽开始时读取的波纹模式（与后端提交时读到的设置保持一致）。 */
    rippleMode: RippleMode;
    /** 波纹跟随集：clipId → 初始起点（仅波纹开启且有跟随对象时非空）。 */
    rippleFollowers: RippleFollowerMap;
    /**
     * 自动交叉淡化实时预览用：受影响 clip（被拖拽 + 同轨邻居）的初始**自动** fade 值。
     * 用于取消/复制时把实时预览的自动 fade 恢复原位（手动 fade 永不被改动）。
     */
    initialAutoFadeById: Record<string, { autoFadeInSec: number; autoFadeOutSec: number }>;
    /**
     * 自动交叉淡化：真正被本次编辑影响的 clip（被拖拽 + 波纹跟随 clip）
     * 及其编辑前直接重叠邻居的最小集合。
     */
    editedXfadeClipIds: string[];
    /**
     * 自动交叉淡化：受影响 clip 在编辑前的每侧重叠关系。
     * 用于“拖开”时只清掉自动交叉淡化、保留手动 fade；预览与提交保持一致。
     */
    initialCrossfadeSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
};

export function useClipDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    rowHeight: number;
    pxPerSec: number;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    dispatch: AppDispatch;
    snapTimeline: (
        sec: number,
        object: "mediaItem",
        opts?: {
            originSec?: number;
            anchorTrackId?: string | null;
            excludeClipIds?: ReadonlySet<string>;
        },
    ) => number;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    trackIdFromClientY: (clientY: number) => string | null;
    setClipDropNewTrack: (v: boolean) => void;
    setMultiSelectedClipIds: (ids: string[]) => void;
    /** modifier.clipSlipEdit 绑定 */
    slipEditKb: Keybinding;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 吸附全局开关 */
    snapEnabled: boolean;
    /** modifier.clipCopyDrag 绑定 */
    copyDragKb: Keybinding;
    /** 自动交叉淡入淡出 */
    autoCrossfadeEnabled: boolean;
    /** 忽略编组 */
    ignoreGrouping: boolean;
    /** Ctrl+点击（未拖动）时的多选切换回调 */
    onCtrlClick?: (clipId: string) => void;
}) {
    const {
        scrollRef,
        sessionRef,
        multiSelectedClipIds,
        multiSelectedSet,
        dispatch,
        pxPerSec,
        snapTimeline,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        snapEnabled,
        copyDragKb,
        autoCrossfadeEnabled,
        ignoreGrouping,
        onCtrlClick,
    } = deps;
    void snapEnabled;

    const clipDragRef = useRef<ClipDragState | null>(null);
    const [ghostDrag, setGhostDrag] = useState<GhostDragInfo | null>(null);
    const [verticalTrackLockTrackId, setVerticalTrackLockTrackId] = useState<string | null>(null);

    function resolveTrackIdByOffset(
        drag: ClipDragState,
        clipId: string,
        trackOffset: number,
    ): string | null {
        const initial = drag.initialById[clipId];
        if (!initial) return null;
        const sourceIndex = drag.initialTrackIndexById[initial.trackId];
        if (!Number.isFinite(sourceIndex)) return null;
        const targetIndex = sourceIndex + trackOffset;
        return drag.initialTrackOrder[targetIndex] ?? null;
    }

    function startSlipDrag(
        e: React.PointerEvent<HTMLDivElement>,
        clipId: string,
        startSlipDragFn: (e: React.PointerEvent<HTMLDivElement>, clipId: string) => void,
    ) {
        startSlipDragFn(e, clipId);
    }

    function startClipDrag(
        e: React.PointerEvent<HTMLDivElement>,
        clipId: string,
        clipstartSec: number,
        _altPressedHint: boolean | undefined,
        startSlipDragFn: (e: React.PointerEvent<HTMLDivElement>, clipId: string) => void,
    ) {
        if (e.button !== 0) return;

        const anchor = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!anchor) return;

        // Only use the slipEdit keybinding to detect slip-edit drag mode.
        // Do NOT use altPressedHint (which carries the stretch modifier state) —
        // that would break Ctrl/Shift selection when those keys are configured
        // as stretch modifiers.
        const isSlipEdit = isModifierActive(slipEditKb, e.nativeEvent);
        if (isSlipEdit) {
            startSlipDrag(e, clipId, startSlipDragFn);
            return;
        }

        const scroller = scrollRef.current;
        if (!scroller) return;
        const bounds = scroller.getBoundingClientRect();
        const beatAtPointer = beatFromClientX(e.clientX, bounds, scroller.scrollLeft);

        // Expand to include selected clips and their group members
        const initialIds =
            multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId)
                ? [...multiSelectedClipIds]
                : [clipId];
        const clipIds = ignoreGrouping
            ? initialIds
            : expandClipIdsWithGroups(
                  initialIds,
                  sessionRef.current.clips,
                  false,
                  sessionRef.current.disabledGroupIds,
              );

        const initialById: Record<string, { startSec: number; trackId: string }> = {};
        let minstartSec = Number.POSITIVE_INFINITY;
        let allowTrackMove = true;
        let baseTrackId: string | null = null;
        const trackOrder = sessionRef.current.tracks.map((t) => String(t.id));
        const trackIndexById = Object.fromEntries(trackOrder.map((id, idx) => [id, idx])) as Record<
            string,
            number
        >;
        for (const id of clipIds) {
            const c = sessionRef.current.clips.find((x) => x.id === id);
            if (!c) continue;
            const startSec = Math.max(0, Number(c.startSec ?? 0));
            initialById[id] = { startSec, trackId: String(c.trackId) };
            minstartSec = Math.min(minstartSec, startSec);
            if (baseTrackId == null) baseTrackId = String(c.trackId);
        }
        if (!Number.isFinite(minstartSec)) minstartSec = 0;

        // 波纹（自动跟进）实时预览：拖拽开始时快照“后续跟随剪辑”的初始位置。
        // 起点 = 被拖拽选择的最早起点；作用域轨道 = 各被拖拽剪辑的初始轨道。
        const rippleMode = sessionRef.current.rippleMode;
        const rippleFollowers = buildRippleFollowers(
            sessionRef.current.clips,
            new Set(clipIds),
            minstartSec,
            rippleMode,
            new Set(Object.values(initialById).map((i) => String(i.trackId))),
        );

        // 自动交叉淡化实时预览：受影响 clip = 被拖拽 clip + 波纹跟随 clip
        // （都是“被编辑”的 clip）+ 编辑前与它们直接重叠的同轨邻居。
        // 只服务真正因本次编辑而变化的淡入淡出侧；同轨但无关的 clip 不会被波及。
        const editedXfadeClipIds = Array.from(
            new Set<string>([...clipIds, ...Object.keys(rippleFollowers)]),
        );
        const initialCrossfadeSides = computeInitialCrossfadeSides(
            sessionRef.current.clips,
            editedXfadeClipIds,
        );
        const xfadeAffectedIds = new Set<string>(editedXfadeClipIds);
        for (const id of Object.keys(initialCrossfadeSides)) {
            xfadeAffectedIds.add(id);
        }
        const initialAutoFadeById: Record<string, { autoFadeInSec: number; autoFadeOutSec: number }> =
            {};
        for (const id of xfadeAffectedIds) {
            const c = sessionRef.current.clips.find((x) => x.id === id);
            if (!c) continue;
            initialAutoFadeById[id] = {
                autoFadeInSec: Number(c.autoFadeInSec ?? 0),
                autoFadeOutSec: Number(c.autoFadeOutSec ?? 0),
            };
        }

        const hasMixedTrackSelection = clipIds.some((id) => {
            const initial = initialById[id];
            return initial && baseTrackId != null && initial.trackId !== baseTrackId;
        });

        const initialTrackId = anchor.trackId;
        const anchorTrackIndex = trackIndexById[initialTrackId];
        if (!Number.isFinite(anchorTrackIndex)) {
            allowTrackMove = false;
        }

        const trackBounds = computeTrackMoveBounds({
            trackCount: trackOrder.length,
            clipIds,
            initialById,
            trackIndexById,
        });
        if (!trackBounds) {
            allowTrackMove = false;
        }
        const minTrackOffset = trackBounds?.minTrackOffset ?? 0;
        const maxTrackOffset = trackBounds?.maxTrackOffset ?? 0;

        const targetTrackId = trackIdFromClientY(e.clientY) ?? initialTrackId;
        // 允许对混合轨道选择也创建新轨（后续释放时会根据源轨跨度创建多条轨道）
        const allowDropToNewTrackComputed = true;
        clipDragRef.current = {
            pointerId: e.pointerId,
            anchorClipId: clipId,
            clipIds,
            offsetBeat: beatAtPointer - clipstartSec,
            initialById,
            minstartSec,
            allowTrackMove,
            initialAnchorstartSec: clipstartSec,
            initialAnchorTrackId: initialTrackId,
            initialTrackOrder: trackOrder,
            initialTrackIndexById: trackIndexById,
            minTrackOffset,
            maxTrackOffset,
            allowDropToNewTrack: allowDropToNewTrackComputed,
            hasMixedTrackSelection,
            lastTrackOffset: 0,
            lastTrackId: targetTrackId,
            lastDeltaBeat: 0,
            copyMode: false,
            // Ctrl+点击（无拖动）多选切换标记；仅在未发生拖动时生效。
            // 拖动开始后此标记被清除，由拖拽逻辑接管（Ctrl 拖动 = 复制模式）。
            ctrlSelectionToggle: e.ctrlKey || e.metaKey,
            startClientX: e.clientX,
            startClientY: e.clientY,
            hasMoved: false,
            rippleMode,
            rippleFollowers,
            editedXfadeClipIds,
            initialAutoFadeById,
            initialCrossfadeSides,
        };
        setVerticalTrackLockTrackId(null);
        scroller.setPointerCapture(e.pointerId);

        function onMove(ev: PointerEvent) {
            const drag = clipDragRef.current;
            const el = scrollRef.current;
            if (!drag || drag.pointerId !== e.pointerId || !el) return;

            if (!drag.hasMoved) {
                const dx = ev.clientX - drag.startClientX;
                const dy = ev.clientY - drag.startClientY;
                if (dx * dx + dy * dy < 9) return;
                drag.hasMoved = true;
                drag.ctrlSelectionToggle = false;
                // 拖动开始时根据当前按键状态决定是否为复制拖动
                drag.copyMode = resolveClipDragCopyMode({
                    existingCopyMode: drag.copyMode,
                    ctrlKey: ev.ctrlKey,
                    metaKey: ev.metaKey,
                    modifierActive: isModifierActive(copyDragKb, ev),
                });
                if (!drag.copyMode) {
                    dispatch(checkpointHistory());
                    dispatch(beginInteraction());
                    // Begin backend undo group so that move_clip + auto-crossfade
                    // share a single backend undo entry.
                    void webApi.beginUndoGroup();
                }
            }

            // 拖动过程中允许 copyMode 随按键变化（但不会从 true 变回 false）
            const copyMode = resolveClipDragCopyMode({
                existingCopyMode: drag.copyMode,
                ctrlKey: ev.ctrlKey,
                metaKey: ev.metaKey,
                modifierActive: isModifierActive(copyDragKb, ev),
            });
            if (copyMode !== drag.copyMode) {
                drag.copyMode = copyMode;
            }
            const b = el.getBoundingClientRect();
            const beatNow = beatFromClientX(ev.clientX, b, el.scrollLeft);
            let nextStart = Math.max(0, beatNow - drag.offsetBeat);
            const noSnapActive = isModifierActive(noSnapKb, ev);
            const effectiveSnap = snapEnabled && !noSnapActive;
            if (effectiveSnap) {
                nextStart = snapTimeline(nextStart, "mediaItem", {
                    originSec: drag.initialAnchorstartSec,
                    anchorTrackId: drag.initialAnchorTrackId,
                    excludeClipIds: new Set(drag.clipIds),
                });
            }

            let deltaBeat = nextStart - drag.initialAnchorstartSec;
            deltaBeat = Math.max(deltaBeat, -drag.minstartSec);
            drag.lastDeltaBeat = deltaBeat;

            const hoveredTrackId = trackIdFromClientY(ev.clientY);
            const hoveredTrackIndex =
                hoveredTrackId != null ? drag.initialTrackIndexById[hoveredTrackId] : undefined;

            let nextTrackOffset = drag.lastTrackOffset;
            if (Number.isFinite(hoveredTrackIndex)) {
                const rawOffset =
                    Number(hoveredTrackIndex) -
                    Number(drag.initialTrackIndexById[drag.initialAnchorTrackId]);
                nextTrackOffset = Math.max(
                    drag.minTrackOffset,
                    Math.min(drag.maxTrackOffset, rawOffset),
                );
            }
            const nextTrackId = drag.allowTrackMove
                ? hoveredTrackId == null
                    ? drag.allowDropToNewTrack
                        ? null
                        : resolveTrackIdByOffset(drag, drag.anchorClipId, nextTrackOffset)
                    : resolveTrackIdByOffset(drag, drag.anchorClipId, nextTrackOffset)
                : drag.initialAnchorTrackId;

            if (drag.allowTrackMove) {
                drag.lastTrackOffset = nextTrackOffset;
                drag.lastTrackId = nextTrackId;
                setClipDropNewTrack(drag.allowDropToNewTrack && nextTrackId == null);
            } else {
                drag.lastTrackOffset = 0;
                drag.lastTrackId = drag.initialAnchorTrackId;
                setClipDropNewTrack(false);
            }

            const horizontalPx = Math.abs(ev.clientX - drag.startClientX);
            const trackLock = computeTimelineTrackDragLock({
                initialTrackId: drag.initialAnchorTrackId,
                hoveredTrackId,
                horizontalDeltaPx: horizontalPx,
                thresholdPx: computeTimelineTrackDragLockThresholdPx(pxPerSec),
            });
            setVerticalTrackLockTrackId(trackLock.lockedTrackId);
            if (trackLock.locked) {
                deltaBeat = 0;
                drag.lastDeltaBeat = 0;
            }

            // copyMode 时不移动原 clip，只更新 ghost 预览位置
            if (copyMode) {
                setGhostDrag((prev) => {
                    if (
                        prev &&
                        prev.deltaSec === deltaBeat &&
                        prev.targetTrackId === nextTrackId &&
                        prev.targetTrackOffset === drag.lastTrackOffset &&
                        prev.allowTrackMove === drag.allowTrackMove &&
                        prev.clipIds === drag.clipIds &&
                        prev.initialById === drag.initialById
                    ) {
                        return prev;
                    }
                    return {
                        clipIds: drag.clipIds,
                        initialById: drag.initialById,
                        deltaSec: deltaBeat,
                        targetTrackId: nextTrackId,
                        targetTrackOffset: drag.lastTrackOffset,
                        allowTrackMove: drag.allowTrackMove,
                    };
                });
                // 复制模式下原 clip 不移动，波纹跟随集保持原位（覆盖拖动中途切到复制的残留预览）。
                applyRippleFollowerShift(dispatch, drag.rippleFollowers, 0);
                // 复制不移动原片，实时预览的交叉淡化也恢复原位。
                restoreInitialAutoFades(dispatch, drag.initialAutoFadeById);
            } else {
                batch(() => {
                    for (const id of drag.clipIds) {
                        const initial = drag.initialById[id];
                        if (!initial) continue;
                        dispatch(
                            moveClipStart({
                                clipId: id,
                                startSec: Math.max(0, initial.startSec + deltaBeat),
                            }),
                        );
                        if (drag.allowTrackMove) {
                            const resolvedTrackId =
                                nextTrackId == null
                                    ? NEW_TRACK_SENTINEL
                                    : (resolveTrackIdByOffset(drag, id, drag.lastTrackOffset) ??
                                      initial.trackId);
                            dispatch(
                                moveClipTrack({
                                    clipId: id,
                                    trackId: resolvedTrackId,
                                }),
                            );
                        }
                    }
                    // 波纹（自动跟进）实时预览：后续剪辑随拖拽同步平移。
                    // 与后端“右缘位移”规则一致：同一拖拽位移量同时作用于所有跟随剪辑。
                    if (drag.rippleMode !== "off") {
                        applyRippleFollowerShift(dispatch, drag.rippleFollowers, drag.lastDeltaBeat);
                    }
                    // 自动交叉淡化实时预览：按当前（乐观）位置计算重叠并实时更新自动 fade 包络。
                    // affectedSides = 拖拽前的每侧重叠关系（分开时仅清自动交叉淡化、保留手动 fade）。
                    // 注意：这里必须用 store.getState().session（同步新鲜），而不是 sessionRef.current——
                    // React-Redux batch 会延迟 store.subscribe 回调，batch 内 sessionRef 仍是上一帧位置，
                    // 否则“拖开瞬间”预览会滞留最后一帧自动淡化长度，松手后跳变为手动美化。
                    if (autoCrossfadeEnabled) {
                        previewAutoCrossfade(
                            store.getState().session,
                            drag.editedXfadeClipIds,
                            dispatch,
                            drag.initialCrossfadeSides,
                        );
                    }
                });
            }
        }

        function end() {
            const drag = clipDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            clipDragRef.current = null;
            setClipDropNewTrack(false);
            setVerticalTrackLockTrackId(null);

            const maybeSelectTargetTrack = (targetTrackId: string | null) => {
                if (!targetTrackId) return;
                if (targetTrackId === drag.initialAnchorTrackId) return;
                if (sessionRef.current.selectedTrackId === targetTrackId) return;
                void dispatch(selectTrackRemote(targetTrackId));
            };

            // 清除 ghost 预览
            setGhostDrag(null);

            if (!drag.hasMoved) {
                // Ctrl+点击（未移动）：执行多选切换
                if (drag.ctrlSelectionToggle && onCtrlClick) {
                    onCtrlClick(drag.anchorClipId);
                }
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", end);
                window.removeEventListener("pointercancel", end);
                return;
            }

            const session = sessionRef.current;
            const dropToNewTrack =
                drag.allowTrackMove && drag.allowDropToNewTrack && drag.lastTrackId == null;

            async function createNewTrackForDrop(): Promise<string | null> {
                const before = new Set(sessionRef.current.tracks.map((t) => t.id));
                const res = (await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap()) as {
                    tracks?: Array<{ id?: string }>;
                    selected_track_id?: string | null;
                };
                const nextTracks = Array.isArray(res?.tracks) ? res.tracks : [];
                const created = nextTracks.find((t) => !before.has(String(t?.id)));
                return (
                    (created && String(created.id)) ||
                    (res?.selected_track_id ? String(res.selected_track_id) : null)
                );
            }

            async function createNewTracksForDrop(count: number): Promise<string[]> {
                const createdIds: string[] = [];
                for (let i = 0; i < count; i += 1) {
                    const before = new Set(sessionRef.current.tracks.map((t) => t.id));
                    const res = (await dispatch(
                        addTrackRemote({
                            name: undefined,
                            parentTrackId: null,
                        }),
                    ).unwrap()) as {
                        tracks?: Array<{ id?: string }>;
                        selected_track_id?: string | null;
                    };
                    const nextTracks = Array.isArray(res?.tracks) ? res.tracks : [];
                    const created = nextTracks.find((t) => !before.has(String(t?.id)));
                    const id =
                        (created && String(created.id)) ||
                        (res?.selected_track_id ? String(res.selected_track_id) : null) ||
                        nextTracks[nextTracks.length - 1]?.id ||
                        null;
                    if (id) createdIds.push(String(id));
                }
                return createdIds;
            }

            const applyOptimisticMoves = (
                moves: Array<{
                    clipId: string;
                    startSec: number;
                    trackId: string;
                }>,
            ) => {
                batch(() => {
                    for (const move of moves) {
                        dispatch(
                            moveClipStart({
                                clipId: move.clipId,
                                startSec: move.startSec,
                            }),
                        );
                        dispatch(
                            moveClipTrack({
                                clipId: move.clipId,
                                trackId: move.trackId,
                            }),
                        );
                    }
                });
            };

            if (drag.copyMode) {
                // copyMode 下原 clip 未被移动，直接根据 ghost 偏移量计算副本位置
                // copyMode 不使用交互锁（原 clip 未被拖动改变位置）
                // 复制不产生波纹，松手前把跟随集恢复原位（覆盖预览残留）。
                applyRippleFollowerShift(dispatch, drag.rippleFollowers, 0);
                restoreInitialAutoFades(dispatch, drag.initialAutoFadeById);
                void (async () => {
                    const sourceClipIds = drag.clipIds.filter((id) =>
                        sessionRef.current.clips.some((clip) => clip.id === id),
                    );
                    if (sourceClipIds.length === 0) {
                        return;
                    }
                    dispatch(checkpointHistory());
                    void (async () => {
                        // Begin backend undo group for copy-drag + auto-crossfade
                        await webApi.beginUndoGroup();
                        try {
                            const targetTrackIdByClipId = new Map<string, string>();
                            if (dropToNewTrack) {
                                if (drag.hasMixedTrackSelection) {
                                    const spanInfo = computeSelectedTrackSpan({
                                        clipIds: drag.clipIds,
                                        initialById: drag.initialById,
                                        trackIndexById: drag.initialTrackIndexById,
                                    });
                                    if (!spanInfo) throw new Error("create_track_failed");
                                    const created = await createNewTracksForDrop(spanInfo.span);
                                    if (created.length !== spanInfo.span) {
                                        throw new Error("create_track_failed");
                                    }
                                    for (const clipId of sourceClipIds) {
                                        const initial = drag.initialById[clipId];
                                        if (!initial) continue;
                                        const srcIdx = drag.initialTrackIndexById[initial.trackId];
                                        if (!Number.isFinite(srcIdx)) continue;
                                        const offset = Number(srcIdx) - spanInfo.minTrackIndex;
                                        const targetTrackId = created[offset];
                                        if (targetTrackId) {
                                            targetTrackIdByClipId.set(clipId, targetTrackId);
                                        }
                                    }
                                } else {
                                    const newTrackId = await createNewTrackForDrop();
                                    if (!newTrackId) throw new Error("create_track_failed");
                                    for (const clipId of sourceClipIds) {
                                        targetTrackIdByClipId.set(clipId, newTrackId);
                                    }
                                }
                            } else {
                                for (const clipId of sourceClipIds) {
                                    const initial = drag.initialById[clipId];
                                    if (!initial) continue;
                                    const targetTrackId =
                                        drag.allowTrackMove && drag.lastTrackOffset !== 0
                                            ? (resolveTrackIdByOffset(
                                                  drag,
                                                  clipId,
                                                  drag.lastTrackOffset,
                                              ) ?? initial.trackId)
                                            : initial.trackId;
                                    targetTrackIdByClipId.set(clipId, targetTrackId);
                                }
                            }

                            const firstTargetTrackId = targetTrackIdByClipId.get(sourceClipIds[0]);
                            if (firstTargetTrackId) {
                                maybeSelectTargetTrack(firstTargetTrackId);
                            }
                            const trackMapping = new Map<string, string>();
                            for (const clipId of sourceClipIds) {
                                const initial = drag.initialById[clipId];
                                const targetTrackId = targetTrackIdByClipId.get(clipId);
                                if (!initial || !targetTrackId) continue;
                                trackMapping.set(initial.trackId, targetTrackId);
                            }
                            if (trackMapping.size === 0) return;
                            const trackMode = Array.from(trackMapping.entries()).every(
                                ([sourceTrackId, targetTrackId]) => sourceTrackId === targetTrackId,
                            )
                                ? { kind: "same_track" }
                                : {
                                      kind: "explicit_mapping",
                                      mapping: Object.fromEntries(trackMapping),
                                  };
                            const payload = await dispatch(
                                duplicateClipsBulkRemote(
                                    buildDuplicateClipsBulkPayload({
                                        sourceClipIds,
                                        deltaSec: drag.lastDeltaBeat,
                                        copyLinkedParams: sessionRef.current.lockParamLinesEnabled,
                                        applyAutoCrossfade: autoCrossfadeEnabled,
                                        trackMode,
                                        renameCopies: false,
                                    }),
                                ),
                            ).unwrap();
                            const created: string[] = payload?.createdClipIds ?? [];
                            if (!Array.isArray(created) || created.length === 0) return;
                            setMultiSelectedClipIds(created);
                            void dispatch(selectClipRemote(created[0]));
                            // 复制拖动后，将播放光标定位到目标时间点（所有副本中最靠前的起始位置）
                            const targetStartSec = sourceClipIds.reduce((min, clipId) => {
                                const initial = drag.initialById[clipId];
                                if (!initial) return min;
                                return Math.min(
                                    min,
                                    Math.max(0, initial.startSec + drag.lastDeltaBeat),
                                );
                            }, Infinity);
                            if (Number.isFinite(targetStartSec)) {
                                dispatch(setplayheadSec(targetStartSec));
                                void dispatch(seekPlayhead(targetStartSec));
                            }
                            // 复制拖动后，尝试对新创建的 clip 应用自动交叉淡化
                            if (autoCrossfadeEnabled) {
                                const allClips = (payload?.clips ?? []) as Array<{
                                    id?: string;
                                    track_id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                    fade_in_sec?: number;
                                    fade_out_sec?: number;
                                }>;
                                const fadeUpdates = computeAutoCrossfadeFromPayload(
                                    allClips,
                                    created,
                                );
                                if (fadeUpdates.length > 0) {
                                    // 复制后的自动交叉淡化写入“自动 fade”（与手动 fade 分离）。
                                    for (const u of fadeUpdates) {
                                        dispatch(
                                            setClipAutoFades({
                                                clipId: u.clipId,
                                                autoFadeInSec: u.autoFadeInSec,
                                                autoFadeOutSec: u.autoFadeOutSec,
                                            }),
                                        );
                                        await webApi.setClipState({
                                            clipId: u.clipId,
                                            autoFadeInSec: u.autoFadeInSec,
                                            autoFadeOutSec: u.autoFadeOutSec,
                                            checkpoint: false,
                                        });
                                    }
                                }
                            }
                        } finally {
                            void webApi.endUndoGroup();
                        }
                    })().catch(() => undefined);
                })().catch(() => undefined);
            } else {
                // 非 copyMode：交互锁在最终持久化请求完成后才释放，
                // 避免 endInteraction() 到 fulfilled 之间的窗口内，
                // 其他 in-flight thunk（如 selectClipRemote）的旧快照覆盖前端乐观更新导致闪烁。

                if (dropToNewTrack) {
                    void (async () => {
                        try {
                            if (drag.hasMixedTrackSelection) {
                                const spanInfo = computeSelectedTrackSpan({
                                    clipIds: drag.clipIds,
                                    initialById: drag.initialById,
                                    trackIndexById: drag.initialTrackIndexById,
                                });
                                if (!spanInfo) throw new Error("create_track_failed");

                                const minIdx = spanInfo.minTrackIndex;
                                const span = spanInfo.span;
                                const created = await createNewTracksForDrop(span);
                                if (created.length !== span) throw new Error("create_track_failed");
                                maybeSelectTargetTrack(created[0] ?? null);
                                const moves = buildDropToNewTrackMoves({
                                    clipIds: drag.clipIds,
                                    initialById: drag.initialById,
                                    deltaSec: drag.lastDeltaBeat,
                                    resolveTargetTrackId: (_clipId, initialTrackId) => {
                                        const srcIdx = drag.initialTrackIndexById[initialTrackId];
                                        if (!Number.isFinite(srcIdx)) return null;
                                        const offset = Number(srcIdx) - minIdx;
                                        return created[offset] ?? null;
                                    },
                                });
                                if (moves.length === 0) throw new Error("create_track_failed");
                                applyOptimisticMoves(moves);
                                if (moves.length > 1) {
                                    await dispatch(
                                        moveClipsRemote({
                                            moves,
                                            moveLinkedParams:
                                                sessionRef.current.lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                } else if (moves.length === 1) {
                                    await dispatch(
                                        moveClipRemote({
                                            clipId: moves[0].clipId,
                                            startSec: moves[0].startSec,
                                            trackId: moves[0].trackId,
                                            moveLinkedParams:
                                                sessionRef.current.lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                }
                            } else {
                                const newTrackId = await createNewTrackForDrop();
                                if (!newTrackId) throw new Error("create_track_failed");
                                maybeSelectTargetTrack(newTrackId);
                                const moves = buildDropToNewTrackMoves({
                                    clipIds: drag.clipIds,
                                    initialById: drag.initialById,
                                    deltaSec: drag.lastDeltaBeat,
                                    resolveTargetTrackId: () => newTrackId,
                                });
                                if (moves.length === 0) throw new Error("create_track_failed");
                                applyOptimisticMoves(moves);
                                if (moves.length > 1) {
                                    await dispatch(
                                        moveClipsRemote({
                                            moves,
                                            moveLinkedParams:
                                                sessionRef.current.lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                } else if (moves.length === 1) {
                                    await dispatch(
                                        moveClipRemote({
                                            clipId: moves[0].clipId,
                                            startSec: moves[0].startSec,
                                            trackId: moves[0].trackId,
                                            moveLinkedParams:
                                                sessionRef.current.lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                }
                            }
                            if (autoCrossfadeEnabled) {
                                const latestSession = sessionRef.current;
                                await applyAutoCrossfade(latestSession, drag.editedXfadeClipIds, dispatch, {
                                    affectedSides: drag.initialCrossfadeSides,
                                });
                            } else {
                                // 开关关闭时只清理“已脱离重叠”的自动交叉淡化，
                                // 保证导入/遗留的自动值不会盖住分离后应该恢复的手动 fade。
                                const latestSession = sessionRef.current;
                                await applyDetachedAutoCrossfadeClears(
                                    latestSession,
                                    drag.editedXfadeClipIds,
                                    dispatch,
                                    drag.initialCrossfadeSides,
                                );
                            }
                        } catch {
                            batch(() => {
                                for (const id of drag.clipIds) {
                                    const initial = drag.initialById[id];
                                    if (!initial) continue;
                                    dispatch(
                                        moveClipStart({
                                            clipId: id,
                                            startSec: Math.max(0, initial.startSec),
                                        }),
                                    );
                                    dispatch(
                                        moveClipTrack({
                                            clipId: id,
                                            trackId: initial.trackId,
                                        }),
                                    );
                                }
                                // 波纹跟随集同样回滚到初始位置。
                                applyRippleFollowerShift(dispatch, drag.rippleFollowers, 0);
                                // 自动交叉淡化（实时预览）的 fade 也回滚到初始值。
                                restoreInitialAutoFades(dispatch, drag.initialAutoFadeById);
                            });
                        } finally {
                            void webApi.endUndoGroup();
                            dispatch(endInteraction());
                        }
                    })();
                    window.removeEventListener("pointermove", onMove);
                    window.removeEventListener("pointerup", end);
                    window.removeEventListener("pointercancel", end);
                    return;
                }

                maybeSelectTargetTrack(drag.lastTrackId ?? null);

                const moves = drag.clipIds
                    .map((id) => {
                        const initial = drag.initialById[id];
                        if (!initial) return null;
                        const startSec = Math.max(0, initial.startSec + drag.lastDeltaBeat);
                        const trackId =
                            drag.allowTrackMove && drag.lastTrackOffset !== 0
                                ? (resolveTrackIdByOffset(drag, id, drag.lastTrackOffset) ??
                                  initial.trackId)
                                : initial.trackId;
                        const changedBeat = Math.abs(startSec - initial.startSec) > 1e-6;
                        const changedTrack = trackId !== initial.trackId;
                        if (!changedBeat && !changedTrack) return null;
                        return {
                            clipId: id,
                            startSec,
                            trackId,
                        };
                    })
                    .filter(
                        (
                            move,
                        ): move is {
                            clipId: string;
                            startSec: number;
                            trackId: string;
                        } => move != null,
                    );

                // Auto crossfade: 等所有 move 完成后再计算并持久化交叉淡化
                if (moves.length > 0) {
                    const movePromise =
                        moves.length > 1
                            ? dispatch(
                                  moveClipsRemote({
                                      moves,
                                      moveLinkedParams: sessionRef.current.lockParamLinesEnabled,
                                  }),
                              ).unwrap()
                            : dispatch(
                                  moveClipRemote({
                                      clipId: moves[0].clipId,
                                      startSec: moves[0].startSec,
                                      trackId: moves[0].trackId,
                                      moveLinkedParams: sessionRef.current.lockParamLinesEnabled,
                                  }),
                              ).unwrap();
                    void (async () => {
                        try {
                            await movePromise;
                        } finally {
                            if (autoCrossfadeEnabled) {
                                const latestSession = sessionRef.current;
                                await applyAutoCrossfade(latestSession, drag.editedXfadeClipIds, dispatch, {
                                    affectedSides: drag.initialCrossfadeSides,
                                });
                            } else {
                                const latestSession = sessionRef.current;
                                await applyDetachedAutoCrossfadeClears(
                                    latestSession,
                                    drag.editedXfadeClipIds,
                                    dispatch,
                                    drag.initialCrossfadeSides,
                                );
                            }
                            await webApi.endUndoGroup();
                            dispatch(endInteraction());
                        }
                    })().catch(() => undefined);
                } else {
                    void (async () => {
                        if (autoCrossfadeEnabled) {
                            await applyAutoCrossfade(session, drag.editedXfadeClipIds, dispatch, {
                                affectedSides: drag.initialCrossfadeSides,
                            });
                        } else {
                            await applyDetachedAutoCrossfadeClears(
                                session,
                                drag.editedXfadeClipIds,
                                dispatch,
                                drag.initialCrossfadeSides,
                            );
                        }
                        await webApi.endUndoGroup();
                        dispatch(endInteraction());
                    })().catch(() => undefined);
                }
            }
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    return { clipDragRef, startClipDrag, ghostDrag, verticalTrackLockTrackId };
}
