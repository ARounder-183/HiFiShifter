import { useRef, useState } from "react";
import { batch } from "react-redux";
import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    addTrackRemote,
    checkpointHistory,
    createClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    moveClipStart,
    moveClipTrack,
    selectClipRemote,
    setClipStateRemote,
    selectTrackRemote,
    seekPlayhead,
    setplayheadSec,
    beginInteraction,
    endInteraction,
} from "../../../../features/session/sessionSlice";
import type { ClipTemplate } from "../../../../features/session/sessionTypes";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import {
    applyAutoCrossfade,
    computeAutoCrossfadeFromPayload,
} from "./autoCrossfade";
import { webApi } from "../../../../services/webviewApi";

export const NEW_TRACK_SENTINEL = "__hs_new_track__";

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
    startClientX: number;
    startClientY: number;
    hasMoved: boolean;
};

export function useClipDrag(deps: {
    scrollRef: React.RefObject<HTMLDivElement | null>;
    sessionRef: React.RefObject<SessionState>;
    rowHeight: number;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    dispatch: AppDispatch;
    snapBeat: (beat: number) => number;
    beatFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
    ) => number;
    trackIdFromClientY: (clientY: number) => string | null;
    setClipDropNewTrack: (v: boolean) => void;
    setMultiSelectedClipIds: (ids: string[]) => void;
    /** modifier.clipSlipEdit 绑定 */
    slipEditKb: Keybinding;
    /** modifier.clipNoSnap 绑定 */
    noSnapKb: Keybinding;
    /** 网格吸附全局开关 */
    gridSnapEnabled: boolean;
    /** modifier.clipCopyDrag 绑定 */
    copyDragKb: Keybinding;
    /** 自动交叉淡入淡出 */
    autoCrossfadeEnabled: boolean;
    /** Ctrl+点击（未拖动）时的多选切换回调 */
    onCtrlClick?: (clipId: string) => void;
}) {
    const {
        scrollRef,
        sessionRef,
        multiSelectedClipIds,
        multiSelectedSet,
        dispatch,
        snapBeat,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        gridSnapEnabled,
        copyDragKb,
        autoCrossfadeEnabled,
        onCtrlClick,
    } = deps;
    void gridSnapEnabled;

    const clipDragRef = useRef<ClipDragState | null>(null);
    const [ghostDrag, setGhostDrag] = useState<GhostDragInfo | null>(null);

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
        startSlipDragFn: (
            e: React.PointerEvent<HTMLDivElement>,
            clipId: string,
        ) => void,
    ) {
        startSlipDragFn(e, clipId);
    }

    function startClipDrag(
        e: React.PointerEvent<HTMLDivElement>,
        clipId: string,
        clipstartSec: number,
        altPressedHint: boolean | undefined,
        startSlipDragFn: (
            e: React.PointerEvent<HTMLDivElement>,
            clipId: string,
        ) => void,
    ) {
        if (e.button !== 0) return;

        const anchor = sessionRef.current.clips.find((c) => c.id === clipId);
        if (!anchor) return;

        const alt = Boolean(
            altPressedHint || isModifierActive(slipEditKb, e.nativeEvent),
        );
        if (alt) {
            startSlipDrag(e, clipId, startSlipDragFn);
            return;
        }

        const scroller = scrollRef.current;
        if (!scroller) return;
        const bounds = scroller.getBoundingClientRect();
        const beatAtPointer = beatFromClientX(
            e.clientX,
            bounds,
            scroller.scrollLeft,
        );

        const clipIds =
            multiSelectedClipIds.length > 0 && multiSelectedSet.has(clipId)
                ? [...multiSelectedClipIds]
                : [clipId];

        const initialById: Record<
            string,
            { startSec: number; trackId: string }
        > = {};
        let minstartSec = Number.POSITIVE_INFINITY;
        let allowTrackMove = true;
        let baseTrackId: string | null = null;
        const trackOrder = sessionRef.current.tracks.map((t) => String(t.id));
        const trackIndexById = Object.fromEntries(
            trackOrder.map((id, idx) => [id, idx]),
        ) as Record<string, number>;
        for (const id of clipIds) {
            const c = sessionRef.current.clips.find((x) => x.id === id);
            if (!c) continue;
            const startSec = Math.max(0, Number(c.startSec ?? 0));
            initialById[id] = { startSec, trackId: String(c.trackId) };
            minstartSec = Math.min(minstartSec, startSec);
            if (baseTrackId == null) baseTrackId = String(c.trackId);
        }
        if (!Number.isFinite(minstartSec)) minstartSec = 0;

        const hasMixedTrackSelection = clipIds.some((id) => {
            const initial = initialById[id];
            return (
                initial &&
                baseTrackId != null &&
                initial.trackId !== baseTrackId
            );
        });

        const initialTrackId = anchor.trackId;
        const anchorTrackIndex = trackIndexById[initialTrackId];
        if (!Number.isFinite(anchorTrackIndex)) {
            allowTrackMove = false;
        }

        let minTrackOffset = 0;
        let maxTrackOffset = 0;
        for (const id of clipIds) {
            const initial = initialById[id];
            if (!initial) continue;
            const idx = trackIndexById[initial.trackId];
            if (!Number.isFinite(idx)) {
                allowTrackMove = false;
                continue;
            }
            minTrackOffset = Math.min(minTrackOffset, -idx);
            maxTrackOffset = Math.max(
                maxTrackOffset,
                trackOrder.length - 1 - idx,
            );
        }

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
            copyMode: isModifierActive(copyDragKb, e.nativeEvent),
            startClientX: e.clientX,
            startClientY: e.clientY,
            hasMoved: false,
        };
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
                if (!drag.copyMode) {
                    dispatch(checkpointHistory());
                    dispatch(beginInteraction());
                    // Begin backend undo group so that move_clip + auto-crossfade
                    // share a single backend undo entry.
                    void webApi.beginUndoGroup();
                }
            }
            const b = el.getBoundingClientRect();
            const beatNow = beatFromClientX(ev.clientX, b, el.scrollLeft);
            let nextStart = Math.max(0, beatNow - drag.offsetBeat);
            const noSnapActive = isModifierActive(noSnapKb, ev);
            const effectiveSnap = gridSnapEnabled
                ? !noSnapActive
                : noSnapActive;
            if (effectiveSnap) {
                nextStart = snapBeat(nextStart);
            }

            let deltaBeat = nextStart - drag.initialAnchorstartSec;
            deltaBeat = Math.max(deltaBeat, -drag.minstartSec);
            drag.lastDeltaBeat = deltaBeat;

            const hoveredTrackId = trackIdFromClientY(ev.clientY);
            const hoveredTrackIndex =
                hoveredTrackId != null
                    ? drag.initialTrackIndexById[hoveredTrackId]
                    : undefined;

            let nextTrackOffset = drag.lastTrackOffset;
            if (Number.isFinite(hoveredTrackIndex)) {
                const rawOffset =
                    Number(hoveredTrackIndex) -
                    Number(
                        drag.initialTrackIndexById[drag.initialAnchorTrackId],
                    );
                nextTrackOffset = Math.max(
                    drag.minTrackOffset,
                    Math.min(drag.maxTrackOffset, rawOffset),
                );
            }
            const nextTrackId = drag.allowTrackMove
                ? hoveredTrackId == null
                    ? drag.allowDropToNewTrack
                        ? null
                        : resolveTrackIdByOffset(
                              drag,
                              drag.anchorClipId,
                              nextTrackOffset,
                          )
                    : resolveTrackIdByOffset(
                          drag,
                          drag.anchorClipId,
                          nextTrackOffset,
                      )
                : drag.initialAnchorTrackId;

            if (drag.allowTrackMove) {
                drag.lastTrackOffset = nextTrackOffset;
                drag.lastTrackId = nextTrackId;
                setClipDropNewTrack(
                    drag.allowDropToNewTrack && nextTrackId == null,
                );
            } else {
                drag.lastTrackOffset = 0;
                drag.lastTrackId = drag.initialAnchorTrackId;
                setClipDropNewTrack(false);
            }

            // ── 轴锁定：垂直跨轨道拖拽时，水平偏移小于阈值则冻结水平位移 ──
            const HORIZONTAL_LOCK_THRESHOLD = 30; // px
            const horizontalPx = Math.abs(ev.clientX - drag.startClientX);
            const isTrackChanging = drag.lastTrackOffset !== 0 ||
                (hoveredTrackId == null && drag.allowDropToNewTrack);
            if (isTrackChanging && horizontalPx < HORIZONTAL_LOCK_THRESHOLD) {
                deltaBeat = 0;
                drag.lastDeltaBeat = 0;
            }

            // copyMode 时不移动原 clip，只更新 ghost 预览位置
            if (drag.copyMode) {
                setGhostDrag({
                    clipIds: drag.clipIds,
                    initialById: drag.initialById,
                    deltaSec: deltaBeat,
                    targetTrackId: nextTrackId,
                    targetTrackOffset: drag.lastTrackOffset,
                    allowTrackMove: drag.allowTrackMove,
                });
            } else {
                batch(() => {
                    for (const id of drag.clipIds) {
                        const initial = drag.initialById[id];
                        if (!initial) continue;
                        dispatch(
                            moveClipStart({
                                clipId: id,
                                startSec: Math.max(
                                    0,
                                    initial.startSec + deltaBeat,
                                ),
                            }),
                        );
                        if (drag.allowTrackMove) {
                            const resolvedTrackId =
                                nextTrackId == null
                                    ? NEW_TRACK_SENTINEL
                                    : (resolveTrackIdByOffset(
                                          drag,
                                          id,
                                          drag.lastTrackOffset,
                                      ) ?? nextTrackId);
                            dispatch(
                                moveClipTrack({
                                    clipId: id,
                                    trackId: resolvedTrackId,
                                }),
                            );
                        }
                    }
                });
            }
        }

        function end() {
            const drag = clipDragRef.current;
            if (!drag || drag.pointerId !== e.pointerId) return;
            clipDragRef.current = null;
            setClipDropNewTrack(false);

            const maybeSelectTargetTrack = (targetTrackId: string | null) => {
                if (!targetTrackId) return;
                if (targetTrackId === drag.initialAnchorTrackId) return;
                if (sessionRef.current.selectedTrackId === targetTrackId)
                    return;
                void dispatch(selectTrackRemote(targetTrackId));
            };

            // 清除 ghost 预览
            setGhostDrag(null);

            if (!drag.hasMoved) {
                // Ctrl+点击（未移动）：执行多选切换
                if (drag.copyMode && onCtrlClick) {
                    onCtrlClick(drag.anchorClipId);
                }
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", end);
                window.removeEventListener("pointercancel", end);
                return;
            }

            const session = sessionRef.current;
            const dropToNewTrack =
                drag.allowTrackMove &&
                drag.allowDropToNewTrack &&
                drag.lastTrackId == null;

            async function createNewTrackForDrop(): Promise<string | null> {
                const before = new Set(
                    sessionRef.current.tracks.map((t) => t.id),
                );
                const res = (await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap()) as {
                    tracks?: Array<{ id?: string }>;
                    selected_track_id?: string | null;
                };
                const nextTracks = Array.isArray(res?.tracks) ? res.tracks : [];
                const created = nextTracks.find(
                    (t) => !before.has(String(t?.id)),
                );
                return (
                    (created && String(created.id)) ||
                    (res?.selected_track_id
                        ? String(res.selected_track_id)
                        : null)
                );
            }

            async function createNewTracksForDrop(
                count: number,
            ): Promise<string[]> {
                const createdIds: string[] = [];
                for (let i = 0; i < count; i += 1) {
                    const before = new Set(
                        sessionRef.current.tracks.map((t) => t.id),
                    );
                    const res = (await dispatch(
                        addTrackRemote({
                            name: undefined,
                            parentTrackId: null,
                        }),
                    ).unwrap()) as {
                        tracks?: Array<{ id?: string }>;
                        selected_track_id?: string | null;
                    };
                    const nextTracks = Array.isArray(res?.tracks)
                        ? res.tracks
                        : [];
                    const created = nextTracks.find(
                        (t) => !before.has(String(t?.id)),
                    );
                    const id =
                        (created && String(created.id)) ||
                        (res?.selected_track_id
                            ? String(res.selected_track_id)
                            : null) ||
                        nextTracks[nextTracks.length - 1]?.id ||
                        null;
                    if (id) createdIds.push(String(id));
                }
                return createdIds;
            }

            if (drag.copyMode) {
                // copyMode 下原 clip 未被移动，直接根据 ghost 偏移量计算副本位置
                // copyMode 不使用交互锁（原 clip 未被拖动改变位置）
                void (async () => {
                    const templateInputs = drag.clipIds
                        .map((id) => {
                            const initial = drag.initialById[id];
                            const now = sessionRef.current.clips.find(
                                (c) => c.id === id,
                            );
                            if (!initial || !now) return null;
                            return { id, initial, now };
                        })
                        .filter(
                            (
                                input,
                            ): input is {
                                id: string;
                                initial: { startSec: number; trackId: string };
                                now: (typeof sessionRef.current.clips)[number];
                            } => input != null,
                        );

                    const linkedParamsResults = await Promise.all(
                        templateInputs.map((input) =>
                            webApi.getClipLinkedParams(input.id),
                        ),
                    );

                    const templates: ClipTemplate[] = templateInputs.map(
                        (input, index) => {
                            const { initial, now } = input;
                            const targetTrackId = drag.allowTrackMove
                                ? drag.lastTrackId == null
                                    ? null
                                    : resolveTrackIdByOffset(
                                          drag,
                                          input.id,
                                          drag.lastTrackOffset,
                                      )
                                : initial.trackId;
                            const linkedParamsResult =
                                linkedParamsResults[index];
                            return {
                                trackId: targetTrackId ?? initial.trackId,
                                name: String(now.name),
                                startSec: Math.max(
                                    0,
                                    initial.startSec + drag.lastDeltaBeat,
                                ),
                                lengthSec: Number(now.lengthSec),
                                sourcePath: now.sourcePath,
                                durationSec: now.durationSec,
                                gain: Number(now.gain ?? 1) || 1,
                                muted: Boolean(now.muted),
                                sourceStartSec:
                                    Number(now.sourceStartSec ?? 0) || 0,
                                sourceEndSec:
                                    Number(now.sourceEndSec ?? 0) || 0,
                                playbackRate:
                                    Number(now.playbackRate ?? 1) || 1,
                                fadeInSec: Number(now.fadeInSec ?? 0) || 0,
                                fadeOutSec: Number(now.fadeOutSec ?? 0) || 0,
                                fadeInCurve: now.fadeInCurve,
                                fadeOutCurve: now.fadeOutCurve,
                                linkedParams: linkedParamsResult.ok
                                    ? linkedParamsResult.linkedParams
                                    : undefined,
                            };
                        },
                    );

                    if (templates.length === 0) {
                        return;
                    }
                    dispatch(checkpointHistory());
                    void (async () => {
                        // Begin backend undo group for copy-drag + auto-crossfade
                        await webApi.beginUndoGroup();
                        try {
                            if (dropToNewTrack) {
                                if (drag.hasMixedTrackSelection) {
                                    // mixed selection: create multiple new tracks matching source span
                                    const idxs = Object.values(
                                        drag.initialTrackIndexById,
                                    );
                                    const minIdx = Math.min(...idxs);
                                    const maxIdx = Math.max(...idxs);
                                    const span = maxIdx - minIdx + 1;
                                    const created =
                                        await createNewTracksForDrop(span);
                                    if (created.length === span) {
                                        for (const tpl of templates) {
                                            const srcIdx =
                                                drag.initialTrackIndexById[
                                                    tpl.trackId
                                                ] ?? null;
                                            if (srcIdx == null) continue;
                                            const offset = srcIdx - minIdx;
                                            tpl.trackId =
                                                created[offset] ?? tpl.trackId;
                                        }
                                        maybeSelectTargetTrack(
                                            created[0] ?? null,
                                        );
                                    } else {
                                        // fallback to single new track
                                        const newTrackId =
                                            await createNewTrackForDrop();
                                        if (newTrackId) {
                                            for (const tpl of templates)
                                                tpl.trackId = newTrackId;
                                            maybeSelectTargetTrack(newTrackId);
                                        }
                                    }
                                } else {
                                    const newTrackId =
                                        await createNewTrackForDrop();
                                    if (newTrackId) {
                                        for (const tpl of templates) {
                                            tpl.trackId = newTrackId;
                                        }
                                        maybeSelectTargetTrack(newTrackId);
                                    }
                                }
                            } else {
                                maybeSelectTargetTrack(
                                    drag.lastTrackId ?? null,
                                );
                            }
                            const payload = await dispatch(
                                createClipsRemote({ templates }),
                            ).unwrap();
                            const created: string[] =
                                payload?.createdClipIds ?? [];
                            if (!Array.isArray(created) || created.length === 0)
                                return;
                            setMultiSelectedClipIds(created);
                            void dispatch(selectClipRemote(created[0]));
                            // 复制拖动后，将播放光标定位到目标时间点（所有副本中最靠前的起始位置）
                            const targetStartSec = templates.reduce(
                                (min, t) => Math.min(min, t.startSec),
                                Infinity,
                            );
                            if (Number.isFinite(targetStartSec)) {
                                dispatch(setplayheadSec(targetStartSec));
                                void dispatch(seekPlayhead(targetStartSec));
                            }
                            // 复制拖动后，尝试对新创建的 clip 应用自动交叉淡化
                            if (autoCrossfadeEnabled) {
                                const allClips = (payload?.clips ??
                                    []) as Array<{
                                    id?: string;
                                    track_id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                    fade_in_sec?: number;
                                    fade_out_sec?: number;
                                }>;
                                const fadeUpdates =
                                    computeAutoCrossfadeFromPayload(
                                        allClips,
                                        created,
                                    );
                                if (fadeUpdates.length > 0) {
                                    const fadePromises = fadeUpdates.map((u) =>
                                        dispatch(
                                            setClipStateRemote({
                                                clipId: u.clipId,
                                                fadeInSec: u.fadeInSec,
                                                fadeOutSec: u.fadeOutSec,
                                            }),
                                        ).unwrap(),
                                    );
                                    await Promise.allSettled(fadePromises);
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
                                const idxs = Object.values(
                                    drag.initialTrackIndexById,
                                );
                                const minIdx = Math.min(...idxs);
                                const maxIdx = Math.max(...idxs);
                                const span = maxIdx - minIdx + 1;
                                const created =
                                    await createNewTracksForDrop(span);
                                if (created.length !== span)
                                    throw new Error("create_track_failed");
                                maybeSelectTargetTrack(created[0] ?? null);
                                const moves = drag.clipIds
                                    .map((id) => {
                                        const initial = drag.initialById[id];
                                        const now =
                                            sessionRef.current.clips.find(
                                                (c) => c.id === id,
                                            );
                                        if (!initial || !now) return null;
                                        const srcIdx =
                                            drag.initialTrackIndexById[
                                                initial.trackId
                                            ];
                                        const offset = srcIdx - minIdx;
                                        const targetTrack = created[offset];
                                        return targetTrack
                                            ? {
                                                  clipId: id,
                                                  startSec: Number(
                                                      now.startSec,
                                                  ),
                                                  trackId: targetTrack,
                                              }
                                            : null;
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
                                if (moves.length > 1) {
                                    await dispatch(
                                        moveClipsRemote({
                                            moves,
                                            moveLinkedParams:
                                                sessionRef.current
                                                    .lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                } else if (moves.length === 1) {
                                    await dispatch(
                                        moveClipRemote({
                                            clipId: moves[0].clipId,
                                            startSec: moves[0].startSec,
                                            trackId: moves[0].trackId,
                                            moveLinkedParams:
                                                sessionRef.current
                                                    .lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                }
                            } else {
                                const newTrackId =
                                    await createNewTrackForDrop();
                                if (!newTrackId)
                                    throw new Error("create_track_failed");
                                maybeSelectTargetTrack(newTrackId);
                                const moves = drag.clipIds
                                    .map((id) => {
                                        const initial = drag.initialById[id];
                                        const now =
                                            sessionRef.current.clips.find(
                                                (c) => c.id === id,
                                            );
                                        if (!initial || !now) return null;
                                        return {
                                            clipId: id,
                                            startSec: Number(now.startSec),
                                            trackId: newTrackId,
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
                                if (moves.length > 1) {
                                    await dispatch(
                                        moveClipsRemote({
                                            moves,
                                            moveLinkedParams:
                                                sessionRef.current
                                                    .lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                } else if (moves.length === 1) {
                                    await dispatch(
                                        moveClipRemote({
                                            clipId: moves[0].clipId,
                                            startSec: moves[0].startSec,
                                            trackId: moves[0].trackId,
                                            moveLinkedParams:
                                                sessionRef.current
                                                    .lockParamLinesEnabled,
                                        }),
                                    ).unwrap();
                                }
                            }
                            if (autoCrossfadeEnabled) {
                                await new Promise((r) => setTimeout(r, 0));
                                const latestSession = sessionRef.current;
                                applyAutoCrossfade(
                                    latestSession,
                                    drag.clipIds,
                                    dispatch,
                                );
                            }
                        } catch {
                            for (const id of drag.clipIds) {
                                const initial = drag.initialById[id];
                                if (!initial) continue;
                                dispatch(
                                    moveClipTrack({
                                        clipId: id,
                                        trackId: initial.trackId,
                                    }),
                                );
                            }
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
                        const now = session.clips.find((c) => c.id === id);
                        if (!initial || !now) return null;
                        const changedBeat =
                            Math.abs(Number(now.startSec) - initial.startSec) >
                            1e-6;
                        const changedTrack =
                            String(now.trackId) !== initial.trackId;
                        if (!changedBeat && !changedTrack) return null;
                        return {
                            clipId: id,
                            startSec: Number(now.startSec),
                            trackId: String(now.trackId),
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
                    const movedIds = drag.clipIds;
                    const movePromise =
                        moves.length > 1
                            ? dispatch(
                                  moveClipsRemote({
                                      moves,
                                      moveLinkedParams:
                                          sessionRef.current
                                              .lockParamLinesEnabled,
                                  }),
                              ).unwrap()
                            : dispatch(
                                  moveClipRemote({
                                      clipId: moves[0].clipId,
                                      startSec: moves[0].startSec,
                                      trackId: moves[0].trackId,
                                      moveLinkedParams:
                                          sessionRef.current
                                              .lockParamLinesEnabled,
                                  }),
                              ).unwrap();
                    void Promise.resolve(movePromise).finally(() => {
                        if (autoCrossfadeEnabled) {
                            const latestSession = sessionRef.current;
                            applyAutoCrossfade(
                                latestSession,
                                movedIds,
                                dispatch,
                            );
                        }
                        void webApi.endUndoGroup();
                        dispatch(endInteraction());
                    });
                } else {
                    if (autoCrossfadeEnabled) {
                        applyAutoCrossfade(session, drag.clipIds, dispatch);
                    }
                    void webApi.endUndoGroup();
                    dispatch(endInteraction());
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

    return { clipDragRef, startClipDrag, ghostDrag };
}
