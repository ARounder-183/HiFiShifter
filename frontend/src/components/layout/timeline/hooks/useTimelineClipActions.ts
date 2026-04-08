/**
 * useTimelineClipActions — Clip 多选管理 + 操作回调
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - multiSelectedClipIds 管理（Redux ↔ local ref）
 * - contextMenu / trackAreaMenu / importModeMenu / renamingClipId 状态
 * - selectionRect hook 桥接
 * - clipboard (copy / cut / paste)
 * - normalizeClips / replaceClipSources / splitClips / glueClips
 * - TrackLane 操作回调（ensureSelected, selectClip, toggleMuted, rename, gain ...）
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { AppDispatch, RootState } from "../../../../app/store";
import { useAppSelector } from "../../../../app/hooks";
import { useI18n } from "../../../../i18n/I18nProvider";
import type { ClipTemplate } from "../../../../features/session/sessionTypes";
import {
    checkpointHistory,
    createClipsRemote,
    seekPlayhead,
    selectClipRemote,
    setClipGain,
    setClipMuted,
    setClipStateRemote,
    setMultiSelectedClipIds as setMultiSelectedClipIdsAction,
    setplayheadSec,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    replaceClipSourceRemote,
    splitClipRemote,
} from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";
import { waveformMipmapStore } from "../../../../utils/waveformMipmapStore";
import { dbToGain } from "../math";
import { computeAutoCrossfadeFromPayload } from "./autoCrossfade";
import { useTimelineSelectionRect } from "../";
import { readSystemClipboardObject } from "../../../../utils/systemClipboard";

// ── Args / Result 类型 ────────────────────────────────────────

export interface UseTimelineClipActionsArgs {
    sessionRef: React.MutableRefObject<RootState["session"]>;
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    lastClickedClipIdRef: React.MutableRefObject<string | null>;
    pxPerSec: number;
    pxPerBeat: number;
    rowHeight: number;
}

export interface UseTimelineClipActionsResult {
    // Multi-select
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
    multiSelectedClipIdsRef: React.MutableRefObject<string[]>;
    multiSelectedSetRef: React.MutableRefObject<Set<string>>;
    setMultiSelectedClipIds: (ids: string[] | ((prev: string[]) => string[])) => void;

    // Context menus
    contextMenu: {
        x: number;
        y: number;
        clipId: string;
        overlappingClipIds?: string[];
    } | null;
    setContextMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            clipId: string;
            overlappingClipIds?: string[];
        } | null>
    >;
    trackAreaMenu: {
        x: number;
        y: number;
        trackId: string;
    } | null;
    setTrackAreaMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            trackId: string;
        } | null>
    >;
    importModeMenu: {
        x: number;
        y: number;
        audioPaths: string[];
        trackId: string | null;
        startSec: number;
    } | null;
    setImportModeMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            audioPaths: string[];
            trackId: string | null;
            startSec: number;
        } | null>
    >;
    renamingClipId: string | null;
    setRenamingClipId: React.Dispatch<React.SetStateAction<string | null>>;

    // Selection rect
    selectionRect: {
        x1: number;
        y1: number;
        x2: number;
        y2: number;
    } | null;
    onSelectionRectPointerDown: (e: React.PointerEvent<HTMLDivElement>) => void;

    // Clipboard
    clipClipboardRef: React.MutableRefObject<ClipTemplate[] | null>;
    buildClipClipboardTemplates: (ids: string[]) => Promise<ClipTemplate[]>;

    // Clip operations
    normalizeClips: (ids: string[]) => void;
    replaceClipSources: (ids: string[]) => Promise<void>;
    splitClipIdsAtPlayhead: (clipIds: string[]) => string[];
    splitSelectedAtPlayhead: () => void;
    selectClipRangeByRect: (targetClipId: string, anchorClipIdOverride?: string | null) => void;
    rangeSelectAnchorClipId: string | null;
    pasteClipsAtPlayhead: () => void;
    clearContextMenu: () => void;

    // TrackLane callbacks
    ensureTrackLaneSelected: (clipId: string) => void;
    selectTrackLaneClipRemote: (clipId: string) => void;
    openTrackLaneContextMenu: (clipId: string, clientX: number, clientY: number) => void;
    seekFromTrackLaneClientX: (clientX: number, commit: boolean) => void;
    toggleTrackLaneClipMuted: (clipId: string, nextMuted: boolean) => void;
    toggleTrackLaneCtrlSelection: (clipId: string) => void;
    toggleTrackLaneMultiSelect: (clipId: string) => void;
    commitTrackLaneRename: (clipId: string, newName: string) => void;
    handleTrackLaneRenameDone: () => void;
    commitTrackLaneGain: (clipId: string, db: number) => void;

    // sameSourceConfirm helpers (forwarded from state)
    sameSourceConfirmResolverRef: React.MutableRefObject<((confirmed: boolean) => void) | null>;
}

// ── Hook 实现 ─────────────────────────────────────────────────

export function useTimelineClipActions(
    args: UseTimelineClipActionsArgs & {
        dispatch: AppDispatch;
        sameSourceConfirmResolverRef: React.MutableRefObject<((confirmed: boolean) => void) | null>;
        setSameSourceConfirmOpen: React.Dispatch<React.SetStateAction<boolean>>;
        setPlayheadFromClientX: (
            clientX: number,
            bounds: DOMRect,
            xScroll: number,
            commit: boolean,
        ) => number;
    },
): UseTimelineClipActionsResult {
    const {
        sessionRef,
        scrollRef,
        lastClickedClipIdRef,
        pxPerSec,
        rowHeight,
        dispatch,
        sameSourceConfirmResolverRef,
        setSameSourceConfirmOpen,
        setPlayheadFromClientX,
    } = args;

    const { t } = useI18n();

    // ── multiSelectedClipIds ─────────────────────────────────
    const multiSelectedClipIds = useAppSelector(
        (state: RootState) => state.session.multiSelectedClipIds,
    );
    const multiSelectedClipIdsRef = useRef(multiSelectedClipIds);
    useEffect(() => {
        multiSelectedClipIdsRef.current = multiSelectedClipIds;
    }, [multiSelectedClipIds]);

    const setMultiSelectedClipIds = React.useCallback(
        (ids: string[] | ((prev: string[]) => string[])) => {
            if (typeof ids === "function") {
                const next = ids(multiSelectedClipIdsRef.current);
                dispatch(setMultiSelectedClipIdsAction(next));
            } else {
                dispatch(setMultiSelectedClipIdsAction(ids));
            }
        },
        [dispatch],
    );

    // 切换工具时清除多选
    const s = useAppSelector((state: RootState) => state.session);
    useEffect(() => {
        dispatch(setMultiSelectedClipIdsAction([]));
    }, [s.toolMode, dispatch]);

    const multiSelectedSet = useMemo(() => new Set(multiSelectedClipIds), [multiSelectedClipIds]);
    const multiSelectedSetRef = useRef(multiSelectedSet);
    useEffect(() => {
        multiSelectedSetRef.current = multiSelectedSet;
    }, [multiSelectedSet]);

    // ── Context menus ────────────────────────────────────────
    const [contextMenu, setContextMenu] = useState<{
        x: number;
        y: number;
        clipId: string;
        overlappingClipIds?: string[];
    } | null>(null);
    const [trackAreaMenu, setTrackAreaMenu] = useState<{
        x: number;
        y: number;
        trackId: string;
    } | null>(null);
    const [importModeMenu, setImportModeMenu] = useState<{
        x: number;
        y: number;
        audioPaths: string[];
        trackId: string | null;
        startSec: number;
    } | null>(null);
    const [renamingClipId, setRenamingClipId] = useState<string | null>(null);

    const clearContextMenu = React.useCallback(() => {
        setContextMenu(null);
    }, []);

    // ── Selection rect ───────────────────────────────────────
    const handleSelectionRectSingleSelect = React.useCallback(
        (clipId: string) => {
            void dispatch(selectClipRemote(clipId));
        },
        [dispatch],
    );

    const { selectionRect, onPointerDown: onSelectionRectPointerDown } = useTimelineSelectionRect({
        scrollRef,
        sessionRef,
        pxPerBeat: pxPerSec,
        rowHeight,
        clearContextMenu,
        setMultiSelectedClipIds,
        onSingleSelect: handleSelectionRectSingleSelect,
    });

    // ── Clipboard ────────────────────────────────────────────
    const clipClipboardRef = useRef<ClipTemplate[] | null>(null);

    const buildClipClipboardTemplates = React.useCallback(async (ids: string[]) => {
        const clips = sessionRef.current.clips.filter((c) => ids.includes(c.id));
        return Promise.all(
            clips.map(async (clip) => {
                const linkedParamsResult = await webApi.getClipLinkedParams(clip.id);
                return {
                    ...clip,
                    waveformPreview: sessionRef.current.clipWaveforms[clip.id],
                    linkedParams: linkedParamsResult.ok
                        ? linkedParamsResult.linkedParams
                        : undefined,
                };
            }),
        );
    }, []);

    // ── normalizeClips ───────────────────────────────────────
    const normalizeClips = React.useCallback(
        (ids: string[]) => {
            for (const id of ids) {
                const clip = sessionRef.current.clips.find((c) => c.id === id);
                if (!clip?.sourcePath || !clip.durationSec || clip.durationSec <= 0) {
                    continue;
                }
                const sourceStartSec = Number(clip.sourceStartSec ?? 0) || 0;
                const sourceEndSec =
                    Number(clip.sourceEndSec ?? clip.durationSec) || clip.durationSec;
                const playbackRate = Math.max(1e-6, Number(clip.playbackRate ?? 1) || 1);
                const clipSourceSpanSec = Math.max(
                    0,
                    Math.min(clip.lengthSec * playbackRate, sourceEndSec - sourceStartSec),
                );
                if (clipSourceSpanSec <= 0) continue;

                const slice = waveformMipmapStore.getInterleavedSlice(
                    clip.sourcePath,
                    0,
                    sourceStartSec,
                    clipSourceSpanSec,
                );
                if (!slice || slice.interleaved.length < 2) continue;
                const data = slice.interleaved;
                let peak = 0;
                for (let i = 0; i < data.length; i++) {
                    const v = Math.abs(data[i]);
                    if (v > peak) peak = v;
                }
                waveformMipmapStore.releaseInterleaved(data);
                if (peak <= 0) continue;
                const newGain = Math.min(Math.max(1.0 / peak, dbToGain(-12)), dbToGain(12));
                dispatch(setClipGain({ clipId: id, gain: newGain }));
                void dispatch(setClipStateRemote({ clipId: id, gain: newGain }));
            }
        },
        [dispatch, sessionRef],
    );

    // ── replaceClipSources ───────────────────────────────────
    const replaceClipSources = React.useCallback(
        async (ids: string[]) => {
            const selected = sessionRef.current.clips.filter((c) => ids.includes(c.id));
            if (selected.length === 0) return;

            const picked = await webApi.openAudioDialog();
            if (!picked.ok || picked.canceled || !picked.path) return;

            const selectedSourcePaths = new Set(
                selected
                    .map((c) => c.sourcePath)
                    .filter((v): v is string => Boolean(v && v.trim().length)),
            );

            let replaceSameSource = false;
            if (selectedSourcePaths.size > 0) {
                const hasOtherClipsWithSameSource = sessionRef.current.clips.some(
                    (clip) =>
                        !ids.includes(clip.id) &&
                        Boolean(clip.sourcePath && selectedSourcePaths.has(clip.sourcePath)),
                );
                if (hasOtherClipsWithSameSource) {
                    replaceSameSource = await new Promise<boolean>((resolve) => {
                        sameSourceConfirmResolverRef.current = resolve;
                        setSameSourceConfirmOpen(true);
                    });
                }
            }

            await dispatch(
                replaceClipSourceRemote({
                    clipIds: ids,
                    newSourcePath: picked.path,
                    replaceSameSource,
                }),
            );
        },
        [dispatch, t],
    );

    // ── splitClipIdsAtPlayhead ────────────────────────────────
    const splitClipIdsAtPlayhead = React.useCallback(
        (clipIds: string[]) => {
            const splitSec = Math.max(0, Number(sessionRef.current.playheadSec ?? 0) || 0);
            const eligibleIds = clipIds.filter((id) => {
                const c = sessionRef.current.clips.find((clip) => clip.id === id);
                if (!c) return false;
                return splitSec >= c.startSec && splitSec <= c.startSec + c.lengthSec;
            });
            for (const clipId of eligibleIds) {
                void dispatch(splitClipRemote({ clipId, splitSec }));
            }
            return eligibleIds;
        },
        [dispatch],
    );

    const splitSelectedAtPlayhead = React.useCallback(() => {
        const selectedIds =
            multiSelectedClipIdsRef.current.length > 0
                ? [...multiSelectedClipIdsRef.current]
                : sessionRef.current.selectedClipId
                  ? [sessionRef.current.selectedClipId]
                  : [];
        if (selectedIds.length === 0) return;
        splitClipIdsAtPlayhead(selectedIds);
    }, [splitClipIdsAtPlayhead]);

    // ── selectClipRangeByRect ────────────────────────────────
    const selectClipRangeByRect = React.useCallback(
        (targetClipId: string, anchorClipIdOverride?: string | null) => {
            const session = sessionRef.current;
            const target = session.clips.find((c) => c.id === targetClipId);
            if (!target) return;

            const anchorId =
                anchorClipIdOverride ??
                lastClickedClipIdRef.current ??
                session.selectedClipId ??
                targetClipId;
            const anchor = session.clips.find((c) => c.id === anchorId) ?? target;

            const trackIndexById = new Map(session.tracks.map((track, index) => [track.id, index]));
            const anchorTrackIndex = trackIndexById.get(anchor.trackId);
            const targetTrackIndex = trackIndexById.get(target.trackId);
            if (anchorTrackIndex == null || targetTrackIndex == null) {
                setMultiSelectedClipIds([targetClipId]);
                dispatch(setSelectedClip(targetClipId));
                lastClickedClipIdRef.current = targetClipId;
                return;
            }

            const minTrack = Math.min(anchorTrackIndex, targetTrackIndex);
            const maxTrack = Math.max(anchorTrackIndex, targetTrackIndex);
            const minStartSec = Math.min(anchor.startSec, target.startSec);
            const maxEndSec = Math.max(
                anchor.startSec + anchor.lengthSec,
                target.startSec + target.lengthSec,
            );

            const selected = session.clips
                .filter((clip) => {
                    const trackIndex = trackIndexById.get(clip.trackId);
                    if (trackIndex == null || trackIndex < minTrack || trackIndex > maxTrack) {
                        return false;
                    }
                    const clipStart = clip.startSec;
                    const clipEnd = clip.startSec + clip.lengthSec;
                    return clipEnd >= minStartSec && clipStart <= maxEndSec;
                })
                .map((clip) => clip.id);

            const next = selected.length > 0 ? selected : [targetClipId];
            setMultiSelectedClipIds(next);
            dispatch(setSelectedClip(targetClipId));
            lastClickedClipIdRef.current = targetClipId;
        },
        [dispatch, setMultiSelectedClipIds],
    );

    // ── pasteClipsAtPlayhead ─────────────────────────────────
    const pasteClipsAtPlayhead = React.useCallback(() => {
        void (async () => {
            let tpl = clipClipboardRef.current;
            try {
                const fromSystem = await readSystemClipboardObject("clip");
                if (fromSystem?.kind === "clip" && Array.isArray(fromSystem.templates)) {
                    tpl = fromSystem.templates;
                    clipClipboardRef.current = fromSystem.templates;
                }
            } catch {
                // ignore and fallback to internal clipboard
            }
            if (!tpl || tpl.length === 0) return;

            const playhead = sessionRef.current.playheadSec ?? 0;
            const minStart = tpl
                .map((c) => c.startSec)
                .reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
            const delta =
                Number.isFinite(minStart) && minStart !== Number.POSITIVE_INFINITY
                    ? playhead - minStart
                    : 0;
            const templates = tpl.map((c) => ({
                ...c,
                startSec: Math.max(0, c.startSec + delta),
            }));

            dispatch(checkpointHistory());
            await webApi.beginUndoGroup();
            try {
                const payload = await dispatch(
                    createClipsRemote({
                        templates,
                        options: { placeOnSelectedTrack: true },
                    }),
                ).unwrap();
                const created: string[] = payload?.createdClipIds ?? [];
                if (!Array.isArray(created) || created.length === 0) return;

                setMultiSelectedClipIds(created);
                void dispatch(selectClipRemote(created[0]));
                const targetStartSec = templates.reduce(
                    (min, t) => Math.min(min, t.startSec),
                    Number.POSITIVE_INFINITY,
                );
                if (Number.isFinite(targetStartSec)) {
                    dispatch(setplayheadSec(targetStartSec));
                    void dispatch(seekPlayhead(targetStartSec));
                }

                if (sessionRef.current.autoCrossfadeEnabled) {
                    const allClips = (payload?.clips ?? []) as Array<{
                        id?: string;
                        track_id?: string;
                        start_sec?: number;
                        length_sec?: number;
                        fade_in_sec?: number;
                        fade_out_sec?: number;
                    }>;
                    const fadeUpdates = computeAutoCrossfadeFromPayload(allClips, created);
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
    }, [dispatch, setMultiSelectedClipIds]);

    // ── TrackLane callbacks ───────────────────────────────────
    const ensureTrackLaneSelected = React.useCallback(
        (clipId: string) => {
            lastClickedClipIdRef.current = clipId;
            const selectedIds = multiSelectedClipIdsRef.current;
            const selectedSet = multiSelectedSetRef.current;
            if (selectedIds.length === 0 || !selectedSet.has(clipId)) {
                setMultiSelectedClipIds([clipId]);
            }
        },
        [setMultiSelectedClipIds],
    );

    const selectTrackLaneClipRemote = React.useCallback(
        (clipId: string) => {
            lastClickedClipIdRef.current = clipId;
            const clip = sessionRef.current.clips.find((entry) => entry.id === clipId);
            const clipTrackId = clip?.trackId ?? null;
            if (
                sessionRef.current.selectedClipId === clipId &&
                clipTrackId != null &&
                clipTrackId === sessionRef.current.selectedTrackId
            ) {
                return;
            }
            const preserveTrackFocus = Boolean(
                clip && clip.trackId === sessionRef.current.selectedTrackId,
            );
            void dispatch(
                selectClipRemote({
                    clipId,
                    preserveTrackFocus,
                }),
            );
        },
        [dispatch],
    );

    const toggleTrackLaneCtrlSelection = React.useCallback(
        (clipId: string) => {
            lastClickedClipIdRef.current = clipId;

            const currentSelectionIds =
                multiSelectedClipIdsRef.current.length > 0
                    ? [...multiSelectedClipIdsRef.current]
                    : sessionRef.current.selectedClipId
                      ? [sessionRef.current.selectedClipId]
                      : [];

            const alreadySelected = currentSelectionIds.includes(clipId);
            const nextSelectionIds = alreadySelected
                ? currentSelectionIds.filter((id) => id !== clipId)
                : [...currentSelectionIds, clipId];

            setMultiSelectedClipIds(nextSelectionIds);

            if (nextSelectionIds.length === 0) {
                dispatch(setSelectedClipPreservingTrack(null));
                return;
            }

            const nextPrimaryClipId = alreadySelected
                ? (nextSelectionIds[nextSelectionIds.length - 1] ?? null)
                : clipId;
            if (!nextPrimaryClipId) {
                dispatch(setSelectedClipPreservingTrack(null));
                return;
            }

            const nextPrimaryClip = sessionRef.current.clips.find(
                (entry) => entry.id === nextPrimaryClipId,
            );
            const preserveTrackFocus = Boolean(
                nextPrimaryClip && nextPrimaryClip.trackId === sessionRef.current.selectedTrackId,
            );

            void dispatch(
                selectClipRemote({
                    clipId: nextPrimaryClipId,
                    preserveTrackFocus,
                }),
            );
        },
        [dispatch, setMultiSelectedClipIds],
    );

    const rangeSelectAnchorClipId =
        lastClickedClipIdRef.current ?? sessionRef.current.selectedClipId ?? null;

    const openTrackLaneContextMenu = React.useCallback(
        (clipId: string, clientX: number, clientY: number) => {
            setTrackAreaMenu(null);
            setContextMenu({
                x: clientX,
                y: clientY,
                clipId,
            });
        },
        [],
    );

    const seekFromTrackLaneClientX = React.useCallback(
        (clientX: number, commit: boolean) => {
            const scroller = scrollRef.current;
            if (!scroller) return;
            const bounds = scroller.getBoundingClientRect();
            setPlayheadFromClientX(clientX, bounds, scroller.scrollLeft, commit);
        },
        [setPlayheadFromClientX],
    );

    const toggleTrackLaneClipMuted = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            dispatch(
                setClipMuted({
                    clipId,
                    muted: nextMuted,
                }),
            );
            void dispatch(
                setClipStateRemote({
                    clipId,
                    muted: nextMuted,
                }),
            );
        },
        [dispatch],
    );

    const toggleTrackLaneMultiSelect = React.useCallback(
        (clipId: string) => {
            setMultiSelectedClipIds((prev) => {
                if (prev.includes(clipId)) {
                    return prev.filter((id) => id !== clipId);
                }
                return [...prev, clipId];
            });
        },
        [setMultiSelectedClipIds],
    );

    const commitTrackLaneRename = React.useCallback(
        (clipId: string, newName: string) => {
            void dispatch(
                setClipStateRemote({
                    clipId,
                    name: newName,
                }),
            );
        },
        [dispatch],
    );

    const handleTrackLaneRenameDone = React.useCallback(() => {
        setRenamingClipId(null);
    }, []);

    const commitTrackLaneGain = React.useCallback(
        (clipId: string, db: number) => {
            const gain = Math.pow(10, db / 20);
            void dispatch(
                setClipStateRemote({
                    clipId,
                    gain,
                }),
            );
        },
        [dispatch],
    );

    // ── Return ───────────────────────────────────────────────
    return {
        multiSelectedClipIds,
        multiSelectedSet,
        multiSelectedClipIdsRef,
        multiSelectedSetRef,
        setMultiSelectedClipIds,

        contextMenu,
        setContextMenu,
        trackAreaMenu,
        setTrackAreaMenu,
        importModeMenu,
        setImportModeMenu,
        renamingClipId,
        setRenamingClipId,

        selectionRect,
        onSelectionRectPointerDown,

        clipClipboardRef,
        buildClipClipboardTemplates,

        normalizeClips,
        replaceClipSources,
        splitClipIdsAtPlayhead,
        splitSelectedAtPlayhead,
        selectClipRangeByRect,
        rangeSelectAnchorClipId,
        pasteClipsAtPlayhead,
        clearContextMenu,

        ensureTrackLaneSelected,
        selectTrackLaneClipRemote,
        openTrackLaneContextMenu,
        seekFromTrackLaneClientX,
        toggleTrackLaneClipMuted,
        toggleTrackLaneCtrlSelection,
        toggleTrackLaneMultiSelect,
        commitTrackLaneRename,
        handleTrackLaneRenameDone,
        commitTrackLaneGain,

        sameSourceConfirmResolverRef,
    };
}
