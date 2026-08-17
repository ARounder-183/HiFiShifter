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
import {
    pasteTimelineClipboardRemote,
    removeClipsRemote,
    removeTrackRemote,
    seekPlayhead,
    selectClipRemote,
    setClipGain,
    setClipMuted,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setMultiSelectedClipIds as setMultiSelectedClipIdsAction,
    setplayheadSec,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    replaceClipSourceRemote,
    splitClipsAtRemote,
} from "../../../../features/session/sessionSlice";
import {
    groupClipsRemote,
    ungroupClipsRemote,
    toggleGroupDisabledRemote,
} from "../../../../features/session/thunks/timelineThunks";
import { webApi } from "../../../../services/webviewApi";
import { waveformMipmapStore } from "../../../../utils/waveformMipmapStore";
import { snapTimelinePosition } from "../../../../utils/timelineSnapping";
import { computeAutoCrossfadeFromPayload } from "./autoCrossfade";
import { useTimelineSelectionRect } from "../";
import { getBulkEditableClipIds } from "./bulkClipEdit";
import { getGroupClipIds } from "./useGroupExpansion";
import { buildBulkClipStateUpdates } from "./bulkClipRemotePayloads";
import { computeClipNormalizationGain } from "../../../../features/session/clipNormalization";

// ── Args / Result 类型 ────────────────────────────────────────

export interface UseTimelineClipActionsArgs {
    sessionRef: React.MutableRefObject<RootState["session"]>;
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    lastClickedClipIdRef: React.MutableRefObject<string | null>;
    lastClickedClientXRef: React.MutableRefObject<number | null>;
    pxPerSec: number;
    pxPerBeat: number;
    rowHeight: number;
    ignoreGrouping: boolean;
    disabledGroupIds: string[];
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
    clipboardAvailable: boolean;
    copyClips: (ids: string[]) => Promise<boolean>;
    cutClips: (ids: string[]) => void;
    copyTracks: (ids: string[]) => Promise<boolean>;
    cutTracks: (ids: string[]) => void;
    copyClipsToReaper: (ids: string[]) => Promise<boolean>;

    // Clip operations
    groupClips: (ids: string[]) => void;
    ungroupClips: (ids: string[]) => void;
    toggleGroupDisabled: (groupId: string) => void;
    normalizeClips: (ids: string[]) => void;
    replaceClipSources: (ids: string[]) => Promise<void>;
    splitClipIdsAtPlayhead: (clipIds: string[]) => string[];
    splitSelectedAtPlayhead: () => void;
    selectClipRangeByRect: (
        targetClipId: string,
        anchorClipIdOverride?: string | null,
        targetClientX?: number,
    ) => void;
    rangeSelectAnchorClipId: string | null;
    recordLastClickPosition: (clientX: number) => void;
    pasteClipsAtPlayhead: (mode?: "selected" | "new_tracks") => void;
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
        lastClickedClientXRef,
        pxPerSec,
        rowHeight,
        dispatch,
        sameSourceConfirmResolverRef,
        setSameSourceConfirmOpen,
        setPlayheadFromClientX,
        ignoreGrouping,
        disabledGroupIds,
    } = args;

    const { t } = useI18n();

    // ── multiSelectedClipIds ─────────────────────────────────
    const multiSelectedClipIds = useAppSelector(
        (state: RootState) => state.session.multiSelectedClipIds,
    );
    const selectedClipId = useAppSelector((state: RootState) => state.session.selectedClipId);
    const [rangeSelectAnchorClipIdState, setRangeSelectAnchorClipIdState] = useState<
        string | null
    >(null);
    const updateRangeSelectAnchor = React.useCallback(
        (clipId: string | null) => {
            lastClickedClipIdRef.current = clipId;
            setRangeSelectAnchorClipIdState(clipId);
        },
        [lastClickedClipIdRef],
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
    const toolMode = useAppSelector((state: RootState) => state.session.toolMode);
    useEffect(() => {
        dispatch(setMultiSelectedClipIdsAction([]));
    }, [toolMode, dispatch]);

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

    // ── Group / Ungroup ───────────────────────────────────────
    const groupClips = React.useCallback(
        (ids: string[]) => {
            if (ids.length < 2) return;
            void dispatch(groupClipsRemote(ids));
        },
        [dispatch],
    );

    const ungroupClips = React.useCallback(
        (ids: string[]) => {
            void dispatch(ungroupClipsRemote(ids));
        },
        [dispatch],
    );

    const toggleGroupDisabled = React.useCallback(
        (groupId: string) => {
            void dispatch(toggleGroupDisabledRemote(groupId));
        },
        [dispatch],
    );

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
    const [clipboardAvailable, setClipboardAvailable] = useState(false);

    useEffect(() => {
        let cancelled = false;
        const refresh = () => {
            void webApi
                .hasTimelineClipboard()
                .then((result) => {
                    if (!cancelled) setClipboardAvailable(Boolean(result?.ok && result?.available));
                })
                .catch(() => {
                    if (!cancelled) setClipboardAvailable(false);
                });
        };
        refresh();
        const timer = window.setInterval(refresh, 2000);
        return () => {
            cancelled = true;
            window.clearInterval(timer);
        };
    }, []);

    const copyClips = React.useCallback(async (ids: string[]) => {
        if (ids.length === 0) return false;
        try {
            const result = await webApi.copyTimelineClips(ids);
            setClipboardAvailable(Boolean(result?.ok));
            return Boolean(result?.ok);
        } catch {
            setClipboardAvailable(false);
            return false;
        }
    }, []);

    const cutClips = React.useCallback(
        (ids: string[]) => {
            void (async () => {
                const copied = await copyClips(ids);
                if (!copied) return;
                setMultiSelectedClipIds([]);
                void dispatch(removeClipsRemote(ids));
            })();
        },
        [copyClips, dispatch, setMultiSelectedClipIds],
    );

    const copyTracks = React.useCallback(async (ids: string[]) => {
        if (ids.length === 0) return false;
        try {
            const result = await webApi.copyTimelineTracks(ids);
            setClipboardAvailable(Boolean(result?.ok));
            return Boolean(result?.ok);
        } catch {
            setClipboardAvailable(false);
            return false;
        }
    }, []);

    const cutTracks = React.useCallback(
        (ids: string[]) => {
            void (async () => {
                const copied = await copyTracks(ids);
                if (!copied) return;
                void dispatch(removeTrackRemote(ids[0]));
            })();
        },
        [copyTracks, dispatch],
    );

    const copyClipsToReaper = React.useCallback(async (ids: string[]) => {
        if (ids.length === 0) return false;
        try {
            const result = await webApi.copyClipsToReaperClipboard(ids);
            if (result?.ok) {
                setClipboardAvailable(false);
            }
            return Boolean(result?.ok);
        } catch {
            return false;
        }
    }, []);

    // ── normalizeClips ───────────────────────────────────────
    const normalizeClips = React.useCallback(
        (ids: string[]) => {
            for (const id of ids) {
                const clip = sessionRef.current.clips.find((c) => c.id === id);
                if (!clip) continue;
                const newGain = computeClipNormalizationGain(clip, {
                    getInterleavedSlice: (sourcePath, _channel, sourceStartSec, sourceSpanSec) =>
                        waveformMipmapStore.getInterleavedSlice(
                            sourcePath,
                            0,
                            sourceStartSec,
                            sourceSpanSec,
                        ),
                    releaseInterleaved: (data) =>
                        waveformMipmapStore.releaseInterleaved(data as Float32Array),
                });
                if (newGain == null) continue;
                dispatch(setClipGain({ clipId: id, gain: newGain }));
                void dispatch(setClipStateRemote({ clipId: id, gain: newGain }));
            }
        },
        [dispatch, sessionRef],
    );

    // ── replaceClipSources ───────────────────────────────────
    const replaceClipSources = React.useCallback(
        async (ids: string[]) => {
            // 过滤掉音高参考块（没有音频源文件）
            const audioOnlyIds = ids.filter((id) => {
                const c = sessionRef.current.clips.find((clip) => clip.id === id);
                return c && c.midiNoteCount == null;
            });
            if (audioOnlyIds.length === 0) return;

            const selected = sessionRef.current.clips.filter((c) => audioOnlyIds.includes(c.id));
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
                        !audioOnlyIds.includes(clip.id) &&
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
                    clipIds: audioOnlyIds,
                    newSourcePath: picked.path,
                    replaceSameSource,
                }),
            );
        },
        [dispatch, t, sessionRef, sameSourceConfirmResolverRef, setSameSourceConfirmOpen],
    );

    // ── splitClipIdsAtPlayhead ────────────────────────────────
    const splitClipIdsAtPlayhead = React.useCallback(
        (clipIds: string[]) => {
            const session = sessionRef.current;
            let splitSec = Math.max(0, Number(session.playheadSec ?? 0) || 0);
            if (session.timelineSnap.enabled && session.timelineSnap.snapRazorEdits) {
                const snapped = snapTimelinePosition(
                    {
                        settings: session.timelineSnap,
                        grid: session.grid,
                        bpm: session.bpm,
                        beatsPerBar: session.beats,
                        tempoMap: session.tempoMap,
                        pxPerSec: Math.max(1e-9, pxPerSec),
                        clips: session.clips,
                        tracks: session.tracks,
                        selectedClipIds: clipIds,
                        playheadSec: splitSec,
                        object: "cursor",
                        anchorTrackId: session.selectedTrackId,
                    },
                    splitSec,
                );
                splitSec = snapped.sec;
            }

            // Expand to include all group members of any input clip
            const expandedIds = new Set(clipIds);
            if (!ignoreGrouping) {
                for (const id of clipIds) {
                    const groupMembers = getGroupClipIds(
                        id,
                        sessionRef.current.clips,
                        disabledGroupIds,
                    );
                    if (groupMembers) {
                        for (const gid of groupMembers) expandedIds.add(gid);
                    }
                }
            }

            const eligibleIds = Array.from(expandedIds).filter((id) => {
                const c = sessionRef.current.clips.find((clip) => clip.id === id);
                if (!c) return false;
                return splitSec > c.startSec + 1e-6 && splitSec < c.startSec + c.lengthSec - 1e-6;
            });
            if (eligibleIds.length > 0) {
                void dispatch(splitClipsAtRemote({ clipIds: eligibleIds, splitSec }));
            }
            return eligibleIds;
        },
        [dispatch, pxPerSec, ignoreGrouping, disabledGroupIds, sessionRef],
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
    }, [splitClipIdsAtPlayhead, sessionRef]);

    // ── recordLastClickPosition ──────────────────────────────
    const recordLastClickPosition = React.useCallback(
        (clientX: number) => {
            lastClickedClientXRef.current = clientX;
        },
        [lastClickedClientXRef],
    );

    // ── selectClipRangeByRect ────────────────────────────────
    const selectClipRangeByRect = React.useCallback(
        (targetClipId: string, anchorClipIdOverride?: string | null, targetClientX?: number) => {
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
                updateRangeSelectAnchor(targetClipId);
                lastClickedClientXRef.current = targetClientX ?? null;
                return;
            }

            const minTrack = Math.min(anchorTrackIndex, targetTrackIndex);
            const maxTrack = Math.max(anchorTrackIndex, targetTrackIndex);

            // 使用鼠标点击位置（时间秒）构建选择矩形，避免长 clip 导致的过度选择
            let anchorClickSec: number;
            let targetClickSec: number;

            const scroller = scrollRef.current;
            const anchorClientX = lastClickedClientXRef.current;
            if (scroller && anchorClientX != null && targetClientX != null) {
                const bounds = scroller.getBoundingClientRect();
                const xScroll = scroller.scrollLeft;
                anchorClickSec = Math.max(0, (anchorClientX - bounds.left + xScroll) / pxPerSec);
                targetClickSec = Math.max(0, (targetClientX - bounds.left + xScroll) / pxPerSec);
            } else {
                // 降级：使用 clip 的 startSec 作为点击时间近似
                anchorClickSec = anchor.startSec;
                targetClickSec = target.startSec;
            }

            const minStartSec = Math.min(anchorClickSec, targetClickSec);
            const maxEndSec = Math.max(anchorClickSec, targetClickSec);

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
            updateRangeSelectAnchor(targetClipId);
            lastClickedClientXRef.current = targetClientX ?? null;
        },
        [
            dispatch,
            setMultiSelectedClipIds,
            pxPerSec,
            sessionRef,
            lastClickedClipIdRef,
            updateRangeSelectAnchor,
            lastClickedClientXRef,
            scrollRef,
        ],
    );

    // ── pasteClipsAtPlayhead ─────────────────────────────────
    const pasteClipsAtPlayhead = React.useCallback(
        (mode?: "selected" | "new_tracks") => {
            void (async () => {
                try {
                    const result = await dispatch(pasteTimelineClipboardRemote(mode)).unwrap();
                    setClipboardAvailable(true);
                    const created = result.newClipIds ?? [];
                    if (created.length === 0) return;

                    setMultiSelectedClipIds(created);
                    void dispatch(selectClipRemote(created[0]));
                    dispatch(setplayheadSec(result.timeline?.playhead_sec ?? 0));
                    void dispatch(seekPlayhead(result.timeline?.playhead_sec ?? 0));

                    if (sessionRef.current.autoCrossfadeEnabled) {
                        const allClips = (result.timeline?.clips ?? []) as Array<{
                            id?: string;
                            track_id?: string;
                            start_sec?: number;
                            length_sec?: number;
                            fade_in_sec?: number;
                            fade_out_sec?: number;
                        }>;
                        const fadeUpdates = computeAutoCrossfadeFromPayload(allClips, created);
                        if (fadeUpdates.length > 0) {
                            const changesById = new Map(
                                fadeUpdates.map((u) => [
                                    u.clipId,
                                    {
                                        fadeInSec: u.fadeInSec,
                                        fadeOutSec: u.fadeOutSec,
                                    },
                                ]),
                            );
                            await dispatch(
                                setClipsStateBulkRemote({
                                    updates: buildBulkClipStateUpdates({
                                        clipIds: [...changesById.keys()],
                                        changesById,
                                    }),
                                    checkpoint: false,
                                }),
                            ).unwrap();
                        }
                    }
                } catch {
                    setClipboardAvailable(false);
                }
            })();
        },
        [dispatch, setMultiSelectedClipIds, sessionRef],
    );

    // ── TrackLane callbacks ───────────────────────────────────
    const ensureTrackLaneSelected = React.useCallback(
        (clipId: string) => {
            updateRangeSelectAnchor(clipId);
            const selectedIds = multiSelectedClipIdsRef.current;
            const selectedSet = multiSelectedSetRef.current;
            if (!selectedSet.has(clipId) || selectedIds.length > 1) {
                setMultiSelectedClipIds([clipId]);
            }
        },
        [setMultiSelectedClipIds, updateRangeSelectAnchor],
    );

    const selectTrackLaneClipRemote = React.useCallback(
        (clipId: string) => {
            updateRangeSelectAnchor(clipId);
            const clip = sessionRef.current.clips.find((entry) => entry.id === clipId);
            const clipTrackId = clip?.trackId ?? null;
            if (
                sessionRef.current.selectedClipId === clipId &&
                clipTrackId != null &&
                clipTrackId === sessionRef.current.selectedTrackId
            ) {
                return;
            }
            const preserveTrackFocus =
                !sessionRef.current.paramEditorTimelineClickSelectTrackEnabled ||
                Boolean(clip && clip.trackId === sessionRef.current.selectedTrackId);
            void dispatch(
                selectClipRemote({
                    clipId,
                    preserveTrackFocus,
                }),
            );
        },
        [dispatch, updateRangeSelectAnchor, sessionRef],
    );

    const toggleTrackLaneCtrlSelection = React.useCallback(
        (clipId: string) => {
            updateRangeSelectAnchor(clipId);

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
            const preserveTrackFocus =
                !sessionRef.current.paramEditorTimelineClickSelectTrackEnabled ||
                Boolean(
                    nextPrimaryClip &&
                        nextPrimaryClip.trackId === sessionRef.current.selectedTrackId,
                );

            void dispatch(
                selectClipRemote({
                    clipId: nextPrimaryClipId,
                    preserveTrackFocus,
                }),
            );
        },
        [dispatch, setMultiSelectedClipIds, updateRangeSelectAnchor, sessionRef],
    );

    const rangeSelectAnchorClipId = rangeSelectAnchorClipIdState ?? selectedClipId;

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
        [setPlayheadFromClientX, scrollRef],
    );

    const toggleTrackLaneClipMuted = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            const targetIds = getBulkEditableClipIds({
                activeClipId: clipId,
                multiSelectedClipIds: multiSelectedClipIdsRef.current,
                multiSelectedSet: multiSelectedSetRef.current,
            });
            const changesById = new Map(
                targetIds.map((targetId) => [targetId, { muted: nextMuted }] as const),
            );
            for (const targetId of targetIds) {
                dispatch(
                    setClipMuted({
                        clipId: targetId,
                        muted: nextMuted,
                    }),
                );
            }
            void dispatch(
                setClipsStateBulkRemote({
                    updates: buildBulkClipStateUpdates({
                        clipIds: targetIds,
                        changesById,
                    }),
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
            const targetIds = getBulkEditableClipIds({
                activeClipId: clipId,
                multiSelectedClipIds: multiSelectedClipIdsRef.current,
                multiSelectedSet: multiSelectedSetRef.current,
            });
            const changesById = new Map(targetIds.map((targetId) => [targetId, { gain }] as const));
            for (const targetId of targetIds) {
                dispatch(setClipGain({ clipId: targetId, gain }));
            }
            void dispatch(
                setClipsStateBulkRemote({
                    updates: buildBulkClipStateUpdates({
                        clipIds: targetIds,
                        changesById,
                    }),
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

        clipboardAvailable,
        copyClips,
        cutClips,
        copyTracks,
        cutTracks,
        copyClipsToReaper,

        groupClips,
        ungroupClips,
        toggleGroupDisabled,
        normalizeClips,
        replaceClipSources,
        splitClipIdsAtPlayhead,
        splitSelectedAtPlayhead,
        selectClipRangeByRect,
        rangeSelectAnchorClipId,
        recordLastClickPosition,
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
