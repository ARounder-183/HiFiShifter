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
import { useAppSelector, useAppStore } from "../../../../app/hooks";
import {
    pasteTimelineClipboardRemote,
    removeClipsRemote,
    selectClipRemote,
    setClipAutoFades,
    setClipGain,
    setClipMuted,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipboardOperationFailed,
    setMultiSelectedClipIds as setMultiSelectedClipIdsAction,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    replaceClipSourceRemote,
    renameClipTakeRemote,
    splitClipsAtRemote,
    bumpParamsEpoch,
} from "../../../../features/session/sessionSlice";
import { resolveRootTrackId } from "../../../../features/session/trackUtils";
import { paramsApi } from "../../../../services/api";
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
    /** 点击轨道空白区：清空 clip 选中（单选 + 多选），保留轨道焦点。 */
    deselectAllTrackLaneClips: () => void;
    openTrackLaneContextMenu: (clipId: string, clientX: number, clientY: number) => void;
    seekFromTrackLaneClientX: (clientX: number, commit: boolean) => void;
    toggleTrackLaneClipMuted: (clipId: string, nextMuted: boolean) => void;
    toggleTrackLaneCtrlSelection: (clipId: string) => void;
    toggleTrackLaneMultiSelect: (clipId: string) => void;
    commitTrackLaneRename: (clipId: string, newName: string) => void;
    handleTrackLaneRenameDone: () => void;
    commitTrackLaneGain: (clipId: string, db: number) => void;
    /**
     * 提交新的播放速率（有效速率， badge 展示值）。多选时批量应用到所有
     * 可编辑 Clip；adjustLength（默认 true）按新旧速率反比缩放各 Clip
     * 时长，使源消费窗口保持不变。
     */
    commitTrackLaneRate: (
        clipId: string,
        timing: {
            /** 有效速率（角标展示值）；缺省 = 不修改速率。 */
            rate?: number;
            /** 直接设定时长（秒）；优先于 rate 的自动调长。 */
            durationSec?: number;
            /** 速率变化时是否自动调整时长（默认 true）。 */
            autoLength?: boolean;
        },
    ) => void;
    /** 角标行内编辑目标（镜像 renamingClipId 管线）。 */
    editingBadge: { clipId: string; field: "rate" | "gain" } | null;
    setEditingBadge: (next: { clipId: string; field: "rate" | "gain" } | null) => void;
    handleBadgeEditDone: () => void;

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

    // 实时 store：copy/cut 时以 store 最新状态过滤失效 Clip id，
    // 避免闭包/ref 里的过期选区把死 id 传给后端。
    const store = useAppStore();

    // ── multiSelectedClipIds ─────────────────────────────────
    const multiSelectedClipIds = useAppSelector(
        (state: RootState) => state.session.multiSelectedClipIds,
    );
    const selectedClipId = useAppSelector((state: RootState) => state.session.selectedClipId);
    const [rangeSelectAnchorClipIdState, setRangeSelectAnchorClipIdState] = useState<string | null>(
        null,
    );
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
    // 角标（播放速率/增益）行内编辑目标：镜像 renamingClipId 的管线。
    const [editingBadge, setEditingBadge] = useState<{
        clipId: string;
        field: "rate" | "gain";
    } | null>(null);

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
        pxPerSec,
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

    const copyClips = React.useCallback(
        async (ids: string[], op: "copy" | "cut" = "copy"): Promise<boolean> => {
            // 只复制当前仍存在的 Clip：失效 id（删除/胶合/拆分替换后的残留）
            // 会被过滤，避免后端 no_clips_selected 静默失败。
            const currentClips = store.getState().session.clips;
            const liveIds = ids.filter((id) => currentClips.some((clip) => clip.id === id));
            if (liveIds.length === 0) return false;

            const attempt = async (): Promise<boolean> => {
                try {
                    const result = await webApi.copyTimelineClips(liveIds);
                    if (!result?.ok) {
                        setClipboardAvailable(false);
                        return false;
                    }
                    // 单剪贴板纪律：Clip 复制替换整个应用剪贴板（copy/cut 共用
                    // 本入口），显式失效参数编辑器的内部参数线剪贴板缓存 ——
                    // 否则"复制 Clip 后在参数编辑器粘贴"会把更早复制、已被
                    // 剪贴板替换掉的参数线数据从内部缓存复活。
                    window.dispatchEvent(
                        new CustomEvent("hifi:clipboardReplaced", {
                            detail: { kind: "clips" },
                        }),
                    );
                    setClipboardAvailable(true);
                    return true;
                } catch {
                    setClipboardAvailable(false);
                    return false;
                }
            };

            let copied = await attempt();
            if (!copied) {
                // 系统剪贴板瞬时被占用（剪贴板管理器 / RDP / 其它进程的延迟
                // 渲染等）时自动重试一次，让“第一次失败、立刻重试成功”的场景
                // 无需用户手动反复重试。
                await new Promise((resolve) => window.setTimeout(resolve, 300));
                copied = await attempt();
            }
            if (!copied) {
                // 最终失败：状态栏给出可见反馈（此前是静默失败，用户无法区分
                // “快捷键没触发”与“操作失败”，只能靠反复重选+重试碰运气）。
                dispatch(setClipboardOperationFailed({ op }));
            }
            return copied;
        },
        [dispatch, store],
    );

    const cutClips = React.useCallback(
        (ids: string[]) => {
            void (async () => {
                const copied = await copyClips(ids, "cut");
                if (!copied) return;
                setMultiSelectedClipIds([]);
                void dispatch(removeClipsRemote(ids));
            })();
        },
        [copyClips, dispatch, setMultiSelectedClipIds],
    );

    // ── normalizeClips ───────────────────────────────────────
    const normalizeClips = React.useCallback(
        (ids: string[]) => {
            const changesById = new Map<string, { gain: number }>();
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
                changesById.set(id, { gain: newGain });
            }
            if (changesById.size === 0) return;
            const clipIds = [...changesById.keys()];
            // 批量归一化 = 单个撤销步：undo group 内一次 bulk 提交
            //（逐个 setClipStateRemote 会产生 N 步撤销 + N 次中间快照）。
            void (async () => {
                await webApi.beginUndoGroup();
                try {
                    await dispatch(
                        setClipsStateBulkRemote({
                            updates: buildBulkClipStateUpdates({ clipIds, changesById }),
                            checkpoint: false,
                        }),
                    ).unwrap();
                } catch {
                    // 非致命：乐观值保留，此后权威快照会纠正。
                } finally {
                    await webApi.endUndoGroup();
                }
            })().catch(() => undefined);
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
        [dispatch, sessionRef, sameSourceConfirmResolverRef, setSameSourceConfirmOpen],
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
    // 粘贴链状态：idle=空闲；busy=一次粘贴在途；queued=在途期间又收到新的
    // 粘贴请求（如长按 Ctrl+V 的连续重复粘贴），当前粘贴完成后立即接续，
    // 避免并发粘贴在后端产生竞态。
    const pasteChainStateRef = React.useRef<"idle" | "busy" | "queued">("idle");
    const pasteClipsAtPlayhead = React.useCallback(
        (mode?: "selected" | "new_tracks") => {
            if (pasteChainStateRef.current !== "idle") {
                pasteChainStateRef.current = "queued";
                return;
            }
            pasteChainStateRef.current = "busy";
            void (async () => {
                try {
                    for (;;) {
                        try {
                            const result = await dispatch(
                                pasteTimelineClipboardRemote(mode),
                            ).unwrap();
                            setClipboardAvailable(true);
                            const created = result.newClipIds ?? [];
                            if (created.length > 0) {
                                setMultiSelectedClipIds(created);
                                void dispatch(selectClipRemote(created[0]));
                                // 播放光标已由 paste thunk 同步到"新 Clip 最靠右结束位置"
                                // （transport + 本地状态），此处无需再设置。

                                if (sessionRef.current.autoCrossfadeEnabled) {
                                    const allClips = (result.timeline?.clips ?? []) as Array<{
                                        id?: string;
                                        track_id?: string;
                                        start_sec?: number;
                                        length_sec?: number;
                                        auto_fade_in_sec?: number;
                                        auto_fade_out_sec?: number;
                                    }>;
                                    const fadeUpdates = computeAutoCrossfadeFromPayload(
                                        allClips,
                                        created,
                                    );
                                    if (fadeUpdates.length > 0) {
                                        // 粘贴后的自动交叉淡化写入“自动 fade”（与手动 fade 分离）。
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
                            }
                        } catch {
                            // 粘贴失败（如剪贴板为空）时终止粘贴链，避免长按期间
                            // 以固定节奏反复触发必然失败的请求。
                            setClipboardAvailable(false);
                            break;
                        }
                        // 没有排队中的粘贴请求则结束；有则立即接续下一次。
                        if (pasteChainStateRef.current !== "queued") break;
                        pasteChainStateRef.current = "busy";
                    }
                } finally {
                    pasteChainStateRef.current = "idle";
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

    // 点击轨道空白区：清空 clip 选中（单选 + 多选）。保留轨道焦点 —— 空白点击
    // 是"取消 clip 目标"，不是"切换轨道目标"（DAW 通用约定）。
    const deselectAllTrackLaneClips = React.useCallback(() => {
        if (multiSelectedClipIdsRef.current.length === 0 && !sessionRef.current.selectedClipId) {
            return;
        }
        setMultiSelectedClipIds([]);
        dispatch(setSelectedClipPreservingTrack(null));
    }, [dispatch, setMultiSelectedClipIds, sessionRef]);

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
            const clip = sessionRef.current?.clips.find((entry) => entry.id === clipId);
            const takes = clip?.takes ?? [];
            // 仅多 Take Clip 的改名写入 active take（UI 展示名此时取 take 名）。
            // 单 Take / 无 takes 时展示名是容器 name，必须走容器改名 ——
            // 后端 rename_take 不回写容器名，误路由会让重命名"看起来无效"。
            if (takes.length > 1) {
                const activeTake =
                    takes.find((entry) => entry.id === clip?.activeTakeId) ?? takes[0];
                void dispatch(
                    renameClipTakeRemote({ clipId, takeId: activeTake.id, name: newName }),
                );
                return;
            }
            void dispatch(
                setClipStateRemote({
                    clipId,
                    name: newName,
                }),
            );
        },
        [dispatch, sessionRef],
    );

    const handleTrackLaneRenameDone = React.useCallback(() => {
        setRenamingClipId(null);
    }, []);

    const handleBadgeEditDone = React.useCallback(() => {
        setEditingBadge(null);
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

    const commitTrackLaneRate = React.useCallback(
        (
            clipId: string,
            timing: {
                /** 有效速率（角标展示值）；缺省 = 不修改速率。 */
                rate?: number;
                /** 直接设定时长（秒）；优先于 rate 的自动调长。 */
                durationSec?: number;
                /** 速率变化时是否自动调整时长（默认 true）。 */
                autoLength?: boolean;
            },
        ) => {
            const session = sessionRef.current;
            const autoLength = timing.autoLength !== false;
            // 与 set_clip_state / 乐观分支同口径：有效速率钳制 0.1~10。
            const requestedRate =
                timing.rate != null && Number.isFinite(timing.rate) && timing.rate > 0
                    ? Math.min(10, Math.max(0.1, timing.rate))
                    : null;
            if (requestedRate == null && timing.durationSec == null) return;
            const targetIds = getBulkEditableClipIds({
                activeClipId: clipId,
                multiSelectedClipIds: multiSelectedClipIdsRef.current,
                multiSelectedSet: multiSelectedSetRef.current,
            });
            const changesById = new Map<
                string,
                { clipPlaybackRate?: number; lengthSec?: number }
            >();
            // “锁定参数线”启用且时长变化时：链接参数线时域映射（与边缘拉伸
            // 同一管线，按根轨道分组批量提交）。
            const mappingsByRootTrack = new Map<
                string,
                Array<{
                    oldStartSec: number;
                    oldLengthSec: number;
                    newStartSec: number;
                    newLengthSec: number;
                }>
            >();
            for (const targetId of targetIds) {
                const clip = session.clips.find((entry) => entry.id === targetId);
                if (!clip) continue;
                const takes = clip.takes ?? [];
                const activeTake =
                    takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
                // 有效速率 = Clip 级 × take 级：改写 Clip 级以保持 take 速率
                // 不变（与 reducer / 乐观分支的 previousTakeRate 口径一致）。
                const takeRate = activeTake
                    ? Number(activeTake.playbackRate) || 1
                    : (Number(clip.playbackRate) || 1) / (Number(clip.clipPlaybackRate) || 1);
                const oldEffective = Number(clip.playbackRate) || 1;
                const oldLengthSec = Number(clip.lengthSec) || 0;
                const change: { clipPlaybackRate?: number; lengthSec?: number } = {};
                // effective 按 clip 独立推导：多选 clip 的长度/速率各异，
                // “时长即拉伸”反推出的速率绝不能跨 clip 复用（旧实现把第一
                // 个 clip 的反推值泄漏给其余 clip，且第一个 clip 自己的速率
                // 反而漏写）。
                let effective = requestedRate;
                if (effective != null) {
                    change.clipPlaybackRate = Math.min(
                        10,
                        Math.max(0.1, effective / takeRate),
                    );
                }
                // 时长：显式时长优先，且**时长即拉伸**——源窗口保持不变，
                // 由时长反推有效速率（改时长同时改倍率）；无显式时长时，
                // 速率变化 + 自动调长才反比缩放时长（加速 → 变短）。
                let nextLengthSec: number | null = null;
                if (timing.durationSec != null && Number.isFinite(timing.durationSec)) {
                    nextLengthSec = Math.max(0, timing.durationSec);
                    if (effective == null && nextLengthSec > 1e-6) {
                        effective = Math.max(
                            0.1,
                            Math.min(10, (oldLengthSec * oldEffective) / nextLengthSec),
                        );
                        change.clipPlaybackRate = Math.min(
                            10,
                            Math.max(0.1, effective / takeRate),
                        );
                    }
                } else if (effective != null && autoLength) {
                    nextLengthSec = Math.max(0, oldLengthSec * (oldEffective / effective));
                }
                if (nextLengthSec != null && Math.abs(nextLengthSec - oldLengthSec) > 1e-6) {
                    change.lengthSec = nextLengthSec;
                    if (session.lockParamLinesEnabled) {
                        const rootTrackId = resolveRootTrackId(session.tracks, clip.trackId);
                        if (rootTrackId) {
                            const trackMappings =
                                mappingsByRootTrack.get(rootTrackId) ?? [];
                            trackMappings.push({
                                oldStartSec: clip.startSec,
                                oldLengthSec: oldLengthSec,
                                newStartSec: clip.startSec,
                                newLengthSec: nextLengthSec,
                            });
                            mappingsByRootTrack.set(rootTrackId, trackMappings);
                        }
                    }
                }
                changesById.set(targetId, change);
            }
            const persistPromise = dispatch(
                setClipsStateBulkRemote({
                    updates: buildBulkClipStateUpdates({ clipIds: targetIds, changesById }),
                    checkpoint: true,
                }),
            );
            // “锁定参数线”：后端应用新长度后，将旧范围的参数线时域映射到
            // 新范围，并 bump 参数纪元让参数编辑器重新拉取。
            if (mappingsByRootTrack.size > 0) {
                const stretchTasks = Array.from(mappingsByRootTrack, ([trackId, mappings]) =>
                    paramsApi.stretchTrackLinkedParams(trackId, mappings, false),
                );
                void Promise.resolve(persistPromise)
                    .then(() => Promise.allSettled(stretchTasks))
                    .finally(() => dispatch(bumpParamsEpoch()));
            }
        },
        [dispatch, multiSelectedClipIdsRef, multiSelectedSetRef, sessionRef],
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
        deselectAllTrackLaneClips,
        openTrackLaneContextMenu,
        seekFromTrackLaneClientX,
        toggleTrackLaneClipMuted,
        toggleTrackLaneCtrlSelection,
        toggleTrackLaneMultiSelect,
        commitTrackLaneRename,
        handleTrackLaneRenameDone,
        commitTrackLaneGain,
        commitTrackLaneRate,
        editingBadge,
        setEditingBadge,
        handleBadgeEditDone,

        sameSourceConfirmResolverRef,
    };
}
