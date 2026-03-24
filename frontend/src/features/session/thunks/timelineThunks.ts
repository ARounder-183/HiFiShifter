import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { TimelineState } from "../../../types/api";
import type { ClipTemplate } from "../sessionTypes";

// 注意：这�?thunk 依赖 SessionState（目前仍�?sessionSlice.ts 内部定义）�?
// 我们在此处用 type-only import，避免运行时循环依赖�?
import type { SessionState } from "../sessionSlice";

export const addTrackRemote = createAsyncThunk(
    "session/addTrackRemote",
    async (payload: { name?: string; parentTrackId?: string | null }) => {
        return webApi.addTrackNested(payload);
    },
);

export const removeTrackRemote = createAsyncThunk(
    "session/removeTrackRemote",
    async (trackId: string) => {
        return webApi.removeTrack(trackId);
    },
);

export const duplicateTrackRemote = createAsyncThunk(
    "session/duplicateTrackRemote",
    async (trackId: string) => {
        return webApi.duplicateTrack(trackId);
    },
);

export const moveTrackRemote = createAsyncThunk(
    "session/moveTrackRemote",
    async (payload: {
        trackId: string;
        targetIndex: number;
        parentTrackId?: string | null;
    }) => {
        return webApi.moveTrack(payload);
    },
);

export const selectTrackRemote = createAsyncThunk(
    "session/selectTrackRemote",
    async (trackId: string) => {
        return webApi.selectTrack(trackId);
    },
);

export const setProjectLengthRemote = createAsyncThunk(
    "session/setProjectLengthRemote",
    async (projectSec: number) => {
        return webApi.setProjectLength(projectSec);
    },
);

export const fetchSelectedTrackSummary = createAsyncThunk(
    "session/fetchSelectedTrackSummary",
    async (_, { getState }) => {
        const state = getState() as { session: SessionState };
        return webApi.getTrackSummary(
            state.session.selectedTrackId ?? undefined,
        );
    },
);

export const addClipOnTrack = createAsyncThunk(
    "session/addClipOnTrack",
    async (payload: { trackId?: string }) => {
        return webApi.addClip({ trackId: payload.trackId });
    },
);

export const createClipsRemote = createAsyncThunk(
    "session/createClipsRemote",
    async (
        payload: {
            templates: ClipTemplate[];
            options?: {
                /**
                 * 粘贴时将模板按源轨道相对顺序重映射到当前选中轨道，
                 * 并在轨道不足时自动创建新轨道。
                 */
                placeOnSelectedTrack?: boolean;
            };
        },
        { getState, dispatch, rejectWithValue },
    ) => {
        let templates = payload.templates;
        const shouldApplyLinkedParams =
            (getState() as { session: SessionState }).session
                .lockParamLinesEnabled;

        if (payload.options?.placeOnSelectedTrack && templates.length > 0) {
            const state = getState() as { session: SessionState };
            const selectedTrackId = state.session.selectedTrackId;
            const selectedTrackIndex = selectedTrackId
                ? state.session.tracks.findIndex((t) => t.id === selectedTrackId)
                : -1;

            if (selectedTrackId && selectedTrackIndex >= 0) {
                const trackOrder = new Map<string, number>();
                for (let i = 0; i < state.session.tracks.length; i += 1) {
                    trackOrder.set(state.session.tracks[i].id, i);
                }

                const sourceTrackIds = Array.from(
                    new Set(
                        templates
                            .map((t) => t.trackId)
                            .filter((id): id is string => Boolean(id)),
                    ),
                ).sort((a, b) => {
                    const ai = trackOrder.get(a) ?? Number.MAX_SAFE_INTEGER;
                    const bi = trackOrder.get(b) ?? Number.MAX_SAFE_INTEGER;
                    if (ai !== bi) return ai - bi;
                    return a.localeCompare(b);
                });

                const sourceGroupKeys =
                    sourceTrackIds.length > 0 ? sourceTrackIds : ["__default__"];

                let workingTracks = state.session.tracks.map((t) => ({
                    id: t.id,
                }));
                const neededLastIndex =
                    selectedTrackIndex + sourceGroupKeys.length - 1;

                while (workingTracks.length - 1 < neededLastIndex) {
                    const beforeIds = new Set(workingTracks.map((t) => t.id));
                    const added = await dispatch(
                        addTrackRemote({ name: undefined, parentTrackId: null }),
                    ).unwrap();
                    workingTracks = (added.tracks ?? []).map((t) => ({
                        id: t.id,
                    }));

                    const createdTrackId =
                        workingTracks.find((t) => !beforeIds.has(t.id))?.id ??
                        added.selected_track_id ??
                        workingTracks[workingTracks.length - 1]?.id ??
                        null;

                    if (!createdTrackId) {
                        return rejectWithValue("add_track_failed");
                    }
                }

                const sourceToTargetTrack = new Map<string, string>();
                for (let i = 0; i < sourceGroupKeys.length; i += 1) {
                    const targetTrack = workingTracks[selectedTrackIndex + i];
                    if (!targetTrack?.id) {
                        return rejectWithValue("add_track_failed");
                    }
                    sourceToTargetTrack.set(sourceGroupKeys[i], targetTrack.id);
                }

                const defaultTargetTrack =
                    sourceToTargetTrack.get(sourceGroupKeys[0]) ?? selectedTrackId;
                templates = templates.map((tpl) => {
                    const key = tpl.trackId && sourceToTargetTrack.has(tpl.trackId)
                        ? tpl.trackId
                        : sourceGroupKeys[0];
                    return {
                        ...tpl,
                        trackId: sourceToTargetTrack.get(key) ?? defaultTargetTrack,
                    };
                });
            }
        }

        const state0 = getState() as { session: SessionState };
        const knownIds = new Set(state0.session.clips.map((c) => c.id));
        const createdIdsInBatch = new Set<string>();

        // 并行创建所�?clip，提升批量操作性能
        const results = await Promise.all(
            templates.map(async (tpl) => {
                const added = await webApi.addClip({
                    trackId: tpl.trackId,
                    name: tpl.name,
                    startSec: tpl.startSec,
                    lengthSec: tpl.lengthSec,
                    sourcePath: tpl.sourcePath,
                });
                if (!(added as { ok?: boolean }).ok) {
                    throw new Error(
                        (added as { error?: { message?: string } }).error?.message ??
                            "add_clip_failed",
                    );
                }

                const addedTimeline = added as TimelineState;
                const createdId =
                    addedTimeline.clips.find(
                        (c) =>
                            !knownIds.has(c.id) &&
                            !createdIdsInBatch.has(c.id),
                    )?.id ??
                    (addedTimeline.selected_clip_id &&
                    !knownIds.has(addedTimeline.selected_clip_id) &&
                    !createdIdsInBatch.has(addedTimeline.selected_clip_id)
                        ? addedTimeline.selected_clip_id
                        : null);
                if (!createdId) {
                    throw new Error("add_clip_failed");
                }
                createdIdsInBatch.add(createdId);

                const updated = await webApi.setClipState({
                    clipId: createdId,
                    lengthSec: tpl.lengthSec,
                    gain: tpl.gain,
                    muted: tpl.muted,
                    sourceStartSec: tpl.sourceStartSec,
                    sourceEndSec: tpl.sourceEndSec,
                    playbackRate: tpl.playbackRate,
                    fadeInSec: tpl.fadeInSec,
                    fadeOutSec: tpl.fadeOutSec,
                });
                if (!(updated as { ok?: boolean }).ok) {
                    throw new Error(
                        (updated as { error?: { message?: string } }).error?.message ??
                            "set_clip_state_failed",
                    );
                }

                let finalTimeline = updated as TimelineState;
                if (shouldApplyLinkedParams && tpl.linkedParams) {
                    const linkedApplied = await webApi.applyClipLinkedParams({
                        clipId: createdId,
                        linkedParams: tpl.linkedParams,
                    });
                    if (!(linkedApplied as { ok?: boolean }).ok) {
                        throw new Error(
                            (linkedApplied as { error?: { message?: string } }).error
                                ?.message ?? "apply_clip_linked_params_failed",
                        );
                    }
                    finalTimeline = linkedApplied as TimelineState;
                }

                // Keep waveform visible after cut→paste even when backend cannot
                // reconstruct preview metadata from source_path (e.g. stale/relative path).
                if (Array.isArray(tpl.waveformPreview)) {
                    const createdClip = finalTimeline.clips.find(
                        (c) => c.id === createdId,
                    );
                    if (
                        createdClip &&
                        (!Array.isArray(createdClip.waveform_preview) ||
                            createdClip.waveform_preview.length === 0)
                    ) {
                        createdClip.waveform_preview = tpl.waveformPreview;
                    }
                }

                return { createdId, timeline: finalTimeline };
            }),
        ).catch((err: unknown) => {
            return rejectWithValue(
                err instanceof Error ? err.message : "create_clips_failed",
            );
        });

        if (!results || !Array.isArray(results)) {
            return results as ReturnType<typeof rejectWithValue>;
        }

        const createdClipIds = results.map((r) => r.createdId);
        // 取最后一�?timeline 作为最终状态（�?clip �?setClipState 结果�?
        const lastTimeline = results[results.length - 1]?.timeline ?? null;
        if (!lastTimeline) {
            return rejectWithValue("create_clips_failed");
        }
        return {
            ...(lastTimeline as object),
            createdClipIds,
        } as TimelineState & { createdClipIds: string[] };
    },
);

export const removeClipRemote = createAsyncThunk(
    "session/removeClipRemote",
    async (clipId: string) => {
        return webApi.removeClip(clipId);
    },
);

export const moveClipRemote = createAsyncThunk(
    "session/moveClipRemote",
    async (payload: {
        clipId: string;
        startSec: number;
        trackId?: string;
        moveLinkedParams?: boolean;
    }) => {
        return webApi.moveClip(payload);
    },
);

export const moveClipsRemote = createAsyncThunk(
    "session/moveClipsRemote",
    async (payload: {
        moves: Array<{
            clipId: string;
            startSec: number;
            trackId?: string;
        }>;
        moveLinkedParams?: boolean;
    }) => {
        return webApi.moveClips(payload);
    },
);

export const setClipStateRemote = createAsyncThunk(
    "session/setClipStateRemote",
    async (payload: {
        clipId: string;
        name?: string;
        color?: string;
        startSec?: number;
        lengthSec?: number;
        gain?: number;
        muted?: boolean;
        sourceStartSec?: number;
        sourceEndSec?: number;
        playbackRate?: number;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInCurve?: string;
        fadeOutCurve?: string;
    }) => {
        return webApi.setClipState(payload);
    },
);

export const replaceClipSourceRemote = createAsyncThunk(
    "session/replaceClipSourceRemote",
    async (payload: {
        clipIds: string[];
        newSourcePath: string;
        replaceSameSource?: boolean;
    }) => {
        return webApi.replaceClipSource(payload);
    },
);

export const splitClipRemote = createAsyncThunk(
    "session/splitClipRemote",
    async (payload: { clipId: string; splitSec: number }) => {
        return webApi.splitClip(payload.clipId, payload.splitSec);
    },
);

export const glueClipsRemote = createAsyncThunk(
    "session/glueClipsRemote",
    async (clipIds: string[]) => {
        return webApi.glueClips(clipIds);
    },
);

export const selectClipRemote = createAsyncThunk(
    "session/selectClipRemote",
    async (clipId: string | null) => {
        return webApi.selectClip(clipId);
    },
);
