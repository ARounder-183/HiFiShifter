import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { TimelineState } from "../../../types/api";
import type { ClipTemplate } from "../sessionTypes";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";

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
    async (payload: { trackId: string; targetIndex: number; parentTrackId?: string | null }) => {
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
        return webApi.getTrackSummary(state.session.selectedTrackId ?? undefined);
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
        const shouldApplyLinkedParams = (getState() as { session: SessionState }).session
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
                        templates.map((t) => t.trackId).filter((id): id is string => Boolean(id)),
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
                const neededLastIndex = selectedTrackIndex + sourceGroupKeys.length - 1;

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
                    const key =
                        tpl.trackId && sourceToTargetTrack.has(tpl.trackId)
                            ? tpl.trackId
                            : sourceGroupKeys[0];
                    return {
                        ...tpl,
                        trackId: sourceToTargetTrack.get(key) ?? defaultTargetTrack,
                    };
                });
            }
        }

        const normalizedTemplates = templates.map((tpl) => ({
            ...tpl,
            ...(shouldApplyLinkedParams ? {} : { linkedParams: undefined }),
        }));

        const result = await webApi.createClipsBulk({
            templates: normalizedTemplates,
            selectCreatedClips: true,
        });

        if (!(result as { ok?: boolean }).ok) {
            return rejectWithValue("create_clips_failed");
        }

        const timeline = result as TimelineState & {
            createdClipIds?: string[];
            created_clip_ids?: string[];
        };
        const createdClipIds = Array.isArray(timeline.created_clip_ids)
            ? timeline.created_clip_ids
            : Array.isArray(timeline.createdClipIds)
              ? timeline.createdClipIds
              : [];

        if (createdClipIds.length === 0) {
            return rejectWithValue("create_clips_failed");
        }

        for (let i = 0; i < createdClipIds.length; i += 1) {
            const createdId = createdClipIds[i];
            const tpl = normalizedTemplates[i];
            if (!createdId || !tpl || !Array.isArray(tpl.waveformPreview)) continue;
            const createdClip = timeline.clips.find((clip) => clip.id === createdId);
            if (
                createdClip &&
                (!Array.isArray(createdClip.waveform_preview) ||
                    createdClip.waveform_preview.length === 0)
            ) {
                createdClip.waveform_preview = tpl.waveformPreview;
            }
        }

        return {
            ...(timeline as object),
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

export const removeClipsRemote = createAsyncThunk(
    "session/removeClipsRemote",
    async (clipIds: string[]) => {
        return webApi.removeClips(clipIds);
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
        reversed?: boolean;
        fadeInSec?: number;
        fadeOutSec?: number;
        fadeInCurve?: string;
        fadeOutCurve?: string;
        formantMorph?: {
            enabled: boolean;
            targetF1Hz: number;
            targetF2Hz: number;
            strength: number;
        };
        checkpoint?: boolean;
    }) => {
        return webApi.setClipState(payload);
    },
);

export const setClipsStateBulkRemote = createAsyncThunk(
    "session/setClipsStateBulkRemote",
    async (payload: {
        updates: Array<{
            clipId: string;
            gain?: number;
            muted?: boolean;
            fadeInSec?: number;
            fadeOutSec?: number;
        }>;
        checkpoint?: boolean;
    }) => {
        return webApi.setClipsStateBulk(payload);
    },
);

export const duplicateClipsBulkRemote = createAsyncThunk<
    TimelineState & { createdClipIds?: string[]; created_clip_ids?: string[] },
    {
        sourceClipIds: string[];
        deltaSec: number;
        trackMode: Record<string, unknown>;
        copyLinkedParams?: boolean;
        selectCreatedClips?: boolean;
        applyAutoCrossfade?: boolean;
        placeOnSelectedTrack?: boolean;
        renameCopies?: boolean;
    }
>("session/duplicateClipsBulkRemote", async (payload) => {
    const result = await webApi.duplicateClipsBulk(payload);
    if (result && typeof result === "object" && "clips" in result) {
        const typed = result as TimelineState & { created_clip_ids?: string[] };
        return {
            ...typed,
            createdClipIds: Array.isArray(typed.created_clip_ids)
                ? typed.created_clip_ids
                : undefined,
        };
    }
    return result as TimelineState & { createdClipIds?: string[]; created_clip_ids?: string[] };
});

export const replaceClipSourceRemote = createAsyncThunk(
    "session/replaceClipSourceRemote",
    async (payload: { clipIds: string[]; newSourcePath: string; replaceSameSource?: boolean }) => {
        const result = await webApi.replaceClipSource(payload);
        // 使前端的波形 mipmap 缓存失效，确保下次渲染时重新从后端拉取新文件的波形数据
        if (result?.ok) {
            waveformMipmapStore.invalidate(payload.newSourcePath);
        }
        return result;
    },
);

export const replaceMidiClipDataRemote = createAsyncThunk(
    "session/replaceMidiClipDataRemote",
    async (payload: {
        clipId: string;
        midiPath: string;
        trackIndices: number[];
        fillGaps?: boolean;
        noteBpmMode?: string;
        specifiedBpm?: number;
        importMidiBpmAsProject?: boolean;
        closeLeadingGap?: boolean;
    }) => {
        return webApi.replaceMidiClipData(
            payload.clipId,
            payload.midiPath,
            payload.trackIndices,
            payload.fillGaps,
            payload.noteBpmMode,
            payload.specifiedBpm,
            payload.importMidiBpmAsProject,
            undefined,
            payload.closeLeadingGap,
        );
    },
);

export const splitClipRemote = createAsyncThunk(
    "session/splitClipRemote",
    async (payload: { clipId: string; splitSec: number }) => {
        return webApi.splitClip(payload.clipId, payload.splitSec);
    },
);

export const splitClipsAtRemote = createAsyncThunk(
    "session/splitClipsAtRemote",
    async (payload: { clipIds: string[]; splitSec: number }) => {
        return webApi.splitClipsAt(payload.clipIds, payload.splitSec);
    },
);

export const glueClipsRemote = createAsyncThunk(
    "session/glueClipsRemote",
    async (clipIds: string[]) => {
        return webApi.glueClips(clipIds);
    },
);

export const groupClipsRemote = createAsyncThunk(
    "session/groupClipsRemote",
    async (clipIds: string[]) => {
        return webApi.groupClips(clipIds);
    },
);

export const ungroupClipsRemote = createAsyncThunk(
    "session/ungroupClipsRemote",
    async (clipIds: string[]) => {
        return webApi.ungroupClips(clipIds);
    },
);

export const toggleGroupDisabledRemote = createAsyncThunk(
    "session/toggleGroupDisabledRemote",
    async (groupId: string) => {
        return webApi.toggleGroupDisabled(groupId);
    },
);

export const convertClipsToPitchReferenceRemote = createAsyncThunk(
    "session/convertClipsToPitchReferenceRemote",
    async (clipIds: string[]) => {
        return webApi.convertClipsToPitchReference(clipIds);
    },
);

export const updatePitchReferenceRemote = createAsyncThunk(
    "session/updatePitchReferenceRemote",
    async (clipIds: string[]) => {
        return webApi.updatePitchReference(clipIds);
    },
);

export const selectClipRemote = createAsyncThunk(
    "session/selectClipRemote",
    async (
        arg:
            | string
            | null
            | {
                  clipId: string | null;
                  preserveTrackFocus?: boolean;
              },
    ) => {
        const clipId =
            typeof arg === "object" && arg !== null && "clipId" in arg ? arg.clipId : arg;
        const preserveTrackFocus =
            typeof arg === "object" && arg !== null ? Boolean(arg.preserveTrackFocus) : false;

        const payload = await webApi.selectClip(clipId);
        if (payload && typeof payload === "object") {
            return {
                ...payload,
                __preserveTrackFocus: preserveTrackFocus,
            };
        }
        return payload;
    },
);
