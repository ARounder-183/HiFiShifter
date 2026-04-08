import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { SessionState } from "../sessionSlice";

import { addTrackRemote, setClipStateRemote } from "./timelineThunks";
import { computeAutoCrossfadeFromPayload } from "../../../components/layout/timeline/hooks/autoCrossfade";

type RawTimelineClip = {
    id?: string;
    track_id?: string;
    start_sec?: number;
    length_sec?: number;
    fade_in_sec?: number;
    fade_out_sec?: number;
};

async function syncAutoCrossfadeFromLatestTimeline(args: {
    dispatch: (action: unknown) => Promise<unknown> & { unwrap: () => Promise<unknown> };
    getState: () => unknown;
    newClipIds: string[];
}) {
    const { dispatch, getState, newClipIds } = args;
    if (newClipIds.length === 0) {
        return null;
    }

    const session = (getState() as { session: SessionState }).session;
    if (!session.autoCrossfadeEnabled) {
        return null;
    }

    const latestTimeline = await webApi.getTimelineState();
    const allClips = (latestTimeline as { clips?: RawTimelineClip[] }).clips ?? [];
    const fadeUpdates = computeAutoCrossfadeFromPayload(allClips, newClipIds);
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
    return latestTimeline;
}

const setAudioPathAction = (path: string) => ({
    type: "session/setAudioPath" as const,
    payload: path,
});
export const importAudioFromDialog = createAsyncThunk(
    "session/importAudioFromDialog",
    async (_, { dispatch, rejectWithValue, getState }) => {
        const picked = await webApi.openAudioDialogMultiple();
        if (!picked.ok) {
            return rejectWithValue("open_audio_dialog_failed");
        }
        const pickedPaths = Array.isArray(picked.paths)
            ? picked.paths.filter((p): p is string => Boolean(p))
            : [];
        if (picked.canceled || pickedPaths.length === 0) {
            return { ok: true, canceled: true };
        }

        // Use current playhead position as import start and preserve playhead.
        const state = getState() as { session: SessionState };
        const startSec = state.session.playheadSec ?? 0;
        const trackId = state.session.selectedTrackId ?? null;
        const firstPath = pickedPaths[0];

        dispatch(setAudioPathAction(firstPath));

        if (pickedPaths.length > 1) {
            return {
                ok: true,
                canceled: false,
                path: firstPath,
                requiresModeChoice: true,
                audioPaths: pickedPaths,
                trackId,
                startSec,
            };
        }

        // Delegate to importAudioAtPosition so imported clips start at playhead
        // and selection/undo handling is consistent with other import flows.
        try {
            const res = await dispatch(
                importAudioAtPosition({
                    audioPath: firstPath,
                    trackId,
                    startSec,
                }),
            ).unwrap();
            return {
                ok: true,
                canceled: false,
                path: firstPath,
                imported: res.imported ?? res,
                newClipIds: res.newClipIds,
            };
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "import_audio_item_failed");
        }
    },
);

export const importAudioFromPath = createAsyncThunk(
    "session/importAudioFromPath",
    async (audioPath: string, { dispatch, rejectWithValue }) => {
        dispatch(setAudioPathAction(audioPath));
        const imported = await webApi.importAudioItem(audioPath);
        if (!(imported as { ok?: boolean }).ok) {
            return rejectWithValue(
                (imported as { error?: { message?: string } }).error?.message ??
                    "import_audio_item_failed",
            );
        }
        return {
            ok: true,
            path: audioPath,
            imported,
        };
    },
);

export const importAudioAtPosition = createAsyncThunk(
    "session/importAudioAtPosition",
    async (
        payload: {
            audioPath: string;
            trackId?: string | null;
            startSec?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        dispatch(setAudioPathAction(payload.audioPath));

        await webApi.beginUndoGroup();
        try {
            let targetTrackId: string | undefined;
            if (payload.trackId === null) {
                const state = getState() as { session: SessionState };
                const beforeIds = new Set(state.session.tracks.map((t) => t.id));
                try {
                    const added = await dispatch(
                        addTrackRemote({ name: undefined, parentTrackId: null }),
                    ).unwrap();
                    const createdId =
                        added.tracks.find((t) => !beforeIds.has(t.id))?.id ??
                        added.selected_track_id ??
                        added.tracks[added.tracks.length - 1]?.id ??
                        null;
                    if (!createdId) {
                        return rejectWithValue("add_track_failed");
                    }
                    targetTrackId = createdId;
                } catch (err) {
                    return rejectWithValue(err instanceof Error ? err.message : "add_track_failed");
                }
            } else {
                targetTrackId = payload.trackId ?? undefined;
            }

            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const imported = await webApi.importAudioItem(
                payload.audioPath,
                targetTrackId,
                payload.startSec,
            );
            if (!(imported as { ok?: boolean }).ok) {
                return rejectWithValue(
                    (imported as { error?: { message?: string } }).error?.message ??
                        "import_audio_item_failed",
                );
            }

            const result = imported as { clips?: Array<{ id?: string }> };
            const newClipIds = (result.clips ?? [])
                .map((c) => c.id)
                .filter((id): id is string => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到第一个音频块的起始位置
            const importedResult = latestTimeline ?? imported;
            if (importedResult && typeof payload.startSec === "number") {
                (importedResult as unknown as Record<string, unknown>).playhead_sec =
                    payload.startSec;
            }

            return {
                ok: true,
                imported: importedResult,
                newClipIds,
            };
        } finally {
            void webApi.endUndoGroup();
        }
    },
);

export const importAudioFileAtPosition = createAsyncThunk(
    "session/importAudioFileAtPosition",
    async (
        payload: { file: File; trackId?: string | null; startSec?: number },
        { dispatch, rejectWithValue, getState },
    ) => {
        await webApi.beginUndoGroup();
        try {
            let targetTrackId: string | undefined;
            if (payload.trackId === null) {
                const state = getState() as { session: SessionState };
                const beforeIds = new Set(state.session.tracks.map((t) => t.id));
                const added = await dispatch(
                    addTrackRemote({ name: undefined, parentTrackId: null }),
                ).unwrap();
                const createdId =
                    added.tracks.find((t) => !beforeIds.has(t.id))?.id ??
                    added.selected_track_id ??
                    added.tracks[added.tracks.length - 1]?.id ??
                    null;
                if (!createdId) {
                    return rejectWithValue("add_track_failed");
                }
                targetTrackId = createdId;
            } else {
                targetTrackId = payload.trackId ?? undefined;
            }

            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const fileName = String(payload.file.name ?? "dropped-audio");
            const dataUrl = await new Promise<string>((resolve, reject) => {
                const reader = new FileReader();
                reader.onerror = () => reject(new Error("read_failed"));
                reader.onload = () => resolve(String(reader.result ?? ""));
                reader.readAsDataURL(payload.file);
            });

            const commaIdx = dataUrl.indexOf(",");
            const base64 = commaIdx !== -1 ? dataUrl.substring(commaIdx + 1) : dataUrl;

            const imported = await webApi.importAudioBytes(
                fileName,
                base64,
                targetTrackId,
                payload.startSec,
            );
            if (!(imported as { ok?: boolean }).ok) {
                return rejectWithValue(
                    (imported as { error?: { message?: string } }).error?.message ??
                        "import_audio_bytes_failed",
                );
            }

            const result = imported as { clips?: Array<{ id?: string }> };
            const newClipIds = (result.clips ?? [])
                .map((c) => c.id)
                .filter((id): id is string => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到第一个音频块的起始位置
            const importedResult = latestTimeline ?? imported;
            if (importedResult && typeof payload.startSec === "number") {
                (importedResult as unknown as Record<string, unknown>).playhead_sec =
                    payload.startSec;
            }

            return {
                ok: true,
                imported: importedResult,
                newClipIds,
            };
        } catch (err) {
            return rejectWithValue(
                err instanceof Error ? err.message : "import_audio_bytes_failed",
            );
        } finally {
            void webApi.endUndoGroup();
        }
    },
);

/**
 * 多文件导入，支持两种模式:
 * - "across-time": 在同一轨道依次排列（按顺序首尾相连）
 * - "across-tracks": 每个文件分配到不同的新轨道，起始位置相同
 */
export const importMultipleAudioAtPosition = createAsyncThunk(
    "session/importMultipleAudioAtPosition",
    async (
        payload: {
            audioPaths: string[];
            mode: "across-time" | "across-tracks";
            trackId?: string | null;
            startSec?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        const { audioPaths, mode, startSec = 0 } = payload;
        if (audioPaths.length === 0) return { ok: true };

        // Single file → delegate to importAudioAtPosition
        if (audioPaths.length === 1) {
            return dispatch(
                importAudioAtPosition({
                    audioPath: audioPaths[0],
                    trackId: payload.trackId,
                    startSec,
                }),
            ).unwrap();
        }

        // Create a single undo checkpoint for the entire batch
        dispatch({ type: "session/checkpointHistory" });

        await webApi.beginUndoGroup();
        try {
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            let lastImported: unknown = null;
            let firstImported: unknown = null;
            const accumulatedNewClipIds: string[] = [];

            if (mode === "across-time") {
                // Import files sequentially on the same track
                let cursor = startSec;
                let targetTrackId: string | undefined;

                if (payload.trackId === null) {
                    // Create a new track
                    const state = getState() as { session: SessionState };
                    const beforeIds = new Set(state.session.tracks.map((t) => t.id));
                    try {
                        const added = await dispatch(
                            addTrackRemote({ name: undefined, parentTrackId: null }),
                        ).unwrap();
                        targetTrackId =
                            added.tracks.find((t) => !beforeIds.has(t.id))?.id ??
                            added.selected_track_id ??
                            added.tracks[added.tracks.length - 1]?.id ??
                            undefined;
                    } catch {
                        return rejectWithValue("add_track_failed");
                    }
                } else {
                    targetTrackId = payload.trackId ?? undefined;
                }

                for (const audioPath of audioPaths) {
                    const imported = await webApi.importAudioItem(audioPath, targetTrackId, cursor);
                    if (!(imported as { ok?: boolean }).ok) continue;
                    if (!firstImported) firstImported = imported;
                    lastImported = imported;
                    const result = imported as {
                        clips?: Array<{ id?: string; start_sec?: number; length_sec?: number }>;
                    };
                    const allClips = result.clips ?? [];
                    for (const c of allClips) {
                        if (c.id) accumulatedNewClipIds.push(c.id);
                    }
                    const newClip = allClips.find(
                        (c) => Math.abs((c.start_sec ?? 0) - cursor) < 0.01,
                    );
                    cursor += newClip?.length_sec ?? 0;
                }

                // Override playhead to the start of the FIRST imported clip
                if (lastImported && firstImported) {
                    const li = lastImported as Record<string, unknown>;
                    li.playhead_sec = startSec;
                }
            } else {
                // "across-tracks" — start from current track, then use subsequent existing tracks,
                // only creating new tracks when we run out of existing ones.
                const state = getState() as { session: SessionState };
                // Get root-level tracks sorted by order/index
                const rootTracks = state.session.tracks
                    .filter((t) => !t.parentId)
                    .sort((a, b) => {
                        // Use the index in the tracks array as order proxy
                        const ai = state.session.tracks.indexOf(a);
                        const bi = state.session.tracks.indexOf(b);
                        return ai - bi;
                    });

                // Find the starting index: the track the user dropped onto
                let startIdx = 0;
                if (payload.trackId) {
                    const idx = rootTracks.findIndex((t) => t.id === payload.trackId);
                    if (idx >= 0) startIdx = idx;
                }

                for (let i = 0; i < audioPaths.length; i++) {
                    const audioPath = audioPaths[i];
                    const trackIdx = startIdx + i;
                    let targetTrackId: string | undefined;

                    if (trackIdx < rootTracks.length) {
                        // Use existing track
                        targetTrackId = rootTracks[trackIdx].id;
                    } else {
                        // Need to create a new track
                        const curState = getState() as { session: SessionState };
                        const beforeTrackIds = new Set(curState.session.tracks.map((t) => t.id));
                        try {
                            const added = await dispatch(
                                addTrackRemote({ name: undefined, parentTrackId: null }),
                            ).unwrap();
                            targetTrackId =
                                added.tracks.find((t) => !beforeTrackIds.has(t.id))?.id ??
                                added.selected_track_id ??
                                added.tracks[added.tracks.length - 1]?.id ??
                                undefined;
                        } catch {
                            continue;
                        }
                    }

                    try {
                        const imported = await webApi.importAudioItem(
                            audioPath,
                            targetTrackId,
                            startSec,
                        );
                        if ((imported as { ok?: boolean }).ok) {
                            lastImported = imported;
                            const result = imported as {
                                clips?: Array<{
                                    id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                }>;
                            };
                            for (const c of result.clips ?? []) {
                                if (c.id) accumulatedNewClipIds.push(c.id);
                            }
                        }
                    } catch {
                        // Continue with remaining files
                    }
                }
            }

            // Detect new clips from all import responses
            const newClipIds = accumulatedNewClipIds.filter((id) => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到第一个音频块的起始位置
            const importedResult = latestTimeline ?? lastImported;
            if (importedResult) {
                (importedResult as Record<string, unknown>).playhead_sec = startSec;
            }

            return { ok: true, imported: importedResult, newClipIds };
        } finally {
            void webApi.endUndoGroup();
        }
    },
);

export const importMultipleAudioFilesAtPosition = createAsyncThunk(
    "session/importMultipleAudioFilesAtPosition",
    async (
        payload: {
            files: File[];
            mode: "across-time" | "across-tracks";
            trackId?: string | null;
            startSec?: number;
        },
        { dispatch, rejectWithValue, getState },
    ) => {
        const { files, mode, startSec = 0 } = payload;
        if (!files || files.length === 0) return { ok: true };

        // Single file → delegate to importAudioFileAtPosition
        if (files.length === 1) {
            return dispatch(
                importAudioFileAtPosition({ file: files[0], trackId: payload.trackId, startSec }),
            ).unwrap();
        }

        dispatch({ type: "session/checkpointHistory" });

        await webApi.beginUndoGroup();
        try {
            const beforeClipIds = new Set(
                (getState() as { session: SessionState }).session.clips.map((c) => c.id),
            );

            const accumulatedNewClipIds: string[] = [];

            let lastImported: unknown = null;
            let firstImported: unknown = null;

            if (mode === "across-time") {
                let cursor = startSec;
                let targetTrackId: string | undefined;

                if (payload.trackId === null) {
                    const state = getState() as { session: SessionState };
                    const beforeIds = new Set(state.session.tracks.map((t) => t.id));
                    try {
                        const added = await dispatch(
                            addTrackRemote({ name: undefined, parentTrackId: null }),
                        ).unwrap();
                        targetTrackId =
                            added.tracks.find((t) => !beforeIds.has(t.id))?.id ??
                            added.selected_track_id ??
                            added.tracks[added.tracks.length - 1]?.id ??
                            undefined;
                    } catch {
                        return rejectWithValue("add_track_failed");
                    }
                } else {
                    targetTrackId = payload.trackId ?? undefined;
                }

                for (const file of files) {
                    const fileName = String(file.name ?? "dropped-audio");
                    const dataUrl = await new Promise<string>((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onerror = () => reject(new Error("read_failed"));
                        reader.onload = () => resolve(String(reader.result ?? ""));
                        reader.readAsDataURL(file);
                    });
                    const base64 = dataUrl.includes(",")
                        ? dataUrl.split(",").slice(1).join(",")
                        : dataUrl;

                    const imported = await webApi.importAudioBytes(
                        fileName,
                        base64,
                        targetTrackId,
                        cursor,
                    );
                    if (!(imported as { ok?: boolean }).ok) continue;
                    if (!firstImported) firstImported = imported;
                    lastImported = imported;
                    const result = imported as {
                        clips?: Array<{ id?: string; start_sec?: number; length_sec?: number }>;
                    };
                    for (const c of result.clips ?? []) {
                        if (c.id) accumulatedNewClipIds.push(c.id);
                    }
                    const newClip = result.clips?.find(
                        (c) => Math.abs((c.start_sec ?? 0) - cursor) < 0.01,
                    );
                    cursor += newClip?.length_sec ?? 0;
                }
            } else {
                // across-tracks: similar to importMultipleAudioAtPosition
                const state = getState() as { session: SessionState };
                const rootTracks = state.session.tracks
                    .filter((t) => !t.parentId)
                    .sort(
                        (a, b) => state.session.tracks.indexOf(a) - state.session.tracks.indexOf(b),
                    );
                let startIdx = 0;
                if (payload.trackId) {
                    const idx = rootTracks.findIndex((t) => t.id === payload.trackId);
                    if (idx >= 0) startIdx = idx;
                }

                for (let i = 0; i < files.length; i++) {
                    const file = files[i];
                    const trackIdx = startIdx + i;
                    let targetTrackId: string | undefined;
                    if (trackIdx < rootTracks.length) {
                        targetTrackId = rootTracks[trackIdx].id;
                    } else {
                        const curState = getState() as { session: SessionState };
                        const beforeTrackIds = new Set(curState.session.tracks.map((t) => t.id));
                        try {
                            const added = await dispatch(
                                addTrackRemote({ name: undefined, parentTrackId: null }),
                            ).unwrap();
                            targetTrackId =
                                added.tracks.find((t) => !beforeTrackIds.has(t.id))?.id ??
                                added.selected_track_id ??
                                added.tracks[added.tracks.length - 1]?.id ??
                                undefined;
                        } catch {
                            continue;
                        }
                    }

                    const fileName = String(file.name ?? "dropped-audio");
                    const dataUrl = await new Promise<string>((resolve, reject) => {
                        const reader = new FileReader();
                        reader.onerror = () => reject(new Error("read_failed"));
                        reader.onload = () => resolve(String(reader.result ?? ""));
                        reader.readAsDataURL(file);
                    });
                    const base64 = dataUrl.includes(",")
                        ? dataUrl.split(",").slice(1).join(",")
                        : dataUrl;

                    try {
                        const imported = await webApi.importAudioBytes(
                            fileName,
                            base64,
                            targetTrackId,
                            startSec,
                        );
                        if ((imported as { ok?: boolean }).ok) {
                            lastImported = imported;
                            const result = imported as {
                                clips?: Array<{
                                    id?: string;
                                    start_sec?: number;
                                    length_sec?: number;
                                }>;
                            };
                            for (const c of result.clips ?? []) {
                                if (c.id) accumulatedNewClipIds.push(c.id);
                            }
                        }
                    } catch {
                        // continue
                    }
                }
            }

            const newClipIds = accumulatedNewClipIds.filter((id) => !!id && !beforeClipIds.has(id));

            const latestTimeline = await syncAutoCrossfadeFromLatestTimeline({
                dispatch: dispatch as unknown as (
                    action: unknown,
                ) => Promise<unknown> & { unwrap: () => Promise<unknown> },
                getState,
                newClipIds,
            });

            // 导入后将光标定位到第一个音频块的起始位置
            const importedResult = latestTimeline ?? lastImported;
            if (importedResult) {
                (importedResult as Record<string, unknown>).playhead_sec = startSec;
            }

            return { ok: true, imported: importedResult, newClipIds };
        } finally {
            void webApi.endUndoGroup();
        }
    },
);
