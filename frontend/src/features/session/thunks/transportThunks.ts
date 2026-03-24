import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";

import type { SessionState } from "../sessionSlice";

export const fetchTimeline = createAsyncThunk(
    "session/fetchTimeline",
    async () => {
        return webApi.getTimelineState();
    },
);

export const stopAudioPlayback = createAsyncThunk(
    "session/stopAudioPlayback",
    async (options: { restoreAnchor?: boolean } | void, { getState }) => {
        const restoreAnchor = Boolean((options as { restoreAnchor?: boolean } | undefined)?.restoreAnchor);
        const state = getState() as { session: SessionState };
        const wasPlaying = Boolean(state.session.runtime.isPlaying);
        const anchorSec = state.session.playbackAnchorSec;
        const result = await webApi.stopAudio();
        // Only restore when this stop action actually interrupted active playback.
        if (
            restoreAnchor &&
            wasPlaying &&
            anchorSec !== undefined &&
            anchorSec !== null
        ) {
            await webApi.setTransport({ playheadSec: anchorSec });
        }
        return { ...result, restoreAnchor, wasPlaying, anchorSec };
    },
);

export const seekPlayhead = createAsyncThunk(
    "session/seekPlayhead",
    async (sec: number) => {
        return webApi.setTransport({ playheadSec: sec });
    },
);

export const updateTransportBpm = createAsyncThunk(
    "session/updateTransportBpm",
    async (bpm: number) => {
        return webApi.setTransport({ bpm });
    },
);

export const syncPlaybackState = createAsyncThunk(
    "session/syncPlaybackState",
    async () => {
        return webApi.getPlaybackState();
    },
);

export const playOriginal = createAsyncThunk(
    "session/playOriginal",
    async (_, { getState }) => {
        const state = getState() as { session: SessionState };
        const anchorSec = state.session.playheadSec;
        // Ensure backend transport is in sync before starting playback.
        await webApi.setTransport({ playheadSec: anchorSec });
        const result = await webApi.playOriginal(0);
        return {
            ...result,
            clipId: null,
            anchorSec,
        };
    },
);


