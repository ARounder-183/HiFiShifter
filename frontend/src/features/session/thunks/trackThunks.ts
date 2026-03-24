import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import type { SessionState } from "../sessionSlice";

export const setTrackStateRemote = createAsyncThunk(
    "session/setTrackStateRemote",
    async (payload: {
        trackId: string;
        muted?: boolean;
        solo?: boolean;
        volume?: number;
        composeEnabled?: boolean;
        pitchAnalysisAlgo?: string;
        color?: string;
        name?: string;
    }) => {
        return webApi.setTrackState(payload);
    },
);

export const removeSelectedClipRemote = createAsyncThunk(
    "session/removeSelectedClipRemote",
    async (_, { getState, rejectWithValue }) => {
        const state = getState() as { session: SessionState };
        const selectedClipId = state.session.selectedClipId;
        if (!selectedClipId) {
            return rejectWithValue("no_selected_clip");
        }
        return webApi.removeClip(selectedClipId);
    },
);
