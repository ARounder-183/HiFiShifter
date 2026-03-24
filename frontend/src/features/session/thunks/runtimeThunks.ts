import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import { settingsApi } from "../../../services/api";
import type { SessionState } from "../sessionSlice";

export const refreshRuntime = createAsyncThunk(
    "session/refreshRuntime",
    async () => {
        return webApi.getRuntimeInfo();
    },
);

export const clearWaveformCacheRemote = createAsyncThunk(
    "session/clearWaveformCacheRemote",
    async () => {
        return webApi.clearWaveformCache();
    },
);

export const loadUiSettings = createAsyncThunk(
    "session/loadUiSettings",
    async () => {
        return settingsApi.getUiSettings();
    },
);

/** Read current UI toggle state from Redux and persist to backend config. */
export const persistUiSettings = createAsyncThunk(
    "session/persistUiSettings",
    async (_, { getState }) => {
        const s = (getState() as { session: SessionState }).session;
        return settingsApi.saveUiSettings({
            autoCrossfade: s.autoCrossfadeEnabled,
            gridSnap: s.gridSnapEnabled,
            pitchSnap: s.pitchSnapEnabled,
            pitchSnapUnit: s.pitchSnapUnit,
            pitchSnapToleranceCents: s.pitchSnapToleranceCents,
            scaleHighlightMode: s.scaleHighlightMode,
            playheadZoom: s.playheadZoomEnabled,
            autoScroll: s.autoScrollEnabled,
            paramEditorSeekPlayhead: s.paramEditorSeekPlayheadEnabled,
            showClipboardPreview: s.showClipboardPreview,
            showParamValuePopup: s.showParamValuePopup,
            lockParamLines: s.lockParamLinesEnabled,
            dragDirection: s.selectDragDirection,
            selectDragDirection: s.selectDragDirection,
            drawDragDirection: s.drawDragDirection,
            lineVibratoDragDirection: s.lineVibratoDragDirection,
            smoothnessPercent: s.edgeSmoothnessPercent,
            customScalePresets: s.customScalePresets,
        });
    },
);
