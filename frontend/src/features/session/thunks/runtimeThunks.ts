/**
 * 运行时相关异步 thunk。
 *
 * 负责刷新运行时状态、清理波形缓存，以及读写 UI 持久化设置。
 */
import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";
import { settingsApi } from "../../../services/api";
import { waveformMipmapStore } from "../../../utils/waveformMipmapStore";
import type { SessionState } from "../sessionSlice";

export const refreshRuntime = createAsyncThunk("session/refreshRuntime", async () => {
    return webApi.getRuntimeInfo();
});

export const clearWaveformCacheRemote = createAsyncThunk(
    "session/clearWaveformCacheRemote",
    async () => {
        const result = await webApi.clearWaveformCache();
        // 同步清除前端内存中的 mipmap 缓存，确保后端磁盘缓存和前端内存缓存一致
        waveformMipmapStore.clear();
        return result;
    },
);

export const loadUiSettings = createAsyncThunk("session/loadUiSettings", async () => {
    return settingsApi.getUiSettings();
});

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
            selectDragDirection: s.selectDragDirection,
            drawDragDirection: s.drawDragDirection,

            lineVibratoDragDirection: s.lineVibratoDragDirection,
            smoothnessPercent: s.edgeSmoothnessPercent,
            customScalePresets: s.customScalePresets,
        });
    },
);
