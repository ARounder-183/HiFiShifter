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

/**
 * 清除全部渲染缓存（磁盘）。
 *
 * 与「清除波形缓存」不同，渲染缓存没有前端镜像需要在本地同步清理；清理只
 * 删除磁盘文件，正在播放/等待渲染的片段仍持有内存 PCM，因此不会中断播放。
 */
export const clearRenderCacheRemote = createAsyncThunk(
    "session/clearRenderCacheRemote",
    async () => {
        return webApi.clearRenderCache("all");
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
            showAllTakes: s.showAllTakes,
            syncEditsAcrossTakes: s.syncEditsAcrossTakes,
            loopNewClips: s.loopNewClipsEnabled,
            splitTransitionEnabled: s.splitTransitionEnabled,
            splitTransitionMode: s.splitTransitionMode,
            splitTransitionDurationUnit: s.splitTransitionDurationUnit,
            splitTransitionDurationSec: s.splitTransitionDurationSec,
            splitTransitionDurationPercent: s.splitTransitionDurationPercent,
            splitTransitionCurve: s.splitTransitionCurve,
            splitTransitionOverlapCrossfade: s.splitTransitionOverlapCrossfade,
            snapEnabled: s.snapEnabled,
            timelineSnap: s.timelineSnap,
            tempoMapVisible: s.tempoMapVisible,
            primaryTimeUnit: s.primaryTimeUnit,
            secondaryTimeUnit: s.secondaryTimeUnit,
            rulerLabelSpacingPx: s.rulerLabelSpacingPx,
            showPlayheadTimeInTrackHeader: s.showPlayheadTimeInTrackHeader,
            paramEditorSyncTimeline: s.paramEditorSyncTimeline,
            paramEditorTimelineClickSelectTrack: s.paramEditorTimelineClickSelectTrackEnabled,
            pitchSnap: s.pitchSnapEnabled,
            pitchSnapUnit: s.pitchSnapUnit,
            pitchSnapScale: s.pitchSnapScale,
            pitchSnapToleranceCents: s.pitchSnapToleranceCents,
            scaleHighlightMode: s.scaleHighlightMode,
            ignoreGrouping: s.ignoreGrouping,
            rippleMode: s.rippleMode,
            playheadZoom: s.playheadZoomEnabled,
            autoScroll: s.autoScrollEnabled,
            paramEditorSeekPlayhead: s.paramEditorSeekPlayheadEnabled,
            showClipboardPreview: s.showClipboardPreview,
            showParamValuePopup: s.showParamValuePopup,
            paramAxisUnits: s.paramAxisUnits,
            lockParamLines: s.lockParamLinesEnabled,
            metronomeEnabled: s.metronomeEnabled,
            metronomeGain: s.metronomeGain,
            metronomeMode: s.metronomeMode,
            metronomeAccent: s.metronomeAccent,
            metronomeSound: s.metronomeSound,
            silenceDetectOptions: s.silenceDetectOptions,
            quickSearchAutoNormalize: s.quickSearchAutoNormalizeEnabled,
            saveUndoHistoryByDefault: s.saveUndoHistoryByDefault,
            visibleReferenceRootTrackIds: s.visibleReferenceRootTrackIds,
            defaultStretchAlgorithm: s.defaultStretchAlgorithm,
            defaultHifiganMelStretch: s.defaultHifiganMelStretch,
            ortEp: s.ortEp,
            gpuDeviceId: s.gpuDeviceId,
            ortDeviceId: s.ortDeviceId,
            autoBackgroundRender: s.autoBackgroundRender,
            renderCache: s.renderCache,
            selectDragDirection: s.selectDragDirection,
            drawDragDirection: s.drawDragDirection,

            lineVibratoDragDirection: s.lineVibratoDragDirection,
            smoothnessPercent: s.edgeSmoothnessPercent,
            customScalePresets: s.customScalePresets,
        });
    },
);
