/* eslint-disable @typescript-eslint/no-explicit-any */

// 注意：该文件作为“门面层（Facade）”保留历史接口 `webApi`，
// 以兼容现有调用方（例如 sessionSlice / 各类面板组件）。
//
// 新代码规范：
// - 具体后端命令调用应收口到 `frontend/src/services/api/*` 分组模块
// - 统一通过 `frontend/src/services/invoke.ts::invoke` 处理 Tauri/pywebview 兼容与错误包装

import type {
    ModelConfigResult,
    PlaybackStateResult,
    ParamFramesPayload,
    ProcessAudioResult,
    RuntimeInfo,
    SynthesizeResult,
    TrackSummaryResult,
    TimelineResult,
    WaveformPeaksSegmentPayload,
} from "../types/api";

import { coreApi, paramsApi, projectApi, timelineApi, waveformApi } from "./api";

export const webApi = {
    // Core
    ping: coreApi.ping,
    getRuntimeInfo: coreApi.getRuntimeInfo,
    getPlaybackState: coreApi.getPlaybackState,
    openAudioDialog: coreApi.openAudioDialog,
    openAudioDialogMultiple: coreApi.openAudioDialogMultiple,
    pickOutputPath: coreApi.pickOutputPath,
    closeWindow: coreApi.closeWindow,

    clearWaveformCache: coreApi.clearWaveformCache,

    // Model / processing
    loadDefaultModel: coreApi.loadDefaultModel,
    loadModel: coreApi.loadModel,
    processAudio: coreApi.processAudio,
    setPitchShift: coreApi.setPitchShift,
    synthesize: coreApi.synthesize,
    saveSynthesized: coreApi.saveSynthesized,
    saveSeparated: coreApi.saveSeparated,
    exportAudioAdvanced: coreApi.exportAudioAdvanced,
    playOriginal: coreApi.playOriginal,
    stopAudio: coreApi.stopAudio,

    // Undo/Redo (backend-authoritative)
    undoTimeline: timelineApi.undoTimeline,
    redoTimeline: timelineApi.redoTimeline,
    beginUndoGroup: timelineApi.beginUndoGroup,
    endUndoGroup: timelineApi.endUndoGroup,

    // Project
    getProjectMeta: projectApi.getProjectMeta,
    newProject: projectApi.newProject,
    openProjectDialog: projectApi.openProjectDialog,
    openProject: projectApi.openProject,
    saveProject: projectApi.saveProject,
    saveProjectAs: projectApi.saveProjectAs,
    setProjectBaseScale: projectApi.setProjectBaseScale,
    setProjectCustomScale: projectApi.setProjectCustomScale,
    setProjectTimelineSettings: projectApi.setProjectTimelineSettings,

    openVocalShifterDialog: projectApi.openVocalShifterDialog,
    importVocalShifterProject: projectApi.importVocalShifterProject,

    openReaperDialog: projectApi.openReaperDialog,
    importReaperProject: projectApi.importReaperProject,

    // Waveform peaks (Mix)
    getRootMixWaveformPeaksSegment: waveformApi.getRootMixWaveformPeaksSegment,
    getTrackMixWaveformPeaksSegment: waveformApi.getTrackMixWaveformPeaksSegment,

    // Param curves (frame-based)
    getParamFrames: paramsApi.getParamFrames,
    setParamFrames: paramsApi.setParamFrames,
    restoreParamFrames: paramsApi.restoreParamFrames,
    pasteVocalShifterClipboard: paramsApi.pasteVocalShifterClipboard,
    pasteReaperClipboard: paramsApi.pasteReaperClipboard,

    // Timeline
    getTimelineState: timelineApi.getTimelineState,
    importAudioItem: timelineApi.importAudioItem,
    importAudioBytes: timelineApi.importAudioBytes,

    addTrack: timelineApi.addTrack,
    addTrackNested: timelineApi.addTrackNested,
    removeTrack: timelineApi.removeTrack,
    duplicateTrack: timelineApi.duplicateTrack,
    moveTrack: timelineApi.moveTrack,
    setTrackState: timelineApi.setTrackState,
    selectTrack: timelineApi.selectTrack,
    getTrackSummary: timelineApi.getTrackSummary,

    addClip: timelineApi.addClip,
    removeClip: timelineApi.removeClip,
    removeClips: timelineApi.removeClips,
    moveClip: timelineApi.moveClip,
    moveClips: timelineApi.moveClips,
    getClipLinkedParams: timelineApi.getClipLinkedParams,
    applyClipLinkedParams: timelineApi.applyClipLinkedParams,
    setClipState: timelineApi.setClipState,
    replaceClipSource: timelineApi.replaceClipSource,
    splitClip: timelineApi.splitClip,
    glueClips: timelineApi.glueClips,
    selectClip: timelineApi.selectClip,

    setTransport: timelineApi.setTransport,
    setProjectLength: timelineApi.setProjectLength,
};

// 保留旧类型导入的“锚点”，以降低大范围改动时的冲突概率（不影响运行时）。
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const __webApiTypeAnchors = {
    ModelConfigResult: null as unknown as ModelConfigResult,
    PlaybackStateResult: null as unknown as PlaybackStateResult,
    ParamFramesPayload: null as unknown as ParamFramesPayload,
    ProcessAudioResult: null as unknown as ProcessAudioResult,
    RuntimeInfo: null as unknown as RuntimeInfo,
    SynthesizeResult: null as unknown as SynthesizeResult,
    TrackSummaryResult: null as unknown as TrackSummaryResult,
    TimelineResult: null as unknown as TimelineResult,
    WaveformPeaksSegmentPayload: null as unknown as WaveformPeaksSegmentPayload,
} as const;

void __webApiTypeAnchors;
