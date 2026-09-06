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

import { coreApi, paramsApi, projectApi, recordingApi, timelineApi, waveformApi } from "./api";

export const webApi = {
    // Core
    ping: coreApi.ping,
    getRuntimeInfo: coreApi.getRuntimeInfo,
    getPlaybackState: coreApi.getPlaybackState,
    openAudioDialog: coreApi.openAudioDialog,
    openAudioDialogMultiple: coreApi.openAudioDialogMultiple,
    openAudioDialogForSource: coreApi.openAudioDialogForSource,
    openMidiDialog: coreApi.openMidiDialog,
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
    startBackgroundRender: coreApi.startBackgroundRender,
    cancelBackgroundRender: coreApi.cancelBackgroundRender,

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
    saveProjectToPath: projectApi.saveProjectToPath,
    setProjectBaseScale: projectApi.setProjectBaseScale,
    setProjectCustomScale: projectApi.setProjectCustomScale,
    setProjectStretchSettings: projectApi.setProjectStretchSettings,
    setProjectTimelineSettings: projectApi.setProjectTimelineSettings,

    getRecordingSettings: recordingApi.getSettings,
    saveRecordingSettings: recordingApi.saveSettings,
    getRecordingDevices: recordingApi.getDevices,
    getRecordingApps: recordingApi.getApps,
    startRecording: recordingApi.startRecording,
    stopRecording: recordingApi.stopRecording,
    getRecordingState: recordingApi.getState,

    openVocalShifterDialog: projectApi.openVocalShifterDialog,
    importVocalShifterProject: projectApi.importVocalShifterProject,

    openReaperDialog: projectApi.openReaperDialog,
    importReaperProject: projectApi.importReaperProject,
    importProjectDialog: projectApi.importProjectDialog,
    importProject: projectApi.importProject,

    // Waveform peaks (Mix)
    getRootMixWaveformPeaksSegment: waveformApi.getRootMixWaveformPeaksSegment,
    getTrackMixWaveformPeaksSegment: waveformApi.getTrackMixWaveformPeaksSegment,

    // Param curves (frame-based)
    getParamFrames: paramsApi.getParamFrames,
    setParamFrames: paramsApi.setParamFrames,
    restoreParamFrames: paramsApi.restoreParamFrames,
    stretchTrackLinkedParams: paramsApi.stretchTrackLinkedParams,
    pasteVocalShifterClipboard: paramsApi.pasteVocalShifterClipboard,
    pasteReaperClipboard: paramsApi.pasteReaperClipboard,

    // Timeline
    getTimelineState: timelineApi.getTimelineState,
    importAudioItem: timelineApi.importAudioItem,
    importAudioBytes: timelineApi.importAudioBytes,
    importMidiAsClip: paramsApi.importMidiAsClip,
    replaceMidiClipData: paramsApi.replaceMidiClipData,
    getMidiTracks: paramsApi.getMidiTracks,
    readMidiClipboardToMemory: paramsApi.readMidiClipboardToMemory,

    addTrack: timelineApi.addTrack,
    addTrackNested: timelineApi.addTrackNested,
    removeTrack: timelineApi.removeTrack,
    duplicateTrack: timelineApi.duplicateTrack,
    moveTrack: timelineApi.moveTrack,
    setTrackState: timelineApi.setTrackState,
    selectTrack: timelineApi.selectTrack,
    getTrackSummary: timelineApi.getTrackSummary,

    addClip: timelineApi.addClip,
    createClipsBulk: timelineApi.createClipsBulk,
    removeClip: timelineApi.removeClip,
    removeClips: timelineApi.removeClips,
    moveClip: timelineApi.moveClip,
    moveClips: timelineApi.moveClips,
    getClipLinkedParams: timelineApi.getClipLinkedParams,
    applyClipLinkedParams: timelineApi.applyClipLinkedParams,
    setClipState: timelineApi.setClipState,
    setClipsStateBulk: timelineApi.setClipsStateBulk,
    setClipActiveTake: timelineApi.setClipActiveTake,
    cycleClipTakes: timelineApi.cycleClipTakes,
    packClipsIntoTakes: timelineApi.packClipsIntoTakes,
    explodeClipTakes: timelineApi.explodeClipTakes,
    duplicateClipTake: timelineApi.duplicateClipTake,
    removeClipTake: timelineApi.removeClipTake,
    renameClipTake: timelineApi.renameClipTake,
    setClipTakeReversed: timelineApi.setClipTakeReversed,
    addClipTakeFromMedia: timelineApi.addClipTakeFromMedia,
    importMediaFilesAsTakes: timelineApi.importMediaFilesAsTakes,
    duplicateClipsBulk: timelineApi.duplicateClipsBulk,
    replaceClipSource: timelineApi.replaceClipSource,
    searchSourceFileReplacements: timelineApi.searchSourceFileReplacements,
    splitClip: timelineApi.splitClip,
    splitClipsAt: timelineApi.splitClipsAt,
    glueClips: timelineApi.glueClips,
    groupClips: timelineApi.groupClips,
    ungroupClips: timelineApi.ungroupClips,
    toggleGroupDisabled: timelineApi.toggleGroupDisabled,
    convertClipsToPitchReference: timelineApi.convertClipsToPitchReference,
    updatePitchReference: timelineApi.updatePitchReference,
    selectClip: timelineApi.selectClip,
    copyTimelineClips: timelineApi.copyTimelineClips,
    copyTimelineTracks: timelineApi.copyTimelineTracks,
    pasteTimelineClipboard: timelineApi.pasteTimelineClipboard,
    hasTimelineClipboard: timelineApi.hasTimelineClipboard,
    clipboardKind: timelineApi.clipboardKind,
    hasReaperClipboard: timelineApi.hasReaperClipboard,

    // 检查已导入媒体源文件是否被外部修改或删除
    checkSourceFilesChanged: timelineApi.checkSourceFilesChanged,

    setTransport: timelineApi.setTransport,
    setProjectLength: timelineApi.setProjectLength,
};

// 保留旧类型导入的“锚点”，以降低大范围改动时的冲突概率（不影响运行时）。
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
