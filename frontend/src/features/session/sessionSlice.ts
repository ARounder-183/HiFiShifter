// Session 同步 reducers。拆自 sessionSlice，按原顺序汇总。
import { createSlice, createAsyncThunk, type PayloadAction } from "@reduxjs/toolkit";
import { isPluginMode } from "../../services/hostCapabilities";
import type { HistoryRecordSummary, TimelineState } from "../../types/api";
import type {
    ClipFormantAnalysisState,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    PitchSnapUnit,
    SplitTransitionCurveType,
    TimelineSnapSettings,
    TimeUnit,
    TimeUnitChoice,
    SilenceDetectSettings,
    TrackMeterInfo,
    ToolMode,
} from "./sessionTypes";
import { normalizeSearchSettings, type SearchSettings } from "../search/searchSettings";
import {
    DEFAULT_SORT_DESCENDING,
    normalizeFileBrowserViewOptions,
    type FileBrowserViewOptions,
} from "../fileBrowser/fileBrowserViewOptions";
import {
    normalizeFolderImportOptions,
    type FolderImportOptions,
} from "../fileBrowser/folderImportOptions";
import { clampGain } from "../../utils/gainDb";
import { persistUiSettings } from "./thunks/runtimeThunks";
import {
    type ChannelImportPolicy,
    type PenInputSettings,
    type RenderCacheSettings,
} from "../../services/api/settings";
import type { ScaleLike } from "../../utils/musicalScales";
import {
    nextParamAxisUnit,
    resolveParamAxisUnit,
    supportsParamAxisUnit,
} from "../../components/layout/pianoRoll/paramAxisUnits";
import type { CustomScalePreset } from "../../utils/customScales";
import { sanitizeCustomScalePreset } from "../../utils/customScales";
import {
    MAX_VIBRATO_PRESETS,
    isBuiltinVibratoPresetId,
    sanitizeVibratoPreset,
} from "../../features/vibrato/vibratoPresets";
import {
    activeIdAfterRemoval,
    cycleVibratoPresetId,
    effectiveBuiltinPresetOrder,
    enabledVibratoPresets,
    reorderBuiltinPresetIds,
    reorderUserVibratoPresets,
    resolveVibratoPresets,
} from "../../features/vibrato/vibratoPresetList";
import { SYSTEM_VIBRATO_PRESETS } from "../../features/vibrato/systemPresets";
import type { BaselineMode, VibratoPreset } from "../../features/vibrato/vibratoTypes";
import type { TempoMap } from "../../utils/tempoMap";
import { clampDenominator, normalizeTempoMap } from "../../utils/tempoMap";
import { markProjectDirty } from "./sessionDirtyState";
import type { ClipColor, SessionState, StretchAlgorithmOption } from "./sessionSlice/types";
import { resolveTrackIdForClipSelection } from "./selectionFocus";
import { applyHistoryDepths } from "./sessionSlice/history";
import { applyTimelineState } from "./sessionSlice/helpersTimeline";
import { intentPlayheadSec } from "./sessionSlice/playheadIntent";
import {
    applyToolMode,
    normalizeTimelineSnapSettings,
    VALID_TIME_UNITS,
    MAX_TRACK_VOLUME,
} from "./sessionSlice/settings";
import { clamp, normalizeClipColor } from "./sessionSlice/helpersCore";
import { getClipRateMultiplier, updateActiveTakeFromFlat } from "./sessionSlice/helpersTakes";
import { initialState } from "./sessionSlice/initialState";
import { registerRuntime } from "./sessionSlice/extraReducers/runtime";
import { registerAudio } from "./sessionSlice/extraReducers/audio";
import { registerTransport } from "./sessionSlice/extraReducers/transport";
import { registerHistory } from "./sessionSlice/extraReducers/history";
import { registerProject } from "./sessionSlice/extraReducers/project";
import { registerTimelineEdits } from "./sessionSlice/extraReducers/timelineEdits";
import { registerClipState } from "./sessionSlice/extraReducers/clipState";
import { registerTrack } from "./sessionSlice/extraReducers/track";
import { registerMatchers } from "./sessionSlice/extraReducers/matchers";
export { rejectedMessage } from "./sessionSlice/helpersCore";
export type { FadeCurveType } from "./sessionTypes";
export type {
    ClipInfo,
    ClipTemplate,
    DrawToolMode,
    DragDirection,
    DrawDragDirection,
    EditParam,
    GridSize,
    ToolMode,
    ToolModeGroup,
    TrackInfo,
} from "./sessionTypes";
export type { ParamSelectionSnapshot, SessionState } from "./sessionSlice/types";

export {
    undoRemote,
    redoRemote,
    setHistoryPositionRemote,
    setProjectSaveUndoHistoryRemote,
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    saveProjectToPathRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectStretchSettingsRemote,
    setProjectTimelineSettingsRemote,
} from "./thunks/projectThunks";

export {
    fetchTimeline,
    seekPlayhead,
    updateTransportBpm,
    syncPlaybackState,
    playOriginal,
    stopAudioPlayback,
} from "./thunks/transportThunks";

export {
    addTrackRemote,
    createHostTrackRemote,
    removeTrackRemote,
    moveTrackRemote,
    selectTrackRemote,
    setProjectLengthRemote,
    fetchSelectedTrackSummary,
    addClipOnTrack,
    createClipsRemote,
    pasteTimelineClipboardRemote,
    removeClipRemote,
    removeClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    closeTrackGapsRemote,
    analyzeSilenceRemote,
    removeSilenceRemote,
    duplicateClipsBulkRemote,
    duplicateTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipActiveTakeRemote,
    cycleClipTakesRemote,
    packClipsIntoTakesRemote,
    explodeClipTakesRemote,
    duplicateClipTakeRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipTakeReversedRemote,
    setClipTakeChannelModeRemote,
    scanAndConvertFakeStereoRemote,
    addClipTakeFromMediaRemote,
    replaceClipSourceRemote,
    replaceMidiClipDataRemote,
    splitClipRemote,
    splitClipsAtRemote,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    glueClipsRemote,
    selectClipRemote,
} from "./thunks/timelineThunks";

export { setTrackStateRemote, removeSelectedClipRemote } from "./thunks/trackThunks";

export {
    refreshRuntime,
    clearWaveformCacheRemote,
    clearRenderCacheRemote,
    loadUiSettings,
    persistUiSettings,
} from "./thunks/runtimeThunks";

export { updateMetronome } from "./thunks/transportThunks";
export { loadModel, loadDefaultModel } from "./thunks/modelThunks";

export {
    processAudio,
    pickOutputPath,
    applyPitchShift,
    synthesizeAudio,
    exportAudio,
    exportAudioAdvanced,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
} from "./thunks/audioThunks";

export {
    importAudioFromDialog,
    importAudioFromPath,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMidiAsClip,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks";

export { createDefaultTimelineSnapSettings } from "./sessionSlice/settings";

const sessionSlice = createSlice({
    name: "session",
    initialState: () =>
        isPluginMode() ? { ...initialState, tracks: [], selectedTrackId: null } : initialState,
    reducers: {
        /**
         * 标记连续交互开始（拖动/滑动等）。
         * 在交互期间，连续操作类 thunk 的 fulfilled handler 会跳过 applyTimelineState()。
         */
        beginInteraction(state) {
            state._interactionLockCount = Math.max(0, state._interactionLockCount) + 1;
            if (isPluginMode()) state._pluginTimelineEpoch += 1;
        },
        /**
         * 标记连续交互结束。计数器归零后恢复正常的后端状态同步。
         */
        endInteraction(state) {
            state._interactionLockCount = Math.max(0, state._interactionLockCount - 1);
            if (isPluginMode()) state._pluginTimelineEpoch += 1;
        },
        /** 乐观更新轨道名称（立即反映到 UI，不等后端响应） */
        setTrackName(state, action: PayloadAction<{ trackId: string; name: string }>) {
            const track = state.tracks.find((entry) => entry.id === action.payload.trackId);
            if (track) {
                track.name = action.payload.name;
                // 同上：轨道名是工程数据。
                markProjectDirty(state.project);
            }
        },
        setTrackMeters(state, action: PayloadAction<Record<string, TrackMeterInfo>>) {
            state.trackMeters = action.payload;
        },
        clearTrackMeters(state) {
            state.trackMeters = {};
        },
        /**
         * 同步「操作记录」与撤销/重做可用性（`history_state` 后端事件 /
         * 挂载时的 get_history_state）。深度为 0 时对应的菜单项置灰、
         * 快捷键静默失败。
         */
        setHistoryState(
            state,
            action: PayloadAction<{
                undoDepth: number;
                redoDepth: number;
                backend?: "local" | "host";
                records?: HistoryRecordSummary[];
            }>,
        ) {
            applyHistoryDepths(state, action.payload.undoDepth, action.payload.redoDepth);
            const records = action.payload.records;
            if (Array.isArray(records)) {
                // 把顶层 `backend` 盖到每条记录上：`labelOf` 要按**记录来源**决定是否
                // 加 `history_op_` 前缀（宿主字符串原样显示，本地 op key 才查表）。
                const backend = action.payload.backend ?? "local";
                state.historyRecords = records.map((record) => ({
                    ...record,
                    backend: record.backend ?? backend,
                }));
            }
        },
        checkpointHistory(state) {
            // 撤销/重做由后端权威管理（undo_timeline / redo_timeline 返回完整
            // 时间线快照）。前端不再维护平行的快照栈——旧实现的本检查点与
            // 后端检查点在深度与内容上都可能错位，撤销时会先乐观渲染出任意
            // 旧状态、再被后端快照纠正，造成轨道视图闪屏。此 action 现在
            // 仅负责：1) 标记工程已修改；2) 递增 paramsEpoch 让参数编辑器
            // 重新取数（如 shiftParam 系列直接写曲线后的刷新）。
            markProjectDirty(state.project);
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        /** 供录音等后端直接导入时间轴的命令同步完整快照。 */
        applyTimelinePayload(state, action: PayloadAction<TimelineState>) {
            if (isPluginMode()) state._pluginTimelineEpoch += 1;
            // 后端直接导入的权威快照（如录音导入）：采纳其后端播放头。
            applyTimelineState(state, action.payload, { force: true, adoptPlayhead: true });
        },
        bumpParamsEpoch(state) {
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        setToolMode(state, action: PayloadAction<ToolMode>) {
            // 工具切换是纯视图状态：不修改工程内容。
            // 不入 undo 历史（否则 Ctrl+Z 会先回退一次"不存在的内容变更"），
            // 也不标记 project.dirty（否则仅切工具就会触发"未保存"退出确认）。
            applyToolMode(state, action.payload);
        },
        setEditParam(state, action: PayloadAction<EditParam>) {
            state.editParam = action.payload;
        },
        setBpm(state, action: PayloadAction<number>) {
            // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
            state.bpm = clamp(action.payload, 10, 960);
        },
        setBeats(state, action: PayloadAction<number>) {
            state.beats = clamp(action.payload, 1, 32);
        },
        setGrid(state, action: PayloadAction<GridSize>) {
            state.grid = action.payload;
        },
        setPrimaryTimeUnit(state, action: PayloadAction<TimeUnit>) {
            if (VALID_TIME_UNITS.has(action.payload)) {
                state.primaryTimeUnit = action.payload;
            }
        },
        setSecondaryTimeUnit(state, action: PayloadAction<TimeUnitChoice>) {
            if (action.payload === "none" || VALID_TIME_UNITS.has(action.payload)) {
                state.secondaryTimeUnit = action.payload;
            }
        },
        setRulerLabelSpacingPx(state, action: PayloadAction<number>) {
            state.rulerLabelSpacingPx = Math.max(
                40,
                Math.min(320, Math.round(Number(action.payload) || 110)),
            );
        },
        setShowPlayheadTimeInTrackHeader(state, action: PayloadAction<boolean>) {
            state.showPlayheadTimeInTrackHeader = Boolean(action.payload);
        },
        setParamEditorSyncTimeline(state, action: PayloadAction<boolean>) {
            state.paramEditorSyncTimeline = Boolean(action.payload);
        },
        toggleAutoCrossfade(state) {
            state.autoCrossfadeEnabled = !state.autoCrossfadeEnabled;
        },
        toggleShowAllTakes(state) {
            state.showAllTakes = !state.showAllTakes;
        },
        toggleSyncEditsAcrossTakes(state) {
            state.syncEditsAcrossTakes = !state.syncEditsAcrossTakes;
        },
        toggleSplitTransition(state) {
            state.splitTransitionEnabled = !state.splitTransitionEnabled;
        },
        setSplitTransitionMode(state, action: PayloadAction<"fade" | "overlap">) {
            state.splitTransitionMode = action.payload;
        },
        setSplitTransitionDurationUnit(state, action: PayloadAction<"seconds" | "percent">) {
            state.splitTransitionDurationUnit = action.payload;
        },
        setSplitTransitionDurationSec(state, action: PayloadAction<number>) {
            state.splitTransitionDurationSec = clamp(Number(action.payload) || 0.01, 0.001, 10);
        },
        setSplitTransitionDurationPercent(state, action: PayloadAction<number>) {
            state.splitTransitionDurationPercent = clamp(Number(action.payload) || 1, 0.01, 100);
        },
        setSplitTransitionCurve(state, action: PayloadAction<SplitTransitionCurveType>) {
            state.splitTransitionCurve = action.payload;
        },
        setSplitTransitionOverlapCrossfade(state, action: PayloadAction<"auto" | "always">) {
            state.splitTransitionOverlapCrossfade = action.payload;
        },
        toggleSnap(state) {
            const next = !state.timelineSnap.enabled;
            state.timelineSnap.enabled = next;
            state.snapEnabled = next;
        },
        setTimelineSnapSettings(state, action: PayloadAction<Partial<TimelineSnapSettings>>) {
            state.timelineSnap = normalizeTimelineSnapSettings(state.timelineSnap, action.payload);
            state.snapEnabled = state.timelineSnap.enabled;
        },
        togglePitchSnap(state) {
            state.pitchSnapEnabled = !state.pitchSnapEnabled;
        },
        setPitchSnapUnit(state, action: PayloadAction<PitchSnapUnit>) {
            state.pitchSnapUnit = action.payload;
        },
        setPitchSnapScale(
            state,
            action: PayloadAction<import("../../utils/musicalScales").ScaleKey>,
        ) {
            state.pitchSnapScale = action.payload;
        },
        setPitchSnapToleranceCents(state, action: PayloadAction<number>) {
            state.pitchSnapToleranceCents = clamp(action.payload, 0, 1000);
        },
        setScaleHighlightMode(state, action: PayloadAction<"always" | "off">) {
            state.scaleHighlightMode = action.payload;
        },
        setTempoMap(state, action: PayloadAction<TempoMap | null>) {
            const projectScale: ScaleLike | null =
                state.project.useCustomScale && state.project.customScale
                    ? state.project.customScale.notes
                    : state.project.baseScale;
            state.tempoMap = normalizeTempoMap(action.payload, state.bpm, state.beats || 4, {
                projectScale: projectScale ?? undefined,
                projectScaleName: state.project.useCustomScale
                    ? (state.project.customScale?.name ?? undefined)
                    : undefined,
                projectDenominator: state.project.timeSignatureDenominator,
            });
            // 保持工程基准值（bpm / 拍号）与 0 位置点一致，删除 Tempo Map 后回退一致。
            if (state.tempoMap) {
                const first = state.tempoMap.points[0];
                state.bpm = clamp(first.bpm, 10, 960);
                const firstSig = first.timeSignature ?? { numerator: 4, denominator: 4 };
                state.beats = Math.min(32, Math.max(1, Math.round(firstSig.numerator)));
                state.project.timeSignatureDenominator = clampDenominator(firstSig.denominator);
            }
        },
        setTempoMapVisible(state, action: PayloadAction<boolean>) {
            state.tempoMapVisible = action.payload;
        },
        toggleTempoMapVisible(state) {
            state.tempoMapVisible = !state.tempoMapVisible;
        },
        upsertCustomScalePreset(state, action: PayloadAction<CustomScalePreset>) {
            const incoming = sanitizeCustomScalePreset(action.payload);
            const idx = state.customScalePresets.findIndex((preset) => preset.id === incoming.id);
            if (idx >= 0) {
                state.customScalePresets[idx] = incoming;
            } else {
                state.customScalePresets.push(incoming);
            }
        },
        removeCustomScalePreset(state, action: PayloadAction<string>) {
            const presetId = action.payload;
            state.customScalePresets = state.customScalePresets.filter(
                (preset) => preset.id !== presetId,
            );
        },
        /** 新增或覆盖一个用户颤音预设（按 id 匹配）。 */
        upsertVibratoPreset(state, action: PayloadAction<VibratoPreset>) {
            const incoming = sanitizeVibratoPreset(action.payload);
            // 系统预设只能来自代码；混进用户列表会让"只读"失效。
            if (isBuiltinVibratoPresetId(incoming.id)) return;
            const idx = state.vibratoPresets.findIndex((preset) => preset.id === incoming.id);
            if (idx >= 0) {
                state.vibratoPresets[idx] = incoming;
                return;
            }
            if (state.vibratoPresets.length >= MAX_VIBRATO_PRESETS) return;
            state.vibratoPresets.push(incoming);
        },
        /**
         * 删除一个用户颤音预设，并把活动 id 迁移到一个仍然存在的预设上。
         *
         * 【为什么在这里迁移】活动 id 指向被删的预设会让"当前预设"悬空。迁移
         * 规则收在 `activeIdAfterRemoval` 里（纯函数、可单测），与预设编辑器
         * 各自维护一份相比不会漂移。
         */
        removeVibratoPreset(state, action: PayloadAction<string>) {
            const presetId = action.payload;
            const all = [...SYSTEM_VIBRATO_PRESETS, ...state.vibratoPresets];
            state.vibratoPresets = state.vibratoPresets.filter((preset) => preset.id !== presetId);
            state.activeVibratoPresetId = activeIdAfterRemoval(
                all,
                state.activeVibratoPresetId,
                presetId,
            );
            // 顺手把停用名单里的这条清掉：id 是随机生成的，留着不会误伤别的预设，
            // 但会让持久化的名单随着"建了又删"一直变长。
            state.disabledVibratoPresetIds = state.disabledVibratoPresetIds.filter(
                (id) => id !== presetId,
            );
        },
        /** 移动用户颤音预设的位置（仅用户段内部）。 */
        reorderVibratoPreset(state, action: PayloadAction<{ id: string; toIndex: number }>) {
            state.vibratoPresets = reorderUserVibratoPresets(
                state.vibratoPresets,
                action.payload.id,
                action.payload.toIndex,
            );
        },
        /**
         * 移动**系统预设**的位置。
         *
         * 顺序以 id 列表持久化：先取当前有效顺序（含出厂顺序兜底），再移动一项后整体
         * 写回 —— 这样"从未排过序"与"排过序"最终都落在同一份完整列表上。
         */
        reorderBuiltinVibratoPreset(state, action: PayloadAction<{ id: string; toIndex: number }>) {
            state.builtinVibratoPresetOrder = reorderBuiltinPresetIds(
                effectiveBuiltinPresetOrder(state.builtinVibratoPresetOrder),
                action.payload.id,
                action.payload.toIndex,
            );
        },
        /** 设定当前活动颤音预设。 */
        setActiveVibratoPreset(state, action: PayloadAction<string>) {
            state.activeVibratoPresetId = action.payload;
        },
        /**
         * 启用 / 停用一条颤音预设。
         *
         * 停用只影响本机的工具栏列表与循环切换，预设本身、以及"当前使用"的选择都
         * 不受影响 —— 停用正在用的那一条是允许的，它仍然是当前预设。
         */
        /**
         * 设置「摆放方式」（添加颤音时的参数）。
         *
         * 与预设无关，因此不碰任何预设数据；调用方随后 `persistUiSettings()` 落盘。
         */
        setVibratoBaseline(state, action: PayloadAction<BaselineMode>) {
            state.vibratoBaseline = action.payload;
        },
        toggleVibratoPresetEnabled(state, action: PayloadAction<string>) {
            const id = action.payload;
            const index = state.disabledVibratoPresetIds.indexOf(id);
            if (index >= 0) state.disabledVibratoPresetIds.splice(index, 1);
            else state.disabledVibratoPresetIds.push(id);
        },
        /**
         * 环绕切换活动颤音预设。
         *
         * `delta` 的语义与 `clip.cycleTake`、`layout.focusNext` 一致：一个实现
         * 同时服务"上一个"与"下一个"。被停用的预设**跳过**。
         */
        cycleActiveVibratoPreset(state, action: PayloadAction<1 | -1>) {
            const all = resolveVibratoPresets(
                state.vibratoPresets,
                state.builtinVibratoPresetOrder,
            ).all;
            const next = cycleVibratoPresetId(
                enabledVibratoPresets(all, state.disabledVibratoPresetIds),
                state.activeVibratoPresetId,
                action.payload,
            );
            if (next) state.activeVibratoPresetId = next;
        },
        togglePlayheadZoom(state) {
            state.playheadZoomEnabled = !state.playheadZoomEnabled;
        },
        toggleLockParamLines(state) {
            state.lockParamLinesEnabled = !state.lockParamLinesEnabled;
        },
        /** 更新节拍器配置（引擎应用 + 持久化由 thunk `updateMetronome` 负责）。 */
        setMetronomeConfig(
            state,
            action: PayloadAction<
                Partial<{
                    metronomeEnabled: boolean;
                    metronomeGain: number;
                    metronomeMode: "grid" | "beat" | "bar";
                    metronomeSound: "click" | "woodblock" | "beep";
                    metronomeAccent: boolean;
                }>
            >,
        ) {
            const p = action.payload;
            if (p.metronomeEnabled != null) state.metronomeEnabled = p.metronomeEnabled;
            if (p.metronomeGain != null)
                state.metronomeGain = Math.min(1, Math.max(0, Number(p.metronomeGain) || 0));
            if (
                p.metronomeMode === "grid" ||
                p.metronomeMode === "beat" ||
                p.metronomeMode === "bar"
            )
                state.metronomeMode = p.metronomeMode;
            if (
                p.metronomeSound === "click" ||
                p.metronomeSound === "woodblock" ||
                p.metronomeSound === "beep"
            )
                state.metronomeSound = p.metronomeSound;
            if (p.metronomeAccent != null) state.metronomeAccent = p.metronomeAccent;
        },
        /** 设置/清除静音检测预览覆盖层（对话框打开期间存在，任何时间线提交即失效）。 */
        setSilencePreview(
            state,
            action: PayloadAction<Record<string, Array<[number, number]>> | null>,
        ) {
            state.silencePreviewSegments = action.payload;
            // 清除预览（关闭 / 应用）时一并作废在途分析请求：否则对话框
            // 关闭后迟到的 fulfilled 仍通过 requestId 守卫，把过期的静音
            // 区域重新画回时间线（甚至画到切除后的新剪辑上）。
            if (action.payload === null) {
                state._silencePreviewRequestId = null;
            }
        },
        /** 更新静音检测对话框参数（持久化由调用方走 persistUiSettings）。 */
        setSilenceDetectOptions(state, action: PayloadAction<Partial<SilenceDetectSettings>>) {
            state.silenceDetectOptions = {
                ...state.silenceDetectOptions,
                ...action.payload,
            };
        },
        /**
         * 更新搜索匹配设置（部分字段；持久化由调用方走 persistUiSettings）。
         *
         * 合并后统一过一遍 `normalizeSearchSettings`：调用点只传改动的那一项，
         * 未传的项保持不变，非法值（旧配置 / 手改文件）在这里被收敛。
         */
        setSearchSettings(state, action: PayloadAction<Partial<SearchSettings>>) {
            state.searchSettings = normalizeSearchSettings({
                ...state.searchSettings,
                ...action.payload,
            });
        },
        setSearchSettingsDialogOpen(state, action: PayloadAction<boolean>) {
            state.searchSettingsDialogOpen = action.payload;
        },
        /**
         * 更新文件浏览器视图选项（部分字段；持久化由调用方走 `persistUiSettings`）。
         *
         * 合并后统一过一遍 `normalizeFileBrowserViewOptions`：调用点只传改动的那一项，
         * 非法值（旧配置 / 手改文件）在这里被收敛。
         *
         * 【切换排序依据时为什么要连带改方向】"按日期"几乎总是想看最新的，
         * 沿用上一次的方向会让它默认排成最旧在前。见 `DEFAULT_SORT_DESCENDING`。
         */
        setFileBrowserView(state, action: PayloadAction<Partial<FileBrowserViewOptions>>) {
            const patch = action.payload;
            const next = normalizeFileBrowserViewOptions({
                ...state.fileBrowserView,
                ...patch,
            });
            if (patch.sortMode !== undefined && patch.sortDescending === undefined) {
                next.sortDescending = DEFAULT_SORT_DESCENDING[patch.sortMode];
            }
            state.fileBrowserView = next;
        },
        /**
         * 目录导入选项（部分覆盖）。
         *
         * 【为什么与视图选项一样"改动即生效"】它们是**下次导入的默认值**，没有
         * "应用"这一步可言 —— 用户改完就关掉对话框，不该还需要再点一次确定。
         */
        setFolderImportOptions(state, action: PayloadAction<Partial<FolderImportOptions>>) {
            state.folderImportOptions = normalizeFolderImportOptions({
                ...state.folderImportOptions,
                ...action.payload,
            });
        },
        /** 覆盖常用位置列表（顺序即显示顺序；去重与去空白在此收口）。 */
        setFileBrowserFavorites(state, action: PayloadAction<string[]>) {
            const seen = new Set<string>();
            state.fileBrowserFavorites = action.payload
                .map((path) => path.trim())
                .filter((path) => {
                    if (!path || seen.has(path)) return false;
                    seen.add(path);
                    return true;
                });
        },
        toggleQuickSearchAutoNormalize(state) {
            state.quickSearchAutoNormalizeEnabled = !state.quickSearchAutoNormalizeEnabled;
        },
        setSaveUndoHistoryByDefault(state, action: PayloadAction<boolean>) {
            state.saveUndoHistoryByDefault = Boolean(action.payload);
        },
        setDefaultStretchAlgorithm(state, action: PayloadAction<StretchAlgorithmOption>) {
            state.defaultStretchAlgorithm = action.payload;
        },
        setDefaultHifiganMelStretch(state, action: PayloadAction<boolean>) {
            state.defaultHifiganMelStretch = action.payload;
        },
        setOrtEp(state, action: PayloadAction<string>) {
            state.ortEp = action.payload;
        },
        /** 记录 vslib 可用性（启动探测一次）。null = 未知，按不可用处理。 */
        setVslibAvailable(state, action: PayloadAction<boolean | null>) {
            state.vslibAvailable = action.payload;
        },
        setOrtDeviceKey(state, action: PayloadAction<string | null>) {
            state.ortDeviceKey = action.payload;
        },
        toggleAutoBackgroundRender(state) {
            state.autoBackgroundRender = !state.autoBackgroundRender;
        },
        /** 覆盖整块渲染缓存设置（对话框保存时调用）。 */
        setRenderCacheSettings(state, action: PayloadAction<Partial<RenderCacheSettings>>) {
            state.renderCache = { ...state.renderCache, ...action.payload };
        },
        /** 覆盖整块导入声道策略（对话框保存时调用）。 */
        setChannelImportPolicy(state, action: PayloadAction<Partial<ChannelImportPolicy>>) {
            state.channelImportPolicy = { ...state.channelImportPolicy, ...action.payload };
        },
        /** 覆盖整块指针设备偏好（设置面板保存时调用）。 */
        setPenInputSettings(state, action: PayloadAction<Partial<PenInputSettings>>) {
            state.penInput = { ...state.penInput, ...action.payload };
        },
        /** 镜像后端 `playback_rendering_state` 事件的 active/target（进度走 App 本地状态）。
         *  `blocking` 为阻塞式前台预渲染（target="original"）的独立镜像；缺省时按
         *  active+target 推导（测试夹具与旧调用方的兼容路径）。 */
        setPlaybackRenderingState(
            state,
            action: PayloadAction<{
                active: boolean;
                target: string | null;
                blocking?: boolean;
            }>,
        ) {
            state.playbackRenderingActive = action.payload.active;
            state.playbackRenderingTarget = action.payload.target;
            state.playbackBlockingRenderActive =
                action.payload.blocking ??
                (action.payload.active && action.payload.target === "original");
        },
        setVisibleReferenceRootTrackIds(state, action: PayloadAction<string[]>) {
            state.visibleReferenceRootTrackIds = Array.from(
                new Set(
                    action.payload.filter(
                        (id): id is string => typeof id === "string" && id.trim().length > 0,
                    ),
                ),
            );
        },
        toggleVisibleReferenceRootTrackId(state, action: PayloadAction<string>) {
            const trackId = action.payload;
            if (state.visibleReferenceRootTrackIds.includes(trackId)) {
                state.visibleReferenceRootTrackIds = state.visibleReferenceRootTrackIds.filter(
                    (id) => id !== trackId,
                );
                return;
            }
            const track = state.tracks.find((t) => t.id === trackId);
            if (!track || track.parentId) return;
            let currentRootId: string | null = state.selectedTrackId;
            if (currentRootId) {
                let cursor: { id: string; parentId?: string | null } | undefined =
                    state.tracks.find((t) => t.id === currentRootId);
                while (cursor?.parentId) {
                    cursor = state.tracks.find((t) => t.id === cursor!.parentId);
                }
                currentRootId = cursor?.id ?? null;
            }
            if (trackId === currentRootId) return;
            state.visibleReferenceRootTrackIds = [...state.visibleReferenceRootTrackIds, trackId];
        },
        toggleAutoScroll(state) {
            state.autoScrollEnabled = !state.autoScrollEnabled;
        },
        toggleIgnoreGrouping(state) {
            state.ignoreGrouping = !state.ignoreGrouping;
        },
        /** 循环波纹编辑模式：off → track → all → off（对应 REAPER 波纹编辑按钮的三态切换）。 */
        cycleRippleMode(state) {
            state.rippleMode =
                state.rippleMode === "off" ? "track" : state.rippleMode === "track" ? "all" : "off";
        },
        setRippleMode(state, action: PayloadAction<"off" | "track" | "all">) {
            if (
                action.payload === "off" ||
                action.payload === "track" ||
                action.payload === "all"
            ) {
                state.rippleMode = action.payload;
            }
        },
        toggleGroupDisabledLocal(state, action: PayloadAction<string>) {
            const idx = state.disabledGroupIds.indexOf(action.payload);
            if (idx >= 0) {
                state.disabledGroupIds.splice(idx, 1);
            } else {
                state.disabledGroupIds.push(action.payload);
            }
        },
        toggleParamEditorSeekPlayhead(state) {
            state.paramEditorSeekPlayheadEnabled = !state.paramEditorSeekPlayheadEnabled;
        },
        toggleParamEditorTimelineClickSelectTrack(state) {
            state.paramEditorTimelineClickSelectTrackEnabled =
                !state.paramEditorTimelineClickSelectTrackEnabled;
        },
        toggleClipboardPreview(state) {
            state.showClipboardPreview = !state.showClipboardPreview;
        },
        toggleParamValuePopup(state) {
            state.showParamValuePopup = !state.showParamValuePopup;
        },
        /**
         * 切换某参数纵轴标尺的展示单位（倍率 ↔ dB）。
         *
         * 只对支持切换的参数生效（音量 / 动态）；其余参数点击纵轴不应有任何反应，
         * 因此这里静默忽略而不是写入无效键 —— 无效键会让持久化里堆出无意义配置。
         */
        toggleParamAxisUnit(state, action: PayloadAction<string>) {
            const param = action.payload;
            if (!supportsParamAxisUnit(param)) return;
            const next = nextParamAxisUnit(resolveParamAxisUnit(state.paramAxisUnits, param));
            state.paramAxisUnits = { ...state.paramAxisUnits, [param]: next };
        },
        cycleDragDirection(state, action: PayloadAction<"select" | "draw" | "vibrato">) {
            if (action.payload === "select") {
                const order: DragDirection[] = ["free", "x-only", "y-only"];
                const idx = order.indexOf(state.selectDragDirection);
                state.selectDragDirection = order[(idx + 1) % order.length];
                return;
            }
            const order: DrawDragDirection[] = ["free", "x-only"];
            if (action.payload === "draw") {
                const idx = order.indexOf(state.drawDragDirection);
                state.drawDragDirection = order[(idx + 1) % order.length];
                return;
            }
            const idx = order.indexOf(state.lineVibratoDragDirection);
            state.lineVibratoDragDirection = order[(idx + 1) % order.length];
        },
        setDragDirection(
            state,
            action: PayloadAction<{
                tool: "select" | "draw" | "vibrato";
                direction: DragDirection | DrawDragDirection;
            }>,
        ) {
            const { tool, direction } = action.payload;
            if (tool === "select") {
                if (["free", "x-only", "y-only"].includes(direction)) {
                    state.selectDragDirection = direction as DragDirection;
                }
                return;
            }
            if (tool === "draw") {
                if (["free", "x-only"].includes(direction)) {
                    state.drawDragDirection = direction as DrawDragDirection;
                }
                return;
            }
            if (["free", "x-only"].includes(direction)) {
                state.lineVibratoDragDirection = direction as DrawDragDirection;
            }
        },
        setEdgeSmoothnessPercent(state, action: PayloadAction<number>) {
            state.edgeSmoothnessPercent = clamp(Number(action.payload) || 0, 0, 100);
        },
        setPlayheadSec(state, action: PayloadAction<number>) {
            // 【为什么这里是全仓的单一闸门】乐观写播放光标只有这一个入口，而"跳转"
            // 在非 ARA 宿主里必然失败（`set_transport` 需要 ARA 播放请求）。闸在
            // 入口上，十几处"先写乐观值再打后端"的调用点就一起被覆盖 —— 否则每一处
            // 都要自己判一次，而漏掉任何一处都会重现"先跳过去再跳回来"。
            if (!intentPlayheadSec(state, action.payload)) return;
            state.playheadLocalAtMs = performance.now();
        },
        /** 设置/清除待执行的“聚焦播放光标”请求（见 pendingPlayheadRevealSec 注释）。 */
        setPendingPlayheadReveal(state, action: PayloadAction<number | null>) {
            state.pendingPlayheadRevealSec = action.payload;
        },
        setAudioPath(state, action: PayloadAction<string>) {
            state.audioPath = action.payload;
        },
        setProjectNotesMarkdown(state, action: PayloadAction<string>) {
            const next = action.payload;
            if (state.project.notesMarkdown === next) {
                return;
            }
            state.project.notesMarkdown = next;
            state.project.dirty = true;
        },
        closeVocalShifterSkippedFilesDialog(state) {
            state.vocalShifterSkippedFilesDialog = null;
        },
        closeReaperSkippedFilesDialog(state) {
            state.reaperSkippedFilesDialog = null;
        },
        closeSaveVersionConflictDialog(state) {
            state.saveVersionConflictDialog = null;
        },
        setSelectedClip(state, action: PayloadAction<string | null>) {
            state.selectedClipId = action.payload;
            state.selectionContext = "clips";
            if (action.payload) {
                const nextTrackId = resolveTrackIdForClipSelection({
                    currentTrackId: state.selectedTrackId,
                    clips: state.clips,
                    clipId: action.payload,
                });
                if (nextTrackId !== state.selectedTrackId) {
                    state.selectedTrackId = nextTrackId;
                }
            }
        },
        setSelectedClipPreservingTrack(state, action: PayloadAction<string | null>) {
            state.selectedClipId = action.payload;
            state.selectionContext = "clips";
        },
        setMultiSelectedClipIds(state, action: PayloadAction<string[]>) {
            state.multiSelectedClipIds = action.payload;
            // 本 action 是**用户显式选择**的入口（框选 / 修饰键点击 / 范围选择 /
            // 全选 / 取消选择），因此标记为有意图。
            state.multiSelectionIntentional = true;
            state.selectionContext = "clips";
        },
        /**
         * 动作驱动的批量选中（粘贴 / 复制拖拽 / 分割 / 导入等）。
         *
         * 与 `setMultiSelectedClipIds` 的唯一差别是把
         * `multiSelectionIntentional` 置为 `false`：这些动作只是"把新产生的东西
         * 设为当前选中"，用户并未表达"我要整组拖动"，因此内核拖拽起手会收敛选区。
         */
        setMultiSelectedClipIdsFromAction(state, action: PayloadAction<string[]>) {
            state.multiSelectedClipIds = action.payload;
            state.multiSelectionIntentional = false;
            state.selectionContext = "clips";
        },
        /** 参数线选区存在性同步（PianoRollPanel 在 selectionUi 变化时派发），
         *  同时把剪贴板上下文标记为参数侧。拖拽期间的重复派发值不变，
         *  immer 判定无修改直接跳过，不触发重渲染。 */
        setParamSelectionActive(state, action: PayloadAction<boolean>) {
            state.paramSelectionActive = action.payload;
            state.selectionContext = "param";
        },
        /** 复制/剪切失败（系统剪贴板被占用等）时在状态栏给出可见反馈。 */
        setClipboardOperationFailed(state, action: PayloadAction<{ op: "copy" | "cut" }>) {
            state.status =
                action.payload.op === "cut" ? "Clipboard cut failed" : "Clipboard copy failed";
        },
        moveClipStart(state, action: PayloadAction<{ clipId: string; startSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.startSec = Math.max(0, action.payload.startSec);
                // 拖动超出边界时自动扩展工程时长
                const clipEnd = clip.startSec + clip.lengthSec;
                if (clipEnd > state.projectSec) {
                    state.projectSec = Math.ceil(clipEnd);
                }
            }
        },
        /**
         * 批量起点平移（波纹实时预览用）。
         *
         * 与 `moveClipStart` 语义逐条一致（钳零 + 越界自动扩展工程时长），但
         * 波纹 "all" 模式每帧要对 K 个跟随 clip 派发；逐条 `moveClipStart` 各自
         * 跑一次 immer producer 并做 O(N) 的 `clips.find`，合计 O(K·N)/帧，大工程
         * 拖拽明显卡顿。这里先建一次 id 索引，复杂度压回 O(N+K)（与
         * `applyOptimisticBulkClipState` 同法）。
         */
        moveClipsStartBulk(
            state,
            action: PayloadAction<Array<{ clipId: string; startSec: number }>>,
        ) {
            const clipsById = new Map(state.clips.map((clip) => [clip.id, clip]));
            for (const { clipId, startSec } of action.payload) {
                const clip = clipsById.get(clipId);
                if (!clip) continue;
                clip.startSec = Math.max(0, startSec);
                // 拖动超出边界时自动扩展工程时长
                const clipEnd = clip.startSec + clip.lengthSec;
                if (clipEnd > state.projectSec) {
                    state.projectSec = Math.ceil(clipEnd);
                }
            }
        },
        moveClipTrack(state, action: PayloadAction<{ clipId: string; trackId: string }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.trackId = action.payload.trackId;
            }
        },
        setClipLength(state, action: PayloadAction<{ clipId: string; lengthSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (clip) {
                clip.lengthSec = Math.max(0.0, action.payload.lengthSec);
                // 与后端及 applyOptimisticClipState 一致：偏移必须落在
                // [0, length] 内，否则留下越界的"幻影吸附目标"。
                if (clip.snapOffsetSec > clip.lengthSec) {
                    clip.snapOffsetSec = clip.lengthSec;
                }
            }
        },
        /** SnapOffset（吸附偏移）乐观更新：拖拽三角手柄实时预览。 */
        setClipSnapOffset(state, action: PayloadAction<{ clipId: string; snapOffsetSec: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            // 与后端一致：偏移必须落在 [0, length] 内。
            const offset = Math.max(0, Number(action.payload.snapOffsetSec) || 0);
            clip.snapOffsetSec = Math.min(offset, clip.lengthSec);
        },
        setClipPlaybackRate(
            state,
            action: PayloadAction<{ clipId: string; clipPlaybackRate: number }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            const takes = clip.takes ?? [];
            const activeTake = takes.find((entry) => entry.id === clip.activeTakeId) ?? takes[0];
            const takeRate = activeTake
                ? activeTake.playbackRate
                : clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
            clip.clipPlaybackRate = clamp(action.payload.clipPlaybackRate, 0.1, 10);
            clip.playbackRate = clamp(clip.clipPlaybackRate * takeRate, 0.1, 10);
            // "同步编辑所有 Take"开启时，后端会把全部 take 的速率统一为
            // 有效速率 ÷ 倍率；乐观阶段同步镜像，否则拖拽期间 inactive lane
            // 按旧速率渲染、persist+fulfilled 才收敛（视觉瞬态失真）。
            if (state.syncEditsAcrossTakes && takes.length > 1) {
                const mirrored = clamp(clip.playbackRate / getClipRateMultiplier(clip), 0.1, 10);
                for (const take of takes) {
                    take.playbackRate = mirrored;
                }
            }
            state.playbackRateVersion = (Number(state.playbackRateVersion) || 0) + 1;
        },
        setClipSourceRange(
            state,
            action: PayloadAction<{
                clipId: string;
                sourceStartSec?: number;
                sourceEndSec?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.sourceStartSec !== undefined) {
                // 与 sourceEndSec 同一口径：**不得钳制到 ≥0**（Slip 向左滑出媒体
                // 起点 ⇒ 负起点 = 前导静音，是既定合法状态；见下方注释）。这里用
                // `Number.isFinite` 而不是 `Number(x) || 0`：后者虽然也保留负数
                // （-3 是真值），但会把 NaN 静默变成 0，与对侧行为不一致。
                const value = Number(action.payload.sourceStartSec);
                clip.sourceStartSec = Number.isFinite(value) ? value : clip.sourceStartSec;
            }
            if (action.payload.sourceEndSec !== undefined) {
                // 注意：**不得钳制到 ≥0**。倒放 Clip 的消费窗口锚定 se，
                // 合法状态包含 se<0（整窗在媒体下方的纯静音段）与 se>D
                // （前导静音）；此处一旦拍扁，渲染/音频/后续编辑全部基于
                // 被摧毁的窗口工作。
                const value = Number(action.payload.sourceEndSec);
                clip.sourceEndSec = Number.isFinite(value) ? value : clip.sourceEndSec;
            }

            // Slip 的实时预览写在 active-take 投影上；展开全部 Take 时，
            // 必须同步对应 lane，否则拖拽期间波形不会移动。
            updateActiveTakeFromFlat(clip);
            if (state.syncEditsAcrossTakes) {
                for (const take of clip.takes ?? []) {
                    if (action.payload.sourceStartSec !== undefined) {
                        take.sourceStartSec = clip.sourceStartSec;
                    }
                    if (action.payload.sourceEndSec !== undefined) {
                        take.sourceEndSec = clip.sourceEndSec;
                    }
                }
            }
        },
        setClipFades(
            state,
            action: PayloadAction<{
                clipId: string;
                fadeInSec?: number;
                fadeOutSec?: number;
                fadeInShape?: number;
                fadeOutShape?: number;
                fadeInDir?: number;
                fadeOutDir?: number;
                fadeInS?: number;
                fadeOutS?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.fadeInSec !== undefined) {
                clip.fadeInSec = Math.max(0, action.payload.fadeInSec);
            }
            if (action.payload.fadeOutSec !== undefined) {
                clip.fadeOutSec = Math.max(0, action.payload.fadeOutSec);
            }
            // 形状原样透传（REAPER 允许小数变体）；曲率夹紧到 [-1, 1]。
            if (action.payload.fadeInShape !== undefined) {
                clip.fadeInShape = action.payload.fadeInShape;
            }
            if (action.payload.fadeOutShape !== undefined) {
                clip.fadeOutShape = action.payload.fadeOutShape;
            }
            if (action.payload.fadeInDir !== undefined) {
                clip.fadeInDir = Math.min(1, Math.max(-1, action.payload.fadeInDir));
            }
            if (action.payload.fadeOutDir !== undefined) {
                clip.fadeOutDir = Math.min(1, Math.max(-1, action.payload.fadeOutDir));
            }
            // S 轴：REAPER ≥7.81 的第二根连续轴，同样夹紧到 [-1, 1]。
            if (action.payload.fadeInS !== undefined) {
                clip.fadeInS = Math.min(1, Math.max(-1, Number(action.payload.fadeInS) || 0));
            }
            if (action.payload.fadeOutS !== undefined) {
                clip.fadeOutS = Math.min(1, Math.max(-1, Number(action.payload.fadeOutS) || 0));
            }
        },
        /** 自动交叉淡化长度（与手动 fade 分离，见 autoCrossfade.ts 的模型说明）。 */
        setClipAutoFades(
            state,
            action: PayloadAction<{
                clipId: string;
                autoFadeInSec?: number;
                autoFadeOutSec?: number;
            }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            if (action.payload.autoFadeInSec !== undefined) {
                clip.autoFadeInSec = Math.max(0, action.payload.autoFadeInSec);
            }
            if (action.payload.autoFadeOutSec !== undefined) {
                clip.autoFadeOutSec = Math.max(0, action.payload.autoFadeOutSec);
            }
        },
        /**
         * 乐观预览"旋钮的增益"。
         *
         * 【为什么写 clip_gain 而不是 gain】旋钮编辑的是 **item** 层（REAPER 的 D_VOL），
         * take 层归 take 属性面板。写错层会让 header 读数与实际听感分叉。
         */
        setClipGain(state, action: PayloadAction<{ clipId: string; gain: number }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.clip_gain = clampGain(Number(action.payload.gain));
        },
        setClipMuted(state, action: PayloadAction<{ clipId: string; muted: boolean }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.muted = Boolean(action.payload.muted);
        },
        setClipsGroupId(
            state,
            action: PayloadAction<{ clipIds: string[]; groupId: string | null }>,
        ) {
            const ids = new Set(action.payload.clipIds);
            for (const clip of state.clips) {
                if (ids.has(clip.id)) {
                    clip.groupId = action.payload.groupId ?? undefined;
                }
            }
        },
        /** 乐观更新 clip 颜色（立即反映到 UI，后端确认前先行生效�?*/
        optimisticUpdateClipColor(
            state,
            action: PayloadAction<{ clipId: string; color: ClipColor }>,
        ) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        /** 回滚 clip 颜色（后端失败时恢复到旧值） */
        rollbackClipColor(state, action: PayloadAction<{ clipId: string; color: ClipColor }>) {
            const clip = state.clips.find((entry) => entry.id === action.payload.clipId);
            if (!clip) return;
            clip.color = normalizeClipColor(action.payload.color);
        },
        setTrackVolume(state, action: PayloadAction<{ trackId: string; volume: number }>) {
            const track = state.tracks.find((entry) => entry.id === action.payload.trackId);
            if (track) {
                track.volume = clamp(action.payload.volume, 0, MAX_TRACK_VOLUME);
                // 轨道音量写进工程文件：改了不标脏，切工程时会被静默丢弃。
                markProjectDirty(state.project);
            }
        },
        /** 更新某个 clip 的音高曲线（来自后端 clip_pitch_data 事件）*/
        setClipPitchData(
            state,
            action: PayloadAction<{
                clipId: string;
                curveStartSec: number;
                midiCurve: number[];
                framePeriodMs: number;
            }>,
        ) {
            const { clipId, curveStartSec, midiCurve, framePeriodMs } = action.payload;
            state.clipPitchCurves[clipId] = {
                curveStartSec,
                midiCurve,
                framePeriodMs,
            };
            // 同步触发轨道总体音高线刷新
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
        },
        setClipFormantStatus(
            state,
            action: PayloadAction<{
                clipId: string;
                status: "ready" | "rebuilding" | "failed";
            }>,
        ) {
            state.clipFormantStatus[action.payload.clipId] = action.payload.status;
        },
        /** 写入某个 clip 的源共振峰分析结果（loading/ready/failed 三态）。 */
        setClipFormantAnalysis(
            state,
            action: PayloadAction<{ clipId: string; analysis: ClipFormantAnalysisState }>,
        ) {
            state.clipFormantAnalysis[action.payload.clipId] = action.payload.analysis;
        },
        openClipFormantToolWindow(
            state,
            action: PayloadAction<{
                clipId: string;
                anchor: { x: number; y: number };
            }>,
        ) {
            const { clipId, anchor } = action.payload;
            state.clipFormantToolWindow.open = true;
            state.clipFormantToolWindow.clipId = clipId;
            if (!state.clipFormantToolWindow.hasMoved) {
                state.clipFormantToolWindow.x = Math.max(12, Math.round(anchor.x));
                state.clipFormantToolWindow.y = Math.max(12, Math.round(anchor.y));
            }
        },
        setClipFormantToolWindowPosition(
            state,
            action: PayloadAction<{
                x: number;
                y: number;
            }>,
        ) {
            state.clipFormantToolWindow.x = Math.max(0, Math.round(action.payload.x));
            state.clipFormantToolWindow.y = Math.max(0, Math.round(action.payload.y));
            state.clipFormantToolWindow.hasMoved = true;
        },
        closeClipFormantToolWindow(state) {
            state.clipFormantToolWindow.open = false;
            state.clipFormantToolWindow.clipId = null;
        },
    },
    extraReducers: (builder) => {
        registerRuntime(builder);
        registerAudio(builder);
        registerTransport(builder);
        registerHistory(builder);
        registerProject(builder);
        registerTimelineEdits(builder);
        registerClipState(builder);
        registerTrack(builder);
        registerMatchers(builder);
    },
});

export const {
    beginInteraction,
    endInteraction,
    setTrackName,
    setTrackMeters,
    clearTrackMeters,
    setHistoryState,
    checkpointHistory,
    applyTimelinePayload,
    setToolMode,
    setEditParam,
    setBpm,
    setBeats,
    setGrid,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setRulerLabelSpacingPx,
    setShowPlayheadTimeInTrackHeader,
    setParamEditorSyncTimeline,
    toggleAutoCrossfade,
    toggleShowAllTakes,
    toggleSyncEditsAcrossTakes,
    toggleSplitTransition,
    setSplitTransitionMode,
    setSplitTransitionDurationUnit,
    setSplitTransitionDurationSec,
    setSplitTransitionDurationPercent,
    setSplitTransitionCurve,
    setSplitTransitionOverlapCrossfade,
    toggleSnap,
    setTimelineSnapSettings,
    setTempoMap,
    setTempoMapVisible,
    toggleTempoMapVisible,
    togglePitchSnap,
    setPitchSnapUnit,
    setPitchSnapScale,
    togglePlayheadZoom,
    toggleAutoScroll,
    toggleIgnoreGrouping,
    cycleRippleMode,
    setRippleMode,
    toggleParamEditorSeekPlayhead,
    toggleParamEditorTimelineClickSelectTrack,
    toggleClipboardPreview,
    toggleParamValuePopup,
    toggleParamAxisUnit,
    cycleDragDirection,
    setDragDirection,
    setEdgeSmoothnessPercent,
    setPlayheadSec,
    setPlaybackRenderingState,
    setPendingPlayheadReveal,
    setAudioPath,
    setProjectNotesMarkdown,
    closeVocalShifterSkippedFilesDialog,
    closeReaperSkippedFilesDialog,
    closeSaveVersionConflictDialog,
    setPitchSnapToleranceCents,
    setScaleHighlightMode,
    upsertCustomScalePreset,
    removeCustomScalePreset,
    upsertVibratoPreset,
    removeVibratoPreset,
    reorderVibratoPreset,
    reorderBuiltinVibratoPreset,
    setActiveVibratoPreset,
    cycleActiveVibratoPreset,
    setVibratoBaseline,
    toggleVibratoPresetEnabled,
    toggleLockParamLines,
    setMetronomeConfig,
    setSilencePreview,
    setSilenceDetectOptions,
    setSearchSettings,
    setFileBrowserView,
    setFolderImportOptions,
    setFileBrowserFavorites,
    setSearchSettingsDialogOpen,
    toggleQuickSearchAutoNormalize,
    setSaveUndoHistoryByDefault,
    setDefaultStretchAlgorithm,
    setDefaultHifiganMelStretch,
    setOrtEp,
    setVslibAvailable,
    setOrtDeviceKey,
    toggleAutoBackgroundRender,
    setRenderCacheSettings,
    setChannelImportPolicy,
    setPenInputSettings,
    setVisibleReferenceRootTrackIds,
    toggleVisibleReferenceRootTrackId,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    setMultiSelectedClipIds,
    setMultiSelectedClipIdsFromAction,
    setParamSelectionActive,
    setClipboardOperationFailed,
    moveClipStart,
    moveClipsStartBulk,
    moveClipTrack,
    setClipLength,
    setClipSnapOffset,
    setClipPlaybackRate,
    setClipSourceRange,
    setClipFades,
    setClipAutoFades,
    setClipGain,
    setClipMuted,
    setClipsGroupId,
    optimisticUpdateClipColor,
    rollbackClipColor,
    setTrackVolume,
    setClipPitchData,
    setClipFormantStatus,
    setClipFormantAnalysis,
    openClipFormantToolWindow,
    setClipFormantToolWindowPosition,
    closeClipFormantToolWindow,
    bumpParamsEpoch,
} = sessionSlice.actions;

/**
 * 切换参数编辑器工具并落盘。
 *
 * 【为什么要一个专门的入口】`setToolMode` 在 App 与参数编辑器面板里共有 8 个
 * dispatch 点。逐个补一句 `persistUiSettings()`，等于埋 8 个"以后新增入口时忘了
 * 补"的坑；工具选择是低频动作（一次点击或一次按键），立即写一次盘的代价可以忽略。
 *
 * 【已经在这个工具上时直接返回】这让"声明式地断言工具"变成免费操作：拖拽中的
 * 预设轮转每次都要确保自己落在正确的工具上（见 `applyVibratoChoice`），若每次都
 * 写一遍盘，一次轮转就会产生两次设置写入。跳过重复项之后，只有真的换了工具才落盘。
 *
 * 【为什么里面要转型】`persistUiSettings` 自己是个 thunk，而 `createAsyncThunk`
 * 默认的 `dispatch` 只认 plain action —— 与 `importThunks` 的
 * `dispatch as unknown as TrackDispatch` 是同一手法。
 */
export const setToolModePersistent = createAsyncThunk<ToolMode, ToolMode>(
    "session/setToolModePersistent",
    (mode, { dispatch, getState }) => {
        // 读的是**派发前**的状态：派发之后它必然等于 `mode`，那样判断永远为真。
        if ((getState() as { session: SessionState }).session.toolMode === mode) return mode;
        dispatch(setToolMode(mode));
        (dispatch as unknown as (action: unknown) => void)(persistUiSettings());
        return mode;
    },
);

export default sessionSlice.reducer;
