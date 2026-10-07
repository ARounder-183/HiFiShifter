import {
    Suspense,
    lazy,
    useCallback,
    useEffect,
    useMemo,
    useRef,
    useState,
    type ReactNode,
} from "react";
import { Flex, Button } from "@radix-ui/themes";
import { MenuBar } from "./components/layout/MenuBar";
import { ActionBar } from "./components/layout/ActionBar";
import { AraHostPanel } from "./features/ara/AraHostPanel";
import { PluginApplyStatus } from "./features/ara/PluginApplyStatus";
import { isPluginMode, pluginAllowsAction, pluginAllowsEditChannel } from "./services/hostCapabilities";
import { loadStandaloneWindowApi } from "./services/hostWindow";
import { TimelinePanel } from "./components/layout/TimelinePanel";
import { PianoRollPanel } from "./components/layout/PianoRollPanel";
import { useAppDispatch, useAppSelector } from "./app/hooks";
import { webApi } from "./services/webviewApi";
import { settingsApi } from "./services/api/settings";
import { fileBrowserApi } from "./services/api/fileBrowser";
import { IS_LINUX } from "./utils/platform";
import { allowsNativeTextSelection, isEditableTarget } from "./utils/nativeSelectionGuards";
import { clipboardErrorKey } from "./utils/clipboardError";
import { resolveStatusText } from "./utils/statusText";
import {
    closeVocalShifterSkippedFilesDialog,
    closeReaperSkippedFilesDialog,
    closeSaveVersionConflictDialog,
    fetchTimeline,
    refreshRuntime,
    loadUiSettings,
    syncPlaybackState,
    stopAudioPlayback,
    playOriginal,
    updateMetronome,
    undoRemote,
    redoRemote,
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromPath,
    openVocalShifterFromDialog,
    openReaperFromPath,
    openReaperFromDialog,
    importAudioFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    saveProjectToPathRemote,
    setTrackMeters,
    setToolModePersistent,
    setVslibAvailable,
    setPlaybackRenderingState,
    checkpointHistory,
    addTrackRemote,
    duplicateTrackRemote,
    removeTrackRemote,
    replaceClipSourceRemote,
    setTrackStateRemote,
    cycleDragDirection,
    persistUiSettings,
} from "./features/session/sessionSlice";
import { resolveTransportShortcutCommand } from "./features/session/transportShortcuts";
import {
    installRightDragContextMenuGuard,
    disposeRightDragContextMenuGuard,
} from "./utils/rightDragContextMenuGuard";
import { emitPinch, pinchDeltaFromWheel } from "./utils/pinchGesture";
import { useI18n } from "./i18n/I18nProvider";
import { useClipPitchDataListener } from "./hooks/useClipPitchDataListener";
import { useHistoryStateListener } from "./hooks/useHistoryStateListener";
import { PitchAnalysisProvider, usePitchAnalysis } from "./contexts/PitchAnalysisContext";
import { ParamDataLoadingChip } from "./components/layout/ParamDataLoadingChip";
import { AppStatusProgressChips } from "./components/layout/AppStatusProgressChips";
import { appStatusProgressBus } from "./utils/appStatusProgressBus";
import { FileBrowserPanel } from "./components/layout/FileBrowserPanel";
import { AppearanceSettingsPanel } from "./components/layout/AppearanceSettingsPanel";
import { UndoHistoryPanel } from "./components/layout/UndoHistoryPanel";
import { DockRoot } from "./components/dock/DockRoot";
import { registerBuiltinPanels } from "./components/dock/registerBuiltinPanels";
import { attachBuiltinPanelComponents } from "./components/dock/attachBuiltinPanelComponents";
import {
    PANEL_ARA_HOST,
    PANEL_FILE_BROWSER,
    PANEL_NOTEBOOK,
    PANEL_PARAM_EDITOR,
    PANEL_TIMELINE,
    PANEL_UNDO_HISTORY,
    PANEL_APPEARANCE,
} from "./components/dock/registerBuiltinPanels";
import { setPanelRenderer } from "./features/dock/panelRenderer";
import {
    addEmptyPanel,
    cycleFocus,
    dissolvePanelCommand,
    maximizeActive,
    toggleFloatActive,
} from "./features/dock/dockApi";
import { hydrateDock } from "./features/dock/dockSlice";
import {
    finalizeDockHydration,
    loadDockSettings,
    persistDockSettings,
} from "./features/dock/dockThunks";

// 面板注册必须在首次渲染前完成：布局归一化要按注册表判定"这个面板还在不在"。
registerBuiltinPanels();
attachBuiltinPanelComponents();

// 记事本按需加载：TipTap/ProseMirror/Turndown 加起来几百 KB，只有真正打开
// 记事本时才需要 —— 静态导入会把这些全塞进首屏主包。
const NotebookPanel = lazy(() =>
    import("./components/layout/notebook/NotebookPanel").then((module) => ({
        default: module.NotebookPanel,
    })),
);
// 记事本自带的错误边界与按需加载放在一起：动态 chunk 加载失败或模块求值抛错
// 都由它兜住，不会把整个窗口打空（应用没有根级 ErrorBoundary）。
const NotebookErrorBoundary = lazy(() =>
    import("./components/layout/notebook/NotebookErrorBoundary").then((module) => ({
        default: module.NotebookErrorBoundary,
    })),
);
import { ImportProjectDialog } from "./components/layout/ImportProjectDialog";
import { AppDialog } from "./ui/Dialog";
import { AppStatusChip } from "./ui";
import { QuickSearchPopup } from "./components/layout/QuickSearchPopup";
import { FolderImportHost } from "./components/layout/FolderImportHost";
import { useKeybindings } from "./features/keybindings/useKeybindings";
import { selectMergedKeybindings } from "./features/keybindings/keybindingsSlice";
import { beginHoldRepeat } from "./features/keybindings/holdRepeat";
import {
    ACTION_TO_EDIT_OP,
    resolveCopyCutRoute,
    resolveEditOpRoute,
    resolvePasteRoute,
    type EditOpChannel,
} from "./features/keybindings/focusRouting";
import { installFocusSurfaceTracking, getActiveSurface } from "./features/uiFocus/focusSurface";
import type { ActionId } from "./features/keybindings/types";
import { store } from "./app/store";
import { resolveRootTrackId, computeInsertBelowPlacement } from "./features/session/trackUtils";
import { getParamShiftStep } from "./components/layout/pianoRoll/paramShiftStep";
import { resolveParamShiftIntent } from "./features/keybindings/paramShiftActions";
import { isSelectionParamEditInFlight } from "./features/session/selectionEditInFlight";
import { runConfirmedExitClose } from "./confirmedExitClose";
import { paramsApi } from "./services/api";
import { coreApi } from "./services/api/core";
import type { SourceFileChange, SourceFileMatchCandidate } from "./services/api/timeline";
import { waveformMipmapStore } from "./utils/waveformMipmapStore";
import { projectApi, type AutoBackupSettings } from "./services/api/project";
import type { ParamFramesPayload, ProcessorParamDescriptor } from "./types/api";
import {
    IMPORT_MIDI_PATH_EVENT,
    IMPORT_PROJECT_PICK_EVENT,
    OPEN_PROJECT_PATH_EVENT,
    type ExternalFileActionDetail,
    type ExternalFileActionKind,
    type ImportMidiRequestDetail,
} from "./features/session/projectOpenEvents";
import { detectExternalPathAction } from "./components/layout/timeline/dnd";
import type { MessageKey } from "./i18n/messages";
import type { CloseRequestedEvent } from "@tauri-apps/api/window";
import { useAutoBackupScheduler } from "./hooks/useAutoBackupScheduler";
import { useClipFormantStatusListener } from "./hooks/useClipFormantStatusListener";
import { useRecordingListener } from "./hooks/useRecordingListener";
import {
    cancelRecordingCountdown,
    loadRecordingSettings,
    startRecordingFlow,
    stopRecordingFlow,
} from "./features/recording/recordingSlice";

const statusKey: Record<string, string> = {
    Ready: "status_ready",
    Failed: "status_failed",
    "Runtime updated": "status_runtime_updated",
    "Runtime update failed": "status_runtime_update_failed",
    "Clear waveform cache failed": "status_clear_waveform_cache_failed",
    "Render cache cleared": "status_render_cache_cleared",
    "Clear render cache failed": "status_clear_render_cache_failed",
    "Fake-stereo scan rejected": "status_fake_stereo_scan_rejected",
    "Import canceled": "status_import_canceled",
    "Pick output canceled": "status_pick_output_canceled",
    "Output path selected": "status_output_path_selected",
    "New project": "status_new_project",
    "Open canceled": "status_open_canceled",
    "Opening project...": "status_opening_project",
    "Open failed": "status_open_failed",
    "Project version confirmation required": "status_project_version_confirmation",
    "Project opened": "status_project_opened",
    "Save canceled": "status_save_canceled",
    "Save failed": "status_save_failed",
    "Save As canceled": "status_save_as_canceled",
    "Save As failed": "status_save_as_failed",
    "Save version confirmation required": "status_save_version_confirmation",
    "Project saved": "status_project_saved",
    "Clips created": "status_clips_created",
    "Glue done": "status_glue_done",
    "Export done": "status_export_done",
    "Export failed": "status_export_failed",
    "Export separated done": "status_export_separated_done",
    "Export separated failed": "status_export_separated_failed",
    "Clipboard copy failed": "status_clipboard_copy_failed",
    "Clipboard cut failed": "status_clipboard_cut_failed",
    "VocalShifter imported with skipped files": "vs_import_skipped_header",
    // 播放/停止传输状态
    "Playing original": "status_playing_original",
    "Play original failed": "status_play_original_failed",
    "Stopping audio...": "status_stopping_audio",
    "Audio stopped": "status_audio_stopped",
    "Stop audio failed": "status_stop_audio_failed",
    // 导入类状态
    "Audio imported": "status_audio_imported",
    "Dropped audio imported": "status_dropped_audio_imported",
    "Clips duplicated": "status_clips_duplicated",
    "Clip edit rejected": "status_clip_edit_rejected",
    "Import done": "status_import_done",
    "Import failed": "status_import_failed",
    "Import audio failed": "status_import_audio_failed",
    "Project imported": "status_project_imported",
    "VocalShifter project imported": "status_vocalshifter_project_imported",
    "Reaper project imported": "status_reaper_project_imported",
    "Pasted VocalShifter clipboard data": "status_pasted_vocalshifter_clipboard",
    "Pasted Reaper clipboard data": "status_pasted_reaper_clipboard",
    "Timeline clipboard pasted": "status_timeline_clipboard_pasted",
    "Paste timeline clipboard failed": "status_paste_timeline_failed",
    "Tempo map updated": "status_tempo_map_updated",
    "Waiting for import options": "status_waiting_import_options",
    "Waveform cache cleared": "status_waveform_cache_cleared",
    // 各 thunk 的完成 / 失败结果状态（sessionSlice fulfilled 分支写出）
    "Default model loaded": "status_default_model_loaded",
    "Load default model failed": "status_load_default_model_failed",
    "Model loaded": "status_model_loaded",
    "Load model failed": "status_load_model_failed",
    "Audio processed": "status_audio_processed",
    "Process audio failed": "status_process_audio_failed",
    "MIDI clip created": "status_midi_clip_created",
    "MIDI import failed": "status_midi_import_failed",
    "Pitch shift applied": "status_pitch_shift_applied",
    "Pitch shift failed": "status_pitch_shift_failed",
    "Synthesis done": "status_synthesis_done",
    "Synthesis failed": "status_synthesis_failed",
    // 进行中状态（setPending）
    "Importing folder...": "status_importing_folder",
    "Applying pitch shift...": "status_applying_pitch_shift",
    "Clearing waveform cache...": "status_clearing_waveform_cache",
    "Clearing render cache...": "status_clearing_render_cache",
    "Exporting WAV...": "status_exporting_wav",
    "Exporting audio...": "status_exporting_audio",
    "Exporting separated tracks...": "status_exporting_separated",
    "Importing MIDI clip...": "status_importing_midi",
    "Importing Reaper project...": "status_importing_reaper",
    "Importing VocalShifter project...": "status_importing_vocalshifter",
    "Importing audio...": "status_importing_audio",
    "Importing dropped audio...": "status_importing_dropped",
    "Importing multiple audio files...": "status_importing_multiple",
    "Importing project...": "status_importing_project",
    "Loading default model...": "status_loading_default_model",
    "Loading model...": "status_loading_model",
    "Pasting Reaper clipboard data...": "status_pasting_reaper",
    "Pasting VocalShifter clipboard data...": "status_pasting_vocalshifter",
    "Pasting timeline clipboard...": "status_pasting_timeline",
    "Picking project to import...": "status_picking_project",
    "Playing original...": "status_playing_original_busy",
    "Processing audio...": "status_processing_audio",
    "Refreshing runtime...": "status_refreshing_runtime",
    "Selecting output path...": "status_selecting_output",
    "Synthesizing...": "status_synthesizing",
};

// 后端返回的错误码 → i18n key 映射
const errorCodeKey: Record<string, string> = {
    clipboard_not_found: "vs_paste_clipboard_not_found",
    clipboard_invalid_format: "vs_paste_clipboard_invalid_format",
    clipboard_io_error: "vs_paste_clipboard_io_error",
    no_pitch_line_selected: "vs_paste_no_pitch_line",
    import_read_failed: "vs_import_read_failed",
    import_parse_failed: "vs_import_parse_failed",
    /* 前端合成码：音频导入 fulfilled 但 ok=false（原版只写灰色 status，失败不可辨） */
    import_audio_failed: "status_import_audio_failed",
    // Take 操作被拒绝 / 失败（sessionSlice 以原文写入 error 通道）
    "Take switch rejected": "status_take_switch_rejected",
    "Take cycle rejected": "status_take_cycle_rejected",
    "Take reverse rejected": "status_take_reverse_rejected",
    "Add take from media failed": "status_add_take_failed",
    "Duplicate take failed": "status_duplicate_take_failed",
    "Explode takes failed": "status_explode_takes_failed",
    "Pack into takes failed": "status_pack_takes_failed",
    "Remove take failed": "status_remove_take_failed",
    "Rename take failed": "status_rename_take_failed",
    "Take channel mode rejected": "status_take_channel_mode_rejected",
};

// 这些状态表示工程内容刚被替换/导入，需立即执行一次缺失媒体检测，
// 不依赖窗口 focus（例如启动时通过命令行打开工程时窗口可能一直保持聚焦）。
const SOURCE_FILE_CHECK_TRIGGER_STATUSES = new Set([
    "Project opened",
    "Project imported",
    "VocalShifter project imported",
    "Reaper project imported",
    "Pasted VocalShifter clipboard data",
    "Pasted Reaper clipboard data",
]);

const DEFAULT_AUTO_BACKUP_SETTINGS: AutoBackupSettings = {
    saveOnSaveEnabled: true,
    timedBackupEnabled: false,
    timedBackupIntervalSec: 300,
    timedBackupPathTemplate:
        "<ProjectFolder>/HiFiShifter Backup/<ProjectName>_%Y-%m-%d-%H-%M-%S.hshp",
};

type SourceFileChangeAction =
    | "pending"
    | "processing"
    | "ignored"
    | "reloaded"
    | "replaced"
    | "failed";

type SourceFileChangedItem = SourceFileChange & {
    action: SourceFileChangeAction;
    /** 重新加载成功后实际使用的新文件路径。 */
    reloadedPath?: string;
    /** 已修改文件是否已经尝试过“重新加载”。之后按钮转为“替换”。 */
    reloadAttempted?: boolean;
    /** 搜索得到的候选文件（哈希完全匹配项排在前面）。 */
    candidates?: SourceFileMatchCandidate[];
    /** 用户在候选列表中当前选中的路径。 */
    selectedCandidatePath?: string;
};

function normalizeSourceFileChanges(rawChanges: SourceFileChange[]): SourceFileChange[] {
    const seen = new Set<string>();
    return rawChanges.filter((change) => {
        if (!change || typeof change.source_path !== "string" || !change.source_path.trim()) {
            return false;
        }
        if (change.change !== "deleted" && change.change !== "modified") return false;
        const key = `${change.source_path}::${change.change}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

function markMissingSourceFilesUnavailable(changes: SourceFileChange[]): void {
    for (const change of changes) {
        if (change.change === "deleted") {
            waveformMipmapStore.markUnavailable(change.source_path);
        }
    }
}

function resetSourceFileItemToPending(item: SourceFileChangedItem): SourceFileChangedItem {
    return {
        ...item,
        action: "pending",
        reloadedPath: undefined,
        reloadAttempted: false,
    };
}

interface WheelSelectProps {
    value: string;
    onValueChange: (value: string) => void;
    disabled?: boolean;
    className?: string;
    children: ReactNode;
}

/**
 * 支持滚轮调整选项的下拉框。
 * 监听原生 wheel 事件（非 passive），滚动时切换选项并阻止默认滚动传播，
 * 避免其所在的可滚动网格列表被一并滚动。
 */
function WheelSelect({ value, onValueChange, disabled, className, children }: WheelSelectProps) {
    const selectRef = useRef<HTMLSelectElement | null>(null);

    useEffect(() => {
        const select = selectRef.current;
        if (!select || disabled) return;

        const handleWheel = (event: WheelEvent) => {
            event.preventDefault();
            event.stopPropagation();
            const total = select.options.length;
            if (total === 0) return;
            const current = select.selectedIndex;
            const step = event.deltaY > 0 ? 1 : -1;
            const next = Math.max(0, Math.min(total - 1, current + step));
            if (next !== current) {
                const nextValue = select.options[next]?.value;
                if (nextValue !== undefined) {
                    onValueChange(nextValue);
                }
            }
        };

        select.addEventListener("wheel", handleWheel, { passive: false });
        return () => select.removeEventListener("wheel", handleWheel);
    }, [value, disabled, onValueChange]);

    return (
        <select
            ref={selectRef}
            value={value}
            disabled={disabled}
            className={className}
            onChange={(event) => onValueChange(event.target.value)}
        >
            {children}
        </select>
    );
}

/** 若条目保留着候选列表但没有当前选择，则默认选回第一个候选。 */
function defaultSelectedCandidatePath(item: SourceFileChangedItem): SourceFileChangedItem {
    if (item.selectedCandidatePath) return item;
    const first = item.candidates?.[0]?.path;
    return first ? { ...item, selectedCandidatePath: first } : item;
}

function mergeLatestSourceFileChanges(
    items: SourceFileChangedItem[],
    rawChanges: SourceFileChange[],
): SourceFileChangedItem[] {
    const rawByClipId = new Map(rawChanges.map((change) => [change.clip_id, change]));
    const rawByPath = new Map(rawChanges.map((change) => [change.source_path, change]));
    const consumedClipIds = new Set<string>();

    const merged: SourceFileChangedItem[] = items.map((item) => {
        // 优先按 clip_id 匹配：即使替换后路径从 A 变为 B/C，也仍然属于同一项目。
        const latest =
            rawByClipId.get(item.clip_id) ??
            rawByPath.get(item.source_path) ??
            (item.reloadedPath ? rawByPath.get(item.reloadedPath) : undefined);
        if (!latest) return item;
        consumedClipIds.add(latest.clip_id);

        const handled = item.action === "reloaded" || item.action === "replaced";
        if (latest.change === "deleted") {
            if (item.action === "ignored") {
                return { ...item, change: "deleted", reloadedPath: undefined };
            }
            if (handled) {
                // 替换/重新加载后的文件又失效了：回到原始文件缺失状态。
                return resetSourceFileItemToPending({
                    ...item,
                    change: "deleted",
                });
            }
            return {
                ...item,
                change: "deleted",
                action: "pending",
            };
        }

        // latest.change === "modified"
        if (item.action === "ignored") {
            return { ...item, change: "modified", reloadedPath: undefined };
        }
        if (handled) {
            // 已处理文件再次被修改：重新变为未处理，保留原路径以便重新加载。
            return resetSourceFileItemToPending({
                ...item,
                change: "modified",
                reloadedPath: latest.source_path,
            });
        }
        return {
            ...item,
            change: "modified",
        };
    });

    const knownPaths = new Set(
        merged.flatMap((item) => [item.source_path, item.reloadedPath].filter(Boolean) as string[]),
    );
    for (const change of rawChanges) {
        if (consumedClipIds.has(change.clip_id) || knownPaths.has(change.source_path)) {
            continue;
        }
        merged.push({ ...change, action: "pending" as const });
        knownPaths.add(change.source_path);
    }

    return merged;
}

/**
 * 判定路径对应的外部文件动作种类。
 *
 * 【为什么不再内联一份正则】这里原先重写了与 `timeline/dnd` 完全相同的四组正则，
 * 而且**漏掉 MIDI**——于是从启动参数 / 外部文件事件进来的 `.mid` 文件会被判为
 * `null` 而被静默丢弃，与拖放路径的行为不一致（同一文件两种命运）。
 * 现在直接复用 `detectExternalPathAction`：准入判据只剩一份，不可能再分叉。
 *
 * @param path 文件路径。
 * @returns 动作种类；非受支持类型时为 null。
 */
function detectExternalActionKindFromPath(path: string): ExternalFileActionKind | null {
    const kind = detectExternalPathAction(path);
    // `importMidi` 不属于本事件通道的动作集合（`ExternalFileActionKind` 没有它）：
    // MIDI 走"导入 MIDI clip"的独立流程，而不是"打开/导入工程"。这里显式排除，
    // 而不是用类型断言硬转——否则 MIDI 路径会被当成工程打开。
    // `importFolder` 同理：目录导入有自己的入口（拖放 / 右键菜单 → 选项对话框），
    // 而这条通道只带一个路径，表达不了落点与导入选项。
    if (kind === null || kind === "importMidi" || kind === "importFolder") return null;
    return kind;
}

function AppInner() {
    const dispatch = useAppDispatch();
    const { t, tf, tVars, plural } = useI18n();
    const pitchAnalysis = usePitchAnalysis();

    const status = useAppSelector((state) => state.session.status);
    const error = useAppSelector((state) => state.session.error);

    const runtimeIsPlaying = useAppSelector((state) => state.session.runtime.isPlaying);
    const runtimeHasSynthesized = useAppSelector((state) => state.session.runtime.hasSynthesized);
    const toolMode = useAppSelector((state) => state.session.toolMode);
    const drawToolMode = useAppSelector((state) => state.session.drawToolMode);
    const projectDirty = useAppSelector((state) => state.session.project.dirty);
    // playheadSec / selectedTrackId 均只在 handleImportMidiFromMenu 打开对话框
    // 的瞬间需要快照，经 store.getState() 读取即可 —— 订阅它们会让 AppInner
    // 随播放头移动 / seek 高频重渲。
    const paramsEpoch = useAppSelector((state) => state.session.paramsEpoch);
    const recordingActive = useAppSelector((state) => state.recording.active);
    const recordingSettings = useAppSelector((state) => state.recording.settings);
    const recordingStartSec = useAppSelector((state) => state.recording.startSec);
    const selectedClipId = useAppSelector((state) => state.session.selectedClipId);
    const multiSelectedClipIds = useAppSelector((state) => state.session.multiSelectedClipIds);
    const sessionClips = useAppSelector((state) => state.session.clips);
    // 注意：playbackPositionSec 以 ~30Hz 持续变化，这里不能订阅（否则每次
    // 播放 tick 都重渲 AppInner）；需要它的地方（录音自动停止）在回调内经
    // store.getState() 同步读取最新值。
    // 使用 ref 桥接最新的工程修改状态
    const projectDirtyRef = useRef(projectDirty);
    useEffect(() => {
        projectDirtyRef.current = projectDirty;
    }, [projectDirty]);
    const projectPath = useAppSelector((state) => state.session.project.path);
    const hasExistingTempoMap = useAppSelector((state) => Boolean(state.session.tempoMap));
    // 当工程路径变更时（新建/打开/关闭工程），重置已忽略的源文件路径集合
    useEffect(() => {
        ignoredSourcePathsRef.current = new Set();
    }, [projectPath]);

    const vocalShifterSkippedFilesDialog = useAppSelector(
        (state) => state.session.vocalShifterSkippedFilesDialog,
    );
    const reaperSkippedFilesDialog = useAppSelector(
        (state) => state.session.reaperSkippedFilesDialog,
    );
    const saveVersionConflictDialog = useAppSelector(
        (state) => state.session.saveVersionConflictDialog,
    );

    const [quickSearchOpen, setQuickSearchOpen] = useState(false);
    const dockLayout = useAppSelector((state) => state.dock.layout);
    const dockSettings = useAppSelector((state) => state.dock.settings);
    const dockHydrated = useAppSelector((state) => state.dock.hydrated);
    const dockMaximized = useAppSelector((state) => Boolean(state.dock.maximized));
    const [autoBackupSettings, setAutoBackupSettings] = useState<AutoBackupSettings>(
        DEFAULT_AUTO_BACKUP_SETTINGS,
    );
    const [unsavedDialog, setUnsavedDialog] = useState<{
        open: boolean;
        mode: "switch" | "exit";
    }>({ open: false, mode: "switch" });
    // 打开工程时发现文件版本高于当前程序：等待用户确认是否继续尝试加载。
    const [projectVersionDialog, setProjectVersionDialog] = useState<{
        open: boolean;
        path: string;
        fileVersion: number;
        currentVersion: number;
    }>({ open: false, path: "", fileVersion: 0, currentVersion: 0 });
    const [projectImportPick, setProjectImportPick] = useState<{
        open: boolean;
        path: string | null;
    }>({ open: false, path: null });
    // 检测/处理互斥：避免窗口 focus、工程打开、文件选择对话框等事件叠加触发重复检测。
    const sourceFileCheckBusyRef = useRef(false);
    const sourceFileChangeHandlingRef = useRef(false);
    const sourceFileDialogOpenRef = useRef(false);
    const sourceFileInitialChangesRef = useRef<SourceFileChangedItem[]>([]);
    // 重新捕获缺失媒体对话框（窗口重新获得焦点或工程内容变更后触发）
    const [sourceFileChangedDialog, setSourceFileChangedDialog] = useState<{
        open: boolean;
        changes: SourceFileChangedItem[];
    }>({ open: false, changes: [] });
    const [sourceFileSearchBusy, setSourceFileSearchBusy] = useState(false);
    const [sourceFileSearchMode, setSourceFileSearchMode] = useState<
        "file_name" | "extension_hash"
    >("file_name");
    useEffect(() => {
        sourceFileDialogOpenRef.current = sourceFileChangedDialog.open;
    }, [sourceFileChangedDialog.open]);
    const pendingUnsavedActionRef = useRef<null | (() => Promise<void>)>(null);
    const allowWindowCloseRef = useRef(false);
    const processorParamCacheRef = useRef(new Map<string, ProcessorParamDescriptor[]>());
    // 当前会话中已忽略的缺失媒体路径集合（用户点击"忽略"后不再重复弹窗）
    const ignoredSourcePathsRef = useRef<Set<string>>(new Set());

    // MIDI clip import dialog state (lifted from TimelinePanel)
    const [midiClipDialogOpen, setMidiClipDialogOpen] = useState(false);
    const [midiClipPath, setMidiClipPath] = useState<string | null>(null);
    const [midiClipStartSec, setMidiClipStartSec] = useState(0);
    const [midiClipTrackId, setMidiClipTrackId] = useState<string | null>(null);
    const [midiClipClipboardGuid, setMidiClipClipboardGuid] = useState<string | null>(null);
    const [fillGaps, setFillGaps] = useState(false);
    const [multiTrackMerge, setMultiTrackMerge] = useState(true);
    const [importBpmAsProject, setImportBpmAsProject] = useState(false);
    const [noteBpmMode, setNoteBpmMode] = useState<string>("midi");
    const [specifiedBpm, setSpecifiedBpm] = useState<number>(120);
    const [importPosition, setImportPosition] = useState<string>("selection");
    const [closeLeadingGap, setCloseLeadingGap] = useState(true);
    const [importTempoMapEnabled, setImportTempoMapEnabled] = useState(false);
    const [importTempoMapTempo, setImportTempoMapTempo] = useState(true);
    const [importTempoMapTimeSignature, setImportTempoMapTimeSignature] = useState(true);
    const [importTempoMapKeySignature, setImportTempoMapKeySignature] = useState(false);
    const [midiImportTargetMenu, setMidiImportTargetMenu] = useState<string>("pitchRef");
    const [midiImportTargetDragDrop, setMidiImportTargetDragDrop] = useState<string>("pitchRef");
    const [midiDialogSource, setMidiDialogSource] = useState<"menu" | "dragDrop">("menu");
    const [autoReloadModifiedMedia, setAutoReloadModifiedMedia] = useState(true);
    // 为新的音频块启用循环（Loop / 循环源，默认开启）
    const [loopNewClips, setLoopNewClips] = useState(true);

    // 加载 UI 持久化设置，并把 MIDI 相关字段回填进本地对话框状态。
    //
    // 【为什么唯一一次 get_ui_settings 由这里发起】后端的 get_ui_settings 不是
    // 纯读；启动路径此前有两轮往返（boot effect 里的 thunk + 本处的直读）。
    // 现在这次加载由本 effect 持有：unwrap 得到的载荷喂给下面的 setState，
    // 同时经 loadUiSettings.fulfilled 进入 sessionSlice（autoCrossfade / 吸附
    // 等全局项的权威来源仍是那个 reducer）。
    useEffect(() => {
        let cancelled = false;
        dispatch(loadUiSettings())
            .unwrap()
            .then((s) => {
                if (cancelled) return;
                if (s?.midiFillGaps != null) {
                    setFillGaps(s.midiFillGaps);
                }
                if (s?.midiMultiTrackMerge != null) {
                    setMultiTrackMerge(s.midiMultiTrackMerge);
                }
                if (s?.midiImportBpmAsProject != null) {
                    setImportBpmAsProject(s.midiImportBpmAsProject);
                }
                if (s?.midiNoteBpmMode != null) {
                    setNoteBpmMode(s.midiNoteBpmMode);
                }
                if (s?.midiSpecifiedBpm != null) {
                    setSpecifiedBpm(s.midiSpecifiedBpm);
                }
                if (s?.midiImportPosition != null) {
                    setImportPosition(s.midiImportPosition);
                }
                if (s?.midiCloseLeadingGap != null) {
                    setCloseLeadingGap(s.midiCloseLeadingGap);
                }
                if (s?.midiImportAsTempoMap != null) {
                    setImportTempoMapEnabled(Boolean(s.midiImportAsTempoMap));
                }
                if (s?.midiImportTempoMapTempo != null) {
                    setImportTempoMapTempo(Boolean(s.midiImportTempoMapTempo));
                }
                if (s?.midiImportTempoMapTimeSignature != null) {
                    setImportTempoMapTimeSignature(Boolean(s.midiImportTempoMapTimeSignature));
                }
                if (s?.midiImportTempoMapKeySignature != null) {
                    setImportTempoMapKeySignature(Boolean(s.midiImportTempoMapKeySignature));
                }
                if (s?.midiImportTargetMenu != null) {
                    setMidiImportTargetMenu(s.midiImportTargetMenu);
                }
                if (s?.midiImportTargetDragDrop != null) {
                    setMidiImportTargetDragDrop(s.midiImportTargetDragDrop);
                }
                if (typeof s?.autoReloadModifiedMedia === "boolean") {
                    setAutoReloadModifiedMedia(s.autoReloadModifiedMedia);
                }
                if (typeof s?.loopNewClips === "boolean") {
                    setLoopNewClips(s.loopNewClips);
                }
            })
            .catch(() => {
                // 读不到设置时保持出厂默认；sessionSlice 的 reducer 同样不会执行。
            });
        return () => {
            cancelled = true;
        };
    }, [dispatch]);

    /*
     * vslib 能力探测：启动时问一次后端，供算法列表过滤掉不可用的 vslib。
     *
     * 【为什么是"一次"而不是轮询】可用性是编译期 + 链接期决定的静态事实，
     * 运行期不会变（DLL 缺失会让进程根本起不来）。探测失败时保持 `null`
     * （未知），算法列表按"不可用"处理 —— 详见 pitchAlgoOptions.ts。
     */
    useEffect(() => {
        let cancelled = false;
        void webApi
            .getVslibStatus()
            .then((status) => {
                if (cancelled) return;
                dispatch(setVslibAvailable(Boolean(status?.available)));
            })
            .catch(() => {
                // 取不到状态：保持 null（未知 → 隐藏 vslib）。
            });
        return () => {
            cancelled = true;
        };
    }, [dispatch]);

    const handleImportMidiFromMenu = useCallback(() => {
        const session = store.getState().session;
        setMidiDialogSource("menu");
        setMidiClipPath(null);
        setMidiClipClipboardGuid(null);
        setMidiClipStartSec(session.playheadSec ?? 0);
        setMidiClipTrackId(session.selectedTrackId ?? null);
        setMidiClipDialogOpen(true);
    }, []);

    const handleFillGapsChange = useCallback((v: boolean) => {
        setFillGaps(v);
        settingsApi.saveUiSettings({ midiFillGaps: v });
    }, []);

    const handleAutoReloadModifiedMediaChange = useCallback((v: boolean) => {
        setAutoReloadModifiedMedia(v);
        settingsApi.saveUiSettings({ autoReloadModifiedMedia: v });
    }, []);

    const handleLoopNewClipsChange = useCallback((v: boolean) => {
        setLoopNewClips(v);
        settingsApi.saveUiSettings({ loopNewClips: v });
    }, []);

    const handleMultiTrackMergeChange = useCallback((v: boolean) => {
        setMultiTrackMerge(v);
        settingsApi.saveUiSettings({ midiMultiTrackMerge: v });
    }, []);

    const handleImportBpmAsProjectChange = useCallback((v: boolean) => {
        setImportBpmAsProject(v);
        settingsApi.saveUiSettings({ midiImportBpmAsProject: v });
    }, []);

    const handleNoteBpmModeChange = useCallback((v: string) => {
        setNoteBpmMode(v);
        settingsApi.saveUiSettings({ midiNoteBpmMode: v });
    }, []);

    const handleSpecifiedBpmChange = useCallback((v: number) => {
        setSpecifiedBpm(v);
        settingsApi.saveUiSettings({ midiSpecifiedBpm: v });
    }, []);

    const handleImportPositionChange = useCallback((position: string) => {
        setImportPosition(position);
        settingsApi.saveUiSettings({ midiImportPosition: position });
    }, []);

    const handleCloseLeadingGapChange = useCallback((v: boolean) => {
        setCloseLeadingGap(v);
        settingsApi.saveUiSettings({ midiCloseLeadingGap: v });
    }, []);

    const handleImportTempoMapEnabledChange = useCallback((v: boolean) => {
        setImportTempoMapEnabled(v);
        settingsApi.saveUiSettings({ midiImportAsTempoMap: v });
    }, []);
    const handleImportTempoMapTempoChange = useCallback((v: boolean) => {
        setImportTempoMapTempo(v);
        settingsApi.saveUiSettings({ midiImportTempoMapTempo: v });
    }, []);
    const handleImportTempoMapTimeSignatureChange = useCallback((v: boolean) => {
        setImportTempoMapTimeSignature(v);
        settingsApi.saveUiSettings({ midiImportTempoMapTimeSignature: v });
    }, []);
    const handleImportTempoMapKeySignatureChange = useCallback((v: boolean) => {
        setImportTempoMapKeySignature(v);
        settingsApi.saveUiSettings({ midiImportTempoMapKeySignature: v });
    }, []);

    const handleImportTargetMenuChange = useCallback((v: string) => {
        setMidiImportTargetMenu(v);
        settingsApi.saveUiSettings({ midiImportTargetMenu: v });
    }, []);

    const handleImportTargetDragDropChange = useCallback((v: string) => {
        setMidiImportTargetDragDrop(v);
        settingsApi.saveUiSettings({ midiImportTargetDragDrop: v });
    }, []);

    const statusText = useMemo(
        () =>
            resolveStatusText(status, statusKey, (key, count) =>
                count === undefined
                    ? (t(key as MessageKey) as string)
                    : plural(key as MessageKey, count),
            ),
        [status, t, plural],
    );

    // 监听后端 clip_pitch_data 事件，将 per-clip MIDI 曲线存入 store
    useClipPitchDataListener();
    useClipFormantStatusListener();
    useRecordingListener();
    // 撤销/重做可用性（栈深度）镜像：菜单置灰与快捷键前置判断都读它
    useHistoryStateListener();

    // 阻止浏览器默认的 Ctrl+F 搜索、右键菜单和 Alt 键

    // 改用 useRef，取消重绘
    const isModifierRef = useRef(false);

    /**
     * 触控板捏合是否接管缩放（设置项）。
     *
     * 【为什么用 ref】下面的全局监听只挂载一次（`useEffect(..., [])`），直接闭包
     * 会永远读到初始值。经 ref 转发后设置一改即生效，不必重挂监听器。
     */
    const pinchZoomEnabled = useAppSelector((state) => state.session.penInput.trackpadPinchZoom);
    const pinchZoomEnabledRef = useRef(pinchZoomEnabled);
    useEffect(() => {
        pinchZoomEnabledRef.current = pinchZoomEnabled;
    });

    useEffect(() => {
        // WebKitGTK fires `contextmenu` on right-button press instead of
        // release. Track the right-button state on Linux and re-dispatch the
        // deferred event on pointerup so right-click menus (and right-drag
        // decisions made by local handlers) follow Windows-like timing.
        let linuxRightButtonDown = false;
        let linuxDeferredContextMenu: {
            clientX: number;
            clientY: number;
            target: EventTarget | null;
        } | null = null;
        const trackLinuxRightButton = (event: PointerEvent) => {
            if (IS_LINUX && event.button === 2) {
                linuxRightButtonDown = true;
            }
        };
        const flushLinuxDeferredContextMenu = () => {
            const pending = linuxDeferredContextMenu;
            linuxDeferredContextMenu = null;
            linuxRightButtonDown = false;
            if (!pending) return;
            window.setTimeout(() => {
                const clientX = pending.clientX;
                const clientY = pending.clientY;
                let target = pending.target;
                if (!(target instanceof Element) || !document.contains(target)) {
                    target = document.elementFromPoint(clientX, clientY);
                }
                target?.dispatchEvent(
                    new MouseEvent("contextmenu", {
                        bubbles: true,
                        cancelable: true,
                        clientX,
                        clientY,
                        button: 2,
                        buttons: 0,
                        view: window,
                    }),
                );
            }, 0);
        };
        const cancelLinuxDeferredContextMenu = () => {
            linuxDeferredContextMenu = null;
            linuxRightButtonDown = false;
        };
        const handleLinuxPointerUp = (event: PointerEvent) => {
            if (!IS_LINUX || event.button !== 2) return;
            flushLinuxDeferredContextMenu();
        };

        function preventNativeTextSelection(e: Event) {
            if (allowsNativeTextSelection(e.target)) return;
            // 阻止 WebView 双击/拖选文本；不阻止传播，因此自定义双击逻辑仍会执行。
            e.preventDefault();
        }

        function clearNativeTextSelection(e: Event) {
            const selection = window.getSelection();
            if (!selection || selection.isCollapsed) return;
            if (allowsNativeTextSelection(e.target)) return;
            selection.removeAllRanges();
        }

        function preventNativeDragStart(e: Event) {
            const target = e.target as HTMLElement | null;
            if (!target) return;
            if (isEditableTarget(target)) return;
            if (target.closest?.('[data-hs-native-drag="true"]') || target.draggable === true) {
                return;
            }
            // 阻止 WebView 原生拖拽（选中文本/图片/链接），自定义 onDragStart 仍会收到事件。
            e.preventDefault();
        }

        function preventMiddleClickNative(e: MouseEvent) {
            if (e.button !== 1) return;
            if (isEditableTarget(e.target)) return;
            // 关闭 WebView 中键自动滚动；应用自身的中键平移通过 pointerdown 实现，不受影响。
            e.preventDefault();
        }

        function preventBrowserZoomWheel(e: WheelEvent) {
            if (!(e.ctrlKey || e.metaKey)) return;
            if (isEditableTarget(e.target)) return;
            // 禁用 Ctrl/Cmd+滚轮的 WebView 页面缩放；应用内的缩放滚轮绑定仍可正常执行。
            e.preventDefault();
            /*
             * 同一批事件正是 Web 平台上报**触控板捏合**的唯一方式（`WheelEvent`
             * 不带 `pointerType`，没有别的信号可用）。此前这里只是把它吞掉，于是
             * 触控板用户既没有浏览器缩放、也没有应用内缩放 —— 手势完全没反应。
             * 现在继续阻止页面缩放，但把同一批事件识别成捏合派发给当前表面。
             */
            if (!pinchZoomEnabledRef.current) return;
            const delta = pinchDeltaFromWheel(e);
            if (delta == null) return;
            emitPinch({ clientX: e.clientX, clientY: e.clientY, delta });
        }

        function preventBrowserFind(e: KeyboardEvent) {
            const isMac = navigator.platform?.toLowerCase().includes("mac");
            const mod = isMac ? e.metaKey : e.ctrlKey;
            const key = e.key.toLowerCase();
            if (mod && (key === "f" || key === "p" || key === "g")) {
                e.preventDefault();
            }
            // Ctrl/Cmd+R 是应用内的"开始/停止录音"快捷键；阻止 WebView 刷新但不阻断应用绑定。
            if (mod && key === "r") {
                e.preventDefault();
            }
            // 阻止浏览器页面缩放快捷键；若用户绑定 Ctrl/Cmd+数字键，应用逻辑仍会收到事件。
            if (mod && (key === "=" || key === "+" || key === "-" || key === "0")) {
                e.preventDefault();
            }
            if (e.key === "F5") {
                e.preventDefault();
            }
            if (e.key === "F3") {
                e.preventDefault();
            }
        }
        function preventContextMenu(e: MouseEvent) {
            if (IS_LINUX && linuxRightButtonDown && !e.defaultPrevented) {
                // Linux/WebKitGTK emits this event while the button is still
                // down. Hold it back and replay on pointerup; local right-drag
                // handlers that already called preventDefault/stopPropagation
                // keep full control of their own drag/context-menu flow.
                e.preventDefault();
                e.stopImmediatePropagation();
                linuxDeferredContextMenu = {
                    clientX: e.clientX,
                    clientY: e.clientY,
                    target: e.target,
                };
                return;
            }
            // 完全禁用 WebView 默认右键菜单。只调用 preventDefault，
            // 不阻止传播，因此应用内基于 contextmenu 事件实现的
            // 自定义菜单/右键拖拽仍可正常工作。
            e.preventDefault();
        }

        function preventContextMenuKey(e: KeyboardEvent) {
            // 同时屏蔽键盘触发的 WebView 默认菜单（Menu/ContextMenu 键与 Shift+F10）。
            if (e.key === "ContextMenu" || (e.key === "F10" && e.shiftKey)) {
                e.preventDefault();
            }
        }

        function altKeyDown(e: KeyboardEvent) {
            if (e.key !== "Alt") isModifierRef.current = true;
        }

        function altKeyUp(e: KeyboardEvent) {
            if (e.key === "Alt" && !isModifierRef.current) {
                e.preventDefault();
            }
            isModifierRef.current = false;
        }

        window.addEventListener("keydown", altKeyDown, true);
        window.addEventListener("keyup", altKeyUp, true);
        window.addEventListener("keydown", preventBrowserFind, true);
        window.addEventListener("keydown", preventContextMenuKey, true);
        if (IS_LINUX) {
            window.addEventListener("pointerdown", trackLinuxRightButton, true);
            window.addEventListener("pointerup", handleLinuxPointerUp, true);
            window.addEventListener("pointercancel", cancelLinuxDeferredContextMenu, true);
        }
        document.addEventListener("contextmenu", preventContextMenu, true);
        // 右键拖拽的收尾守卫：拖拽后在**任何**表面松开右键都不弹该表面的菜单
        // （菜单由松开位置的元素决定，而发起手势的表面范围有限——见模块头注释）。
        installRightDragContextMenuGuard();
        document.addEventListener("selectstart", preventNativeTextSelection, true);
        document.addEventListener("pointerdown", clearNativeTextSelection, true);
        // WebKitGTK may create the selection during the drag rather than on
        // `selectstart`; clear it again on release (editable/selectable
        // targets are left untouched).
        document.addEventListener("mouseup", clearNativeTextSelection, true);
        document.addEventListener("dragstart", preventNativeDragStart, true);
        document.addEventListener("mousedown", preventMiddleClickNative, true);
        window.addEventListener("wheel", preventBrowserZoomWheel, {
            capture: true,
            passive: false,
        });
        return () => {
            window.removeEventListener("keydown", preventBrowserFind, true);
            window.removeEventListener("keydown", preventContextMenuKey, true);
            window.removeEventListener("keydown", altKeyDown, true);
            window.removeEventListener("keyup", altKeyUp, true);
            if (IS_LINUX) {
                window.removeEventListener("pointerdown", trackLinuxRightButton, true);
                window.removeEventListener("pointerup", handleLinuxPointerUp, true);
                window.removeEventListener("pointercancel", cancelLinuxDeferredContextMenu, true);
            }
            document.removeEventListener("contextmenu", preventContextMenu, true);
            disposeRightDragContextMenuGuard();
            document.removeEventListener("selectstart", preventNativeTextSelection, true);
            document.removeEventListener("pointerdown", clearNativeTextSelection, true);
            document.removeEventListener("mouseup", clearNativeTextSelection, true);
            document.removeEventListener("dragstart", preventNativeDragStart, true);
            document.removeEventListener("mousedown", preventMiddleClickNative, true);
            window.removeEventListener("wheel", preventBrowserZoomWheel, {
                capture: true,
            } as EventListenerOptions);
        };
    }, []);

    // 剪贴板错误码可能带回退链后缀/细节，先经 token 精确映射为 i18n key
    // （未收录的码回退显示原文，保留诊断信息）。
    const mappedErrorKey = error ? (errorCodeKey[error] ?? clipboardErrorKey(error)) : "";
    const errorText = error
        ? tVars("common_label_value", {
              label: t("status_error_prefix"),
              value: mappedErrorKey ? t(mappedErrorKey as MessageKey) : error,
          })
        : statusText;

    // 构建 pitch 分析进度文本（分析中时显示在状态栏左侧）
    const pitchAnalysisText = pitchAnalysis.pending
        ? (() => {
              const parts: string[] = [t("status_analyzing_pitch")];
              if (pitchAnalysis.currentClip) {
                  parts.push(`"${pitchAnalysis.currentClip}"`);
              }
              if (pitchAnalysis.totalClips != null && pitchAnalysis.totalClips > 0) {
                  parts.push(`(${pitchAnalysis.completedClips ?? 0}/${pitchAnalysis.totalClips})`);
              }
              if (pitchAnalysis.progress != null && Number.isFinite(pitchAnalysis.progress)) {
                  parts.push(`${Math.round(pitchAnalysis.progress * 100)}%`);
              }
              return parts.join(" ");
          })()
        : null;

    // 后端"播放/预渲染"状态：active/target 镜像进 Redux（state.session
    // .playbackRenderingActive/Target）供状态栏展示；blocking（阻塞式前台
    // 预渲染的独立镜像）供播放轮询 reducer 与本文件的轮询 tick 守卫拒采
    // 陈旧传输态。进度百分比只影响状态栏展示，留在本地状态避免高频
    // progress 事件惊动订阅全量 session 的组件。
    const renderingActive = useAppSelector((state) => state.session.playbackRenderingActive);
    const renderingTarget = useAppSelector((state) => state.session.playbackRenderingTarget);
    const renderingBlocking = useAppSelector((state) => state.session.playbackBlockingRenderActive);
    const rendering = {
        active: renderingActive,
        target: renderingTarget,
        blocking: renderingBlocking,
    };

    // ── 导入等待提示（延迟点亮）────────────────────────────────────────────
    // 导入走的是盘 IO + 容器探测 + 声道判定，正常只在毫秒级结束，因此**不能**
    // 一发起就点亮加载态 —— 那会让每次导入都闪一下，比不提示更烦人。超过门槛
    // 还没结束才说明这次真的慢（慢盘 / 网络盘 / 超大文件），此时才点。
    const IMPORT_BUSY_DELAY_MS = 300;
    const importInFlight = useAppSelector((state) => state.session.importInFlight);
    const [importBusy, setImportBusy] = useState(false);
    useEffect(() => {
        if (importInFlight <= 0) {
            setImportBusy(false);
            return;
        }
        const timer = setTimeout(() => setImportBusy(true), IMPORT_BUSY_DELAY_MS);
        return () => clearTimeout(timer);
    }, [importInFlight]);

    // Listen for backend stretch progress notifications (Tauri only).
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "stretch_progress",
                    (event: { payload?: { active?: boolean; clipName?: string | null } }) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            active?: boolean;
                            clipName?: string | null;
                        };
                        const active = Boolean(payload?.active);
                        const clipName =
                            typeof payload?.clipName === "string" ? payload.clipName : null;
                        appStatusProgressBus.setStretching({ active, clipName });
                    },
                );
                // cleanup 可能发生在 await resolve 之前：已卸载则立即反注册，
                // 否则该监听器会泄漏（StrictMode 双挂载时尤其明显）。
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // Safe no-op for non-Tauri builds.
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, []);

    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "track_meter",
                    (event: {
                        payload?: {
                            tracks?: Array<{
                                trackId?: string;
                                peakLinear?: number;
                                maxPeakLinear?: number;
                                clipped?: boolean;
                            }>;
                        };
                    }) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            tracks?: Array<{
                                trackId?: string;
                                peakLinear?: number;
                                maxPeakLinear?: number;
                                clipped?: boolean;
                            }>;
                        };
                        const next: Record<
                            string,
                            {
                                peakLinear: number;
                                maxPeakLinear: number;
                                clipped: boolean;
                            }
                        > = {};

                        for (const entry of payload?.tracks ?? []) {
                            if (typeof entry?.trackId !== "string" || !entry.trackId) {
                                continue;
                            }
                            next[entry.trackId] = {
                                peakLinear:
                                    typeof entry.peakLinear === "number" &&
                                    Number.isFinite(entry.peakLinear)
                                        ? Math.max(0, entry.peakLinear)
                                        : 0,
                                maxPeakLinear:
                                    typeof entry.maxPeakLinear === "number" &&
                                    Number.isFinite(entry.maxPeakLinear)
                                        ? Math.max(0, entry.maxPeakLinear)
                                        : 0,
                                clipped: Boolean(entry.clipped),
                            };
                        }

                        dispatch(setTrackMeters(next));
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // Safe no-op for non-Tauri builds.
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);

    // 监听后端波形分析进度事件 (waveform_analysis_progress)
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;
        let fadeOutTimer: ReturnType<typeof setTimeout> | null = null;
        // 跟踪当前显示的进度值，用于防止进度回退导致的跳动
        let currentProgress = -1;
        // 跟踪当前正在 computing 的 sourcePath，用于判断是否为同一文件
        let currentComputingPath: string | null = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "waveform_analysis_progress",
                    (event: {
                        payload?: { sourcePath?: string; progress?: number; status?: string };
                    }) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            sourcePath?: string;
                            progress?: number;
                            status?: string;
                        };
                        const status = payload?.status ?? "";
                        const sourcePath =
                            typeof payload?.sourcePath === "string" ? payload.sourcePath : null;
                        const p =
                            typeof payload?.progress === "number" &&
                            Number.isFinite(payload.progress)
                                ? Math.max(0, Math.min(1, payload.progress))
                                : null;

                        // 已被用户忽略的源文件不显示任何波形分析进度；若此前正显示该
                        // 文件的进度，立即清除，避免后端仍在收尾时导致状态条停留。
                        if (sourcePath && ignoredSourcePathsRef.current.has(sourcePath)) {
                            if (currentComputingPath === sourcePath) {
                                if (fadeOutTimer) {
                                    clearTimeout(fadeOutTimer);
                                    fadeOutTimer = null;
                                }
                                currentProgress = -1;
                                currentComputingPath = null;
                                appStatusProgressBus.setWaveformAnalysis({
                                    active: false,
                                    sourcePath: null,
                                    progress: null,
                                });
                            }
                            return;
                        }

                        if (status === "computing") {
                            // 如果已在显示进度且新进度比当前低，忽略（防止并发去重后
                            // 残留的事件或不同触发点导致进度回退）
                            if (
                                currentProgress > 0 &&
                                p !== null &&
                                p < currentProgress &&
                                // 同一文件的进度回退才忽略；不同文件的 0 是正常的
                                currentComputingPath === sourcePath
                            ) {
                                return;
                            }

                            // 清除之前的淡出定时器
                            if (fadeOutTimer) {
                                clearTimeout(fadeOutTimer);
                                fadeOutTimer = null;
                            }
                            currentProgress = p ?? 0;
                            currentComputingPath = sourcePath;
                            // 提取文件名（不含路径和扩展名）
                            const fileName = sourcePath
                                ? (sourcePath
                                      .replace(/\\/g, "/")
                                      .split("/")
                                      .pop()
                                      ?.replace(/\.[^.]+$/, "") ?? sourcePath)
                                : null;
                            appStatusProgressBus.setWaveformAnalysis({
                                active: true,
                                sourcePath: fileName,
                                progress: p,
                            });
                        } else if (status === "done" || status === "cached") {
                            // 波形数据就绪信号：清除该文件的 mipmap 失败负缓存
                            // 并按需重载——打开工程瞬间的抢先请求常早于后端分析
                            // 完成，若无此重试，波形要等用户滚动/缩放才出现。
                            if (sourcePath) {
                                waveformMipmapStore.refresh(sourcePath);
                            }
                            // 完成后延迟 1.5 秒隐藏，让用户有时间看到 100%
                            if (status === "done") {
                                currentProgress = 1.0;
                                currentComputingPath = null;
                                appStatusProgressBus.setWaveformAnalysis({
                                    active: true,
                                    sourcePath: null,
                                    progress: 1.0,
                                });
                                fadeOutTimer = setTimeout(() => {
                                    if (!disposed) {
                                        currentProgress = -1;
                                        appStatusProgressBus.setWaveformAnalysis({
                                            active: false,
                                            sourcePath: null,
                                            progress: null,
                                        });
                                    }
                                }, 1500);
                            } else if (currentComputingPath === sourcePath) {
                                // 缓存命中是终态：如果之前曾进入 computing，立刻结束进度显示。
                                if (fadeOutTimer) {
                                    clearTimeout(fadeOutTimer);
                                    fadeOutTimer = null;
                                }
                                currentProgress = -1;
                                currentComputingPath = null;
                                appStatusProgressBus.setWaveformAnalysis({
                                    active: false,
                                    sourcePath: null,
                                    progress: null,
                                });
                            }
                            // cached 状态不显示进度条
                        } else if (status === "failed") {
                            // 计算失败（例如文件缺失）也是终态：立即清除“正在分析波形”，
                            // 避免缺失/被忽略的文件让左下角状态永久停留。
                            if (sourcePath === null || currentComputingPath === sourcePath) {
                                if (fadeOutTimer) {
                                    clearTimeout(fadeOutTimer);
                                    fadeOutTimer = null;
                                }
                                currentProgress = -1;
                                currentComputingPath = null;
                                appStatusProgressBus.setWaveformAnalysis({
                                    active: false,
                                    sourcePath: null,
                                    progress: null,
                                });
                            }
                        }
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // Safe no-op for non-Tauri builds.
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
            if (fadeOutTimer) clearTimeout(fadeOutTimer);
        };
    }, []);

    // Listen for backend playback priming notifications (Tauri only).
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "playback_rendering_state",
                    (event: {
                        payload?: {
                            active?: boolean;
                            progress?: number | null;
                            target?: string | null;
                            pass?: number | null;
                        };
                    }) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            active?: boolean;
                            progress?: number | null;
                            target?: string | null;
                            pass?: number | null;
                        };
                        const active = Boolean(payload?.active);
                        const pRaw = payload?.progress;
                        const p =
                            typeof pRaw === "number" && Number.isFinite(pRaw)
                                ? Math.max(0, Math.min(1, pRaw))
                                : null;
                        const target = typeof payload?.target === "string" ? payload.target : null;
                        const passRaw = payload?.pass;
                        const pass =
                            typeof passRaw === "number" && Number.isFinite(passRaw)
                                ? passRaw
                                : null;

                        // 按 target 分 ref 跟踪两类渲染的活跃状态：前台（阻塞式）
                        // 与后台渲染线程会并发发射事件，单一布尔镜像会被互相覆盖
                        // —— 后台渲染的完成事件（active=false）会把前台预渲染的
                        // 拒采窗口提前关闭，让渲染期间派发的陈旧轮询响应溜进
                        // reducer（光标跳变）。写入前先捕获该 target 的活跃值，
                        // 供下方完成跃迁判定使用。
                        //
                        // ★ 无 target 的事件按"后台"归属处理：这类载荷来自启动
                        // 收集阶段（进度回调尚未挂上）等旧前端无法归属 target 的
                        // 路径。若按"未知"跳过 ref 更新，活跃镜像（ Redux）与
                        // ref 会分叉 —— 后台完成事件来时 wasActiveForTarget 仍
                        // 判定 false，徽标会卡在与 ref 不一致的状态上。
                        const targetKey = target ?? "background";
                        let wasActiveForTarget = false;
                        if (targetKey === "original") {
                            wasActiveForTarget = originalRenderActiveRef.current;
                            originalRenderActiveRef.current = active;
                        } else {
                            wasActiveForTarget = backgroundRenderActiveRef.current;
                            backgroundRenderActiveRef.current = active;
                        }
                        const anyActive =
                            originalRenderActiveRef.current || backgroundRenderActiveRef.current;

                        // 镜像进 Redux：playbackRenderingActive/Target 驱动状态栏
                        // （任意渲染），playbackBlockingRenderActive 专供播放轮询
                        // reducer 在阻塞式预渲染期间拒采陈旧传输态（App.tsx 的
                        // 轮询 tick 守卫同样读取该状态）。进度只驱动状态栏，留在
                        // 本地状态。
                        dispatch(
                            setPlaybackRenderingState({
                                active: anyActive,
                                target: targetKey,
                                blocking: originalRenderActiveRef.current,
                            }),
                        );
                        // 进度写入状态栏。两条放行条件（见 renderProgressRef 的说明）：
                        // - 没有任何渲染在活跃（`!anyActive`）：上一次渲染已结束，
                        //   高水位作废，下一次从头开始；
                        // - pass 序号变化：后端开了新一轮 pass（打开大工程时音高
                        //   分析逐批解锁，每批一轮），这一轮必须能从 0% 重新起始。
                        // 其余情况一律要求单调不减 —— 回退对用户是纯粹的故障信号。
                        const passChanged = pass != null && renderPassRef.current !== pass;
                        if (pass != null) renderPassRef.current = pass;
                        if (!anyActive || passChanged) renderProgressRef.current = 0;
                        if (p == null) {
                            appStatusProgressBus.setRenderingProgress(null);
                        } else if (p >= renderProgressRef.current) {
                            renderProgressRef.current = p;
                            appStatusProgressBus.setRenderingProgress(p);
                        }

                        // 渲染从 active→inactive（完成）时，延迟同步一次播放状态，
                        // 使前端能感知后端已真正开始播放。跃迁按 target 判定：
                        // 两类渲染并发时共享单一 was-active 标志会被互相覆盖
                        //（后发的完成事件吞掉先发 target 的完成同步）。同样携带
                        // 当前传输纪元与派发时刻，与 30Hz 轮询响应共享乱序丢弃
                        // 与时延外推。
                        if (!active && wasActiveForTarget) {
                            setTimeout(() => {
                                dispatch(
                                    syncPlaybackState({
                                        epoch: store.getState().session._transportEpoch,
                                        dispatchedAtMs: performance.now(),
                                    }),
                                );
                            }, 200);
                        }
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // Safe no-op for non-Tauri builds.
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);

    // ── 后端一次性提示位 ────────────────────────────────────────────────────
    // 状态栏只有一个短暂提示位：渲染缓存命中统计与自动声道折叠的结论都写这里
    //（两者都是"后台做完了一件事，顺带告诉用户一声"，同时出现时后到的覆盖前者
    // 即可，不值得为它们各占一行）。
    const renderCacheShowHitStats = useAppSelector(
        (state) => state.session.renderCache.showHitStats,
    );
    const [noticeText, setNoticeText] = useState("");
    const showNotice = useCallback((text: string, holdMs = 15_000) => {
        if (!text) return;
        setNoticeText(text);
        if (noticeHideTimerRef.current) clearTimeout(noticeHideTimerRef.current);
        noticeHideTimerRef.current = setTimeout(() => setNoticeText(""), holdMs);
    }, []);
    const noticeHideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    useEffect(() => {
        if (!renderCacheShowHitStats) return;
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "render_cache_summary",
                    (event: {
                        payload?: {
                            diskHits?: number;
                            total?: number;
                        };
                    }) => {
                        if (disposed) return;
                        const payload = event?.payload ?? {};
                        // 后端保证：这些是**工程级**累计值（分子按 clip 去重、分母是
                        // 工程需要渲染的 clip 总数），且一次工程加载只在收敛后上报
                        // 一次。不要在这里对多次事件做累加 —— 那正是"逐轮数字"的
                        // 老毛病（6/6 → 5/35 → 156/465）。
                        const hits = Number(payload.diskHits ?? 0);
                        const total = Number(payload.total ?? 0);
                        // 没有磁盘命中就不打扰用户（首次打开工程本就无缓存）。
                        if (!(Number.isFinite(hits) && hits > 0 && total > 0)) return;
                        // 各段都是完整分句、自身不带前导分隔符，由这里统一用
                        // " · " 连接。
                        const hitText = tf("status_render_cache_summary")
                            .replace("{hits}", String(hits))
                            .replace("{total}", String(total));
                        showNotice(hitText);
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // 非 Tauri 环境（浏览器调试）无事件系统：静默跳过。
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [renderCacheShowHitStats, showNotice, tf]);

    // ── 后台声道折叠反馈 ────────────────────────────────────────────────────
    // 打开工程 / 导入 / 换源后，没有权威判定档案的 Take 会被后台扫描按策略折叠为
    // 单声道（渲染耗时减半）。折叠改的是 Take 的 `channel_mode`，也就是时间线的
    // 真实状态，而前端不会轮询时间线 —— 必须让它看到，否则界面一直显示旧的
    // 声道带数，用户会以为软件抽风。
    //
    // 【为什么要合并 + 延迟，而不是每批都处理】扫描按 64 个 Take 一批推进，一批
    // 一次事件；逐批重拉时间线是拿一整份时间线换几十毫秒的收敛。但也不能只在
    // 整轮结束才处理 —— 大工程上那会让界面长时间停在陈旧状态。
    //
    // 折中：**延迟一个很短的门槛再动手**。绝大多数扫描在门槛内就跑完了，于是
    // 行为和以前完全一致（界面变化与解释同时出现，不会先变后解释）；只有真正
    // 慢的扫描才会先渐进收敛。提示语仍只在整轮结束时给出一次，用累计计数，
    // 保证数字是准的而不是流水账。
    const CHANNEL_SCAN_FEEDBACK_DELAY_MS = 400;
    const channelScanFeedbackRef = useRef<{
        timer: ReturnType<typeof setTimeout> | null;
        folded: number;
        pending: number;
        finished: boolean;
    }>({ timer: null, folded: 0, pending: 0, finished: false });

    const flushChannelScanFeedback = useCallback(() => {
        const state = channelScanFeedbackRef.current;
        if (state.timer != null) {
            clearTimeout(state.timer);
            state.timer = null;
        }
        const folded = state.folded;
        const pending = state.pending;
        // 折叠改的是时间线的真实状态，必须主动重拉一次才能收敛到界面。
        if (folded > 0) void dispatch(fetchTimeline());
        if (!state.finished) return;
        if (folded <= 0 && pending <= 0) return;
        // 各段完整成句、无前导标点，这里统一连接。
        const parts: string[] = [];
        if (folded > 0) {
            parts.push(plural("status_channel_scan_folded", folded));
        }
        if (pending > 0) {
            parts.push(plural("status_channel_scan_pending", pending));
        }
        showNotice(parts.join(" · "), 20_000);
    }, [dispatch, showNotice, plural]);

    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;
        const feedback = channelScanFeedbackRef.current;

        async function setup() {
            try {
                const mod = window.__HFS_PLUGIN_BOOTSTRAP__ ? await import("./services/hostEvents") : await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "channel_scan_progress",
                    (event: {
                        payload?: {
                            done?: number;
                            total?: number;
                            folded?: number;
                            pending?: number;
                            finished?: boolean;
                        };
                    }) => {
                        if (disposed) return;
                        const payload = event?.payload ?? {};
                        const folded = Number(payload.folded ?? 0);
                        const pending = Number(payload.pending ?? 0);
                        // 计数是**整轮累计值**，取 max 只作防御（乱序事件不应让数字倒退）。
                        feedback.folded = Math.max(feedback.folded, folded);
                        feedback.pending = Math.max(feedback.pending, pending);
                        if (payload.finished) {
                            feedback.finished = true;
                            flushChannelScanFeedback();
                            // 整轮已结束，为下一轮复位（后台扫描会连跑多轮）。
                            feedback.finished = false;
                            feedback.folded = 0;
                            feedback.pending = 0;
                            return;
                        }
                        // 还没有折叠：没有任何需要让界面收敛的状态，不值得排一次重拉。
                        if (feedback.folded <= 0 || feedback.timer != null) return;
                        feedback.timer = setTimeout(() => {
                            feedback.timer = null;
                            flushChannelScanFeedback();
                        }, CHANNEL_SCAN_FEEDBACK_DELAY_MS);
                    },
                );
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // 非 Tauri 环境（浏览器调试）无事件系统：静默跳过。
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (feedback.timer != null) {
                clearTimeout(feedback.timer);
                feedback.timer = null;
            }
            if (unlisten) unlisten();
        };
    }, [flushChannelScanFeedback]);

    const runtimeRef = useRef({
        isPlaying: false,
        hasSynthesized: false,
        toolMode: "draw" as import("./features/session/sessionTypes").ToolMode,
        drawToolMode: "draw" as import("./features/session/sessionTypes").DrawToolMode,
    });

    const playbackSyncInFlightRef = useRef(false);
    // 按 target 隔离的渲染活跃状态（见 playback_rendering_state 监听器）：
    // 前台（阻塞式）与后台渲染线程并发发射事件，必须分 ref 跟踪才能把
    // "阻塞式前台预渲染"窗口的开关与后台渲染的生命周期解耦。
    const originalRenderActiveRef = useRef(false);
    const backgroundRenderActiveRef = useRef(false);
    // 渲染进度的高水位与 pass 序号（见 playback_rendering_state 监听器）：
    // 后端已保证同轮单调（renderer/progress 的单调闸门），这里再守一道是因为
    // Tauri 事件是异步投递的 —— 旧代渲染线程的迟到事件有可能在 active=false
    // 之后才到达。pass 序号用来区分"新一轮重新起始"与"同一轮回退"：
    // 前者必须放行（打开大工程时每批各跑一轮，每轮都该从 0% 涨到 100%），
    // 后者必须拦下。
    const renderPassRef = useRef<number | null>(null);
    const renderProgressRef = useRef(0);

    const closeWindowNow = useCallback(async () => {
        try {
            await runConfirmedExitClose({
                markAllowClose: () => {
                    allowWindowCloseRef.current = true;
                },
                destroyWindow: async () => {
                    const mod = await loadStandaloneWindowApi();
                    const currentWindow = mod.getCurrentWindow();
                    await currentWindow.destroy();
                },
                closeWindow: async () => {
                    await coreApi.closeWindow();
                },
            });
        } catch (error) {
            allowWindowCloseRef.current = false;
            throw error;
        }
    }, []);

    const promptUnsavedAction = useCallback(
        (mode: "switch" | "exit", action: () => Promise<void>) => {
            pendingUnsavedActionRef.current = action;
            setUnsavedDialog({ open: true, mode });
        },
        [],
    );

    const runOrPromptUnsavedAction = useCallback(
        (mode: "switch" | "exit", action: () => Promise<void>) => {
            if (!projectDirty) {
                // action 可能 reject（如 openProjectFromDialog().unwrap() 后端失败、
                // closeWindowNow 双双失败）：干净工程这一支没有弹窗，若不接住，
                // 拒绝会变成 unhandledrejection 被全局上报当成崩溃。错误状态已由
                // Redux 呈现给用户，这里只需静默吞掉。
                void action().catch(() => {});
                return;
            }
            promptUnsavedAction(mode, action);
        },
        [projectDirty, promptUnsavedAction],
    );

    const executePendingUnsavedAction = useCallback(async () => {
        const action = pendingUnsavedActionRef.current;
        const mode = unsavedDialog.mode;
        pendingUnsavedActionRef.current = null;
        setUnsavedDialog((current) => ({ ...current, open: false }));
        if (action) {
            try {
                await action();
            } catch (error) {
                pendingUnsavedActionRef.current = action;
                setUnsavedDialog({ open: true, mode });
                throw error;
            }
        }
    }, [unsavedDialog.mode]);

    const cancelUnsavedAction = useCallback(() => {
        pendingUnsavedActionRef.current = null;
        setUnsavedDialog((current) => ({ ...current, open: false }));
    }, []);

    const discardUnsavedAndContinue = useCallback(() => {
        void executePendingUnsavedAction().catch(() => {});
    }, [executePendingUnsavedAction]);

    const saveUnsavedAndContinue = useCallback(() => {
        void (async () => {
            try {
                const result = (await dispatch(
                    projectPath ? saveProjectRemote() : saveProjectAsRemote(),
                ).unwrap()) as { canceled?: boolean; versionConflict?: boolean };
                // 取消 或 命中"目标版本不一致"确认框：暂不执行后续操作，
                // 等待用户完成保存（继续保存成功后由后续入口继续，或重新操作）。
                if (result?.canceled || result?.versionConflict) {
                    return;
                }
                await executePendingUnsavedAction();
            } catch {
                // Keep the dialog open so the user can retry or cancel.
            }
        })();
    }, [dispatch, executePendingUnsavedAction, projectPath]);

    const showProjectVersionConfirmationIfNeeded = useCallback((result: unknown) => {
        const payload = result as
            | {
                  projectVersionTooNew?: boolean;
                  path?: string;
                  projectFileVersion?: number;
                  currentProjectFileVersion?: number;
              }
            | undefined;
        if (payload?.projectVersionTooNew && payload.path) {
            setProjectVersionDialog({
                open: true,
                path: payload.path,
                fileVersion: Number(payload.projectFileVersion ?? 0),
                currentVersion: Number(payload.currentProjectFileVersion ?? 0),
            });
        }
    }, []);

    const confirmContinueLoadingNewerProject = useCallback(() => {
        const path = projectVersionDialog.path;
        setProjectVersionDialog((current) => ({ ...current, open: false }));
        if (!path) return;
        void dispatch(openProjectFromPathForced(path));
    }, [dispatch, projectVersionDialog.path]);

    const cancelContinueLoadingNewerProject = useCallback(() => {
        setProjectVersionDialog((current) => ({ ...current, open: false }));
    }, []);

    // ── 保存/另存为目标存在版本不一致工程文件时的确认操作 ──
    const cancelSaveVersionConflict = useCallback(() => {
        dispatch(closeSaveVersionConflictDialog());
    }, [dispatch]);

    // 用户选择"另存为"：关闭确认框并重新弹出另存为选择器。
    const saveAsFromVersionConflict = useCallback(() => {
        dispatch(closeSaveVersionConflictDialog());
        void dispatch(saveProjectAsRemote());
    }, [dispatch]);

    // 用户确认"继续保存"：向已选路径执行强制覆盖保存。
    const continueForceSave = useCallback(() => {
        const path = saveVersionConflictDialog?.path;
        dispatch(closeSaveVersionConflictDialog());
        if (!path) return;
        void dispatch(saveProjectToPathRemote(path));
    }, [dispatch, saveVersionConflictDialog?.path]);

    const handleNewProject = useCallback(() => {
        runOrPromptUnsavedAction("switch", async () => {
            await dispatch(newProjectRemote()).unwrap();
        });
    }, [dispatch, runOrPromptUnsavedAction]);

    const handleOpenProject = useCallback(() => {
        runOrPromptUnsavedAction("switch", async () => {
            const result = await dispatch(openProjectFromDialog()).unwrap();
            showProjectVersionConfirmationIfNeeded(result);
        });
    }, [dispatch, runOrPromptUnsavedAction, showProjectVersionConfirmationIfNeeded]);

    const handleOpenRecentProject = useCallback(
        (path: string) => {
            runOrPromptUnsavedAction("switch", async () => {
                const result = await dispatch(openProjectFromPath(path)).unwrap();
                showProjectVersionConfirmationIfNeeded(result);
            });
        },
        [dispatch, runOrPromptUnsavedAction, showProjectVersionConfirmationIfNeeded],
    );

    const handleImportProject = useCallback(async () => {
        try {
            const picked = await dispatch(pickProjectToImport()).unwrap();
            if (!picked?.ok || picked.canceled || !picked.path) {
                return;
            }
            setProjectImportPick({ open: true, path: picked.path });
        } catch {
            // Reducer already surfaces the error.
        }
    }, [dispatch]);

    const handleImportProjectConfirmed = useCallback(
        (options: { placeAtPlayhead: boolean; importTempoMap: boolean }) => {
            const { path } = projectImportPick;
            setProjectImportPick({ open: false, path: null });
            if (!path) return;
            void dispatch(
                importProjectFromPath({
                    projectPath: path,
                    placeAtPlayhead: options.placeAtPlayhead,
                    importTempoMap: options.importTempoMap,
                }),
            );
        },
        [dispatch, projectImportPick],
    );

    const handleExternalFileAction = useCallback(
        (kind: ExternalFileActionKind, path: string) => {
            const normalized = String(path ?? "").trim();
            if (!normalized) return;
            if (kind === "openProject") {
                runOrPromptUnsavedAction("switch", async () => {
                    const result = await dispatch(openProjectFromPath(normalized)).unwrap();
                    showProjectVersionConfirmationIfNeeded(result);
                });
                return;
            }
            if (kind === "importVocalShifter") {
                void dispatch(openVocalShifterFromPath(normalized));
                return;
            }
            if (kind === "importReaper") {
                void dispatch(openReaperFromPath(normalized));
                return;
            }
            if (kind === "importAudio") {
                void dispatch(importAudioFromPath(normalized));
            }
        },
        [dispatch, runOrPromptUnsavedAction, showProjectVersionConfirmationIfNeeded],
    );

    const handleExitApp = useCallback(() => {
        runOrPromptUnsavedAction("exit", closeWindowNow);
    }, [closeWindowNow, runOrPromptUnsavedAction]);

    const handleAutoBackupSettingsSaved = useCallback((settings: AutoBackupSettings) => {
        const interval = Number(settings.timedBackupIntervalSec);
        setAutoBackupSettings({
            ...DEFAULT_AUTO_BACKUP_SETTINGS,
            ...settings,
            timedBackupIntervalSec: Number.isFinite(interval)
                ? Math.max(1, Math.floor(interval))
                : DEFAULT_AUTO_BACKUP_SETTINGS.timedBackupIntervalSec,
            timedBackupPathTemplate:
                String(settings.timedBackupPathTemplate ?? "").trim() ||
                DEFAULT_AUTO_BACKUP_SETTINGS.timedBackupPathTemplate,
        });
    }, []);

    useAutoBackupScheduler({
        settings: autoBackupSettings,
        paramsEpoch,
        projectDirty,
        status,
    });

    useEffect(() => {
        void dispatch(fetchTimeline());
        void dispatch(refreshRuntime());
        // loadUiSettings 不在这里发起：UI 设置的唯一一次加载由上方"加载 UI
        // 持久化设置"的 effect 持有（unwrap 后回填 MIDI 字段），否则启动会有
        // 两轮 get_ui_settings 往返（后端的 get_ui_settings 不是纯读）。
        if (!isPluginMode()) void dispatch(loadRecordingSettings());
        // 【必须显式 hydrate】thunk 只负责取回磁盘内容，把结果写进切片是这里的
        // 责任。漏掉这一步的后果不是"界面不好看"，而是**布局永远不落盘**：
        // `hydrated` 闸门始终为 false，持久化副作用永不触发（曾实际发生）。
        void dispatch(loadDockSettings())
            .unwrap()
            .then((payload) => {
                dispatch(hydrateDock(payload));
                // 归一化与面板注册都已完成，此时才能安全处理"套用启动预设"与
                // "把浮窗收回停靠位"这两件事（见 `finalizeDockHydration`）。
                finalizeDockHydration(dispatch, store.getState);
            })
            .catch(() => {
                // 读不到设置时保持出厂布局；`hydrated` 仍为 false，因此不会把
                // 默认布局写回去覆盖磁盘内容。
            });
    }, [dispatch]);

    useEffect(() => {
        let cancelled = false;

        async function loadAutoBackupSettings() {
            if (isPluginMode()) return;
            try {
                const settings = await projectApi.getAutoBackupSettings();
                if (cancelled || !settings) return;
                handleAutoBackupSettingsSaved(settings);
            } catch {
                // 保持默认配置。
            }
        }

        void loadAutoBackupSettings();
        return () => {
            cancelled = true;
        };
    }, [handleAutoBackupSettingsSaved]);

    // ── 后台预渲染：paramsEpoch 变更时自动触发 ──────────────────────────────────
    const autoBackgroundRender = useAppSelector((state) => state.session.autoBackgroundRender);
    const prevParamsEpochRef = useRef(paramsEpoch);
    useEffect(() => {
        if (isPluginMode() || !autoBackgroundRender) return;
        // 跳过初始加载（prevParamsEpochRef 与当前 epoch 相同时跳过）
        if (prevParamsEpochRef.current === paramsEpoch) return;
        prevParamsEpochRef.current = paramsEpoch;

        // 防抖：延迟 200ms 后再触发，避免连续编辑时频繁启动渲染线程
        const timer = setTimeout(() => {
            void (async () => {
                try {
                    const result = await webApi.startBackgroundRender();
                    if (result?.skipped) {
                        // 已在渲染中，无需重复启动
                        return;
                    }
                } catch {
                    // 静默失败；后台渲染为可选增强功能
                }
            })();
        }, 200);

        return () => clearTimeout(timer);
    }, [paramsEpoch, autoBackgroundRender]);

    // consume 语义只应执行一次：依赖 handleExternalFileAction（其随
    // projectDirty 翻转而重建）会让该 effect 在首次编辑后重复发起 IPC。
    const consumeStartupProjectPathRef = useRef(handleExternalFileAction);
    useEffect(() => {
        consumeStartupProjectPathRef.current = handleExternalFileAction;
    }, [handleExternalFileAction]);

    useEffect(() => {
        let canceled = false;

        async function consumeStartupProjectPath() {
            try {
                const result = await projectApi.consumeStartupProjectPath();
                const startupPath = String(result?.path ?? "").trim();
                const kind = detectExternalActionKindFromPath(startupPath);
                if (!canceled && startupPath && kind) {
                    consumeStartupProjectPathRef.current(kind, startupPath);
                }
            } catch {
                // no-op
            }
        }

        void consumeStartupProjectPath();
        return () => {
            canceled = true;
        };
    }, []);

    useEffect(() => {
        function onOpenProjectPath(event: Event) {
            const detail = (event as CustomEvent<ExternalFileActionDetail>).detail;
            const path = String(detail?.path ?? "").trim();
            const kind = detail?.kind ?? detectExternalActionKindFromPath(path);
            if (!path || !kind) return;
            handleExternalFileAction(kind, path);
        }

        window.addEventListener(OPEN_PROJECT_PATH_EVENT, onOpenProjectPath as EventListener);
        return () => {
            window.removeEventListener(OPEN_PROJECT_PATH_EVENT, onOpenProjectPath as EventListener);
        };
    }, [handleExternalFileAction]);

    // 文件浏览器拖拽工程文件 → "导入工程"：携带路径打开导入选项对话框。
    useEffect(() => {
        function onImportProjectPick(event: Event) {
            const path = String(
                (event as CustomEvent<{ path?: string }>).detail?.path ?? "",
            ).trim();
            if (!path) return;
            setProjectImportPick({ open: true, path });
        }

        window.addEventListener(IMPORT_PROJECT_PICK_EVENT, onImportProjectPick as EventListener);
        return () => {
            window.removeEventListener(
                IMPORT_PROJECT_PICK_EVENT,
                onImportProjectPick as EventListener,
            );
        };
    }, []);

    /*
     * 文件浏览器右键「导入 MIDI…」：MIDI 导入对话框的十余项选项状态都由本组件持有
     * （见 `midiClip*` 一组 state），只有它能在任何面板布局下渲染那个对话框。
     * 文件浏览器因此只发一条请求，这里接住并用自己的默认值打开。
     */
    useEffect(() => {
        function onImportMidi(event: Event) {
            const detail = (event as CustomEvent<ImportMidiRequestDetail>).detail;
            const path = String(detail?.path ?? "").trim();
            if (!path) return;
            setMidiClipPath(path);
            setMidiClipStartSec(detail?.startSec ?? 0);
            setMidiClipTrackId(detail?.trackId ?? null);
            setMidiClipClipboardGuid(null);
            setMidiClipDialogOpen(true);
        }

        window.addEventListener(IMPORT_MIDI_PATH_EVENT, onImportMidi as EventListener);
        return () => {
            window.removeEventListener(IMPORT_MIDI_PATH_EVENT, onImportMidi as EventListener);
        };
    }, []);

    useEffect(() => {
        runtimeRef.current = {
            isPlaying: Boolean(runtimeIsPlaying),
            hasSynthesized: Boolean(runtimeHasSynthesized),
            toolMode,
            drawToolMode,
        };
    }, [runtimeIsPlaying, runtimeHasSynthesized, toolMode, drawToolMode]);

    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = await loadStandaloneWindowApi();
                const currentWindow = mod.getCurrentWindow();
                unlisten = await currentWindow.onCloseRequested((event: CloseRequestedEvent) => {
                    if (allowWindowCloseRef.current) {
                        allowWindowCloseRef.current = false;
                        return;
                    }
                    // 读取 ref 的值，无需重建整个监听器
                    if (!projectDirtyRef.current) {
                        return;
                    }
                    event.preventDefault();
                    if (!disposed) {
                        promptUnsavedAction("exit", closeWindowNow);
                    }
                });
            } catch {
                // 非 Tauri 环境：忽略窗口关闭事件监听失败
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [closeWindowNow, promptUnsavedAction]); // 剔除 projectDirty 依赖，只绑定一次

    // 检测已导入的媒体源文件是否被外部修改或删除。
    // 触发时机：窗口重新获得焦点，以及工程/导入内容刚替换完成时。
    const checkSourceFileChanges = useCallback(async () => {
        if (isPluginMode()) return;
        if (
            sourceFileCheckBusyRef.current ||
            sourceFileChangeHandlingRef.current ||
            sourceFileDialogOpenRef.current
        ) {
            return;
        }
        sourceFileCheckBusyRef.current = true;
        try {
            const result = (await webApi.checkSourceFilesChanged()) as
                | { changed?: SourceFileChange[] }
                | undefined;
            const ignored = ignoredSourcePathsRef.current;
            const changes = normalizeSourceFileChanges(result?.changed ?? []).filter(
                (change) => !ignored.has(change.source_path),
            );

            const modified = changes.filter((change) => change.change === "modified");

            // 设置开启时，后台自动重新加载“已修改”的文件，不弹出确认窗口。
            if (autoReloadModifiedMedia && modified.length > 0) {
                sourceFileChangeHandlingRef.current = true;
                try {
                    const uniqueByPath = new Map<string, SourceFileChange>();
                    for (const change of modified) {
                        if (!uniqueByPath.has(change.source_path)) {
                            uniqueByPath.set(change.source_path, change);
                        }
                    }
                    for (const item of uniqueByPath.values()) {
                        try {
                            await dispatch(
                                replaceClipSourceRemote({
                                    clipIds: item.clip_id ? [item.clip_id] : [],
                                    newSourcePath: item.source_path,
                                    replaceSameSource: true,
                                }),
                            ).unwrap();
                        } catch {
                            // 单个文件失败时继续处理其他文件；稍后统一复核。
                        }
                    }
                } finally {
                    sourceFileChangeHandlingRef.current = false;
                }

                const refreshed = (await webApi.checkSourceFilesChanged()) as
                    | { changed?: SourceFileChange[] }
                    | undefined;
                const remaining = normalizeSourceFileChanges(refreshed?.changed ?? []).filter(
                    (change) => !ignored.has(change.source_path),
                );

                if (remaining.length > 0) {
                    markMissingSourceFilesUnavailable(remaining);
                    const items = remaining.map((change) => ({
                        ...change,
                        action: "pending" as const,
                    }));
                    sourceFileInitialChangesRef.current = items;
                    sourceFileDialogOpenRef.current = true;
                    setSourceFileChangedDialog({ open: true, changes: items });
                }
                return;
            }

            if (changes.length > 0) {
                markMissingSourceFilesUnavailable(changes);
                const items = changes.map((change) => ({ ...change, action: "pending" as const }));
                sourceFileInitialChangesRef.current = items;
                sourceFileDialogOpenRef.current = true;
                setSourceFileChangedDialog({
                    open: true,
                    changes: items,
                });
            }
        } catch {
            // 静默失败；此检测为可选增强功能
        } finally {
            sourceFileCheckBusyRef.current = false;
        }
    }, [autoReloadModifiedMedia, dispatch]);

    // 窗口已打开时执行“刷新”：按 clip_id / 路径匹配后端最新状态。
    // 现有条目永不删除；已替换文件失效时回退为原始文件的“已删除”状态。
    const refreshSourceFileChanges = useCallback(async () => {
        if (sourceFileCheckBusyRef.current || sourceFileChangeHandlingRef.current) {
            return;
        }
        sourceFileCheckBusyRef.current = true;
        try {
            const result = (await webApi.checkSourceFilesChanged()) as
                | { changed?: SourceFileChange[] }
                | undefined;
            const rawChanges = normalizeSourceFileChanges(result?.changed ?? []);

            setSourceFileChangedDialog((prev) => {
                const changes = mergeLatestSourceFileChanges(prev.changes, rawChanges);
                const knownClipIds = new Set(prev.changes.map((item) => item.clip_id));
                for (const change of rawChanges) {
                    if (!knownClipIds.has(change.clip_id)) {
                        const existingInitial = sourceFileInitialChangesRef.current.some(
                            (item) => item.clip_id === change.clip_id,
                        );
                        if (!existingInitial) {
                            sourceFileInitialChangesRef.current = [
                                ...sourceFileInitialChangesRef.current,
                                { ...change, action: "pending" as const },
                            ];
                        }
                    }
                }
                return { ...prev, open: true, changes };
            });

            markMissingSourceFilesUnavailable(rawChanges);
        } catch {
            // 刷新失败时保持当前列表，避免误导用户。
        } finally {
            sourceFileCheckBusyRef.current = false;
        }
    }, []);

    useEffect(() => {
        function onFocus() {
            // 窗口仍然打开时执行刷新而不是重新弹窗；窗口未打开时才执行常规检测。
            if (sourceFileDialogOpenRef.current) {
                void refreshSourceFileChanges();
            } else {
                void checkSourceFileChanges();
            }
        }
        window.addEventListener("focus", onFocus);
        return () => {
            window.removeEventListener("focus", onFocus);
        };
    }, [checkSourceFileChanges, refreshSourceFileChanges]);

    useEffect(() => {
        if (SOURCE_FILE_CHECK_TRIGGER_STATUSES.has(status)) {
            void checkSourceFileChanges();
        }
    }, [status, checkSourceFileChanges]);

    // 文件菜单：重新捕获缺失媒体。
    // 清除本会话的忽略记录，重新拉取所有缺失/修改的媒体并始终打开窗口（列表可为空）。
    const handleRecaptureMissingMedia = useCallback(async () => {
        if (sourceFileCheckBusyRef.current || sourceFileChangeHandlingRef.current) return;
        ignoredSourcePathsRef.current.clear();
        sourceFileDialogOpenRef.current = false;
        sourceFileCheckBusyRef.current = true;
        try {
            const result = (await webApi.checkSourceFilesChanged()) as
                | { changed?: SourceFileChange[] }
                | undefined;
            const changes = normalizeSourceFileChanges(result?.changed ?? []);
            markMissingSourceFilesUnavailable(changes);
            const items = changes.map((change) => ({ ...change, action: "pending" as const }));
            sourceFileInitialChangesRef.current = items;
            sourceFileDialogOpenRef.current = true;
            setSourceFileChangedDialog({ open: true, changes: items });
        } catch {
            // 读取失败时也打开空窗口，让用户可再次手动操作。
            sourceFileInitialChangesRef.current = [];
            sourceFileDialogOpenRef.current = true;
            setSourceFileChangedDialog({ open: true, changes: [] });
        } finally {
            sourceFileCheckBusyRef.current = false;
        }
    }, []);

    const updateSourceFileChangeItem = useCallback(
        (
            clipId: string,
            patch: Partial<
                Pick<
                    SourceFileChangedItem,
                    | "action"
                    | "reloadedPath"
                    | "reloadAttempted"
                    | "change"
                    | "candidates"
                    | "selectedCandidatePath"
                >
            >,
        ) => {
            setSourceFileChangedDialog((prev) => ({
                ...prev,
                changes: prev.changes.map((item) =>
                    item.clip_id === clipId ? { ...item, ...patch } : item,
                ),
            }));
        },
        [],
    );

    const ignoreSourceFileChangeItem = useCallback(
        (item: SourceFileChangedItem) => {
            if (item.action === "processing") return;
            // 本次会话内不再重复提示该源文件路径。
            ignoredSourcePathsRef.current.add(item.source_path);
            // 被忽略的文件不再继续请求/展示波形分析；若已缓存的数据仍可正常显示。
            waveformMipmapStore.markUnavailable(item.source_path);
            updateSourceFileChangeItem(item.clip_id, { action: "ignored" });
        },
        [updateSourceFileChangeItem],
    );

    const ignoreAllSourceFileChanges = useCallback(() => {
        if (sourceFileChangeHandlingRef.current) return;
        const targets = sourceFileChangedDialog.changes.filter(
            (item) =>
                item.action !== "ignored" &&
                item.action !== "reloaded" &&
                item.action !== "replaced" &&
                item.action !== "processing",
        );
        if (targets.length === 0) return;

        for (const item of targets) {
            ignoredSourcePathsRef.current.add(item.source_path);
            waveformMipmapStore.markUnavailable(item.source_path);
        }
        const ignoredClipIds = new Set(targets.map((item) => item.clip_id));
        setSourceFileChangedDialog((prev) => ({
            ...prev,
            changes: prev.changes.map((item) =>
                ignoredClipIds.has(item.clip_id) ? { ...item, action: "ignored" } : item,
            ),
        }));
    }, [sourceFileChangedDialog.changes]);

    const applySourceFileReplacement = useCallback(
        async (
            item: SourceFileChangedItem,
            replacementPath: string,
            mode: "reload" | "replace",
        ) => {
            if (item.action === "processing" || sourceFileChangeHandlingRef.current) {
                return false;
            }

            updateSourceFileChangeItem(item.clip_id, { action: "processing" });
            sourceFileChangeHandlingRef.current = true;
            try {
                await dispatch(
                    replaceClipSourceRemote({
                        clipIds: item.clip_id ? [item.clip_id] : [],
                        newSourcePath: replacementPath,
                        replaceSameSource: true,
                    }),
                ).unwrap();
                updateSourceFileChangeItem(item.clip_id, {
                    action: mode === "reload" ? "reloaded" : "replaced",
                    reloadedPath: replacementPath,
                    selectedCandidatePath: undefined,
                });
                return true;
            } catch {
                // 已处理过的条目再次重选失败时保留原状态；未处理条目才显示失败。
                if (item.action === "reloaded" || item.action === "replaced") {
                    updateSourceFileChangeItem(item.clip_id, { action: item.action });
                } else if (item.action === "ignored") {
                    updateSourceFileChangeItem(item.clip_id, { action: "ignored" });
                } else {
                    updateSourceFileChangeItem(item.clip_id, { action: "failed" });
                }
                return false;
            } finally {
                sourceFileChangeHandlingRef.current = false;
            }
        },
        [dispatch, updateSourceFileChangeItem],
    );

    const replaceSourceFileChangeItem = useCallback(
        async (item: SourceFileChangedItem) => {
            if (item.action === "processing" || sourceFileChangeHandlingRef.current) {
                return;
            }

            const dialogTitle = t("recapture_missing_media_replace_dialog_title").replace(
                "{name}",
                item.clip_name || item.source_path,
            );
            const picked = await coreApi.openAudioDialogForSource(item.source_path, dialogTitle);
            if (!picked?.ok || picked.canceled || !picked.path) {
                // 取消时保留原状态（含“已忽略 / 已重新加载 / 已替换”）。
                return;
            }
            await applySourceFileReplacement(item, picked.path, "replace");
        },
        [applySourceFileReplacement, t],
    );

    const reloadSourceFileChangeItem = useCallback(
        async (item: SourceFileChangedItem) => {
            if (
                item.action === "processing" ||
                item.change !== "modified" ||
                sourceFileChangeHandlingRef.current
            ) {
                return;
            }

            // 按下“重新加载”后，该按钮语义立即转为“替换”。
            updateSourceFileChangeItem(item.clip_id, {
                action: "processing",
                reloadAttempted: true,
            });
            sourceFileChangeHandlingRef.current = true;
            let becameMissing = false;
            try {
                const result = (await webApi.checkSourceFilesChanged()) as
                    | { changed?: SourceFileChange[] }
                    | undefined;
                const latest = normalizeSourceFileChanges(result?.changed ?? []).find(
                    (change) =>
                        change.clip_id === item.clip_id ||
                        change.source_path === item.source_path ||
                        (item.reloadedPath && change.source_path === item.reloadedPath),
                );
                if (latest?.change === "deleted") {
                    // 用户在点击后中途删除了文件：状态从“已修改”变为“已删除”，
                    // 按钮转成“替换”，无需额外弹窗。
                    updateSourceFileChangeItem(item.clip_id, {
                        action: "failed",
                        change: "deleted",
                    });
                    becameMissing = true;
                }
            } catch {
                // 检测失败时不中断，继续尝试按原路径重新加载。
            } finally {
                sourceFileChangeHandlingRef.current = false;
            }

            if (!becameMissing) {
                await applySourceFileReplacement(
                    item,
                    item.reloadedPath ?? item.source_path,
                    "reload",
                );
            }
        },
        [applySourceFileReplacement, updateSourceFileChangeItem],
    );

    const reloadAllModifiedSourceFiles = useCallback(async () => {
        const targets = sourceFileChangedDialog.changes.filter(
            (item) =>
                item.change === "modified" &&
                !item.reloadAttempted &&
                item.action !== "ignored" &&
                item.action !== "reloaded" &&
                item.action !== "replaced" &&
                item.action !== "processing",
        );
        for (const target of targets) {
            await reloadSourceFileChangeItem(target);
        }
    }, [reloadSourceFileChangeItem, sourceFileChangedDialog.changes]);

    const searchSourceFileReplacements = useCallback(async () => {
        const targets = sourceFileChangedDialog.changes.filter(
            (item) =>
                item.action !== "ignored" &&
                item.action !== "reloaded" &&
                item.action !== "replaced" &&
                item.action !== "processing",
        );
        if (targets.length === 0 || sourceFileSearchBusy || sourceFileChangeHandlingRef.current) {
            return;
        }

        let picked: { ok: boolean; canceled?: boolean; path?: string } | undefined;
        try {
            picked = await fileBrowserApi.pickDirectory();
        } catch {
            return;
        }
        if (!picked?.ok || picked.canceled || !picked.path) return;

        setSourceFileSearchBusy(true);
        try {
            const result = await webApi.searchSourceFileReplacements(
                picked.path,
                targets.map((item) => item.clip_id),
                sourceFileSearchMode,
            );
            const targetIds = new Set(targets.map((item) => item.clip_id));
            setSourceFileChangedDialog((prev) => ({
                ...prev,
                changes: prev.changes.map((item) => {
                    if (!targetIds.has(item.clip_id)) return item;
                    const candidates = result.matches?.[item.clip_id] ?? [];
                    return {
                        ...item,
                        candidates,
                        selectedCandidatePath: candidates[0]?.path,
                    };
                }),
            }));
        } catch {
            // 搜索失败保持现有状态；用户可以重新选择文件夹再试。
        } finally {
            setSourceFileSearchBusy(false);
        }
    }, [sourceFileChangedDialog.changes, sourceFileSearchBusy, sourceFileSearchMode]);

    const selectSourceFileMatchCandidate = useCallback(
        (clipId: string, candidatePath: string) => {
            updateSourceFileChangeItem(clipId, { selectedCandidatePath: candidatePath });
        },
        [updateSourceFileChangeItem],
    );

    const applySelectedSourceFileMatch = useCallback(
        async (item: SourceFileChangedItem) => {
            const candidatePath = item.selectedCandidatePath;
            if (!candidatePath) return;
            await applySourceFileReplacement(item, candidatePath, "replace");
        },
        [applySourceFileReplacement],
    );

    const applyAllExactSourceFileMatches = useCallback(async () => {
        const targets = sourceFileChangedDialog.changes.flatMap((item) => {
            if (
                item.action === "ignored" ||
                item.action === "reloaded" ||
                item.action === "replaced" ||
                item.action === "processing"
            ) {
                return [];
            }
            const candidates = item.candidates ?? [];
            const exactCandidates = candidates.filter((candidate) => candidate.exact_hash);
            if (exactCandidates.length === 0) return [];
            // 若用户当前选中的恰是哈希完全匹配项，优先尊重用户的选择；
            // 否则按候选顺序取第一个哈希完全匹配项。
            const selected = item.selectedCandidatePath
                ? candidates.find((candidate) => candidate.path === item.selectedCandidatePath)
                : undefined;
            const chosen = selected?.exact_hash ? selected : exactCandidates[0];
            return [{ item, path: chosen.path }];
        });
        for (const target of targets) {
            await applySourceFileReplacement(target.item, target.path, "replace");
        }
    }, [applySourceFileReplacement, sourceFileChangedDialog.changes]);

    const applyAllSelectedSourceFileMatches = useCallback(async () => {
        const targets = sourceFileChangedDialog.changes.flatMap((item) => {
            if (
                item.action === "ignored" ||
                item.action === "reloaded" ||
                item.action === "replaced" ||
                item.action === "processing"
            ) {
                return [];
            }
            const path = item.selectedCandidatePath;
            return path ? [{ item, path }] : [];
        });
        for (const target of targets) {
            await applySourceFileReplacement(target.item, target.path, "replace");
        }
    }, [applySourceFileReplacement, sourceFileChangedDialog.changes]);

    // 全部重置：先刷新最新状态，再恢复为窗口初始的“未处理”快照。
    // 重置时保留“搜索文件夹”已搜索到的候选列表与当前选择，方便用户继续操作。
    const resetAllSourceFileChanges = useCallback(async () => {
        if (sourceFileCheckBusyRef.current || sourceFileChangeHandlingRef.current) return;
        const searchResultsByClipId = new Map(
            sourceFileChangedDialog.changes.map((item) => [
                item.clip_id,
                {
                    candidates: item.candidates,
                    selectedCandidatePath: item.selectedCandidatePath,
                },
            ]),
        );
        const attachSearchResults = (item: SourceFileChangedItem): SourceFileChangedItem => {
            const saved = searchResultsByClipId.get(item.clip_id);
            if (!saved) return item;
            return { ...item, ...saved };
        };

        sourceFileCheckBusyRef.current = true;
        try {
            const result = (await webApi.checkSourceFilesChanged()) as
                | { changed?: SourceFileChange[] }
                | undefined;
            const rawChanges = normalizeSourceFileChanges(result?.changed ?? []);
            const baseItems = sourceFileInitialChangesRef.current.map((item) => ({ ...item }));
            const changes = mergeLatestSourceFileChanges(baseItems, rawChanges)
                .map((item) => resetSourceFileItemToPending(item))
                .map(attachSearchResults)
                .map(defaultSelectedCandidatePath);
            for (const item of changes) {
                ignoredSourcePathsRef.current.delete(item.source_path);
            }
            sourceFileInitialChangesRef.current = changes.map((item) => ({ ...item }));
            setSourceFileChangedDialog((prev) => ({ ...prev, open: true, changes }));
        } catch {
            // 读取失败时仍恢复到初始快照，避免列表停留在错误状态。
            const changes = sourceFileInitialChangesRef.current
                .map((item) => ({
                    ...resetSourceFileItemToPending(item),
                }))
                .map(attachSearchResults)
                .map(defaultSelectedCandidatePath);
            for (const item of changes) {
                ignoredSourcePathsRef.current.delete(item.source_path);
            }
            setSourceFileChangedDialog((prev) => ({ ...prev, open: true, changes }));
        } finally {
            sourceFileCheckBusyRef.current = false;
        }
    }, [sourceFileChangedDialog.changes]);

    // 单条重置：仅将该条目恢复为窗口打开时的初始状态。
    // 重置时保留该条目“搜索文件夹”搜索到的候选列表与当前选择。
    const resetSourceFileChangeItem = useCallback((clipId: string) => {
        setSourceFileChangedDialog((prev) => ({
            ...prev,
            changes: prev.changes.map((item) => {
                if (item.clip_id !== clipId) return item;
                const initial = sourceFileInitialChangesRef.current.find(
                    (candidate) => candidate.clip_id === clipId,
                );
                const restored = initial
                    ? resetSourceFileItemToPending({ ...initial })
                    : resetSourceFileItemToPending(item);
                const merged = defaultSelectedCandidatePath({
                    ...restored,
                    candidates: item.candidates,
                    selectedCandidatePath: item.selectedCandidatePath,
                } as SourceFileChangedItem);
                if (restored.source_path) {
                    ignoredSourcePathsRef.current.delete(restored.source_path);
                }
                return merged;
            }),
        }));
    }, []);

    const closeSourceFileChangedDialog = useCallback(() => {
        setSourceFileChangedDialog((prev) => ({ ...prev, open: false }));
    }, []);

    // 参数线上移/下移的长按重复：上一拍尚未完成（后端请求仍在途中）时
    // 跳过本拍，避免 50ms 节奏下 IPC 与历史检查点堆积。
    const paramShiftBusyRef = useRef(false);

    // 统一快捷键处理（通过 keybindings 模块管理，用户可自定义）
    const handleKeybindingAction = useCallback(
        (actionId: ActionId) => {
            if (!pluginAllowsAction(actionId, getActiveSurface())) return false;
            // ── 编辑操作统一路由 ──
            // clip.* 与 pianoRoll.* 的同义绑定（Ctrl+C/X/V）归一为同一编辑 op
            // 后定向派发到唯一执行者 —— 事件名即契约：hifi:editOp 只属于
            // 参数编辑器，hifi:timelineEditOp 只属于时间轴，消费者不再自行
            // 判断焦点。裁决规则见 focusRouting：复制/剪切按活动编辑表面
            // （focusSurface，由最后一次 pointerdown 落点驱动）；粘贴按剪贴板
            // 载荷类型（last-copy-wins，内容路由见 resolvePasteRoute），槽位
            // 为空/外来数据时才按活动表面兜底。
            const editOp = ACTION_TO_EDIT_OP[actionId];
            if (editOp) {
                if (editOp === "paste") {
                    // 内容路由（last-copy-wins）：探测剪贴板载荷类型后定向派发。
                    // 键盘事件已在 useKeybindings 同步消费，此处异步探测不影响
                    // 焦点语义；探测失败按外来源/空处理，回退表面裁决。
                    //
                    // 长按重复在这里布防，且**必须与 keydown 同步**（所以在探测
                    // 之前）：holdRepeat 只靠 keyup / blur 终止，若等异步探测返回
                    // 再布防，用户"快速点按"（keyup 早于探测返回）就会留下一个
                    // 永远等不到松键的计时器 —— 一次点按变成无限粘贴。合成派发方
                    // （菜单项、记事本暂存块）不经过这里，因此不会被误装长按。
                    // 重复的每一拍只在通道确认为时间轴时派发，保持"参数编辑器
                    // 粘贴不重复"的既有语义。
                    let channel: EditOpChannel | null = null;
                    const pasteKb = selectMergedKeybindings(store.getState())["clip.paste"];
                    if (pasteKb) {
                        beginHoldRepeat(pasteKb, () => {
                            if (channel === "hifi:timelineEditOp" && pluginAllowsEditChannel(channel, "paste")) {
                                window.dispatchEvent(
                                    new CustomEvent(channel, { detail: { op: "paste" } }),
                                );
                            }
                        });
                    }
                    void (async () => {
                        let kind: string | null = null;
                        try {
                            kind = (await webApi.clipboardKind()).kind ?? null;
                        } catch {
                            // 探测失败不阻塞粘贴。
                        }
                        channel = resolvePasteRoute(kind, getActiveSurface());
                        if (pluginAllowsEditChannel(channel, "paste")) {
                            window.dispatchEvent(
                                new CustomEvent(channel, { detail: { op: "paste" } }),
                            );
                        }
                    })();
                    return;
                }
                if (editOp === "copy" || editOp === "cut") {
                    // 选择路由（复制时内容尚不存在，看"当前选中了什么"）：仅
                    // 参数线选区 → 参数复制；仅 Clip 选区 → Clip 复制；两者并存
                    // 按 selectionContext（最近被触碰的选区上下文，换轨计入参数
                    // 侧）仲裁；皆空 → 无操作。
                    const session = store.getState().session;
                    const channel = resolveCopyCutRoute({
                        surface: getActiveSurface(),
                        clipSelectionActive:
                            session.multiSelectedClipIds.length > 0 || !!session.selectedClipId,
                        paramSelectionActive: session.paramSelectionActive,
                        selectionContext: session.selectionContext,
                    });
                    if (pluginAllowsEditChannel(channel, editOp)) {
                        window.dispatchEvent(new CustomEvent(channel, { detail: { op: editOp } }));
                    }
                    return;
                }
                if (
                    editOp === "addClipsToParamSelection" ||
                    editOp === "removeClipsFromParamSelection"
                ) {
                    // 音频块范围 → 参数编辑器选区：消费端**只有**参数编辑器，
                    // 因此不按活动表面裁决 —— 焦点在时间轴上时按快捷键同样生效。
                    // 选中了哪些音频块由消费端从 session 读取（选区是权威来源），
                    // 与右键菜单传 clipIds 的路径共用同一实现。
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", { detail: { op: editOp } }),
                    );
                    return;
                }
                const channel = resolveEditOpRoute(
                    getActiveSurface(),
                    editOp,
                    store.getState().session.toolMode,
                );
                if (pluginAllowsEditChannel(channel, editOp)) {
                    window.dispatchEvent(new CustomEvent(channel, { detail: { op: editOp } }));
                }
                return;
            }
            switch (actionId) {
                case "playback.toggle":
                case "playback.stop": {
                    // 以 store 实时状态判定播放态：runtimeRef 在 effect 提交后才
                    // 刷新，快速连续按键（播放/停止连打）时会基于过期值对同一
                    // 状态双重派发（连按两次 Space 派发两次 play/两次 stop），
                    // 第二次会把刚建立的播放重新拉回起点（光标小跳）。
                    //
                    // 语义（与 DAW 惯例严格对齐，裁决表见 transportShortcuts）：
                    // - toggle（Space）= 播放 / **暂停**：播放中暂停，光标留在当前
                    //   播放位置；空闲时从光标起播。
                    // - stop（Enter）= 播放 / **停止**：播放中停止，光标回到本次
                    //   起播位置（restoreAnchor）；空闲时同样从光标起播。
                    //
                    // 空闲时的起播分支必须保留：`playOriginal` 已幂等（播放中调用
                    // 为完全 no-op，见 transportThunks），所以这里不会再产生 77553e61
                    // 要修的那种"重复触发把传输层拽回起点"——但缺了它，默认 Enter
                    // 在空闲时什么都不做，与标签「播放 / 停止」和手册相矛盾。
                    const isPlayingNow = Boolean(store.getState().session.runtime.isPlaying);
                    const command = resolveTransportShortcutCommand(actionId, isPlayingNow);
                    if (command === "play") {
                        void dispatch(playOriginal());
                    } else if (command === "stop") {
                        void dispatch(stopAudioPlayback({ restoreAnchor: true }));
                    } else {
                        void dispatch(stopAudioPlayback());
                    }
                    break;
                }
                case "playback.metronome":
                    void dispatch(
                        updateMetronome({
                            metronomeEnabled: !store.getState().session.metronomeEnabled,
                        }),
                    );
                    break;
                case "playback.focusCursor":
                    window.dispatchEvent(new CustomEvent("hifi:focusCursor"));
                    break;
                case "playback.seekLeft":
                    window.dispatchEvent(
                        new CustomEvent("hifi:nudgePlayhead", {
                            detail: { direction: -1 },
                        }),
                    );
                    break;
                case "playback.seekRight":
                    window.dispatchEvent(
                        new CustomEvent("hifi:nudgePlayhead", {
                            detail: { direction: 1 },
                        }),
                    );
                    break;
                case "timeline.zoomIn":
                    window.dispatchEvent(
                        new CustomEvent("hifi:zoomTimelineFocus", {
                            detail: { factor: 1.1 },
                        }),
                    );
                    break;
                case "timeline.zoomOut":
                    window.dispatchEvent(
                        new CustomEvent("hifi:zoomTimelineFocus", {
                            detail: { factor: 0.9 },
                        }),
                    );
                    break;
                case "edit.undo": {
                    // 空栈时静默失败：后端回 ok=false，前端不套用任何快照 ——
                    // 界面零刷新零变更、无任何提示（见 sessionSlice 的
                    // undoRemote.fulfilled 空栈分支）。这里仍照常派发，绝不
                    // 用前端镜像把撤销「挡」在门外：镜像万一滞后，被吞掉的
                    // 是用户真实的撤销意图，而多一次空往返没有代价。
                    // 长按 Ctrl+Z = 连续撤销（每拍撤销一步；后端按消息队列
                    // 串行处理，无需忙守卫）；深度镜像（history_state 事件）
                    // 仅用于撤到空栈后停止长按重复。
                    const fire = () => {
                        const hasUndoableStep = store.getState().session.historyUndoDepth > 0;
                        void dispatch(undoRemote({ parametersOnly: isPluginMode() && getActiveSurface() === "pianoRoll" }));
                        return hasUndoableStep;
                    };
                    if (fire()) {
                        beginHoldRepeat(
                            selectMergedKeybindings(store.getState())["edit.undo"],
                            fire,
                        );
                    }
                    break;
                }
                case "edit.redo": {
                    // 空栈时同样静默失败；长按 Ctrl+Y = 连续重做（同上）。
                    const fire = () => {
                        const hasRedoableStep = store.getState().session.historyRedoDepth > 0;
                        void dispatch(redoRemote({ parametersOnly: isPluginMode() && getActiveSurface() === "pianoRoll" }));
                        return hasRedoableStep;
                    };
                    if (fire()) {
                        beginHoldRepeat(
                            selectMergedKeybindings(store.getState())["edit.redo"],
                            fire,
                        );
                    }
                    break;
                }
                // edit.selectAll / edit.deselect 由顶部「编辑操作统一路由」按
                // 活动编辑表面定向派发（select 工具下的参数编辑器全选走
                // hifi:editOp，其余走 hifi:timelineEditOp），此处不再重复派发。
                // 布局：全部走 dockApi，与「布局」菜单共用同一份行为实现。
                case "layout.toggleFloat":
                    toggleFloatActive(dispatch, store.getState);
                    break;
                case "layout.focusNext":
                    cycleFocus(dispatch, store.getState, 1);
                    break;
                case "layout.focusPrev":
                    cycleFocus(dispatch, store.getState, -1);
                    break;
                case "layout.maximize":
                    maximizeActive(dispatch);
                    break;
                case "layout.newPanel":
                    addEmptyPanel(dispatch);
                    break;
                case "layout.dissolvePanel":
                    dissolvePanelCommand(dispatch, store.getState);
                    break;
                case "project.new":
                    handleNewProject();
                    break;
                case "project.open":
                    handleOpenProject();
                    break;
                case "project.save":
                    void dispatch(saveProjectRemote());
                    break;
                case "project.saveAs":
                    void dispatch(saveProjectAsRemote());
                    break;
                case "project.export":
                    window.dispatchEvent(
                        new CustomEvent("hifi:openEditDialog", {
                            detail: { dialog: "exportAudio" },
                        }),
                    );
                    break;
                case "project.importMedia":
                    // 多文件/多音轨选择等交互由 MenuBar 的流程处理（与菜单项一致）。
                    window.dispatchEvent(new CustomEvent("hifi:importMediaFromMenu"));
                    break;
                case "project.importMidi":
                    handleImportMidiFromMenu();
                    break;
                case "project.importHifishifter":
                    void handleImportProject();
                    break;
                case "project.importReaper":
                    void dispatch(openReaperFromDialog());
                    break;
                case "project.importVocalShifter":
                    void dispatch(openVocalShifterFromDialog());
                    break;
                case "mode.toggle": {
                    const cur = runtimeRef.current.toolMode;
                    if (cur === "select") {
                        void dispatch(setToolModePersistent(runtimeRef.current.drawToolMode));
                    } else {
                        void dispatch(setToolModePersistent("select"));
                    }
                    break;
                }
                case "mode.selectTool":
                    void dispatch(setToolModePersistent("select"));
                    break;
                case "mode.drawTool":
                    void dispatch(setToolModePersistent("draw"));
                    break;
                case "mode.lineTool":
                    void dispatch(setToolModePersistent("line"));
                    break;
                case "mode.vibratoTool":
                    void dispatch(setToolModePersistent("vibrato"));
                    break;
                case "quickSearch.open":
                    setQuickSearchOpen(true);
                    break;
                case "track.add": {
                    // 新建轨道继承当前选中轨道的轨道层级（同 parentId），
                    // 并紧跟在选中轨道下方插入（同级列表紧后一位）。
                    // 长按 Ctrl+T = 连续添加（每拍读取最新选区，轨道依次向下排）。
                    const fire = () => {
                        const ss = store.getState().session;
                        const placement = computeInsertBelowPlacement(
                            ss.tracks,
                            ss.selectedTrackId,
                        );
                        void dispatch(
                            addTrackRemote({
                                parentTrackId: placement.parentTrackId,
                                index: placement.index,
                            }),
                        );
                        return true;
                    };
                    fire();
                    beginHoldRepeat(selectMergedKeybindings(store.getState())["track.add"], fire);
                    break;
                }
                case "track.clone": {
                    // 长按 Ctrl+D = 连续克隆（每拍克隆当前选中轨道；克隆后后端
                    // 会选中新克隆，因此连续克隆依次向下堆叠）。
                    const fire = () => {
                        const ss = store.getState().session;
                        const selectedId = ss.selectedTrackId;
                        if (!selectedId) return false;
                        void dispatch(duplicateTrackRemote(selectedId));
                        return true;
                    };
                    if (fire()) {
                        beginHoldRepeat(
                            selectMergedKeybindings(store.getState())["track.clone"],
                            fire,
                        );
                    }
                    break;
                }
                case "track.delete": {
                    const ss = store.getState().session;
                    const selectedId = ss.selectedTrackId;
                    if (!selectedId) break;
                    const selected = ss.tracks.find((t) => t.id === selectedId);
                    // 与菜单一致：只剩下最后一个根轨道时禁止删除根轨道。
                    if (!selected) break;
                    if (!selected.parentId && ss.tracks.filter((t) => !t.parentId).length <= 1) {
                        break;
                    }
                    void dispatch(removeTrackRemote(selectedId));
                    break;
                }
                case "track.selectUp":
                    window.dispatchEvent(
                        new CustomEvent("hifi:selectAdjacentTrack", {
                            detail: { direction: -1 },
                        }),
                    );
                    break;
                case "track.selectDown":
                    window.dispatchEvent(
                        new CustomEvent("hifi:selectAdjacentTrack", {
                            detail: { direction: 1 },
                        }),
                    );
                    break;
                case "track.toggleMute": {
                    // 默认无键位：由用户在快捷键设置中自行绑定。作用于当前
                    // 选中轨道（与轨道头 M 按钮同一后端命令）。
                    const ss = store.getState().session;
                    const trackId = ss.selectedTrackId;
                    const track = trackId ? ss.tracks.find((t) => t.id === trackId) : null;
                    if (!track) break;
                    void dispatch(setTrackStateRemote({ trackId: track.id, muted: !track.muted }));
                    break;
                }
                case "track.toggleSolo": {
                    const ss = store.getState().session;
                    const trackId = ss.selectedTrackId;
                    const track = trackId ? ss.tracks.find((t) => t.id === trackId) : null;
                    if (!track) break;
                    void dispatch(setTrackStateRemote({ trackId: track.id, solo: !track.solo }));
                    break;
                }
                case "pianoRoll.cycleDragDirection": {
                    // 循环切换当前活动工具的拖动方向（与工具栏方向按钮同源）。
                    // 左键拖拽参数线期间按下同一键时，参数编辑器内的本地监听会
                    // 同步切换本次拖拽的方向 —— 触控板用户的「右键切换」替代。
                    const ss = store.getState().session;
                    // 直线与颤音共用同一份拖动方向（同一个"起点 → 终点"手势），
                    // 因此两者都归到 `"vibrato"` 这一路。
                    const tool =
                        ss.toolMode === "select"
                            ? ("select" as const)
                            : ss.drawToolMode === "draw"
                              ? ("draw" as const)
                              : ("vibrato" as const);
                    dispatch(cycleDragDirection(tool));
                    void dispatch(persistUiSettings());
                    break;
                }
                case "pianoRoll.shiftParamUp":
                case "pianoRoll.shiftParamDown":
                case "pianoRoll.shiftParamUpLarge":
                case "pianoRoll.shiftParamDownLarge":
                case "pianoRoll.shiftParamUpSmall":
                case "pianoRoll.shiftParamDownSmall": {
                    // 三档幅度：默认 / 大幅（Shift，音高 = 一个八度等）/
                    // 微调（Ctrl，音高 = 1 音分等），步长见 getParamShiftStep。
                    // 方向/幅度由 resolveParamShiftIntent 统一解析（变体 id
                    // 以 Large/Small 结尾，方向判定不能用 endsWith）。
                    const intent = resolveParamShiftIntent(actionId);
                    if (!intent || intent.selectionOp) break;
                    const isUp = intent.isUp;
                    const magnitude = intent.magnitude;
                    const busyRef = paramShiftBusyRef;
                    // 长按 "=" / "-" / Shift+=" / Ctrl+=" = 连续上移/下移
                    // 参数线。异步链路进行中时跳过本拍，避免 50ms 节奏下
                    // IPC 与历史检查点堆积。
                    const fire = (): boolean => {
                        if (busyRef.current) return false;
                        const ss = store.getState().session;
                        const rootTrkId = resolveRootTrackId(ss.tracks, ss.selectedTrackId);
                        if (!rootTrkId) return false;
                        const editP = ss.editParam;
                        const rootTrk = ss.tracks.find((tr) => tr.id === rootTrkId);
                        // pitch 参数需要 pitch 分析可用才能操作
                        if (editP === "pitch") {
                            if (!rootTrk?.composeEnabled || rootTrk.pitchAnalysisAlgo === "none") {
                                return false;
                            }
                        }
                        const selClipId = ss.selectedClipId;
                        // 优先使用多选 clip 列表，否则 fallback 到单选
                        const multiIds = ss.multiSelectedClipIds;
                        const clipIds =
                            multiIds.length >= 1 ? multiIds : selClipId ? [selClipId] : [];
                        if (clipIds.length === 0) return false;
                        const selClips = ss.clips.filter((c) => clipIds.includes(c.id));
                        if (selClips.length === 0) return false;
                        const minSec = Math.min(...selClips.map((c) => c.startSec));
                        const maxSec = Math.max(...selClips.map((c) => c.startSec + c.lengthSec));
                        // 默认 framePeriodMs = 5
                        const fp = 5;
                        const startFrame = Math.max(0, Math.floor((minSec * 1000) / fp));
                        const frameCount = Math.max(
                            1,
                            Math.min(200_000, Math.ceil(((maxSec - minSec) * 1000) / fp)),
                        );
                        busyRef.current = true;
                        void (async () => {
                            try {
                                let descriptor: ProcessorParamDescriptor | undefined;
                                if (editP !== "pitch" && rootTrk?.pitchAnalysisAlgo) {
                                    const algo = rootTrk.pitchAnalysisAlgo;
                                    let descriptors = processorParamCacheRef.current.get(algo);
                                    if (!descriptors) {
                                        try {
                                            descriptors = await paramsApi.getProcessorParams(algo);
                                            processorParamCacheRef.current.set(algo, descriptors);
                                        } catch {
                                            descriptors = undefined;
                                        }
                                    }
                                    descriptor = descriptors?.find((param) => param.id === editP);
                                }
                                const step = getParamShiftStep(editP, descriptor, magnitude);
                                const delta = isUp ? step : -step;
                                const clampNum = (v: number, minV: number, maxV: number) =>
                                    Math.min(maxV, Math.max(minV, v));
                                const smoothness = clampNum(
                                    Number(ss.edgeSmoothnessPercent) || 0,
                                    0,
                                    100,
                                );
                                const maxTransitionFrames = Math.floor(frameCount / 2);
                                const transitionFrames =
                                    smoothness > 0 && maxTransitionFrames > 0
                                        ? Math.round((smoothness / 100) * maxTransitionFrames)
                                        : 0;
                                const halfSpan = transitionFrames > 0 ? transitionFrames / 2 : 0;
                                const extend = Math.max(0, Math.ceil(halfSpan));
                                const extStart = Math.max(0, startFrame - extend);
                                const extCount =
                                    frameCount + Math.max(0, startFrame - extStart) + extend;
                                const selOffset = startFrame - extStart;

                                const extRes = await paramsApi.getParamFrames(
                                    rootTrkId,
                                    editP,
                                    extStart,
                                    extCount,
                                    1,
                                );
                                if (!extRes?.ok) return;
                                const extPayload = extRes as ParamFramesPayload;
                                const beforeDense = (extPayload.edit ?? []).map(
                                    (v) => Number(v) || 0,
                                );
                                if (beforeDense.length === 0) return;

                                const selEnd = Math.min(
                                    beforeDense.length - 1,
                                    selOffset + frameCount - 1,
                                );
                                if (
                                    selOffset < 0 ||
                                    selOffset >= beforeDense.length ||
                                    selEnd < selOffset
                                ) {
                                    return;
                                }
                                const actualSelLen = selEnd - selOffset + 1;
                                const editedDense = beforeDense.slice();
                                for (let i = 0; i < actualSelLen; i += 1) {
                                    const orig = beforeDense[selOffset + i] ?? 0;
                                    editedDense[selOffset + i] = orig + delta;
                                }

                                if (smoothness > 0 && transitionFrames > 0) {
                                    const calcMean = (arr: number[]) => {
                                        let sum = 0;
                                        let count = 0;
                                        for (let i = 0; i < actualSelLen; i += 1) {
                                            const v = Number(arr[selOffset + i] ?? 0);
                                            if (editP === "pitch" && v === 0) continue;
                                            sum += v;
                                            count += 1;
                                        }
                                        return { sum, count };
                                    };

                                    const beforeMean = calcMean(beforeDense);
                                    const afterMean = calcMean(editedDense);
                                    const meanDelta =
                                        beforeMean.count > 0 && afterMean.count > 0
                                            ? Math.abs(
                                                  afterMean.sum / afterMean.count -
                                                      beforeMean.sum / beforeMean.count,
                                              )
                                            : 0;

                                    let boundaryDelta = 0;
                                    let boundaryCount = 0;
                                    if (selOffset > 0) {
                                        boundaryDelta += Math.abs(
                                            Number(beforeDense[selOffset] ?? 0) -
                                                Number(beforeDense[selOffset - 1] ?? 0),
                                        );
                                        boundaryCount += 1;
                                    }
                                    if (selEnd < beforeDense.length - 1) {
                                        boundaryDelta += Math.abs(
                                            Number(beforeDense[selEnd] ?? 0) -
                                                Number(beforeDense[selEnd + 1] ?? 0),
                                        );
                                        boundaryCount += 1;
                                    }
                                    const boundaryMean =
                                        boundaryCount > 0 ? boundaryDelta / boundaryCount : 0;
                                    const changeFactor = clampNum(
                                        meanDelta / (meanDelta + boundaryMean + 1e-6),
                                        0,
                                        1,
                                    );

                                    if (changeFactor > 0) {
                                        const snapshot = editedDense.slice();
                                        const span = Math.max(1e-9, 2 * halfSpan);
                                        if (selOffset > 0) {
                                            const left = Math.max(
                                                0,
                                                Math.floor(selOffset - halfSpan),
                                            );
                                            const right = Math.min(
                                                editedDense.length - 1,
                                                Math.ceil(selOffset + halfSpan),
                                            );
                                            for (let idx = left; idx <= right; idx += 1) {
                                                const t = clampNum(
                                                    (idx - (selOffset - halfSpan)) / span,
                                                    0,
                                                    1,
                                                );
                                                const outsideIdx = Math.min(selOffset - 1, idx);
                                                const insideIdx = Math.max(selOffset, idx);
                                                const outsideVal =
                                                    snapshot[outsideIdx] ?? editedDense[idx];
                                                const insideVal =
                                                    snapshot[insideIdx] ?? editedDense[idx];
                                                const smoothed =
                                                    outsideVal + (insideVal - outsideVal) * t;
                                                editedDense[idx] =
                                                    snapshot[idx] +
                                                    (smoothed - snapshot[idx]) * changeFactor;
                                            }
                                        }
                                        if (selEnd < editedDense.length - 1) {
                                            const left = Math.max(0, Math.floor(selEnd - halfSpan));
                                            const right = Math.min(
                                                editedDense.length - 1,
                                                Math.ceil(selEnd + halfSpan),
                                            );
                                            for (let idx = left; idx <= right; idx += 1) {
                                                const t = clampNum(
                                                    (idx - (selEnd - halfSpan)) / span,
                                                    0,
                                                    1,
                                                );
                                                const insideIdx = Math.min(selEnd, idx);
                                                const outsideIdx = Math.max(selEnd + 1, idx);
                                                const insideVal =
                                                    snapshot[insideIdx] ?? editedDense[idx];
                                                const outsideVal =
                                                    snapshot[outsideIdx] ?? editedDense[idx];
                                                const smoothed =
                                                    insideVal + (outsideVal - insideVal) * t;
                                                editedDense[idx] =
                                                    snapshot[idx] +
                                                    (smoothed - snapshot[idx]) * changeFactor;
                                            }
                                        }
                                    }
                                }

                                await paramsApi.setParamFrames(
                                    rootTrkId,
                                    editP,
                                    extStart,
                                    editedDense,
                                    true,
                                );
                                // 通知 PianoRoll 刷新曲线
                                dispatch(checkpointHistory());
                            } finally {
                                busyRef.current = false;
                            }
                        })();
                        return true;
                    };
                    if (fire()) {
                        beginHoldRepeat(selectMergedKeybindings(store.getState())[actionId], fire);
                    }
                    break;
                }
                case "pianoRoll.shiftParamUpSelection":
                case "pianoRoll.shiftParamDownSelection":
                case "pianoRoll.shiftParamUpSelectionLarge":
                case "pianoRoll.shiftParamDownSelectionLarge":
                case "pianoRoll.shiftParamUpSelectionSmall":
                case "pianoRoll.shiftParamDownSelectionSmall": {
                    // 选择范围平移同样分三档幅度；op 名沿用消费端（PianoRollPanel）
                    // 的既有契约，幅度经事件 detail 透传（解析同上）。
                    const intent = resolveParamShiftIntent(actionId);
                    if (!intent || !intent.selectionOp) break;
                    const op = intent.selectionOp;
                    const magnitude = intent.magnitude;
                    // 长按 "]" / "["（及 Shift/Ctrl 变体）= 连续上移/下移选区
                    // 范围参数线。执行体在 PianoRollPanel，在途标记经
                    // selectionEditInFlight 共享 —— 上一拍未完成时跳过本拍。
                    // 首拍与音频块范围平移同构的守卫：无选区 / 无根轨道 /
                    // pitch 不可用时不布防长按（否则空转重复直至松键）。
                    const fire = (): boolean => {
                        if (isSelectionParamEditInFlight()) return false;
                        const ss = store.getState().session;
                        if (!ss.paramSelectionActive) return false;
                        const rootTrkId = resolveRootTrackId(ss.tracks, ss.selectedTrackId);
                        if (!rootTrkId) return false;
                        if (ss.editParam === "pitch") {
                            const rootTrk = ss.tracks.find((tr) => tr.id === rootTrkId);
                            if (!rootTrk?.composeEnabled || rootTrk.pitchAnalysisAlgo === "none") {
                                return false;
                            }
                        }
                        window.dispatchEvent(
                            new CustomEvent("hifi:editOp", {
                                detail: { op, magnitude },
                            }),
                        );
                        return true;
                    };
                    if (fire()) {
                        beginHoldRepeat(selectMergedKeybindings(store.getState())[actionId], fire);
                    }
                    break;
                }
                case "edit.pasteVocalShifter":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "pasteVocalShifter" },
                        }),
                    );
                    break;
                case "recording.toggle": {
                    const rec = store.getState().recording;
                    if (rec.active) {
                        void dispatch(stopRecordingFlow());
                    } else if (rec.countdownRemaining > 0) {
                        void dispatch(cancelRecordingCountdown());
                    } else {
                        void dispatch(startRecordingFlow());
                    }
                    break;
                }
                case "edit.pasteTracks": {
                    // 长按 = 连续作为新轨道组粘贴。接收端 pasteClipsAtPlayhead
                    // 自带粘贴链守卫（busy/queued），重复事件排队处理、不会并发。
                    // pasteTracks 是时间轴专有操作，固定路由到时间轴通道。
                    const op = "pasteTracks" as const;
                    const fire = () => {
                        window.dispatchEvent(
                            new CustomEvent("hifi:timelineEditOp", { detail: { op } }),
                        );
                        return true;
                    };
                    fire();
                    beginHoldRepeat(
                        selectMergedKeybindings(store.getState())["edit.pasteTracks"],
                        fire,
                    );
                    break;
                }
                // 注：edit.pasteVocalShifter（文件型剪贴板）未启用长按重复 ——
                // 其接收端（钢琴卷帘）无粘贴链守卫，50ms 节奏下重复导入同一
                // 剪贴板文件可能并发；且该操作边际收益低。
                // 注：clip.* / pianoRoll.copy|cut|paste / edit.selectAll 等
                // 编辑操作已在函数顶部按活动编辑表面统一路由（focusRouting）。
                default:
                    break;
            }
        },
        [
            dispatch,
            handleNewProject,
            handleOpenProject,
            handleImportMidiFromMenu,
            handleImportProject,
        ],
    );

    useKeybindings(handleKeybindingAction);

    // 活动编辑表面跟踪：document 捕获阶段解析 pointerdown / focusin 落点，
    // 为编辑快捷键（复制/剪切/粘贴等）的归属裁决提供单一事实源 —— 捕获
    // 阶段位于传播链最前端，子元素的 preventDefault / stopPropagation 均
    // 无法逃过解析（时间轴正是靠 preventDefault 自管焦点的）。
    useEffect(() => installFocusSurfaceTracking(), []);

    useEffect(() => {
        // ★ 常驻轮询（自愈）：播放中 ~30Hz；未播放时低频（400ms）作看门狗。
        //
        // 旧实现在未播放时直接 return（不建 interval），而前端 `isPlaying` 只是
        // 引擎状态的**镜像**：任何一次误翻转（竞态采样的迟到响应、刷新、外部
        // 命令）都会让轮询彻底停摆 —— 前端**再也无法自愈**，于是出现"引擎确实
        // 在播放、音频在响，但播放光标永久冻结"，并且按空格被误判为"未播放"
        // 而重新播放（把传输层 seek 回播放头，音频从头开始）。低频看门狗使镜像
        // 在 ≤~400ms 内自我纠正：首个纠正采样即翻转 isPlaying、恢复 30Hz 与
        // 光标推进。
        const intervalMs = runtimeIsPlaying ? 33 : 400;
        const id = window.setInterval(() => {
            // 阻塞式前台预渲染（target="original"）阶段后端还未真正进入 playing，
            // 若此时同步会把前端"准备播放"状态误判为停止，导致 stop 锚点丢失。
            // 后台预渲染（target="background"）是独立线程，播放应正常同步。
            // 读取按 target 隔离的 blocking 镜像：两类渲染并发发射事件，
            // 后台渲染的完成事件不得提前关闭本守卫。
            if (rendering.blocking) return;
            if (playbackSyncInFlightRef.current) return;
            playbackSyncInFlightRef.current = true;
            // 派发时刻的传输纪元与时钟读数：响应迟到且期间发生过
            // 播放/停止/seek/isPlaying 翻转时，reducer 把该响应按乱序丢弃；
            // 时钟读数用于把采样位置外推到处理时刻（消除 IPC 时延滞后/抖动
            // 造成的视觉跳变）。
            const dispatchedAtMs = performance.now();
            const p = dispatch(
                syncPlaybackState({
                    epoch: store.getState().session._transportEpoch,
                    dispatchedAtMs,
                }),
            ) as unknown as Promise<unknown>;
            p.finally(() => {
                playbackSyncInFlightRef.current = false;
            });
        }, intervalMs);
        return () => window.clearInterval(id);
    }, [dispatch, runtimeIsPlaying, rendering.blocking]);

    useEffect(() => {
        if (!recordingActive || !recordingSettings.autoStopAtSelectionEnd) return;
        const selectedIds = new Set<string>();
        if (selectedClipId) selectedIds.add(selectedClipId);
        for (const id of multiSelectedClipIds) selectedIds.add(id);
        const selected = sessionClips.filter((clip) => selectedIds.has(clip.id));
        if (selected.length === 0) return;
        const endSec = Math.max(
            ...selected.map((clip) => Number(clip.startSec) + Number(clip.lengthSec)),
        );
        if (!Number.isFinite(endSec)) return;
        if (recordingStartSec == null || Number(recordingStartSec) > endSec + 0.05) return;

        const id = window.setInterval(() => {
            // 不能把 playbackPositionSec 放进闭包/依赖：它随播放轮询 ~33ms 变化，
            // 会让本 effect 不停地销毁重建这个 100ms interval，回调永远等不到触发。
            // 经 store 同步读取最新播放位置。
            const positionSec = Number(store.getState().session.runtime.playbackPositionSec ?? 0);
            if (positionSec >= endSec - 0.05) {
                void dispatch(stopRecordingFlow());
            }
        }, 100);
        return () => window.clearInterval(id);
    }, [
        dispatch,
        multiSelectedClipIds,
        recordingActive,
        recordingSettings.autoStopAtSelectionEnd,
        recordingStartSec,
        selectedClipId,
        sessionClips,
    ]);

    const sourceFileSearchMatchTotal = sourceFileChangedDialog.changes.reduce(
        (total, item) =>
            item.action === "pending" || item.action === "failed"
                ? total + (item.candidates?.length ?? 0)
                : total,
        0,
    );
    const sourceFileSearchExactTotal = sourceFileChangedDialog.changes.reduce(
        (total, item) =>
            item.action === "pending" || item.action === "failed"
                ? total + (item.candidates?.filter((candidate) => candidate.exact_hash).length ?? 0)
                : total,
        0,
    );
    const sourceFileExactApplyTotal = sourceFileChangedDialog.changes.reduce((total, item) => {
        if (item.action !== "pending" && item.action !== "failed") {
            return total;
        }
        return total + (item.candidates?.some((candidate) => candidate.exact_hash) ? 1 : 0);
    }, 0);
    const sourceFileSelectedApplyTotal = sourceFileChangedDialog.changes.reduce((total, item) => {
        if (
            (item.action !== "pending" && item.action !== "failed") ||
            !item.selectedCandidatePath
        ) {
            return total;
        }
        return total + 1;
    }, 0);
    const sourceFileReloadAllTotal = sourceFileChangedDialog.changes.reduce((total, item) => {
        if (
            item.change === "modified" &&
            !item.reloadAttempted &&
            (item.action === "pending" || item.action === "failed")
        ) {
            return total + 1;
        }
        return total;
    }, 0);
    const sourceFileAnyProcessing = sourceFileChangedDialog.changes.some(
        (item) => item.action === "processing",
    );

    // ── 布局持久化 ──────────────────────────────────────────────────
    //
    // 去抖后写回：拖分隔条/移浮窗会在松手瞬间各触发一次布局变化，而后端
    // `save_ui_settings` 是"读-改-写整个配置文件 + 原子替换 + 备份"（约 8 次
    // 文件操作），不宜按次调用。
    //
    // 【闸门】必须等 `hydrated` 为真：切片初始状态是出厂布局，若在读到磁盘
    // 内容之前就写回，用户的布局会被默认值覆盖 —— 也就是"打开应用发现界面
    // 被重置"这类最恼人的故障。
    //
    // 【最大化时也必须跳过】最大化把 `state.layout` 的某个根临时换成单组树，
    // 而 `maximized` 本身刻意不持久化（重启后回到用户排好的布局）。若此刻仍
    // 落盘，写下的就是那棵临时树；用户没还原就退出，原排布被永久覆盖且无从
    // 恢复。`dockMaximized` 进入依赖：还原时它变回 false，effect 重新运行，
    // 恢复后的真实布局随即被保存，去抖不会永久失效。
    useEffect(() => {
        if (!dockHydrated || dockMaximized) return;
        const timer = window.setTimeout(() => {
            void dispatch(persistDockSettings());
        }, dockSettings.saveDebounceMs);
        return () => window.clearTimeout(timer);
    }, [dockLayout, dockSettings, dockHydrated, dockMaximized, dispatch]);

    // ── 面板渲染函数登记 ─────────────────────────────────────────────
    //
    // 停靠系统只负责"把面板摆在哪儿"，面板需要什么 props 仍由 App 提供 ——
    // 这些状态（MIDI 导入对话框、文件浏览器的关闭回调等）的所有者自始至终
    // 是 App，搬进注册表只会变成第二份拷贝。写在渲染期是安全的：写入幂等，
    // 且读取发生在同一趟渲染里更靠后的子组件（见 `panelRenderer` 注释）。
    setPanelRenderer(PANEL_TIMELINE, () => (
        <TimelinePanel
            midiClipDialogOpen={midiClipDialogOpen}
            midiClipPath={midiClipPath}
            midiClipStartSec={midiClipStartSec}
            midiClipTrackId={midiClipTrackId}
            midiClipClipboardGuid={midiClipClipboardGuid}
            fillGaps={fillGaps}
            multiTrackMerge={multiTrackMerge}
            importBpmAsProject={importBpmAsProject}
            noteBpmMode={noteBpmMode}
            specifiedBpm={specifiedBpm}
            importPosition={importPosition}
            closeLeadingGap={closeLeadingGap}
            onMidiClipDialogOpenChange={setMidiClipDialogOpen}
            onMidiClipPathChange={setMidiClipPath}
            onMidiClipStartSecChange={setMidiClipStartSec}
            onMidiClipTrackIdChange={setMidiClipTrackId}
            onFillGapsChange={handleFillGapsChange}
            onMultiTrackMergeChange={handleMultiTrackMergeChange}
            onImportBpmAsProjectChange={handleImportBpmAsProjectChange}
            onNoteBpmModeChange={handleNoteBpmModeChange}
            onSpecifiedBpmChange={handleSpecifiedBpmChange}
            onImportPositionChange={handleImportPositionChange}
            onCloseLeadingGapChange={handleCloseLeadingGapChange}
            importTempoMapEnabled={importTempoMapEnabled}
            onImportTempoMapEnabledChange={handleImportTempoMapEnabledChange}
            importTempoMapTempo={importTempoMapTempo}
            onImportTempoMapTempoChange={handleImportTempoMapTempoChange}
            importTempoMapTimeSignature={importTempoMapTimeSignature}
            onImportTempoMapTimeSignatureChange={handleImportTempoMapTimeSignatureChange}
            importTempoMapKeySignature={importTempoMapKeySignature}
            onImportTempoMapKeySignatureChange={handleImportTempoMapKeySignatureChange}
            midiDialogSource={midiDialogSource}
            onMidiDialogSourceChange={setMidiDialogSource}
            importTargetMenu={midiImportTargetMenu}
            onImportTargetMenuChange={handleImportTargetMenuChange}
            importTargetDragDrop={midiImportTargetDragDrop}
            onImportTargetDragDropChange={handleImportTargetDragDropChange}
        />
    ));
    // 把窗体身份传进去：参数编辑器要判断"我是否与时间轴上下堆叠"，以决定同步
    // 偏移怎么算（多实例时每个窗体各判各的）。
    setPanelRenderer(PANEL_PARAM_EDITOR, (form) => <PianoRollPanel dockFormId={form.id} />);
    setPanelRenderer(PANEL_FILE_BROWSER, () => <FileBrowserPanel />);
    setPanelRenderer(PANEL_UNDO_HISTORY, () => <UndoHistoryPanel />);
    // 外观设置：曾经是独立 OS 窗口（`appearance.html` + 独立 React 根），现在复用
    // 停靠机制 —— 居中浮出、不可停靠、不进「窗口」菜单（见注册表声明）。
    setPanelRenderer(PANEL_APPEARANCE, (form) => <AppearanceSettingsPanel formId={form.id} />);
    // ARA 宿主会话（仅独立 App）：默认关闭的浮出面板，入口在「视图」菜单。
    // 曾经是一整条常驻横条，一直占着工作区高度（见注册表声明里的取舍）。
    setPanelRenderer(PANEL_ARA_HOST, () => (
        <AraHostPanel
            dirty={projectDirty}
            onTimelineChanged={async () => {
                await dispatch(fetchTimeline()).unwrap();
            }}
        />
    ));
    // 记事本走 Suspense：TipTap 那几百 KB 只在真正打开时才拉取。
    setPanelRenderer(PANEL_NOTEBOOK, () => (
        <Suspense fallback={null}>
            <NotebookErrorBoundary>
                <NotebookPanel />
            </NotebookErrorBoundary>
        </Suspense>
    ));

    return (
        <Flex
            direction="column"
            className="h-screen w-screen bg-qt-window text-qt-text overflow-hidden font-sans text-qt-md selection:bg-qt-highlight selection:text-white"
        >
            <AppDialog
                open={Boolean(vocalShifterSkippedFilesDialog?.length)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeVocalShifterSkippedFilesDialog());
                    }
                }}
                title={t("status_error_prefix")}
                description={t("vs_import_skipped_header")}
                size="lg"
                actions={[
                    {
                        id: "ok",
                        label: t("ok"),
                        intent: "primary",
                        onClick: () => {
                            dispatch(closeVocalShifterSkippedFilesDialog());
                        },
                    },
                ]}
            >
                {/*
                 * 外层不滚、内层滚。
                 *
                 * 对话框的 body 本身就是 `overflow-y-auto`（见 AppDialog）。这里再放
                 * 一个带 `max-h-[240px]` 的滚动盒，当"描述 + 列表"超过 body 高度时就会
                 * 出现**两条竖直滚动条**，而里面那条滚下去什么也看不到 —— 用户只会以为
                 * 下面还有内容。
                 *
                 * 改法：列表参与 body 的 flex 布局（`flex-1 min-h-0`），body 因此永不
                 * 溢出，只剩一条滚动条。间距用 `pt-2` 而不是 `mt-2`：外边距会把总高撑过
                 * 容器的 100%，又造出溢出。
                 */}
                <div className="flex h-full min-h-0 flex-col pt-2">
                    <div className="hs-scroll-gutter-flush min-h-0 flex-1 overflow-auto rounded border border-qt-border bg-qt-base p-2 text-qt-xs">
                        {(vocalShifterSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" data-tooltip={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                </div>
            </AppDialog>

            <AppDialog
                open={Boolean(reaperSkippedFilesDialog?.length)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeReaperSkippedFilesDialog());
                    }
                }}
                title={t("status_error_prefix")}
                description={t("reaper_import_skipped_header")}
                size="lg"
                actions={[
                    {
                        id: "ok",
                        label: t("ok"),
                        intent: "primary",
                        onClick: () => {
                            dispatch(closeReaperSkippedFilesDialog());
                        },
                    },
                ]}
            >
                {/*
                 * 外层不滚、内层滚。
                 *
                 * 对话框的 body 本身就是 `overflow-y-auto`（见 AppDialog）。这里再放
                 * 一个带 `max-h-[240px]` 的滚动盒，当"描述 + 列表"超过 body 高度时就会
                 * 出现**两条竖直滚动条**，而里面那条滚下去什么也看不到 —— 用户只会以为
                 * 下面还有内容。
                 *
                 * 改法：列表参与 body 的 flex 布局（`flex-1 min-h-0`），body 因此永不
                 * 溢出，只剩一条滚动条。间距用 `pt-2` 而不是 `mt-2`：外边距会把总高撑过
                 * 容器的 100%，又造出溢出。
                 */}
                <div className="flex h-full min-h-0 flex-col pt-2">
                    <div className="hs-scroll-gutter-flush min-h-0 flex-1 overflow-auto rounded border border-qt-border bg-qt-base p-2 text-qt-xs">
                        {(reaperSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" data-tooltip={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                </div>
            </AppDialog>

            <AppDialog
                open={unsavedDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        cancelUnsavedAction();
                    }
                }}
                title={t("unsaved_changes_title")}
                /*
                 * 主消息走 `message`（13px 正文色），不再借用副标题槽位
                 * （11px 弱化色）—— 这是全应用最需要被读到的一句话之一。
                 */
                message={t(
                    unsavedDialog.mode === "exit"
                        ? "unsaved_changes_exit_desc"
                        : "unsaved_changes_switch_desc",
                )}
                tone="danger"
                size="sm"
                actions={[
                    {
                        id: "cancel",
                        label: t("progress_cancel"),
                        onClick: cancelUnsavedAction,
                    },
                    {
                        id: "discard",
                        label: t("unsaved_changes_discard"),
                        // 丢弃是本对话框里唯一不可逆的选择：严重度落在后果上。
                        intent: "danger",
                        onClick: discardUnsavedAndContinue,
                    },
                    {
                        id: "save",
                        label: t("menu_save_project"),
                        intent: "primary",
                        // 保存取消/失败/命中版本冲突时对话框须保持打开以便重试，
                        // 关闭由 saveUnsavedAndContinue 内部决定，故不自动关闭。
                        autoClose: false,
                        onClick: saveUnsavedAndContinue,
                    },
                ]}
            />

            {/* Project file version newer than this build — ask before attempting load */}
            <AppDialog
                open={projectVersionDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        setProjectVersionDialog((current) => ({ ...current, open: false }));
                    }
                }}
                title={t("project_version_too_new_title")}
                message={t("project_version_too_new_desc")
                    .replace("{fileVersion}", String(projectVersionDialog.fileVersion || "?"))
                    .replace(
                        "{currentVersion}",
                        String(projectVersionDialog.currentVersion || "?"),
                    )}
                tone="warning"
                size="sm"
                actions={[
                    {
                        id: "cancel",
                        label: t("progress_cancel"),
                        onClick: cancelContinueLoadingNewerProject,
                    },
                    {
                        id: "continue",
                        label: t("project_version_too_new_continue"),
                        intent: "primary",
                        onClick: confirmContinueLoadingNewerProject,
                    },
                ]}
            />

            {/* 保存/另存为目标已存在版本不一致的工程文件 — 覆盖前询问用户 */}
            <AppDialog
                open={Boolean(saveVersionConflictDialog)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeSaveVersionConflictDialog());
                    }
                }}
                title={t("save_version_conflict_title")}
                /*
                 * 这是全应用最长的单条提示（343 字符），而它讲的是"覆盖会降级、
                 * 可能丢参数"。此前它以 11px 弱化色渲染在 400px 列里 —— 约 7 行
                 * 全应用最小最淡的字。改走 `message` 并标为 danger。
                 */
                message={
                    saveVersionConflictDialog?.existingIsNewer
                        ? t("save_version_conflict_desc_higher")
                              .replace(
                                  "{existingVersion}",
                                  String(saveVersionConflictDialog.existingVersion),
                              )
                              .replace(
                                  "{currentVersion}",
                                  String(saveVersionConflictDialog.currentVersion),
                              )
                        : t("save_version_conflict_desc_lower")
                              .replace(
                                  "{existingVersion}",
                                  String(saveVersionConflictDialog?.existingVersion ?? "?"),
                              )
                              .replace(
                                  "{currentVersion}",
                                  String(saveVersionConflictDialog?.currentVersion ?? "?"),
                              )
                }
                tone="danger"
                size="md"
                actions={[
                    {
                        id: "cancel",
                        label: t("progress_cancel"),
                        onClick: cancelSaveVersionConflict,
                    },
                    {
                        id: "save-as",
                        label: t("save_version_conflict_save_as"),
                        onClick: saveAsFromVersionConflict,
                    },
                    {
                        id: "continue",
                        label: t("save_version_conflict_continue"),
                        intent: "danger",
                        onClick: continueForceSave,
                    },
                ]}
            />

            {/* Recapture missing media dialog — grid: file status / file / processing status / ignore / action */}
            <AppDialog
                open={sourceFileChangedDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        closeSourceFileChangedDialog();
                    }
                }}
                title={t("recapture_missing_media_title")}
                description={t("recapture_missing_media_desc")}
                size="xl"
                actions={[
                    {
                        id: "ok",
                        label: t("ok"),
                        intent: "primary",
                        disabled:
                            sourceFileAnyProcessing ||
                            sourceFileChangedDialog.changes.some(
                                (item) => item.action === "pending",
                            ),
                        onClick: closeSourceFileChangedDialog,
                    },
                ]}
            >
                {/*
                 * 外层不滚、内层滚（见下面列表上的注释）：这一层只是把 body 变成
                 * flex 列，好让列表用 `flex-1 min-h-0` 吃掉剩余高度。
                 */}
                <div className="flex h-full min-h-0 flex-col">
                    <Flex justify="between" align="center" gap="2" mt="2">
                        <Flex gap="1" align="center" className="shrink-0">
                            <span className="shrink-0 text-qt-micro text-qt-text-muted">
                                {t("recapture_missing_media_search_mode_label")}
                            </span>
                            <WheelSelect
                                className="h-6 shrink-0 rounded border border-qt-border bg-qt-base px-1 py-0.5 text-qt-micro text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30"
                                value={sourceFileSearchMode}
                                disabled={
                                    sourceFileSearchBusy ||
                                    sourceFileAnyProcessing ||
                                    !sourceFileChangedDialog.changes.some(
                                        (item) =>
                                            item.action === "pending" || item.action === "failed",
                                    )
                                }
                                onValueChange={(value) =>
                                    setSourceFileSearchMode(value as "file_name" | "extension_hash")
                                }
                            >
                                <option value="file_name">
                                    {t("recapture_missing_media_search_mode_file_name")}
                                </option>
                                <option value="extension_hash">
                                    {t("recapture_missing_media_search_mode_extension_hash")}
                                </option>
                            </WheelSelect>
                            <Button
                                size="1"
                                variant="soft"
                                disabled={
                                    sourceFileSearchBusy ||
                                    sourceFileAnyProcessing ||
                                    !sourceFileChangedDialog.changes.some(
                                        (item) =>
                                            item.action === "pending" || item.action === "failed",
                                    )
                                }
                                onClick={() => void searchSourceFileReplacements()}
                            >
                                {sourceFileSearchBusy
                                    ? t("recapture_missing_media_searching")
                                    : t("recapture_missing_media_search_folder")}
                            </Button>
                        </Flex>
                        <Flex gap="1" align="center" className="shrink-0">
                            <Button
                                size="1"
                                variant="soft"
                                disabled={sourceFileAnyProcessing || sourceFileReloadAllTotal === 0}
                                onClick={() => void reloadAllModifiedSourceFiles()}
                            >
                                {t("recapture_missing_media_reload_all")}
                            </Button>
                            <Button
                                size="1"
                                variant="soft"
                                disabled={sourceFileAnyProcessing || sourceFileSearchBusy}
                                onClick={() => void refreshSourceFileChanges()}
                            >
                                {t("recapture_missing_media_refresh")}
                            </Button>
                            <Button
                                size="1"
                                variant="soft"
                                color="gray"
                                disabled={sourceFileAnyProcessing}
                                onClick={() => void resetAllSourceFileChanges()}
                            >
                                {t("recapture_missing_media_reset_all")}
                            </Button>
                            <Button
                                size="1"
                                variant="soft"
                                color="gray"
                                disabled={
                                    sourceFileAnyProcessing ||
                                    !sourceFileChangedDialog.changes.some(
                                        (item) =>
                                            item.action !== "ignored" &&
                                            item.action !== "reloaded" &&
                                            item.action !== "replaced" &&
                                            item.action !== "processing",
                                    )
                                }
                                onClick={ignoreAllSourceFileChanges}
                            >
                                {t("recapture_missing_media_ignore_all")}
                            </Button>
                        </Flex>
                    </Flex>

                    {sourceFileSearchMatchTotal > 0 && (
                        <Flex
                            justify="between"
                            align="center"
                            gap="2"
                            mt="1"
                            className="rounded border border-qt-border bg-qt-base px-2 py-1.5"
                        >
                            <span className="hs-type-label min-w-0 truncate">
                                {t("recapture_missing_media_search_result_summary")
                                    .replace("{total}", String(sourceFileSearchMatchTotal))
                                    .replace("{exact}", String(sourceFileSearchExactTotal))
                                    .replace("{selected}", String(sourceFileSelectedApplyTotal))}
                            </span>
                            <Flex gap="1" align="center" className="shrink-0">
                                {(sourceFileExactApplyTotal > 0 ||
                                    sourceFileSelectedApplyTotal > 0) && (
                                    <>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="green"
                                            disabled={
                                                sourceFileAnyProcessing ||
                                                sourceFileExactApplyTotal === 0
                                            }
                                            onClick={() => void applyAllExactSourceFileMatches()}
                                        >
                                            {t("recapture_missing_media_apply_all_exact")}
                                        </Button>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            disabled={
                                                sourceFileAnyProcessing ||
                                                sourceFileSelectedApplyTotal === 0
                                            }
                                            onClick={() => void applyAllSelectedSourceFileMatches()}
                                        >
                                            {t("recapture_missing_media_apply_all_selected")}
                                        </Button>
                                    </>
                                )}
                            </Flex>
                        </Flex>
                    )}

                    {/*
                     * 对话框 body 已经是 `overflow-y-auto`，这里若再给一个固定的
                     * `max-h-[320px]` 就会出现两条竖直滚动条，而里面那条滚下去什么
                     * 也看不到。改为参与上面的 flex 列：`flex-1 min-h-0` 让列表吃
                     * 掉剩余高度，body 因此永不溢出，只剩一条滚动条。
                     */}
                    <div className="hs-scroll-gutter-flush mt-2 min-h-0 flex-1 overflow-auto rounded border border-qt-border bg-qt-base p-1">
                        <div className="grid grid-cols-[64px_minmax(0,1fr)_76px_64px_84px_48px] items-center gap-2 border-b border-qt-border px-1 py-1 text-qt-micro font-semibold text-qt-text-muted">
                            <div>{t("recapture_missing_media_col_file_status")}</div>
                            <div>{t("recapture_missing_media_col_file")}</div>
                            <div>{t("recapture_missing_media_col_process_status")}</div>
                            <div />
                            <div className="text-right">
                                {t("recapture_missing_media_col_action")}
                            </div>
                            <div className="text-right">
                                {t("recapture_missing_media_reset_item")}
                            </div>
                        </div>
                        {sourceFileChangedDialog.changes.map((item) => {
                            const itemKey = `${item.clip_id}::${item.change}`;
                            const isBusy = item.action === "processing";
                            const isProcessed =
                                item.action === "ignored" ||
                                item.action === "reloaded" ||
                                item.action === "replaced";
                            const statusBadgeClass =
                                item.action === "ignored"
                                    ? "border border-qt-border bg-qt-base text-qt-text-muted"
                                    : item.action === "reloaded" || item.action === "replaced"
                                      ? "border border-qt-success-border bg-qt-success-bg text-qt-success-text"
                                      : item.action === "failed"
                                        ? "border border-qt-danger-border bg-qt-danger-bg text-qt-danger-text"
                                        : item.action === "processing"
                                          ? "border border-qt-info-border bg-qt-info-bg text-qt-info-text"
                                          : "border border-qt-border bg-qt-base text-qt-text-muted";
                            const statusLabel =
                                item.action === "ignored"
                                    ? t("recapture_missing_media_item_ignored")
                                    : item.action === "reloaded"
                                      ? t("recapture_missing_media_item_reloaded")
                                      : item.action === "replaced"
                                        ? t("recapture_missing_media_item_replaced")
                                        : item.action === "failed"
                                          ? t("recapture_missing_media_item_failed")
                                          : item.action === "processing"
                                            ? t("recapture_missing_media_item_processing")
                                            : t("recapture_missing_media_item_pending");
                            const useReplaceAction =
                                item.change === "deleted" ||
                                Boolean(item.reloadAttempted) ||
                                item.action === "ignored" ||
                                item.action === "reloaded" ||
                                item.action === "replaced";
                            return (
                                <div
                                    key={itemKey}
                                    className="grid grid-cols-[64px_minmax(0,1fr)_76px_64px_84px_48px] items-center gap-2 border-b border-qt-border px-1 py-1.5 text-qt-xs last:border-b-0"
                                >
                                    <div
                                        className={`shrink-0 whitespace-nowrap rounded px-1 py-0.5 text-center text-qt-micro font-semibold leading-none ${
                                            item.change === "deleted"
                                                ? "border border-qt-danger-border bg-qt-danger-bg text-qt-danger-text"
                                                : "border border-qt-warning-border bg-qt-warning-bg text-qt-warning-text"
                                        }`}
                                    >
                                        {item.change === "deleted"
                                            ? t("recapture_missing_media_status_deleted")
                                            : t("recapture_missing_media_status_modified")}
                                    </div>
                                    <div className="min-w-0">
                                        <div className="truncate" data-tooltip={item.source_path}>
                                            <span className="font-medium">{item.clip_name}</span>
                                            <span className="text-qt-text-muted">
                                                {" "}
                                                — {item.source_path}
                                            </span>
                                        </div>
                                        {item.reloadedPath && (
                                            <div
                                                className="mt-0.5 flex items-center gap-1 truncate text-qt-micro text-qt-success-text"
                                                data-tooltip={item.reloadedPath}
                                            >
                                                <span className="shrink-0 font-semibold">
                                                    {item.action === "reloaded"
                                                        ? t("recapture_missing_media_reloaded_path")
                                                        : t(
                                                              "recapture_missing_media_replaced_path",
                                                          )}
                                                </span>
                                                <span className="truncate">
                                                    {item.reloadedPath}
                                                </span>
                                            </div>
                                        )}
                                        {!isProcessed &&
                                            item.candidates &&
                                            item.candidates.length > 0 && (
                                                <div className="mt-1 flex items-center gap-1">
                                                    <WheelSelect
                                                        className="min-w-0 flex-1 rounded border border-qt-border bg-qt-base px-1 py-0.5 text-qt-micro text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30"
                                                        value={item.selectedCandidatePath ?? ""}
                                                        disabled={isBusy}
                                                        onValueChange={(value) =>
                                                            selectSourceFileMatchCandidate(
                                                                item.clip_id,
                                                                value,
                                                            )
                                                        }
                                                    >
                                                        {item.candidates.map((candidate) => (
                                                            <option
                                                                key={candidate.path}
                                                                value={candidate.path}
                                                            >
                                                                {candidate.exact_hash
                                                                    ? tVars(
                                                                          "recapture_missing_media_exact_option",
                                                                          { path: candidate.path },
                                                                      )
                                                                    : candidate.path}
                                                            </option>
                                                        ))}
                                                    </WheelSelect>
                                                    <Button
                                                        size="1"
                                                        variant="soft"
                                                        disabled={
                                                            isBusy || !item.selectedCandidatePath
                                                        }
                                                        onClick={() =>
                                                            void applySelectedSourceFileMatch(item)
                                                        }
                                                    >
                                                        {t("recapture_missing_media_use_selected")}
                                                    </Button>
                                                </div>
                                            )}
                                        {!isProcessed &&
                                            item.candidates &&
                                            item.candidates.length === 0 && (
                                                <div className="mt-1 truncate text-qt-micro text-qt-text-muted">
                                                    {t("recapture_missing_media_search_no_matches")}
                                                </div>
                                            )}
                                    </div>
                                    <div
                                        className={`shrink-0 whitespace-nowrap rounded px-1.5 py-0.5 text-center text-qt-micro font-semibold leading-none ${statusBadgeClass}`}
                                    >
                                        {statusLabel}
                                    </div>
                                    <div className="flex justify-start">
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            disabled={isBusy || isProcessed}
                                            onClick={() => void ignoreSourceFileChangeItem(item)}
                                        >
                                            {t("recapture_missing_media_ignore")}
                                        </Button>
                                    </div>
                                    <div className="flex justify-end">
                                        <Button
                                            size="1"
                                            variant="soft"
                                            disabled={isBusy}
                                            onClick={() =>
                                                useReplaceAction
                                                    ? void replaceSourceFileChangeItem(item)
                                                    : void reloadSourceFileChangeItem(item)
                                            }
                                        >
                                            {useReplaceAction
                                                ? t("recapture_missing_media_replace")
                                                : t("recapture_missing_media_reload")}
                                        </Button>
                                    </div>
                                    <div className="flex justify-end">
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            disabled={isBusy}
                                            onClick={() => resetSourceFileChangeItem(item.clip_id)}
                                        >
                                            {t("recapture_missing_media_reset_item")}
                                        </Button>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                    <Flex justify="between" align="center" gap="2" mt="3">
                        <span className="hs-type-label">
                            {t("recapture_missing_media_summary")
                                .replace(
                                    "{ignored}",
                                    String(
                                        sourceFileChangedDialog.changes.filter(
                                            (item) => item.action === "ignored",
                                        ).length,
                                    ),
                                )
                                .replace(
                                    "{reloaded}",
                                    String(
                                        sourceFileChangedDialog.changes.filter(
                                            (item) => item.action === "reloaded",
                                        ).length,
                                    ),
                                )
                                .replace(
                                    "{replaced}",
                                    String(
                                        sourceFileChangedDialog.changes.filter(
                                            (item) => item.action === "replaced",
                                        ).length,
                                    ),
                                )
                                .replace("{total}", String(sourceFileChangedDialog.changes.length))}
                        </span>
                    </Flex>
                </div>
            </AppDialog>

            <ImportProjectDialog
                key={projectImportPick.open ? (projectImportPick.path ?? "open") : "closed"}
                open={projectImportPick.open}
                projectPath={projectImportPick.path}
                hasExistingTempoMap={hasExistingTempoMap}
                onOpenChange={(open) => setProjectImportPick((prev) => ({ ...prev, open }))}
                onConfirm={handleImportProjectConfirmed}
            />

            <MenuBar
                onNewProject={handleNewProject}
                onOpenProject={handleOpenProject}
                onOpenRecentProject={handleOpenRecentProject}
                onRecaptureMissingMedia={handleRecaptureMissingMedia}
                onImportProject={handleImportProject}
                onExit={handleExitApp}
                onImportMidiFromMenu={handleImportMidiFromMenu}
                autoBackupSettings={autoBackupSettings}
                onAutoBackupSettingsSaved={handleAutoBackupSettingsSaved}
                autoReloadModifiedMedia={autoReloadModifiedMedia}
                onAutoReloadModifiedMediaChange={handleAutoReloadModifiedMediaChange}
                loopNewClips={loopNewClips}
                onLoopNewClipsChange={handleLoopNewClipsChange}
            />
            <ActionBar />

            {/*
             * 工作区：全部可停靠窗体由布局树驱动。
             *
             * 这里取代了原先写死的"时间轴 / 分隔条 / 参数编辑器 + 右侧固定宽度栏"
             * 结构。每个面板的组件实例只挂载一次，靠搬 DOM 宿主换位置（见
             * `components/dock/panelHostRegistry`），因此停靠重排不会重建
             * WebGL 上下文、不会丢滚动位置。面板自己的 props 由
             * `usePanelRenderers` 注入，App 仍是这些状态的唯一所有者。
             */}
            <DockRoot />
            {/* Quick Search Popup */}
            <QuickSearchPopup open={quickSearchOpen} onClose={() => setQuickSearchOpen(false)} />
            {/*
              目录导入的宿主：监听"导入文件夹"请求、扫描、必要时弹选项对话框。
              挂在最外层是因为三个入口（系统拖放 / 文件浏览器拖拽 / 右键菜单）分散在
              不同位置 —— 只有一份对话框状态，才不会出现"两个面板各弹一个"。
            */}
            <FolderImportHost />

            {/* Status Bar */}
            <Flex
                align="center"
                justify="between"
                className="h-qt-bar-status bg-qt-window border-t border-qt-border px-1 select-none gap-2"
            >
                <Flex align="center" gap="1" className="truncate min-w-0">
                    {/* 排列规则：**长时效的提示在前，短时效的进度片在后**。
                        绿色提示位（后台缓存统计 / 自动声道折叠）会挂十几秒，而
                        紫色进度片（拉伸、波形分析、渲染…）只是几百毫秒的过客。
                        若把进度片放在前面，它每出现/消失一次，后面所有片就整体
                        横移一次 —— 用户视线里"提示在乱动"。把长时效的钉在最左，
                        短的插在它右侧，左侧位置就永远稳定。 */}
                    {noticeText ? <AppStatusChip tone="success">{noticeText}</AppStatusChip> : null}
                    {/* 导入等待提示：只在真慢时点亮（见 IMPORT_BUSY_DELAY_MS），
                        位置紧跟长时效提示位之后、短时效进度片之前。 */}
                    {importBusy ? (
                        <AppStatusChip tone="accent">{t("status_importing")}</AppStatusChip>
                    ) : null}
                    {pitchAnalysisText ? (
                        <AppStatusChip tone="accent">{pitchAnalysisText}</AppStatusChip>
                    ) : null}
                    {/* 参数曲线取数提示：**独立订阅**外部 store，不参与本组件重渲染
                        （见 ParamDataLoadingChip 的说明）。 */}
                    <ParamDataLoadingChip />
                    {/* 高频进度三片（拉伸 / 波形分析 / 渲染进度）：**独立订阅**外部
                        store，不参与本组件重渲染（见 AppStatusProgressChips 的说明）。 */}
                    <AppStatusProgressChips />
                    <span
                        className="hs-type-label truncate"
                        style={error ? { color: "var(--qt-danger-text)" } : undefined}
                    >
                        {errorText}
                    </span>
                </Flex>
                {/*
                  状态栏右侧槽位：插件宿主的「自动应用」状态。
                  它曾经是 `ActionBar` 与工作区之间的一整行横条 —— 在插件那个小窗口里
                  一整行高度很贵，而这段内容本来就是状态读数，归状态栏。
                  （见 `PluginApplyStatus`：片独立订阅外部 store，250 ms 的轮询不会
                  带动本组件重渲染。）
                */}
                {isPluginMode() ? (
                    <PluginApplyStatus
                        onTimelineChanged={async () => {
                            await dispatch(fetchTimeline()).unwrap();
                        }}
                    />
                ) : null}
            </Flex>
        </Flex>
    );
}

function App() {
    return (
        <PitchAnalysisProvider>
            <AppInner />
        </PitchAnalysisProvider>
    );
}

export default App;
