import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Flex, Box, Text, Dialog, Button } from "@radix-ui/themes";
import { MenuBar } from "./components/layout/MenuBar";
import { ActionBar } from "./components/layout/ActionBar";
import { TimelinePanel } from "./components/layout/TimelinePanel";
import { PianoRollPanel } from "./components/layout/PianoRollPanel";
import { useAppDispatch, useAppSelector } from "./app/hooks";
import { webApi } from "./services/webviewApi";
import { settingsApi } from "./services/api/settings";
import { fileBrowserApi } from "./services/api/fileBrowser";
import { IS_LINUX } from "./utils/platform";
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
    undoRemote,
    redoRemote,
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromPath,
    openReaperFromPath,
    importAudioFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    saveProjectToPathRemote,
    setTrackMeters,
    setToolMode,
    checkpointHistory,
    addTrackRemote,
    replaceClipSourceRemote,
} from "./features/session/sessionSlice";
import { useI18n } from "./i18n/I18nProvider";
import { useClipPitchDataListener } from "./hooks/useClipPitchDataListener";
import { PitchAnalysisProvider, usePitchAnalysis } from "./contexts/PitchAnalysisContext";
import { PianoRollStatusProvider, usePianoRollStatus } from "./contexts/PianoRollStatusContext";
import { FileBrowserPanel } from "./components/layout/FileBrowserPanel";
import { NotebookPanel } from "./components/layout/NotebookPanel";
import { ImportProjectDialog } from "./components/layout/ImportProjectDialog";
import { QuickSearchPopup } from "./components/layout/QuickSearchPopup";
import { useKeybindings } from "./features/keybindings/useKeybindings";
import type { ActionId } from "./features/keybindings/types";
import { store } from "./app/store";
import { resolveRootTrackId } from "./features/session/trackUtils";
import { getParamShiftStep } from "./components/layout/pianoRoll/paramShiftStep";
import { runConfirmedExitClose } from "./confirmedExitClose";
import { paramsApi } from "./services/api";
import { coreApi } from "./services/api/core";
import type { SourceFileChange, SourceFileMatchCandidate } from "./services/api/timeline";
import { waveformMipmapStore } from "./utils/waveformMipmapStore";
import { projectApi, type AutoBackupSettings } from "./services/api/project";
import type { ParamFramesPayload, ProcessorParamDescriptor } from "./types/api";
import {
    OPEN_PROJECT_PATH_EVENT,
    type ExternalFileActionDetail,
    type ExternalFileActionKind,
} from "./features/session/projectOpenEvents";
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
    "VocalShifter imported with skipped files": "vs_import_skipped_header",
};

// 后端返回的错误码 → i18n key 映射
const errorCodeKey: Record<string, string> = {
    clipboard_not_found: "vs_paste_clipboard_not_found",
    clipboard_invalid_format: "vs_paste_clipboard_invalid_format",
    clipboard_io_error: "vs_paste_clipboard_io_error",
    no_pitch_line_selected: "vs_paste_no_pitch_line",
    import_read_failed: "vs_import_read_failed",
    import_parse_failed: "vs_import_parse_failed",
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

function detectExternalActionKindFromPath(path: string): ExternalFileActionKind | null {
    const normalized = String(path ?? "").trim();
    if (!normalized) return null;
    if (/\.(hshp|hsp|json)$/i.test(normalized)) return "openProject";
    if (/\.rpp$/i.test(normalized)) return "importReaper";
    if (/\.(vshp|vsp)$/i.test(normalized)) return "importVocalShifter";
    if (
        /\.(wav|flac|mp3|ogg|oga|opus|aac|m4a|aif|aiff|wma|ac3|eac3|ape|wv|mp2|mpa|dts|amr|mp4|m4v|mov|mkv|webm|avi|flv|wmv|ts|mts|m2ts|vob|mpg|mpeg|3gp|3g2|ogv|rm|rmvb)$/i.test(
            normalized,
        )
    ) {
        return "importAudio";
    }
    return null;
}

function AppInner() {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const pitchAnalysis = usePitchAnalysis();
    const pianoRollStatus = usePianoRollStatus();

    const status = useAppSelector((state) => state.session.status);
    const error = useAppSelector((state) => state.session.error);

    const runtimeIsPlaying = useAppSelector((state) => state.session.runtime.isPlaying);
    const runtimeHasSynthesized = useAppSelector((state) => state.session.runtime.hasSynthesized);
    const fileBrowserVisible = useAppSelector((state) => state.fileBrowser.visible);
    const notebookVisible = useAppSelector((state) => state.notebook.visible);
    const toolMode = useAppSelector((state) => state.session.toolMode);
    const drawToolMode = useAppSelector((state) => state.session.drawToolMode);
    const projectDirty = useAppSelector((state) => state.session.project.dirty);
    const playheadSec = useAppSelector((state) => state.session.playheadSec);
    const selectedTrackId = useAppSelector((state) => state.session.selectedTrackId);
    const paramsEpoch = useAppSelector((state) => state.session.paramsEpoch);
    const recordingActive = useAppSelector((state) => state.recording.active);
    const recordingSettings = useAppSelector((state) => state.recording.settings);
    const recordingStartSec = useAppSelector((state) => state.recording.startSec);
    const selectedClipId = useAppSelector((state) => state.session.selectedClipId);
    const multiSelectedClipIds = useAppSelector((state) => state.session.multiSelectedClipIds);
    const sessionClips = useAppSelector((state) => state.session.clips);
    const playbackPositionSec = useAppSelector(
        (state) => state.session.runtime.playbackPositionSec,
    );
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

    const containerRef = useRef<HTMLDivElement | null>(null);
    const dragRef = useRef<{ pointerId: number } | null>(null);
    const [splitRatio, setSplitRatio] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.splitRatio"));
        return Number.isFinite(stored) ? Math.min(0.85, Math.max(0.15, stored)) : 0.6;
    });
    const splitRatioRef = useRef(splitRatio);
    const [isDragging, setIsDragging] = useState(false);
    const [quickSearchOpen, setQuickSearchOpen] = useState(false);
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

    // 加载 MIDI 相关设置
    useEffect(() => {
        settingsApi.getUiSettings().then((s) => {
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
        });
    }, []);

    const handleImportMidiFromMenu = useCallback(() => {
        setMidiDialogSource("menu");
        setMidiClipPath(null);
        setMidiClipClipboardGuid(null);
        setMidiClipStartSec(playheadSec ?? 0);
        setMidiClipTrackId(selectedTrackId ?? null);
        setMidiClipDialogOpen(true);
    }, [playheadSec, selectedTrackId]);

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

    const splitter = useMemo(() => {
        const minTopPx = 200;
        const minBottomPx = 150;
        const handlePx = 8;

        function clamp(v: number, minV: number, maxV: number) {
            return Math.min(maxV, Math.max(minV, v));
        }

        // 提取纯计算逻辑，不在此处触发 React 状态更新
        function calculateRatio(clientY: number) {
            const el = containerRef.current;
            if (!el) return null;
            const rect = el.getBoundingClientRect();
            const total = rect.height;
            if (!Number.isFinite(total) || total <= minTopPx + minBottomPx + handlePx) {
                return null;
            }
            const y = clientY - rect.top;
            const maxTop = total - handlePx - minBottomPx;
            const nextTop = clamp(y, minTopPx, maxTop);
            return clamp(nextTop / total, 0.15, 0.85);
        }

        function onPointerMove(e: PointerEvent) {
            if (!dragRef.current) return;
            const nextRatio = calculateRatio(e.clientY);
            if (nextRatio === null) return;

            // 拖拽时直接修改 DOM 的 flexGrow，绕过 React 重绘
            const container = containerRef.current;
            if (container && container.children.length >= 3) {
                const topPanel = container.children[0] as HTMLElement;
                const bottomPanel = container.children[2] as HTMLElement;
                topPanel.style.flexGrow = String(nextRatio);
                bottomPanel.style.flexGrow = String(1 - nextRatio);
            }

            splitRatioRef.current = nextRatio;
        }

        function endDrag() {
            if (!dragRef.current) return;
            dragRef.current = null;
            setIsDragging(false);

            // 只在松开鼠标的最后一刻，才把最终状态同步给 React 并持久化
            setSplitRatio(splitRatioRef.current);
            localStorage.setItem("hifishifter.splitRatio", String(splitRatioRef.current));

            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", endDrag);
            window.removeEventListener("pointercancel", endDrag);
        }

        function startDrag(e: React.PointerEvent<HTMLDivElement>) {
            if (e.button !== 0) return;
            dragRef.current = { pointerId: e.pointerId };
            setIsDragging(true);
            (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);

            // 按下的瞬间也走一次 DOM 直通更新
            const nextRatio = calculateRatio(e.clientY);
            if (nextRatio !== null) {
                splitRatioRef.current = nextRatio;
                const container = containerRef.current;
                if (container && container.children.length >= 3) {
                    const topPanel = container.children[0] as HTMLElement;
                    const bottomPanel = container.children[2] as HTMLElement;
                    topPanel.style.flexGrow = String(nextRatio);
                    bottomPanel.style.flexGrow = String(1 - nextRatio);
                }
            }

            window.addEventListener("pointermove", onPointerMove);
            window.addEventListener("pointerup", endDrag);
            window.addEventListener("pointercancel", endDrag);
        }

        return { startDrag };
    }, []);

    const statusText = useMemo(() => {
        // 精确匹配
        if (statusKey[status]) return t(statusKey[status] as MessageKey);
        // 前缀匹配：支持 "Export done — path" 等带后缀的状态
        for (const key of Object.keys(statusKey)) {
            if (status.startsWith(key) && status.length > key.length) {
                const suffix = status.slice(key.length);
                return t(statusKey[key] as MessageKey) + suffix;
            }
        }
        return status;
    }, [status, t]);

    // 监听后端 clip_pitch_data 事件，将 per-clip MIDI 曲线存入 store
    useClipPitchDataListener();
    useClipFormantStatusListener();
    useRecordingListener();

    // 阻止浏览器默认的 Ctrl+F 搜索、右键菜单和 Alt 键

    // 改用 useRef，取消重绘
    const isModifierRef = useRef(false);

    useEffect(() => {
        function isEditableTarget(target: EventTarget | null): boolean {
            const el = target as HTMLElement | null;
            if (!el) return false;
            const tag = (el.tagName ?? "").toLowerCase();
            if (tag === "input" || tag === "textarea" || tag === "select") return true;
            if (el.isContentEditable) return true;
            return el.closest?.('input,textarea,select,[contenteditable="true"]') != null;
        }

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

        // 只允许可编辑控件和显式声明可选择/拖拽的区域使用 WebView 原生选择逻辑。
        function allowsNativeTextSelection(target: EventTarget | null): boolean {
            if (isEditableTarget(target)) return true;
            let node = target instanceof Element ? target : null;
            while (node) {
                if (node.getAttribute?.("data-hs-selectable") === "true") return true;
                try {
                    const style = window.getComputedStyle(node) as CSSStyleDeclaration & {
                        webkitUserSelect?: string;
                    };
                    const userSelect = style.userSelect || style.webkitUserSelect || "";
                    if (userSelect === "text" || userSelect === "all") return true;
                    if (userSelect === "none") return false;
                } catch {
                    // ignore
                }
                node = node.parentElement;
            }
            return false;
        }

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

    const errorText = error
        ? `${t("status_error_prefix")}：${errorCodeKey[error] ? t(errorCodeKey[error] as MessageKey) : error}`
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

    const [rendering, setRendering] = useState<{
        active: boolean;
        progress: number | null;
        target: string | null;
    }>({ active: false, progress: null, target: null });

    const [stretching, setStretching] = useState<{
        active: boolean;
        clipName: string | null;
    }>({ active: false, clipName: null });

    // 波形分析进度状态
    const [waveformAnalysis, setWaveformAnalysis] = useState<{
        active: boolean;
        sourcePath: string | null;
        progress: number | null;
    }>({ active: false, sourcePath: null, progress: null });

    // Listen for backend stretch progress notifications (Tauri only).
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen("stretch_progress", (event: any) => {
                    if (disposed) return;
                    const payload = (event?.payload ?? {}) as {
                        active?: boolean;
                        clipName?: string | null;
                    };
                    const active = Boolean(payload?.active);
                    const clipName =
                        typeof payload?.clipName === "string" ? payload.clipName : null;
                    setStretching({ active, clipName });
                });
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
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen("track_meter", (event: any) => {
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
                });
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
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen("waveform_analysis_progress", (event: any) => {
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
                        typeof payload?.progress === "number" && Number.isFinite(payload.progress)
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
                            setWaveformAnalysis({
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
                        setWaveformAnalysis({
                            active: true,
                            sourcePath: fileName,
                            progress: p,
                        });
                    } else if (status === "done" || status === "cached") {
                        // 完成后延迟 1.5 秒隐藏，让用户有时间看到 100%
                        if (status === "done") {
                            currentProgress = 1.0;
                            currentComputingPath = null;
                            setWaveformAnalysis({
                                active: true,
                                sourcePath: null,
                                progress: 1.0,
                            });
                            fadeOutTimer = setTimeout(() => {
                                if (!disposed) {
                                    currentProgress = -1;
                                    setWaveformAnalysis({
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
                            setWaveformAnalysis({
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
                            setWaveformAnalysis({
                                active: false,
                                sourcePath: null,
                                progress: null,
                            });
                        }
                    }
                });
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
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen("playback_rendering_state", (event: any) => {
                    if (disposed) return;
                    const payload = (event?.payload ?? {}) as {
                        active?: boolean;
                        progress?: number | null;
                        target?: string | null;
                    };
                    const active = Boolean(payload?.active);
                    const pRaw = payload?.progress;
                    const p =
                        typeof pRaw === "number" && Number.isFinite(pRaw)
                            ? Math.max(0, Math.min(1, pRaw))
                            : null;
                    const target = typeof payload?.target === "string" ? payload.target : null;

                    setRendering({ active, progress: p, target });

                    // 渲染从 active→inactive（完成）时，延迟同步一次播放状态，
                    // 使前端能感知后端已真正开始播放。
                    if (!active && renderingWasActiveRef.current) {
                        setTimeout(() => {
                            dispatch(syncPlaybackState());
                        }, 200);
                    }
                    renderingWasActiveRef.current = active;
                });
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

    const runtimeRef = useRef({
        isPlaying: false,
        hasSynthesized: false,
        toolMode: "draw" as import("./features/session/sessionTypes").ToolMode,
        drawToolMode: "draw" as import("./features/session/sessionTypes").DrawToolMode,
    });

    const playbackSyncInFlightRef = useRef(false);
    const renderingWasActiveRef = useRef(false);

    const closeWindowNow = useCallback(async () => {
        try {
            await runConfirmedExitClose({
                markAllowClose: () => {
                    allowWindowCloseRef.current = true;
                },
                destroyWindow: async () => {
                    const mod = await import("@tauri-apps/api/window");
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
                void action();
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
                ).unwrap()) as any;
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
        void dispatch(loadUiSettings());
        void dispatch(loadRecordingSettings());
    }, [dispatch]);

    useEffect(() => {
        let cancelled = false;

        async function loadAutoBackupSettings() {
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
        if (!autoBackgroundRender) return;
        // 跳过初始加载（prevParamsEpochRef 与当前 epoch 相同时跳过）
        if (prevParamsEpochRef.current === paramsEpoch) return;
        prevParamsEpochRef.current = paramsEpoch;

        // 防抖：延迟 200ms 后再触发，避免连续编辑时频繁启动渲染线程
        const timer = setTimeout(() => {
            void (async () => {
                try {
                    const result = await webApi.startBackgroundRender();
                    if ((result as any)?.skipped) {
                        // 已在渲染中，无需重复启动
                        return;
                    }
                } catch (e) {
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
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅挂载时执行一次
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
                const mod = await import("@tauri-apps/api/window");
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
            } catch {}
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

    // 统一快捷键处理（通过 keybindings 模块管理，用户可自定义）
    const handleKeybindingAction = useCallback(
        (actionId: ActionId) => {
            switch (actionId) {
                case "playback.toggle":
                    if (runtimeRef.current.isPlaying) {
                        void dispatch(stopAudioPlayback());
                    } else {
                        void dispatch(playOriginal());
                    }
                    break;
                case "playback.stop":
                    if (runtimeRef.current.isPlaying) {
                        void dispatch(stopAudioPlayback({ restoreAnchor: true }));
                    } else {
                        void dispatch(playOriginal());
                    }
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
                case "edit.undo":
                    void dispatch(undoRemote());
                    break;
                case "edit.redo":
                    void dispatch(redoRemote());
                    break;
                case "edit.selectAll":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "selectAll" },
                        }),
                    );
                    break;
                case "edit.deselect":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "deselect" },
                        }),
                    );
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
                case "mode.toggle": {
                    const cur = runtimeRef.current.toolMode;
                    if (cur === "select") {
                        dispatch(setToolMode(runtimeRef.current.drawToolMode));
                    } else {
                        dispatch(setToolMode("select"));
                    }
                    break;
                }
                case "mode.selectTool":
                    dispatch(setToolMode("select"));
                    break;
                case "mode.drawTool":
                    dispatch(setToolMode("draw"));
                    break;
                case "mode.lineTool":
                    dispatch(setToolMode("vibrato"));
                    break;
                case "quickSearch.open":
                    setQuickSearchOpen(true);
                    break;
                case "track.add": {
                    const ss = store.getState().session;
                    const parentId = ss.selectedTrackId ?? null;
                    void dispatch(addTrackRemote({ parentTrackId: parentId }));
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
                case "pianoRoll.shiftParamUp":
                case "pianoRoll.shiftParamDown": {
                    const isUp = actionId === "pianoRoll.shiftParamUp";
                    const ss = store.getState().session;
                    const rootTrkId = resolveRootTrackId(ss.tracks, ss.selectedTrackId);
                    if (!rootTrkId) break;
                    const editP = ss.editParam;
                    const rootTrk = ss.tracks.find((tr) => tr.id === rootTrkId);
                    // pitch 参数需要 pitch 分析可用才能操作
                    if (editP === "pitch") {
                        if (!rootTrk?.composeEnabled || rootTrk.pitchAnalysisAlgo === "none") break;
                    }
                    const selClipId = ss.selectedClipId;
                    // 优先使用多选 clip 列表，否则 fallback 到单选
                    const multiIds = ss.multiSelectedClipIds;
                    const clipIds = multiIds.length >= 1 ? multiIds : selClipId ? [selClipId] : [];
                    if (clipIds.length === 0) break;
                    const selClips = ss.clips.filter((c) => clipIds.includes(c.id));
                    if (selClips.length === 0) break;
                    const minSec = Math.min(...selClips.map((c) => c.startSec));
                    const maxSec = Math.max(...selClips.map((c) => c.startSec + c.lengthSec));
                    // 默认 framePeriodMs = 5
                    const fp = 5;
                    const startFrame = Math.max(0, Math.floor((minSec * 1000) / fp));
                    const frameCount = Math.max(
                        1,
                        Math.min(200_000, Math.ceil(((maxSec - minSec) * 1000) / fp)),
                    );
                    void (async () => {
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
                        const step = getParamShiftStep(editP, descriptor);
                        const delta = isUp ? step : -step;
                        const clampNum = (v: number, minV: number, maxV: number) =>
                            Math.min(maxV, Math.max(minV, v));
                        const smoothness = clampNum(Number(ss.edgeSmoothnessPercent) || 0, 0, 100);
                        const maxTransitionFrames = Math.floor(frameCount / 2);
                        const transitionFrames =
                            smoothness > 0 && maxTransitionFrames > 0
                                ? Math.round((smoothness / 100) * maxTransitionFrames)
                                : 0;
                        const halfSpan = transitionFrames > 0 ? transitionFrames / 2 : 0;
                        const extend = Math.max(0, Math.ceil(halfSpan));
                        const extStart = Math.max(0, startFrame - extend);
                        const extCount = frameCount + Math.max(0, startFrame - extStart) + extend;
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
                        const beforeDense = (extPayload.edit ?? []).map((v) => Number(v) || 0);
                        if (beforeDense.length === 0) return;

                        const selEnd = Math.min(beforeDense.length - 1, selOffset + frameCount - 1);
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
                                    const left = Math.max(0, Math.floor(selOffset - halfSpan));
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
                                        const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                                        const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                                        const smoothed = outsideVal + (insideVal - outsideVal) * t;
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
                                        const insideVal = snapshot[insideIdx] ?? editedDense[idx];
                                        const outsideVal = snapshot[outsideIdx] ?? editedDense[idx];
                                        const smoothed = insideVal + (outsideVal - insideVal) * t;
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
                    })();
                    break;
                }
                case "pianoRoll.shiftParamUpSelection":
                case "pianoRoll.shiftParamDownSelection": {
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: {
                                op:
                                    actionId === "pianoRoll.shiftParamUpSelection"
                                        ? "shiftParamUpSelection"
                                        : "shiftParamDownSelection",
                            },
                        }),
                    );
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
                case "edit.pasteTracks":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "pasteTracks" },
                        }),
                    );
                    break;
                // clip.* 操作由 TimelinePanel 的 useKeyboardShortcuts 处理
                default:
                    break;
            }
        },
        [dispatch, handleNewProject, handleOpenProject],
    );

    useKeybindings(handleKeybindingAction);

    useEffect(() => {
        if (!runtimeIsPlaying) return;
        // Keep playhead following backend audio clock.
        // 用 in-flight guard 防止轮询请求堆积；并适度降频以降低 Redux/React 压力。
        // Increase playhead sync frequency to ~30Hz for smoother playhead updates
        const intervalMs = 33;
        const id = window.setInterval(() => {
            // 阻塞式预渲染（target=”original”）阶段后端还未真正进入 playing，
            // 若此时同步会把前端”准备播放”状态误判为停止，导致 stop 锚点丢失。
            // 后台预渲染（target=”background”）是独立线程，播放应正常同步。
            if (rendering.active && rendering.target === "original") return;
            if (playbackSyncInFlightRef.current) return;
            playbackSyncInFlightRef.current = true;
            const p = dispatch(syncPlaybackState()) as unknown as Promise<unknown>;
            p.finally(() => {
                playbackSyncInFlightRef.current = false;
            });
        }, intervalMs);
        return () => window.clearInterval(id);
    }, [dispatch, runtimeIsPlaying, rendering.active, rendering.target]);

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
            if (Number(playbackPositionSec ?? 0) >= endSec - 0.05) {
                void dispatch(stopRecordingFlow());
            }
        }, 100);
        return () => window.clearInterval(id);
    }, [
        dispatch,
        multiSelectedClipIds,
        playbackPositionSec,
        recordingActive,
        recordingSettings.autoStopAtSelectionEnd,
        recordingStartSec,
        selectedClipId,
        sessionClips,
    ]);

    useEffect(() => {
        splitRatioRef.current = splitRatio;
    }, [splitRatio]);

    useEffect(() => {
        if (!isDragging) return;
        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "ns-resize";
        document.body.style.userSelect = "none";
        return () => {
            document.body.style.cursor = prevCursor;
            document.body.style.userSelect = prevSelect;
        };
    }, [isDragging]);

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

    return (
        <Flex
            direction="column"
            className="h-screen w-screen bg-qt-window text-qt-text overflow-hidden font-sans text-sm selection:bg-qt-highlight selection:text-white"
        >
            <Dialog.Root
                open={Boolean(vocalShifterSkippedFilesDialog?.length)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeVocalShifterSkippedFilesDialog());
                    }
                }}
            >
                <Dialog.Content maxWidth="620px">
                    <Dialog.Title>{t("status_error_prefix")}</Dialog.Title>
                    <Dialog.Description>{t("vs_import_skipped_header")}</Dialog.Description>
                    <div className="mt-2 max-h-[240px] overflow-auto rounded border border-qt-border bg-qt-base p-2 text-xs">
                        {(vocalShifterSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" data-tooltip={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                    <Flex justify="end" mt="3">
                        <Button onClick={() => dispatch(closeVocalShifterSkippedFilesDialog())}>
                            {"OK"}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <Dialog.Root
                open={Boolean(reaperSkippedFilesDialog?.length)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeReaperSkippedFilesDialog());
                    }
                }}
            >
                <Dialog.Content maxWidth="620px">
                    <Dialog.Title>{t("status_error_prefix")}</Dialog.Title>
                    <Dialog.Description>{t("reaper_import_skipped_header")}</Dialog.Description>
                    <div className="mt-2 max-h-[240px] overflow-auto rounded border border-qt-border bg-qt-base p-2 text-xs">
                        {(reaperSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" data-tooltip={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                    <Flex justify="end" mt="3">
                        <Button onClick={() => dispatch(closeReaperSkippedFilesDialog())}>
                            {"OK"}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <Dialog.Root
                open={unsavedDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        cancelUnsavedAction();
                    }
                }}
            >
                <Dialog.Content maxWidth="460px">
                    <Dialog.Title>{t("unsaved_changes_title")}</Dialog.Title>
                    <Dialog.Description>
                        {t(
                            unsavedDialog.mode === "exit"
                                ? "unsaved_changes_exit_desc"
                                : "unsaved_changes_switch_desc",
                        )}
                    </Dialog.Description>
                    <Flex justify="end" gap="2" mt="4">
                        <Button variant="soft" color="gray" onClick={cancelUnsavedAction}>
                            {t("progress_cancel")}
                        </Button>
                        <Button variant="soft" color="gray" onClick={discardUnsavedAndContinue}>
                            {t("unsaved_changes_discard")}
                        </Button>
                        <Button onClick={saveUnsavedAndContinue}>{t("menu_save_project")}</Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            {/* Project file version newer than this build — ask before attempting load */}
            <Dialog.Root
                open={projectVersionDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        setProjectVersionDialog((current) => ({ ...current, open: false }));
                    }
                }}
            >
                <Dialog.Content maxWidth="480px">
                    <Dialog.Title>{t("project_version_too_new_title")}</Dialog.Title>
                    <Dialog.Description>
                        {t("project_version_too_new_desc")
                            .replace(
                                "{fileVersion}",
                                String(projectVersionDialog.fileVersion || "?"),
                            )
                            .replace(
                                "{currentVersion}",
                                String(projectVersionDialog.currentVersion || "?"),
                            )}
                    </Dialog.Description>
                    <Flex justify="end" gap="2" mt="4">
                        <Button
                            variant="soft"
                            color="gray"
                            onClick={cancelContinueLoadingNewerProject}
                        >
                            {t("progress_cancel")}
                        </Button>
                        <Button color="amber" onClick={confirmContinueLoadingNewerProject}>
                            {t("project_version_too_new_continue")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            {/* 保存/另存为目标已存在版本不一致的工程文件 — 覆盖前询问用户 */}
            <Dialog.Root
                open={Boolean(saveVersionConflictDialog)}
                onOpenChange={(open) => {
                    if (!open) {
                        dispatch(closeSaveVersionConflictDialog());
                    }
                }}
            >
                <Dialog.Content maxWidth="520px">
                    <Dialog.Title>{t("save_version_conflict_title")}</Dialog.Title>
                    <Dialog.Description>
                        {saveVersionConflictDialog?.existingIsNewer
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
                                  )}
                    </Dialog.Description>
                    <Flex justify="end" gap="2" mt="4">
                        <Button variant="soft" color="gray" onClick={cancelSaveVersionConflict}>
                            {t("progress_cancel")}
                        </Button>
                        <Button variant="soft" onClick={saveAsFromVersionConflict}>
                            {t("save_version_conflict_save_as")}
                        </Button>
                        <Button color="red" onClick={continueForceSave}>
                            {t("save_version_conflict_continue")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            {/* Recapture missing media dialog — grid: file status / file / processing status / ignore / action */}
            <Dialog.Root
                open={sourceFileChangedDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        closeSourceFileChangedDialog();
                    }
                }}
            >
                <Dialog.Content maxWidth="860px">
                    <Dialog.Title>{t("recapture_missing_media_title")}</Dialog.Title>
                    <Dialog.Description>{t("recapture_missing_media_desc")}</Dialog.Description>

                    <Flex justify="between" align="center" gap="2" mt="2">
                        <Flex gap="1" align="center" className="shrink-0">
                            <span className="shrink-0 text-[10px] text-qt-text-muted">
                                {t("recapture_missing_media_search_mode_label")}
                            </span>
                            <WheelSelect
                                className="h-6 shrink-0 rounded border border-qt-border bg-qt-base px-1 py-0.5 text-[10px] text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30"
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
                            <Text size="1" color="gray" className="min-w-0 truncate">
                                {t("recapture_missing_media_search_result_summary")
                                    .replace("{total}", String(sourceFileSearchMatchTotal))
                                    .replace("{exact}", String(sourceFileSearchExactTotal))
                                    .replace("{selected}", String(sourceFileSelectedApplyTotal))}
                            </Text>
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

                    <div className="mt-2 max-h-[320px] overflow-auto rounded border border-qt-border bg-qt-base p-1">
                        <div className="grid grid-cols-[64px_minmax(0,1fr)_76px_64px_84px_48px] items-center gap-2 border-b border-qt-border px-1 py-1 text-[10px] font-semibold text-qt-text-muted">
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
                                    ? "border border-gray-500/25 bg-gray-500/10 text-gray-600"
                                    : item.action === "reloaded" || item.action === "replaced"
                                      ? "border border-green-500/25 bg-green-500/10 text-green-600"
                                      : item.action === "failed"
                                        ? "border border-red-500/25 bg-red-500/10 text-red-600"
                                        : item.action === "processing"
                                          ? "border border-blue-500/25 bg-blue-500/10 text-blue-600"
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
                                    className="grid grid-cols-[64px_minmax(0,1fr)_76px_64px_84px_48px] items-center gap-2 border-b border-qt-border px-1 py-1.5 text-xs last:border-b-0"
                                >
                                    <div
                                        className={`shrink-0 whitespace-nowrap rounded px-1 py-0.5 text-center text-[10px] font-semibold leading-none ${
                                            item.change === "deleted"
                                                ? "border border-red-500/25 bg-red-500/10 text-red-600"
                                                : "border border-amber-500/25 bg-amber-500/10 text-amber-600"
                                        }`}
                                    >
                                        {item.change === "deleted"
                                            ? t("recapture_missing_media_status_deleted")
                                            : t("recapture_missing_media_status_modified")}
                                    </div>
                                    <div className="min-w-0">
                                        <div className="truncate" data-tooltip={item.source_path}>
                                            <span className="font-medium">{item.clip_name}</span>
                                            <span className="text-gray-500">
                                                {" "}
                                                — {item.source_path}
                                            </span>
                                        </div>
                                        {item.reloadedPath && (
                                            <div
                                                className="mt-0.5 flex items-center gap-1 truncate text-[10px] text-green-600"
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
                                                        className="min-w-0 flex-1 rounded border border-qt-border bg-qt-base px-1 py-0.5 text-[10px] text-qt-text focus:outline-none focus:ring-1 focus:ring-qt-highlight/30"
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
                                                                    ? `✓ ${t("recapture_missing_media_match_exact")} · `
                                                                    : ""}
                                                                {candidate.path}
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
                                                <div className="mt-1 truncate text-[10px] text-qt-text-muted">
                                                    {t("recapture_missing_media_search_no_matches")}
                                                </div>
                                            )}
                                    </div>
                                    <div
                                        className={`shrink-0 whitespace-nowrap rounded px-1.5 py-0.5 text-center text-[10px] font-semibold leading-none ${statusBadgeClass}`}
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
                        <Text size="1" color="gray">
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
                        </Text>
                        <Flex gap="2" align="center" className="shrink-0">
                            <Button
                                disabled={
                                    sourceFileAnyProcessing ||
                                    sourceFileChangedDialog.changes.some(
                                        (item) => item.action === "pending",
                                    )
                                }
                                onClick={closeSourceFileChangedDialog}
                            >
                                {t("ok")}
                            </Button>
                        </Flex>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

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

            {/* Main Content Area: Splitter + optional right-side panels */}
            <Flex className="flex-1 min-h-0">
                {/* Left: Timeline / PianoRoll vertical splitter */}
                <div ref={containerRef} className="flex-1 min-w-0 min-h-0 flex flex-col">
                    {/* Top: Timeline / Tracks */}
                    <Box
                        className="min-h-[200px] border-b border-qt-border relative bg-qt-base"
                        style={{ flexGrow: splitRatio, flexBasis: 0 }}
                    >
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
                            onImportTempoMapTimeSignatureChange={
                                handleImportTempoMapTimeSignatureChange
                            }
                            importTempoMapKeySignature={importTempoMapKeySignature}
                            onImportTempoMapKeySignatureChange={
                                handleImportTempoMapKeySignatureChange
                            }
                            midiDialogSource={midiDialogSource}
                            onMidiDialogSourceChange={setMidiDialogSource}
                            importTargetMenu={midiImportTargetMenu}
                            onImportTargetMenuChange={handleImportTargetMenuChange}
                            importTargetDragDrop={midiImportTargetDragDrop}
                            onImportTargetDragDropChange={handleImportTargetDragDropChange}
                        />
                    </Box>

                    {/* Splitter */}
                    <div
                        className="h-2 bg-qt-window border-y border-qt-border cursor-ns-resize shrink-0"
                        onPointerDown={splitter.startDrag}
                        role="separator"
                        aria-orientation="horizontal"
                        aria-label={t("aria_resize_panels")}
                    />

                    {/* Bottom: Parameter / Piano Roll */}
                    <Box
                        className="min-h-[150px] relative bg-qt-base"
                        style={{ flexGrow: 1 - splitRatio, flexBasis: 0 }}
                    >
                        <PianoRollPanel />
                    </Box>
                </div>

                {(fileBrowserVisible || notebookVisible) && (
                    <Flex className="shrink-0 min-h-0 border-l border-qt-border bg-qt-window">
                        {fileBrowserVisible ? (
                            <div className="w-[280px] shrink-0 border-r border-qt-border bg-qt-window flex flex-col">
                                <FileBrowserPanel />
                            </div>
                        ) : null}
                        {notebookVisible ? (
                            <div className="w-[320px] shrink-0 bg-qt-window flex flex-col">
                                <NotebookPanel />
                            </div>
                        ) : null}
                    </Flex>
                )}
            </Flex>

            {/* Quick Search Popup */}
            <QuickSearchPopup open={quickSearchOpen} onClose={() => setQuickSearchOpen(false)} />

            {/* Status Bar */}
            <Flex
                align="center"
                justify="between"
                className="h-6 bg-qt-window border-t border-qt-border px-1 select-none gap-2"
            >
                <Flex align="center" gap="1" className="truncate min-w-0">
                    {stretching.active ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {t("status_stretching")}
                            {stretching.clipName ? ` "${stretching.clipName}"` : ""}
                        </span>
                    ) : null}
                    {waveformAnalysis.active ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {"Analyzing waveform"}
                            {waveformAnalysis.sourcePath ? ` "${waveformAnalysis.sourcePath}"` : ""}
                            {waveformAnalysis.progress != null
                                ? ` ${Math.round(waveformAnalysis.progress * 100)}%`
                                : ""}
                        </span>
                    ) : null}
                    {pitchAnalysisText ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {pitchAnalysisText}
                        </span>
                    ) : null}
                    {pianoRollStatus.dataLoading ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {t("loading")}
                        </span>
                    ) : null}
                    {pianoRollStatus.asyncRefreshActive ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {t("refreshing_pitch_data") || "Refreshing pitch data"}
                            {pianoRollStatus.asyncRefreshProgress > 0
                                ? ` ${Math.round(pianoRollStatus.asyncRefreshProgress)}%`
                                : ""}
                        </span>
                    ) : null}
                    {rendering.active ? (
                        <span
                            className="shrink-0 rounded px-1 py-0 text-xs font-medium"
                            style={{
                                background: "var(--accent-3)",
                                color: "var(--accent-11)",
                                fontSize: "11px",
                                lineHeight: "16px",
                            }}
                        >
                            {t("rendering")}
                            {rendering.progress != null
                                ? ` ${Math.round(rendering.progress * 100)}%`
                                : ""}
                        </span>
                    ) : null}
                    <Text size="1" color={error ? "red" : "gray"} className="truncate">
                        {errorText}
                    </Text>
                </Flex>
            </Flex>
        </Flex>
    );
}

function App() {
    return (
        <PitchAnalysisProvider>
            <PianoRollStatusProvider>
                <AppInner />
            </PianoRollStatusProvider>
        </PitchAnalysisProvider>
    );
}

export default App;
