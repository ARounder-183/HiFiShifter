import React, { useCallback, useEffect, useMemo, useState } from "react";
import { DropdownMenu, Flex } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import { store, type RootState } from "../../app/store";
import {
    openReaperFromDialog,
    openVocalShifterFromDialog,
    addTrackRemote,
    removeTrackRemote,
    duplicateTrackRemote,
    refreshRuntime,
    clearWaveformCacheRemote,
    persistUiSettings,
    undoRemote,
    redoRemote,
    saveProjectRemote,
    saveProjectAsRemote,
    setDefaultHifiganMelStretch,
    setDefaultStretchAlgorithm,
    setOrtEp,
    setOrtDeviceId,
    setPrimaryTimeUnit,
    setSecondaryTimeUnit,
    setSearchSettingsDialogOpen,
    toggleAutoBackgroundRender,
    toggleShowAllTakes,
    toggleSyncEditsAcrossTakes,
    toggleClipboardPreview,
    toggleParamValuePopup,
    toggleTempoMapVisible,
    setProjectStretchSettingsRemote,
} from "../../features/session/sessionSlice";
import type { TimeUnit } from "../../features/session/sessionTypes";
import { TIME_UNITS, TIME_UNIT_CHOICES } from "./timeline/timeFormat";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { SnapGridSettingsDialog } from "./SnapGridSettingsDialog";
import { DockLayoutMenus, DockLayoutDialogs } from "../dock/DockLayoutMenu";
import { scaleChangesInRange, scaleLikeEquals } from "../../utils/tempoMap";
import type { ScaleLike } from "../../utils/musicalScales";
import {
    getPianoRollSelection,
    subscribePianoRollSelection,
} from "../../utils/pianoRollSelectionBus";
import { computeInsertBelowPlacement } from "../../features/session/trackUtils";

import {
    importAudioAtPosition,
    importAudioFromDialog,
    importMultipleAudioAtPosition,
} from "../../features/session/thunks/importThunks";
import type { MediaAudioStream } from "../../services/api/fileBrowser";
import { useAppTheme } from "../../theme/AppThemeProvider";
import { GlobeIcon } from "@radix-ui/react-icons";
import {
    selectMergedKeybindings,
    formatKeybinding,
    isNoneBinding,
} from "../../features/keybindings/keybindingsSlice";
import type { ActionId } from "../../features/keybindings/types";
import {
    resolveCopyCutRoute,
    resolveEditOpRoute,
    resolvePasteRoute,
} from "../../features/keybindings/focusRouting";
import { getActiveSurface } from "../../features/uiFocus/focusSurface";
import { webApi } from "../../services/webviewApi";
import { KeybindingsDialog } from "./KeybindingsDialog";
import {
    TransposeCentsDialog,
    TransposeDegreesDialog,
    SetPitchDialog,
    AverageDialog,
    SmoothDialog,
    QuantizeDialog,
    MeanQuantizeDialog,
} from "../editDialogs/EditDialogs";
import { SCALE_LABELS } from "../../utils/musicalScales";
import { ExportAudioDialog } from "./ExportAudioDialog";
import { AutoBackupDialog } from "./AutoBackupDialog";
import { RenderCacheDialog } from "./RenderCacheDialog";
import { VibratoDialog } from "./VibratoDialog";
import { ChannelImportDialog } from "./ChannelImportDialog";
import { PenInputDialog } from "./PenInputDialog";
import { SearchSettingsDialog } from "./SearchSettingsDialog";
import { RecordingSettingsDialog } from "./RecordingSettingsDialog";
import { BenchmarkDialog } from "./BenchmarkDialog";
import { AboutDialog } from "./AboutDialog";

import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "./pianoRoll/childPitchOffsetParams";
import { isDynParam } from "./pianoRoll/paramRanges";
import type { AutoBackupSettings } from "../../services/api/project";
import { AppDialog } from "../../ui/Dialog";
import { AppChoiceList } from "../../ui/ChoiceList";
import { togglePanelVisible } from "../../features/dock/dockApi";
import { PANEL_APPEARANCE } from "../dock/registerBuiltinPanels";
import { AppBusy, AppConfirmDialog, AppNoticeDialog } from "../../ui";
// import type { VibratoParams } from "../editDialogs/EditDialogs"; // 已移除无效导入

interface MenuBarProps {
    onNewProject: () => void;
    onOpenProject: () => void;
    onOpenRecentProject: (projectPath: string) => void;
    onRecaptureMissingMedia: () => void;
    onExit: () => void;
    onImportMidiFromMenu: () => void;
    onImportProject: () => void;
    autoBackupSettings: AutoBackupSettings;
    onAutoBackupSettingsSaved: (settings: AutoBackupSettings) => void;
    autoReloadModifiedMedia: boolean;
    onAutoReloadModifiedMediaChange: (value: boolean) => void;
    /** 为新的音频块启用循环（Loop / 循环源，默认开启）。 */
    loopNewClips: boolean;
    onLoopNewClipsChange: (value: boolean) => void;
}

function timeUnitLabelKey(unit: TimeUnit): string {
    switch (unit) {
        case "barBeats":
            return "time_unit_bar_beats";
        case "barDivisions":
            return "time_unit_bar_divisions";
        case "seconds":
            return "time_unit_seconds";
        case "clock":
            return "time_unit_clock";
    }
}

/** MenuBar 实际消费的 session 字段子集（配合 shallowEqual 阻断播放轮询的重渲染）。
 *  新增消费字段时必须同步补充到这里。 */
const selectMenuBarSession = (state: RootState) => {
    const session = state.session;
    return {
        autoBackgroundRender: session.autoBackgroundRender,
        defaultHifiganMelStretch: session.defaultHifiganMelStretch,
        defaultStretchAlgorithm: session.defaultStretchAlgorithm,
        edgeSmoothnessPercent: session.edgeSmoothnessPercent,
        editParam: session.editParam,
        historyRedoDepth: session.historyRedoDepth,
        historyUndoDepth: session.historyUndoDepth,
        multiSelectedClipIds: session.multiSelectedClipIds,
        ortDeviceId: session.ortDeviceId,
        ortEp: session.ortEp,
        paramSelectionActive: session.paramSelectionActive,
        pitchSnapToleranceCents: session.pitchSnapToleranceCents,
        playheadSec: session.playheadSec,
        primaryTimeUnit: session.primaryTimeUnit,
        project: session.project,
        projectSec: session.projectSec,
        searchSettingsDialogOpen: session.searchSettingsDialogOpen,
        secondaryTimeUnit: session.secondaryTimeUnit,
        selectedClipId: session.selectedClipId,
        selectedTrackId: session.selectedTrackId,
        selectionContext: session.selectionContext,
        showAllTakes: session.showAllTakes,
        showClipboardPreview: session.showClipboardPreview,
        showParamValuePopup: session.showParamValuePopup,
        syncEditsAcrossTakes: session.syncEditsAcrossTakes,
        tempoMap: session.tempoMap,
        tempoMapVisible: session.tempoMapVisible,
        toolMode: session.toolMode,
        tracks: session.tracks,
    };
};

export const MenuBar: React.FC<MenuBarProps> = ({
    onNewProject,
    onOpenProject,
    onOpenRecentProject,
    onRecaptureMissingMedia,
    onExit,
    onImportMidiFromMenu,
    onImportProject,
    autoBackupSettings,
    onAutoBackupSettingsSaved,
    autoReloadModifiedMedia,
    onAutoReloadModifiedMediaChange,
    loopNewClips,
    onLoopNewClipsChange,
}) => {
    const { t, tf, setLocale, plural } = useI18n();
    const dispatch = useAppDispatch();
    // 只选取本组件实际消费的字段子集并以 shallowEqual 比较：播放期间
    // runtime.playbackPositionSec 每 ~33ms 变一次，整片 session 的对象引用
    // 随之失效，若直接订阅 state.session，全部菜单定义会以 ≥30Hz 重建。
    const s = useAppSelector(selectMenuBarSession, shallowEqual);
    const gpuBackend = useAppSelector((state: RootState) => state.session.runtime.gpuBackend);
    const theme = useAppTheme();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const [kbDialogOpen, setKbDialogOpen] = useState(false);
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = useState(false);
    const [snapSettingsOpen, setSnapSettingsOpen] = useState(false);
    const [exportDialogOpen, setExportDialogOpen] = useState(false);
    const [autoBackupDialogOpen, setAutoBackupDialogOpen] = useState(false);
    const [recordingDialogOpen, setRecordingDialogOpen] = useState(false);
    const [benchmarkDialogOpen, setBenchmarkDialogOpen] = useState(false);
    // 导出诊断信息（含基准测试，耗时较长）进行中：菜单项禁用 + 模态提示，
    // 避免用户以为点击没生效而反复触发。
    const [diagnosticsExporting, setDiagnosticsExporting] = useState(false);
    const [aboutDialogOpen, setAboutDialogOpen] = useState(false);
    // 纯通知对话框（取代此前的 window.alert）：错误报告用标题 + 正文两段。
    const [notice, setNotice] = useState<{ title: string; message: string } | null>(null);
    const [renderCacheDialogOpen, setRenderCacheDialogOpen] = useState(false);
    /** 清空波形缓存确认框。缓存重建代价高，且下拉菜单关闭即卸载，故由常驻菜单栏托管。 */
    const [waveformCacheConfirmOpen, setWaveformCacheConfirmOpen] = useState(false);
    const [channelImportDialogOpen, setChannelImportDialogOpen] = useState(false);
    const [penInputDialogOpen, setPenInputDialogOpen] = useState(false);
    const [dmlAdapters, setDmlAdapters] = useState<
        { deviceId: number; name: string; memoryMb: number }[]
    >([]);

    // Fetch DML adapters on mount for GPU device selector
    useEffect(() => {
        import("../../services/api/core")
            .then(({ coreApi }) => coreApi.getDmlAdapters())
            .then((result) => {
                if (result.adapters && result.adapters.length > 0) {
                    setDmlAdapters(
                        result.adapters.map((a) => ({
                            deviceId: a.deviceId,
                            name: a.name,
                            memoryMb: a.dedicatedVideoMemoryMb,
                        })),
                    );
                }
            })
            .catch(() => {
                // DML adapters unavailable — device selector won't show
            });
    }, []);

    // Edit dialog states
    const [transposeCentsOpen, setTransposeCentsOpen] = useState(false);
    const [transposeDegreesOpen, setTransposeDegreesOpen] = useState(false);
    const [setPitchOpen, setSetPitchOpen] = useState(false);
    const [averageOpen, setAverageOpen] = useState(false);
    const [smoothOpen, setSmoothOpen] = useState(false);
    const [vibratoDialogOpen, setVibratoDialogOpen] = useState(false);
    const [quantizeOpen, setQuantizeOpen] = useState(false);
    const [meanQuantizeOpen, setMeanQuantizeOpen] = useState(false);
    const [menuImportMode, setMenuImportMode] = useState<{
        audioPaths: string[];
        trackId: string | null;
        startSec: number;
    } | null>(null);
    const [mediaStreamImport, setMediaStreamImport] = useState<{
        path: string;
        streams: MediaAudioStream[];
        trackId: string | null;
        startSec: number;
    } | null>(null);

    const isPitchParam = s.editParam === "pitch";
    const isChildCentsParam = isChildPitchOffsetCentsParam(s.editParam);
    const isChildDegreesParam = isChildPitchOffsetDegreesParam(s.editParam);
    const setToDefaultValue =
        s.editParam === "pitch"
            ? 60
            : isChildCentsParam || isChildDegreesParam
              ? 0
              : s.editParam === "volume" || isDynParam(s.editParam)
                ? 1
                : 0;
    const setToValueLabel = s.editParam === "pitch" ? tf("dlg_midi_note") : tf("dlg_value");
    const quantizeDefaultUnit = (() => {
        if (isChildCentsParam) return 100;
        if (isChildDegreesParam) return 1;
        switch (s.editParam) {
            case "volume":
            case "dyn":
            case "dyn_edit":
                return 0.05;
            case "formant_shift_cents":
                return 100;
            case "breath_gain":
            case "hifigan_tension":
                return 0.05;
            case "pan":
                return 0.1;
            case "breathiness":
                return 250;
            default:
                return 1;
        }
    })();
    const projectScaleLabel =
        s.project.useCustomScale && s.project.customScale
            ? `${tf("project_scale_prefix")} (${tf("custom_scale_short")})`
            : `${tf("project_scale_prefix")} (${SCALE_LABELS[s.project.baseScale]})`;

    // ── “工程音阶”选项受 Tempo Map 影响的提示 ─────────────────────────
    // 读取参数编辑器当前选区（帧范围）判断是否跨过音阶变化点；
    // 无选区时按整个工程判断（存在音阶变化点即提示）。
    const [selectionVersion, setSelectionVersion] = useState(0);
    useEffect(
        () =>
            subscribePianoRollSelection(() => {
                setSelectionVersion((v) => v + 1);
            }),
        [],
    );
    const tempoMapScaleHint = useMemo(() => {
        if (!s.tempoMap) return null;
        const sel = getPianoRollSelection();
        const startSec = sel ? (sel.startFrame * sel.framePeriodMs) / 1000 : 0;
        const endSec = sel
            ? ((sel.startFrame + Math.max(0, sel.frameCount)) * sel.framePeriodMs) / 1000
            : Math.max(1, s.projectSec);
        const projectScale: ScaleLike | null =
            s.project.useCustomScale && s.project.customScale
                ? s.project.customScale.notes
                : s.project.baseScale;
        // 仅当范围内存在“与工程音阶不同”的音阶变化（或管辖范围起点的
        // 变化与工程音阶不同）时才提示 —— 变化点音阶等于工程音阶时
        // 选区并未真正受到影响。
        const changes = scaleChangesInRange(s.tempoMap, startSec, endSec);
        if (changes.length === 0) return null;
        if (changes.every((c) => scaleLikeEquals(c.scale, projectScale))) {
            return null;
        }
        return tf("project_scale_tempo_map_hint");
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [s.tempoMap, s.projectSec, s.project, tf, selectionVersion]);
    const projectScaleLabelWithHint = tempoMapScaleHint
        ? `${projectScaleLabel} ${tempoMapScaleHint}`
        : projectScaleLabel;
    const effectiveProjectStretchAlgorithm =
        s.project.stretchAlgorithmOverride ?? s.defaultStretchAlgorithm;
    const effectiveProjectHifiganMelStretch =
        s.project.hifiganMelStretchOverride ?? s.defaultHifiganMelStretch;

    const stretchAlgorithmLabel = (value: "linear" | "signalsmith" | "soundtouch") => {
        switch (value) {
            case "linear":
                return tf("stretch_option_linear");
            case "signalsmith":
                return tf("stretch_option_signalsmith");
            case "soundtouch":
            default:
                return tf("stretch_option_soundtouch");
        }
    };

    const withCheck = (active: boolean, label: string) => `${active ? "●" : "○"} ${label}`;

    const resolveScaleToken = (scaleValue: string) =>
        scaleValue === "__project__" ? "__project__" : scaleValue;

    // Listen for context menu → open dialog requests
    useEffect(() => {
        const handler = (e: Event) => {
            const dialog = (e as CustomEvent).detail?.dialog as string;
            switch (dialog) {
                case "transposeCents":
                    setTransposeCentsOpen(true);
                    break;
                case "transposeDegrees":
                    setTransposeDegreesOpen(true);
                    break;
                case "setPitch":
                    setSetPitchOpen(true);
                    break;
                case "average":
                    setAverageOpen(true);
                    break;
                case "smooth":
                    setSmoothOpen(true);
                    break;
                case "quantize":
                    setQuantizeOpen(true);
                    break;
                case "meanQuantize":
                    setMeanQuantizeOpen(true);
                    break;
                case "exportAudio":
                    setExportDialogOpen(true);
                    break;
            }
        };
        window.addEventListener("hifi:openEditDialog", handler);
        return () => window.removeEventListener("hifi:openEditDialog", handler);
    }, []);

    /** 获取某个操作的快捷键显示文本（"None" 绑定时返回空字符串，不显示） */
    function shortcutLabel(actionId: ActionId): string {
        const kb = keybindings[actionId];
        if (!kb || isNoneBinding(kb)) return "";
        return formatKeybinding(kb, "");
    }

    /**
     * 派发编辑操作事件：与键盘快捷键共用同一裁决（focusRouting）。
     * copy/cut 按"当前选中了什么"路由（resolveCopyCutRoute，双选区并存时
     * 按 selectionContext 仲裁）；paste 按剪贴板载荷类型路由
     * （last-copy-wins，resolvePasteRoute），槽位为空/外来数据时才退回活动
     * 表面兜底。未收录的操作（参数编辑器专有：对话框确认、pasteVocalShifter
     * 等）落回参数编辑器通道，与既有行为一致。
     */
    const dispatchEditOp = useCallback(
        (op: string, data?: Record<string, unknown>) => {
            if (op === "copy" || op === "cut") {
                const channel = resolveCopyCutRoute({
                    surface: getActiveSurface(),
                    clipSelectionActive: s.multiSelectedClipIds.length > 0 || !!s.selectedClipId,
                    paramSelectionActive: s.paramSelectionActive,
                    selectionContext: s.selectionContext,
                });
                if (channel) {
                    window.dispatchEvent(new CustomEvent(channel, { detail: { op, ...data } }));
                }
                return;
            }
            if (op === "paste") {
                void (async () => {
                    let kind: string | null = null;
                    try {
                        kind = (await webApi.clipboardKind()).kind ?? null;
                    } catch {
                        // 探测失败不阻塞粘贴。
                    }
                    const channel = resolvePasteRoute(kind, getActiveSurface());
                    if (channel) {
                        window.dispatchEvent(new CustomEvent(channel, { detail: { op, ...data } }));
                    }
                })();
                return;
            }
            const channel = resolveEditOpRoute(getActiveSurface(), op, s.toolMode) ?? "hifi:editOp";
            window.dispatchEvent(new CustomEvent(channel, { detail: { op, ...data } }));
        },
        [s],
    );

    const handleImportAudioFromMenu = useCallback(async () => {
        try {
            const res = (await dispatch(importAudioFromDialog()).unwrap()) as {
                canceled?: boolean;
                requiresModeChoice?: boolean;
                requiresStreamChoice?: boolean;
                audioPaths?: string[];
                mediaAudioStreams?: MediaAudioStream[];
                path?: string;
                trackId?: string | null;
                startSec?: number;
            };
            if (res?.canceled) {
                return;
            }
            if (res?.requiresStreamChoice && res.path) {
                setMediaStreamImport({
                    path: res.path,
                    streams: res.mediaAudioStreams ?? [],
                    trackId: res.trackId ?? s.selectedTrackId ?? null,
                    startSec:
                        typeof res.startSec === "number" ? res.startSec : (s.playheadSec ?? 0),
                });
                return;
            }
            if (!res?.requiresModeChoice) {
                return;
            }
            if (!Array.isArray(res.audioPaths) || res.audioPaths.length <= 1) {
                return;
            }
            setMenuImportMode({
                audioPaths: res.audioPaths,
                trackId: res.trackId ?? s.selectedTrackId ?? null,
                startSec: typeof res.startSec === "number" ? res.startSec : (s.playheadSec ?? 0),
            });
        } catch {
            // Error state is already handled by session thunk reducers.
        }
    }, [dispatch, s.playheadSec, s.selectedTrackId]);

    const handleImportMidiFromMenu = useCallback(() => {
        onImportMidiFromMenu();
    }, [onImportMidiFromMenu]);

    /**
     * 导出诊断信息：先弹原生保存对话框选路径，再在后台打包并跑基准测试（较慢，
     * 约 20–60 秒）。期间用模态提示「仍在进行」并禁用菜单项，避免用户误以为点击
     * 没有反应而反复触发。
     */
    const handleExportDiagnostics = useCallback(async () => {
        const { pickDiagnosticsOutputPath, exportDiagnostics } =
            await import("../../services/api/diagnostics");
        try {
            const pick = await pickDiagnosticsOutputPath();
            if (!pick?.ok || !pick.path) return; // 用户取消
            setDiagnosticsExporting(true);
            const res = await exportDiagnostics(pick.path);
            if (!res.ok) {
                setNotice({
                    title: tf("status_error_prefix"),
                    message: res.error || tf("menu_export_diagnostics_failed"),
                });
            }
        } catch (e) {
            setNotice({ title: tf("status_error_prefix"), message: String(e) });
        } finally {
            setDiagnosticsExporting(false);
        }
    }, [tf]);

    // 快捷键「导入媒体文件」→ 复用文件菜单的导入流程（多文件/多音轨选择）。
    useEffect(() => {
        const handler = () => {
            void handleImportAudioFromMenu();
        };
        window.addEventListener("hifi:importMediaFromMenu", handler);
        return () => window.removeEventListener("hifi:importMediaFromMenu", handler);
    }, [handleImportAudioFromMenu]);

    return (
        <Flex
            align="center"
            className="h-qt-bar-main bg-qt-panel border-b border-qt-border px-1 select-none z-50 flex-nowrap gap-1 overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/**
             * Note: @radix-ui/themes DropdownMenu.Trigger does not support asChild.
             * Use Trigger as the actual button element to avoid nesting <button>.
             */}
            {/* File Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_file")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={onNewProject}>
                        {t("menu_new_project")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.new")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={onOpenProject}>
                        {t("menu_open_project")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.open")}
                        </div>
                    </DropdownMenu.Item>

                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {t("menu_recent_projects")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {s.project.recent.length ? (
                                s.project.recent.slice(0, 12).map((p) => (
                                    <DropdownMenu.Item
                                        key={p}
                                        onSelect={() => onOpenRecentProject(p)}
                                    >
                                        {p}
                                    </DropdownMenu.Item>
                                ))
                            ) : (
                                <DropdownMenu.Item disabled>
                                    {t("menu_recent_empty")}
                                </DropdownMenu.Item>
                            )}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    <DropdownMenu.Item onSelect={onRecaptureMissingMedia}>
                        {t("menu_recapture_missing_media")}
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Item onSelect={() => void dispatch(saveProjectRemote())}>
                        {t("menu_save_project")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.save")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(saveProjectAsRemote())}>
                        {t("menu_save_project_as")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.saveAs")}
                        </div>
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Item
                        onSelect={() => {
                            void handleImportAudioFromMenu();
                        }}
                    >
                        {t("menu_import_media")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.importMedia")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => {
                            void handleImportMidiFromMenu();
                        }}
                    >
                        {t("menu_import_midi")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.importMidi")}
                        </div>
                    </DropdownMenu.Item>
                    {/* 导入外部工程（HiFiShifter / Reaper / VocalShifter）*/}
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tf("menu_import_external_project")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Item onSelect={onImportProject}>
                                {t("menu_import_hifishifter")}
                                <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                                    {shortcutLabel("project.importHifishifter")}
                                </div>
                            </DropdownMenu.Item>
                            <DropdownMenu.Item
                                onSelect={() => void dispatch(openReaperFromDialog())}
                            >
                                {t("menu_import_reaper")}
                                <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                                    {shortcutLabel("project.importReaper")}
                                </div>
                            </DropdownMenu.Item>
                            <DropdownMenu.Item
                                onSelect={() => void dispatch(openVocalShifterFromDialog())}
                            >
                                {t("menu_import_vocalshifter")}
                                <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                                    {shortcutLabel("project.importVocalShifter")}
                                </div>
                            </DropdownMenu.Item>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                    <DropdownMenu.Item onSelect={() => setExportDialogOpen(true)}>
                        {t("menu_export_audio")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("project.export")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setAutoBackupDialogOpen(true)}>
                        {tf("menu_auto_backup")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setRecordingDialogOpen(true)}>
                        {tf("menu_recording_settings")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={onExit} color="red">
                        {t("menu_exit")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Edit Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_edit")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    {/* 无可撤销/可重做状态时置灰（不发请求、不刷新界面）：
                        深度镜像由后端 history_state 广播实时维护。 */}
                    <DropdownMenu.Item
                        disabled={s.historyUndoDepth <= 0}
                        onSelect={() => void dispatch(undoRemote())}
                    >
                        {t("menu_undo")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.undo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        disabled={s.historyRedoDepth <= 0}
                        onSelect={() => void dispatch(redoRemote())}
                    >
                        {t("menu_redo")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.redo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 剪贴板：剪切 / 复制 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("cut")}>
                        {tf("menu_cut")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("clip.cut")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("copy")}>
                        {tf("menu_copy")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.copy")}
                        </div>
                    </DropdownMenu.Item>
                    {/* 剪贴板：粘贴 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("paste")}>
                        {tf("menu_paste")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.paste")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteTracks")}>
                        {t("menu_paste_new_tracks")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteTracks")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 外部剪贴板交换 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteVocalShifter")}>
                        {t("menu_paste_vocalshifter_clipboard")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteVocalShifter")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 选择 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("selectAll")}>
                        {tf("menu_select_all")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.selectAll")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("deselect")}>
                        {tf("menu_deselect")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.deselect")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 音频块范围 → 参数编辑器选区（批量入口）。标签直接复用
                        快捷键设置里的动作名：菜单与设置面板共用同一份文案，
                        避免两处翻译漂移。 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("addClipsToParamSelection")}>
                        {tf("kb_edit_add_clips_to_param_selection")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.addClipsToParamSelection")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => dispatchEditOp("removeClipsFromParamSelection")}
                    >
                        {tf("kb_edit_remove_clips_from_param_selection")}{" "}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("edit.removeClipsFromParamSelection")}
                        </div>
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Track Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_track")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={() => {
                            // 新建轨道继承选中轨道的层级，并紧跟在选中轨道下方插入。
                            const placement = computeInsertBelowPlacement(
                                s.tracks,
                                s.selectedTrackId,
                            );
                            dispatch(
                                addTrackRemote({
                                    parentTrackId: placement.parentTrackId,
                                    index: placement.index,
                                }),
                            );
                        }}
                    >
                        {t("track_add")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("track.add")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        disabled={!s.selectedTrackId}
                        onSelect={() =>
                            s.selectedTrackId && dispatch(duplicateTrackRemote(s.selectedTrackId))
                        }
                    >
                        {tf("menu_clone_selected_track")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("track.clone")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        disabled={
                            !s.selectedTrackId ||
                            // 只剩最后一个根轨道时，禁止删除根轨道
                            (s.tracks.filter((t) => !t.parentId).length <= 1 &&
                                !s.tracks.find((t) => t.id === s.selectedTrackId)?.parentId)
                        }
                        onSelect={() =>
                            s.selectedTrackId && dispatch(removeTrackRemote(s.selectedTrackId))
                        }
                    >
                        {t("track_remove_selected")}
                        <div className="ml-auto pl-4 text-qt-xs text-qt-text-muted">
                            {shortcutLabel("track.delete")}
                        </div>
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* View Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_view")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    {/* 窗口 / 布局：原顶层「布局」选项卡并入视图。窗体显隐收进
                        「窗口」，其余布局级操作（预设、导入导出、重置）收进
                        「布局」二级菜单 —— 两个组件的停靠订阅都在 DockLayoutMenu
                        模块内部，不进 MenuBar。 */}
                    <DockLayoutMenus withCheck={withCheck} />
                    <DropdownMenu.Separator />

                    {/* 时间轴 / 编辑器里的显示开关。 */}
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleTempoMapVisible());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.tempoMapVisible, tf("menu_view_tempo_map"))}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleShowAllTakes());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.showAllTakes, tf("options_show_all_takes"))}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleClipboardPreview());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.showClipboardPreview, t("clipboard_preview"))}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleParamValuePopup());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.showParamValuePopup, t("param_value_popup"))}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />

                    {/* 时间与外观：低频的展示设置。 */}
                    {/* Time Display */}
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>{tf("time_display")}</DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {tf("time_unit_primary")}:{" "}
                                    {tf(timeUnitLabelKey(s.primaryTimeUnit))}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    {TIME_UNITS.map((unit) => (
                                        <DropdownMenu.Item
                                            key={unit}
                                            onSelect={() => {
                                                dispatch(setPrimaryTimeUnit(unit));
                                                void dispatch(persistUiSettings());
                                            }}
                                        >
                                            {withCheck(
                                                s.primaryTimeUnit === unit,
                                                tf(timeUnitLabelKey(unit)),
                                            )}
                                        </DropdownMenu.Item>
                                    ))}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {tf("time_unit_secondary")}:{" "}
                                    {s.secondaryTimeUnit === "none"
                                        ? tf("time_unit_none")
                                        : tf(timeUnitLabelKey(s.secondaryTimeUnit as TimeUnit))}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    {TIME_UNIT_CHOICES.map((unit) => (
                                        <DropdownMenu.Item
                                            key={unit}
                                            onSelect={() => {
                                                dispatch(setSecondaryTimeUnit(unit));
                                                void dispatch(persistUiSettings());
                                            }}
                                        >
                                            {withCheck(
                                                s.secondaryTimeUnit === unit,
                                                unit === "none"
                                                    ? tf("time_unit_none")
                                                    : tf(timeUnitLabelKey(unit as TimeUnit)),
                                            )}
                                        </DropdownMenu.Item>
                                    ))}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Separator />
                            <DropdownMenu.Item onSelect={() => setTimeDisplaySettingsOpen(true)}>
                                {tf("timeline_display_settings")}
                            </DropdownMenu.Item>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {`${t("common_theme")}: ${tf(`theme_${theme.modeSetting}`)}`}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {(["auto", "dark", "light"] as const).map((mode) => (
                                <DropdownMenu.Item
                                    key={mode}
                                    onSelect={() => {
                                        theme.applySettings({
                                            mode,
                                            accentColor: theme.accentColor,
                                            grayColor: theme.grayColor,
                                            radius: theme.radius,
                                            fontFamily: theme.fontFamily,
                                            activeCustomThemeId: theme.activeCustomThemeId,
                                        });
                                    }}
                                >
                                    {withCheck(theme.modeSetting === mode, tf(`theme_${mode}`))}
                                </DropdownMenu.Item>
                            ))}
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                    {/*
                      外观设置现在是停靠面板（居中浮出、不可停靠、不进「窗口」菜单）。
                      `togglePanelVisible` 让它与「窗口」菜单里的其它面板行为一致：
                      已打开则关闭、关闭则打开 —— 比"每次新建一个"更符合用户预期，
                      而且它声明了 `singleton: true`，重复打开本来就只会聚焦同一个窗体。
                    */}
                    <DropdownMenu.Item
                        onSelect={() =>
                            togglePanelVisible(dispatch, store.getState, PANEL_APPEARANCE)
                        }
                    >
                        {tf("menu_appearance_settings")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />

                    {/* 刷新与缓存清理是维护性操作，沉底。 */}
                    <DropdownMenu.Item onSelect={() => dispatch(refreshRuntime())}>
                        {t("action_refresh")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setWaveformCacheConfirmOpen(true)}>
                        {t("menu_clear_waveform_cache")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_options")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tf("stretch_project_override")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tf("stretch_algorithm")}: ${stretchAlgorithmLabel(effectiveProjectStretchAlgorithm)}`}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    <DropdownMenu.Item
                                        onSelect={() =>
                                            void dispatch(
                                                setProjectStretchSettingsRemote({
                                                    stretchAlgorithmOverride: null,
                                                    hifiganMelStretchOverride:
                                                        s.project.hifiganMelStretchOverride,
                                                }),
                                            )
                                        }
                                    >
                                        {withCheck(
                                            s.project.stretchAlgorithmOverride == null,
                                            `${tf("stretch_inherit_global")} (${stretchAlgorithmLabel(s.defaultStretchAlgorithm)})`,
                                        )}
                                    </DropdownMenu.Item>
                                    {(["linear", "signalsmith", "soundtouch"] as const).map(
                                        (algorithm) => (
                                            <DropdownMenu.Item
                                                key={algorithm}
                                                onSelect={() =>
                                                    void dispatch(
                                                        setProjectStretchSettingsRemote({
                                                            stretchAlgorithmOverride: algorithm,
                                                            hifiganMelStretchOverride:
                                                                s.project.hifiganMelStretchOverride,
                                                        }),
                                                    )
                                                }
                                            >
                                                {withCheck(
                                                    s.project.stretchAlgorithmOverride ===
                                                        algorithm,
                                                    stretchAlgorithmLabel(algorithm),
                                                )}
                                            </DropdownMenu.Item>
                                        ),
                                    )}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tf("stretch_hifigan_mel")}: ${effectiveProjectHifiganMelStretch ? tf("stretch_toggle_on") : tf("stretch_toggle_off")}`}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    <DropdownMenu.Item
                                        onSelect={() =>
                                            void dispatch(
                                                setProjectStretchSettingsRemote({
                                                    stretchAlgorithmOverride:
                                                        s.project.stretchAlgorithmOverride,
                                                    hifiganMelStretchOverride: null,
                                                }),
                                            )
                                        }
                                    >
                                        {withCheck(
                                            s.project.hifiganMelStretchOverride == null,
                                            `${tf("stretch_inherit_global")} (${s.defaultHifiganMelStretch ? tf("stretch_toggle_on") : tf("stretch_toggle_off")})`,
                                        )}
                                    </DropdownMenu.Item>
                                    <DropdownMenu.Item
                                        onSelect={() =>
                                            void dispatch(
                                                setProjectStretchSettingsRemote({
                                                    stretchAlgorithmOverride:
                                                        s.project.stretchAlgorithmOverride,
                                                    hifiganMelStretchOverride: true,
                                                }),
                                            )
                                        }
                                    >
                                        {withCheck(
                                            s.project.hifiganMelStretchOverride === true,
                                            tf("stretch_toggle_on"),
                                        )}
                                    </DropdownMenu.Item>
                                    <DropdownMenu.Item
                                        onSelect={() =>
                                            void dispatch(
                                                setProjectStretchSettingsRemote({
                                                    stretchAlgorithmOverride:
                                                        s.project.stretchAlgorithmOverride,
                                                    hifiganMelStretchOverride: false,
                                                }),
                                            )
                                        }
                                    >
                                        {withCheck(
                                            s.project.hifiganMelStretchOverride === false,
                                            tf("stretch_toggle_off"),
                                        )}
                                    </DropdownMenu.Item>
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tf("stretch_global_default")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tf("stretch_algorithm")}: ${stretchAlgorithmLabel(s.defaultStretchAlgorithm)}`}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    {(["linear", "signalsmith", "soundtouch"] as const).map(
                                        (algorithm) => (
                                            <DropdownMenu.Item
                                                key={algorithm}
                                                onSelect={() => {
                                                    dispatch(setDefaultStretchAlgorithm(algorithm));
                                                    void dispatch(persistUiSettings());
                                                }}
                                            >
                                                {withCheck(
                                                    s.defaultStretchAlgorithm === algorithm,
                                                    stretchAlgorithmLabel(algorithm),
                                                )}
                                            </DropdownMenu.Item>
                                        ),
                                    )}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tf("stretch_hifigan_mel")}: ${s.defaultHifiganMelStretch ? tf("stretch_toggle_on") : tf("stretch_toggle_off")}`}
                                </DropdownMenu.SubTrigger>
                                <DropdownMenu.SubContent>
                                    <DropdownMenu.Item
                                        onSelect={() => {
                                            dispatch(setDefaultHifiganMelStretch(true));
                                            void dispatch(persistUiSettings());
                                        }}
                                    >
                                        {withCheck(
                                            s.defaultHifiganMelStretch,
                                            tf("stretch_toggle_on"),
                                        )}
                                    </DropdownMenu.Item>
                                    <DropdownMenu.Item
                                        onSelect={() => {
                                            dispatch(setDefaultHifiganMelStretch(false));
                                            void dispatch(persistUiSettings());
                                        }}
                                    >
                                        {withCheck(
                                            !s.defaultHifiganMelStretch,
                                            tf("stretch_toggle_off"),
                                        )}
                                    </DropdownMenu.Item>
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    <DropdownMenu.Separator />

                    {/* Inference Device */}
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {`${t("menu_inference_device")}: ${
                                s.ortEp === "auto"
                                    ? `${t("menu_inference_auto")}${gpuBackend ? ` (${gpuBackend})` : ""}`
                                    : s.ortEp === "cpu"
                                      ? t("menu_inference_cpu")
                                      : `${t("menu_inference_gpu")} (${gpuBackend || "GPU"})`
                            }`}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {(["auto", "cpu", "gpu"] as const).map((ep) => {
                                const labels: Record<string, string> = {
                                    auto: t("menu_inference_auto"),
                                    cpu: t("menu_inference_cpu"),
                                    gpu: t("menu_inference_gpu"),
                                };
                                return (
                                    <DropdownMenu.Item
                                        key={ep}
                                        onSelect={() => {
                                            dispatch(setOrtEp(ep));
                                            void dispatch(persistUiSettings());
                                        }}
                                    >
                                        {withCheck(s.ortEp === ep, labels[ep])}
                                    </DropdownMenu.Item>
                                );
                            })}

                            {/* GPU Device Selector — shown when GPU EP is selected */}
                            {s.ortEp === "gpu" && dmlAdapters.length > 0 && (
                                <>
                                    <DropdownMenu.Separator />
                                    <DropdownMenu.Label>{tf("menu_gpu_device")}</DropdownMenu.Label>
                                    <DropdownMenu.Item
                                        onSelect={() => {
                                            dispatch(setOrtDeviceId(null));
                                            void dispatch(persistUiSettings());
                                        }}
                                    >
                                        {withCheck(
                                            s.ortDeviceId == null,
                                            tf("menu_gpu_auto_select"),
                                        )}
                                    </DropdownMenu.Item>
                                    {dmlAdapters.map((adapter) => (
                                        <DropdownMenu.Item
                                            key={adapter.deviceId}
                                            onSelect={() => {
                                                dispatch(setOrtDeviceId(adapter.deviceId));
                                                void dispatch(persistUiSettings());
                                            }}
                                        >
                                            {withCheck(
                                                s.ortDeviceId === adapter.deviceId,
                                                `${adapter.name} (${adapter.memoryMb} MB)`,
                                            )}
                                        </DropdownMenu.Item>
                                    ))}
                                </>
                            )}

                            <DropdownMenu.Separator />
                            <DropdownMenu.Item onSelect={() => setBenchmarkDialogOpen(true)}>
                                {t("menu_run_benchmark")}
                            </DropdownMenu.Item>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    {/* Background Pre-render — same level as Inference Device, no separator */}
                    <DropdownMenu.Item
                        onSelect={async () => {
                            dispatch(toggleAutoBackgroundRender());
                            await dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.autoBackgroundRender, tf("menu_background_prerender"))}
                    </DropdownMenu.Item>

                    {/* Take display / editing options */}
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleSyncEditsAcrossTakes());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.syncEditsAcrossTakes, tf("sync_edits_across_takes"))}
                    </DropdownMenu.Item>

                    {/* Auto-reload modified media — same level as Background Pre-render */}
                    <DropdownMenu.Item
                        onSelect={() => onAutoReloadModifiedMediaChange(!autoReloadModifiedMedia)}
                    >
                        {withCheck(
                            autoReloadModifiedMedia,
                            tf("options_auto_reload_modified_media"),
                        )}
                    </DropdownMenu.Item>

                    {/* Loop for new clips — 为新的音频块启用循环（默认开启） */}
                    <DropdownMenu.Item onSelect={() => onLoopNewClipsChange(!loopNewClips)}>
                        {withCheck(loopNewClips, tf("options_loop_new_clips"))}
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    {/* Snap/Grid Settings — above Keyboard Shortcuts */}
                    <DropdownMenu.Item onSelect={() => setSnapSettingsOpen(true)}>
                        {tf("snap_grid_settings_title")}
                    </DropdownMenu.Item>

                    {/* Search matching settings — 作用于全部搜索面，因此与
                        吸附/网格同级，而不是塞进某个面板自己的设置页。 */}
                    <DropdownMenu.Item onSelect={() => dispatch(setSearchSettingsDialogOpen(true))}>
                        {tf("search_settings_title")}
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    {/* Render cache manager — above Keyboard Shortcuts */}
                    <DropdownMenu.Item onSelect={() => setRenderCacheDialogOpen(true)}>
                        {tf("menu_render_cache_manager")}
                    </DropdownMenu.Item>

                    {/* 颤音预设库。与上下文菜单用**不同**的文案：选项菜单这一层
                        没有"颤音"语境，只写"管理预设"没人知道管的是哪一种。 */}
                    <DropdownMenu.Item onSelect={() => setVibratoDialogOpen(true)}>
                        {tf("menu_vibrato_presets")}
                    </DropdownMenu.Item>

                    {/* Import channel policy（假立体声 → 单声道） */}
                    <DropdownMenu.Item onSelect={() => setChannelImportDialogOpen(true)}>
                        {tf("menu_channel_import_settings")}
                    </DropdownMenu.Item>

                    {/* 指针设备（触控板 / 数位板 / 触控笔 / 触摸）输入偏好 */}
                    <DropdownMenu.Item onSelect={() => setPenInputDialogOpen(true)}>
                        {tf("menu_pen_input_settings")}
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />

                    {/* Keyboard Shortcuts — at the bottom */}
                    <DropdownMenu.Item onSelect={() => setKbDialogOpen(true)}>
                        {t("menu_keybindings")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Help Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_help")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={async () => {
                            const { openLogFolder } =
                                await import("../../services/api/diagnostics");
                            try {
                                const res = await openLogFolder();
                                if (!res.ok) {
                                    setNotice({
                                        title: tf("status_error_prefix"),
                                        message: res.error || tf("menu_open_log_folder_failed"),
                                    });
                                }
                            } catch (e) {
                                setNotice({
                                    title: tf("status_error_prefix"),
                                    message: String(e),
                                });
                            }
                        }}
                    >
                        {tf("menu_open_log_folder")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        disabled={diagnosticsExporting}
                        onSelect={() => void handleExportDiagnostics()}
                    >
                        {tf("menu_export_diagnostics")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setAboutDialogOpen(true)}>
                        {t("menu_about")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <Flex ml="auto" gap="2" align="center" className="shrink-0">
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                        <Flex align="center" gap="1">
                            <GlobeIcon width={14} height={14} />
                            <span>{t("common_language")}</span>
                        </Flex>
                    </DropdownMenu.Trigger>
                    <DropdownMenu.Content>
                        <DropdownMenu.Item onSelect={() => setLocale("en-US")}>
                            {t("lang_en")}
                        </DropdownMenu.Item>
                        <DropdownMenu.Item onSelect={() => setLocale("zh-CN")}>
                            {t("lang_zh")}
                        </DropdownMenu.Item>
                        <DropdownMenu.Item onSelect={() => setLocale("zh-TW")}>
                            {t("lang_zh_tw")}
                        </DropdownMenu.Item>
                        <DropdownMenu.Item onSelect={() => setLocale("ja-JP")}>
                            {t("lang_ja")}
                        </DropdownMenu.Item>
                        <DropdownMenu.Item onSelect={() => setLocale("ko-KR")}>
                            {t("lang_ko")}
                        </DropdownMenu.Item>
                    </DropdownMenu.Content>
                </DropdownMenu.Root>
            </Flex>

            {/* 快捷键设置对话框 */}
            <KeybindingsDialog open={kbDialogOpen} onOpenChange={setKbDialogOpen} />

            {/* 停靠窗体的「窗口 / 布局」二级菜单所触发的对话框（常驻，
                不能放进视图菜单的 Content —— 菜单关闭时那里会卸载）。 */}
            <DockLayoutDialogs />

            <TimelineDisplaySettingsDialog
                open={timeDisplaySettingsOpen}
                onOpenChange={setTimeDisplaySettingsOpen}
            />

            <SnapGridSettingsDialog open={snapSettingsOpen} onOpenChange={setSnapSettingsOpen} />

            <ExportAudioDialog open={exportDialogOpen} onOpenChange={setExportDialogOpen} />

            <AutoBackupDialog
                open={autoBackupDialogOpen}
                settings={autoBackupSettings}
                onOpenChange={setAutoBackupDialogOpen}
                onSettingsSaved={onAutoBackupSettingsSaved}
            />

            <SearchSettingsDialog
                open={s.searchSettingsDialogOpen}
                onOpenChange={(open) => dispatch(setSearchSettingsDialogOpen(open))}
            />

            <ChannelImportDialog
                open={channelImportDialogOpen}
                onOpenChange={setChannelImportDialogOpen}
            />
            <PenInputDialog
                open={penInputDialogOpen}
                onOpenChange={setPenInputDialogOpen}
            />
            <RenderCacheDialog
                open={renderCacheDialogOpen}
                onOpenChange={setRenderCacheDialogOpen}
            />

            {/* 颤音预设库：与右键菜单的「管理预设…」共用同一个对话框 */}
            <VibratoDialog
                open={vibratoDialogOpen}
                onOpenChange={setVibratoDialogOpen}
                editParam={s.editParam}
            />

            <RecordingSettingsDialog
                open={recordingDialogOpen}
                onOpenChange={setRecordingDialogOpen}
            />

            {/* Inference device benchmark */}
            <BenchmarkDialog open={benchmarkDialogOpen} onOpenChange={setBenchmarkDialogOpen} />

            {/* 导出诊断信息进行中：含基准测试（约 20–60 秒），给出明确的进行中提示。
                不可中断（后端没有取消通道），因此屏蔽 Esc / 点击遮罩关闭
                —— 这一条现在由 `dismissible={false}` 表达，不再靠
                `onOpenChange={() => {}}` 把回调整个废掉。 */}
            <AppDialog
                open={diagnosticsExporting}
                onOpenChange={setDiagnosticsExporting}
                title={tf("menu_export_diagnostics")}
                size="sm"
                dismissible={false}
            >
                <Flex align="center" gap="3">
                    <AppBusy size="md" label={tf("menu_export_diagnostics_running")} />
                </Flex>
            </AppDialog>

            {/* 清空波形缓存：代价高的破坏性维护操作，先确认再执行。 */}
            <AppConfirmDialog
                open={waveformCacheConfirmOpen}
                onOpenChange={setWaveformCacheConfirmOpen}
                title={tf("menu_clear_waveform_cache")}
                message={tf("menu_clear_waveform_cache_confirm")}
                confirmLabel={tf("menu_clear_waveform_cache")}
                cancelLabel={tf("cancel")}
                intent="danger"
                onConfirm={() => {
                    void dispatch(clearWaveformCacheRemote());
                }}
            />

            {/* 关于对话框：简介 + 版本 + Commit（可点击跳转源码快照）+ 仓库链接 */}
            <AboutDialog open={aboutDialogOpen} onOpenChange={setAboutDialogOpen} />

            {/* 错误报告：取代此前的 window.alert（原生弹窗在 Tauri 里不可主题化、
                按钮不可本地化）。标题与正文分开，便于放系统错误原文。 */}
            <AppNoticeDialog
                open={notice !== null}
                onOpenChange={(open) => {
                    if (!open) setNotice(null);
                }}
                title={notice?.title ?? ""}
                message={notice?.message ?? ""}
                closeLabel={t("ok")}
            />

            {/*
              菜单导入模式选择（多文件）。
              
              此前是手写模态：缺 Esc、无焦点管理、不抑制全局快捷键（框内按空格会触发
              播放）、无 Enter 默认动作，标题 13px（其余对话框是 20px），宽度写死
              380px（四档是 400/520/640/800）。改走 `AppDialog` 后这些一次对齐。
             */}
            <AppDialog
                open={menuImportMode !== null}
                onOpenChange={(open) => {
                    if (!open) setMenuImportMode(null);
                }}
                title={tf("import_dialog_title")}
                message={
                    menuImportMode
                        ? plural("import_files_selected", menuImportMode.audioPaths.length)
                        : null
                }
                size="sm"
                actions={[
                    {
                        id: "cancel",
                        label: tf("cancel"),
                        onClick: () => setMenuImportMode(null),
                    },
                ]}
            >
                <AppChoiceList
                    options={[
                        { id: "across-time", label: t("import_across_time") },
                        { id: "across-tracks", label: t("import_across_tracks") },
                        { id: "as-takes", label: t("import_as_takes") },
                    ]}
                    onSelect={(id) => {
                        const mode = menuImportMode;
                        if (!mode) return;
                        setMenuImportMode(null);
                        void dispatch(
                            importMultipleAudioAtPosition({
                                audioPaths: mode.audioPaths,
                                mode: id as "across-time" | "across-tracks" | "as-takes",
                                trackId: mode.trackId,
                                startSec: mode.startSec,
                            }),
                        );
                    }}
                />
            </AppDialog>

            {/* 多音轨媒体：选择要导入的音轨（同样从手写模态迁入 UI 系统） */}
            <AppDialog
                open={mediaStreamImport !== null}
                onOpenChange={(open) => {
                    if (!open) setMediaStreamImport(null);
                }}
                title={tf("media_stream_select_title")}
                message={tf("media_stream_select_hint")}
                description={
                    mediaStreamImport ? (
                        <span className="block truncate">{mediaStreamImport.path}</span>
                    ) : null
                }
                size="md"
                actions={[
                    {
                        id: "cancel",
                        label: tf("cancel"),
                        onClick: () => setMediaStreamImport(null),
                    },
                ]}
            >
                <AppChoiceList
                    options={(mediaStreamImport?.streams ?? []).map((stream) => {
                        const meta = [
                            stream.codec,
                            stream.channels > 0 ? `${stream.channels} ch` : null,
                            stream.sampleRate > 0 ? `${stream.sampleRate} Hz` : null,
                            stream.title || stream.language,
                            stream.durationSec > 0 ? `${stream.durationSec.toFixed(2)} s` : null,
                        ].filter(Boolean);
                        return {
                            id: String(stream.index),
                            label: `${tf("media_stream_track")} ${stream.index + 1}`,
                            description: meta.join(" · "),
                        };
                    })}
                    onSelect={(id) => {
                        const m = mediaStreamImport;
                        if (!m) return;
                        setMediaStreamImport(null);
                        void dispatch(
                            importAudioAtPosition({
                                audioPath: m.path,
                                trackId: m.trackId,
                                startSec: m.startSec,
                                mediaAudioStreamIndex: Number(id),
                            }),
                        );
                    }}
                />
            </AppDialog>

            {/* Edit operation dialogs */}
            <TransposeCentsDialog
                open={transposeCentsOpen}
                onOpenChange={setTransposeCentsOpen}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(cents, edgeSmoothnessPercent) =>
                    dispatchEditOp("transposeCents", {
                        cents,
                        edgeSmoothnessPercent,
                    })
                }
            />
            <TransposeDegreesDialog
                open={transposeDegreesOpen}
                onOpenChange={setTransposeDegreesOpen}
                defaultScale={s.project.baseScale}
                defaultUseProjectScale={true}
                projectScaleLabel={projectScaleLabelWithHint}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(degrees, scaleValue, edgeSmoothnessPercent) =>
                    dispatchEditOp("transposeDegrees", {
                        degrees,
                        scale: resolveScaleToken(scaleValue),
                        edgeSmoothnessPercent,
                    })
                }
            />
            <SetPitchDialog
                open={setPitchOpen}
                onOpenChange={setSetPitchOpen}
                titleText={isPitchParam ? tf("menu_set_pitch") : tf("menu_set_value")}
                valueLabelText={setToValueLabel}
                defaultValue={setToDefaultValue}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(value, edgeSmoothnessPercent) =>
                    dispatchEditOp("setPitch", {
                        value,
                        edgeSmoothnessPercent,
                    })
                }
            />
            <AverageDialog
                open={averageOpen}
                onOpenChange={setAverageOpen}
                onConfirm={(strength) => {
                    dispatchEditOp("average", { strength });
                }}
            />
            <SmoothDialog
                open={smoothOpen}
                onOpenChange={setSmoothOpen}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(strength) => dispatchEditOp("smooth", { strength })}
            />
            <QuantizeDialog
                open={quantizeOpen}
                onOpenChange={setQuantizeOpen}
                valueMode={!isPitchParam}
                defaultQuantizeUnit={quantizeDefaultUnit}
                defaultTolerance={0}
                defaultScale={s.project.baseScale}
                defaultUseProjectScale={true}
                projectScaleLabel={projectScaleLabelWithHint}
                defaultToleranceCents={s.pitchSnapToleranceCents}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(
                    unit,
                    scaleValue,
                    toleranceCents,
                    quantizeUnit,
                    edgeSmoothnessPercent,
                ) =>
                    dispatchEditOp("quantize", {
                        unit,
                        scale: resolveScaleToken(scaleValue),
                        toleranceCents,
                        tolerance: toleranceCents,
                        quantizeUnit,
                        edgeSmoothnessPercent,
                    })
                }
            />
            <MeanQuantizeDialog
                open={meanQuantizeOpen}
                onOpenChange={setMeanQuantizeOpen}
                valueMode={!isPitchParam}
                defaultQuantizeUnit={quantizeDefaultUnit}
                defaultTolerance={0}
                defaultScale={s.project.baseScale}
                defaultUseProjectScale={true}
                projectScaleLabel={projectScaleLabelWithHint}
                defaultToleranceCents={s.pitchSnapToleranceCents}
                defaultSmoothness={s.edgeSmoothnessPercent}
                onConfirm={(
                    unit,
                    scaleValue,
                    toleranceCents,
                    quantizeUnit,
                    edgeSmoothnessPercent,
                ) =>
                    dispatchEditOp("meanQuantize", {
                        unit,
                        scale: resolveScaleToken(scaleValue),
                        toleranceCents,
                        tolerance: toleranceCents,
                        quantizeUnit,
                        edgeSmoothnessPercent,
                    })
                }
            />
        </Flex>
    );
};
