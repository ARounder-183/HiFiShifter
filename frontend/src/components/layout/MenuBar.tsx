import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Flex, DropdownMenu } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import {
    openReaperFromDialog,
    openVocalShifterFromDialog,
    addTrackRemote,
    removeTrackRemote,
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
    toggleAutoBackgroundRender,
    toggleClipboardPreview,
    toggleParamValuePopup,
    toggleTempoMapVisible,
    setProjectStretchSettingsRemote,
} from "../../features/session/sessionSlice";
import type { TimeUnit } from "../../features/session/sessionTypes";
import { TIME_UNITS, TIME_UNIT_CHOICES } from "./timeline/timeFormat";
import { TimelineDisplaySettingsDialog } from "./TimelineDisplaySettingsDialog";
import { SnapGridSettingsDialog } from "./SnapGridSettingsDialog";
import { scaleChangesInRange, scaleLikeEquals } from "../../utils/tempoMap";
import type { ScaleLike } from "../../utils/musicalScales";
import {
    getPianoRollSelection,
    subscribePianoRollSelection,
} from "../../utils/pianoRollSelectionBus";



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
import { KeybindingsDialog } from "./KeybindingsDialog";
import { AppearanceSettingsDialog } from "./AppearanceSettingsDialog";
import {
    TransposeCentsDialog,
    TransposeDegreesDialog,
    SetPitchDialog,
    AverageDialog,
    SmoothDialog,
    VibratoDialog,
    QuantizeDialog,
    MeanQuantizeDialog,
} from "../editDialogs/EditDialogs";
import { SCALE_LABELS } from "../../utils/musicalScales";
import { ExportAudioDialog } from "./ExportAudioDialog";
import { AutoBackupDialog } from "./AutoBackupDialog";
import { RecordingSettingsDialog } from "./RecordingSettingsDialog";
import { BenchmarkDialog } from "./BenchmarkDialog";

import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "./pianoRoll/childPitchOffsetParams";
import type { AutoBackupSettings } from "../../services/api/project";
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
    const { t, setLocale } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const theme = useAppTheme();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const [kbDialogOpen, setKbDialogOpen] = useState(false);
    const [appearanceDialogOpen, setAppearanceDialogOpen] = useState(false);
    const [timeDisplaySettingsOpen, setTimeDisplaySettingsOpen] = useState(false);
    const [snapSettingsOpen, setSnapSettingsOpen] = useState(false);
    const [exportDialogOpen, setExportDialogOpen] = useState(false);
    const [autoBackupDialogOpen, setAutoBackupDialogOpen] = useState(false);
    const [recordingDialogOpen, setRecordingDialogOpen] = useState(false);
    const [benchmarkDialogOpen, setBenchmarkDialogOpen] = useState(false);
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
    const [vibratoOpen, setVibratoOpen] = useState(false);
    const [vibratoParamRange, setVibratoParamRange] = useState<
        { min: number; max: number } | undefined
    >(undefined);
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
              : s.editParam === "volume" || s.editParam === "dyn_edit"
                ? 1
                : 0;
    const setToValueLabel = s.editParam === "pitch" ? tAny("dlg_midi_note") : tAny("dlg_value");
    const quantizeDefaultUnit = (() => {
        if (isChildCentsParam) return 100;
        if (isChildDegreesParam) return 1;
        switch (s.editParam) {
            case "volume":
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
            ? `${tAny("project_scale_prefix")} (${tAny("custom_scale_short")})`
            : `${tAny("project_scale_prefix")} (${SCALE_LABELS[s.project.baseScale]})`;

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
        if (
            changes.every((c) => scaleLikeEquals(c.scale, projectScale))
        ) {
            return null;
        }
        return tAny("project_scale_tempo_map_hint");
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [s.tempoMap, s.projectSec, s.project, tAny, selectionVersion]);
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
                return tAny("stretch_option_linear");
            case "signalsmith":
                return tAny("stretch_option_signalsmith");
            case "soundtouch":
            default:
                return tAny("stretch_option_soundtouch");
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
                case "addVibrato":
                    setVibratoParamRange((e as CustomEvent).detail?.paramRange);
                    setVibratoOpen(true);
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

    /** 派发编辑操作事件给 PianoRollPanel / TimelinePanel */
    const dispatchEditOp = useCallback((op: string, data?: Record<string, unknown>) => {
        const active = document.activeElement as HTMLElement | null;
        const inPianoRoll =
            active?.hasAttribute("data-piano-roll-scroller") ||
            active?.closest?.("[data-piano-roll-scroller]") ||
            document.body.getAttribute("data-hs-focus-window") === "pianoRoll";
        const inTrackHeader =
            Boolean(active?.closest?.("[data-track-list-panel]")) ||
            document.body.getAttribute("data-hs-focus-window") === "trackHeader";
        const inTimeline =
            active?.hasAttribute("data-timeline-scroller") ||
            active?.closest?.("[data-timeline-scroller]") ||
            document.body.getAttribute("data-hs-focus-window") === "timeline";

        if (
            (op === "copy" ||
                op === "cut" ||
                op === "paste" ||
                op === "pasteTracks") &&
            inTimeline &&
            !inPianoRoll &&
            !inTrackHeader
        ) {
            window.dispatchEvent(
                new CustomEvent("hifi:timelineEditOp", { detail: { op, ...data } }),
            );
            return;
        }

        window.dispatchEvent(new CustomEvent("hifi:editOp", { detail: { op, ...data } }));
    }, []);

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

    return (
        <Flex
            align="center"
            className="h-8 bg-qt-panel border-b border-qt-border px-1 select-none z-50 flex-nowrap gap-1 overflow-x-auto overflow-y-hidden min-w-0 custom-scrollbar"
        >
            {/**
             * Note: @radix-ui/themes DropdownMenu.Trigger does not support asChild.
             * Use Trigger as the actual button element to avoid nesting <button>.
             */}
            {/* File Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_file")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={onNewProject}>
                        {t("menu_new_project")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.new")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={onOpenProject}>
                        {t("menu_open_project")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
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
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.save")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(saveProjectAsRemote())}>
                        {t("menu_save_project_as")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
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
                    </DropdownMenu.Item>
                    <DropdownMenu.Item
                        onSelect={() => {
                            void handleImportMidiFromMenu();
                        }}
                    >
                        {t("menu_import_midi")}{" "}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={onImportProject}>
                        {t("menu_import_hifishifter")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(openReaperFromDialog())}>
                        {t("menu_import_reaper")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(openVocalShifterFromDialog())}>
                        {t("menu_import_vocalshifter")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setExportDialogOpen(true)}>
                        {t("menu_export_audio")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("project.export")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setAutoBackupDialogOpen(true)}>
                        {tAny("menu_auto_backup")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setRecordingDialogOpen(true)}>
                        {tAny("menu_recording_settings")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={onExit} color="red">
                        {t("menu_exit")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Edit Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_edit")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={() => void dispatch(undoRemote())}>
                        {t("menu_undo")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.undo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(redoRemote())}>
                        {t("menu_redo")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.redo")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 剪贴板：剪切 / 复制 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("cut")}>
                        {tAny("menu_cut")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("clip.cut")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("copy")}>
                        {tAny("menu_copy")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.copy")}
                        </div>
                    </DropdownMenu.Item>
                    {/* 剪贴板：粘贴 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("paste")}>
                        {tAny("menu_paste")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.paste")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteTracks")}>
                        {t("menu_paste_new_tracks")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteTracks")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 外部剪贴板交换 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteVocalShifter")}>
                        {t("menu_paste_vocalshifter_clipboard")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteVocalShifter")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    {/* 选择 */}
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("selectAll")}>
                        {tAny("menu_select_all")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.selectAll")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("deselect")}>
                        {tAny("menu_deselect")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.deselect")}
                        </div>
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* Track Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_track")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={() => dispatch(addTrackRemote({}))}>
                        {t("track_add")}
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
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            {/* View Menu */}
            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_view")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={() => dispatch(refreshRuntime())}>
                        {t("action_refresh")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => void dispatch(clearWaveformCacheRemote())}>
                        {t("menu_clear_waveform_cache")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
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
                    <DropdownMenu.Item
                        onSelect={() => {
                            dispatch(toggleTempoMapVisible());
                            void dispatch(persistUiSettings());
                        }}
                    >
                        {withCheck(s.tempoMapVisible, tAny("menu_view_tempo_map"))}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />

                    {/* Time Display */}
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tAny("time_display")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {tAny("time_unit_primary")}:{" "}
                                    {tAny(timeUnitLabelKey(s.primaryTimeUnit))}
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
                                                tAny(timeUnitLabelKey(unit)),
                                            )}
                                        </DropdownMenu.Item>
                                    ))}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {tAny("time_unit_secondary")}:{" "}
                                    {s.secondaryTimeUnit === "none"
                                        ? tAny("time_unit_none")
                                        : tAny(timeUnitLabelKey(s.secondaryTimeUnit as TimeUnit))}
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
                                                    ? tAny("time_unit_none")
                                                    : tAny(timeUnitLabelKey(unit as TimeUnit)),
                                            )}
                                        </DropdownMenu.Item>
                                    ))}
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                            <DropdownMenu.Separator />
                            <DropdownMenu.Item onSelect={() => setTimeDisplaySettingsOpen(true)}>
                                {tAny("timeline_display_settings")}
                            </DropdownMenu.Item>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item
                        onSelect={() => {
                            const nextMode = theme.mode === "dark" ? "light" : "dark";
                            theme.applySettings({
                                mode: nextMode,
                                accentColor: theme.accentColor,
                                grayColor: theme.grayColor,
                                radius: theme.radius,
                                fontFamily: theme.fontFamily,
                                activeCustomThemeId: theme.activeCustomThemeId,
                            });
                        }}
                    >
                        {t("theme")}: {theme.mode === "dark" ? t("theme_dark") : t("theme_light")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setAppearanceDialogOpen(true)}>
                        {tAny("menu_appearance_settings")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <DropdownMenu.Root>
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_options")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item onSelect={() => setSnapSettingsOpen(true)}>
                        {tAny("snap_grid_settings_title")}
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tAny("stretch_project_override")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tAny("stretch_algorithm")}: ${stretchAlgorithmLabel(effectiveProjectStretchAlgorithm)}`}
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
                                            `${tAny("stretch_inherit_global")} (${stretchAlgorithmLabel(s.defaultStretchAlgorithm)})`,
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
                                    {`${tAny("stretch_hifigan_mel")}: ${effectiveProjectHifiganMelStretch ? tAny("stretch_toggle_on") : tAny("stretch_toggle_off")}`}
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
                                            `${tAny("stretch_inherit_global")} (${s.defaultHifiganMelStretch ? tAny("stretch_toggle_on") : tAny("stretch_toggle_off")})`,
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
                                            tAny("stretch_toggle_on"),
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
                                            tAny("stretch_toggle_off"),
                                        )}
                                    </DropdownMenu.Item>
                                </DropdownMenu.SubContent>
                            </DropdownMenu.Sub>
                        </DropdownMenu.SubContent>
                    </DropdownMenu.Sub>

                    <DropdownMenu.Separator />

                    <DropdownMenu.Sub>
                        <DropdownMenu.SubTrigger>
                            {tAny("stretch_global_default")}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            <DropdownMenu.Sub>
                                <DropdownMenu.SubTrigger>
                                    {`${tAny("stretch_algorithm")}: ${stretchAlgorithmLabel(s.defaultStretchAlgorithm)}`}
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
                                    {`${tAny("stretch_hifigan_mel")}: ${s.defaultHifiganMelStretch ? tAny("stretch_toggle_on") : tAny("stretch_toggle_off")}`}
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
                                            tAny("stretch_toggle_on"),
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
                                            tAny("stretch_toggle_off"),
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
                            {`${t("menu_inference_device")}: ${s.ortEp === "auto" ? t("menu_inference_auto") : s.ortEp === "cpu" ? t("menu_inference_cpu") : `${t("menu_inference_gpu")} (${s.runtime.gpuBackend || "GPU"})`}`}
                        </DropdownMenu.SubTrigger>
                        <DropdownMenu.SubContent>
                            {(["auto", "cpu", "gpu"] as const).map((ep) => {
                                const labels: Record<string, string> = {
                                    auto: t("menu_inference_auto"),
                                    cpu: t("menu_inference_cpu"),
                                    gpu: `${t("menu_inference_gpu")} (${s.runtime.gpuBackend || "GPU"})`,
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
                                    <DropdownMenu.Label>
                                        {tAny("menu_gpu_device")}
                                    </DropdownMenu.Label>
                                    <DropdownMenu.Item
                                        onSelect={() => {
                                            dispatch(setOrtDeviceId(null));
                                            void dispatch(persistUiSettings());
                                        }}
                                    >
                                        {withCheck(
                                            s.ortDeviceId == null,
                                            tAny("menu_gpu_auto_select"),
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
                        {withCheck(s.autoBackgroundRender, tAny("menu_background_prerender"))}
                    </DropdownMenu.Item>

                    {/* Auto-reload modified media — same level as Background Pre-render */}
                    <DropdownMenu.Item
                        onSelect={() => onAutoReloadModifiedMediaChange(!autoReloadModifiedMedia)}
                    >
                        {withCheck(
                            autoReloadModifiedMedia,
                            tAny("options_auto_reload_modified_media"),
                        )}
                    </DropdownMenu.Item>

                    {/* Loop for new clips — 为新的音频块启用循环（默认开启） */}
                    <DropdownMenu.Item onSelect={() => onLoopNewClipsChange(!loopNewClips)}>
                        {withCheck(loopNewClips, tAny("options_loop_new_clips"))}
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
                <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                    <span>{t("menu_help")}</span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    <DropdownMenu.Item
                        onSelect={async () => {
                            const { openUrl } = await import("@tauri-apps/plugin-opener");
                            openUrl("https://github.com/ARounder-183/HiFiShifter");
                        }}
                    >
                        {t("menu_about")}
                    </DropdownMenu.Item>
                </DropdownMenu.Content>
            </DropdownMenu.Root>

            <Flex ml="auto" gap="2" align="center" className="shrink-0">
                <DropdownMenu.Root>
                    <DropdownMenu.Trigger className="shrink-0 rounded px-2 py-1 text-xs text-qt-text hover:bg-qt-highlight hover:text-white">
                        <Flex align="center" gap="1">
                            <GlobeIcon width={14} height={14} />
                            <span>{t("language")}</span>
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

            {/* 外观设置对话框 */}
            <AppearanceSettingsDialog
                open={appearanceDialogOpen}
                onOpenChange={setAppearanceDialogOpen}
            />

            <TimelineDisplaySettingsDialog
                open={timeDisplaySettingsOpen}
                onOpenChange={setTimeDisplaySettingsOpen}
            />

            <SnapGridSettingsDialog
                open={snapSettingsOpen}
                onOpenChange={setSnapSettingsOpen}
            />

            <ExportAudioDialog open={exportDialogOpen} onOpenChange={setExportDialogOpen} />

            <AutoBackupDialog
                open={autoBackupDialogOpen}
                settings={autoBackupSettings}
                onOpenChange={setAutoBackupDialogOpen}
                onSettingsSaved={onAutoBackupSettingsSaved}
            />

            <RecordingSettingsDialog
                open={recordingDialogOpen}
                onOpenChange={setRecordingDialogOpen}
            />

            {/* Inference device benchmark */}
            <BenchmarkDialog open={benchmarkDialogOpen} onOpenChange={setBenchmarkDialogOpen} />

            {/* 菜单导入模式选择（多文件） */}
            {menuImportMode && (
                <div
                    className="fixed inset-0 z-[9999] bg-qt-overlay flex items-center justify-center"
                    onClick={() => setMenuImportMode(null)}
                >
                    <div
                        className="w-[380px] max-w-[92vw] bg-qt-panel border border-qt-border rounded-xl shadow-[0_20px_44px_rgba(0,0,0,0.28)]"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="px-4 py-3 border-b border-qt-border">
                            <div className="text-sm font-medium text-qt-text">
                                {tAny("import_dialog_title") || t("menu_import_media")}
                            </div>
                            <div className="mt-1 text-xs text-qt-text-muted">
                                {menuImportMode.audioPaths.length} file(s) selected
                            </div>
                        </div>

                        <div className="px-3 py-3 flex flex-col gap-2">
                            <button
                                className="w-full text-left px-3 py-2 rounded-lg text-sm text-qt-text border border-qt-border hover:bg-qt-hover"
                                onClick={() => {
                                    const m = menuImportMode;
                                    setMenuImportMode(null);
                                    void dispatch(
                                        importMultipleAudioAtPosition({
                                            audioPaths: m.audioPaths,
                                            mode: "across-time",
                                            trackId: m.trackId,
                                            startSec: m.startSec,
                                        }),
                                    );
                                }}
                            >
                                {t("import_across_time")}
                            </button>
                            <button
                                className="w-full text-left px-3 py-2 rounded-lg text-sm text-qt-text border border-qt-border hover:bg-qt-hover"
                                onClick={() => {
                                    const m = menuImportMode;
                                    setMenuImportMode(null);
                                    void dispatch(
                                        importMultipleAudioAtPosition({
                                            audioPaths: m.audioPaths,
                                            mode: "across-tracks",
                                            trackId: m.trackId,
                                            startSec: m.startSec,
                                        }),
                                    );
                                }}
                            >
                                {t("import_across_tracks")}
                            </button>
                        </div>

                        <div className="px-3 py-2 border-t border-qt-border flex justify-end">
                            <button
                                className="px-3 py-1.5 text-xs text-qt-text hover:bg-qt-hover rounded-lg"
                                onClick={() => setMenuImportMode(null)}
                            >
                                {tAny("cancel") || "Cancel"}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* 多音轨媒体：选择要导入的音轨 */}
            {mediaStreamImport && (
                <div
                    className="fixed inset-0 z-[9999] bg-qt-overlay flex items-center justify-center"
                    onClick={() => setMediaStreamImport(null)}
                >
                    <div
                        className="w-[420px] max-w-[92vw] bg-qt-panel border border-qt-border rounded-xl shadow-[0_20px_44px_rgba(0,0,0,0.28)]"
                        onClick={(e) => e.stopPropagation()}
                    >
                        <div className="px-4 py-3 border-b border-qt-border">
                            <div className="text-sm font-medium text-qt-text">
                                {tAny("media_stream_select_title") || "Select audio stream"}
                            </div>
                            <div className="mt-1 text-xs text-qt-text-muted truncate">
                                {mediaStreamImport.path}
                            </div>
                            <div className="mt-1 text-xs text-qt-text-muted">
                                {tAny("media_stream_select_hint") ||
                                    "This file contains multiple audio streams. Select one to extract and import."}
                            </div>
                        </div>

                        <div className="px-3 py-3 flex flex-col gap-2 max-h-[50vh] overflow-y-auto custom-scrollbar">
                            {mediaStreamImport.streams.map((stream) => {
                                const meta = [
                                    stream.codec,
                                    stream.channels > 0 ? `${stream.channels} ch` : null,
                                    stream.sampleRate > 0 ? `${stream.sampleRate} Hz` : null,
                                    stream.title || stream.language,
                                    stream.durationSec > 0
                                        ? `${stream.durationSec.toFixed(2)} s`
                                        : null,
                                ].filter(Boolean);
                                return (
                                    <button
                                        key={stream.index}
                                        className="w-full text-left px-3 py-2 rounded-lg text-sm text-qt-text border border-qt-border hover:bg-qt-hover"
                                        onClick={() => {
                                            const m = mediaStreamImport;
                                            setMediaStreamImport(null);
                                            void dispatch(
                                                importAudioAtPosition({
                                                    audioPath: m.path,
                                                    trackId: m.trackId,
                                                    startSec: m.startSec,
                                                    mediaAudioStreamIndex: stream.index,
                                                }),
                                            );
                                        }}
                                    >
                                        <span className="font-medium">
                                            {tAny("media_stream_track") || "Track"}{" "}
                                            {stream.index + 1}
                                        </span>
                                        <span className="ml-2 text-xs text-qt-text-muted">
                                            {meta.join(" · ")}
                                        </span>
                                    </button>
                                );
                            })}
                        </div>

                        <div className="px-3 py-2 border-t border-qt-border flex justify-end">
                            <button
                                className="px-3 py-1.5 text-xs text-qt-text hover:bg-qt-hover rounded-lg"
                                onClick={() => setMediaStreamImport(null)}
                            >
                                {tAny("cancel") || "Cancel"}
                            </button>
                        </div>
                    </div>
                </div>
            )}

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
                titleText={isPitchParam ? tAny("menu_set_pitch") : tAny("menu_set_value")}
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
            <VibratoDialog
                open={vibratoOpen}
                onOpenChange={setVibratoOpen}
                editParam={s.editParam}
                paramRange={vibratoParamRange}
                onConfirm={(amplitude, rate, attack, release, phase) =>
                    dispatchEditOp("addVibrato", { amplitude, rate, attack, release, phase })
                }
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
                onConfirm={(unit, scaleValue, toleranceCents, quantizeUnit) =>
                    dispatchEditOp("quantize", {
                        unit,
                        scale: resolveScaleToken(scaleValue),
                        toleranceCents,
                        tolerance: toleranceCents,
                        quantizeUnit,
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
                onConfirm={(unit, scaleValue, toleranceCents, quantizeUnit) =>
                    dispatchEditOp("meanQuantize", {
                        unit,
                        scale: resolveScaleToken(scaleValue),
                        toleranceCents,
                        tolerance: toleranceCents,
                        quantizeUnit,
                    })
                }
            />
        </Flex>
    );
};
