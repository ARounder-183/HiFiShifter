import React, { useCallback, useEffect, useState } from "react";
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
    undoRemote,
    redoRemote,
    saveProjectRemote,
    saveProjectAsRemote,
} from "../../features/session/sessionSlice";
import {
    importAudioFromDialog,
    importMultipleAudioAtPosition,
} from "../../features/session/thunks/importThunks";
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
import {
    isChildPitchOffsetCentsParam,
    isChildPitchOffsetDegreesParam,
} from "./pianoRoll/childPitchOffsetParams";
// import type { VibratoParams } from "../editDialogs/EditDialogs"; // 已移除无效导入

interface MenuBarProps {
    onNewProject: () => void;
    onOpenProject: () => void;
    onOpenRecentProject: (projectPath: string) => void;
    onExit: () => void;
}

export const MenuBar: React.FC<MenuBarProps> = ({
    onNewProject,
    onOpenProject,
    onOpenRecentProject,
    onExit,
}) => {
    const { t, setLocale } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();
    const s = useAppSelector((state: RootState) => state.session);
    const theme = useAppTheme();
    const keybindings = useAppSelector(selectMergedKeybindings);
    const [kbDialogOpen, setKbDialogOpen] = useState(false);
    const [appearanceDialogOpen, setAppearanceDialogOpen] = useState(false);
    const [exportDialogOpen, setExportDialogOpen] = useState(false);

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

    /** 派发编辑操作事件给 PianoRollPanel */
    const dispatchEditOp = useCallback((op: string, data?: Record<string, unknown>) => {
        window.dispatchEvent(new CustomEvent("hifi:editOp", { detail: { op, ...data } }));
    }, []);

    const handleImportAudioFromMenu = useCallback(async () => {
        try {
            const res = (await dispatch(importAudioFromDialog()).unwrap()) as {
                canceled?: boolean;
                requiresModeChoice?: boolean;
                audioPaths?: string[];
                trackId?: string | null;
                startSec?: number;
            };
            if (res?.canceled || !res?.requiresModeChoice) {
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
                        {t("menu_import_audio")}{" "}
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
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("copy")}>
                        {tAny("menu_copy")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.copy")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("cut")}>
                        {tAny("menu_cut")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("clip.cut")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("paste")}>
                        {tAny("menu_paste")}{" "}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("pianoRoll.paste")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
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
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("initialize")}>
                        {tAny("menu_initialize")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.initialize")}
                        </div>
                    </DropdownMenu.Item>

                    {isPitchParam && (
                        <>
                            <DropdownMenu.Separator />
                            <DropdownMenu.Item onSelect={() => setTransposeCentsOpen(true)}>
                                {tAny("menu_transpose_cents")}
                                <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                                    {shortcutLabel("edit.transposeCents")}
                                </div>
                            </DropdownMenu.Item>
                            <DropdownMenu.Item onSelect={() => setTransposeDegreesOpen(true)}>
                                {tAny("menu_transpose_degrees")}
                                <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                                    {shortcutLabel("edit.transposeDegrees")}
                                </div>
                            </DropdownMenu.Item>
                        </>
                    )}
                    <DropdownMenu.Item onSelect={() => setSetPitchOpen(true)}>
                        {isPitchParam ? tAny("menu_set_pitch") : tAny("menu_set_value")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.setPitch")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setAverageOpen(true)}>
                        {tAny("menu_average")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.average")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setSmoothOpen(true)}>
                        {tAny("menu_smooth")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.smooth")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setVibratoOpen(true)}>
                        {tAny("menu_add_vibrato")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.addVibrato")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setQuantizeOpen(true)}>
                        {tAny("menu_quantize")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.quantize")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => setMeanQuantizeOpen(true)}>
                        {tAny("menu_mean_quantize")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.meanQuantize")}
                        </div>
                    </DropdownMenu.Item>

                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteReaper")}>
                        {t("menu_paste_reaper_clipboard")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteReaper")}
                        </div>
                    </DropdownMenu.Item>
                    <DropdownMenu.Item onSelect={() => dispatchEditOp("pasteVocalShifter")}>
                        {t("menu_paste_vocalshifter_clipboard")}
                        <div className="ml-auto pl-4 text-xs text-qt-text-muted">
                            {shortcutLabel("edit.pasteVocalShifter")}
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
                    <DropdownMenu.Separator />
                    <DropdownMenu.Item onSelect={() => setKbDialogOpen(true)}>
                        {(t as (key: string) => string)("menu_keybindings")}
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

            <ExportAudioDialog open={exportDialogOpen} onOpenChange={setExportDialogOpen} />

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
                                {tAny("import_dialog_title") || t("menu_import_audio")}
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
                projectScaleLabel={projectScaleLabel}
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
                projectScaleLabel={projectScaleLabel}
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
                projectScaleLabel={projectScaleLabel}
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
