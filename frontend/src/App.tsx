import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Flex, Box, Text, Dialog, Button } from "@radix-ui/themes";
import { MenuBar } from "./components/layout/MenuBar";
import { ActionBar } from "./components/layout/ActionBar";
import { TimelinePanel } from "./components/layout/TimelinePanel";
import { PianoRollPanel } from "./components/layout/PianoRollPanel";
import { useAppDispatch, useAppSelector } from "./app/hooks";
import {
    closeVocalShifterSkippedFilesDialog,
    closeReaperSkippedFilesDialog,
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
    openVocalShifterFromPath,
    openReaperFromPath,
    importAudioFromPath,
    saveProjectRemote,
    saveProjectAsRemote,
    exportAudio,
    pickOutputPath,
    setToolMode,
    checkpointHistory,
    addTrackRemote,
} from "./features/session/sessionSlice";
import { useI18n } from "./i18n/I18nProvider";
import { useClipPitchDataListener } from "./hooks/useClipPitchDataListener";
import {
    PitchAnalysisProvider,
    usePitchAnalysis,
} from "./contexts/PitchAnalysisContext";
import {
    PianoRollStatusProvider,
    usePianoRollStatus,
} from "./contexts/PianoRollStatusContext";
import { FileBrowserPanel } from "./components/layout/FileBrowserPanel";
import { QuickSearchPopup } from "./components/layout/QuickSearchPopup";
import { useKeybindings } from "./features/keybindings/useKeybindings";
import type { ActionId } from "./features/keybindings/types";
import { store } from "./app/store";
import { resolveRootTrackId } from "./features/session/trackUtils";
import { getParamShiftStep } from "./components/layout/pianoRoll/paramShiftStep";
import { runConfirmedExitClose } from "./confirmedExitClose";
import { paramsApi } from "./services/api";
import { coreApi } from "./services/api/core";
import { projectApi } from "./services/api/project";
import type { ParamFramesPayload, ProcessorParamDescriptor } from "./types/api";
import { MISSING_FILE_CONFIRM_EVENT } from "./features/session/thunks/missingFilePrompt";
import {
    OPEN_PROJECT_PATH_EVENT,
    type ExternalFileActionDetail,
    type ExternalFileActionKind,
} from "./features/session/projectOpenEvents";
import type { MessageKey } from "./i18n/messages";
import type { CloseRequestedEvent } from "@tauri-apps/api/window";

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
    "Project opened": "status_project_opened",
    "Save canceled": "status_save_canceled",
    "Save failed": "status_save_failed",
    "Save As canceled": "status_save_as_canceled",
    "Save As failed": "status_save_as_failed",
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

function detectExternalActionKindFromPath(
    path: string,
): ExternalFileActionKind | null {
    const normalized = String(path ?? "").trim();
    if (!normalized) return null;
    if (/\.(hshp|hsp)$/i.test(normalized)) return "openProject";
    if (/\.rpp$/i.test(normalized)) return "importReaper";
    if (/\.(vshp|vsp)$/i.test(normalized)) return "importVocalShifter";
    if (/\.(wav|flac|mp3|ogg|m4a|aac|aif|aiff|wma|opus)$/i.test(normalized)) {
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

    const runtimeIsPlaying = useAppSelector(
        (state) => state.session.runtime.isPlaying,
    );
    const runtimeHasSynthesized = useAppSelector(
        (state) => state.session.runtime.hasSynthesized,
    );
    const fileBrowserVisible = useAppSelector(
        (state) => state.fileBrowser.visible,
    );
    const toolMode = useAppSelector((state) => state.session.toolMode);
    const drawToolMode = useAppSelector((state) => state.session.drawToolMode);
    const outputPath = useAppSelector((state) => state.session.outputPath);
    const projectDirty = useAppSelector((state) => state.session.project.dirty);
    // 使用 ref 桥接最新的工程修改状态
    const projectDirtyRef = useRef(projectDirty);
    useEffect(() => {
        projectDirtyRef.current = projectDirty;
    }, [projectDirty]);
    const projectPath = useAppSelector((state) => state.session.project.path);
    const vocalShifterSkippedFilesDialog = useAppSelector(
        (state) => state.session.vocalShifterSkippedFilesDialog,
    );
    const reaperSkippedFilesDialog = useAppSelector(
        (state) => state.session.reaperSkippedFilesDialog,
    );

    const containerRef = useRef<HTMLDivElement | null>(null);
    const dragRef = useRef<{ pointerId: number } | null>(null);
    const [splitRatio, setSplitRatio] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.splitRatio"));
        return Number.isFinite(stored)
            ? Math.min(0.85, Math.max(0.15, stored))
            : 0.6;
    });
    const splitRatioRef = useRef(splitRatio);
    const [isDragging, setIsDragging] = useState(false);
    const [quickSearchOpen, setQuickSearchOpen] = useState(false);
    const [unsavedDialog, setUnsavedDialog] = useState<{
        open: boolean;
        mode: "switch" | "exit";
    }>({ open: false, mode: "switch" });
    const [missingFileDialog, setMissingFileDialog] = useState<{
        open: boolean;
        missingPath: string;
    }>({ open: false, missingPath: "" });
    const pendingUnsavedActionRef = useRef<null | (() => Promise<void>)>(null);
    const allowWindowCloseRef = useRef(false);
    const missingFileResolverRef = useRef<
        ((shouldPick: boolean) => void) | null
    >(null);
    const processorParamCacheRef = useRef(
        new Map<string, ProcessorParamDescriptor[]>(),
    );

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
            if (
                !Number.isFinite(total) ||
                total <= minTopPx + minBottomPx + handlePx
            ) {
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
            localStorage.setItem(
                "hifishifter.splitRatio",
                String(splitRatioRef.current),
            );

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

    // 阻止浏览器默认的 Ctrl+F 搜索、右键菜单和 Alt 键

    // 改用 useRef，取消重绘
    const isModifierRef = useRef(false);

    useEffect(() => {
        function preventBrowserFind(e: KeyboardEvent) {
            const isMac = navigator.platform?.toLowerCase().includes("mac");
            const mod = isMac ? e.metaKey : e.ctrlKey;
            if (mod && e.key.toLowerCase() === "f") {
                e.preventDefault();
            }
            if (mod && e.key.toLowerCase() === "p") {
                e.preventDefault();
            }
        }
        function preventContextMenu(e: MouseEvent) {
            const target = e.target as HTMLElement | null;
            if (target?.tagName === "INPUT" || target?.tagName === "TEXTAREA")
                return;
            if (target?.closest?.("[data-hs-context-menu]")) return;
            e.preventDefault();
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
        document.addEventListener("contextmenu", preventContextMenu, true);
        return () => {
            window.removeEventListener("keydown", preventBrowserFind, true);
            window.removeEventListener("keydown", altKeyDown, true);
            window.removeEventListener("keyup", altKeyUp, true);
            document.removeEventListener(
                "contextmenu",
                preventContextMenu,
                true,
            );
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
              if (
                  pitchAnalysis.totalClips != null &&
                  pitchAnalysis.totalClips > 0
              ) {
                  parts.push(
                      `(${pitchAnalysis.completedClips ?? 0}/${pitchAnalysis.totalClips})`,
                  );
              }
              if (
                  pitchAnalysis.progress != null &&
                  Number.isFinite(pitchAnalysis.progress)
              ) {
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
                unlisten = await mod.listen(
                    "stretch_progress",
                    (event: any) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            active?: boolean;
                            clipName?: string | null;
                        };
                        const active = Boolean(payload?.active);
                        const clipName =
                            typeof payload?.clipName === "string"
                                ? payload.clipName
                                : null;
                        setStretching({ active, clipName });
                    },
                );
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

    // 监听后端波形分析进度事件 (waveform_analysis_progress)
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;
        let fadeOutTimer: ReturnType<typeof setTimeout> | null = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "waveform_analysis_progress",
                    (event: any) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            sourcePath?: string;
                            progress?: number;
                            status?: string;
                        };
                        const status = payload?.status ?? "";
                        const sourcePath =
                            typeof payload?.sourcePath === "string"
                                ? payload.sourcePath
                                : null;
                        const p =
                            typeof payload?.progress === "number" &&
                            Number.isFinite(payload.progress)
                                ? Math.max(0, Math.min(1, payload.progress))
                                : null;

                        if (status === "computing") {
                            // 清除之前的淡出定时器
                            if (fadeOutTimer) {
                                clearTimeout(fadeOutTimer);
                                fadeOutTimer = null;
                            }
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
                                setWaveformAnalysis({
                                    active: true,
                                    sourcePath: null,
                                    progress: 1.0,
                                });
                                fadeOutTimer = setTimeout(() => {
                                    if (!disposed) {
                                        setWaveformAnalysis({
                                            active: false,
                                            sourcePath: null,
                                            progress: null,
                                        });
                                    }
                                }, 1500);
                            }
                            // cached 状态不显示进度条
                        }
                    },
                );
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
                unlisten = await mod.listen(
                    "playback_rendering_state",
                    (event: any) => {
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
                        const target =
                            typeof payload?.target === "string"
                                ? payload.target
                                : null;

                        setRendering({ active, progress: p, target });

                        // 渲染从 active→inactive（完成）时，延迟同步一次播放状态，
                        // 使前端能感知后端已真正开始播放。
                        if (!active && renderingWasActiveRef.current) {
                            setTimeout(() => {
                                dispatch(syncPlaybackState());
                            }, 200);
                        }
                        renderingWasActiveRef.current = active;
                    },
                );
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
        drawToolMode:
            "draw" as import("./features/session/sessionTypes").DrawToolMode,
    });
    const outputPathRef = useRef(outputPath);

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
                const result = await dispatch(
                    projectPath ? saveProjectRemote() : saveProjectAsRemote(),
                ).unwrap();
                if ((result as { canceled?: boolean } | undefined)?.canceled) {
                    return;
                }
                await executePendingUnsavedAction();
            } catch {
                // Keep the dialog open so the user can retry or cancel.
            }
        })();
    }, [dispatch, executePendingUnsavedAction, projectPath]);

    const handleNewProject = useCallback(() => {
        runOrPromptUnsavedAction("switch", async () => {
            await dispatch(newProjectRemote()).unwrap();
        });
    }, [dispatch, runOrPromptUnsavedAction]);

    const handleOpenProject = useCallback(() => {
        runOrPromptUnsavedAction("switch", async () => {
            await dispatch(openProjectFromDialog()).unwrap();
        });
    }, [dispatch, runOrPromptUnsavedAction]);

    const handleOpenRecentProject = useCallback(
        (path: string) => {
            runOrPromptUnsavedAction("switch", async () => {
                await dispatch(openProjectFromPath(path)).unwrap();
            });
        },
        [dispatch, runOrPromptUnsavedAction],
    );

    const handleExternalFileAction = useCallback(
        (kind: ExternalFileActionKind, path: string) => {
            const normalized = String(path ?? "").trim();
            if (!normalized) return;
            if (kind === "openProject") {
                runOrPromptUnsavedAction("switch", async () => {
                    await dispatch(openProjectFromPath(normalized)).unwrap();
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
        [dispatch, runOrPromptUnsavedAction],
    );

    const handleExitApp = useCallback(() => {
        runOrPromptUnsavedAction("exit", closeWindowNow);
    }, [closeWindowNow, runOrPromptUnsavedAction]);

    useEffect(() => {
        void dispatch(fetchTimeline());
        void dispatch(refreshRuntime());
        void dispatch(loadUiSettings());
    }, [dispatch]);

    useEffect(() => {
        let canceled = false;

        async function consumeStartupProjectPath() {
            try {
                const result = await projectApi.consumeStartupProjectPath();
                const startupPath = String(result?.path ?? "").trim();
                const kind = detectExternalActionKindFromPath(startupPath);
                if (!canceled && startupPath && kind) {
                    handleExternalFileAction(kind, startupPath);
                }
            } catch {
                // no-op
            }
        }

        void consumeStartupProjectPath();
        return () => {
            canceled = true;
        };
    }, [handleExternalFileAction]);

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
            window.removeEventListener(
                OPEN_PROJECT_PATH_EVENT,
                onOpenProjectPath as EventListener,
            );
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
        outputPathRef.current = outputPath;
    }, [outputPath]);

    useEffect(() => {
        const handler = (event: Event) => {
            const detail = (
                event as CustomEvent<{
                    missingPath?: string;
                    resolve?: (shouldPick: boolean) => void;
                }>
            ).detail;
            if (!detail || typeof detail.resolve !== "function") return;
            missingFileResolverRef.current = detail.resolve;
            setMissingFileDialog({
                open: true,
                missingPath:
                    typeof detail.missingPath === "string"
                        ? detail.missingPath
                        : "",
            });
        };
        window.addEventListener(
            MISSING_FILE_CONFIRM_EVENT,
            handler as EventListener,
        );
        return () => {
            window.removeEventListener(
                MISSING_FILE_CONFIRM_EVENT,
                handler as EventListener,
            );
            if (missingFileResolverRef.current) {
                missingFileResolverRef.current(false);
                missingFileResolverRef.current = null;
            }
        };
    }, []);

    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/window");
                const currentWindow = mod.getCurrentWindow();
                unlisten = await currentWindow.onCloseRequested(
                    (event: CloseRequestedEvent) => {
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
                    },
                );
            } catch {}
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [closeWindowNow, promptUnsavedAction]); // 剔除 projectDirty 依赖，只绑定一次

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
                        void dispatch(
                            stopAudioPlayback({ restoreAnchor: true }),
                        );
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
                    void (async () => {
                        const curPath = outputPathRef.current?.trim();
                        if (!curPath) {
                            const picked =
                                await dispatch(pickOutputPath()).unwrap();
                            if (picked.ok && !picked.canceled && picked.path) {
                                await dispatch(exportAudio(picked.path));
                            }
                            return;
                        }
                        await dispatch(exportAudio(curPath));
                    })();
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
                case "pianoRoll.shiftParamUp":
                case "pianoRoll.shiftParamDown": {
                    const isUp = actionId === "pianoRoll.shiftParamUp";
                    const ss = store.getState().session;
                    const rootTrkId = resolveRootTrackId(
                        ss.tracks,
                        ss.selectedTrackId,
                    );
                    if (!rootTrkId) break;
                    const editP = ss.editParam;
                    const rootTrk = ss.tracks.find((tr) => tr.id === rootTrkId);
                    // pitch 参数需要 pitch 分析可用才能操作
                    if (editP === "pitch") {
                        if (
                            !rootTrk?.composeEnabled ||
                            rootTrk.pitchAnalysisAlgo === "none"
                        )
                            break;
                    }
                    const selClipId = ss.selectedClipId;
                    // 优先使用多选 clip 列表，否则 fallback 到单选
                    const multiIds = ss.multiSelectedClipIds;
                    const clipIds =
                        multiIds.length >= 1
                            ? multiIds
                            : selClipId
                              ? [selClipId]
                              : [];
                    if (clipIds.length === 0) break;
                    const selClips = ss.clips.filter((c) =>
                        clipIds.includes(c.id),
                    );
                    if (selClips.length === 0) break;
                    const minSec = Math.min(...selClips.map((c) => c.startSec));
                    const maxSec = Math.max(
                        ...selClips.map((c) => c.startSec + c.lengthSec),
                    );
                    // 默认 framePeriodMs = 5
                    const fp = 5;
                    const startFrame = Math.max(
                        0,
                        Math.floor((minSec * 1000) / fp),
                    );
                    const frameCount = Math.max(
                        1,
                        Math.min(
                            200_000,
                            Math.ceil(((maxSec - minSec) * 1000) / fp),
                        ),
                    );
                    void (async () => {
                        let descriptor: ProcessorParamDescriptor | undefined;
                        if (editP !== "pitch" && rootTrk?.pitchAnalysisAlgo) {
                            const algo = rootTrk.pitchAnalysisAlgo;
                            let descriptors =
                                processorParamCacheRef.current.get(algo);
                            if (!descriptors) {
                                try {
                                    descriptors =
                                        await paramsApi.getProcessorParams(
                                            algo,
                                        );
                                    processorParamCacheRef.current.set(
                                        algo,
                                        descriptors,
                                    );
                                } catch {
                                    descriptors = undefined;
                                }
                            }
                            descriptor = descriptors?.find(
                                (param) => param.id === editP,
                            );
                        }
                        const step = getParamShiftStep(editP, descriptor);
                        const delta = isUp ? step : -step;
                        const clampNum = (
                            v: number,
                            minV: number,
                            maxV: number,
                        ) => Math.min(maxV, Math.max(minV, v));
                        const smoothness = clampNum(
                            Number(ss.edgeSmoothnessPercent) || 0,
                            0,
                            100,
                        );
                        const maxTransitionFrames = Math.floor(frameCount / 2);
                        const transitionFrames =
                            smoothness > 0 && maxTransitionFrames > 0
                                ? Math.round(
                                      (smoothness / 100) * maxTransitionFrames,
                                  )
                                : 0;
                        const halfSpan =
                            transitionFrames > 0 ? transitionFrames / 2 : 0;
                        const extend = Math.max(0, Math.ceil(halfSpan));
                        const extStart = Math.max(0, startFrame - extend);
                        const extCount =
                            frameCount +
                            Math.max(0, startFrame - extStart) +
                            extend;
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
                                boundaryCount > 0
                                    ? boundaryDelta / boundaryCount
                                    : 0;
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
                                    for (
                                        let idx = left;
                                        idx <= right;
                                        idx += 1
                                    ) {
                                        const t = clampNum(
                                            (idx - (selOffset - halfSpan)) /
                                                span,
                                            0,
                                            1,
                                        );
                                        const outsideIdx = Math.min(
                                            selOffset - 1,
                                            idx,
                                        );
                                        const insideIdx = Math.max(
                                            selOffset,
                                            idx,
                                        );
                                        const outsideVal =
                                            snapshot[outsideIdx] ??
                                            editedDense[idx];
                                        const insideVal =
                                            snapshot[insideIdx] ??
                                            editedDense[idx];
                                        const smoothed =
                                            outsideVal +
                                            (insideVal - outsideVal) * t;
                                        editedDense[idx] =
                                            snapshot[idx] +
                                            (smoothed - snapshot[idx]) *
                                                changeFactor;
                                    }
                                }
                                if (selEnd < editedDense.length - 1) {
                                    const left = Math.max(
                                        0,
                                        Math.floor(selEnd - halfSpan),
                                    );
                                    const right = Math.min(
                                        editedDense.length - 1,
                                        Math.ceil(selEnd + halfSpan),
                                    );
                                    for (
                                        let idx = left;
                                        idx <= right;
                                        idx += 1
                                    ) {
                                        const t = clampNum(
                                            (idx - (selEnd - halfSpan)) / span,
                                            0,
                                            1,
                                        );
                                        const insideIdx = Math.min(selEnd, idx);
                                        const outsideIdx = Math.max(
                                            selEnd + 1,
                                            idx,
                                        );
                                        const insideVal =
                                            snapshot[insideIdx] ??
                                            editedDense[idx];
                                        const outsideVal =
                                            snapshot[outsideIdx] ??
                                            editedDense[idx];
                                        const smoothed =
                                            insideVal +
                                            (outsideVal - insideVal) * t;
                                        editedDense[idx] =
                                            snapshot[idx] +
                                            (smoothed - snapshot[idx]) *
                                                changeFactor;
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
                                    actionId ===
                                    "pianoRoll.shiftParamUpSelection"
                                        ? "shiftParamUpSelection"
                                        : "shiftParamDownSelection",
                            },
                        }),
                    );
                    break;
                }
                case "edit.pasteReaper":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "pasteReaper" },
                        }),
                    );
                    break;
                case "edit.pasteVocalShifter":
                    window.dispatchEvent(
                        new CustomEvent("hifi:editOp", {
                            detail: { op: "pasteVocalShifter" },
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
            // 预渲染阶段后端还未真正进入 playing，
            // 若此时同步会把前端“准备播放”状态误判为停止，导致 stop 锚点丢失。
            if (rendering.active) return;
            if (playbackSyncInFlightRef.current) return;
            playbackSyncInFlightRef.current = true;
            const p = dispatch(
                syncPlaybackState(),
            ) as unknown as Promise<unknown>;
            p.finally(() => {
                playbackSyncInFlightRef.current = false;
            });
        }, intervalMs);
        return () => window.clearInterval(id);
    }, [dispatch, runtimeIsPlaying, rendering.active]);

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
                    <Dialog.Description>
                        {t("vs_import_skipped_header")}
                    </Dialog.Description>
                    <div className="mt-2 max-h-[240px] overflow-auto rounded border border-qt-border bg-qt-base p-2 text-xs">
                        {(vocalShifterSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" title={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                    <Flex justify="end" mt="3">
                        <Button
                            onClick={() =>
                                dispatch(closeVocalShifterSkippedFilesDialog())
                            }
                        >
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
                    <Dialog.Description>
                        {t("reaper_import_skipped_header")}
                    </Dialog.Description>
                    <div className="mt-2 max-h-[240px] overflow-auto rounded border border-qt-border bg-qt-base p-2 text-xs">
                        {(reaperSkippedFilesDialog ?? []).map((file) => (
                            <div key={file} className="truncate" title={file}>
                                • {file}
                            </div>
                        ))}
                    </div>
                    <Flex justify="end" mt="3">
                        <Button
                            onClick={() =>
                                dispatch(closeReaperSkippedFilesDialog())
                            }
                        >
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
                        <Button
                            variant="soft"
                            color="gray"
                            onClick={cancelUnsavedAction}
                        >
                            {t("progress_cancel")}
                        </Button>
                        <Button
                            variant="soft"
                            color="gray"
                            onClick={discardUnsavedAndContinue}
                        >
                            {t("unsaved_changes_discard")}
                        </Button>
                        <Button onClick={saveUnsavedAndContinue}>
                            {t("menu_save_project")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <Dialog.Root
                open={missingFileDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        setMissingFileDialog((prev) => ({
                            ...prev,
                            open: false,
                        }));
                        if (missingFileResolverRef.current) {
                            missingFileResolverRef.current(false);
                            missingFileResolverRef.current = null;
                        }
                    }
                }}
            >
                <Dialog.Content maxWidth="560px">
                    <Dialog.Title>
                        {t("missing_file_replace_title")}
                    </Dialog.Title>
                    <Dialog.Description>
                        {t("missing_file_replace_desc")}
                    </Dialog.Description>
                    <div className="mt-2 rounded border border-qt-border bg-qt-base p-2 text-xs break-all">
                        {missingFileDialog.missingPath}
                    </div>
                    <Flex justify="end" gap="2" mt="4">
                        <Button
                            variant="soft"
                            color="gray"
                            onClick={() => {
                                setMissingFileDialog((prev) => ({
                                    ...prev,
                                    open: false,
                                }));
                                if (missingFileResolverRef.current) {
                                    missingFileResolverRef.current(false);
                                    missingFileResolverRef.current = null;
                                }
                            }}
                        >
                            {t("cancel")}
                        </Button>
                        <Button
                            onClick={() => {
                                setMissingFileDialog((prev) => ({
                                    ...prev,
                                    open: false,
                                }));
                                if (missingFileResolverRef.current) {
                                    missingFileResolverRef.current(true);
                                    missingFileResolverRef.current = null;
                                }
                            }}
                        >
                            {t("missing_file_replace_pick")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>

            <MenuBar
                onNewProject={handleNewProject}
                onOpenProject={handleOpenProject}
                onOpenRecentProject={handleOpenRecentProject}
                onExit={handleExitApp}
            />
            <ActionBar />

            {/* Main Content Area: Splitter + optional File Browser */}
            <Flex className="flex-1 min-h-0">
                {/* Left: Timeline / PianoRoll vertical splitter */}
                <div
                    ref={containerRef}
                    className="flex-1 min-w-0 min-h-0 flex flex-col"
                >
                    {/* Top: Timeline / Tracks */}
                    <Box
                        className="min-h-[200px] border-b border-qt-border relative bg-qt-base"
                        style={{ flexGrow: splitRatio, flexBasis: 0 }}
                    >
                        <TimelinePanel />
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

                {/* Right: File Browser Panel (可收起) */}
                {fileBrowserVisible && (
                    <div className="w-[280px] shrink-0 border-l border-qt-border bg-qt-window flex flex-col">
                        <FileBrowserPanel />
                    </div>
                )}
            </Flex>

            {/* Quick Search Popup */}
            <QuickSearchPopup
                open={quickSearchOpen}
                onClose={() => setQuickSearchOpen(false)}
            />

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
                            {stretching.clipName
                                ? ` "${stretching.clipName}"`
                                : ""}
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
                            {waveformAnalysis.sourcePath
                                ? ` "${waveformAnalysis.sourcePath}"`
                                : ""}
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
                            {t("refreshing_pitch_data") ||
                                "Refreshing pitch data"}
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
                    <Text
                        size="1"
                        color={error ? "red" : "gray"}
                        className="truncate"
                    >
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
