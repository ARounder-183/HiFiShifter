import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Flex, Dialog, Button, Text } from "@radix-ui/themes";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    addTrackRemote,
    checkpointHistory,
    createClipsRemote,
    duplicateTrackRemote,
    removeTrackRemote,
    selectTrackRemote,
    setTrackStateRemote,
    seekPlayhead,
    selectClipRemote,
    setplayheadSec,
    moveTrackRemote,
    setClipMuted,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMultipleAudioAtPosition,
    setClipStateRemote,
    replaceClipSourceRemote,
    setClipGain,
    setClipFades,
    glueClipsRemote,
    removeClipRemote,
    splitClipRemote,
    setMultiSelectedClipIds as setMultiSelectedClipIdsAction,
    setSelectedClip,
    setSelectedClipPreservingTrack,
    setTrackName,
} from "../../features/session/sessionSlice";

import type { ClipTemplate } from "../../features/session/sessionTypes";
import { dbToGain } from "./timeline/math";
import { computeAnchoredHorizontalZoom } from "../../utils/horizontalZoom";
import {
    NEW_TRACK_SENTINEL,
    useClipDrag,
} from "./timeline/hooks/useClipDrag";
import { useEditDrag } from "./timeline/hooks/useEditDrag";
import { useSlipDrag } from "./timeline/hooks/useSlipDrag";
import { useKeyboardShortcuts } from "./timeline/hooks/useKeyboardShortcuts";
import { computeAutoCrossfadeFromPayload } from "./timeline/hooks/autoCrossfade";
import { collectFadeContextClips } from "./timeline/clipFadeContext";
import { selectKeybinding } from "../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../features/keybindings/types";
import { webApi } from "../../services/webviewApi";
import { getDynamicProjectSec } from "../../features/session/projectBoundary";
import { emitExternalFileAction } from "../../features/session/projectOpenEvents";
import { waveformMipmapStore } from "../../utils/waveformMipmapStore";

import {
    BackgroundGrid,
    ClipContextMenu,
    DEFAULT_PX_PER_SEC,
    DEFAULT_ROW_HEIGHT,
    MAX_PX_PER_SEC,
    MAX_ROW_HEIGHT,
    MIN_PX_PER_SEC,
    MIN_ROW_HEIGHT,
    TRACK_ADD_ROW_HEIGHT,
    TrackAreaContextMenu,
    TimelineScrollArea,
    TimeRuler,
    TrackLane,
    TrackList,
    detectExternalPathAction,
    findFirstExternalPathAction,
    useTimelineSelectionRect,
    extractLocalFilePath,
    gridStepBeats,
    hasFileDrag,
} from "./timeline";

export const TimelinePanel: React.FC = () => {
    const dispatch = useAppDispatch();
    const { t } = useI18n();
    const s = useAppSelector((state: RootState) => state.session);
    const sessionRef = useRef(s);
    useEffect(() => {
        sessionRef.current = s;
    }, [s]);
    const scrollRef = useRef<HTMLDivElement | null>(null);
    const trackListScrollRef = useRef<HTMLDivElement | null>(null);
    const rulerContentRef = useRef<HTMLDivElement | null>(null);
    const scrollLeftRef = useRef(0);
    const scrollStateRafRef = useRef<number | null>(null);
    const playheadDragRef = useRef<{
        pointerId: number;
        lastBeat: number;
    } | null>(null);
    const lastClickedClipIdRef = useRef<string | null>(null);
    const playheadRef = useRef<HTMLDivElement | null>(null);
    const dropPreviewRef = useRef<HTMLDivElement | null>(null);
    const pendingDropDurationPathRef = useRef<string | null>(null);
    const [scrollLeft, setScrollLeft] = useState(0);
    const [pxPerSec, setPxPerSec] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.pxPerSec"));
        return Number.isFinite(stored) && stored > 0
            ? Math.min(MAX_PX_PER_SEC, Math.max(MIN_PX_PER_SEC, stored))
            : DEFAULT_PX_PER_SEC;
    });
    const pxPerSecRef = useRef(pxPerSec);
    useEffect(() => {
        pxPerSecRef.current = pxPerSec;
    }, [pxPerSec]);
    const keyboardZoomPendingRef = useRef<{
        nextScale: number;
        nextScrollLeft: number;
    } | null>(null);
    const [viewportWidth, setViewportWidth] = useState(0);
    const [sameSourceConfirmOpen, setSameSourceConfirmOpen] = useState(false);
    const sameSourceConfirmResolverRef = useRef<((confirmed: boolean) => void) | null>(null);
    useEffect(() => {
        scrollLeftRef.current = scrollLeft;
    }, [scrollLeft]);

    useEffect(() => {
        const scroller = scrollRef.current;
        if (!scroller) return;

        const updateViewportWidth = () => {
            setViewportWidth(scroller.clientWidth || 0);
        };

        updateViewportWidth();

        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(() => {
                updateViewportWidth();
            });
            observer.observe(scroller);
            return () => {
                observer.disconnect();
            };
        }

        window.addEventListener("resize", updateViewportWidth);
        return () => {
            window.removeEventListener("resize", updateViewportWidth);
        };
    }, []);

    function syncScrollLeft(next: number) {
        scrollLeftRef.current = next;
        if (rulerContentRef.current) {
            rulerContentRef.current.style.transform = `translateX(${-next}px)`;
        }
        if (scrollStateRafRef.current == null) {
            scrollStateRafRef.current = requestAnimationFrame(() => {
                scrollStateRafRef.current = null;
                setScrollLeft(scrollLeftRef.current);
            });
        }
    }

    const setScrollLeftAction: React.Dispatch<React.SetStateAction<number>> = (
        action,
    ) => {
        const next =
            typeof action === "function"
                ? (action as (prev: number) => number)(scrollLeftRef.current)
                : action;
        syncScrollLeft(next);
    };

    useLayoutEffect(() => {
        const pending = keyboardZoomPendingRef.current;
        if (!pending) return;
        if (Math.abs(pending.nextScale - pxPerSec) > 1e-9) return;
        const scroller = scrollRef.current;
        if (!scroller) return;

        keyboardZoomPendingRef.current = null;
        scroller.scrollLeft = pending.nextScrollLeft;
        syncScrollLeft(pending.nextScrollLeft);
    }, [pxPerSec]);
    // 渲染时根据 BPM 计算 pxPerBeat，仅用于网格/标尺渲染
    // pxPerBeat = pxPerSec × secPerBeat = pxPerSec × (60 / bpm)
    const secPerBeat = 60 / Math.max(1, s.bpm);
    const pxPerBeat = pxPerSec * secPerBeat;
    // clip 位置/宽度/交互坐标统一使用 pxPerSec（不随 BPM 变化）

    const [rowHeight, setRowHeight] = useState(() => {
        const stored = Number(localStorage.getItem("hifishifter.rowHeight"));
        return Number.isFinite(stored)
            ? Math.min(MAX_ROW_HEIGHT, Math.max(MIN_ROW_HEIGHT, stored))
            : DEFAULT_ROW_HEIGHT;
    });

    // 静默访问
    const rowHeightRef = useRef(rowHeight);
    useEffect(() => { rowHeightRef.current = rowHeight; }, [rowHeight]);

    const panRef = useRef<{
        pointerId: number | null;
        startX: number;
        startY: number;
        scrollLeft: number;
        scrollTop: number;
    } | null>(null);

    // clip ��ק���༭��ק��slip ��ק�� ref �����������ɸ��� hook �ṩ��
    // ���·� snapBeat / beatFromClientX / trackIdFromClientY ������ʼ����

    const [trackVolumeUi, setTrackVolumeUi] = useState<Record<string, number>>(
        {},
    );
    const [dropPreview, setDropPreview] = useState<{
        path: string;
        fileName: string;
        trackId: string | null;
        startSec: number;
        durationSec: number;
    } | null>(null);

    const [clipDropNewTrack, setClipDropNewTrack] = useState(false);

    const [altPressed, setAltPressed] = useState(false);

    // 从 store 读取 modifier.clipStretch 绑定，动态检测对应修饰键
    const stretchKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipStretch"),
    );
    const slipEditKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipSlipEdit"),
    );
    const noSnapKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipNoSnap"),
    );
    const copyDragKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipCopyDrag"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    const stretchKbRef = useRef<Keybinding>(stretchKb);
    useEffect(() => {
        stretchKbRef.current = stretchKb;
    }, [stretchKb]);

    const clipClipboardRef = useRef<ClipTemplate[] | null>(null);

    const buildClipClipboardTemplates = React.useCallback(
        async (ids: string[]) => {
            const clips = sessionRef.current.clips.filter((c) =>
                ids.includes(c.id),
            );
            return Promise.all(
                clips.map(async (clip) => {
                    const linkedParamsResult = await webApi.getClipLinkedParams(
                        clip.id,
                    );
                    return {
                        ...clip,
                        waveformPreview:
                            sessionRef.current.clipWaveforms[clip.id],
                        linkedParams: linkedParamsResult.ok
                            ? linkedParamsResult.linkedParams
                            : undefined,
                    };
                }),
            );
        },
        [],
    );

    const tauriDraggedPathRef = useRef<string | null>(null);
    const tauriLastDropPathRef = useRef<string | null>(null);
    const tauriDropHandledAtRef = useRef<number>(0);

    useEffect(() => {
        /** 检测当前按下的键是否匹配 modifier.clipStretch 绑定 */
        function isStretchModifier(e: KeyboardEvent): boolean {
            const kb = stretchKbRef.current;
            if (kb.ctrl && (e.key === "Control" || e.ctrlKey || e.metaKey))
                return true;
            if (kb.alt && (e.key === "Alt" || e.altKey)) return true;
            if (kb.shift && (e.key === "Shift" || e.shiftKey)) return true;
            return false;
        }
        function checkStretchState(e: KeyboardEvent): boolean {
            const kb = stretchKbRef.current;
            if (kb.ctrl) return e.ctrlKey || e.metaKey;
            if (kb.alt) return e.altKey;
            if (kb.shift) return e.shiftKey;
            return false;
        }
        function onKeyDown(e: KeyboardEvent) {
            if (isStretchModifier(e)) setAltPressed(true);
        }
        function onKeyUp(e: KeyboardEvent) {
            if (!checkStretchState(e)) setAltPressed(false);
        }
        function onBlur() {
            setAltPressed(false);
        }
        window.addEventListener("keydown", onKeyDown, true);
        window.addEventListener("keyup", onKeyUp, true);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keydown", onKeyDown, true);
            window.removeEventListener("keyup", onKeyUp, true);
            window.removeEventListener("blur", onBlur);
        };
    }, []);

    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        const debugDnd = localStorage.getItem("hifishifter.debugDnd") === "1";

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/window");
                const win = mod.getCurrentWindow();

                if (debugDnd) {
                    console.log("[dnd] attaching tauri drag-drop listener");
                }

                type TauriDragDropPayload = {
                    type?: string;
                    event?: string;
                    paths?: string[];
                    position?: { x?: number; y?: number };
                    pos?: { x?: number; y?: number };
                    cursorPosition?: { x?: number; y?: number };
                };

                type TauriDragDropEvent =
                    | { payload?: TauriDragDropPayload }
                    | TauriDragDropPayload;

                unlisten = await win.onDragDropEvent(
                    (event: TauriDragDropEvent) => {
                        if (disposed) return;
                        const payload = (
                            "payload" in event ? event.payload : event
                        ) as TauriDragDropPayload | undefined;
                        const type = String(
                            payload?.type ?? payload?.event ?? "",
                        );
                        const paths: string[] = Array.isArray(payload?.paths)
                            ? payload.paths
                            : [];

                        if (debugDnd) {
                            console.log("[dnd] tauri event", {
                                type,
                                pathsCount: paths.length,
                                hasPosition: Boolean(
                                    payload?.position ??
                                    payload?.pos ??
                                    payload?.cursorPosition,
                                ),
                            });
                        }

                        // Some platforms don't dispatch DOM dragover/drop for external file drags.
                        // Use Tauri's drag-drop event as the source of truth, including ghost + import.
                        const scroller = scrollRef.current;
                        const bounds =
                            scroller?.getBoundingClientRect() ?? null;
                        const pos = (payload?.position ??
                            payload?.pos ??
                            payload?.cursorPosition) as
                            | { x?: number; y?: number }
                            | undefined;
                        // Tauri reports physical (screen) pixels; convert to CSS logical pixels.
                        const dpr = window.devicePixelRatio || 1;
                        const clientX =
                            typeof pos?.x === "number"
                                ? pos.x / dpr
                                : undefined;
                        const clientY =
                            typeof pos?.y === "number"
                                ? pos.y / dpr
                                : undefined;
                        const fallbackBeat =
                            sessionRef.current.playheadSec ?? 0;
                        const beat =
                            clientX !== undefined && bounds && scroller
                                ? beatFromClientX(
                                    clientX,
                                    bounds,
                                    scroller.scrollLeft,
                                )
                                : fallbackBeat;
                        const trackId =
                            clientY !== undefined
                                ? trackIdFromClientY(clientY)
                                : null;

                        const primaryPath = paths.length > 0 ? paths[0] : null;
                        function fileNameFromPath(p: string) {
                            return String(p.split(/[\\/]/).pop() ?? p);
                        }

                        if (type === "enter" || type === "over") {
                            if (primaryPath) {
                                tauriDraggedPathRef.current = primaryPath;
                                if (detectExternalPathAction(primaryPath) === "importAudio") {
                                    ensureDropPreviewDuration(primaryPath);
                                }
                            }
                            // On some platforms, `paths` may only be populated on `drop`.
                            // Still update the ghost position on every `over`.
                            setDropPreview((prev) => {
                                const path = primaryPath ?? tauriDraggedPathRef.current ?? prev?.path ?? null;
                                if (!path) return prev;
                                if (detectExternalPathAction(path) !== "importAudio") {
                                    return null;
                                }
                                const nextFileName = fileNameFromPath(path);

                                // 直接修改 DOM
                                if (prev && dropPreviewRef.current) {
                                    dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                                    dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                                    dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                                }

                                if (!prev || prev.trackId !== trackId || prev.path !== path) {
                                    return {
                                        path,
                                        fileName: prev?.fileName ?? nextFileName,
                                        trackId,
                                        startSec: beat,
                                        durationSec: prev?.durationSec ?? 0,
                                    };
                                }
                                return prev;
                            });
                            return;
                        }

                        if (type === "leave") {
                            tauriDraggedPathRef.current = null;
                            setDropPreview(null);
                            return;
                        }

                        if (type === "drop") {
                            if (primaryPath) {
                                tauriDraggedPathRef.current = primaryPath;
                                tauriLastDropPathRef.current = primaryPath;
                            }
                            if (
                                primaryPath &&
                                detectExternalPathAction(primaryPath) === "importAudio"
                            ) {
                                ensureDropPreviewDuration(primaryPath);
                            }
                            setDropPreview(null);

                            const externalAction = findFirstExternalPathAction(paths);
                            if (
                                externalAction &&
                                externalAction.kind !== "importAudio"
                            ) {
                                tauriDropHandledAtRef.current = Date.now();
                                tauriDraggedPathRef.current = null;
                                tauriLastDropPathRef.current = null;
                                emitExternalFileAction(
                                    externalAction.kind,
                                    externalAction.path,
                                );
                                return;
                            }

                            // Multi-file drop
                            if (paths.length > 1) {
                                tauriDropHandledAtRef.current = Date.now();
                                tauriDraggedPathRef.current = null;
                                tauriLastDropPathRef.current = null;

                                // TODO: Right Drag Support
                                /* const isRightDrag = false;
                                if (isRightDrag) {
                                    // Suppress the contextmenu event that may fire after right-click drag
                                    const suppressCtx = (ev: Event) => {
                                        ev.preventDefault();
                                        ev.stopImmediatePropagation();
                                        window.removeEventListener("contextmenu", suppressCtx, true);
                                    };
                                    window.addEventListener("contextmenu", suppressCtx, true);

                                    setImportModeMenu({
                                        x: clientX ?? window.innerWidth / 2,
                                        y: clientY ?? window.innerHeight / 2,
                                        audioPaths: paths,
                                        trackId,
                                        startSec: beat,
                                    });
                                    return;
                                } */

                                void dispatch(
                                    importMultipleAudioAtPosition({
                                        audioPaths: paths,
                                        mode: "across-time",
                                        trackId,
                                        startSec: beat,
                                    }),
                                );

                                return;
                            }

                            const resolvedPath =
                                primaryPath ||
                                tauriDraggedPathRef.current ||
                                tauriLastDropPathRef.current;
                            if (resolvedPath) {
                                tauriDropHandledAtRef.current = Date.now();
                                tauriDraggedPathRef.current = null;
                                tauriLastDropPathRef.current = null;
                                const actionKind = detectExternalPathAction(resolvedPath);
                                if (actionKind && actionKind !== "importAudio") {
                                    emitExternalFileAction(actionKind, resolvedPath);
                                    return;
                                }
                                void dispatch(
                                    importAudioAtPosition({
                                        audioPath: resolvedPath,
                                        trackId,
                                        startSec: beat,
                                    }),
                                );
                            }
                        }
                    },
                );
            } catch (err) {
                // Safe no-op: browser/pywebview builds won't have the Tauri API.
                if (debugDnd) {
                    console.warn(
                        "Failed to attach Tauri drag-drop listener",
                        err,
                    );
                }
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [dispatch]);

    // ========== 监听文件浏览器面板的自定义拖拽事件 ==========
    useEffect(() => {
        function onHifiFileDrag(e: Event) {
            const detail = (e as CustomEvent).detail as {
                type: string;
                filePath: string;
                fileName: string;
                clientX: number;
                clientY: number;
            };
            if (!detail) return;

            const scroller = scrollRef.current;
            const bounds = scroller?.getBoundingClientRect() ?? null;

            // 判断鼠标是否在 timeline 滚动区域内
            const isOverTimeline =
                bounds &&
                detail.clientX >= bounds.left &&
                detail.clientX <= bounds.right &&
                detail.clientY >= bounds.top &&
                detail.clientY <= bounds.bottom;

            // 异步获取到音频时长后仅更新 ghost 宽度
            if (detail.type === "duration") {
                setDropPreview((prev) => {
                    if (prev && prev.path === detail.filePath) {
                        if (dropPreviewRef.current) {
                            const nextDuration =
                                Number((detail as any).durationSec) || 0;
                            dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(nextDuration)}px`;
                        }
                        return {
                            ...prev,
                            durationSec: (detail as any).durationSec,
                        };
                    }
                    return prev;
                });
                return;
            }

            // 移动时的 DOM 直通与重绘拦截
            if (detail.type === "move" || detail.type === "start") {
                if (isOverTimeline && scroller) {
                    const beat = beatFromClientX(
                        detail.clientX,
                        bounds!,
                        scroller.scrollLeft,
                    );
                    const trackId = trackIdFromClientY(detail.clientY);
                    const path = detail.filePath;
                    const fileName = detail.fileName;

                    if (detectExternalPathAction(path) !== "importAudio") {
                        setDropPreview(null);
                        return;
                    }

                    setDropPreview((prev) => {
                        // 如果幽灵框存在，直接通过 DOM 修改其位置坐标
                        if (prev && dropPreviewRef.current) {
                            dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                            dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                            dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                        }
                        // 如果切换了轨道或文件，才触发真正的 React 重绘
                        if (!prev || prev.trackId !== trackId || prev.path !== path) {
                            ensureDropPreviewDuration(path);
                            return {
                                path,
                                fileName,
                                trackId,
                                startSec: beat,
                                durationSec: prev?.path === path ? prev.durationSec : 2,
                            };
                        }
                        return prev;
                    });
                } else {
                    setDropPreview(null);
                }
                return;
            }

            if (detail.type === "drop") {
                setDropPreview(null);
                if (isOverTimeline && scroller) {
                    const beat = beatFromClientX(
                        detail.clientX,
                        bounds!,
                        scroller.scrollLeft,
                    );
                    const trackId = trackIdFromClientY(detail.clientY);
                    const filePaths: string[] = (detail as any).filePaths;
                    const isRightDrag = !!(detail as any).isRightDrag;
                    const isMulti =
                        Array.isArray(filePaths) && filePaths.length > 1;

                    if (isRightDrag) {
                        // 右键松开后浏览器会立即触发 contextmenu 事件，
                        // 如果不拦截，该事件会被 backdrop 的 onContextMenu 捕获，
                        // 导致刚弹出的导入模式菜单立刻被关闭。
                        // 添加一次性 capturing-phase 拦截器来吞掉那个多余的 contextmenu。
                        const suppressCtx = (ev: Event) => {
                            ev.preventDefault();
                            ev.stopImmediatePropagation();
                            window.removeEventListener(
                                "contextmenu",
                                suppressCtx,
                                true,
                            );
                        };
                        window.addEventListener(
                            "contextmenu",
                            suppressCtx,
                            true,
                        );

                        // 右键拖拽 → 弹出导入模式菜单（跟随鼠标位置）
                        setImportModeMenu({
                            x: detail.clientX,
                            y: detail.clientY,
                            audioPaths: isMulti ? filePaths : [detail.filePath],
                            trackId,
                            startSec: beat,
                        });
                    } else if (isMulti) {
                        const externalAction = findFirstExternalPathAction(filePaths);
                        if (externalAction && externalAction.kind !== "importAudio") {
                            emitExternalFileAction(
                                externalAction.kind,
                                externalAction.path,
                            );
                            return;
                        }
                        // 左键多文件拖拽 → 直接跨时间导入
                        void dispatch(
                            importMultipleAudioAtPosition({
                                audioPaths: filePaths,
                                mode: "across-time",
                                trackId,
                                startSec: beat,
                            }),
                        );
                    } else {
                        const actionKind = detectExternalPathAction(detail.filePath);
                        if (actionKind && actionKind !== "importAudio") {
                            emitExternalFileAction(actionKind, detail.filePath);
                            return;
                        }
                        // 左键单文件拖拽 → 直接导入
                        void dispatch(
                            importAudioAtPosition({
                                audioPath: detail.filePath,
                                trackId,
                                startSec: beat,
                            }),
                        );
                    }
                }
                return;
            }
        }

        window.addEventListener("hifi-file-drag", onHifiFileDrag);
        return () => {
            window.removeEventListener("hifi-file-drag", onHifiFileDrag);
        };
    }, [dispatch, pxPerSec, rowHeight]);

    const multiSelectedClipIds = useAppSelector(
        (state: RootState) => state.session.multiSelectedClipIds,
    );
    const multiSelectedClipIdsRef = useRef(multiSelectedClipIds);
    useEffect(() => {
        multiSelectedClipIdsRef.current = multiSelectedClipIds;
    }, [multiSelectedClipIds]);

    const setMultiSelectedClipIds = React.useCallback(
        (ids: string[] | ((prev: string[]) => string[])) => {
            if (typeof ids === "function") {
                // 支持 callback 形式（与旧 useState dispatch 兼容）
                const next = ids(multiSelectedClipIdsRef.current);
                dispatch(setMultiSelectedClipIdsAction(next));
            } else {
                dispatch(setMultiSelectedClipIdsAction(ids));
            }
        },
        [dispatch],
    );
    // 切换工具时清除多选
    useEffect(() => {
        dispatch(setMultiSelectedClipIdsAction([]));
    }, [s.toolMode, dispatch]);
    const multiSelectedSet = useMemo(
        () => new Set(multiSelectedClipIds),
        [multiSelectedClipIds],
    );
    const multiSelectedSetRef = useRef(multiSelectedSet);
    useEffect(() => {
        multiSelectedSetRef.current = multiSelectedSet;
    }, [multiSelectedSet]);

    const [contextMenu, setContextMenu] = useState<{
        x: number;
        y: number;
        clipId: string;
        overlappingClipIds?: string[];
    } | null>(null);
    const [trackAreaMenu, setTrackAreaMenu] = useState<{
        x: number;
        y: number;
        trackId: string;
    } | null>(null);

    /** 导入模式选择菜单 */
    const [importModeMenu, setImportModeMenu] = useState<{
        x: number;
        y: number;
        audioPaths: string[];
        trackId: string | null;
        startSec: number;
    } | null>(null);

    /** �Ҽ��˵��д����������� clipId */
    const [renamingClipId, setRenamingClipId] = useState<string | null>(null);

    const clearContextMenu = React.useCallback(() => {
        setContextMenu(null);
    }, []);

    const handleSelectionRectSingleSelect = React.useCallback(
        (clipId: string) => {
            void dispatch(selectClipRemote(clipId));
        },
        [dispatch],
    );

    const { selectionRect, onPointerDown: onSelectionRectPointerDown } =
        useTimelineSelectionRect({
            scrollRef,
            sessionRef,
            pxPerBeat: pxPerSec,
            rowHeight,
            clearContextMenu,
            setMultiSelectedClipIds,
            onSingleSelect: handleSelectionRectSingleSelect,
        });

    const dynamicProjectSec = useMemo(
        () => getDynamicProjectSec(s.clips),
        [s.clips],
    );

    const contentWidth = useMemo(
        () => Math.max(1, Math.ceil(dynamicProjectSec * pxPerSec)),
        [dynamicProjectSec, pxPerSec],
    );

    const dropExtraRows =
        (dropPreview && !dropPreview.trackId ? 1 : 0) +
        (clipDropNewTrack ? 1 : 0);
    const contentHeight =
        (s.tracks.length + dropExtraRows) * rowHeight + TRACK_ADD_ROW_HEIGHT;

    const bars = useMemo(() => {
        const beatsPerBar = Math.max(1, Math.round(s.beats || 4));
        const secPerBeatLocal = 60 / Math.max(1, s.bpm);
        // totalBeats 必须用秒/每拍换算，确保覆盖整个 projectSec 范围
        const totalBeats = Math.max(
            1,
            Math.ceil(dynamicProjectSec / secPerBeatLocal),
        );
        const totalBars = Math.max(1, Math.ceil(totalBeats / beatsPerBar));

        let startBarIndex = 0;
        let endBarIndex = totalBars;

        if (Number.isFinite(viewportWidth) && viewportWidth > 0) {
            const beatPx = Math.max(1e-9, secPerBeatLocal * pxPerSec);
            const bufferPx = Math.max(240, viewportWidth * 0.5);
            const leftPx = Math.max(0, scrollLeft - bufferPx);
            const rightPx = scrollLeft + viewportWidth + bufferPx;

            const leftBeat = leftPx / beatPx;
            const rightBeat = rightPx / beatPx;

            startBarIndex = Math.max(
                0,
                Math.floor(leftBeat / beatsPerBar) - 1,
            );
            endBarIndex = Math.min(
                totalBars,
                Math.ceil(rightBeat / beatsPerBar) + 1,
            );
        }

        const result: Array<{ beat: number; label: string }> = [];
        for (let barIndex = startBarIndex; barIndex <= endBarIndex; barIndex += 1) {
            const beat = barIndex * beatsPerBar;
            if (beat > totalBeats) break;
            result.push({ beat, label: `${barIndex + 1}.1` });
        }
        return result;
    }, [
        s.beats,
        dynamicProjectSec,
        s.bpm,
        viewportWidth,
        pxPerSec,
        scrollLeft,
    ]);

    const clipsByTrackId = useMemo(() => {
        const map = new Map<string, typeof s.clips>();
        for (const clip of s.clips) {
            const arr = map.get(clip.trackId);
            if (arr) {
                arr.push(clip);
            } else {
                map.set(clip.trackId, [clip]);
            }
        }

        for (const arr of map.values()) {
            arr.sort((a, b) => {
                const d = (a.startSec ?? 0) - (b.startSec ?? 0);
                if (Math.abs(d) > 1e-9) return d;
                return String(a.id).localeCompare(String(b.id));
            });
        }

        return map;
    }, [s.clips]);

    // ========================================
    // Mipmap 预加载：clip 列表变化时，对新 sourcePath 自动预加载三级波形数据
    // ========================================
    const preloadedPathsRef = useRef(new Set<string>());
    useEffect(() => {
        for (const clip of s.clips) {
            const sp = clip.sourcePath;
            if (sp && !preloadedPathsRef.current.has(sp)) {
                preloadedPathsRef.current.add(sp);
                void waveformMipmapStore.preload(sp);
            }
        }
    }, [s.clips]);

    /** 将客户端 X 坐标转换为秒（seconds-based，不受 BPM 影响） */
    const secFromClientX = React.useCallback(
        (clientX: number, bounds: DOMRect, xScroll: number) => {
            const x = clientX - bounds.left + xScroll;
            // 改用 ref，切断对 pxPerSec state 的依赖
            return Math.max(0, x / pxPerSecRef.current);
        },
        [],
    );

    const beatFromClientX = secFromClientX;

    function trackIdFromClientY(clientY: number) {
        const scroller = scrollRef.current;
        if (!scroller) return null;
        const bounds = scroller.getBoundingClientRect();
        const y = clientY - bounds.top + scroller.scrollTop;
        // ⬇️ 优化核心：改用 ref，切断对 rowHeight state 的依赖
        const idx = Math.floor(y / rowHeightRef.current);
        const tracks = sessionRef.current.tracks;
        if (idx < 0 || idx >= tracks.length) return null;
        return tracks[idx]?.id ?? null;
    }

    function rowTopForTrackId(trackId: string | null) {
        if (!trackId) {
            return s.tracks.length * rowHeight;
        }
        const idx = s.tracks.findIndex((t) => t.id === trackId);
        if (idx < 0) {
            return s.tracks.length * rowHeight;
        }
        return idx * rowHeight;
    }

    function ensureDropPreviewDuration(path: string) {
        if (!path || pendingDropDurationPathRef.current === path) return;
        pendingDropDurationPathRef.current = path;
        void import("../../services/api/fileBrowser")
            .then(({ fileBrowserApi }) => fileBrowserApi.getAudioFileInfo(path))
            .then((info) => {
                setDropPreview((prev) => {
                    if (!prev || prev.path !== path) return prev;
                    return {
                        ...prev,
                        durationSec: Math.max(
                            0,
                            Number(info?.durationSec ?? prev.durationSec) || 0,
                        ),
                    };
                });
            })
            .catch(() => undefined)
            .finally(() => {
                if (pendingDropDurationPathRef.current === path) {
                    pendingDropDurationPathRef.current = null;
                }
            });
    }

    function getDropPreviewWidthPx(durationSec: number) {
        return durationSec > 0
            ? Math.max(1, pxPerSecRef.current * durationSec)
            : 80;
    }

    const setPlayheadFromClientX = React.useCallback(
        (
            clientX: number,
            bounds: DOMRect,
            xScroll: number,
            commit: boolean,
        ) => {
            const beat = beatFromClientX(clientX, bounds, xScroll);

            if (commit) {
                // 只有松手时才触发 Redux 全局重绘
                dispatch(setplayheadSec(beat));
                void dispatch(seekPlayhead(beat));
            } else {
                // 拖拽过程中直通 DOM
                if (playheadRef.current) {
                    playheadRef.current.style.left = `${beat * pxPerSecRef.current}px`;
                }
            }
            return beat;
        },
        [beatFromClientX, dispatch],
    );

    const startDeferredPlayheadSeek = React.useCallback(
        (args: {
            startClientX: number;
            startClientY: number;
            getBounds: () => DOMRect | null;
            getScrollLeft: () => number;
        }) => {
            const { startClientX, startClientY, getBounds, getScrollLeft } = args;
            let moved = false;
            let lastSec = 0;

            const updateAt = (clientX: number, commit: boolean) => {
                const bounds = getBounds();
                if (!bounds) return null;
                const sec = setPlayheadFromClientX(
                    clientX,
                    bounds,
                    getScrollLeft(),
                    commit,
                );
                return sec;
            };

            const onMove = (ev: MouseEvent) => {
                const dx = ev.clientX - startClientX;
                const dy = ev.clientY - startClientY;
                if (!moved && dx * dx + dy * dy >= 9) {
                    moved = true;
                }
                if (!moved) return;
                const sec = updateAt(ev.clientX, false);
                if (sec != null) lastSec = sec;
            };

            const onEnd = (ev: MouseEvent) => {
                window.removeEventListener("mousemove", onMove, true);
                window.removeEventListener("mouseup", onEnd, true);
                window.removeEventListener("mouseleave", onEnd, true);

                if (!moved) {
                    updateAt(ev.clientX, true);
                    return;
                }

                const sec = updateAt(ev.clientX, false);
                const finalSec = sec == null ? lastSec : sec;
                void dispatch(seekPlayhead(finalSec));
            };

            window.addEventListener("mousemove", onMove, true);
            window.addEventListener("mouseup", onEnd, true);
            window.addEventListener("mouseleave", onEnd, true);
        },
        [dispatch, setPlayheadFromClientX],
    );

    const ensureTrackLaneSelected = React.useCallback(
        (clipId: string) => {
            lastClickedClipIdRef.current = clipId;
            const selectedIds = multiSelectedClipIdsRef.current;
            const selectedSet = multiSelectedSetRef.current;
            if (selectedIds.length === 0 || !selectedSet.has(clipId)) {
                setMultiSelectedClipIds([clipId]);
            }
        },
        [setMultiSelectedClipIds],
    );

    const selectTrackLaneClipRemote = React.useCallback(
        (clipId: string) => {
            lastClickedClipIdRef.current = clipId;
            void dispatch(selectClipRemote(clipId));
        },
        [dispatch],
    );

    const openTrackLaneContextMenu = React.useCallback(
        (clipId: string, clientX: number, clientY: number) => {
            setTrackAreaMenu(null);
            setContextMenu({
                x: clientX,
                y: clientY,
                clipId,
            });
        },
        [],
    );

    const seekFromTrackLaneClientX = React.useCallback(
        (clientX: number, commit: boolean) => {
            const scroller = scrollRef.current;
            if (!scroller) return;
            const bounds = scroller.getBoundingClientRect();
            setPlayheadFromClientX(
                clientX,
                bounds,
                scroller.scrollLeft,
                commit,
            );
        },
        [setPlayheadFromClientX],
    );

    const toggleTrackLaneClipMuted = React.useCallback(
        (clipId: string, nextMuted: boolean) => {
            dispatch(
                setClipMuted({
                    clipId,
                    muted: nextMuted,
                }),
            );
            void dispatch(
                setClipStateRemote({
                    clipId,
                    muted: nextMuted,
                }),
            );
        },
        [dispatch],
    );

    const toggleTrackLaneMultiSelect = React.useCallback(
        (clipId: string) => {
            setMultiSelectedClipIds((prev) => {
                if (prev.includes(clipId)) {
                    return prev.filter((id) => id !== clipId);
                }
                return [...prev, clipId];
            });
        },
        [setMultiSelectedClipIds],
    );

    const commitTrackLaneRename = React.useCallback(
        (clipId: string, newName: string) => {
            void dispatch(
                setClipStateRemote({
                    clipId,
                    name: newName,
                }),
            );
        },
        [dispatch],
    );

    const handleTrackLaneRenameDone = React.useCallback(() => {
        setRenamingClipId(null);
    }, []);

    const commitTrackLaneGain = React.useCallback(
        (clipId: string, db: number) => {
            const gain = Math.pow(10, db / 20);
            void dispatch(
                setClipStateRemote({
                    clipId,
                    gain,
                }),
            );
        },
        [dispatch],
    );

    function startPanPointer(e: React.PointerEvent) {
        const scroller = scrollRef.current;
        if (!scroller) return;
        if (e.pointerType !== "mouse") return;
        panRef.current = {
            pointerId: e.pointerId,
            startX: e.clientX,
            startY: e.clientY,
            scrollLeft: scroller.scrollLeft,
            scrollTop: scroller.scrollTop,
        };

        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none";

        try {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        } catch {
            // ignore
        }

        function onMove(ev: PointerEvent) {
            const pan = panRef.current;
            const el = scrollRef.current;
            if (!pan || !el) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            el.scrollLeft = pan.scrollLeft - (ev.clientX - pan.startX);
            el.scrollTop = pan.scrollTop - (ev.clientY - pan.startY);
            syncScrollLeft(el.scrollLeft);
        }

        function end(ev: PointerEvent) {
            const pan = panRef.current;
            if (!pan) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            panRef.current = null;
            document.body.style.cursor = prevCursor;
            document.body.style.userSelect = prevSelect;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    /** 将秒数 snap 到最近的 beat 对齐位置（seconds-based） */
    function snapSec(sec: number) {
        const stepBeats = gridStepBeats(s.grid);
        const stepSec = stepBeats * (60 / Math.max(1, s.bpm));
        return Math.round(sec / stepSec) * stepSec;
    }

    /** 兼容旧名称，内部统一使用 snapSec */
    const snapBeat = snapSec;

    function isEditableTarget(target: EventTarget | null): boolean {
        const el = target as HTMLElement | null;
        if (!el) return false;
        const tag = (el.tagName ?? "").toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") {
            return true;
        }
        if (el.isContentEditable) return true;
        if (el.closest?.('input,textarea,select,[contenteditable="true"]')) {
            return true;
        }
        return false;
    }

    function isPointerOnNativeScrollbar(
        scroller: HTMLDivElement,
        clientX: number,
        clientY: number,
    ): boolean {
        const bounds = scroller.getBoundingClientRect();
        const horizontalScrollbarHeight =
            scroller.offsetHeight - scroller.clientHeight;
        if (
            horizontalScrollbarHeight > 0 &&
            clientY > bounds.bottom - horizontalScrollbarHeight
        ) {
            return true;
        }
        const verticalScrollbarWidth =
            scroller.offsetWidth - scroller.clientWidth;
        if (
            verticalScrollbarWidth > 0 &&
            clientX > bounds.right - verticalScrollbarWidth
        ) {
            return true;
        }
        return false;
    }

    // ���� ��ק hooks ��������������������������������������������������������������������������������������������������������������������
    const { editDragRef: _editDragRef, startEditDrag } = useEditDrag({
        scrollRef,
        sessionRef,
        dispatch,
        snapBeat,
        beatFromClientX,
        noSnapKb,
        gridSnapEnabled: s.gridSnapEnabled,
    });

    const { slipDragRef: _slipDragRef, startSlipDrag } = useSlipDrag({
        scrollRef,
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        multiSelectedSet,
        beatFromClientX,
    });

    const {
        clipDragRef: _clipDragRef,
        startClipDrag: _startClipDragInner,
        ghostDrag,
    } = useClipDrag({
        scrollRef,
        sessionRef,
        rowHeight,
        multiSelectedClipIds,
        multiSelectedSet,
        dispatch,
        snapBeat,
        beatFromClientX,
        trackIdFromClientY,
        setClipDropNewTrack,
        setMultiSelectedClipIds,
        slipEditKb,
        noSnapKb,
        gridSnapEnabled: s.gridSnapEnabled,
        copyDragKb,
        autoCrossfadeEnabled: s.autoCrossfadeEnabled,
        onCtrlClick: (clipId: string) => {
            setMultiSelectedClipIds((prev) => {
                if (prev.includes(clipId)) {
                    return prev.filter((id) => id !== clipId);
                }
                return [...prev, clipId];
            });
            void dispatch(selectClipRemote(clipId));
        },
    });

    const newTrackGhostClips = useMemo(() => {
        if (clipDropNewTrack) {
            const moved = s.clips.filter(
                (clip) => clip.trackId === NEW_TRACK_SENTINEL,
            );
            if (moved.length > 0) return moved;
        }
        if (!ghostDrag || ghostDrag.targetTrackId != null) {
            return [];
        }
        return ghostDrag.clipIds
            .map((clipId) => {
                const initial = ghostDrag.initialById[clipId];
                const clip = s.clips.find((item) => item.id === clipId);
                if (!initial || !clip) return null;
                return {
                    ...clip,
                    startSec: Math.max(
                        0,
                        initial.startSec + ghostDrag.deltaSec,
                    ),
                };
            })
            .filter((clip): clip is typeof s.clips[number] => clip != null);
    }, [clipDropNewTrack, ghostDrag, s.clips]);

    const startClipDrag = React.useCallback(
        (
            e: React.PointerEvent<HTMLDivElement>,
            clipId: string,
            clipstartSec: number,
            altPressedHint?: boolean,
        ) => {
            _startClipDragInner(
                e,
                clipId,
                clipstartSec,
                altPressedHint,
                startSlipDrag,
            );
        },
        [_startClipDragInner, startSlipDrag],
    );

    const viewportStartSec = scrollLeft / Math.max(1e-9, pxPerSec);
    const viewportEndSec =
        (scrollLeft + viewportWidth) / Math.max(1e-9, pxPerSec);

    // ���� ���̿�ݼ� hook ������������������������������������������������������������������������������������������������������������
    // 规格化 clip 音量（快捷键 & 右键菜单共用）
    const normalizeClips = React.useCallback(
        (ids: string[]) => {
            for (const id of ids) {
                const waveform = sessionRef.current.clipWaveforms[id];
                if (!waveform) continue;
                const samples = Array.isArray(waveform)
                    ? waveform
                    : [...waveform.l, ...waveform.r];
                if (samples.length === 0) continue;
                const peak = Math.max(...samples.map(Math.abs));
                if (peak <= 0) continue;
                const newGain = Math.min(
                    Math.max(1.0 / peak, dbToGain(-12)),
                    dbToGain(12),
                );
                dispatch(setClipGain({ clipId: id, gain: newGain }));
                void dispatch(
                    setClipStateRemote({ clipId: id, gain: newGain }),
                );
            }
        },
        [dispatch, sessionRef],
    );

    const replaceClipSources = React.useCallback(
        async (ids: string[]) => {
            const selected = sessionRef.current.clips.filter((c) =>
                ids.includes(c.id),
            );
            if (selected.length === 0) return;

            const picked = await webApi.openAudioDialog();
            if (!picked.ok || picked.canceled || !picked.path) return;

            const selectedSourcePaths = new Set(
                selected
                    .map((c) => c.sourcePath)
                    .filter((v): v is string => Boolean(v && v.trim().length)),
            );

            let replaceSameSource = false;
            if (selectedSourcePaths.size > 0) {
                const hasOtherClipsWithSameSource =
                    sessionRef.current.clips.some(
                        (clip) =>
                            !ids.includes(clip.id) &&
                            Boolean(
                                clip.sourcePath &&
                                selectedSourcePaths.has(clip.sourcePath),
                            ),
                    );
                if (hasOtherClipsWithSameSource) {
                    replaceSameSource = await new Promise<boolean>((resolve) => {
                        sameSourceConfirmResolverRef.current = resolve;
                        setSameSourceConfirmOpen(true);
                    });
                }
            }

            await dispatch(
                replaceClipSourceRemote({
                    clipIds: ids,
                    newSourcePath: picked.path,
                    replaceSameSource,
                }),
            );
        },
        [dispatch, t],
    );

    const splitClipIdsAtPlayhead = React.useCallback(
        (clipIds: string[]) => {
            const splitSec = Math.max(
                0,
                Number(sessionRef.current.playheadSec ?? 0) || 0,
            );
            const eligibleIds = clipIds.filter((id) => {
                const c = sessionRef.current.clips.find(
                    (clip) => clip.id === id,
                );
                if (!c) return false;
                return (
                    splitSec >= c.startSec &&
                    splitSec <= c.startSec + c.lengthSec
                );
            });
            for (const clipId of eligibleIds) {
                void dispatch(splitClipRemote({ clipId, splitSec }));
            }
            return eligibleIds;
        },
        [dispatch],
    );

    const splitSelectedAtPlayhead = React.useCallback(() => {
        // 直接从 ref 读值
        const selectedIds =
            multiSelectedClipIdsRef.current.length > 0
                ? [...multiSelectedClipIdsRef.current]
                : sessionRef.current.selectedClipId
                    ? [sessionRef.current.selectedClipId]
                    : [];
        if (selectedIds.length === 0) return;
        splitClipIdsAtPlayhead(selectedIds);
    }, [splitClipIdsAtPlayhead]); // 去除了 multiSelectedClipIds 依赖

    const selectClipRangeByRect = React.useCallback(
        (targetClipId: string) => {
            const session = sessionRef.current;
            const target = session.clips.find((c) => c.id === targetClipId);
            if (!target) return;

            const anchorId =
                lastClickedClipIdRef.current ??
                session.selectedClipId ??
                targetClipId;
            const anchor =
                session.clips.find((c) => c.id === anchorId) ?? target;

            const trackIndexById = new Map(
                session.tracks.map((track, index) => [track.id, index]),
            );
            const anchorTrackIndex = trackIndexById.get(anchor.trackId);
            const targetTrackIndex = trackIndexById.get(target.trackId);
            if (anchorTrackIndex == null || targetTrackIndex == null) {
                setMultiSelectedClipIds([targetClipId]);
                dispatch(setSelectedClip(targetClipId));
                lastClickedClipIdRef.current = targetClipId;
                return;
            }

            const minTrack = Math.min(anchorTrackIndex, targetTrackIndex);
            const maxTrack = Math.max(anchorTrackIndex, targetTrackIndex);
            const minStartSec = Math.min(anchor.startSec, target.startSec);
            const maxEndSec = Math.max(
                anchor.startSec + anchor.lengthSec,
                target.startSec + target.lengthSec,
            );

            const selected = session.clips
                .filter((clip) => {
                    const trackIndex = trackIndexById.get(clip.trackId);
                    if (
                        trackIndex == null ||
                        trackIndex < minTrack ||
                        trackIndex > maxTrack
                    ) {
                        return false;
                    }
                    const clipStart = clip.startSec;
                    const clipEnd = clip.startSec + clip.lengthSec;
                    return clipStart >= minStartSec && clipEnd <= maxEndSec;
                })
                .map((clip) => clip.id);

            const next = selected.length > 0 ? selected : [targetClipId];
            setMultiSelectedClipIds(next);
            dispatch(setSelectedClip(targetClipId));
            lastClickedClipIdRef.current = targetClipId;
        },
        [dispatch, setMultiSelectedClipIds],
    );

    const pasteClipsAtPlayhead = React.useCallback(() => {
        const tpl = clipClipboardRef.current;
        if (!tpl || tpl.length === 0) return;

        const playhead = sessionRef.current.playheadSec ?? 0;
        const minStart = tpl
            .map((c) => c.startSec)
            .reduce((a, b) => Math.min(a, b), Number.POSITIVE_INFINITY);
        const delta =
            Number.isFinite(minStart) && minStart !== Number.POSITIVE_INFINITY
                ? playhead - minStart
                : 0;
        const templates = tpl.map((c) => ({
            ...c,
            startSec: Math.max(0, c.startSec + delta),
        }));

        dispatch(checkpointHistory());
        void (async () => {
            await webApi.beginUndoGroup();
            try {
                const payload = await dispatch(
                    createClipsRemote({
                        templates,
                        options: { placeOnSelectedTrack: true },
                    }),
                ).unwrap();
                const created: string[] = payload?.createdClipIds ?? [];
                if (!Array.isArray(created) || created.length === 0) return;

                setMultiSelectedClipIds(created);
                void dispatch(selectClipRemote(created[0]));
                const targetStartSec = templates.reduce(
                    (min, t) => Math.min(min, t.startSec),
                    Number.POSITIVE_INFINITY,
                );
                if (Number.isFinite(targetStartSec)) {
                    dispatch(setplayheadSec(targetStartSec));
                    void dispatch(seekPlayhead(targetStartSec));
                }

                if (sessionRef.current.autoCrossfadeEnabled) {
                    const allClips = (payload?.clips ?? []) as Array<{
                        id?: string;
                        track_id?: string;
                        start_sec?: number;
                        length_sec?: number;
                        fade_in_sec?: number;
                        fade_out_sec?: number;
                    }>;
                    const fadeUpdates = computeAutoCrossfadeFromPayload(
                        allClips,
                        created,
                    );
                    if (fadeUpdates.length > 0) {
                        const fadePromises = fadeUpdates.map((u) =>
                            dispatch(
                                setClipStateRemote({
                                    clipId: u.clipId,
                                    fadeInSec: u.fadeInSec,
                                    fadeOutSec: u.fadeOutSec,
                                }),
                            ).unwrap(),
                        );
                        await Promise.allSettled(fadePromises);
                    }
                }
            } finally {
                void webApi.endUndoGroup();
            }
        })().catch(() => undefined);
    }, [dispatch, setMultiSelectedClipIds]);

    useKeyboardShortcuts({
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        clipClipboardRef,
        buildClipClipboardTemplates,
        isEditableTarget,
        onNormalize: normalizeClips,
        onPaste: pasteClipsAtPlayhead,
        onSplitSelected: splitSelectedAtPlayhead,
    });

    useEffect(() => {
        function onEditOp(e: Event) {
            const op = (e as CustomEvent<{ op?: string }>).detail?.op;
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]");
            const deferToPianoRollForSelection =
                inPianoRoll &&
                sessionRef.current.toolMode === "select" &&
                (op === "selectAll" || op === "deselect");
            if (deferToPianoRollForSelection) return;
            if (inPianoRoll && op !== "selectAll" && op !== "deselect") {
                return;
            }

            if (op === "selectAll") {
                const allIds = sessionRef.current.clips.map((clip) => clip.id);
                setMultiSelectedClipIds(allIds);
                dispatch(setSelectedClipPreservingTrack(allIds[0] ?? null));
                return;
            }

            if (op === "deselect") {
                setMultiSelectedClipIds([]);
                dispatch(setSelectedClip(null));
                return;
            }

            if (op === "paste") {
                pasteClipsAtPlayhead();
            }
            if (op === "split") {
                splitSelectedAtPlayhead();
            }
        }
        window.addEventListener("hifi:editOp", onEditOp as EventListener);
        return () =>
            window.removeEventListener(
                "hifi:editOp",
                onEditOp as EventListener,
            );
    }, [pasteClipsAtPlayhead, splitSelectedAtPlayhead]);

    useEffect(() => {
        function onNudge(e: Event) {
            const direction = Number(
                (e as CustomEvent<{ direction?: number }>).detail?.direction ?? 0,
            );
            if (!direction) return;
            const stepSec =
                gridStepBeats(sessionRef.current.grid) *
                (60 / Math.max(1, sessionRef.current.bpm));
            const current = Number(sessionRef.current.playheadSec ?? 0) || 0;
            const next = Math.max(0, current + Math.sign(direction) * stepSec);
            dispatch(setplayheadSec(next));
            void dispatch(seekPlayhead(next));
        }

        window.addEventListener("hifi:nudgePlayhead", onNudge as EventListener);
        return () =>
            window.removeEventListener(
                "hifi:nudgePlayhead",
                onNudge as EventListener,
            );
    }, [dispatch]);

    useEffect(() => {
        function onZoomFocused(e: Event) {
            const active = document.activeElement as HTMLElement | null;
            const inTimeline =
                active?.hasAttribute("data-timeline-scroller") ||
                active?.closest?.("[data-timeline-scroller]") ||
                document.body.getAttribute("data-hs-focus-window") === "timeline";
            if (!inTimeline) return;

            const factor = Number(
                (e as CustomEvent<{ factor?: number }>).detail?.factor ?? 1,
            );
            if (!Number.isFinite(factor) || factor <= 0) return;

            const scroller = scrollRef.current;
            if (!scroller) return;

            const zoom = computeAnchoredHorizontalZoom({
                currentScale: pxPerSecRef.current,
                factor,
                minScale: MIN_PX_PER_SEC,
                maxScale: MAX_PX_PER_SEC,
                scrollLeft: scroller.scrollLeft,
                viewportWidth: scroller.clientWidth,
                anchorSec: Number(sessionRef.current.playheadSec ?? 0) || 0,
                contentSec: sessionRef.current.projectSec,
            });
            if (!zoom) return;

            keyboardZoomPendingRef.current = {
                nextScale: zoom.nextScale,
                nextScrollLeft: zoom.nextScrollLeft,
            };
            setPxPerSec(zoom.nextScale);
        }

        window.addEventListener(
            "hifi:zoomTimelineFocus",
            onZoomFocused as EventListener,
        );
        return () =>
            window.removeEventListener(
                "hifi:zoomTimelineFocus",
                onZoomFocused as EventListener,
            );
    }, []);

    useEffect(() => {
        if (!contextMenu && !trackAreaMenu) return;
        function onAnyPointerDown(e: PointerEvent) {
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-hs-context-menu='1']")) return;
            setContextMenu(null);
            setTrackAreaMenu(null);
        }
        window.addEventListener("pointerdown", onAnyPointerDown, true);
        return () =>
            window.removeEventListener("pointerdown", onAnyPointerDown, true);
    }, [contextMenu, trackAreaMenu]);

    // Auto-scroll: keep playhead visible during playback
    useEffect(() => {
        if (!s.autoScrollEnabled || !s.runtime.isPlaying) return;
        const scroller = scrollRef.current;
        if (!scroller) return;
        const playheadX = s.playheadSec * pxPerSec;
        const viewLeft = scroller.scrollLeft;
        const viewRight = viewLeft + scroller.clientWidth;
        if (playheadX < viewLeft || playheadX > viewRight) {
            const next = Math.max(0, playheadX - scroller.clientWidth / 2);
            scroller.scrollLeft = next;
            syncScrollLeft(next);
        }
    }, [s.autoScrollEnabled, s.runtime.isPlaying, s.playheadSec, pxPerSec]);

    // Focus cursor: scroll to center the playhead in the viewport
    useEffect(() => {
        function handler() {
            const scroller = scrollRef.current;
            if (!scroller) return;
            const playheadX = s.playheadSec * pxPerSec;
            const next = Math.max(0, playheadX - scroller.clientWidth / 2);
            scroller.scrollLeft = next;
            syncScrollLeft(next);
        }
        window.addEventListener("hifi:focusCursor", handler);
        return () => window.removeEventListener("hifi:focusCursor", handler);
    }, [s.playheadSec, pxPerSec]);

    return (
        <Flex className="h-full w-full bg-qt-graph-bg overflow-hidden">
            <TrackList
                t={t}
                tracks={s.tracks}
                selectedTrackId={s.selectedTrackId}
                rowHeight={rowHeight}
                trackVolumeUi={trackVolumeUi}
                listScrollRef={trackListScrollRef}
                onSelectTrack={(trackId) => {
                    dispatch(selectTrackRemote(trackId));
                }}
                onRemoveTrack={(trackId) => {
                    dispatch(removeTrackRemote(trackId));
                }}
                onMoveTrack={(payload) => {
                    dispatch(
                        moveTrackRemote({
                            trackId: payload.trackId,
                            targetIndex: payload.targetIndex,
                            parentTrackId: payload.parentTrackId,
                        }),
                    );
                }}
                onToggleMute={(trackId, nextMuted) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            muted: nextMuted,
                        }),
                    );
                }}
                onToggleSolo={(trackId, nextSolo) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            solo: nextSolo,
                        }),
                    );
                }}
                onToggleCompose={(trackId, nextComposeEnabled) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            composeEnabled: nextComposeEnabled,
                        }),
                    );
                }}
                onVolumeUiChange={(trackId, nextVolume) => {
                    setTrackVolumeUi((prev) => ({
                        ...prev,
                        [trackId]: nextVolume,
                    }));
                }}
                onVolumeCommit={(trackId, nextVolume) => {
                    setTrackVolumeUi((prev) => {
                        const copy = { ...prev };
                        delete copy[trackId];
                        return copy;
                    });
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            volume: nextVolume,
                        }),
                    );
                }}
                onAddTrack={() => {
                    dispatch(addTrackRemote({}));
                }}
                onTrackColorChange={(trackId, color) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            color,
                        }),
                    );
                }}
                onAlgoChange={(trackId, algo) => {
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            pitchAnalysisAlgo: algo,
                        }),
                    );
                }}
                onTrackNameChange={(trackId, name) => {
                    // 乐观更新：立即反映到 UI
                    dispatch(setTrackName({ trackId, name }));
                    dispatch(
                        setTrackStateRemote({
                            trackId,
                            name,
                        }),
                    );
                }}
                onDuplicateTrack={(trackId) => {
                    dispatch(duplicateTrackRemote(trackId));
                }}
                onScrollTopChange={(scrollTop) => {
                    const timelineScroller = scrollRef.current;
                    if (!timelineScroller) return;
                    if (Math.abs(timelineScroller.scrollTop - scrollTop) < 0.5) return;
                    timelineScroller.scrollTop = scrollTop;
                }}
            />

            {/* Timeline View (Right) */}
            <Flex
                direction="column"
                className="flex-1 relative overflow-hidden bg-qt-graph-bg"
            >
                <TimeRuler
                    contentWidth={contentWidth}
                    scrollLeft={scrollLeft}
                    bars={bars}
                    pxPerBeat={pxPerBeat}
                    pxPerSec={pxPerSec}
                    secPerBeat={secPerBeat}
                    viewportWidth={viewportWidth}
                    playheadSec={s.playheadSec}
                    contentRef={rulerContentRef}
                    onMouseDown={(e) => {
                        if (e.button !== 0) return;
                        document.body.setAttribute("data-hs-focus-window", "timeline");
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        const bounds = (
                            e.currentTarget as HTMLDivElement
                        ).getBoundingClientRect();
                        startDeferredPlayheadSeek({
                            startClientX: e.clientX,
                            startClientY: e.clientY,
                            getBounds: () => bounds,
                            getScrollLeft: () => scroller.scrollLeft,
                        });
                    }}
                />

                {/* Tracks Area */}
                <TimelineScrollArea
                    scrollRef={scrollRef}
                    projectSec={dynamicProjectSec}
                    bpm={s.bpm}
                    pxPerSec={pxPerSec}
                    setPxPerSec={setPxPerSec}
                    rowHeight={rowHeight}
                    setRowHeight={setRowHeight}
                    setScrollLeft={setScrollLeftAction}
                    scrollHorizontalKb={scrollHorizontalKb}
                    scrollVerticalKb={scrollVerticalKb}
                    horizontalZoomKb={horizontalZoomKb}
                    playheadSec={s.playheadSec}
                    playheadZoomEnabled={s.playheadZoomEnabled}
                    className="flex-1 bg-qt-graph-bg overflow-auto relative custom-scrollbar"
                    data-timeline-scroller
                    onScroll={(e) => {
                        const el = e.currentTarget as HTMLDivElement;
                        if (trackListScrollRef.current) {
                            if (
                                Math.abs(
                                    trackListScrollRef.current.scrollTop - el.scrollTop,
                                ) >= 0.5
                            ) {
                                trackListScrollRef.current.scrollTop = el.scrollTop;
                            }
                        }
                    }}
                    onMouseDownCapture={(e) => {
                        if (e.button === 1) {
                            e.preventDefault();
                        }
                    }}
                    onAuxClick={(e) => {
                        if (e.button === 1) {
                            e.preventDefault();
                        }
                    }}
                    onContextMenu={(e) => {
                        e.preventDefault();
                        // Clear any existing clip menu state first
                        setContextMenu(null);

                        const target = e.target as HTMLElement | null;
                        // If clicking on an existing custom context menu, ignore
                        if (target?.closest?.("[data-hs-context-menu='1']")) return;

                        const trackId = trackIdFromClientY(e.clientY);
                        if (!trackId) {
                            setTrackAreaMenu(null);
                            return;
                        }

                        // Try to detect clips at this time position (for overlapping clips)
                        const scroller = scrollRef.current;
                        const bounds = scroller?.getBoundingClientRect() ?? null;
                        const timeAtPointer =
                            bounds && scroller
                                ? beatFromClientX(e.clientX, bounds, scroller.scrollLeft)
                                : null;

                        if (timeAtPointer != null) {
                            const clipsHere = sessionRef.current.clips
                                .filter((c) => c.trackId === trackId)
                                .filter((c) => {
                                    const start = Number(c.startSec ?? 0) || 0;
                                    const end = start + (Number(c.lengthSec ?? 0) || 0);
                                    return timeAtPointer >= start && timeAtPointer <= end;
                                })
                                .sort((a, b) => a.startSec - b.startSec);

                            if (clipsHere.length > 0) {
                                // If the click actually landed inside a clip DOM node, let the clip's handler win (it stops propagation).
                                if (target?.closest?.("[data-hs-clip-item='1']")) return;

                                // Open clip context menu for the top-most clip, and pass overlapping ids if multiple
                                const topClip = clipsHere[clipsHere.length - 1];
                                setContextMenu({
                                    x: e.clientX,
                                    y: e.clientY,
                                    clipId: topClip.id,
                                    overlappingClipIds:
                                        clipsHere.length > 1
                                            ? clipsHere.map((c) => c.id)
                                            : undefined,
                                });
                                return;
                            }
                        }

                        // Fallback: open track area menu
                        if (sessionRef.current.selectedTrackId !== trackId) {
                            void dispatch(selectTrackRemote(trackId));
                        }
                        setTrackAreaMenu({
                            x: e.clientX,
                            y: e.clientY,
                            trackId,
                        });
                    }}
                    onPointerDown={onSelectionRectPointerDown}
                    onDragOver={(e) => {
                        const dt = e.dataTransfer;
                        const tauriPath = tauriDraggedPathRef.current;
                        const hasDomFile = Boolean(
                            dt?.files && dt.files.length > 0,
                        );
                        const isTauri = Boolean(
                            (window as unknown as { __TAURI__?: unknown })
                                .__TAURI__,
                        );
                        if (
                            !isTauri &&
                            !hasFileDrag(dt) &&
                            !hasDomFile &&
                            !tauriPath
                        )
                            return;
                        e.preventDefault();
                        const info = extractLocalFilePath(dt);
                        const el = e.currentTarget as HTMLDivElement;
                        const bounds = el.getBoundingClientRect();
                        const beat = beatFromClientX(
                            e.clientX,
                            bounds,
                            el.scrollLeft,
                        );
                        const trackId = trackIdFromClientY(e.clientY);
                        const path = info?.path || tauriPath || "";
                        const fileName =
                            info?.name ||
                            (tauriPath
                                ? String(
                                    tauriPath.split(/[\\/]/).pop() ??
                                    tauriPath,
                                )
                                : hasDomFile
                                    ? String(dt?.files?.[0]?.name ?? "Audio")
                                    : "Audio");
                        if (path && detectExternalPathAction(path) !== "importAudio") {
                            setDropPreview(null);
                            return;
                        }
                        if (path) {
                            ensureDropPreviewDuration(path);
                        }
                        setDropPreview({
                            path,
                            fileName,
                            trackId,
                            startSec: beat,
                            durationSec: 0,
                        });
                    }}
                    onDragLeave={(e) => {
                        const related = e.relatedTarget as Node | null;
                        if (
                            related &&
                            (e.currentTarget as HTMLDivElement).contains(
                                related,
                            )
                        )
                            return;
                        setDropPreview(null);
                    }}
                    onDrop={(e) => {
                        const dt = e.dataTransfer;
                        const tauriPath = tauriDraggedPathRef.current;
                        const lastTauriDropPath = tauriLastDropPathRef.current;
                        const hasDomFile = Boolean(
                            dt?.files && dt.files.length > 0,
                        );
                        const isTauri = Boolean(
                            (window as unknown as { __TAURI__?: unknown })
                                .__TAURI__,
                        );
                        if (
                            !isTauri &&
                            !hasFileDrag(dt) &&
                            !hasDomFile &&
                            !tauriPath
                        )
                            return;
                        e.preventDefault();

                        if (
                            isTauri &&
                            Date.now() - (tauriDropHandledAtRef.current || 0) <
                            500
                        ) {
                            setDropPreview(null);
                            return;
                        }

                        const info = extractLocalFilePath(dt);
                        const el = e.currentTarget as HTMLDivElement;
                        const bounds = el.getBoundingClientRect();
                        const beat = beatFromClientX(
                            e.clientX,
                            bounds,
                            el.scrollLeft,
                        );
                        const trackId = trackIdFromClientY(e.clientY);
                        setDropPreview(null);
                        const resolvedPath =
                            info?.path || lastTauriDropPath || tauriPath;
                        if (resolvedPath) {
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;
                            const actionKind = detectExternalPathAction(resolvedPath);
                            if (actionKind && actionKind !== "importAudio") {
                                emitExternalFileAction(actionKind, resolvedPath);
                                return;
                            }
                            void dispatch(
                                importAudioAtPosition({
                                    audioPath: resolvedPath,
                                    trackId,
                                    startSec: beat,
                                }),
                            );
                            return;
                        }

                        if (isTauri) {
                            window.setTimeout(() => {
                                const p =
                                    tauriLastDropPathRef.current ||
                                    tauriDraggedPathRef.current;
                                if (!p) return;
                                tauriDraggedPathRef.current = null;
                                tauriLastDropPathRef.current = null;
                                const actionKind = detectExternalPathAction(p);
                                if (actionKind && actionKind !== "importAudio") {
                                    emitExternalFileAction(actionKind, p);
                                    return;
                                }
                                void dispatch(
                                    importAudioAtPosition({
                                        audioPath: p,
                                        trackId,
                                        startSec: beat,
                                    }),
                                );
                            }, 0);
                        }

                        const fallbackFile = dt.files?.[0] ?? null;
                        if (fallbackFile) {
                            void dispatch(
                                importAudioFileAtPosition({
                                    file: fallbackFile,
                                    trackId,
                                    startSec: beat,
                                }),
                            );
                        }
                    }}
                    onPointerDownCapture={(e) => {
                        document.body.setAttribute("data-hs-focus-window", "timeline");
                        const scroller = scrollRef.current;
                        if (
                            scroller &&
                            isPointerOnNativeScrollbar(
                                scroller,
                                e.clientX,
                                e.clientY,
                            )
                        ) {
                            return;
                        }
                        if (e.button === 0 || e.button === 2) {
                            const trackId = trackIdFromClientY(e.clientY);
                            if (
                                trackId &&
                                trackId !== sessionRef.current.selectedTrackId
                            ) {
                                void dispatch(selectTrackRemote(trackId));
                            }
                        }
                        if (e.button !== 1) return;
                        if (isEditableTarget(e.target)) return;
                        e.preventDefault();
                        startPanPointer(e);
                    }}
                    onMouseDown={(e) => {
                        if (e.button !== 0) return;
                        setContextMenu(null);
                        setTrackAreaMenu(null);
                        setMultiSelectedClipIds([]);
                        const scroller = scrollRef.current;
                        if (!scroller) return;
                        // Ignore clicks on the native scrollbar region
                        if (
                            isPointerOnNativeScrollbar(
                                scroller,
                                e.clientX,
                                e.clientY,
                            )
                        )
                            return;
                        const trackId = trackIdFromClientY(e.clientY);
                        if (
                            trackId &&
                            trackId !== sessionRef.current.selectedTrackId
                        ) {
                            void dispatch(selectTrackRemote(trackId));
                        }
                        startDeferredPlayheadSeek({
                            startClientX: e.clientX,
                            startClientY: e.clientY,
                            getBounds: () => {
                                const cur = scrollRef.current;
                                return cur ? cur.getBoundingClientRect() : null;
                            },
                            getScrollLeft: () => {
                                const cur = scrollRef.current;
                                return cur ? cur.scrollLeft : scroller.scrollLeft;
                            },
                        });
                    }}
                >
                    {/* Track Lanes */}
                    <div
                        className="relative"
                        style={{ width: contentWidth, height: contentHeight }}
                    >
                        {selectionRect ? (
                            <div
                                className="absolute z-40 pointer-events-none"
                                style={{
                                    left: selectionRect.x1,
                                    top: selectionRect.y1,
                                    width: Math.max(
                                        1,
                                        selectionRect.x2 - selectionRect.x1,
                                    ),
                                    height: Math.max(
                                        1,
                                        selectionRect.y2 - selectionRect.y1,
                                    ),
                                    border: "1px dashed var(--qt-highlight)",
                                    backgroundColor:
                                        "color-mix(in oklab, var(--qt-highlight) 12%, transparent)",
                                }}
                            />
                        ) : null}

                        <BackgroundGrid
                            contentWidth={contentWidth}
                            contentHeight={contentHeight}
                            pxPerBeat={pxPerBeat}
                            grid={s.grid}
                            beatsPerBar={Math.max(1, Math.round(s.beats || 4))}
                        />

                        {clipDropNewTrack ? (
                            <div
                                className="absolute left-0 right-0 pointer-events-none z-20"
                                style={{
                                    top: s.tracks.length * rowHeight,
                                    height: rowHeight,
                                }}
                            >
                                <div
                                    className="absolute inset-0"
                                    style={{
                                        border: "1px dashed var(--qt-highlight)",
                                        backgroundColor:
                                            "color-mix(in oklab, var(--qt-highlight) 10%, transparent)",
                                    }}
                                />
                                {newTrackGhostClips.map((clip) => (
                                    <div
                                        key={`new-track-ghost-${clip.id}`}
                                        className="absolute opacity-60"
                                        style={{
                                            left: Math.max(0, clip.startSec * pxPerSec),
                                            width: Math.max(1, clip.lengthSec * pxPerSec),
                                            top: 0,
                                            height: rowHeight - 8,
                                            paddingTop: 8,
                                        }}
                                    >
                                        <div
                                            className="absolute left-0 right-0 top-0 rounded-t-sm"
                                            style={{
                                                height: 18,
                                                backgroundColor:
                                                    "color-mix(in oklab, var(--qt-highlight) 55%, transparent)",
                                            }}
                                        />
                                        <div
                                            className="absolute left-0 right-0 bottom-0 rounded-sm border border-dashed border-white/70"
                                            style={{
                                                top: 18,
                                                backgroundColor:
                                                    "color-mix(in oklab, var(--qt-highlight) 20%, transparent)",
                                            }}
                                        />
                                    </div>
                                ))}
                            </div>
                        ) : null}

                        {s.tracks.map((track) => {
                            const trackClips =
                                clipsByTrackId.get(track.id) ??
                                ([] as typeof s.clips);

                            return (
                                <TrackLane
                                    key={track.id}
                                    track={track}
                                    allTracks={s.tracks}
                                    trackClips={trackClips}
                                    rowHeight={rowHeight}
                                    pxPerSec={pxPerSec}
                                    bpm={s.bpm}
                                    viewportWidthPx={viewportWidth}
                                    viewportStartSec={viewportStartSec}
                                    viewportEndSec={viewportEndSec}
                                    clipWaveforms={s.clipWaveforms}
                                    altPressed={altPressed}
                                    selectedClipId={s.selectedClipId}
                                    multiSelectedClipIds={multiSelectedClipIds}
                                    multiSelectedSet={multiSelectedSet}
                                    trackColor={track.color || undefined}
                                    ensureSelected={ensureTrackLaneSelected}
                                    selectClipRemote={selectTrackLaneClipRemote}
                                    onShiftRangeSelect={selectClipRangeByRect}
                                    openContextMenu={openTrackLaneContextMenu}
                                    seekFromClientX={seekFromTrackLaneClientX}
                                    ghostDrag={ghostDrag}
                                    allClips={s.clips}
                                    startClipDrag={startClipDrag}
                                    startEditDrag={startEditDrag}
                                    toggleClipMuted={toggleTrackLaneClipMuted}
                                    clearContextMenu={clearContextMenu}
                                    toggleMultiSelect={
                                        toggleTrackLaneMultiSelect
                                    }
                                    renamingClipId={renamingClipId}
                                    onRenameCommit={commitTrackLaneRename}
                                    onRenameDone={handleTrackLaneRenameDone}
                                    onGainCommit={commitTrackLaneGain}
                                />
                            );
                        })}

                        <div
                            className="absolute left-0 right-0 pointer-events-none z-10"
                            style={{
                                top: contentHeight - TRACK_ADD_ROW_HEIGHT,
                                height: TRACK_ADD_ROW_HEIGHT,
                                backgroundColor: "var(--qt-graph-bg)",
                            }}
                        />

                        {/* Drop preview (ghost item) */}
                        {dropPreview ? (
                            <div
                                ref={dropPreviewRef}
                                className="absolute z-30 pointer-events-none"
                                style={{
                                    left: Math.max(
                                        0,
                                        dropPreview.startSec * pxPerSec,
                                    ),
                                    top:
                                        rowTopForTrackId(dropPreview.trackId) +
                                        8,
                                    width:
                                        dropPreview.durationSec > 0
                                            ? Math.max(
                                                1,
                                                pxPerSec *
                                                    dropPreview.durationSec,
                                            )
                                            : 80,
                                    height: rowHeight - 16,
                                }}
                            >
                                <div className="h-full w-full rounded-sm border border-dashed border-qt-highlight bg-[color-mix(in_oklab,var(--qt-highlight)_20%,transparent)]">
                                    <div className="px-2 pt-1 text-[10px] text-qt-text truncate">
                                        {dropPreview.fileName}
                                    </div>
                                </div>
                            </div>
                        ) : null}

                        {/* Playhead Cursor */}
                        <div
                            ref={playheadRef}
                            className="absolute top-0 bottom-0 w-px bg-qt-playhead z-20 cursor-ew-resize"
                            style={{ left: s.playheadSec * pxPerSec }}
                            onPointerDown={(e) => {
                                if (e.button !== 0) return;
                                e.stopPropagation();
                                const scroller = scrollRef.current;
                                if (!scroller) return;
                                const startX = e.clientX;
                                const startY = e.clientY;
                                let moved = false;
                                const bounds = scroller.getBoundingClientRect();
                                const initialSec = s.playheadSec;
                                playheadDragRef.current = {
                                    pointerId: e.pointerId,
                                    lastBeat: initialSec,
                                };
                                (
                                    e.currentTarget as HTMLDivElement
                                ).setPointerCapture(e.pointerId);

                                function onPointerMove(ev: PointerEvent) {
                                    const drag = playheadDragRef.current;
                                    const currentScroller = scrollRef.current;
                                    if (
                                        !drag ||
                                        drag.pointerId !== e.pointerId ||
                                        !currentScroller
                                    ) {
                                        return;
                                    }
                                    const dx = ev.clientX - startX;
                                    const dy = ev.clientY - startY;
                                    if (!moved && dx * dx + dy * dy >= 9) {
                                        moved = true;
                                    }
                                    if (!moved) return;
                                    const currentBounds =
                                        currentScroller.getBoundingClientRect();
                                    drag.lastBeat = setPlayheadFromClientX(
                                        ev.clientX,
                                        currentBounds,
                                        currentScroller.scrollLeft,
                                        false,
                                    );
                                }

                                function endDrag() {
                                    const drag = playheadDragRef.current;
                                    if (!drag || drag.pointerId !== e.pointerId)
                                        return;
                                    playheadDragRef.current = null;
                                    if (!moved) {
                                        drag.lastBeat = setPlayheadFromClientX(
                                            startX,
                                            bounds,
                                            scroller!.scrollLeft,
                                            false,
                                        );
                                    }
                                    void dispatch(seekPlayhead(drag.lastBeat));
                                    window.removeEventListener(
                                        "pointermove",
                                        onPointerMove,
                                    );
                                    window.removeEventListener(
                                        "pointerup",
                                        endDrag,
                                    );
                                    window.removeEventListener(
                                        "pointercancel",
                                        endDrag,
                                    );
                                }

                                window.addEventListener(
                                    "pointermove",
                                    onPointerMove,
                                );
                                window.addEventListener("pointerup", endDrag);
                                window.addEventListener(
                                    "pointercancel",
                                    endDrag,
                                );
                            }}
                        />
                    </div>
                </TimelineScrollArea>

                {/* 导入模式选择菜单 */}
                {importModeMenu && (
                    <div
                        className="fixed inset-0 z-[9999]"
                        onClick={() => setImportModeMenu(null)}
                        onContextMenu={(e) => {
                            e.preventDefault();
                            setImportModeMenu(null);
                        }}
                    >
                        <div
                            className="absolute bg-qt-panel border border-qt-border rounded shadow-lg py-1 min-w-[180px]"
                            style={{
                                left: importModeMenu.x,
                                top: importModeMenu.y,
                            }}
                            onClick={(e) => e.stopPropagation()}
                        >
                            <button
                                className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                onClick={() => {
                                    const m = importModeMenu;
                                    setImportModeMenu(null);
                                    if (m.audioPaths.length === 1) {
                                        void dispatch(
                                            importAudioAtPosition({
                                                audioPath: m.audioPaths[0],
                                                trackId: m.trackId,
                                                startSec: m.startSec,
                                            }),
                                        );
                                    } else {
                                        void dispatch(
                                            importMultipleAudioAtPosition({
                                                audioPaths: m.audioPaths,
                                                mode: "across-time",
                                                trackId: m.trackId,
                                                startSec: m.startSec,
                                            }),
                                        );
                                    }
                                }}
                            >
                                {t("import_across_time" as any) ||
                                    "Import across time (same track)"}
                            </button>
                            <button
                                className="w-full text-left px-3 py-1.5 text-sm text-qt-text hover:bg-qt-hover"
                                onClick={() => {
                                    const m = importModeMenu;
                                    setImportModeMenu(null);
                                    if (m.audioPaths.length === 1) {
                                        void dispatch(
                                            importAudioAtPosition({
                                                audioPath: m.audioPaths[0],
                                                trackId: null,
                                                startSec: m.startSec,
                                            }),
                                        );
                                    } else {
                                        void dispatch(
                                            importMultipleAudioAtPosition({
                                                audioPaths: m.audioPaths,
                                                mode: "across-tracks",
                                                trackId: m.trackId,
                                                startSec: m.startSec,
                                            }),
                                        );
                                    }
                                }}
                            >
                                {t("import_across_tracks" as any) ||
                                    "Import across tracks (one per track)"}
                            </button>
                        </div>
                    </div>
                )}

                {contextMenu
                    ? (() => {
                        const ctxClip = sessionRef.current.clips.find(
                            (c) => c.id === contextMenu.clipId,
                        );
                        if (!ctxClip) return null;

                        const selectedIds =
                            multiSelectedClipIds.length >= 2
                                ? multiSelectedClipIds
                                : [contextMenu.clipId];
                        const selectedClips = sessionRef.current.clips.filter(
                            (c) => selectedIds.includes(c.id),
                        );

                        // Convert context menu X to timeline seconds for fade region detection
                        const _ctxScroller = scrollRef.current;
                        const _ctxBounds =
                            _ctxScroller?.getBoundingClientRect();
                        const contextTimeSec =
                            _ctxBounds && _ctxScroller
                                ? beatFromClientX(
                                    contextMenu.x,
                                    _ctxBounds,
                                    _ctxScroller.scrollLeft,
                                )
                                : ctxClip.startSec;

                        const overlappingFadeClips = collectFadeContextClips({
                            allClips: sessionRef.current.clips,
                            contextClip: ctxClip,
                            contextTimeSec,
                            explicitOverlappingClipIds:
                                contextMenu.overlappingClipIds,
                        });

                        const playheadSec = sessionRef.current.playheadSec;
                        const playheadInClip =
                            playheadSec >= ctxClip.startSec &&
                            playheadSec <=
                            ctxClip.startSec + ctxClip.lengthSec;

                        return (
                            <ClipContextMenu
                                x={contextMenu.x}
                                y={contextMenu.y}
                                clip={ctxClip}
                                selectedClips={selectedClips}
                                overlappingClips={overlappingFadeClips}
                                playheadInClip={playheadInClip}
                                canSplitSelected={selectedClips.some((c) => {
                                    const splitSec = Math.max(
                                        0,
                                        Number(
                                            sessionRef.current.playheadSec ??
                                            0,
                                        ) || 0,
                                    );
                                    return (
                                        splitSec >= c.startSec &&
                                        splitSec <= c.startSec + c.lengthSec
                                    );
                                })}
                                onClose={() => setContextMenu(null)}
                                onDelete={(ids) => {
                                    setContextMenu(null);
                                    setMultiSelectedClipIds([]);
                                    for (const id of ids) {
                                        void dispatch(removeClipRemote(id));
                                    }
                                }}
                                onMute={(ids, muted) => {
                                    for (const id of ids) {
                                        dispatch(
                                            setClipMuted({
                                                clipId: id,
                                                muted,
                                            }),
                                        );
                                        void dispatch(
                                            setClipStateRemote({
                                                clipId: id,
                                                muted,
                                            }),
                                        );
                                    }
                                }}
                                onRename={(clipId) => {
                                    setContextMenu(null);
                                    setRenamingClipId(clipId);
                                }}
                                onCopy={(ids) => {
                                    void (async () => {
                                        const templates =
                                            await buildClipClipboardTemplates(
                                                ids,
                                            );
                                        if (templates.length > 0) {
                                            clipClipboardRef.current =
                                                templates;
                                        }
                                    })();
                                }}
                                onCut={(ids) => {
                                    void (async () => {
                                        const templates =
                                            await buildClipClipboardTemplates(
                                                ids,
                                            );
                                        if (templates.length === 0) return;
                                        clipClipboardRef.current = templates;
                                        setContextMenu(null);
                                        setMultiSelectedClipIds([]);
                                        for (const id of ids) {
                                            void dispatch(
                                                removeClipRemote(id),
                                            );
                                        }
                                    })();
                                }}
                                onReplace={(ids) => {
                                    void replaceClipSources(ids);
                                }}
                                onSplit={(clipIds) => {
                                    setContextMenu(null);
                                    splitClipIdsAtPlayhead(clipIds);
                                }}
                                onGlue={(ids) => {
                                    setContextMenu(null);
                                    if (ids.length >= 2) {
                                        void dispatch(glueClipsRemote(ids));
                                        setMultiSelectedClipIds([]);
                                    }
                                }}
                                onFadeCurveChange={(
                                    clipId,
                                    target,
                                    curve,
                                ) => {
                                    dispatch(
                                        setClipFades({
                                            clipId,
                                            ...(target === "in"
                                                ? { fadeInCurve: curve }
                                                : { fadeOutCurve: curve }),
                                        }),
                                    );
                                    void dispatch(
                                        setClipStateRemote({
                                            clipId,
                                            ...(target === "in"
                                                ? { fadeInCurve: curve }
                                                : { fadeOutCurve: curve }),
                                        }),
                                    );
                                }}
                                onNormalize={normalizeClips}
                            />
                        );
                    })()
                    : null}

                {trackAreaMenu ? (
                    <TrackAreaContextMenu
                        x={trackAreaMenu.x}
                        y={trackAreaMenu.y}
                        canPaste={
                            Boolean(clipClipboardRef.current) &&
                            (clipClipboardRef.current?.length ?? 0) > 0
                        }
                        canSplit={(multiSelectedClipIds.length > 0
                            ? multiSelectedClipIds
                            : sessionRef.current.selectedClipId
                                ? [sessionRef.current.selectedClipId]
                                : []
                        ).some((id) => {
                            const clip = sessionRef.current.clips.find(
                                (c) => c.id === id,
                            );
                            if (!clip) return false;
                            const splitSec = Math.max(
                                0,
                                Number(sessionRef.current.playheadSec ?? 0) ||
                                0,
                            );
                            return (
                                splitSec >= clip.startSec &&
                                splitSec <= clip.startSec + clip.lengthSec
                            );
                        })}
                        onPaste={pasteClipsAtPlayhead}
                        onSplit={splitSelectedAtPlayhead}
                        onClose={() => setTrackAreaMenu(null)}
                    />
                ) : null}

                <Dialog.Root
                    open={sameSourceConfirmOpen}
                    onOpenChange={(open) => {
                        setSameSourceConfirmOpen(open);
                        if (!open && sameSourceConfirmResolverRef.current) {
                            sameSourceConfirmResolverRef.current(false);
                            sameSourceConfirmResolverRef.current = null;
                        }
                    }}
                >
                    <Dialog.Content maxWidth="480px">
                        <Dialog.Title>{t("ctx_replace")}</Dialog.Title>
                        <Dialog.Description>
                            <Text size="2">
                                {t("clip_replace_same_source_confirm" as any)}
                            </Text>
                        </Dialog.Description>
                        <Flex justify="end" gap="2" mt="4">
                            <Button
                                variant="soft"
                                color="gray"
                                onClick={() => {
                                    setSameSourceConfirmOpen(false);
                                    if (sameSourceConfirmResolverRef.current) {
                                        sameSourceConfirmResolverRef.current(false);
                                        sameSourceConfirmResolverRef.current = null;
                                    }
                                }}
                            >
                                {t("cancel")}
                            </Button>
                            <Button
                                onClick={() => {
                                    setSameSourceConfirmOpen(false);
                                    if (sameSourceConfirmResolverRef.current) {
                                        sameSourceConfirmResolverRef.current(true);
                                        sameSourceConfirmResolverRef.current = null;
                                    }
                                }}
                            >
                                {t("ok")}
                            </Button>
                        </Flex>
                    </Dialog.Content>
                </Dialog.Root>
            </Flex>
        </Flex>
    );
};
