/**
 * useTimelineDragDrop — Tauri 原生拖放 + 文件浏览器面板自定义拖拽
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - Tauri onDragDropEvent（enter / over / leave / drop）
 * - hifi-file-drag 自定义事件（文件浏览器面板）
 * - tauriDraggedPathRef / tauriLastDropPathRef / tauriDropHandledAtRef 管理
 */
import { useEffect, useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { RootState } from "../../../../app/store";
import {
    importAudioAtPosition,
    importMultipleAudioAtPosition,
} from "../../../../features/session/sessionSlice";
import { emitExternalFileAction } from "../../../../features/session/projectOpenEvents";
import {
    detectExternalPathAction,
    findFirstExternalPathAction,
} from "../";

export interface UseTimelineDragDropArgs {
    dispatch: AppDispatch;
    scrollRef: React.MutableRefObject<HTMLDivElement | null>;
    sessionRef: React.MutableRefObject<RootState["session"]>;
    pxPerSecRef: React.MutableRefObject<number>;
    rowHeightRef: React.MutableRefObject<number>;
    dropPreviewRef: React.MutableRefObject<HTMLDivElement | null>;
    pendingDropDurationPathRef: React.MutableRefObject<string | null>;
    beatFromClientX: (
        clientX: number,
        bounds: DOMRect,
        xScroll: number,
    ) => number;
    trackIdFromClientY: (clientY: number) => string | null;
    rowTopForTrackId: (trackId: string | null) => number;
    setDropPreview: React.Dispatch<
        React.SetStateAction<{
            path: string;
            fileName: string;
            trackId: string | null;
            startSec: number;
            durationSec: number;
        } | null>
    >;
    ensureDropPreviewDuration: (path: string) => void;
    getDropPreviewWidthPx: (durationSec: number) => number;
    setImportModeMenu: React.Dispatch<
        React.SetStateAction<{
            x: number;
            y: number;
            audioPaths: string[];
            trackId: string | null;
            startSec: number;
        } | null>
    >;
    pxPerSec: number;
    rowHeight: number;
}

export interface UseTimelineDragDropResult {
    tauriDraggedPathRef: React.MutableRefObject<string | null>;
    tauriLastDropPathRef: React.MutableRefObject<string | null>;
    tauriDropHandledAtRef: React.MutableRefObject<number>;
}

export function useTimelineDragDrop(
    args: UseTimelineDragDropArgs,
): UseTimelineDragDropResult {
    const {
        dispatch,
        scrollRef,
        sessionRef,
        pxPerSecRef,
        dropPreviewRef,
        beatFromClientX,
        trackIdFromClientY,
        rowTopForTrackId,
        setDropPreview,
        ensureDropPreviewDuration,
        getDropPreviewWidthPx,
        setImportModeMenu,
        pxPerSec,
        rowHeight,
    } = args;

    const tauriDraggedPathRef = useRef<string | null>(null);
    const tauriLastDropPathRef = useRef<string | null>(null);
    const tauriDropHandledAtRef = useRef<number>(0);

    // ── Tauri 原生拖放 ───────────────────────────────────────
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        const debugDnd =
            localStorage.getItem("hifishifter.debugDnd") === "1";

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

                        const scroller = scrollRef.current;
                        const bounds =
                            scroller?.getBoundingClientRect() ?? null;
                        const pos = (payload?.position ??
                            payload?.pos ??
                            payload?.cursorPosition) as
                            | { x?: number; y?: number }
                            | undefined;
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

                        const primaryPath =
                            paths.length > 0 ? paths[0] : null;
                        function fileNameFromPath(p: string) {
                            return String(p.split(/[\\/]/).pop() ?? p);
                        }

                        if (type === "enter" || type === "over") {
                            if (primaryPath) {
                                tauriDraggedPathRef.current = primaryPath;
                                if (
                                    detectExternalPathAction(primaryPath) ===
                                    "importAudio"
                                ) {
                                    ensureDropPreviewDuration(primaryPath);
                                }
                            }
                            setDropPreview((prev) => {
                                const path =
                                    primaryPath ??
                                    tauriDraggedPathRef.current ??
                                    prev?.path ??
                                    null;
                                if (!path) return prev;
                                if (
                                    detectExternalPathAction(path) !==
                                    "importAudio"
                                ) {
                                    return null;
                                }
                                const nextFileName = fileNameFromPath(path);

                                if (prev && dropPreviewRef.current) {
                                    dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                                    dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                                    dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                                }

                                if (
                                    !prev ||
                                    prev.trackId !== trackId ||
                                    prev.path !== path
                                ) {
                                    return {
                                        path,
                                        fileName:
                                            prev?.fileName ?? nextFileName,
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
                                detectExternalPathAction(primaryPath) ===
                                    "importAudio"
                            ) {
                                ensureDropPreviewDuration(primaryPath);
                            }
                            setDropPreview(null);

                            const externalAction =
                                findFirstExternalPathAction(paths);
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
                                const actionKind =
                                    detectExternalPathAction(resolvedPath);
                                if (
                                    actionKind &&
                                    actionKind !== "importAudio"
                                ) {
                                    emitExternalFileAction(
                                        actionKind,
                                        resolvedPath,
                                    );
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

    // ── 文件浏览器面板的自定义拖拽事件 ───────────────────────
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
                        if (prev && dropPreviewRef.current) {
                            dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                            dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                            dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                        }
                        if (
                            !prev ||
                            prev.trackId !== trackId ||
                            prev.path !== path
                        ) {
                            ensureDropPreviewDuration(path);
                            return {
                                path,
                                fileName,
                                trackId,
                                startSec: beat,
                                durationSec:
                                    prev?.path === path
                                        ? prev.durationSec
                                        : 2,
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

                        setImportModeMenu({
                            x: detail.clientX,
                            y: detail.clientY,
                            audioPaths: isMulti
                                ? filePaths
                                : [detail.filePath],
                            trackId,
                            startSec: beat,
                        });
                    } else if (isMulti) {
                        const externalAction =
                            findFirstExternalPathAction(filePaths);
                        if (
                            externalAction &&
                            externalAction.kind !== "importAudio"
                        ) {
                            emitExternalFileAction(
                                externalAction.kind,
                                externalAction.path,
                            );
                            return;
                        }
                        void dispatch(
                            importMultipleAudioAtPosition({
                                audioPaths: filePaths,
                                mode: "across-time",
                                trackId,
                                startSec: beat,
                            }),
                        );
                    } else {
                        const actionKind = detectExternalPathAction(
                            detail.filePath,
                        );
                        if (actionKind && actionKind !== "importAudio") {
                            emitExternalFileAction(
                                actionKind,
                                detail.filePath,
                            );
                            return;
                        }
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

    return {
        tauriDraggedPathRef,
        tauriLastDropPathRef,
        tauriDropHandledAtRef,
    };
}
