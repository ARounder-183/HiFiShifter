/**
 * useTimelineDragDrop — Tauri 原生拖放 + 文件浏览器面板自定义拖拽
 *
 * 从 TimelinePanel.tsx 拆分而来，负责：
 * - Tauri onDragDropEvent（enter / over / leave / drop）
 * - hifi-file-drag 自定义事件（文件浏览器面板）
 * - tauriDraggedPathRef / tauriLastDropPathRef / tauriDropHandledAtRef 管理
 *
 * 【视口来源必须模式无关】
 * 落点换算需要「容器矩形 + 横向滚动量」。旧实现直接读原生 scroller，但渲染内核
 * 模式下旧容器不挂载（`scrollRef.current === null`），会让**所有**拖入路径静默
 * 失效（落点回退到播放头、预览消失）。因此本 hook 只接受
 * `TimelineViewportAccess`，由它内部区分「原生 scroller / 内核自绘滚动」。
 */
import { useEffect, useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import type { RootState } from "../../../../app/store";
import {
    importAudioAtPosition,
    importMultipleAudioAtPosition,
} from "../../../../features/session/sessionSlice";
import { emitExternalFileAction } from "../../../../features/session/projectOpenEvents";
import { emitFolderImportRequest } from "../../../../features/fileBrowser/folderImportEvents";
import { rootIndexAtDrop } from "../../../../features/session/trackUtils";
import { fileBrowserApi } from "../../../../services/api";
import { detectExternalPathAction, isAcceptedDropPath, partitionDroppedPaths } from "../";
import { SNAP_HIGHLIGHT_GROUP, clearSnapHighlights } from "../../../../utils/snapHighlight";
import type { SnapTimelineFn } from "./useTimelineState";
import type { TimelineViewportAccess } from "./timelineViewportAccess";

export interface UseTimelineDragDropArgs {
    dispatch: AppDispatch;
    /**
     * 模式无关的视口访问器（原生 scroller 或渲染内核）。
     *
     * 必须**引用稳定**（面板用 ref / useMemo 构造）：本 hook 的事件监听只在挂载
     * 时注册一次，监听回调内现读 ref，不依赖该对象的身份变化。
     */
    viewport: TimelineViewportAccess;
    sessionRef: React.MutableRefObject<RootState["session"]>;
    pxPerSecRef: React.MutableRefObject<number>;
    rowHeightRef: React.MutableRefObject<number>;
    dropPreviewRef: React.MutableRefObject<HTMLDivElement | null>;
    pendingDropDurationPathRef: React.MutableRefObject<string | null>;
    beatFromClientX: (clientX: number, bounds: DOMRect, xScroll: number) => number;
    /** 吸附引擎入口（媒体项目对象）。 */
    snapTimeline?: SnapTimelineFn;
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
    /**
     * 工程文件（hshp/hsp）拖放时的「打开工程 / 导入工程」操作菜单。
     */
    setProjectActionMenu?: React.Dispatch<
        React.SetStateAction<{ x: number; y: number; path: string } | null>
    >;
    pxPerSec: number;
    rowHeight: number;
    /** MIDI 文件拖放回调（用于创建 MIDI clip） */
    onMidiDrop?: (payload: { midiPath: string; trackId: string | null; startSec: number }) => void;
}

export interface UseTimelineDragDropResult {
    tauriDraggedPathRef: React.MutableRefObject<string | null>;
    tauriLastDropPathRef: React.MutableRefObject<string | null>;
    tauriDropHandledAtRef: React.MutableRefObject<number>;
}

export function useTimelineDragDrop(args: UseTimelineDragDropArgs): UseTimelineDragDropResult {
    const {
        dispatch,
        viewport,
        sessionRef,
        pxPerSecRef,
        dropPreviewRef,
        beatFromClientX,
        snapTimeline,
        trackIdFromClientY,
        rowTopForTrackId,
        setDropPreview,
        ensureDropPreviewDuration,
        getDropPreviewWidthPx,
        setImportModeMenu,
        setProjectActionMenu,
        pxPerSec,
        rowHeight,
        onMidiDrop,
    } = args;

    const tauriDraggedPathRef = useRef<string | null>(null);
    const tauriLastDropPathRef = useRef<string | null>(null);
    const tauriDropHandledAtRef = useRef<number>(0);

    // ── Tauri 原生拖放 ───────────────────────────────────────
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        const debugDnd = localStorage.getItem("hifishifter.debugDnd") === "1";

        async function setup() {
            try {
                const mod = await (
                    await import("../../../../services/hostWindow")
                ).loadStandaloneWindowApi();
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

                type TauriDragDropEvent = { payload?: TauriDragDropPayload } | TauriDragDropPayload;

                unlisten = await win.onDragDropEvent(async (event: TauriDragDropEvent) => {
                    if (disposed) return;
                    const payload = ("payload" in event ? event.payload : event) as
                        | TauriDragDropPayload
                        | undefined;
                    const type = String(payload?.type ?? payload?.event ?? "");
                    const paths: string[] = Array.isArray(payload?.paths) ? payload.paths : [];

                    if (debugDnd) {
                        console.log("[dnd] tauri event", {
                            type,
                            pathsCount: paths.length,
                            hasPosition: Boolean(
                                payload?.position ?? payload?.pos ?? payload?.cursorPosition,
                            ),
                        });
                    }

                    const bounds = viewport.getRect();
                    const pos = (payload?.position ?? payload?.pos ?? payload?.cursorPosition) as
                        | { x?: number; y?: number }
                        | undefined;
                    const dpr = window.devicePixelRatio || 1;
                    const clientX = typeof pos?.x === "number" ? pos.x / dpr : undefined;
                    const clientY = typeof pos?.y === "number" ? pos.y / dpr : undefined;
                    const fallbackBeat = sessionRef.current.playheadSec ?? 0;
                    const trackId = clientY !== undefined ? trackIdFromClientY(clientY) : null;
                    const rawBeat =
                        clientX !== undefined && bounds
                            ? beatFromClientX(clientX, bounds, viewport.getScrollLeft())
                            : fallbackBeat;
                    const beat =
                        snapTimeline && sessionRef.current.timelineSnap.enabled
                            ? snapTimeline(rawBeat, "clip", {
                                  anchorTrackId: trackId,
                                  originSec: rawBeat,
                                  // drop 预览左缘即被吸附边（无 clipId，行级高亮）。
                                  highlight: { sources: [{ trackId }] },
                              })
                            : rawBeat;

                    const primaryPath = paths.length > 0 ? paths[0] : null;

                    // 改用 $O(1)$ 零内存分配的切片
                    function fileNameFromPath(p: string) {
                        const slashIdx = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
                        return slashIdx >= 0 ? p.substring(slashIdx + 1) : p;
                    }

                    if (type === "enter" || type === "over") {
                        if (primaryPath) {
                            tauriDraggedPathRef.current = primaryPath;
                            // MIDI 文件不需要预加载时长，使用默认值
                            const primaryAction = detectExternalPathAction(primaryPath);
                            if (primaryAction === "importAudio") {
                                ensureDropPreviewDuration(primaryPath);
                            }
                        }
                        setDropPreview((prev) => {
                            const path =
                                primaryPath ?? tauriDraggedPathRef.current ?? prev?.path ?? null;
                            if (!path) return prev;
                            // 准入判据：只接受媒体 / 工程（含 -bak）/ 外部工程 / MIDI。
                            // 未知扩展名不再显示预览（此前它会一路穿透到音频导入分支）。
                            if (!isAcceptedDropPath(path)) return null;
                            const action = detectExternalPathAction(path);
                            if (action !== "importAudio" && action !== "importMidi") {
                                return null;
                            }

                            if (prev && dropPreviewRef.current) {
                                dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                                dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                                dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                            }

                            if (!prev || prev.trackId !== trackId || prev.path !== path) {
                                // 仅仅在真正需要更新对象时，才执行提取文件名的操作
                                return {
                                    path,
                                    fileName: prev?.fileName ?? fileNameFromPath(path),
                                    trackId,
                                    startSec: beat,
                                    durationSec:
                                        action === "importMidi" ? 2 : (prev?.durationSec ?? 0),
                                };
                            }
                            return prev;
                        });
                        return;
                    }

                    if (type === "leave") {
                        tauriDraggedPathRef.current = null;
                        setDropPreview(null);
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }

                    if (type === "drop") {
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
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

                        // ── 统一准入 + 分类（缺陷修复点）──────────────────────
                        // 此前：只对"已知的非音频种类"特殊处理，**未知扩展名穿透**到
                        // 音频导入默认分支；多文件更是把全部路径无差别转发，MIDI /
                        // 工程 / 无关文件都被当成 Clip 导入。
                        // 现在：先按 `partitionDroppedPaths` 分类，只放行白名单内的
                        // 类型（媒体 / 工程含 -bak / 外部工程 / MIDI / 目录），其余整体丢弃。
                        //
                        // 目录类型只能问文件系统：路径字符串看不出它是不是目录。一次
                        // 批量查询，拖入 20 项也只发一次 IPC。查询失败按"都不是目录"
                        // 处理 —— 与改动前的行为一致，不会凭空建轨道。
                        let directories: ReadonlySet<string> = new Set<string>();
                        try {
                            const stats = await fileBrowserApi.statPaths(paths);
                            directories = new Set(
                                stats.filter((stat) => stat.isDir).map((stat) => stat.path),
                            );
                        } catch {
                            /* 见上：降级为"没有目录" */
                        }
                        const partition = partitionDroppedPaths(paths, { directories });
                        if (partition.rejectedPaths.length > 0 && debugDnd) {
                            console.info(
                                "[dnd] 已拒绝拖放（类型不在准入白名单）:",
                                partition.rejectedPaths,
                            );
                        }

                        // 工程文件优先：一次拖放只打开一个（打开会替换当前会话）。
                        if (partition.projectPath !== null) {
                            tauriDropHandledAtRef.current = Date.now();
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;
                            const kind = detectExternalPathAction(partition.projectPath);
                            if (
                                kind !== null &&
                                kind !== "importAudio" &&
                                kind !== "importMidi" &&
                                kind !== "importFolder"
                            ) {
                                emitExternalFileAction(kind, partition.projectPath);
                            }
                            return;
                        }

                        // 目录：展开成媒体文件后按所选模式导入。选项对话框与默认值的
                        // 决定权在 `FolderImportHost`（只有一个对话框状态）。
                        // 目录里的散文件（同时拖入的媒体文件）一并交给它，避免
                        // "目录进对话框、文件走另一条路"这种分叉。
                        if (partition.folderPaths.length > 0) {
                            tauriDropHandledAtRef.current = Date.now();
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;
                            setDropPreview(null);
                            emitFolderImportRequest({
                                dirs: [...partition.folderPaths],
                                looseFiles: [...partition.mediaPaths],
                                trackId,
                                startSec: beat,
                                insertIndex: rootIndexAtDrop(sessionRef.current.tracks, trackId),
                            });
                            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                            return;
                        }

                        const midiPath = partition.midiPaths[0] ?? null;
                        if (midiPath !== null) {
                            tauriDropHandledAtRef.current = Date.now();
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;
                            setDropPreview(null);
                            // 仅当落点位于时间轴区域内才在此导入 MIDI；落在其它区域
                            //（如参数编辑器）时留给对应区域的拖放接收者处理，避免把
                            // “拖到参数编辑器”误判为“拖到时间轴”（导入目标默认值
                            // 也会因此选错场景）。
                            const overTimeline =
                                bounds != null &&
                                clientX !== undefined &&
                                clientY !== undefined &&
                                clientX >= bounds.left &&
                                clientX <= bounds.right &&
                                clientY >= bounds.top &&
                                clientY <= bounds.bottom;
                            if (overTimeline) {
                                onMidiDrop?.({
                                    midiPath,
                                    trackId,
                                    startSec: beat,
                                });
                            }
                            return;
                        }

                        // 媒体文件：多个时按 "across-time" 依次铺开，单个走单文件导入。
                        if (partition.mediaPaths.length > 1) {
                            tauriDropHandledAtRef.current = Date.now();
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;

                            void dispatch(
                                importMultipleAudioAtPosition({
                                    audioPaths: [...partition.mediaPaths],
                                    mode: "across-time",
                                    trackId,
                                    startSec: beat,
                                }),
                            );

                            return;
                        }

                        // 单文件媒体：优先用准入判据通过的那一个（而不是无条件的
                        // `primaryPath`——它可能就是已被拒绝的文件）。
                        const resolvedPath = partition.mediaPaths[0] ?? null;
                        if (resolvedPath !== null) {
                            tauriDropHandledAtRef.current = Date.now();
                            tauriDraggedPathRef.current = null;
                            tauriLastDropPathRef.current = null;
                            void dispatch(
                                importAudioAtPosition({
                                    audioPath: resolvedPath,
                                    trackId,
                                    startSec: beat,
                                }),
                            );
                            return;
                        }

                        // 没有任何可接受的路径：明确吞掉本次 drop（不建 Clip），并
                        // 标记已处理以免外层 HTML5 兜底再试着导入同一个文件。
                        tauriDropHandledAtRef.current = Date.now();
                        tauriDraggedPathRef.current = null;
                        tauriLastDropPathRef.current = null;
                    }
                });

                if (disposed && unlisten) {
                    unlisten();
                }
            } catch (err) {
                if (debugDnd) {
                    console.warn("Failed to attach Tauri drag-drop listener", err);
                }
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
        // The helper functions intentionally read refs to avoid re-registering the
        // native drag-drop listener on every render.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dispatch, snapTimeline]);

    // ── 文件浏览器面板的自定义拖拽事件 ───────────────────────
    const snapDropBeat = (rawBeat: number, trackId: string | null) => {
        if (snapTimeline && sessionRef.current.timelineSnap.enabled) {
            return snapTimeline(rawBeat, "clip", {
                anchorTrackId: trackId,
                originSec: rawBeat,
                // drop 预览左缘即被吸附边（无 clipId，行级高亮）。
                highlight: { sources: [{ trackId }] },
            });
        }
        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
        return rawBeat;
    };

    useEffect(() => {
        function onHifiFileDrag(e: Event) {
            const detail = (e as CustomEvent).detail as {
                type: string;
                filePath: string;
                fileName: string;
                clientX: number;
                clientY: number;
                durationSec: number;
                filePaths: string[];
                isRightDrag: boolean;
                /**
                 * 本次拖拽里**是目录**的路径（面板拖拽时类型已知，无需再问文件系统）。
                 *
                 * 【为什么不在拖拽开始时 stat】面板手里就有 `FileEntry.isDir`，而
                 * 拖拽移动每几毫秒就发一次事件 —— 在这里 stat 等于把 I/O 放进热路径。
                 */
                dirPaths?: string[];
            };
            if (!detail) return;

            const bounds = viewport.getRect();

            const isOverTimeline =
                bounds !== null &&
                detail.clientX >= bounds.left &&
                detail.clientX <= bounds.right &&
                detail.clientY >= bounds.top &&
                detail.clientY <= bounds.bottom;

            // 异步获取到音频时长后仅更新 ghost 宽度
            if (detail.type === "duration") {
                setDropPreview((prev) => {
                    if (prev && prev.path === detail.filePath) {
                        if (dropPreviewRef.current) {
                            const nextDuration = Number(detail.durationSec) || 0;
                            dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(nextDuration)}px`;
                        }
                        return {
                            ...prev,
                            durationSec: detail.durationSec,
                        };
                    }
                    return prev;
                });
                return;
            }

            // 移动时的 DOM 直通与重绘拦截
            if (detail.type === "move" || detail.type === "start") {
                if (isOverTimeline) {
                    const rawBeat = beatFromClientX(
                        detail.clientX,
                        bounds!,
                        viewport.getScrollLeft(),
                    );
                    const trackId = trackIdFromClientY(detail.clientY);
                    const beat = snapDropBeat(rawBeat, trackId);
                    const path = detail.filePath;
                    const fileName = detail.fileName;

                    const moveAction = detectExternalPathAction(path);
                    if (moveAction !== "importAudio" && moveAction !== "importMidi") {
                        // 非媒体文件不产生 drop 预览：snapDropBeat 已发布过吸附
                        // 高亮，这里必须清除，否则悬停期间的高亮会残留。
                        setDropPreview(null);
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }

                    setDropPreview((prev) => {
                        if (prev && dropPreviewRef.current) {
                            dropPreviewRef.current.style.left = `${Math.max(0, beat * pxPerSecRef.current)}px`;
                            dropPreviewRef.current.style.top = `${rowTopForTrackId(trackId) + 8}px`;
                            dropPreviewRef.current.style.width = `${getDropPreviewWidthPx(prev.durationSec)}px`;
                        }
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
                    clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                }
                return;
            }

            // 用户中途放弃了这次拖拽（反向键点击打断 / 指针取消）：只做清理，
            // 不执行任何导入。与 `drop` 各走一条直路，不靠 `canceled` 标志在
            // 同一个分支里分叉。
            if (detail.type === "cancel") {
                setDropPreview(null);
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                return;
            }

            if (detail.type === "drop") {
                setDropPreview(null);
                if (isOverTimeline) {
                    const rawBeat = beatFromClientX(
                        detail.clientX,
                        bounds!,
                        viewport.getScrollLeft(),
                    );
                    const trackId = trackIdFromClientY(detail.clientY);
                    const beat = snapDropBeat(rawBeat, trackId);
                    const filePaths: string[] = detail.filePaths;
                    const isMulti = Array.isArray(filePaths) && filePaths.length > 1;
                    const isRightDrag = !!detail.isRightDrag;
                    const filePath = detail.filePath;
                    const actionKind = detectExternalPathAction(filePath);

                    // 右键松开后浏览器会立即触发 contextmenu 事件；若不拦截，
                    // 该事件会被菜单 backdrop 的 onContextMenu 捕获，导致刚
                    // 弹出的菜单立刻被关闭。注册一次性 capturing 拦截器吞掉它。
                    //
                    // 【拦截器绝不能长生】若只靠 contextmenu 事件自我注销，一旦
                    // 本次右键释放没有产生该事件（窗口外释放 / 手势取消 / 平台
                    // 差异），它会滞留并吞掉用户**下一次**任意位置的右键。因此
                    // 额外挂 pointerdown / blur / 超时三条兜底，任一先到即注销
                    // （注销幂等）。
                    let suppressCtxTimer = 0;
                    const cleanupSuppressCtx = () => {
                        window.removeEventListener("contextmenu", suppressCtx, true);
                        window.removeEventListener("pointerdown", cleanupSuppressCtx, true);
                        window.removeEventListener("blur", cleanupSuppressCtx);
                        window.clearTimeout(suppressCtxTimer);
                    };
                    const suppressCtx = (ev: Event) => {
                        ev.preventDefault();
                        ev.stopImmediatePropagation();
                        cleanupSuppressCtx();
                    };
                    const armSuppressCtx = () => {
                        window.addEventListener("contextmenu", suppressCtx, true);
                        window.addEventListener("pointerdown", cleanupSuppressCtx, true);
                        window.addEventListener("blur", cleanupSuppressCtx);
                        suppressCtxTimer = window.setTimeout(cleanupSuppressCtx, 1000);
                    };

                    // 目录：面板拖拽时**类型是已知的**（`dirPaths` 由行的 `isDir` 得来），
                    // 不需要再去问文件系统 —— 这是与系统拖放的关键差别。
                    const dirPaths = new Set(detail.dirPaths ?? []);
                    const folderPaths = filePaths.filter((path) => dirPaths.has(path));
                    if (folderPaths.length > 0) {
                        if (isRightDrag) armSuppressCtx();
                        emitFolderImportRequest({
                            dirs: folderPaths,
                            // 同时拖入的媒体文件一并交给宿主，别让它们走另一条路。
                            looseFiles: filePaths.filter(
                                (path) =>
                                    !dirPaths.has(path) &&
                                    detectExternalPathAction(path) === "importAudio",
                            ),
                            trackId,
                            startSec: beat,
                            insertIndex: rootIndexAtDrop(sessionRef.current.tracks, trackId),
                            fromExplicitRequest: isRightDrag,
                        });
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }

                    if (actionKind === "openProject") {
                        // 拖入 HiFiShifter 工程（hshp/hsp）：始终弹出
                        // 「打开工程 / 导入工程」操作菜单，不直接执行打开。
                        armSuppressCtx();
                        setProjectActionMenu?.({
                            x: detail.clientX,
                            y: detail.clientY,
                            path: filePath,
                        });
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }

                    if (
                        isRightDrag &&
                        (actionKind === "importAudio" || actionKind === "importMidi")
                    ) {
                        // 右键拖拽媒体文件 → 弹出导入模式菜单
                        // （跨时间添加 / 跨轨道添加 / 作为 Take 添加）。
                        armSuppressCtx();
                        setImportModeMenu({
                            x: detail.clientX,
                            y: detail.clientY,
                            audioPaths: isMulti ? filePaths : [filePath],
                            trackId,
                            startSec: beat,
                        });
                        clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                        return;
                    }

                    if (isMulti) {
                        setImportModeMenu({
                            x: detail.clientX,
                            y: detail.clientY,
                            audioPaths: filePaths,
                            trackId,
                            startSec: beat,
                        });
                    } else {
                        if (actionKind === "importMidi") {
                            onMidiDrop?.({
                                midiPath: filePath,
                                trackId,
                                startSec: beat,
                            });
                        } else if (
                            actionKind &&
                            actionKind !== "importAudio" &&
                            actionKind !== "importFolder"
                        ) {
                            // rpp / vshp / vsp 工程文件：直接视为对应格式的导入。
                            emitExternalFileAction(actionKind, filePath);
                        } else {
                            void dispatch(
                                importAudioAtPosition({
                                    audioPath: filePath,
                                    trackId,
                                    startSec: beat,
                                }),
                            );
                        }
                    }
                }
                // drop 手势结束：上面的 snapDropBeat 在“导入位置恰好吸附”时会
                // 再次发布吸附竖线高亮（先清除再计算会导致此结果），必须在手势
                // 出口统一清除，否则导入完成后高亮会残留在画面上。
                clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
                return;
            }
        }

        window.addEventListener("hifi-file-drag", onHifiFileDrag);
        return () => {
            window.removeEventListener("hifi-file-drag", onHifiFileDrag);
        };
        // Re-registering this DOM listener whenever transient helpers change would
        // interrupt in-flight drag state; the helpers read the latest refs instead.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [dispatch, pxPerSec, rowHeight, snapTimeline]);

    return {
        tauriDraggedPathRef,
        tauriLastDropPathRef,
        tauriDropHandledAtRef,
    };
}
