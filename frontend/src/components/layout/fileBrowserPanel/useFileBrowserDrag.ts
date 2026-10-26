// FileBrowser 的自定义 pointer 拖拽状态机。拆自 FileBrowserPanel。
import { useCallback, useEffect, useRef, useState } from "react";
import { setFileBrowserDragActive } from "../../../features/fileBrowser/fileBrowserDragStore";
import {
    FILE_DRAG_THRESHOLD_PX,
    buildFileDragFinishDetail,
    dragButtonOf,
    interruptCandidateMoved,
    interruptCandidateOnDown,
    isInterruptRelease,
    type DragInterruptCandidate,
    type FileDragSource,
} from "../../../features/fileBrowser/fileBrowserDragGesture";
import { fileBrowserApi, type FileEntry } from "../../../services/api/fileBrowser";
import type { FileBrowserListing } from "./useFileBrowserListing";
import type { FileBrowserSelection } from "./useFileBrowserSelection";

/** 一次进行中的文件拖拽（自定义 pointer 事件，替代 HTML5 drag API）。 */
interface FileDragState extends FileDragSource {
    startX: number;
    startY: number;
    active: boolean;
}

export function useFileBrowserDrag(listing: FileBrowserListing, selection: FileBrowserSelection) {
    const { displayEntries } = listing;
    const { selectedPaths } = selection;
    const [dragState, setDragState] = useState<FileDragState | null>(null);
    const dragStateRef = useRef(dragState);
    useEffect(() => {
        dragStateRef.current = dragState;
    }, [dragState]);

    // ghost 元素跟随鼠标
    const ghostRef = useRef<HTMLDivElement | null>(null);

    const DRAG_THRESHOLD = FILE_DRAG_THRESHOLD_PX; // 越过它才算拖拽（与打断的"点击"判定同一阈值）

    const handlePointerDownForDrag = useCallback(
        (e: React.PointerEvent<HTMLDivElement>, entry: FileEntry) => {
            // 允许左键(0)和右键(2)拖拽
            if (e.button !== 0 && e.button !== 2) return;
            // 拖拽顺序取**列表显示顺序**，与右键菜单的 `selectedEntries` 同源。
            // 此前用 `Array.from(selectedPaths)` —— 那是**点击顺序**，于是同一份
            // 选区在"右键插入"和"拖拽插入"下给出两种顺序。目录导入时顺序决定
            // 轨道顺序，这种分叉会变成"轨道顺序和我看到的不一样"。
            const dragged =
                selectedPaths.size > 0 && selectedPaths.has(entry.path)
                    ? displayEntries.filter((item) => selectedPaths.has(item.path))
                    : [entry];
            // 不拦截 pointer，让 click 事件仍能触发预览
            setDragState({
                filePath: entry.path,
                fileName: entry.name,
                allFilePaths: dragged.map((item) => item.path),
                dirPaths: dragged.filter((item) => item.isDir).map((item) => item.path),
                startX: e.clientX,
                startY: e.clientY,
                active: false,
                isRightDrag: e.button === 2,
            });
        },
        [displayEntries, selectedPaths],
    );

    useEffect(() => {
        if (!dragState) return;

        /**
         * 反向键的一次**点击**（按下后没越过阈值就松开）= 放弃本次拖拽。
         *
         * 【为什么判据是"点击"而不是"按下"】右键拖拽中按下左键（不松开）是
         * "再加一个键"，不是"我后悔了"。判定见 `fileBrowserDragGesture`。
         */
        let interrupt: DragInterruptCandidate | null = null;

        /**
         * 结束一次拖拽：派发 `drop`（在此处放下）或 `cancel`（放弃），并清干净本地状态。
         */
        function finishDrag(
            ds: FileDragState,
            type: "drop" | "cancel",
            clientX: number,
            clientY: number,
        ) {
            window.dispatchEvent(
                new CustomEvent("hifi-file-drag", {
                    detail: buildFileDragFinishDetail(ds, type, clientX, clientY),
                }),
            );
            interrupt = null;
            setFileBrowserDragActive(false);
            setDragState(null);
        }

        /**
         * 吞掉紧随打断（右键点击）之后的那一次原生 `contextmenu`。
         *
         * 左键拖拽中按右键打断时，平台随后会派发 contextmenu；不吞掉的话，
         * "打断"就换来了一个菜单。命中一次即自卸（与 `useTimelineDragDrop`
         * 的 `suppressCtx` 同款）。
         *
         * 【必须有有效期】平台要么在右键松开后立刻派发 contextmenu、要么根本
         * 不派发（如在窗口外松开）。监听器若只等下一次 contextmenu 自卸，
         * 那次"不来"之后它会一直挂着，把用户之后**任意位置的下一次右键菜单**
         * 吞掉。给它一个远大于平台派发延迟的短超时兜底自卸。注意不能用本
         * effect 的 cleanup 兜底：打断收尾会让依赖变化、cleanup 先于平台
         * 派发 contextmenu 跑，会吞不掉该吞的那一次。
         */
        let activeSwallow: ((ev: Event) => void) | null = null;
        let swallowExpiryTimer: number | null = null;
        function removeSwallow() {
            if (swallowExpiryTimer != null) {
                window.clearTimeout(swallowExpiryTimer);
                swallowExpiryTimer = null;
            }
            if (activeSwallow) {
                window.removeEventListener("contextmenu", activeSwallow, true);
                activeSwallow = null;
            }
        }
        function suppressNextContextMenu() {
            if (activeSwallow) return;
            const swallow = (ev: Event) => {
                ev.preventDefault();
                ev.stopImmediatePropagation();
                removeSwallow();
            };
            activeSwallow = swallow;
            window.addEventListener("contextmenu", swallow, true);
            // 与 useTimelineDragDrop 的 suppressCtx 同款超时兜底。
            swallowExpiryTimer = window.setTimeout(removeSwallow, 1000);
        }

        function onInterruptPointerDown(e: PointerEvent) {
            const ds = dragStateRef.current;
            if (!ds) return;
            interrupt = interruptCandidateOnDown({
                active: ds.active,
                isRightDrag: ds.isRightDrag,
                button: e.button,
                x: e.clientX,
                y: e.clientY,
            });
        }

        function onPointerMove(e: PointerEvent) {
            const ds = dragStateRef.current;
            if (!ds) return;

            // 反向键按下后指针又移动超过阈值 → 那不是一次点击，取消打断意图。
            if (interrupt && interruptCandidateMoved(interrupt, e.clientX, e.clientY)) {
                interrupt = null;
            }

            if (!ds.active) {
                const dx = e.clientX - ds.startX;
                const dy = e.clientY - ds.startY;
                if (Math.sqrt(dx * dx + dy * dy) < DRAG_THRESHOLD) return;
                // 激活拖拽
                dragStateRef.current = { ...ds, active: true };
                setDragState(dragStateRef.current);
                // 越过阈值：指针已离开源行，浮标应当消失（见 fileBrowserDragStore）。
                setFileBrowserDragActive(true);
                // 发送拖拽开始事件
                window.dispatchEvent(
                    new CustomEvent("hifi-file-drag", {
                        detail: {
                            type: "start",
                            filePath: ds.filePath,
                            fileName: ds.fileName,
                            filePaths: ds.allFilePaths,
                            dirPaths: ds.dirPaths,
                            clientX: e.clientX,
                            clientY: e.clientY,
                            isRightDrag: ds.isRightDrag,
                        },
                    }),
                );
                // 异步获取音频时长，获取后通知 TimelinePanel 更新 ghost 宽度
                fileBrowserApi
                    .getAudioFileInfo(ds.filePath)
                    .then((info) => {
                        if (info && dragStateRef.current?.filePath === ds.filePath) {
                            window.dispatchEvent(
                                new CustomEvent("hifi-file-drag", {
                                    detail: {
                                        type: "duration",
                                        filePath: ds.filePath,
                                        durationSec: info.durationSec,
                                    },
                                }),
                            );
                        }
                    })
                    .catch(() => {
                        /* 获取失败则保持默认宽度 */
                    });
            }

            // 更新 ghost 位置（clamp 到窗口可视范围内，鼠标超出界面时 ghost 停在边缘。
            // 余量按 ghost 实测尺寸计算：文件名长短不一，固定余量会让 ghost
            // 在窗口右缘脱离光标，长文件名仍会溢出右边界）。
            if (ghostRef.current) {
                const ghost = ghostRef.current;
                const clampedX = Math.max(
                    0,
                    Math.min(e.clientX + 12, window.innerWidth - ghost.offsetWidth - 4),
                );
                const clampedY = Math.max(
                    0,
                    Math.min(e.clientY + 12, window.innerHeight - ghost.offsetHeight - 4),
                );
                ghost.style.left = `${clampedX}px`;
                ghost.style.top = `${clampedY}px`;
            }

            // 发送拖拽移动事件（TimelinePanel 监听）
            window.dispatchEvent(
                new CustomEvent("hifi-file-drag", {
                    detail: {
                        type: "move",
                        filePath: dragStateRef.current!.filePath,
                        fileName: dragStateRef.current!.fileName,
                        filePaths: dragStateRef.current!.allFilePaths,
                        dirPaths: dragStateRef.current!.dirPaths,
                        clientX: e.clientX,
                        clientY: e.clientY,
                        isRightDrag: dragStateRef.current!.isRightDrag,
                    },
                }),
            );
        }

        function onPointerUp(e: PointerEvent) {
            const ds = dragStateRef.current;
            if (!ds) return;
            // 反向键的一次点击 = 打断：放弃本次拖拽，不做 drop。
            if (ds.active && isInterruptRelease(interrupt, e.button)) {
                if (e.button === 2) suppressNextContextMenu();
                finishDrag(ds, "cancel", e.clientX, e.clientY);
                return;
            }
            // 反向键的松开（未构成点击）不结束拖拽：发起键还按着。
            if (e.button !== dragButtonOf(ds.isRightDrag)) return;
            if (ds.active) {
                finishDrag(ds, "drop", e.clientX, e.clientY);
                return;
            }
            // 未越过阈值：这只是一次点击（选中 / 试听），收掉待定的拖拽态。
            setDragState(null);
        }

        // 指针在窗口外（任务栏/另一显示器）松开时 pointerup 不会派发，
        // pointercancel / lostpointercapture 是唯一可靠的收尾信号 —— 否则
        // 拖拽态永久卡死（ghost 滞留、列表行保持半透明、move 事件持续派发）。
        function onPointerCancel() {
            const ds = dragStateRef.current;
            if (ds?.active) {
                finishDrag(ds, "cancel", ds.startX, ds.startY);
                return;
            }
            setDragState(null);
        }

        // 右键拖拽时抑制浏览器原生右键菜单
        function onContextMenu(e: MouseEvent) {
            if (dragStateRef.current?.isRightDrag) {
                e.preventDefault();
            }
        }

        window.addEventListener("pointerdown", onInterruptPointerDown, true);
        window.addEventListener("pointermove", onPointerMove);
        window.addEventListener("pointerup", onPointerUp);
        window.addEventListener("pointercancel", onPointerCancel);
        window.addEventListener("blur", onPointerCancel);
        window.addEventListener("contextmenu", onContextMenu, true);
        return () => {
            window.removeEventListener("pointerdown", onInterruptPointerDown, true);
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("pointercancel", onPointerCancel);
            window.removeEventListener("blur", onPointerCancel);
            window.removeEventListener("contextmenu", onContextMenu, true);
        };
    }, [dragState !== null]); // eslint-disable-line react-hooks/exhaustive-deps

    /*
     * 卸载即收尾。
     *
     * 【为什么单开一个空依赖 effect】上面的拖拽 effect 以 `dragState !== null` 为
     * 依赖 —— 拖拽**开始**时它也会重跑一次清理，若在清理里补发 `cancel` 会立刻把
     * 刚开始的拖拽取消掉。空依赖的清理只在**真正卸载**时执行，语义精确。
     *
     * 【不补发会怎样】面板一旦卸载，pointerup / pointercancel / blur 都不再派发，
     * 而监听 `hifi-file-drag` 的**时间轴仍然挂载** —— 它手里的落点预览与吸附竖线
     * 会永久留在画面上（正是"高亮没清掉"那一类）。
     */
    useEffect(
        () => () => {
            const ds = dragStateRef.current;
            if (!ds?.active) return;
            window.dispatchEvent(
                new CustomEvent("hifi-file-drag", {
                    detail: buildFileDragFinishDetail(ds, "cancel", ds.startX, ds.startY),
                }),
            );
            // 全局拖拽态存在 store 里（不在本组件），卸载后仍需复位，否则文件
            // 浏览器再次挂载时行会以"正在拖拽"的样式出现。
            setFileBrowserDragActive(false);
        },
        [],
    );

    return { dragState, ghostRef, handlePointerDownForDrag };
}

export type FileBrowserDrag = ReturnType<typeof useFileBrowserDrag>;
