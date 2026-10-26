/*
 * 浏览器默认行为守卫：Ctrl/Cmd+F/P/G（查找/打印）、Ctrl+R 刷新、页面缩放键、
 * Ctrl+滚轮缩放（含触控板捏合派发）、原生右键菜单、原生文本选取/拖拽、中键
 * 自动滚动、Alt 单击。
 *
 * 【为什么抽成 hook】这组守卫原来只住在主窗口的 AppInner 里 —— 但独立窗口
 * （detached.html，另一个 JS 上下文）同样是 WebView 页面，一模一样的默认行为
 * 在那边全都存在：Ctrl+滚轮会页面缩放、F5 会刷新、右键会弹原生菜单。抽出来
 * 让两个入口都挂载，行为不再分叉。
 *
 * 【为什么全部 preventDefault 而不阻断传播】只调用 preventDefault：
 * 应用内基于同一事件的自定义逻辑（快捷键、右键拖拽、缩放滚轮）仍会收到事件。
 *
 * 【Linux 右键时序】WebKitGTK 在右键**按下**时就发 contextmenu（Windows 语义
 * 是松开）。Linux 下先扣住，pointerup 重放 —— 本地右键拖拽处理器自己
 * preventDefault/stopPropagation 的仍完全掌控自己的流程。
 */

import { useEffect, useRef } from "react";

import { useAppSelector } from "../app/hooks";
import { IS_LINUX } from "../utils/platform";
import { allowsNativeTextSelection, isEditableTarget } from "../utils/nativeSelectionGuards";
import {
    disposeRightDragContextMenuGuard,
    installRightDragContextMenuGuard,
} from "../utils/rightDragContextMenuGuard";
import { emitPinch, pinchDeltaFromWheel } from "../utils/pinchGesture";

export function useBrowserDefaultGuards(): void {
    // 触控板捏合是否接管缩放（设置项）。全局监听只挂载一次，直接闭包会永远
    // 读到初始值 —— 经 ref 转发后设置一改即生效，不必重挂监听器。
    const pinchZoomEnabled = useAppSelector((state) => state.session.penInput.trackpadPinchZoom);
    const pinchZoomEnabledRef = useRef(pinchZoomEnabled);
    useEffect(() => {
        pinchZoomEnabledRef.current = pinchZoomEnabled;
    });

    // 改用 useRef，取消重绘：Alt 是否"被当作修饰键用过"（决定 keyup 要不要拦）。
    const isModifierRef = useRef(false);

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
        let linuxDeferredDispatchTimer: number | null = null;
        const flushLinuxDeferredContextMenu = () => {
            const pending = linuxDeferredContextMenu;
            linuxDeferredContextMenu = null;
            linuxRightButtonDown = false;
            if (!pending) return;
            // 句柄要留在 effect 里：卸载时清掉这拍延迟派发，避免向已拆除的
            // 事件树重放合成 contextmenu（StrictMode 双挂载窗口尤其如此）。
            linuxDeferredDispatchTimer = window.setTimeout(() => {
                linuxDeferredDispatchTimer = null;
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
            if (linuxDeferredDispatchTimer !== null) {
                window.clearTimeout(linuxDeferredDispatchTimer);
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
        // 只挂载一次；设置项经 ref 转发（见上方 pinchZoomEnabledRef 的说明）。
    }, []);
}
