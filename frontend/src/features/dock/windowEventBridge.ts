/**
 * 跨窗口 UI 命令桥。
 *
 * DOM CustomEvent 只在一个 window 内传播：独立窗口里的文件浏览器发出「打开工程 /
 * 导入 MIDI / 导入文件夹」时，主窗口的 App 收不到；主窗口 MenuBar 发出的编辑
 * 操作也到不了卫星窗口里的参数编辑器。
 *
 * 本模块把「窗口目标」显式化：
 * - main：只由主窗口消费（文件打开、导入、对话框宿主）；
 * - activeEditor：由最近获得焦点的编辑窗口消费（参数编辑命令）；
 * - all：仅用于明确的广播型通知。
 *
 * 投递方式是 Tauri 全局 emit + 本窗口即时 dispatch；收到远端信封后只做本地
 * DOM dispatch，绝不再次 emit（origin + targetWindow 双重过滤防回声）。
 */

import { store } from "../../app/store";
import { emit as emitHostEvent, listen as listenHostEvent } from "../../services/hostEvents";
import { resolveWindowLabel, isSatelliteWindow } from "./detachBridge";

export const WINDOW_EVENT_BRIDGE_EVENT = "hs:window-event";
/** 卫星窗口把非编辑类快捷键交回主窗口执行。 */
export const KEYBINDING_ACTION_EVENT = "hs:keybinding-action";

export type WindowEventTarget = "main" | "activeEditor" | "all";

interface WindowEventEnvelope {
    origin: string;
    /** null = 广播给所有窗口；否则只投递给该窗口标签。 */
    targetWindow: string | null;
    name: string;
    detail: unknown;
}

function localDispatch(name: string, detail: unknown): void {
    if (typeof window === "undefined") return;
    window.dispatchEvent(new CustomEvent(name, { detail }));
}

function resolveTargetWindow(target: WindowEventTarget): string | null {
    if (target === "all") return null;
    if (target === "main") return "main";
    const active = store.getState().interaction.activeEditorWindow;
    if (active !== null) return active;
    // 没有活动编辑窗口时：卫星窗口回退到自己（命令来自本窗口，消费者也在
    // 本窗口）；主窗口回退到 main。
    return isSatelliteWindow() ? resolveWindowLabel() : "main";
}

/**
 * 发送一个 UI 命令。当前窗口就是目标时立即本地 dispatch；否则经 Tauri 广播，
 * 由目标窗口的 bridge 转成同名 DOM CustomEvent。
 */
export function emitWindowEvent(target: WindowEventTarget, name: string, detail: unknown): void {
    if (typeof window === "undefined") return;
    const origin = resolveWindowLabel();
    const targetWindow = resolveTargetWindow(target);
    const local = targetWindow === null || targetWindow === origin;
    if (local) localDispatch(name, detail);
    // 目标只在当前窗口时无需广播；all 与跨窗口目标才发。
    if (local && targetWindow !== null) return;
    void emitHostEvent(WINDOW_EVENT_BRIDGE_EVENT, {
        origin,
        targetWindow,
        name,
        detail,
    } satisfies WindowEventEnvelope).catch(() => undefined);
}

/**
 * 安装本窗口的接收端。App 与 detachedMain 各调用一次。
 *
 * @returns 拆除函数。
 */
export function installWindowEventBridge(): () => void {
    let disposed = false;
    let unlisten: (() => void) | null = null;

    void listenHostEvent<WindowEventEnvelope>(WINDOW_EVENT_BRIDGE_EVENT, (event) => {
        if (disposed) return;
        const envelope = event.payload;
        if (!envelope || typeof envelope.name !== "string") return;
        const origin = resolveWindowLabel();
        if (envelope.origin === origin) return;
        if (envelope.targetWindow !== null && envelope.targetWindow !== origin) return;
        localDispatch(envelope.name, envelope.detail);
    })
        .then((off) => {
            if (disposed) off();
            else unlisten = off;
        })
        .catch(() => undefined);

    return () => {
        disposed = true;
        if (unlisten) unlisten();
        unlisten = null;
    };
}
