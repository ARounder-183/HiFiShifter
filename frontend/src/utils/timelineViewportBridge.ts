/**
 * 跨窗口时间轴视口桥。
 *
 * timelineViewportSync 是模块级单例：独立窗口是另一个 JS 上下文，主窗口的订阅者
 * 收不到卫星窗口里的滚动/缩放，卫星窗口也从未被播种。这里用窗口事件桥广播
 * {origin, scrollLeft, pxPerSec}，而不是把 60Hz 视口放进 Redux（那会让整棵树每帧重渲染）。
 *
 * 只在非插件模式、且布局里确有 osWindow 的时间轴/参数编辑器时广播。没有卫星窗口时
 * 完全 no-op；收到 remote: 来源的写入不再回声。
 */
import { store } from "../app/store";
import { isPluginMode } from "../services/hostCapabilities";
import { frontendLog } from "../services/frontendLog";
import { isPanelForm, collectDockedForms } from "../features/dock/dockTree";
import { PANEL_PARAM_EDITOR, PANEL_TIMELINE } from "../components/dock/registerBuiltinPanels";
import { isSatelliteWindow, resolveWindowLabel } from "../features/dock/detachBridge";
import { emitWindowEvent } from "../features/dock/windowEventBridge";
import { timelineViewportSync } from "./timelineViewportSync";

export const TIMELINE_VIEWPORT_EVENT = "hs:timeline-viewport";
export const TIMELINE_VIEWPORT_REQUEST_EVENT = "hs:timeline-viewport-request";

interface TimelineViewportEnvelope {
    origin: string;
    scrollLeft: number;
    pxPerSec: number;
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value);
}

function formTreeContainsEditor(
    layout: ReturnType<typeof store.getState>["dock"]["layout"],
    formId: string,
    depth: number,
): boolean {
    if (depth > 16) return false;
    const form = layout.forms[formId];
    if (!form) return false;
    if (form.panelId === PANEL_TIMELINE || form.panelId === PANEL_PARAM_EDITOR) return true;
    if (!isPanelForm(form) || !form.childRootId) return false;
    const tree = layout.roots[form.childRootId];
    if (!tree) return false;
    return collectDockedForms(tree).some((memberId) =>
        formTreeContainsEditor(layout, memberId, depth + 1),
    );
}

/** 布局里是否存在拆到独立系统窗口的编辑器窗体（含包含它们的面板）。 */
export function hasOsWindowEditorForm(
    layout: ReturnType<typeof store.getState>["dock"]["layout"],
): boolean {
    return Object.values(layout.forms).some(
        (form) =>
            form.floating === true &&
            form.floatMode === "osWindow" &&
            formTreeContainsEditor(layout, form.id, 0),
    );
}

/**
 * 安装本窗口的视口桥；返回拆除函数。
 *
 * 主窗口与卫星窗口各安装一次。卫星启动后发请求，主窗口把已播种的当前视口回给它；
 * 请求每 500ms 重试一次，最多 10 秒，避免主窗口布局尚未恢复时的启动竞态。
 */
export function installTimelineViewportBridge(): () => void {
    if (typeof window === "undefined" || isPluginMode()) return () => undefined;

    const self = resolveWindowLabel();
    let disposed = false;
    let frame: number | null = null;
    let lastBroadcastAt = 0;
    let requestTimer: number | null = null;
    let requestAttempts = 0;
    let receivedRemote = false;

    const canBroadcast = () => hasOsWindowEditorForm(store.getState().dock.layout);

    const broadcast = () => {
        frame = null;
        if (disposed || !canBroadcast()) return;
        const now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (now - lastBroadcastAt < 33) {
            frame = requestAnimationFrame(broadcast);
            return;
        }
        lastBroadcastAt = now;
        const viewport = timelineViewportSync.get();
        emitWindowEvent("all", TIMELINE_VIEWPORT_EVENT, {
            origin: self,
            scrollLeft: viewport.scrollLeft,
            pxPerSec: viewport.pxPerSec,
        } satisfies TimelineViewportEnvelope);
    };

    const scheduleBroadcast = () => {
        if (disposed || frame !== null) return;
        frame = requestAnimationFrame(broadcast);
    };

    const unsubscribe = timelineViewportSync.subscribe(() => {
        const origin = timelineViewportSync.getOrigin();
        // remote: 来源是刚收到的别的窗口写入，不能再广播回去形成回声。
        if (origin?.startsWith("remote:")) return;
        scheduleBroadcast();
    });

    const onRemote = (event: Event) => {
        if (disposed) return;
        const detail = (event as CustomEvent<TimelineViewportEnvelope>).detail;
        if (!detail || detail.origin === self) return;
        if (!isFiniteNumber(detail.scrollLeft) || !isFiniteNumber(detail.pxPerSec)) return;
        receivedRemote = true;
        timelineViewportSync.setViewport(
            { scrollLeft: detail.scrollLeft, pxPerSec: detail.pxPerSec },
            "remote:" + detail.origin,
        );
        if (requestTimer !== null) {
            window.clearTimeout(requestTimer);
            requestTimer = null;
        }
    };

    const onRequest = (event: Event) => {
        if (disposed) return;
        const detail = (event as CustomEvent<{ origin?: unknown }>).detail;
        if (!detail || detail.origin === self) return;
        if (!canBroadcast() || !timelineViewportSync.isSeeded()) return;
        const viewport = timelineViewportSync.get();
        emitWindowEvent("all", TIMELINE_VIEWPORT_EVENT, {
            origin: self,
            scrollLeft: viewport.scrollLeft,
            pxPerSec: viewport.pxPerSec,
        } satisfies TimelineViewportEnvelope);
    };

    const requestSeed = () => {
        requestTimer = null;
        if (disposed || receivedRemote || requestAttempts >= 20) return;
        requestAttempts += 1;
        emitWindowEvent("all", TIMELINE_VIEWPORT_REQUEST_EVENT, { origin: self });
        requestTimer = window.setTimeout(requestSeed, 500);
    };

    window.addEventListener(TIMELINE_VIEWPORT_EVENT, onRemote);
    window.addEventListener(TIMELINE_VIEWPORT_REQUEST_EVENT, onRequest);
    if (isSatelliteWindow()) {
        requestSeed();
    } else if (canBroadcast()) {
        // 主窗口已有卫星窗口但目前没有发生滚动：先广播一次，卫星安装后也能拿到当前值。
        scheduleBroadcast();
    }

    return () => {
        disposed = true;
        unsubscribe();
        if (frame !== null) cancelAnimationFrame(frame);
        if (requestTimer !== null) window.clearTimeout(requestTimer);
        window.removeEventListener(TIMELINE_VIEWPORT_EVENT, onRemote);
        window.removeEventListener(TIMELINE_VIEWPORT_REQUEST_EVENT, onRequest);
        frontendLog("debug", "[timeline-viewport] bridge disposed");
    };
}
