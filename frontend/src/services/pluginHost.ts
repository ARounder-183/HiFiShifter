// 原GUI的原生插件通信：不伪造Tauri，不让命令/事件串到其它FX实例。
export type HostEvent<T> = { event: string; id: number; payload: T };
export type PluginBootstrap = {
    version: 1;
    viewId: string;
    transportControl?: boolean;
    clipEditing?: boolean;
    clipSplitting?: boolean;
    clipClipboard?: boolean;
    audioImport?: boolean;
    trackGrouping?: boolean;
    /**
     * 宿主用哪一套淡化轴。
     *
     * - `"legacy"`：REAPER ≤7.80，`C_FADE*SHAPE` 决定形状 → 预设按钮可用；
     * - `"continuous"`：REAPER ≥7.81，curvature/S 两轴决定形状 → 只给连续滑杆
     *   （预设到 (curvature, S) 的映射尚未校准，不摆按钮假装能用）；
     * - 缺省/`null`：版本读不出来 → 整块淡变编辑保持只读。
     */
    fadeAxes?: "legacy" | "continuous" | null;
};
export interface WebViewMessagePort {
    postMessage(message: unknown): void;
    postMessageWithAdditionalObjects?(message: unknown, objects: File[]): void;
    addEventListener(name: "message", listener: (event: { data: unknown }) => void): void;
    removeEventListener(name: "message", listener: (event: { data: unknown }) => void): void;
}
export interface PluginHostBridge {
    readonly kind: "plugin";
    invoke<T>(command: string, args?: Record<string, unknown>, files?: File[]): Promise<T>;
    listen<T>(event: string, listener: (event: HostEvent<T>) => void): Promise<() => void>;
    dispose(): void;
}
declare global {
    interface Window {
        __HFS_PLUGIN_BOOTSTRAP__?: PluginBootstrap;
        chrome?: { webview?: WebViewMessagePort };
    }
}
type Pending = {
    resolve(value: unknown): void;
    reject(error: Error): void;
    timer: ReturnType<typeof setTimeout>;
};

/** 宿主IPlugView兜底按键继续经过原GUI监听器，保留输入框、模态及自定义键位的规则。 */
function forwardPluginKeyboard(payload: unknown): void {
    if (!payload || typeof payload !== "object" || typeof document === "undefined") return;
    const key = payload as Record<string, unknown>;
    if (
        (key.type !== "keydown" && key.type !== "keyup") ||
        typeof key.key !== "string" ||
        !["c", "x", "v", "z", "y"].includes(key.key) ||
        key.ctrlKey !== true
    )
        return;
    const target = document.activeElement ?? document.body;
    target.dispatchEvent(
        new KeyboardEvent(key.type, {
            key: key.key,
            code: `Key${key.key.toUpperCase()}`,
            ctrlKey: true,
            shiftKey: key.shiftKey === true,
            altKey: key.altKey === true,
            metaKey: key.metaKey === true,
            repeat: key.repeat === true,
            bubbles: true,
            cancelable: true,
        }),
    );
}

/** 用宿主注入的view身份关联有界请求；只能绑定原生WebView消息口。 */
export function createPluginHost(
    port: WebViewMessagePort,
    boot: PluginBootstrap,
): PluginHostBridge {
    if (boot.version !== 1 || !boot.viewId || boot.viewId.length > 128) {
        throw new Error("Invalid plugin host bootstrap");
    }
    let closed = false;
    let nextId = 1;
    let eventId = 1;
    const pending = new Map<number, Pending>();
    const listeners = new Map<string, Set<(event: HostEvent<unknown>) => void>>();
    function receive(event: { data: unknown }) {
        if (closed || !event.data || typeof event.data !== "object") return;
        const data = event.data as Record<string, unknown>;
        if (data.version !== 1 || data.viewId !== boot.viewId) return;
        if (data.event === "plugin_keyboard") {
            forwardPluginKeyboard(data.payload);
            return;
        }
        if (typeof data.event === "string") {
            const notification = { event: data.event, id: eventId++, payload: data.payload };
            for (const handler of Array.from(listeners.get(data.event) ?? [])) {
                try {
                    handler(notification);
                } catch (error) {
                    console.error("Plugin event handler failed", error);
                }
            }
            return;
        }
        if (typeof data.id !== "number" || !Number.isSafeInteger(data.id)) return;
        const slot = pending.get(data.id);
        if (!slot) return;
        pending.delete(data.id);
        clearTimeout(slot.timer);
        if (data.ok === true) slot.resolve(data.value);
        else
            slot.reject(
                new Error(typeof data.error === "string" ? data.error : "Invalid plugin response"),
            );
    }
    port.addEventListener("message", receive);
    return {
        kind: "plugin",
        invoke<T>(command: string, args?: Record<string, unknown>, files?: File[]): Promise<T> {
            if (closed) return Promise.reject(new Error("Plugin editor closed"));
            if (pending.size >= 128)
                return Promise.reject(new Error("Too many pending plugin requests"));
            if (!command || command.length > 128 || !Number.isSafeInteger(nextId)) {
                return Promise.reject(
                    new Error("Invalid plugin command or exhausted request identity"),
                );
            }
            if (
                files &&
                (!port.postMessageWithAdditionalObjects ||
                    command !== "import_native_audio_file" ||
                    files.length !== 1)
            ) {
                return Promise.reject(
                    new Error("Native File import unavailable; use File menu import"),
                );
            }
            const id = nextId++;
            return new Promise<T>((resolve, reject) => {
                const timer = setTimeout(() => {
                    pending.delete(id);
                    reject(new Error(`Plugin command timed out: ${command}`));
                }, 30_000);
                pending.set(id, { resolve: (value) => resolve(value as T), reject, timer });
                try {
                    const message = { version: 1, viewId: boot.viewId, id, command, args };
                    if (files) port.postMessageWithAdditionalObjects!(message, files);
                    else port.postMessage(message);
                } catch (error) {
                    clearTimeout(timer);
                    pending.delete(id);
                    reject(error instanceof Error ? error : new Error(String(error)));
                }
            });
        },
        async listen<T>(event: string, listener: (event: HostEvent<T>) => void) {
            if (closed) throw new Error("Plugin editor closed");
            const handlers = listeners.get(event) ?? new Set();
            const handler = listener as (event: HostEvent<unknown>) => void;
            handlers.add(handler);
            listeners.set(event, handlers);
            return () => {
                handlers.delete(handler);
                if (!handlers.size) listeners.delete(event);
            };
        },
        dispose() {
            if (closed) return;
            closed = true;
            port.removeEventListener("message", receive);
            for (const slot of pending.values()) {
                clearTimeout(slot.timer);
                slot.reject(new Error("Plugin editor closed"));
            }
            pending.clear();
            listeners.clear();
        },
    };
}

let active: PluginHostBridge | null = null;

/** 独立app没有插件bootstrap，原调用路径不会被WebView2对象本身误判为插件。 */
export function getPluginHost(): PluginHostBridge | null {
    if (typeof window === "undefined" || !window.__HFS_PLUGIN_BOOTSTRAP__) return null;
    if (active) return active;
    const port = window.chrome?.webview;
    if (!port) throw new Error("Plugin bootstrap present but native message port unavailable");
    active = createPluginHost(port, window.__HFS_PLUGIN_BOOTSTRAP__);
    window.addEventListener("pagehide", () => active?.dispose(), { once: true });
    return active;
}
