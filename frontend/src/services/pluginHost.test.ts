// 插件通信回归：覆盖真实bridge的实例隔离、错误、取消和超时，不模拟编辑成功。
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPluginHost, type WebViewMessagePort } from "./pluginHost";

/** 只替代原生WebView边界；Promise关联、事件与生命周期全部执行生产bridge。 */
function nativePort() {
    const listeners = new Set<(event: { data: unknown }) => void>();
    const sent: Record<string, unknown>[] = [];
    const port: WebViewMessagePort = {
        postMessage(message) {
            sent.push(message as Record<string, unknown>);
        },
        addEventListener(_name, listener) {
            listeners.add(listener);
        },
        removeEventListener(_name, listener) {
            listeners.delete(listener);
        },
    };
    return { port, sent, deliver: (data: unknown) => listeners.forEach((fn) => fn({ data })) };
}

afterEach(() => vi.useRealTimers());

describe("plugin host protocol", () => {
    it("existing invoke mapping reaches plugin without pretending to be Tauri", async () => {
        vi.resetModules();
        const native = nativePort();
        const lifecycle = new Set<() => void>();
        vi.stubGlobal("window", {
            __HFS_PLUGIN_BOOTSTRAP__: { version: 1, viewId: "mapped-view" },
            chrome: { webview: native.port },
            addEventListener: (_name: string, fn: () => void) => lifecycle.add(fn),
        });
        try {
            const { invoke } = await import("./invoke");
            const response = invoke("set_transport", 1.5, undefined);
            expect(native.sent[0]).toEqual({
                version: 1,
                viewId: "mapped-view",
                id: 1,
                command: "set_transport",
                args: { playheadSec: 1.5 },
            });
            native.deliver({
                version: 1,
                viewId: "mapped-view",
                id: 1,
                ok: true,
                value: { ok: true },
            });
            await expect(response).resolves.toEqual({ ok: true });
        } finally {
            lifecycle.forEach((close) => close());
            vi.unstubAllGlobals();
        }
    });
    it("resolves only the matching view and request", async () => {
        const native = nativePort();
        const host = createPluginHost(native.port, { version: 1, viewId: "view-a" });
        const response = host.invoke("set_transport", { playheadSec: 1.25 });
        expect(native.sent[0]).toEqual({
            version: 1,
            viewId: "view-a",
            id: 1,
            command: "set_transport",
            args: { playheadSec: 1.25 },
        });
        let settled = false;
        void response.then(() => {
            settled = true;
        });
        native.deliver({ version: 1, viewId: "view-b", id: 1, ok: true, value: "wrong" });
        native.deliver({ version: 2, viewId: "view-a", id: 1, ok: true, value: "wrong" });
        await Promise.resolve();
        expect(settled).toBe(false);
        native.deliver({ version: 1, viewId: "view-a", id: 1, ok: true, value: { ok: true } });
        await expect(response).resolves.toEqual({ ok: true });
        host.dispose();
    });

    it("rejects native failure and timeout, freeing request slots", async () => {
        vi.useFakeTimers();
        const native = nativePort();
        const host = createPluginHost(native.port, { version: 1, viewId: "a" });
        const failed = expect(host.invoke("write_curve")).rejects.toThrow("host model changed");
        native.deliver({ version: 1, viewId: "a", id: 1, ok: false, error: "host model changed" });
        await failed;
        const timed = expect(host.invoke("slow")).rejects.toThrow("timed out");
        await vi.advanceTimersByTimeAsync(30_000);
        await timed;
        const next = host.invoke("ping");
        native.deliver({ version: 1, viewId: "a", id: 3, ok: true, value: 42 });
        await expect(next).resolves.toBe(42);
        host.dispose();
    });

    it("close cancels pending calls and never sends further commands", async () => {
        const native = nativePort();
        const host = createPluginHost(native.port, { version: 1, viewId: "a" });
        const pending = expect(host.invoke("pending")).rejects.toThrow("closed");
        host.dispose();
        host.dispose();
        await pending;
        await expect(host.invoke("after_close")).rejects.toThrow("closed");
        expect(native.sent).toHaveLength(1);
    });

    it("events are view-local and unlisten removes the handler", async () => {
        const native = nativePort();
        const host = createPluginHost(native.port, { version: 1, viewId: "a" });
        const received: unknown[] = [];
        const off = await host.listen("pitch", (event) => received.push(event.payload));
        native.deliver({ version: 1, viewId: "b", event: "pitch", payload: [99] });
        native.deliver({ version: 1, viewId: "a", event: "pitch", payload: [60] });
        off();
        native.deliver({ version: 1, viewId: "a", event: "pitch", payload: [70] });
        expect(received).toEqual([[60]]);
        host.dispose();
    });

    it("bounds unfinished requests and propagates native send errors", async () => {
        const native = nativePort();
        const host = createPluginHost(native.port, { version: 1, viewId: "a" });
        const pending = Array.from({ length: 128 }, () => host.invoke("slow").catch(String));
        await expect(host.invoke("overflow")).rejects.toThrow("Too many");
        expect(native.sent).toHaveLength(128);
        host.dispose();
        await Promise.all(pending);
        native.port.postMessage = () => {
            throw new Error("native gone");
        };
        const broken = createPluginHost(native.port, { version: 1, viewId: "b" });
        await expect(broken.invoke("ping")).rejects.toThrow("native gone");
        broken.dispose();
    });
});
