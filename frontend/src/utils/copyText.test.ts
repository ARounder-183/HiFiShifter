/**
 * `copyTextToClipboard` 的宿主分流契约。
 *
 * 回归背景：菜单里的复制动作经过动态 import + 若干 IPC 后，WebView 的
 * transient user activation 已过期，`navigator.clipboard.writeText` 可能拒绝；
 * 因此复制路径 / 复制链接优先走不受手势限制的原生通道。
 */
// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";

const tauriInvoke = vi.fn();
const pluginInvoke = vi.fn();
const frontendLog = vi.fn();
let pluginHost: { invoke: (...args: unknown[]) => unknown } | null = null;

vi.mock("../services/invoke", async () => {
    const actual = await vi.importActual<typeof import("../services/invoke")>("../services/invoke");
    return { ...actual };
});
vi.mock("../services/pluginHost", () => ({
    getPluginHost: () => pluginHost,
}));
vi.mock("../services/frontendLog", () => ({
    frontendLog: (...args: unknown[]) => frontendLog(...args),
}));

import { copyTextToClipboard } from "./copyText";

let writeText: ReturnType<typeof vi.fn>;
let execCommand: ReturnType<typeof vi.fn>;

beforeEach(() => {
    tauriInvoke.mockReset();
    pluginInvoke.mockReset();
    frontendLog.mockReset();
    pluginHost = { invoke: (...args: unknown[]) => pluginInvoke(...args) };
    writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    execCommand = vi.fn().mockReturnValue(true);
    Object.assign(document, { execCommand });
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    delete window.__TAURI__;
});

test("a standalone native success never falls back to the WebView clipboard", async () => {
    pluginHost = null;
    window.__TAURI__ = { core: { invoke: tauriInvoke } };
    tauriInvoke.mockResolvedValue({ ok: true });
    await expect(copyTextToClipboard("HiFiShifter 0.2.0")).resolves.toBe(true);
    expect(tauriInvoke).toHaveBeenCalledWith("write_clipboard_text", {
        text: "HiFiShifter 0.2.0",
    });
    expect(writeText).not.toHaveBeenCalled();
});

test("a plugin native success uses the WebView2 message port", async () => {
    pluginInvoke.mockResolvedValue({ ok: true });
    await expect(copyTextToClipboard("HiFiShifter 0.2.0")).resolves.toBe(true);
    expect(pluginInvoke).toHaveBeenCalledWith("write_clipboard_text", {
        text: "HiFiShifter 0.2.0",
    });
    expect(writeText).not.toHaveBeenCalled();
});

test("a structured native failure still tries the browser fallback once", async () => {
    pluginInvoke.mockResolvedValue({ ok: false, error: "clipboard_open_failed" });
    await expect(copyTextToClipboard("summary")).resolves.toBe(true);
    expect(writeText).toHaveBeenCalledWith("summary");
});

test("browser preview skips IPC and writes through the WebView clipboard", async () => {
    pluginHost = null;
    await expect(copyTextToClipboard("path")).resolves.toBe(true);
    expect(pluginInvoke).not.toHaveBeenCalled();
    expect(tauriInvoke).not.toHaveBeenCalled();
    expect(writeText).toHaveBeenCalledWith("path");
});

test("all channels failing returns false without throwing", async () => {
    pluginHost = null;
    window.__TAURI__ = { core: { invoke: tauriInvoke } };
    tauriInvoke.mockResolvedValue({ ok: false });
    writeText.mockRejectedValue(new Error("permission denied"));
    execCommand.mockReturnValue(false);
    await expect(copyTextToClipboard("text")).resolves.toBe(false);
    expect(frontendLog).not.toHaveBeenCalled();
});

test("an empty payload never reaches any channel", async () => {
    await expect(copyTextToClipboard("")).resolves.toBe(false);
    expect(pluginInvoke).not.toHaveBeenCalled();
    expect(tauriInvoke).not.toHaveBeenCalled();
    expect(writeText).not.toHaveBeenCalled();
});
