/**
 * Transfer 设置与诊断面板的渲染契约。
 *
 * 【为什么值得一条冒烟测试】这个面板整块重写过（从"录制面板"变成"设置与诊断"），
 * 而它只在**非 ARA 宿主**里出现 —— 那是用户报障最多的形态，也是自动化最难覆盖的形态。
 * 一条渲染测试能挡住"打开面板就白屏"这类事故。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { resetTransferStoreForTests } from "./transferStore";
import { TransferPanel } from "./TransferPanel";
import type { TransferStatus } from "./transferApi";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;

function status(overrides: Partial<TransferStatus> = {}): TransferStatus {
    return {
        state: "ready",
        recording: null,
        capture: null,
        material: {
            clips: 2,
            capturedFrames: 96_000,
            coveredFrames: 96_000,
            spannedFrames: 96_000,
            sampleRate: 48_000,
            silent: false,
        },
        hasMaterial: true,
        span: { startFrame: 0, endFrame: 96_000, sampleRate: 48_000 },
        takes: [
            { id: 1, active: false, clips: 2, startSec: 0, endSec: 2 },
            { id: 2, active: true, clips: 1, startSec: 0.5, endSec: 1 },
        ],
        options: { maxDurationSec: 600, segmentSec: 8, monitor: true, outsidePassthrough: true },
        takeCount: 2,
        activeTake: 2,
        sawContext: true,
        sawInput: true,
        sawOffline: false,
        canCapture: true,
        hostPlaying: true,
        araBound: false,
        ...overrides,
    };
}

beforeEach(() => {
    invokeMock.mockReset();
    resetTransferStoreForTests();
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <TransferPanel />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    await act(async () => {
        await Promise.resolve();
    });
}

function text(): string {
    return container.textContent ?? "";
}

test("列出全部 take，并允许删除其中一个", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    expect(text()).toContain(enUS.transfer_field_takes);
    expect(text()).toContain("2");
    const deletes = Array.from(container.querySelectorAll("button")).filter((b) =>
        b.textContent?.includes(enUS.transfer_take_delete),
    );
    expect(deletes.length, "每个 take 都要能删").toBe(2);
    await act(async () => deletes[1].click());
    await act(async () => {
        await Promise.resolve();
    });
    expect(invokeMock).toHaveBeenCalledWith("transfer_delete_take", 2);
});

test("两个行为选项都可切换，且把补丁发出去", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    expect(text()).toContain(enUS.transfer_option_monitor);
    expect(text()).toContain(enUS.transfer_option_outside);
    const switches = Array.from(container.querySelectorAll("button[role='switch']"));
    expect(switches.length).toBe(2);
    await act(async () => (switches[0] as HTMLButtonElement).click());
    await act(async () => {
        await Promise.resolve();
    });
    expect(invokeMock).toHaveBeenCalledWith("transfer_set_options", { monitor: false });
});

test("录到了但没有内容时说清是输入静音", async () => {
    invokeMock.mockResolvedValue(
        status({
            material: {
                clips: 0,
                capturedFrames: 96_000,
                coveredFrames: 0,
                spannedFrames: 96_000,
                sampleRate: 48_000,
                silent: true,
            },
            takes: [{ id: 1, active: true, clips: 0, startSec: null, endSec: null }],
            takeCount: 1,
        }),
    );
    await render();
    // "录到了 2.0 秒，但输入是静音…"
    expect(text()).toContain(enUS.transfer_recorded_silent.replace("{seconds}", "2.0"));
});

test("ARA 宿主里说不需要 Transfer，而不是显示一个空面板", async () => {
    invokeMock.mockResolvedValue(status({ araBound: true, takes: [], takeCount: 0 }));
    await render();
    expect(text()).toContain(enUS.transfer_not_applicable_ara);
});
