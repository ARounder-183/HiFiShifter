/**
 * 传输栏上 Transfer 控件的契约。
 *
 * 【要钉死什么】用户实测报障原话是"没看到任何 Transfer 相关的功能入口"。控件因此必须
 * 在插件形态下**永远**给出一个可点的键，并且**说清**现在缺哪一步（宿主没播放、宿主不给
 * 位置、宿主正在用 ARA 供音）。store 自己的时序由 `transferStore.test.ts` 覆盖，本文件
 * 只测"控件上显示什么、点了会发哪条命令"。
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
import { TransferControl } from "./TransferControl";
import type { TransferStatus } from "./transferApi";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;

function status(overrides: Partial<TransferStatus> = {}): TransferStatus {
    return {
        state: "empty",
        recording: null,
        capture: null,
        material: null,
        hasMaterial: false,
        span: null,
        takes: [],
        options: { maxDurationSec: 600, segmentSec: 8, monitor: true, outsidePassthrough: true },
        takeCount: 0,
        activeTake: null,
        sawContext: true,
        sawInput: true,
        sawOffline: false,
        canCapture: true,
        hostPlaying: false,
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
                    <TransferControl />
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

function button(label: string): HTMLButtonElement | undefined {
    return Array.from(container.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(label),
    );
}

async function click(target: HTMLButtonElement | undefined) {
    expect(target, "button must exist").toBeTruthy();
    await act(async () => target!.click());
    await act(async () => {
        await Promise.resolve();
    });
}

test("宿主可以录时给出可点的录制键", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    const record = button(enUS.transfer_start);
    expect(record, "用户必须能看到一个录制入口").toBeTruthy();
    expect(record!.disabled).toBe(false);
    await click(record);
    expect(invokeMock).toHaveBeenCalledWith("transfer_start");
});

test("宿主不给播放位置时键在但**说清为什么不能录**", async () => {
    invokeMock.mockResolvedValue(status({ canCapture: false }));
    await render();
    const record = button(enUS.transfer_start);
    expect(record).toBeTruthy();
    expect(record!.disabled, "不能录时不能让它点下去什么都不发生").toBe(true);
    expect(record!.getAttribute("data-tooltip")).toBe(enUS.transfer_cannot_capture);
});

test("ARA 宿主里如实说明不需要 Transfer", async () => {
    invokeMock.mockResolvedValue(status({ araBound: true }));
    await render();
    expect(button(enUS.transfer_start)!.getAttribute("data-tooltip")).toBe(
        enUS.transfer_not_applicable_ara,
    );
});

test("录制中给出停止与取消，并说出「宿主没在播放」", async () => {
    invokeMock.mockResolvedValue(
        status({
            state: "recording",
            hostPlaying: false,
            recording: {
                capturedFrames: 48_000,
                sealedSegments: 0,
                position: 48_000,
                peak: [0.1, 0.1],
                gaps: 0,
                spares: 2,
            },
            capture: { startFrame: 0, endFrame: 48_000, positionFrame: 48_000, sampleRate: 48_000 },
        }),
    );
    await render();
    expect(button(enUS.transfer_stop)).toBeTruthy();
    expect(button(enUS.transfer_cancel), "误触之后要有退路").toBeTruthy();
    // 录了 1 秒 + 宿主没在播放这条提示。
    expect(text()).toContain(enUS.transfer_field_waiting_host);
    await click(button(enUS.transfer_stop));
    expect(invokeMock).toHaveBeenCalledWith("transfer_stop");
});
