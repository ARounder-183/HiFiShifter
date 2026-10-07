/**
 * 状态栏「插件宿主自动应用」状态片的契约。
 *
 * 【要钉死什么】片上的**动作门**：只有宿主就绪时才给「重新载入宿主」按钮，且宿主
 * 报告仍有未应用编辑（`pending`）时**必须先确认**再强制重载 —— 强制重载会丢弃
 * 本地曲线，静默执行是不可接受的。store 自己的时序由
 * `pluginApplyStore.test.ts` 覆盖，本文件只测"片上显示什么、点了会怎样"。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
}));

vi.mock("../../services/hostEvents", () => ({
    listen: async () => () => undefined,
}));

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { resetPluginApplyStoreForTests, type PluginApplyState } from "./pluginApplyStore";
import { PluginApplyStatus } from "./PluginApplyStatus";

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
let refreshes: number;

function hostState(overrides: Partial<PluginApplyState> = {}): PluginApplyState {
    return {
        generation: 4,
        applied_generation: 4,
        pending: false,
        host_version: 1,
        connected: true,
        ready: true,
        error: null,
        ...overrides,
    };
}

beforeEach(() => {
    vi.useFakeTimers();
    invokeMock.mockReset();
    resetPluginApplyStoreForTests();
    refreshes = 0;
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
    vi.useRealTimers();
    vi.restoreAllMocks();
});

async function render() {
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <PluginApplyStatus
                        onTimelineChanged={async () => {
                            refreshes++;
                        }}
                    />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    // 第一轮轮询的微任务链。
    await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
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
        await vi.advanceTimersByTimeAsync(0);
    });
}

test("waiting for the host shows a status word and no reload action", async () => {
    invokeMock.mockResolvedValue(hostState({ ready: false }));
    await render();
    expect(text()).toContain(enUS.plugin_apply_waiting_host);
    expect(button(enUS.plugin_apply_reload_host)).toBeUndefined();
});

test("a ready host offers the reload action and reports the applied state", async () => {
    invokeMock.mockResolvedValue(hostState());
    await render();
    expect(text()).toContain(enUS.plugin_apply_applied);
    expect(button(enUS.plugin_apply_reload_host)).toBeTruthy();
    // 修订数与长说明退到 title，不占状态栏宽度。
    const chip = container.querySelector("[title]");
    expect(chip?.getAttribute("title")).toContain(enUS.plugin_apply_hint);
});

test("reloading with unapplied edits requires an explicit confirmation", async () => {
    invokeMock.mockResolvedValue(hostState({ pending: true }));
    await render();
    expect(text()).toContain(enUS.plugin_apply_pending);
    // 首次轮询也会重取一次时间轴（宿主版本号第一次见到）；下面只看增量。
    const before = refreshes;

    await click(button(enUS.plugin_apply_reload_host));
    // 还没确认：不允许发出强制重载。
    expect(invokeMock).not.toHaveBeenCalledWith("plugin_refresh", true);
    expect(document.body.querySelector('[role="dialog"]')).toBeTruthy();

    const confirm = Array.from(document.body.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(enUS.plugin_apply_reload_confirm),
    );
    await act(async () => confirm!.click());
    await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
    });
    expect(invokeMock).toHaveBeenCalledWith("plugin_refresh", true);
    // 强制重载之后必须重取时间轴：宿主刚换掉了音频，本地时间轴已经过期。
    expect(refreshes).toBe(before + 1);
});

test("reloading a clean host goes straight through without a dialog", async () => {
    invokeMock.mockResolvedValue(hostState());
    await render();
    await click(button(enUS.plugin_apply_reload_host));
    expect(document.body.querySelector('[role="dialog"]')).toBeNull();
    expect(invokeMock).toHaveBeenCalledWith("plugin_refresh", false);
});
