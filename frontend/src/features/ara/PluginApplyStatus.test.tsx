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
    // `pluginApplyStore` 在失败路径上会调用这两个分类辅助函数。不提供它们的话，
    // 任何一次"失败路径"用例都会撞上 `undefined is not a function` —— 而不是
    // 被测的真实行为。
    invokeErrorKind: (error: unknown) =>
        error && typeof error === "object" && (error as { kind?: unknown }).kind === "terminal"
            ? "terminal"
            : "transient",
    invokeErrorCode: (error: unknown) =>
        error && typeof error === "object" && typeof (error as { code?: unknown }).code === "number"
            ? (error as { code: number }).code
            : null,
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
    // 修订数与长说明退到悬停提示，不占状态栏宽度。
    //
    // 【为什么断言 `data-tooltip` 而不是 `title`】提示必须走本项目自定义气泡
    // （`data-tooltip`）：它才能按 `pre-wrap` 渲染两行、并跟随明暗主题；
    // 原生 `title` 会把两行折成一行。此前这条测的是 `[title]`，等于把
    // "提示走原生通道"这个分歧固化下来。
    const chip = container.querySelector("[data-tooltip]");
    expect(chip).not.toBeNull();
    expect(chip?.getAttribute("data-tooltip")).toContain(enUS.plugin_apply_hint);
    // 同一元素上不得再挂原生 title —— 两个气泡会先后冒出来。
    expect(chip?.getAttribute("title")).toBeNull();
});

test("reloading with unapplied edits requires an explicit confirmation", async () => {
    invokeMock.mockResolvedValue(hostState({ pending: true }));
    await render();
    // 【为什么不再断言某句文案】`pending` 现在**不驱动任何文案**：它混合了素材物化
    // 与后台刷新这两件用户感知不到的事。判据改成"有未应用的编辑 ⇒ 强制重载要确认"，
    // 而那件事由下面的对话框断言。
    //
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

/*
 * 【为什么这四条必须存在】`pending / analysis_pending / rendering_active` 三个分量
 * 混在一起时，状态栏在**每一次播放开始**都会说「正在自动应用…」—— 而那一刻后台
 * 跑的只是素材物化，用户什么都没改。四种读数各自对应一个用户能感知的后果，
 * 任何一条读错都会让状态栏再次变成噪音。
 */
// 【为什么拆成四条用例】同一个用例里多次 `render()` 不会重新挂载组件，
// 而 store 的 `publish` 只把快照推给**当前**订阅者 —— 第一条之后界面就停在
// `preparing` 上了。每条用例各自挂载一次，才是"每种读数显示什么"的真判据。
test.each([
    ["preparing", "plugin_apply_waiting_host"],
    ["stale", "plugin_apply_rendering"],
    ["awaiting-analysis", "plugin_apply_analyzing"],
    ["current", "plugin_apply_applied"],
] as const)("audible_state=%s 只说它自己那件事", async (audibleState, expectedKey) => {
    invokeMock.mockResolvedValue(hostState({ audible_state: audibleState }));
    await render();
    expect(text()).toContain(enUS[expectedKey]);
});

test("播放开始时的内务工作不再说「正在应用」", async () => {
    // 后端如实报告"输出就是当前编辑"，同时 `pending` 为真（素材仍在物化）。
    // 状态栏必须**沉默于这件事** —— 它没有用户可感知的后果。
    invokeMock.mockResolvedValue(hostState({ audible_state: "current", pending: true }));
    await render();
    expect(text()).toContain(enUS.plugin_apply_applied);
});
