// @vitest-environment jsdom
/**
 * ARA 面板的提示通道归属。
 *
 * 【这个文件钉住一件事】`features/ara/` 下的悬停提示**只走本项目自定义通道**
 * （`data-tooltip`）。此前三个面板一律用原生 `title`，其中自动应用状态片还把
 * 两行说明 `join("\n")` 塞给 `title` —— 原生 tooltip 会把两行折成一行，
 * 且样式/明暗主题都不受本项目控制（`components/dock/dockTooltips.ts:58-61`
 * 对"不要改用原生 title"有同样的结论）。
 *
 * 【为什么逐面板各测一条，而不是只测原语】原语测试（`ui/StatusChip`）只能保证
 * "传了 `tooltip` 就有 `data-tooltip`"，不能保证调用方**传的是它**。这里守住
 * 调用方：任何一处把提示写回 `title` 都会让对应用例变红。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const invokeMock = vi.fn();

vi.mock("../../services/invoke", () => ({
    invoke: (...args: unknown[]) => invokeMock(...args),
    invokeErrorKind: () => "transient",
    invokeErrorCode: () => null,
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
                    <PluginApplyStatus onTimelineChanged={async () => undefined} />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
    });
}

/**
 * 断言"这个容器里没有原生 title"。
 *
 * 【为什么允许 [] 白名单】对话框/通知的标题槽是 `title` prop，它们经原语渲染成
 * `<h2>` 而不是 DOM 的 `title` 属性，因此正常情况下一个都不该出现。白名单是
 * 给将来合理的例外留的口子，目前为空 —— 出现 `[title]` 就是回归。
 */
function expectNoNativeTitle(context: string) {
    const offenders = Array.from(container.querySelectorAll("[title]")).map(
        (element) => `${element.tagName.toLowerCase()}[title="${element.getAttribute("title")}"]`,
    );
    expect(offenders, `${context} 不得使用原生 title`).toEqual([]);
}

test("the applied chip puts its two-line note on the custom channel", async () => {
    invokeMock.mockResolvedValue(hostState());
    await render();
    const chip = container.querySelector("[data-tooltip]");
    expect(chip, "应用状态片必须经 data-tooltip 提供提示").toBeTruthy();
    const tooltip = chip?.getAttribute("data-tooltip") ?? "";
    // 两行：修订读数 + 操作说明。气泡是 pre-wrap，换行会如实渲染。
    expect(tooltip).toContain(
        enUS.plugin_apply_generations.replace("{edits}", "4").replace("{audio}", "4"),
    );
    expect(tooltip).toContain(enUS.plugin_apply_hint);
    expect(tooltip).toContain("\n");
    expectNoNativeTitle("应用状态片");
});

test("a host that never became ready still explains itself on the custom channel", async () => {
    invokeMock.mockResolvedValue(hostState({ ready: false }));
    await render();
    expect(container.textContent).toContain(enUS.plugin_apply_waiting_host);
    expectNoNativeTitle("等待宿主的状态片");
});

test("an error line carries no duplicate native title", async () => {
    invokeMock.mockResolvedValue(hostState({ ready: false, error: "host refused" }));
    await render();
    // 错误行本身已经是完整的"尚未应用：原因"，再挂任何提示都是重复。
    expect(container.querySelector('[role="alert"]')?.textContent).toContain("host refused");
    expectNoNativeTitle("错误行");
});
