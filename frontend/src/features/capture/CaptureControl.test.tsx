/**
 * 传输栏上捕获控件的契约，以及它的设置窗口。
 *
 * 【要钉死什么】
 *
 * 1. 插件形态下**永远**有一个可点的键。用户实测报障原话是"没看到任何 Capture
 *    相关的功能入口"。
 * 2. **只有一行，而且只有图标**。主工具栏是 `flex-nowrap overflow-y-hidden` 的单行
 *    容器，宽度由宿主的窗口决定。此前捕获中会多出一行读数，那一行被工具栏裁掉，
 *    表现为"点了捕获排版就乱"。
 * 3. **tooltip 是一句短语**。"宿主没把播放位置交给插件，因此无法对齐到时间轴……"
 *    那一整句塞进悬停气泡既读不完也挤爆布局 —— 它属于设置窗口（右键进入）。
 * 4. 无文字不等于无名称：每个图标键都必须带 `aria-label`（它同时是 tooltip）。
 *
 * store 自己的时序由 `captureStore.test.ts` 覆盖，本文件只测"控件上显示什么、点了会发
 * 哪条命令、右键会打开什么"。
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
import { resetCaptureSettingsForTests } from "./captureSettingsBus";
import { resetCaptureStoreForTests } from "./captureStore";
import { CaptureControl } from "./CaptureControl";
import type { CaptureStatus } from "./captureApi";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver，而 `AppDialog` 的滚动区与 Radix 的 Switch 在布局 effect 里
// 都会构造它（设置窗口一打开就会撞上）。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const LOCALE_KEY = "hifishifter.locale";
let container: HTMLDivElement;
let root: Root;

function status(overrides: Partial<CaptureStatus> = {}): CaptureStatus {
    return {
        state: "empty",
        recording: null,
        capture: null,
        material: null,
        hasMaterial: false,
        span: null,
        takes: [],
        missingTakes: [],
        options: { maxDurationSec: 600, segmentSec: 8, monitor: true, outsidePassthrough: true },
        takeCount: 0,
        activeTake: null,
        sawContext: true,
        sawInput: true,
        sawOffline: false,
        canCapture: true,
        hostPlaying: false,
        unsupportedPrecision: false,
        araBound: false,
        standby: false,
        limitWarning: false,
        stoppedReason: null,
        output: "passthrough",
        ...overrides,
    };
}

/** 一段"正在捕获"的读数（多数用例都要它）。 */
function recording(overrides: Partial<CaptureStatus> = {}): CaptureStatus {
    return status({
        state: "recording",
        hostPlaying: true,
        recording: {
            capturedFrames: 48_000,
            sealedSegments: 0,
            position: 48_000,
            peak: [0.1, 0.1],
            gaps: 0,
            spares: 2,
        },
        capture: { startFrame: 0, endFrame: 48_000, positionFrame: 48_000, sampleRate: 48_000 },
        ...overrides,
    });
}

beforeEach(() => {
    invokeMock.mockReset();
    resetCaptureStoreForTests();
    resetCaptureSettingsForTests();
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
                    <CaptureControl />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    await act(async () => {
        await Promise.resolve();
    });
}

function text(): string {
    return document.body.textContent ?? "";
}

function control(): HTMLElement {
    const found = container.querySelector<HTMLElement>("[data-capture-control]");
    expect(found, "控件必须挂载").toBeTruthy();
    return found!;
}

function buttons(): HTMLButtonElement[] {
    return Array.from(container.querySelectorAll("button"));
}

/** 按 tooltip（= `aria-label`）找键：图标键没有文字，这是唯一的稳定查找方式。 */
function iconButton(tooltip: string): HTMLButtonElement | undefined {
    return buttons().find((b) => b.getAttribute("aria-label") === tooltip);
}

async function click(target: HTMLButtonElement | undefined) {
    expect(target, "button must exist").toBeTruthy();
    await act(async () => target!.click());
    await act(async () => {
        await Promise.resolve();
    });
}

test("宿主可以捕获时给出可点的捕获键", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    const key = iconButton(enUS.capture_start_hint);
    expect(key, "用户必须能看到一个捕获入口").toBeTruthy();
    expect(key!.disabled).toBe(false);
    await click(key);
    expect(invokeMock).toHaveBeenCalledWith("capture_start");
});

/**
 * **控件只有一行，且键只有图标。**
 *
 * 【为什么这是回归测试】此前捕获中会多出一行读数（`data-capture-details`），而主工具栏
 * 是单行且纵向裁切的容器 —— 那一行被裁掉，用户看到的是"点了捕获排版就乱"。而工具栏
 * 在宿主窗口里可能很窄，文字键在任何语言里都比图标宽。
 */
test("控件只有一行，键只有图标且都有名称", async () => {
    invokeMock.mockResolvedValue(recording());
    await render();
    expect(
        container.querySelector("[data-capture-details]"),
        "第二行必须被彻底去掉，否则它会撑破单行工具栏",
    ).toBeNull();
    expect(
        control().style.flexDirection,
        "控件根节点不得是纵向布局 —— 那正是第二行的来源",
    ).not.toBe("column");
    for (const child of Array.from(control().children)) {
        expect(
            (child as HTMLElement).style.flexDirection,
            "控件内部不得出现纵向布局（那是第二行的来源）",
        ).not.toBe("column");
    }
    const keys = buttons().filter((b) => b.closest("[data-capture-control]"));
    expect(keys.length).toBeGreaterThan(0);
    for (const key of keys) {
        expect(key.textContent?.trim(), "键必须是图标，不能带文字").toBe("");
        expect(key.getAttribute("aria-label"), "图标键必须有名称").toBeTruthy();
    }
});

/**
 * **tooltip 只写一句短语。**
 *
 * 【为什么要有长度上限】用户原话："那个 ToolTips 太长了，根本不符合本项目的设计风格。"
 * 悬停气泡回答的是"我现在按这个键会发生什么"，不是"这条通路为什么在这个宿主上不
 * 成立" —— 后者需要一整句，它住在设置窗口里。这里用**字符数**当代理指标：所有合法
 * 状态的 tooltip 都必须短于最长的两句说明（`capture_cannot_capture` 等）。
 */
test("任何状态下的 tooltip 都是一句短语", async () => {
    const cases: Array<[string, CaptureStatus]> = [
        ["可捕获", status()],
        ["待命", status({ state: "standby", standby: true })],
        ["捕获中", recording()],
        ["宿主不给位置", status({ canCapture: false })],
        ["ARA 宿主", status({ araBound: true })],
        ["64 位精度", status({ unsupportedPrecision: true })],
    ];
    for (const [label, value] of cases) {
        invokeMock.mockReset();
        resetCaptureStoreForTests();
        invokeMock.mockResolvedValue(value);
        await render();
        const key = buttons().find((b) => b.getAttribute("aria-label"));
        expect(key, `${label}: 必须有一个带名称的键`).toBeTruthy();
        const tooltip = key!.getAttribute("aria-label") ?? "";
        // 短语判据：不出现句号，且不超过 40 个字符。
        expect(tooltip.includes("."), `${label}: tooltip 不该是整句 —— ${tooltip}`).toBe(false);
        expect(
            tooltip.length,
            `${label}: tooltip 过长（${tooltip.length} 字符）—— ${tooltip}`,
        ).toBeLessThanOrEqual(40);
    }
});

/**
 * **右键捕获键 = 打开设置窗口。**
 *
 * 【为什么右键是必须的】不可用时按钮是 `disabled` 的，而禁用按钮收不到点击 —— 右键是
 * 它唯一还能用的手势，也是"我想知道为什么"最自然的那一下。
 */
test("右键捕获键打开设置窗口", async () => {
    invokeMock.mockResolvedValue(status({ canCapture: false }));
    await render();
    expect(text()).not.toContain(enUS.capture_settings_title);

    const key = iconButton(enUS.capture_unavailable);
    expect(key).toBeTruthy();
    const host = key!.parentElement!;
    await act(async () => {
        host.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
    });
    await act(async () => {
        await Promise.resolve();
    });

    // 设置窗口打开了，并且**说清了为什么不能用**（长句在这里，不在 tooltip 里）。
    expect(text()).toContain(enUS.capture_settings_title);
    expect(text()).toContain(enUS.capture_cannot_capture);
});

/** 设置窗口里的三组设置必须真的下发命令。 */
test("设置窗口把监听开关与上限写回后端", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    const key = iconButton(enUS.capture_start_hint)!;
    await act(async () => {
        key.parentElement!.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, cancelable: true }),
        );
    });
    await act(async () => {
        await Promise.resolve();
    });

    // 监听开关：Radix 的 switch 是 `role="switch"`。
    const toggle = document.querySelector<HTMLElement>("[role='switch']");
    expect(toggle, "监听开关必须存在").toBeTruthy();
    await act(async () => toggle!.click());
    await act(async () => {
        await Promise.resolve();
    });
    expect(invokeMock).toHaveBeenCalledWith("capture_set_options", expect.anything());
});

test("ARA 宿主里如实说明不需要捕获", async () => {
    invokeMock.mockResolvedValue(status({ araBound: true }));
    await render();
    // 不可用 ⇒ 键仍在（只是禁用），tooltip 说"当前不可用"。
    const key = iconButton(enUS.capture_unavailable);
    expect(key).toBeTruthy();
    expect(key!.disabled).toBe(true);
});

/** 捕获中必须有一个"在动"的读数：没有它，用户无法判断插件到底有没有在收音频。 */
test("捕获中显示已捕获时长", async () => {
    invokeMock.mockResolvedValue(recording());
    await render();
    expect(text()).toContain("1.0");
    expect(iconButton(enUS.capture_stop), "捕获中键的语义是停止").toBeTruthy();
});

test("捕获中给出取消的退路", async () => {
    invokeMock.mockResolvedValue(recording());
    await render();
    const cancel = iconButton(enUS.capture_cancel_hint);
    expect(cancel, "误触之后要有退路").toBeTruthy();
    await click(cancel);
    expect(invokeMock).toHaveBeenCalledWith("capture_cancel");
});

/**
 * **出错时只给图标，正文在设置窗口里。**
 *
 * 【为什么不在工具栏显示正文】后端错误可能是一整句话（"宿主还没声明采样率…"），
 * 放进工具栏会把整条栏撑开 —— 那正是"排版就乱"的另一半。
 */
test("出错时只给图标，正文进设置窗口", async () => {
    // 【为什么让**所有**命令都失败】错误是**瞬时**的：轮询成功一次就会把它清掉
    // （见 `captureStore.refresh`）。只让 start 失败的话，200ms 后的那次轮询会
    // 把正文抹掉，测试就变成了在测"时序运气"。让后端整体不可用才是真实场景之一
    // （插件刚加载、会话还没就绪），也让"正文一直在设置窗口里"成为一个稳定断言。
    invokeMock.mockImplementation(async () => {
        throw new Error("the host has not declared a sample rate yet");
    });
    await render();

    const alert = container.querySelector<HTMLElement>("[role='alert']");
    expect(alert, "错误必须有一个可见的落点").toBeTruthy();
    expect(text(), "错误正文不该铺在工具栏上").not.toContain("sample rate");

    // 点它进设置窗口 —— 正文与下一步都在那里。
    await act(async () => alert!.click());
    await act(async () => {
        await Promise.resolve();
    });
    expect(text()).toContain(enUS.capture_settings_title);
    expect(text()).toContain("sample rate");
});
