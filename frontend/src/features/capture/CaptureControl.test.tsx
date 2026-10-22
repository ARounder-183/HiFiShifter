/**
 * 传输栏上 Capture 控件的契约。
 *
 * 【要钉死什么】
 * 1. 插件形态下**永远**有一个可点的键，并且说清现在缺哪一步（宿主没播放、宿主不给位置、
 *    宿主正在用 ARA 供音）—— 用户实测报障原话是"没看到任何 Capture 相关的功能入口"。
 * 2. **只有一行，而且只有图标**。主工具栏是 `flex-nowrap overflow-y-hidden` 的单行容器，
 *    而插件形态下它的宽度由宿主的 VST 窗口决定。此前录制中会多出一行读数，那一行被
 *    工具栏裁掉，表现为"点了录制排版就乱"（用户报障）。所以这里既断言"没有第二行"，
 *    也断言"键没有文字"。
 * 3. 无文字不等于无名称：每个图标键都必须带 `aria-label`（它同时是 tooltip）。
 *
 * store 自己的时序由 `captureStore.test.ts` 覆盖，本文件只测"控件上显示什么、点了会发
 * 哪条命令"。
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
import { resetCaptureStoreForTests } from "./captureStore";
import { CaptureControl } from "./CaptureControl";
import type { CaptureStatus } from "./captureApi";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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

/** 一段"正在录"的读数（多数用例都要它）。 */
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
    return container.textContent ?? "";
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

test("宿主可以录时给出可点的录制键", async () => {
    invokeMock.mockResolvedValue(status());
    await render();
    const record = iconButton(enUS.capture_guide_body);
    expect(record, "用户必须能看到一个录制入口").toBeTruthy();
    expect(record!.disabled).toBe(false);
    await click(record);
    expect(invokeMock).toHaveBeenCalledWith("capture_start");
});

/**
 * **控件只有一行，且键只有图标。**
 *
 * 【为什么这是回归测试】此前录制中会多出一行读数（`data-capture-details`），而主工具栏
 * 是单行且纵向裁切的容器 —— 那一行被裁掉，用户看到的是"点了录制排版就乱"。而工具栏在
 * 宿主 VST 窗口里可能很窄，文字键（"录制"/"停止"/"取消"）在任何语言里都比图标宽。
 */
test("控件只有一行，键只有图标且都有名称", async () => {
    invokeMock.mockResolvedValue(recording());
    await render();
    // 第二行必须**完全**不存在（不是隐藏、不是条件渲染失败）。
    expect(
        container.querySelector("[data-capture-details]"),
        "第二行必须被彻底去掉，否则它会撑破单行工具栏",
    ).toBeNull();
    // 根节点是一个横向容器，且它的直接子元素都在同一行里（没有换行的块级行）。
    expect(
        control().style.flexDirection,
        "控件根节点不得是纵向布局 —— 那正是第二行的来源",
    ).not.toBe("column");
    expect(control().children.length).toBeGreaterThan(0);
    for (const child of Array.from(control().children)) {
        expect(
            (child as HTMLElement).style.flexDirection,
            "控件内部不得出现纵向布局（那是第二行的来源）",
        ).not.toBe("column");
    }
    // 键没有文字，但都有可访问名称。
    const keys = buttons();
    expect(keys.length).toBeGreaterThan(0);
    for (const key of keys) {
        expect(key.textContent?.trim(), "键必须是图标，不能带文字").toBe("");
        expect(key.getAttribute("aria-label"), "图标键必须有名称").toBeTruthy();
    }
});

test("宿主不给播放位置时键在但**说清为什么不能录**", async () => {
    invokeMock.mockResolvedValue(status({ canCapture: false }));
    await render();
    const record = iconButton(enUS.capture_cannot_capture);
    expect(record).toBeTruthy();
    expect(record!.disabled, "不能录时不能让它点下去什么都不发生").toBe(true);
});

test("ARA 宿主里如实说明不需要 Capture", async () => {
    invokeMock.mockResolvedValue(status({ araBound: true }));
    await render();
    expect(iconButton(enUS.capture_not_applicable_ara)).toBeTruthy();
});

test("宿主在播放却一帧都没到 ⇒ 直接说是音频没进来", async () => {
    invokeMock.mockResolvedValue(
        recording({
            recording: {
                capturedFrames: 0,
                sealedSegments: 0,
                position: 0,
                peak: [0, 0],
                gaps: 0,
                spares: 2,
            },
            capture: null,
        }),
    );
    await render();
    expect(iconButton(enUS.capture_no_audio_arriving)).toBeTruthy();
    expect(
        iconButton(enUS.capture_host_not_playing),
        "不能把「没在播放」与「没拿到音频」混成一句",
    ).toBeUndefined();
});

test("宿主用 64 位精度时说明原因与补救（这是用户可修的一档）", async () => {
    invokeMock.mockResolvedValue(status({ unsupportedPrecision: true }));
    await render();
    expect(iconButton(enUS.capture_unsupported_precision)).toBeTruthy();
});

test("录制中给出停止与取消，并说出「宿主没在播放」", async () => {
    invokeMock.mockResolvedValue(recording({ hostPlaying: false }));
    await render();
    const stop = iconButton(enUS.capture_host_not_playing);
    expect(stop, "宿主没在播放时必须说清原因").toBeTruthy();
    expect(iconButton(enUS.capture_cancel_hint), "误触之后要有退路").toBeTruthy();
    await click(stop);
    expect(invokeMock).toHaveBeenCalledWith("capture_stop");
});

/**
 * **待命必须说成"等待播放"，而不是"已录 0.0 秒"。**
 *
 * 【为什么这条是用户反馈的回归测试】用户按了录制、宿主还没播，界面显示的是一个
 * "已录 0.0 秒"的读数 —— 那看起来像失败，而实际是"等你去按播放"。这两件事对用户的
 * 下一步完全不同，所以待命必须是一个**独立**的说法，而且**不该有录制读数**。
 */
test("按了录制但宿主还没播放 ⇒ 说成等待播放，而不是 0.0 秒", async () => {
    invokeMock.mockResolvedValue(status({ state: "standby", standby: true }));
    await render();
    expect(iconButton(enUS.capture_field_waiting_host)).toBeTruthy();
    expect(text(), "待命期不该出现录制读数").not.toContain("0.0");
    // 键变成"停止"（本次录制已经开始了，只是还没收到素材）。
    await click(iconButton(enUS.capture_field_waiting_host));
    expect(invokeMock).toHaveBeenCalledWith("capture_stop");
});

/** 距上限不足 10 秒时给出预警 —— 上限到点会自己停。预警折进键与读数的 tooltip。 */
test("接近录制上限时给出预警", async () => {
    invokeMock.mockResolvedValue(recording({ limitWarning: true }));
    await render();
    expect(iconButton(enUS.capture_limit_warning)).toBeTruthy();
});

/** 录制中必须有一个"在动"的读数：没有它，用户无法判断插件到底有没有在收音频。 */
test("录制中显示已录时长", async () => {
    invokeMock.mockResolvedValue(recording());
    await render();
    expect(text()).toContain("1.0");
});

/** 后端错误只给图标，正文进 tooltip（错误可能是一整句话，放进工具栏会撑开整条栏）。 */
test("出错时只给图标，正文在 tooltip 里", async () => {
    invokeMock.mockResolvedValue(status({ canCapture: false }));
    invokeMock.mockImplementation(async (command: string) => {
        if (command === "capture_status") return status({ canCapture: true });
        throw new Error("the host has not declared a sample rate yet");
    });
    await render();
    await click(iconButton(enUS.capture_guide_body));
    const alert = container.querySelector<HTMLElement>("[role='alert']");
    expect(alert, "错误必须有一个可见的落点").toBeTruthy();
    expect(alert!.getAttribute("data-tooltip")).toContain("sample rate");
    expect(text(), "错误正文不该铺在工具栏上").not.toContain("sample rate");
});
