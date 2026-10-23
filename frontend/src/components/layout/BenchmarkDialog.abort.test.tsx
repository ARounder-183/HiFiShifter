// @vitest-environment jsdom
/*
 * 基准测试对话框的「弃用旧运行」契约测试。
 *
 * 【为什么必须有】后端基准测试一次要跑 36–134 秒，而关闭对话框与旧 promise 的落地
 * 之间是纯粹的异步赛跑：曾经关闭只置一个共享布尔、重开又把它复位，于是"中途关掉、
 * 过一会儿重开"的用户会在新会话里看到上一轮的结果（或错误）—— 从未点过 Run 的一次
 * 运行"复活"了。代际计数修复后，本测试把契约钉住：旧 promise 无论 resolve 还是
 * reject，都不得触碰重开后的新会话。
 *
 * 【怎么把 promise 握在手里】mock `coreApi.runVocoderBenchmark` 返回一个 deferred：
 * 测试先让对话框进入 running，走完"关闭 → 重开"，再手动 resolve/reject 旧 promise。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { enUS } from "../../i18n/en-US";
import { zhCN } from "../../i18n/zh-CN";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import type { BenchmarkResult } from "../../types/api";
import sessionReducer from "../../features/session/sessionSlice";
import { BenchmarkDialog } from "./BenchmarkDialog";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * 与 `I18nProvider` 的默认解析一致（jsdom 无存储值，走 `navigator.language`）：
 * 文案从**实际生效**的目录现读，测试不硬编码任何语言。
 */
const catalog = navigator.language.toLowerCase().startsWith("zh") ? zhCN : enUS;

/*
 * jsdom 没有 `ResizeObserver`，而 `AppDialog` 的滚动区（Radix ScrollArea）在布局
 * effect 里会构造它 —— 缺了这层替身，挂载会在提交阶段直接抛错。
 */
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const mocks = vi.hoisted(() => ({
    /** 基准测试命令的替身：返回值由测试用 deferred 手动控制。 */
    runVocoderBenchmark: vi.fn(),
}));

vi.mock("../../services/api/core", async () => {
    const actual =
        await vi.importActual<typeof import("../../services/api/core")>("../../services/api/core");
    return {
        ...actual,
        coreApi: {
            ...actual.coreApi,
            runVocoderBenchmark: mocks.runVocoderBenchmark,
        },
    };
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    mocks.runVocoderBenchmark.mockReset();
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(async () => {
    await act(async () => root.unmount());
    // Radix 通过 portal 把对话框挂到 document.body，不在 host 里。
    document.body.innerHTML = "";
});

/** 与 `BenchmarkResult` 的必填字段对齐的最小夹具（本测试不依赖任何数值）。 */
function fixtureResult(): BenchmarkResult {
    return {
        cpuMedianMs: 500,
        cpuRtFactor: 2,
        benchmarkSamples: 5,
        gpuAvailable: false,
        dmlAvailable: false,
        gpuDeviceId: 0,
        availableProviders: ["CPUExecutionProvider"],
        ortBuildInfo: "test",
        gpuDevices: [],
        dmlAdapters: [],
    };
}

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
    });
    return { promise, resolve, reject };
}

/**
 * 挂载对话框；返回重渲染（模拟父组件的 open 切换）与查询助手。
 * 按钮文案从 zh-CN 目录现读 —— 默认语言即 zh-CN，测试因此不硬编码文案。
 */
async function mountDialog() {
    const onOpenChange = vi.fn();
    const store = configureStore({ reducer: { session: sessionReducer } });
    const render = (open: boolean) => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <BenchmarkDialog open={open} onOpenChange={onOpenChange} />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    };
    await act(async () => render(true));

    const buttonByLabel = (label: string): HTMLElement => {
        const hit = [...document.body.querySelectorAll("button")].find(
            (b) => (b.textContent ?? "").trim() === label,
        );
        if (hit === undefined) {
            throw new Error(
                `找不到「${label}」按钮；页面按钮文案：${JSON.stringify(
                    [...document.body.querySelectorAll("button")].map((b) =>
                        (b.textContent ?? "").trim(),
                    ),
                )}`,
            );
        }
        return hit;
    };

    const click = async (el: HTMLElement) => {
        await act(async () => {
            el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
        });
    };

    /**
     * 等到 Run 的异步前置（先 `await dispatch(stopAudioPlayback(...))`，而 invoke 在
     * jsdom 里要走完 1.5s 的 pywebview 探测才失败落定）结束、基准测试命令真正发出。
     */
    const waitForBenchmarkCall = async () => {
        await act(async () => {
            for (let i = 0; i < 3000 && mocks.runVocoderBenchmark.mock.calls.length === 0; i++) {
                await new Promise((resolve) => setTimeout(resolve, 2));
            }
        });
        expect(mocks.runVocoderBenchmark).toHaveBeenCalledTimes(1);
    };

    return {
        onOpenChange,
        render,
        click,
        waitForBenchmarkCall,
        runButton: () => buttonByLabel(catalog.benchmark_run_btn),
        closeButton: () => buttonByLabel(catalog.benchmark_close),
    };
}

test("关闭后重开，旧运行的 resolve 不得写进新会话", async () => {
    const pending = deferred<BenchmarkResult>();
    mocks.runVocoderBenchmark.mockReturnValue(pending.promise);
    const dialog = await mountDialog();

    await dialog.click(dialog.runButton());
    await dialog.waitForBenchmarkCall();

    // 关闭 → 父组件置 open=false → 再打开：新会话，从未点过 Run。
    await dialog.click(dialog.closeButton());
    expect(dialog.onOpenChange).toHaveBeenCalledWith(false);
    await act(async () => dialog.render(false));
    await act(async () => dialog.render(true));

    pending.resolve(fixtureResult());
    // 让被弃用 promise 的续体在 act 内冲刷完。
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
    });

    // 新会话必须仍是 idle：没有结果表，idle 提示可见。
    expect(document.querySelector("table")).toBeNull();
    expect(document.body.textContent).toContain(catalog.benchmark_idle_hint);
});

test("关闭后重开，旧运行的 reject 不得写进新会话", async () => {
    const pending = deferred<BenchmarkResult>();
    mocks.runVocoderBenchmark.mockReturnValue(pending.promise);
    const dialog = await mountDialog();

    await dialog.click(dialog.runButton());
    await dialog.waitForBenchmarkCall();
    await dialog.click(dialog.closeButton());
    await act(async () => dialog.render(false));
    await act(async () => dialog.render(true));

    pending.reject(new Error("stale failure"));
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 0));
    });

    // 新会话必须仍是 idle：错误横幅（默认文案 / 自定义文案）不得出现。
    expect(document.body.textContent).not.toContain("stale failure");
    expect(document.body.textContent).toContain(catalog.benchmark_idle_hint);
});
