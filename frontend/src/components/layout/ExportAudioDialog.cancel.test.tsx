// @vitest-environment jsdom
/*
 * 导出对话框的「取消」契约测试。
 *
 * 【为什么必须有】界面里有两个取消入口、文案完全相同（都是「取消」），但用户预期不同：
 * 进度条旁边的取消只该结束**本次导出**，页脚的取消/关闭才是离开对话框。两者曾经共用
 * 一个 handler，于是点进度条的取消会把整个对话框一起关掉（用户报告的"功能有点怪"）。
 * 文案相同意味着这条契约**无法靠人眼在回归时稳定复验**，必须由测试钉住。
 *
 * 【怎么把对话框推进"正在导出"】jsdom 里没有 Tauri 事件，所以 mock
 * `@tauri-apps/api/event` 的 `listen` 抓住回调，再用一个合成的进度事件把
 * `exportProgress.active` 置真 —— 进度区（含它自己的取消按钮）因此出现。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import sessionReducer from "../../features/session/sessionSlice";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { ExportAudioDialog } from "./ExportAudioDialog";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/*
 * jsdom 没有 `ResizeObserver`，而 `AppDialog` 的滚动区（Radix ScrollArea）在布局
 * effect 里会构造它 —— 缺了这层替身，挂载会在提交阶段直接抛错。
 * 只需要"存在且可调用"，本测试不依赖任何尺寸回调。
 */
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

const mocks = vi.hoisted(() => ({
    /** 导出进度事件的监听回调（由 mock 的 `listen` 填入）。 */
    progressListener: null as null | ((event: { payload?: unknown }) => void),
    /** 取消导出命令的替身。 */
    cancelExportAudio: vi.fn(async () => ({ ok: true })),
}));

vi.mock("@tauri-apps/api/event", () => ({
    listen: vi.fn(async (_name: string, cb: (event: { payload?: unknown }) => void) => {
        mocks.progressListener = cb;
        return () => {
            mocks.progressListener = null;
        };
    }),
}));

vi.mock("../../services/api/core", async () => {
    const actual =
        await vi.importActual<typeof import("../../services/api/core")>("../../services/api/core");
    return {
        ...actual,
        coreApi: {
            ...actual.coreApi,
            // 打开时的一次性默认值拉取：返回 `ok:false` 让对话框保持内置默认（不需要夹具）。
            getExportAudioDefaults: vi.fn(async () => ({ ok: false })),
            cancelExportAudio: mocks.cancelExportAudio,
        },
    };
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    mocks.cancelExportAudio.mockClear();
    mocks.progressListener = null;
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(async () => {
    await act(async () => root.unmount());
    // Radix 通过 portal 把对话框挂到 document.body，不在 host 里。
    document.body.innerHTML = "";
});

/** 挂载对话框；返回 `onOpenChange` 替身与若干查询助手。 */
async function mountDialog() {
    const onOpenChange = vi.fn();
    const store = configureStore({ reducer: { session: sessionReducer } });
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <ExportAudioDialog open onOpenChange={onOpenChange} />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });

    /** 推进到"正在导出"：进度区出现，进度条自带取消按钮随之出现。 */
    const enterExportingState = async () => {
        expect(mocks.progressListener).not.toBeNull();
        await act(async () => {
            mocks.progressListener?.({
                payload: { active: true, mode: "project", progress: 0.4, current: 1, total: 1 },
            });
        });
        expect(document.querySelector("[data-hs-progress-cancel]")).not.toBeNull();
    };

    const click = async (el: Element | null) => {
        expect(el).not.toBeNull();
        await act(async () => {
            (el as HTMLElement).dispatchEvent(new MouseEvent("click", { bubbles: true }));
        });
    };

    /**
     * 页脚「取消」按钮：文案与进度条取消相同（同一条 i18n 键的语义族），因此
     * **不能靠文案区分** —— 这里显式排除带 `data-hs-progress-cancel` 的那个。
     * 期望文案从进度条取消按钮现读，测试因此不依赖当前语言。
     */
    const footerCancelButton = () => {
        const progressCancel = document.querySelector("[data-hs-progress-cancel]");
        const label = (progressCancel?.textContent ?? "").trim();
        const buttons = [...document.body.querySelectorAll("button")];
        const hit = buttons.find(
            (b) =>
                !b.hasAttribute("data-hs-progress-cancel") &&
                label !== "" &&
                (b.textContent ?? "").trim() === label,
        );
        if (hit === undefined) {
            throw new Error(
                `找不到页脚「${label}」按钮；页面按钮文案：${JSON.stringify(
                    buttons.map((b) => (b.textContent ?? "").trim()),
                )}`,
            );
        }
        return hit;
    };

    return { onOpenChange, enterExportingState, click, footerCancelButton };
}

test("进度条上的取消只取消导出，不关闭对话框", async () => {
    const { onOpenChange, enterExportingState, click } = await mountDialog();
    await enterExportingState();

    await click(document.querySelector("[data-hs-progress-cancel]"));

    expect(mocks.cancelExportAudio).toHaveBeenCalledTimes(1);
    expect(onOpenChange).not.toHaveBeenCalled();
    // 取消后进度区收起：对话框回到"可再次导出"的状态。
    expect(document.querySelector("[data-hs-progress-cancel]")).toBeNull();
});

test("页脚的取消会先取消导出再关闭对话框", async () => {
    const { onOpenChange, enterExportingState, click, footerCancelButton } = await mountDialog();
    await enterExportingState();

    await click(footerCancelButton());

    expect(mocks.cancelExportAudio).toHaveBeenCalledTimes(1);
    expect(onOpenChange).toHaveBeenCalledWith(false);
});
