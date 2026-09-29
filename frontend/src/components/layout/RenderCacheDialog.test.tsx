// @vitest-environment jsdom
/*
 * 渲染缓存管理窗口：数字输入框的滚轮契约。
 *
 * 【为什么必须有】「占用上限」「未使用超期清理」旁边各有一个**预设下拉**，而滚轮
 * 会改下拉的选中项 —— 两个预设列表的**最后一项都是 `0`**（`SIZE_PRESETS_MB` 尾部是
 * 0 = 不限、`AGE_PRESETS_DAYS` 尾部是 0 = 不清理）。于是滚轮向下滚一格，会把
 * 4096 MB / 90 天直接变成"不限 / 永不清理"，用户看到的就是"滚轮一滚就跳到 0"。
 *
 * 本测试把两条契约钉住：
 *   1. 指针在**数字输入框**上滚轮 → 值按步长（整数 ±1）走，且不跳到 0；
 *   2. 指针在**预设下拉**上滚轮 → 不得把值改成那个极端的 0 预设。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import sessionReducer from "../../features/session/sessionSlice";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { RenderCacheDialog } from "./RenderCacheDialog";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver，而 AppDialog 的滚动区在布局 effect 里会构造它。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

vi.mock("../../services/api/core", async () => {
    const actual =
        await vi.importActual<typeof import("../../services/api/core")>("../../services/api/core");
    return {
        ...actual,
        coreApi: {
            ...actual.coreApi,
            getRenderCacheStats: vi.fn(async () => ({
                ok: true,
                totalBytes: 0,
                entries: 0,
                sessionHits: 0,
                sessionMisses: 0,
                maxBytes: 0,
                maxAgeDays: 0,
            })),
        },
    };
});

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(async () => {
    await act(async () => root.unmount());
    document.body.innerHTML = "";
});

async function mountDialog() {
    // `keybindings` 是必需的：`useFineAdjustModifier` 从它读"精细调整"修饰键。
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <RenderCacheDialog open onOpenChange={() => undefined} />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
    // `useNonPassiveWheel` 把元素存进 state、在 effect 里挂监听：必须先把 effect
    // 冲刷掉，否则合成滚轮事件会落在"监听还没挂上"的空窗里（探针曾因此误判
    // "所有输入框都不响应滚轮"）。
    await act(async () => {
        await Promise.resolve();
    });
    return store;
}

test("新增「导出音频时复用渲染缓存」开关，默认开启且可切换", async () => {
    await mountDialog();
    const label = document.querySelector("label");
    // 用文案定位（测试环境默认 en-US）。
    const text = "Reuse the render cache when exporting audio (skips re-synthesis)";
    const row = [...document.body.querySelectorAll("span")].find((el) => el.textContent === text);
    expect(row, `找不到开关文案：${text}`).toBeTruthy();

    const checkbox = row!.parentElement!.querySelector("button[role=checkbox]");
    expect(checkbox, "开关应渲染为 checkbox").toBeTruthy();
    expect(checkbox!.getAttribute("aria-checked"), "默认开启").toBe("true");

    await act(async () => {
        checkbox!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(checkbox!.getAttribute("aria-checked"), "点击后关闭").toBe("false");
    void label;
});
