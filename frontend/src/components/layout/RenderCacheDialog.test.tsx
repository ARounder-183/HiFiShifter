// @vitest-environment jsdom
/*
 * 渲染缓存管理窗口：开关与数字输入框的滚轮契约。
 *
 * 【为什么必须有】「占用上限」「未使用超期清理」曾各配一个**预设下拉**（末项是
 * `0` = 不限 / 不清理），用户报告"滚轮一滚就跳到 0"。真实根因不是滚轮、也不是下拉被
 * 滚动选中，而是 Radix 的隐藏原生 `<select>` 在**受控值不在选项里**时把 `""` 上报成
 * 一次变更（`AppDialog` 的 `<form>` 正是它存在的条件），调用方 `Number("") === 0`
 * 于是把占用上限静默改成"不限"。详见 `ui/Select.test.tsx` 与 `ui/Select.tsx` 的守卫注释。
 *
 * 因此两个字段改回**普通数字输入框**（与下方四个同一种控件）。本测试钉住：
 *   1. 指针在输入框上滚轮 → 按单位步长走一格（整数 ±1），**不得**跳到 0；
 *   2. 按住精细调整修饰键同样按该单位的 fine 步长；
 *   3. 这两行不再有下拉（否则那个哨兵缺陷会以同样的方式回来）。
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

/** 等两帧，让数字框的帧合并提交器落地（与 `wheelThrottle.test.tsx` 同一手法）。 */
async function nextFrame() {
    await act(async () => {
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
        await new Promise((resolve) => requestAnimationFrame(() => resolve(null)));
    });
}

/** 按无障碍名称取数字输入框（en-US 文案）。 */
function numberField(ariaLabel: string): HTMLInputElement {
    const el = document.querySelector<HTMLInputElement>(`input[aria-label="${ariaLabel}"]`);
    expect(el, `找不到数字框：${ariaLabel}`).toBeTruthy();
    return el!;
}

/**
 * 在数字框上滚一格。
 *
 * 【为什么要打在外层包裹 div 上】`AppNumberField` 把非被动 wheel 监听挂在包裹层，
 * 而 Radix 的 `TextField.Root` 在 `<input>` 外还有一层容器；事件必须**冒泡**上去。
 */
async function wheelNumberField(el: HTMLInputElement, deltaY: number, init: WheelEventInit = {}) {
    await act(async () => {
        el.parentElement!.dispatchEvent(
            new WheelEvent("wheel", { deltaY, bubbles: true, cancelable: true, ...init }),
        );
    });
    await nextFrame();
}

test("占用上限：滚轮按 ±1 走一格，不会跳到 0", async () => {
    await mountDialog();
    const field = numberField("Size limit");
    expect(field.value, "默认 4096 MB").toBe("4096");

    await wheelNumberField(field, -100);
    expect(field.value, "向上滚一格 = +1").toBe("4097");

    await wheelNumberField(field, 100);
    await wheelNumberField(field, 100);
    expect(field.value, "向下滚两格 = −2").toBe("4095");
});

test("未使用超期清理：滚轮按 ±1 走一格，不会跳到 0", async () => {
    await mountDialog();
    const field = numberField("Prune unused after");
    expect(field.value, "默认 90 天").toBe("90");

    await wheelNumberField(field, -100);
    expect(field.value, "向上滚一格 = +1").toBe("91");
});

test("精细调整修饰键：整数单位仍是 ±1（与下方四个字段同一口径）", async () => {
    await mountDialog();
    const field = numberField("Size limit");

    await wheelNumberField(field, -100, { ctrlKey: true });
    expect(field.value).toBe("4097");
});

test("这两行不再有预设下拉（否则哨兵缺陷会以同样方式回来）", async () => {
    await mountDialog();
    for (const ariaLabel of ["Size limit", "Prune unused after"]) {
        const row = numberField(ariaLabel).closest(".app-field__control")?.parentElement;
        expect(row, `找不到 ${ariaLabel} 所在行`).toBeTruthy();
        expect(row!.querySelector('[role="combobox"]'), `${ariaLabel} 行内不应再有下拉`).toBeNull();
    }
});
