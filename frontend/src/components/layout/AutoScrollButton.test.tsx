/**
 * 自动滚屏工具栏按钮。
 *
 * 【要钉死什么】四件事，缺一条就会退化成"看起来能用"：
 * 1. **图标真的随模式换**（不是只换 tooltip）—— 用户对模式的认知全靠这一枚图标；
 * 2. 左键仍是开关（右键菜单不得劫持它）；
 * 3. 右键菜单是标准 `AppContextMenu`：标题行 + 两个单选项，当前项打勾；
 * 4. 选模式会回调并关掉菜单。
 *
 * 另有一条**源码级**断言：旧图标 `DoubleArrowRightIcon` 必须彻底消失 —— 它只可能
 * 以 import 的形式回来，运行时快照抓不到。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { AutoScrollButton } from "./AutoScrollButton";
import type { AutoScrollMode } from "../../utils/autoFollowScroll";

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

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    window.__TAURI__ = { core: { invoke: async () => ({}) as never } };
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    delete window.__TAURI__;
    vi.restoreAllMocks();
});

async function render(props: {
    enabled: boolean;
    mode: AutoScrollMode;
    onToggle?: () => void;
    onSelectMode?: (mode: AutoScrollMode) => void;
}) {
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <AutoScrollButton
                        enabled={props.enabled}
                        mode={props.mode}
                        onToggle={props.onToggle ?? (() => undefined)}
                        onSelectMode={props.onSelectMode ?? (() => undefined)}
                    />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    const button = container.querySelector<HTMLButtonElement>("button");
    expect(button, "自动滚屏按钮必须渲染出来").toBeTruthy();
    return button!;
}

test("图标随模式切换（不是只换 tooltip）", async () => {
    await render({ enabled: false, mode: "scroll" });
    expect(container.querySelector('[data-hs-auto-scroll-icon="continuous"]')).toBeTruthy();
    expect(container.querySelector('[data-hs-auto-scroll-icon="page"]')).toBeNull();

    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <AutoScrollButton
                        enabled={false}
                        mode="page"
                        onToggle={() => undefined}
                        onSelectMode={() => undefined}
                    />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });
    expect(container.querySelector('[data-hs-auto-scroll-icon="page"]')).toBeTruthy();
    expect(container.querySelector('[data-hs-auto-scroll-icon="continuous"]')).toBeNull();
});

test("tooltip / aria-label 随模式变化，且带上功能名", async () => {
    const scrollButton = await render({ enabled: false, mode: "scroll" });
    const scrollLabel = scrollButton.getAttribute("aria-label");
    expect(scrollLabel).toContain("Auto Scroll");
    expect(scrollLabel).toContain("Continuous");

    const pageButton = await render({ enabled: false, mode: "page" });
    const pageLabel = pageButton.getAttribute("aria-label");
    expect(pageLabel).toContain("Auto Scroll");
    expect(pageLabel).toContain("Page");
    expect(pageLabel).not.toBe(scrollLabel);
});

test("左键点击触发开关，不打开菜单", async () => {
    const onToggle = vi.fn();
    const button = await render({ enabled: false, mode: "scroll", onToggle });
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});

test("右键打开模式菜单：标题 + 两个单选项，当前项打勾", async () => {
    const button = await render({ enabled: true, mode: "page" });
    await act(async () => {
        button.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 40, clientY: 40 }),
        );
    });

    const menu = document.body.querySelector('[data-hs-context-menu="1"]');
    expect(menu, "右键必须打开菜单").toBeTruthy();
    // 标题行：不是可选条目，用弱化小标题渲染。
    expect(menu!.querySelector(".hs-menu__label")?.textContent).toBe("Auto Scroll");

    const items = Array.from(menu!.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]"));
    expect(items).toHaveLength(2);
    const checked = items.filter((item) => item.getAttribute("aria-checked") === "true");
    expect(checked).toHaveLength(1);
    expect(checked[0].textContent).toContain("Page");
    // 两项各带自己的图标：菜单里就把"图标 ↔ 模式"这个配对教会用户。
    expect(menu!.querySelector('[data-hs-auto-scroll-icon="continuous"]')).toBeTruthy();
    expect(menu!.querySelector('[data-hs-auto-scroll-icon="page"]')).toBeTruthy();
});

test("点击菜单项回调模式并关闭菜单", async () => {
    const onSelectMode = vi.fn();
    const button = await render({ enabled: false, mode: "scroll", onSelectMode });
    await act(async () => {
        button.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 10, clientY: 10 }),
        );
    });
    const menu = document.body.querySelector('[data-hs-context-menu="1"]')!;
    const pageItem = Array.from(menu.querySelectorAll<HTMLElement>("[role=menuitemcheckbox]")).find(
        (item) => item.textContent?.includes("Page"),
    );
    expect(pageItem, "菜单里必须有「翻页式」").toBeTruthy();

    await act(async () => {
        pageItem!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(onSelectMode).toHaveBeenCalledWith("page");
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});

test("旧图标 DoubleArrowRightIcon 已彻底弃用（源码级）", () => {
    /*
     * 只扫**代码行**：两枚新图标的注释里会提到旧图标的名字（"它已随本次改动彻底
     * 弃用"），那是有价值的设计记录，不该被这条门禁逼着删掉。真正的回归只会出现在
     * import 行或 JSX 里，两处都不是注释。
     */
    const codeOnly = (source: string) =>
        source
            .split("\n")
            .filter((line) => {
                const trimmed = line.trim();
                return !trimmed.startsWith("//") && !trimmed.startsWith("*");
            })
            .join("\n");

    for (const file of ["ActionBar.tsx", "AutoScrollButton.tsx"]) {
        // 相对路径与 `keyReferenceIntegrity.test.ts` 同口径：vitest 的工作目录就是 frontend/。
        const source = readFileSync(join("src", "components", "layout", file), "utf8");
        expect(codeOnly(source), `${file} 仍在引用旧图标`).not.toContain("DoubleArrowRightIcon");
    }
});
