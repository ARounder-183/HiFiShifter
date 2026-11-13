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
import { messages } from "../../i18n/messages";
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
    // 右下角小三角：告诉用户"右键还有模式可选"（与参数编辑器的工具族按钮同一个记号）。
    expect(container.querySelector("[data-hs-menu-caret]")).toBeTruthy();

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

test("tooltip / aria-label 随模式变化，且格式只是「功能名：模式名」", async () => {
    /*
     * 【为什么要钉"格式"而不只是"包含"】悬停气泡是一句**标签**，不是说明书：
     * 它只需要回答"这是什么控件、当前什么模式"。曾经这里写着"持续跟随播放头。
     * 右键选模式。"——那是文档，不是标签；把功能名与模式名挖掉之后剩下的必须是
     * 纯分隔符，这一条比"包含某些词"更能防住文案重新膨胀。
     */
    const dict = messages["en-US"] as unknown as Record<string, string>;
    const labels: string[] = [];

    for (const mode of ["scroll", "page"] as const) {
        const button = await render({ enabled: false, mode });
        const label = button.getAttribute("aria-label") ?? "";
        labels.push(label);

        // 功能名与模式名都在（缺一个，读屏用户就不知道这是什么 / 当前是什么）。
        expect(label, `${mode}: 必须带功能名`).toContain(dict.auto_scroll);
        expect(label, `${mode}: 必须带模式名`).toContain(dict[`auto_scroll_mode_${mode}`]);

        // 挖掉两者之后只剩分隔符 —— 任何解释性文字都会在这里现形。
        const rest = label
            .replace(dict.auto_scroll, "")
            .replace(dict[`auto_scroll_mode_${mode}`], "");
        expect(rest.trim(), `${mode}: 气泡里只剩「功能名：模式名」，实际是「${label}」`).toMatch(
            /^[:：\s]*$/,
        );
    }

    expect(labels[0]).not.toBe(labels[1]);
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
    // 菜单里**没有**标题行：只有选项（点开就能选，不需要先读一遍"这是谁的菜单"）。
    expect(menu!.querySelector(".hs-menu__label")).toBeNull();

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
