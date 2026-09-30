// @vitest-environment jsdom
/*
 * 颤音预设管理器的布局契约。
 *
 * 【为什么必须有】这里出过两个只有肉眼能发现的缺陷，而它们都不会抛错、不会被
 * `tsc` 或任何现有门禁拦下：
 *
 * 1. **预览与参数分离**。预览原本是参数流的**第一行**，于是参数一长、竖直滚动条
 *    一出现，用户调下面的参数时就得滚回顶部才能看效果 —— 编辑流程被切断。
 *    现在的契约是：预览**不属于任何滚动区**，因此它在任何滚动位置都可见。
 * 2. **横向滚动条**。`AppNumberField` 内部是 `<input>`，固有最小宽度约 20 字符；
 *    两个数字框 + 一个下拉并排时 min-content 宽度超过控件列，把面板撑出横向
 *    滚动条。修法是让这些并排行**可换行**，而不是横向溢出。
 *
 * 两条都用结构断言钉住 —— 它们与具体文案、具体参数项无关，因此不会随 UI 微调
 * 而失效，但也正是这样才拦得住"下一次重构又把预览塞回参数流"。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { readFileSync } from "node:fs";
import { afterEach, beforeEach, expect, test } from "vitest";

import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import sessionReducer from "../../features/session/sessionSlice";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { VibratoPresetDialog } from "./VibratoPresetDialog";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver；预览画布与对话框的滚动区都会构造它。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;
// 预览画布用 2D context 绘制；jsdom 默认返回 null，会让组件提前返回。
(HTMLCanvasElement.prototype.getContext as unknown) ??= () => null;

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
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <VibratoPresetDialog open onOpenChange={() => undefined} />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
    await act(async () => {
        await Promise.resolve();
    });
    return store;
}

/** Radix ScrollArea 的视口节点：滚动就发生在这里。 */
const SCROLL_VIEWPORT = "[data-radix-scroll-area-viewport]";

test("预览不落在任何滚动区内 —— 调参数时它不会被滚出视野", async () => {
    await mountDialog();

    const canvas = document.querySelector("canvas[role=img]");
    expect(canvas, "预览画布应已渲染").toBeTruthy();

    // 契约：从预览往上找，不应碰到滚动视口。
    expect(
        canvas!.closest(SCROLL_VIEWPORT),
        "预览被放进了滚动区：参数一长它就会滚出去（这正是要修的那个缺陷）",
    ).toBeNull();

    // 反向对照：参数区**必须**在自己那层滚动区里，否则长表单会把对话框撑高，
    // 变成"整张对话框一起滚"（预览也就跟着滚走了）。
    // 用 `section`（`AppFormSection` 的根元素）定位参数区：`AppForm` 渲染的是
    // 普通 `<div>`，而 `document.querySelector("form")` 会命中 `AppDialog` 自己的
    // 外层表单 —— 那是预览与面板的共同祖先，断言它会永远为真。
    const section = document.querySelector("section");
    expect(section, "参数分区应已渲染").toBeTruthy();
    expect(section!.closest(SCROLL_VIEWPORT), "参数区应位于滚动区内").not.toBeNull();
});

test("同排两栏共用一个高度上限，两栏等高", async () => {
    await mountDialog();

    const viewports = [...document.querySelectorAll<HTMLElement>(SCROLL_VIEWPORT)];
    // 预设列表 + 参数表单：两个**并排**的滚动区（不是嵌套的两层）。
    expect(viewports.length).toBeGreaterThanOrEqual(2);

    const heights = viewports
        .map((viewport) => viewport.parentElement?.style.maxHeight ?? "")
        .filter((value) => value !== "");
    expect(heights.length).toBeGreaterThanOrEqual(2);
    expect(new Set(heights).size, `两栏高度上限应相同，实际为 ${heights.join(" / ")}`).toBe(1);
});

/*
 * 并排控件行的换行约束，改成**读源码**的断言。
 *
 * 【为什么不做 DOM 断言】jsdom 没有排版引擎：`getComputedStyle` 拿不到真实的
 * min-content 宽度，也就无法真的测出"会不会溢出"。这里试过按 DOM 结构找
 * "含两个以上控件的行"，结果是**空转**（控件被包在各自的包装元素里，子选择器
 * 一个都匹配不到）—— 一个永远为真的断言比没有断言更糟。
 *
 * 【这条断言守的是什么】`AppNumberField` 渲染的是真实 `<input>`，浏览器给它的
 * 固有最小宽度约 20 字符（≈170px）。两个输入框加一个下拉并排时，min-content
 * 宽度超过控件列 —— 若这一行不可换行，就只能横向溢出。所以：**参数表单里凡是
 * 并排两个以上控件的行，都必须 `wrap="wrap"`**。这是一条源码级代理断言，它不
 * 验证浏览器行为，但能在"下一次重构去掉 wrap"时立刻失败。
 */
test("参数表单里的并排控件行都允许换行（源码级约束）", () => {
    const source = readFileSync("src/components/layout/VibratoPresetDialog.tsx", "utf8");
    const rows = source.match(/<Flex align="center" gap="2"[^>]*>/g) ?? [];
    // 先确认确实扫到了这些行 —— 否则下面的断言是空转（模式改名就会这样）。
    expect(rows.length, "应至少扫到一行并排控件").toBeGreaterThan(0);
    const missing = rows.filter((row) => !row.includes('wrap="wrap"'));
    expect(missing, "这些并排控件行没有 wrap，放不下时会把面板撑出横向滚动条").toEqual([]);
});

/*
 * 导入链路的端到端测试：页脚按钮 → 隐藏文件输入 → 解析 / 净化 / 去重 → 入库 →
 * 列表出现新预设。走的是**真实 File 对象**（jsdom 支持 Blob.text()），因此
 * parse / merge / store 之间的衔接是被真实执行的，不是 mock。
 *
 * 【为什么值得测】"点了没反应"是这条链路最可能的故障形态：任何一步静默失败
 * （change 未触发、text() 抛、净化拒收），UI 上都毫无动静。
 */
test("导入：选中的文件经净化入库，列表出现新预设，反馈可见", async () => {
    const store = await mountDialog();

    const before = store.getState().session.vibratoPresets.length;
    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    expect(input, "隐藏文件输入应已挂载").toBeTruthy();

    const fileText = JSON.stringify({
        kind: "hifishifter-vibrato-presets",
        version: 1,
        presets: [
            {
                id: "builtin.natural",
                name: "Imported One",
                depthCents: 33,
                rateHz: 6,
            },
        ],
    });
    const file = new File([fileText], "presets.json", { type: "application/json" });
    Object.defineProperty(input!, "files", { value: [file] });

    await act(async () => {
        input!.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();
    });

    const after = store.getState().session.vibratoPresets;
    expect(after.length).toBe(before + 1);
    const added = after[after.length - 1];
    // 文件里的 builtin. id 被重写为用户 id —— 否则 upsert 拒收、静默丢条目。
    expect(added?.name).toBe("Imported One");
    expect(added?.id.startsWith("custom_")).toBe(true);
    expect(added?.builtin).toBe(false);
    expect(added?.depthCents).toBe(33);

    // 行内反馈可见，且不是危险色（成功）。
    const notice = document.querySelector('[role="status"]');
    expect(notice?.textContent ?? "").toContain("Imported 1 preset");

    // 列表里真的出现了这个名字。
    expect(document.body.textContent).toContain("Imported One");
});

test("导入拿错的文件（主题 / 布局 JSON）：拒收且给出明确反馈", async () => {
    await mountDialog();

    const input = document.querySelector<HTMLInputElement>('input[type="file"]');
    const file = new File([JSON.stringify({ name: "A", colors: {} })], "theme.json", {
        type: "application/json",
    });
    Object.defineProperty(input!, "files", { value: [file] });

    await act(async () => {
        input!.dispatchEvent(new Event("change", { bubbles: true }));
        await Promise.resolve();
    });

    const notice = document.querySelector('[role="status"]');
    expect(notice?.textContent ?? "").toContain("not a vibrato preset file");
});
