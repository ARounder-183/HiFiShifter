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
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import sessionReducer, {
    setActiveVibratoPreset,
    upsertVibratoPreset,
} from "../../features/session/sessionSlice";
import { sanitizeVibratoPreset } from "../../features/vibrato/vibratoPresets";
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

async function mountDialog(
    prepare?: (store: ReturnType<typeof configureStore>) => void,
    onOpenChange: (open: boolean) => void = () => undefined,
) {
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    prepare?.(store);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <VibratoPresetDialog open onOpenChange={onOpenChange} />
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

test("布局：定高 wrapper + 两栏 flex 填充，且不再依赖 maxHeight（反双滚动条）", async () => {
    await mountDialog();

    /*
     * 【结构契约】对话框正文自身可滚；双竖直滚动条（内层两栏 + 外层正文）
     * 的根因是"内容总高超过正文上限"。现在的结构是：内容包一层**定高**
     * wrapper（min(60vh, 560px)），两栏 `flex-1 min-h-0` 填满剩余高度 ——
     * 条件行（只读提示 / 导入反馈）只压缩栏高，永远把不破外层。
     *
     * 断言三条：
     * 1. 定高 wrapper 恰有一个，两个滚动视口都是它的后代（同一高度语境）；
     * 2. 面板**不再**使用 maxHeight（那正是被替换掉的失败机制）；
     * 3. 面板以 flex 填充（min-h-0 + flex-1），jsdom 无排版引擎，测不了
     *    真实高度，结构属性是能钉住的最强代理。
     */
    const viewports = [...document.querySelectorAll<HTMLElement>(SCROLL_VIEWPORT)];
    expect(viewports.length).toBeGreaterThanOrEqual(2);

    const wrapperSelector = "[data-vibrato-content]";
    const wrappers = document.querySelectorAll(wrapperSelector);
    expect(wrappers.length, "定高 wrapper 应恰有一个").toBe(1);
    for (const viewport of viewports) {
        expect(
            wrappers[0].contains(viewport),
            "滚动视口必须都在定高 wrapper 内（否则外层正文会滚动）",
        ).toBe(true);
    }

    const maxHeighted = viewports.filter(
        (viewport) => (viewport.parentElement?.style.maxHeight ?? "") !== "",
    );
    expect(maxHeighted, "面板不得再用 maxHeight（那是双滚动条的失败机制）").toEqual([]);

    const panes = viewports.map((viewport) => viewport.closest(".min-h-0.flex-1"));
    expect(panes.filter(Boolean).length).toBeGreaterThanOrEqual(2);
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
    // 只扫**参数表单**区域：约束的对象是"两个以上表单控件并排"的行；列表行
    // （glyph + 名称）同样用 Flex，但不属于这条约束。
    const formStart = source.indexOf("<AppForm>");
    const formEnd = source.indexOf("</AppForm>") + "</AppForm>".length;
    expect(formStart).toBeGreaterThan(0);
    const form = source.slice(formStart, formEnd);
    const rows = form.match(/<Flex align="center" gap="2"[^>]*>/g) ?? [];
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

/*
 * R6a：预览画布的可编辑性。
 *
 * 【契约】用户预设的预览可拖（画手柄、接受手势），系统预设的预览只读 ——
 * 与参数表单"系统预设禁用一切字段"一致，避免"为什么别的能拖这里不能"的歧义。
 * 交互性由 `data-testid` 暴露：它只在可拖时出现。
 */
test("用户预设：预览画布可编辑（渲染交互层）", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_grab", name: "Grab Me", depthCents: 40 });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });
    expect(document.querySelector('[data-testid="vibrato-preview-interactive"]')).toBeTruthy();
});

test("系统预设：预览画布只读（不渲染交互层）", async () => {
    await mountDialog();
    // 默认活动预设是系统预设「自然」。
    expect(document.querySelector('[data-testid="vibrato-preview-interactive"]')).toBeNull();
});

/*
 * R6b：手绘周期编辑器的入口。
 *
 * 【契约】用户预设可以展开手绘编辑器（`table` 波形的一等入口）；系统预设只读，
 * 入口按钮禁用。展开后写进草稿的是 `table` 波形 —— 预览 / 试听 / 应用全管线
 * 无差别支持。
 */
test("用户预设：点「手绘…」展开周期编辑器", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_draw", name: "Draw Me", depthCents: 40 });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    expect(document.querySelector('[data-testid="vibrato-cycle-editor"]')).toBeNull();
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    expect(drawButton, "手绘入口应已渲染").toBeTruthy();
    await act(async () => {
        drawButton!.click();
    });
    expect(document.querySelector('[data-testid="vibrato-cycle-editor"]')).toBeTruthy();
});

test("系统预设：手绘入口禁用", async () => {
    await mountDialog();
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    ) as HTMLButtonElement | undefined;
    expect(drawButton, "手绘入口应已渲染").toBeTruthy();
    expect(drawButton!.disabled).toBe(true);
    expect(document.querySelector('[data-testid="vibrato-cycle-editor"]')).toBeNull();
});

/*
 * 交互修正：保存不关闭 + 切换预设不丢编辑。
 *
 * 【为什么值得测】这两条都是"用户能感觉到、但不会抛错"的交互缺陷：
 * 1. 点「保存」把窗口一起收掉 —— 用户想"先存一版、接着调"时只能重开；
 * 2. 编辑到一半切到别的预设，当前改动被静默丢弃 —— 于是"编辑途中不能换预设"。
 * 前者断言"保存不请求关闭"，后者断言"切走时改动已写回库"。
 */
test("保存不关闭对话框（可先存一版接着调）", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_save", name: "Save Me", depthCents: 40 });
    const onOpenChange = vi.fn();
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    }, onOpenChange);

    const saveButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Save",
    );
    expect(saveButton, "保存按钮应已渲染").toBeTruthy();
    await act(async () => {
        saveButton!.click();
    });
    expect(onOpenChange).not.toHaveBeenCalled();
});

test("切换预设时把未保存的改动写回库（编辑途中可换预设）", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_edit", name: "Edit Me", depthCents: 40 });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    // 制造一处未保存的改动：点「手绘…」把草稿的波形换成手绘表。
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    expect(drawButton, "手绘入口应已渲染").toBeTruthy();
    await act(async () => {
        drawButton!.click();
    });

    // 切到系统预设「直线」（出厂顺序首位）。
    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const straightRow = rows.find((row) => row.textContent?.includes("Straight"));
    expect(straightRow, "系统预设行应已渲染").toBeTruthy();
    await act(async () => {
        straightRow!.click();
    });

    // 改动已写回库，而不是被丢弃。
    const saved = store.getState().session.vibratoPresets.find((p) => p.id === "custom_edit");
    expect(saved?.cycle.kind).toBe("table");
});

/*
 * 关闭入口：保存不再关闭窗口之后，页脚必须有一个显式的「关闭」按钮 ——
 * 否则用户只剩 Esc / 点外部两条不显眼的路。
 */
test("页脚有「关闭」按钮，点击请求关闭窗口", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_close", name: "Close Me", depthCents: 40 });
    const onOpenChange = vi.fn();
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    }, onOpenChange);

    const closeButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Close",
    );
    expect(closeButton, "关闭按钮应已渲染").toBeTruthy();
    await act(async () => {
        closeButton!.click();
    });
    expect(onOpenChange).toHaveBeenCalledWith(false);
});

test("导入 / 导出按钮文案不带省略号（省空间）", async () => {
    await mountDialog();
    const labels = [...document.querySelectorAll("button")].map((b) => b.textContent?.trim() ?? "");
    expect(labels).toContain("Import");
    expect(labels).toContain("Export");
    expect(labels.some((label) => label === "Import...")).toBe(false);
    expect(labels.some((label) => label === "Export...")).toBe(false);
});
