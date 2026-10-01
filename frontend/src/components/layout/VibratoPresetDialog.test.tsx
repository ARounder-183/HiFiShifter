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
 * 预览画布拖拽：拖到一半才按下 / 松开「精细调整」，不能打断当前拖拽。
 *
 * 【为什么值得测】根因是每帧用"从起点算起的**总**位移 × 当前比例"重算 —— 一按
 * Ctrl，此前累计的位移被整体重新缩小，数值瞬间跳回去（闪回），正在进行的拖拽
 * 被打断。正确做法是按**增量**缩放：比例变化只影响此后每帧走多少。这里把"累计量
 * 连续"钉在组件层面（换算本身在 `vibratoPreviewGestures.test.ts` 另有单测）。
 *
 * jsdom 没有排版：画布宽度为 0，命中测试因此一律落在**主体**（相位 / 深度）上，
 * 纵向位移与深度 1:1（`centsPerPx` 初值为 1）—— 断言正好干净。
 */
test("预览画布拖拽：中途按下 / 松开精细调整都不闪回", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_fine_drag",
        name: "Fine Drag",
        depthCents: 30,
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const container = document.querySelector<HTMLElement>(
        '[data-testid="vibrato-preview-interactive"]',
    );
    expect(container, "交互层应已渲染").toBeTruthy();
    // jsdom 没有指针捕获 API（画布按下时会调）。
    container!.setPointerCapture = () => undefined;
    container!.releasePointerCapture = () => undefined;
    container!.hasPointerCapture = () => false;

    const depthValue = () =>
        Number(
            document.querySelector<HTMLInputElement>('input[aria-label="Depth"]')?.value ?? "NaN",
        );

    const dragTo = async (clientY: number, ctrlKey: boolean) => {
        // 手势回调挂在画布容器上（React 合成事件），因此要派发到容器而不是 window。
        await act(async () => {
            container!.dispatchEvent(
                new PointerEvent("pointermove", { bubbles: true, clientY, ctrlKey }),
            );
        });
        return depthValue();
    };

    await act(async () => {
        container!.dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, button: 0, clientX: 0, clientY: 0 }),
        );
    });

    // 不按修饰键：向上 20px → 深度 +20。
    const coarse = await dragTo(-20, false);
    expect(coarse).toBeCloseTo(50, 0);

    // 按下 Ctrl 的瞬间：累计量必须还在原处（闪回时这里会掉到 30 出头）。
    const atToggle = await dragTo(-30, true);
    expect(atToggle, "按下 Ctrl 的瞬间不得闪回").toBeGreaterThanOrEqual(coarse);
    // 同样走 10px，现在推进得明显更少（但不为零）。
    expect(atToggle - coarse).toBeGreaterThan(0);
    expect(atToggle - coarse).toBeLessThan(10);

    // 按住 Ctrl 继续走：仍在推进，只是慢。
    const fine = await dragTo(-40, true);
    expect(fine).toBeGreaterThan(atToggle);
    expect(fine - atToggle).toBeLessThan(10);

    // 松开 Ctrl：累计量同样不跳，速度恢复。
    const released = await dragTo(-50, false);
    expect(released).toBeGreaterThan(fine);

    await act(async () => {
        container!.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, clientY: -50 }));
    });
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
 * 右键整体变换要一路走到草稿（不只是编辑器内部的事）。
 *
 * 【契约】右键拖拽改的是**同一个** `cycle: { kind: "table", table }` ——
 * 与画笔、与"从选区提取"完全同源，因此预览 / 试听 / 应用无需任何特殊处理。
 * 这条测试同时钉住"手势的位移真的落进了草稿"，而不只是 `onChange` 被调用过。
 */
test("右键拖拽整体旋转手绘曲线，落地仍是 table 波形", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_rotate",
        name: "Rotate Me",
        depthCents: 40,
    });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    await act(async () => {
        drawButton!.click();
    });

    const canvas = document.querySelector<HTMLCanvasElement>(
        '[data-testid="vibrato-cycle-editor"] canvas',
    );
    expect(canvas, "周期画布的 canvas 应已渲染").toBeTruthy();
    const container = canvas!.parentElement as HTMLDivElement;
    // jsdom 没有排版：手工给出几何，手势的换算才有的算（宽 640 / 64 格 → 每格 10px）。
    const width = 640;
    container.getBoundingClientRect = () =>
        ({
            left: 0,
            top: 0,
            right: width,
            bottom: 120,
            width,
            height: 120,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        }) as DOMRect;
    Object.defineProperty(container, "clientWidth", { value: width, configurable: true });

    const pointer = (type: string, init: PointerEventInit) =>
        new PointerEvent(type, { bubbles: true, cancelable: true, ...init });
    // 向右拖 160px = 1/4 周期 = 16 格。
    await act(async () => {
        canvas!.dispatchEvent(pointer("pointerdown", { button: 2, clientX: 100, clientY: 60 }));
    });
    await act(async () => {
        canvas!.dispatchEvent(pointer("pointermove", { clientX: 260, clientY: 60 }));
    });
    await act(async () => {
        canvas!.dispatchEvent(pointer("pointerup", { button: 2, clientX: 260, clientY: 60 }));
    });

    // 切到系统预设：未保存的改动会被写回库（既有行为），从库里读最终落地的波形。
    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const straightRow = rows.find((row) => row.textContent?.includes("Straight"));
    await act(async () => {
        straightRow!.click();
    });

    const saved = store.getState().session.vibratoPresets.find((p) => p.id === "custom_rotate");
    expect(saved?.cycle.kind).toBe("table");
    const table = (saved!.cycle as { kind: "table"; table: number[] }).table;
    expect(table.length).toBe(64);
    // 正弦的峰原在第 16 格，转过 1/4 周期后落到第 32 格。
    expect(table.indexOf(Math.max(...table))).toBe(32);
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

/*
 * 预览纵轴：一次性拟合到"档位值"，而不是跟着当前深度自适应。
 *
 * 【为什么值得测】自适应标尺会让波形永远填满画布 —— 调深度时只看到整幅在竖直方向
 * 抖一下，读不出幅度大小。契约是：标尺由**拟合档位**给出（这里深度 40 → 量程 50），
 * 编辑期间保持不动，于是波形高度就等于深度。
 */
test("预览纵轴是一次性拟合的档位值（不是当前峰值）", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_axis",
        name: "Axis",
        depthCents: 40,
        irregularity: 0,
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const container = document.querySelector<HTMLElement>("[data-axis-cents]");
    expect(container, "预览画布应暴露纵轴量程").toBeTruthy();
    const axis = Number(container!.getAttribute("data-axis-cents"));
    expect(Number.isFinite(axis)).toBe(true);
    // 量程 ≥ 峰值（静止时不裁切），且落在阶梯档位上而不是等于峰值。
    expect(axis).toBeGreaterThanOrEqual(40);
    expect(axis).toBe(50);
});

test("预览卡片有「适应」按钮（重新拟合纵轴）", async () => {
    await mountDialog();
    const fitButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Fit",
    );
    expect(fitButton, "适应按钮应已渲染").toBeTruthy();
});

/*
 * 停用 / 启用：每行一个按钮，停用后该预设不进工具栏列表、拖拽循环切换也跳过。
 * 这里断言按钮确实写进了切片里的停用名单。
 */
test("预设行有启用 / 停用按钮，点击写进停用名单", async () => {
    const store = await mountDialog();
    const disableButtons = [...document.querySelectorAll("button")].filter(
        (button) => button.getAttribute("aria-label") === "Disable",
    );
    // 每个预设一行，行数 = 系统 + 用户。
    expect(disableButtons.length).toBeGreaterThan(0);

    await act(async () => {
        disableButtons[0].click();
    });

    // 出厂顺序首位是「直线」。
    expect(store.getState().session.disabledVibratoPresetIds).toEqual(["builtin.straight"]);
});

/*
 * 右键菜单：把"针对这一条预设"的动作（启用 / 停用、设为当前、复制、删除）收拢到指针处。
 */
test("右键预设行打开上下文菜单", async () => {
    const store = await mountDialog();
    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const row = rows.find((entry) => entry.textContent?.includes("Straight"));
    expect(row, "预设行应已渲染").toBeTruthy();

    await act(async () => {
        row!.dispatchEvent(
            new MouseEvent("contextmenu", { bubbles: true, clientX: 20, clientY: 30 }),
        );
    });

    const text = document.body.textContent ?? "";
    expect(text).toContain("Disable");
    expect(text).toContain("Use as current");
    expect(text).toContain("Duplicate as mine");
    expect(text).toContain("Delete");
    // 系统预设现在也可排序：菜单里应当有上移 / 下移（首项的上移禁用）。
    expect(text).toContain("Move up");
    expect(text).toContain("Move down");
    const moveUp = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Move up",
    ) as HTMLButtonElement | undefined;
    expect(moveUp, "上移菜单项应已渲染").toBeTruthy();
    expect(moveUp!.disabled, "首项的「上移」应禁用").toBe(true);

    // 系统预设的名字来自词条：重命名不可用。
    const renameItem = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Rename",
    ) as HTMLButtonElement | undefined;
    expect(renameItem, "重命名菜单项应已渲染").toBeTruthy();
    expect(renameItem!.disabled, "系统预设的「重命名」应禁用").toBe(true);

    /*
     * 层级契约：菜单必须在**对话框的 DOM 子树里**。
     *
     * 手写菜单的层级（--qt-z-menu: 999）低于对话框（--qt-z-dialog: 1000），挂在
     * 对话框外面会被整个盖住、点不到；跑到对话框 DOM 之外时 Radix 还会把点击当成
     * "点了弹窗外面"顺手关掉窗口。钉住这条，避免下次重构又把它挪出去。
     */
    const menu = document.querySelector('[role="menu"]');
    expect(menu, "菜单应已渲染").toBeTruthy();
    expect(menu!.closest('[role="dialog"]'), "菜单必须位于对话框内部").not.toBeNull();

    // 菜单里的「停用」与行内按钮同源：点一下即写进名单。
    const menuDisable = [...document.querySelectorAll("button")].find(
        (button) =>
            button.textContent?.trim() === "Disable" && button.getAttribute("role") === "menuitem",
    );
    expect(menuDisable, "菜单项应已渲染").toBeTruthy();
    await act(async () => {
        menuDisable!.click();
    });
    expect(store.getState().session.disabledVibratoPresetIds).toContain("builtin.straight");
});

/*
 * 页脚动作不应关闭窗口。
 *
 * 【为什么值得测】`AppDialog` 对**同步**动作默认 `autoClose: true`，所以「导入」
 * 「新建」「复制为自定义」这类同步动作会顺手把窗口关掉 —— 用户刚点完就被弹出，
 * 只能重新打开继续。异步动作默认不关，因此这个缺陷只落在同步的那几个上，很容易
 * 在"把 async 去掉"的重构里复发。
 */
test("导入 / 新建 / 复制为自定义 都不关闭窗口", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_keep", name: "Keep", depthCents: 40 });
    const onOpenChange = vi.fn();
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    }, onOpenChange);

    for (const label of ["Import", "New", "Duplicate as mine"]) {
        const button = [...document.querySelectorAll("button")].find(
            (entry) => entry.textContent?.trim() === label,
        );
        expect(button, `${label} 按钮应已渲染`).toBeTruthy();
        await act(async () => {
            button!.click();
        });
    }

    expect(onOpenChange).not.toHaveBeenCalled();
});

/*
 * 拖拽排序：用户预设按住行上下拖即可换位（取代了原来的 ▲ / ▼ 按钮）。
 *
 * jsdom 没有排版引擎，`getBoundingClientRect()` 一律返回 0，因此"向下拖"必然落到
 * 列表末尾 —— 这刚好够断言"拖拽确实触发了一次 reorder"，而不必伪造行高。
 */
test("拖拽用户预设行可以调整顺序", async () => {
    const a = sanitizeVibratoPreset({ id: "custom_drag_a", name: "Drag A", depthCents: 30 });
    const b = sanitizeVibratoPreset({ id: "custom_drag_b", name: "Drag B", depthCents: 40 });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(a));
        store.dispatch(upsertVibratoPreset(b));
        store.dispatch(setActiveVibratoPreset(a.id));
    });
    expect(store.getState().session.vibratoPresets.map((preset) => preset.id)).toEqual([
        "custom_drag_a",
        "custom_drag_b",
    ]);

    const row = document.querySelector<HTMLElement>('[data-preset-row="custom_drag_a"]');
    expect(row, "可拖拽的用户预设行应已渲染").toBeTruthy();
    const target = row!.querySelector('[role="option"]') as HTMLElement;

    await act(async () => {
        target.dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, button: 0, clientY: 0 }),
        );
    });
    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointermove", { clientY: 40 }));
    });
    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointerup", { clientY: 40 }));
    });

    expect(store.getState().session.vibratoPresets.map((preset) => preset.id)).toEqual([
        "custom_drag_b",
        "custom_drag_a",
    ]);
});

test("上下调整按钮已被移除（改为拖拽 + 右键菜单）", async () => {
    const a = sanitizeVibratoPreset({ id: "custom_no_btn", name: "No Buttons" });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(a));
        store.dispatch(setActiveVibratoPreset(a.id));
    });
    const labels = [...document.querySelectorAll("button")].map(
        (button) => button.getAttribute("aria-label") ?? "",
    );
    expect(labels).not.toContain("Move up");
    expect(labels).not.toContain("Move down");
});

/*
 * 边缘自动滚动：指针贴住列表下缘不动时，列表也要持续上卷。
 *
 * jsdom 没有排版，这里给滚动视口伪造一个 400px 高的矩形与可读写的 `scrollTop`，
 * 再让 rAF 真跑几帧 —— 断言"拖到下缘之后 scrollTop 变大了"。
 */
test("拖到列表下缘会自动滚动", async () => {
    const presets = Array.from({ length: 6 }, (_, index) =>
        sanitizeVibratoPreset({ id: `custom_scroll_${index}`, name: `Scroll ${index}` }),
    );
    await mountDialog((store) => {
        presets.forEach((preset) => store.dispatch(upsertVibratoPreset(preset)));
        store.dispatch(setActiveVibratoPreset(presets[0].id));
    });

    const viewport = document.querySelector<HTMLElement>("[data-radix-scroll-area-viewport]");
    expect(viewport, "滚动视口应已渲染").toBeTruthy();
    viewport!.getBoundingClientRect = () =>
        ({
            top: 100,
            bottom: 500,
            left: 0,
            right: 200,
            width: 200,
            height: 400,
            x: 0,
            y: 100,
            toJSON: () => ({}),
        }) as DOMRect;
    let scrollTop = 0;
    Object.defineProperty(viewport!, "scrollTop", {
        configurable: true,
        get: () => scrollTop,
        set: (value: number) => {
            // 模拟"滚到底就停"：到顶格后不再变化，rAF 循环随之收敛（真实视口同理）。
            scrollTop = Math.min(30, value);
        },
    });

    const row = document.querySelector<HTMLElement>('[data-preset-row="custom_scroll_0"]');
    expect(row, "可拖拽的用户预设行应已渲染").toBeTruthy();
    const target = row!.querySelector('[role="option"]') as HTMLElement;

    await act(async () => {
        target.dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, button: 0, clientY: 110 }),
        );
    });
    // 拖到视口下缘（500 - 24 = 476 以内）并停住。
    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointermove", { clientY: 495 }));
    });
    // 让 rAF 真跑几帧：指针不动，滚动仍应推进。
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 60));
    });
    expect(scrollTop, "贴住下缘应当持续向下滚动").toBeGreaterThan(0);

    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointerup", { clientY: 495 }));
    });
    // 松手后不再滚动。
    const afterDrop = scrollTop;
    await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 40));
    });
    expect(scrollTop).toBe(afterDrop);
});

/*
 * 系统预设也可排序：拖拽写进 `builtinVibratoPresetOrder`（以 id 列表持久化）。
 *
 * 与用户预设同一套拖拽机制，但两组各排各的 —— 跨组拖拽没有明确语义，也会让
 * "系统预设只读"的边界变糊。
 */
test("拖拽系统预设行可以调整顺序", async () => {
    const store = await mountDialog();
    expect(store.getState().session.builtinVibratoPresetOrder).toEqual([]);

    const row = document.querySelector<HTMLElement>('[data-preset-row="builtin.straight"]');
    expect(row, "系统预设行应已渲染").toBeTruthy();
    const target = row!.querySelector('[role="option"]') as HTMLElement;

    await act(async () => {
        target.dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, button: 0, clientY: 0 }),
        );
    });
    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointermove", { clientY: 40 }));
    });
    await act(async () => {
        window.dispatchEvent(new PointerEvent("pointerup", { clientY: 40 }));
    });

    // jsdom 没有排版：所有行中线都是 0，"向下拖"必然落到末尾。
    const order = store.getState().session.builtinVibratoPresetOrder;
    expect(order).toHaveLength(12);
    expect(order).toContain("builtin.straight");
    expect(order[order.length - 1]).toBe("builtin.straight");
});

/*
 * 有未保存改动时切换预设：改动写回库之后，**选中必须真的切过去**。
 *
 * 【为什么值得测】切走前的落盘会让 `resolved.all` 变化，而"打开时播种草稿"的 effect
 * 依赖它 —— 不设闸的话它会把草稿重新播种回当前活动预设，用户点了另一条却被弹回去，
 * 只能再点一次。
 */
test("有未保存改动时切换预设：改动写回库，且选中确实切过去了", async () => {
    const a = sanitizeVibratoPreset({ id: "custom_sw_a", name: "Switch A", depthCents: 30 });
    const b = sanitizeVibratoPreset({ id: "custom_sw_b", name: "Switch B", depthCents: 40 });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(a));
        store.dispatch(upsertVibratoPreset(b));
        store.dispatch(setActiveVibratoPreset(a.id));
    });

    // 在 A 上制造未保存改动：点「手绘…」把波形换成表。
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    expect(drawButton, "手绘入口应已渲染").toBeTruthy();
    await act(async () => {
        drawButton!.click();
    });

    const bRow = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((row) =>
        row.textContent?.includes("Switch B"),
    );
    expect(bRow, "B 行应已渲染").toBeTruthy();
    await act(async () => {
        bRow!.click();
    });

    // 改动已写回库。
    expect(
        store.getState().session.vibratoPresets.find((preset) => preset.id === "custom_sw_a")?.cycle
            .kind,
    ).toBe("table");

    // 选中的确实是 B，而不是被播种逻辑弹回 A。
    const selected = [...document.querySelectorAll<HTMLElement>('[role="option"]')].filter(
        (row) => row.getAttribute("aria-selected") === "true",
    );
    expect(selected).toHaveLength(1);
    expect(selected[0].textContent ?? "").toContain("Switch B");
});

/*
 * 手绘编辑器的两个复位按钮：回到进入前的波形，或回到下拉框里当前的形状。
 */
test("手绘编辑器提供两个复位按钮：旧形状 + 当前形状", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_reset",
        name: "Reset Me",
        cycle: { kind: "shape", shape: "triangle", skew: 0.5 },
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    await act(async () => {
        drawButton!.click();
    });

    const labels = [...document.querySelectorAll("button")].map(
        (button) => button.textContent?.trim() ?? "",
    );
    expect(labels).toContain("Reset to previous");
    // 进入手绘时来自三角波，因此"当前形状"就是三角 —— 按钮文案要跟着它，而不是
    // 退化成固定的正弦。
    expect(labels).toContain("Reset to Triangle");
});

/*
 * 右键菜单重命名：输入框覆盖在列表里的名字上，Enter 提交、Esc 取消。
 */
test("右键菜单重命名：输入框覆盖在名字上，Enter 提交", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_rn", name: "Old Name", depthCents: 30 });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const row = document.querySelector<HTMLElement>('[data-preset-row="custom_rn"]');
    expect(row, "用户预设行应已渲染").toBeTruthy();
    await act(async () => {
        row!
            .querySelector('[role="option"]')!
            .dispatchEvent(
                new MouseEvent("contextmenu", { bubbles: true, clientX: 5, clientY: 5 }),
            );
    });

    const renameItem = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Rename",
    );
    expect(renameItem, "重命名菜单项应已渲染").toBeTruthy();
    await act(async () => {
        renameItem!.click();
    });

    const input = document.querySelector<HTMLInputElement>('input[aria-label="Rename"]');
    expect(input, "内联输入框应已渲染").toBeTruthy();
    expect(input!.value).toBe("Old Name");
    /*
     * 宽度契约：`<input>` 默认 `size=20`（固有宽度约 170px），比列表列还宽，会把整行
     * 撑出去、连累整个列表横向位移。压到 1 之后它只按 flex 填满行内剩余空间。
     */
    expect(input!.getAttribute("size")).toBe("1");

    // 受控输入：用原生 setter 写值再派发 input 事件，React 才收得到。
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(input!, "New Name");
        input!.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
        input!.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
    });

    expect(
        store.getState().session.vibratoPresets.find((preset) => preset.id === "custom_rn")?.name,
    ).toBe("New Name");
    expect(document.querySelector('input[aria-label="Rename"]')).toBeNull();
});

test("右键菜单重命名：Esc 取消，名字不变", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_rn2", name: "Keep Me", depthCents: 30 });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const row = document.querySelector<HTMLElement>('[data-preset-row="custom_rn2"]');
    await act(async () => {
        row!
            .querySelector('[role="option"]')!
            .dispatchEvent(
                new MouseEvent("contextmenu", { bubbles: true, clientX: 5, clientY: 5 }),
            );
    });
    const renameItem = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Rename",
    );
    await act(async () => {
        renameItem!.click();
    });

    const input = document.querySelector<HTMLInputElement>('input[aria-label="Rename"]')!;
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(input, "Discarded");
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
        input.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });

    expect(document.querySelector('input[aria-label="Rename"]')).toBeNull();
    expect(
        store.getState().session.vibratoPresets.find((preset) => preset.id === "custom_rn2")?.name,
    ).toBe("Keep Me");
});

/*
 * 复制为自定义的预填名。
 *
 * 【为什么值得测】连点两次曾得到 `名字 2 2`、再点变 `名字 2 2 2` —— 编号越叠越长，
 * 读起来像名字本身的一部分。系统预设的 `name` 字段还是空的（名字走词条），不按显示名
 * 预填的话副本会没有名字。
 */
test("复制为自定义：按显示名预填编号，且不会叠成「2 2」", async () => {
    const store = await mountDialog();

    const duplicateButton = () =>
        [...document.querySelectorAll("button")].find(
            (button) => button.textContent?.trim() === "Duplicate as mine",
        );

    await act(async () => {
        duplicateButton()!.click();
    });
    expect(store.getState().session.vibratoPresets.map((preset) => preset.name)).toEqual([
        "Straight 2",
    ]);

    // 第二次复制的是刚生成的 "Straight 2"：剥掉编号后应得到 "Straight 3"。
    await act(async () => {
        duplicateButton()!.click();
    });
    expect(store.getState().session.vibratoPresets.map((preset) => preset.name)).toEqual([
        "Straight 2",
        "Straight 3",
    ]);
});

/*
 * 手绘中偏斜滑块仍要能调。
 *
 * 【为什么值得测】手绘状态下"当前形状"存在 `handDraw` 里而不是草稿里；滑块若只写草稿、
 * 显示却读 `handDraw`，拖动就会立刻弹回原位 —— 表现为"拖了不动"。
 */
test("手绘中偏斜滑块仍可调整", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_skew",
        name: "Skew Me",
        cycle: { kind: "shape", shape: "triangle", skew: 0.5 },
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    await act(async () => {
        drawButton!.click();
    });

    // 偏斜是波形分区里的第一个滑块（Radix 把 aria-label 挂在 Root 上，这里按顺序取）。
    const thumb = document.querySelectorAll<HTMLElement>('[role="slider"]')[0];
    expect(thumb, "偏斜滑块应已渲染").toBeTruthy();
    // 偏斜读数是第一个 `.hs-type-mono`：键盘步进 +1，应当从 50% 变成 51%。
    // 修复前它写的是草稿里的形状、显示的却是 `handDraw.skew`，拖了会弹回 50%。
    const readouts = () =>
        [...document.querySelectorAll(".hs-type-mono")].map((el) => el.textContent);
    expect(readouts()[0]).toBe("50%");

    await act(async () => {
        thumb!.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }));
    });

    expect(readouts()[0]).toBe("51%");
});

/*
 * 直线预设的读数应为 "±0 分"，而不是被保底的 "±1"。
 */
test("完全平直的预设读数显示 ±0", async () => {
    // 默认活动预设就是「直线」（深度 0）。
    await mountDialog();
    const text = document.body.textContent ?? "";
    expect(text).toContain("±0 cents");
    expect(text).not.toContain("±1 cents");
});

/*
 * 骰子按钮（换一种抖动图案）必须真的换掉预览。
 *
 * 【为什么值得测】它改的是预设的 `seed` 字段，而渲染曾经另取种子并兜底为 0 ——
 * 按钮点了、字段也确实变了，画出来的波形却一模一样，用户看到的就是"完全不起作用"。
 * 这里断言"点一下，预览的幅度读数就变"，把整条链路（按钮 → 草稿 → 预览采样）钉住。
 */
test("骰子按钮换抖动图案：预览随之改变", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_dice",
        name: "Dice",
        depthCents: 40,
        irregularity: 60,
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    const peakReadout = () => (document.body.textContent ?? "").match(/±[\d.]+ cents/)?.[0] ?? "";
    const before = peakReadout();
    expect(before, "预览的幅度读数应已渲染").toMatch(/^±[\d.]+ cents$/);

    // 固定骰子结果，断言才是确定的（种子 = floor(0.5 × 100000)）。
    const randomSpy = vi.spyOn(Math, "random").mockReturnValue(0.5);
    try {
        const dice = document.querySelector<HTMLButtonElement>(
            'button[aria-label="Roll a new wobble pattern"]',
        );
        expect(dice, "骰子按钮应已渲染").toBeTruthy();
        await act(async () => {
            dice!.click();
        });
    } finally {
        randomSpy.mockRestore();
    }

    expect(peakReadout(), "换了抖动图案，预览必须跟着变").not.toBe(before);
});

/*
 * 不规则度为 0 时没有图案可换（噪声被整个乘掉），按钮应当停用**并说明原因**，
 * 而不是让用户点了半天看不出变化 —— 新建的预设默认就是 0，很容易撞上。
 */
test("不规则度为 0 时骰子按钮停用并说明原因", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_no_irr",
        name: "No Wobble",
        depthCents: 40,
    });
    expect(custom.irregularity).toBe(0);
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    expect(document.querySelector('button[aria-label="Roll a new wobble pattern"]')).toBeNull();
    const dice = document.querySelector<HTMLButtonElement>(
        'button[aria-label="Set the irregularity above 0 to roll a wobble pattern"]',
    );
    expect(dice, "应改用说明性 tooltip").toBeTruthy();
    expect(dice!.disabled, "没有图案可换时应停用").toBe(true);
});

/*
 * 删除正在编辑的预设：草稿要切到"迁移后的活动预设"，而不是掉进"未选择"的空状态。
 *
 * 【为什么值得测】原来删除时把草稿置空，编辑器于是显示"还没有自定义预设"—— 而库里
 * 明明还有一堆。活动 id 由 reducer 迁移到滑进同位置的预设上，草稿必须跟着走。
 */
test("删除正在编辑的预设：草稿切到迁移后的预设", async () => {
    const a = sanitizeVibratoPreset({ id: "custom_del_a", name: "Del A" });
    const b = sanitizeVibratoPreset({ id: "custom_del_b", name: "Del B" });
    const store = await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(a));
        store.dispatch(upsertVibratoPreset(b));
        store.dispatch(setActiveVibratoPreset(b.id));
    });

    const deleteButtons = () =>
        [...document.querySelectorAll("button")].filter(
            (button) => button.textContent?.trim() === "Delete",
        );
    // 页脚的「删除」→ 二次确认里的「删除」。
    await act(async () => {
        deleteButtons()[0].click();
    });
    await act(async () => {
        deleteButtons().at(-1)!.click();
    });

    expect(store.getState().session.vibratoPresets.map((preset) => preset.id)).toEqual([
        "custom_del_a",
    ]);

    // 草稿切到了剩下的那条（活动 id 迁移到滑进同位置的预设）。
    const selected = [...document.querySelectorAll<HTMLElement>('[role="option"]')].filter(
        (row) => row.getAttribute("aria-selected") === "true",
    );
    expect(selected).toHaveLength(1);
    expect(selected[0].textContent ?? "").toContain("Del A");
    expect(document.body.textContent ?? "").not.toContain("No custom presets yet");
});
