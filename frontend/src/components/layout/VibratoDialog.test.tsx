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
import { SYSTEM_VIBRATO_PRESETS } from "../../features/vibrato/systemPresets";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { VibratoDialog } from "./VibratoDialog";

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

/**
 * 选区宿主（可选）。
 *
 * 给了它，窗口就多出「套用到选区」页签与「应用」「提取」两个动作 —— 与参数编辑器
 * 里那条路径一致；不给就是菜单栏那条纯预设库路径。
 */
interface ApplyTargetOptions {
    loadOriginal?: () => Promise<{ values: number[]; framePeriodMs: number } | null>;
    onApply?: (preset: VibratoPreset) => void;
    onExtract?: () => Promise<VibratoPreset | null>;
}

interface MountOptions {
    /** 打开时编辑哪一条（提取出来的那条走这里）。 */
    initialPresetId?: string;
    /** 选区宿主；省略 = 纯预设库。 */
    applyTarget?: ApplyTargetOptions;
    editParam?: string;
    paramRange?: { min: number; max: number };
}

async function mountDialog(
    prepare?: (store: ReturnType<typeof configureStore>) => void,
    onOpenChange: (open: boolean) => void = () => undefined,
    options: MountOptions = {},
) {
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    prepare?.(store);
    const target = options.applyTarget;
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <VibratoDialog
                            open
                            onOpenChange={onOpenChange}
                            editParam={options.editParam ?? "pitch"}
                            paramRange={options.paramRange}
                            initialPresetId={options.initialPresetId}
                            applyTarget={
                                target
                                    ? {
                                          loadOriginal:
                                              target.loadOriginal ??
                                              (() =>
                                                  Promise.resolve({
                                                      values: ORIGINAL,
                                                      framePeriodMs: 5,
                                                  })),
                                          onApply: target.onApply ?? (() => undefined),
                                          onExtract: target.onExtract,
                                      }
                                    : undefined
                            }
                        />
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

/** 页脚 / 页签上的按钮：按文案找。 */
function findButton(text: string): HTMLButtonElement | undefined {
    return [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === text,
    ) as HTMLButtonElement | undefined;
}

/** 点一个按钮（找不到就抛，避免"点了没反应"被误当成通过）。 */
async function clickButton(text: string): Promise<void> {
    const button = findButton(text);
    expect(button, `按钮「${text}」应已渲染`).toBeTruthy();
    await act(async () => {
        button!.click();
    });
}

/** 往数字输入框里打字（`AppNumberField` 的实时回调每次输入都会触发）。 */
async function typeNumber(ariaLabel: string, value: number): Promise<void> {
    const input = document.querySelector<HTMLInputElement>(`input[aria-label="${ariaLabel}"]`);
    expect(input, `输入框「${ariaLabel}」应已渲染`).toBeTruthy();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(input, String(value));
        input!.dispatchEvent(new Event("input", { bubbles: true }));
    });
}

/** 库里当前的自定义预设（不含系统预设 —— 它们不在 store 里）。 */
function userPresets(store: Awaited<ReturnType<typeof mountDialog>>) {
    return store.getState().session.vibratoPresets;
}

/**
 * 一段真实的音高曲线：C4 附近 ±0.5 个半音。
 *
 * 【为什么从 60 起而不是从 0 起】音高参数里 **0 = 未检测到音高**（见
 * `vibratoPitch.ts`）。用 `sin(i/8) * 0.5` 当音高夹具既不是真实音高，i=0 处又恰好
 * 落在 0 上 —— 那是在测一个不存在的情形，新契约还会把那一帧画成断口。
 */
const ORIGINAL = Array.from({ length: 64 }, (_, i) => 60 + Math.sin(i / 8) * 0.5);

/** 选区窗口时长（ms）：`ORIGINAL` 的 64 帧 × 5ms。 */
const SELECTION_WINDOW_MS = (ORIGINAL.length - 1) * 5;
/** 预设波形页签的窗口时长（ms）：与 `buildVibratoPreview` 的默认几何一致。 */
const PRESET_WINDOW_MS = 320 * 5 - 5;

/** 测试里给画布铺的宽度（CSS 像素）—— 与 `withCanvasLayout` 配套。 */
const CANVAS_WIDTH = 400;

/** 渐入 / 渐出手柄在画布上的横向位置（与 `handleLayoutFor` 同一套换算）。 */
function handleXFor(ms: number, windowMs: number): number {
    return Math.min(1, Math.max(0, ms / windowMs)) * CANVAS_WIDTH;
}

/**
 * 让预览画布在 jsdom 里"有排版"。
 *
 * 【为什么需要它】jsdom 没有排版引擎：`clientWidth` 恒为 0、`getContext("2d")` 返回
 * null，而画布的几何（宽度、cents/px）正是**绘制时**算出来存进 `geometryRef` 的。
 * 宽度为 0 时命中测试一律判成"主体"，渐入 / 渐出手柄根本抓不到。
 *
 * 【为什么值得铺这层脚手架】"手柄按哪个时间轴换算"正是套用页签这次改动最容易错
 * 的地方：它的时间轴是**选区真实帧数**，预设波形是固定窗口。只有真的抓到那个手柄，
 * 才能证明两个页签各自用了自己的窗口。
 *
 * @returns 还原函数 —— 用例结束必须调用，别把全局原型留给下一条用例。
 */
function withCanvasLayout(width: number): () => void {
    const canvasProto = HTMLCanvasElement.prototype as unknown as { getContext: unknown };
    const originalGetContext = canvasProto.getContext;
    // 绘制只写不读，所以"任何属性都是空函数"的代理足够当 2D 上下文。
    const fakeCtx = new Proxy(
        {},
        {
            get: (_target, prop) => (prop === "canvas" ? undefined : () => undefined),
            set: () => true,
        },
    );
    canvasProto.getContext = () => fakeCtx;
    const originalClientWidth = Object.getOwnPropertyDescriptor(
        HTMLElement.prototype,
        "clientWidth",
    );
    Object.defineProperty(HTMLElement.prototype, "clientWidth", {
        configurable: true,
        get: () => width,
    });
    return () => {
        canvasProto.getContext = originalGetContext;
        if (originalClientWidth) {
            Object.defineProperty(HTMLElement.prototype, "clientWidth", originalClientWidth);
        }
    };
}

/** 预览右上角的「摆放方式」下拉（表单里没有第二个，因此这个选择器是唯一的）。 */
function findBaselineTrigger(): HTMLButtonElement | null {
    return document.querySelector<HTMLButtonElement>('button[aria-label="Placement"]');
}

/** 画布容器（手势回调挂在它上面，因此事件要派发到它而不是 window）。 */
function previewInteractive(): HTMLElement {
    const container = document.querySelector<HTMLElement>(
        '[data-testid="vibrato-preview-interactive"]',
    );
    expect(container, "预览交互层应已渲染").toBeTruthy();
    // jsdom 没有指针捕获 API（画布按下时会调）。
    container!.setPointerCapture = () => undefined;
    container!.releasePointerCapture = () => undefined;
    container!.hasPointerCapture = () => false;
    return container!;
}

/** 从 `fromX` 按住并水平拖 `dx` 像素（一次完整手势）。 */
async function dragHorizontally(fromX: number, dx: number): Promise<void> {
    const container = previewInteractive();
    await act(async () => {
        container.dispatchEvent(
            new PointerEvent("pointerdown", {
                bubbles: true,
                button: 0,
                clientX: fromX,
                clientY: 0,
            }),
        );
    });
    await act(async () => {
        container.dispatchEvent(
            new PointerEvent("pointermove", {
                bubbles: true,
                clientX: fromX + dx,
                clientY: 0,
            }),
        );
    });
    await act(async () => {
        container.dispatchEvent(new PointerEvent("pointerup", { bubbles: true }));
    });
}

/** 读一个数字输入框当前显示的值。 */
function numberFieldValue(ariaLabel: string): number {
    return Number(
        document.querySelector<HTMLInputElement>(`input[aria-label="${ariaLabel}"]`)?.value ??
            "NaN",
    );
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
    const source = readFileSync("src/components/layout/VibratoDialog.tsx", "utf8");
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
 * 【契约】预览可拖（画手柄、接受手势）—— 对系统预设**也一样**：拖的是本地草稿，
 * 落盘时变成一份自定义副本（见"保存 = 另存为副本"那几条），出厂预设本身不会被碰。
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

test("系统预设：预览画布同样可编辑（改的是本地草稿）", async () => {
    await mountDialog();
    // 默认活动预设是系统预设；它一样能拖 —— 拖出来的是草稿，不是出厂参数。
    expect(document.querySelector('[data-testid="vibrato-preview-interactive"]')).toBeTruthy();
});

test("套用页签：预览同样可拖（与预设波形共用一套手势）", async () => {
    await mountDialog(undefined, () => undefined, { applyTarget: {} });
    // 有选区数据时套用页签画的是真实曲线，它同样接受手柄与主体拖拽。
    expect(document.querySelector('[data-testid="vibrato-preview-interactive"]')).toBeTruthy();
});

/*
 * 「摆放方式」就地可选：套用页签的波形右上角一个下拉。
 *
 * 【为什么它该在波形旁边】它决定的是"颤音挂在素材的哪条线上" —— 对「添加颤音」来说
 * 是要**边看边定**的参数（换一下，颤音就从"保持歌手原曲线"变成"拉成一条直线"），
 * 让用户去右下角表单里翻太远。
 *
 * 它只出现在套用页：预设波形页没有素材，也就无从判断该挂在哪条线上。
 */
test("套用页签：波形右上角有「摆放方式」，且只在这一页出现", async () => {
    await mountDialog(undefined, () => undefined, { applyTarget: {} });

    const trigger = findBaselineTrigger();
    expect(trigger, "套用页签应当有摆放方式").toBeTruthy();
    // 显示**设置**里当前的值（默认「起点 → 终点」= 抽取成设置之前的既有行为）。
    expect(trigger!.textContent, "应回显设置里的值").toContain("Start → End");

    await clickButton("Preset waveform");
    expect(findBaselineTrigger(), "预设波形页不该有摆放方式").toBeFalsy();
});

/*
 * 摆放方式**不能铺满整行**。
 *
 * 【为什么单独一条】`AppSelect` 默认 `fullWidth`（表单里那样是对的），放进预览的
 * 头部一行就会把同一行的标题挤成一字一行、自己撑到整行宽 —— 实机截图里那一列竖排的
 * "围绕什么摆动"就是这么来的。jsdom 没有排版，量不出"挤没挤"，但可以钉住那个根因：
 * 这个下拉不得带 `w-full`，且要有定宽。
 */
test("摆放方式不抢整行宽度（否则标题会被挤成竖排）", async () => {
    await mountDialog(undefined, () => undefined, { applyTarget: {} });

    const trigger = findBaselineTrigger();
    expect(trigger, "套用页签应当有摆放方式").toBeTruthy();
    expect(
        trigger!.className,
        "下拉不得铺满整行：它会把自己撑到整行宽、并把旁边的标题挤成一字一行",
    ).not.toContain("w-full");
    expect(trigger!.style.minWidth, "定宽避免切选项时整行跳动").toBe("150px");
});

test("管理预设那一面：没有「摆放方式」（那是添加颤音的专属）", async () => {
    await mountDialog();
    expect(findBaselineTrigger()).toBeFalsy();
});

/*
 * 摆放方式就地改：写进**设置**（不是草稿）。
 *
 * 【为什么这条测试变了】它原来是"改草稿、应用时带出去"。抽离成设置之后，它既不该
 * 随预设走，也不该进预设载荷 —— 提交侧（`addVibrato`）自己从设置里读。因此这里断言
 * 两件事：设置被改了（下次打开还是它），而**预设载荷里没有** baseline 这个字段。
 */
test("摆放方式就地改：写进设置，且不进预设载荷", async () => {
    const onApply = vi.fn();
    const store = await mountDialog(undefined, () => undefined, { applyTarget: { onApply } });

    const trigger = findBaselineTrigger()!;
    // Radix 为表单兼容渲染一个隐藏的原生 select，它就是"改这个受控值"的入口
    // （与 `Select.test.tsx` 同一手法）。
    const native = trigger.parentElement?.querySelector("select");
    expect(native, "隐藏的原生 select 应存在").toBeTruthy();
    const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(native, "holdStart");
        native!.dispatchEvent(new Event("change", { bubbles: true }));
    });

    expect(store.getState().session.vibratoBaseline, "应当写进设置").toBe("holdStart");
    expect(trigger.textContent, "选完立刻回显").toContain("Hold start");

    await clickButton("Apply");
    expect(onApply).toHaveBeenCalledTimes(1);
    expect("baseline" in onApply.mock.calls[0][0], "摆放方式不是预设的一部分，不该进载荷").toBe(
        false,
    );
});

/*
 * 手柄按**哪个时间轴**换算：两个页签各用各的。
 *
 * 【为什么值得单独两条】这是套用页签接入手势时最容易错的一处：手柄的横向位置与
 * 水平拖动都是"占整段时长的比例"，而两个页签的时间轴不是一回事 ——
 * 预设波形是固定的 320 帧窗口（1595ms），套用预览画的是**选区真实帧数**
 * （这里 64 帧 × 5ms = 315ms）。
 *
 * 同一对手柄、同一个 100px 位移，在两条时间轴上该改出**不同**的毫秒数：
 *
 * - 预设窗口：90 + 100/400 × 1595 ≈ 489
 * - 选区窗口：90 + 100/400 × 315  ≈ 169
 *
 * 若把窗口取错（例如套用页签沿用预设窗口），这里的数字会差 3 倍多 —— 而"手柄画在
 * 错误的位置"更糟：用户会去拖一个不在斜坡上的点。
 */
test("预设波形页签：拖渐入手柄按预设窗口换算", async () => {
    const restore = withCanvasLayout(CANVAS_WIDTH);
    try {
        const natural = SYSTEM_VIBRATO_PRESETS.find((preset) => preset.id === "builtin.natural");
        expect(natural, "出厂预设「自然」应存在").toBeTruthy();
        await mountDialog((s) => {
            s.dispatch(setActiveVibratoPreset("builtin.natural"));
        });

        await dragHorizontally(handleXFor(natural!.attackMs, PRESET_WINDOW_MS), 100);

        const expected = natural!.attackMs + (100 / CANVAS_WIDTH) * PRESET_WINDOW_MS;
        expect(
            Math.abs(numberFieldValue("Fade in") - expected),
            "渐入应按预设窗口（1595ms）换算",
        ).toBeLessThan(1.5);
    } finally {
        restore();
    }
});

test("套用页签：拖渐入手柄按**选区**窗口换算", async () => {
    const restore = withCanvasLayout(CANVAS_WIDTH);
    try {
        const natural = SYSTEM_VIBRATO_PRESETS.find((preset) => preset.id === "builtin.natural");
        expect(natural, "出厂预设「自然」应存在").toBeTruthy();
        await mountDialog(
            (s) => {
                s.dispatch(setActiveVibratoPreset("builtin.natural"));
            },
            () => undefined,
            { applyTarget: {} },
        );

        // 手柄按**选区**时长摆位（315ms），与预设页签的位置不是同一处。
        await dragHorizontally(handleXFor(natural!.attackMs, SELECTION_WINDOW_MS), 100);

        const expected = natural!.attackMs + (100 / CANVAS_WIDTH) * SELECTION_WINDOW_MS;
        expect(
            Math.abs(numberFieldValue("Fade in") - expected),
            "渐入应按选区窗口（315ms）换算",
        ).toBeLessThan(1.5);
        // 反向对照：按预设窗口算出来的值离得很远（差 3 倍多）。
        const presetWindowResult = natural!.attackMs + (100 / CANVAS_WIDTH) * PRESET_WINDOW_MS;
        expect(Math.abs(numberFieldValue("Fade in") - presetWindowResult)).toBeGreaterThan(100);
    } finally {
        restore();
    }
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

    /**
     * 等一帧。
     *
     * 【为什么必须等】预览画布把一帧内的多个 pointermove **合并**成一次提交
     * （笔 133–266Hz 的采样率不该变成同等数量的 React 渲染，与钢琴卷帘线工具
     * 的 `pendingLineEvent` 同源）。因此派发事件后要等 rAF 跑完才观察得到结果。
     */
    const nextFrame = () =>
        new Promise<void>((resolve) => {
            if (typeof requestAnimationFrame === "function") {
                requestAnimationFrame(() => resolve());
            } else {
                setTimeout(resolve, 0);
            }
        });

    const dragTo = async (clientY: number, ctrlKey: boolean) => {
        // 手势回调挂在画布容器上（React 合成事件），因此要派发到容器而不是 window。
        await act(async () => {
            container!.dispatchEvent(
                new PointerEvent("pointermove", { bubbles: true, clientY, ctrlKey }),
            );
            await nextFrame();
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

test("系统预设：手绘入口同样可用", async () => {
    await mountDialog();
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    ) as HTMLButtonElement | undefined;
    expect(drawButton, "手绘入口应已渲染").toBeTruthy();
    expect(drawButton!.disabled, "系统预设也能手绘（改的是草稿）").toBe(false);
    await act(async () => {
        drawButton!.click();
    });
    expect(document.querySelector('[data-testid="vibrato-cycle-editor"]')).toBeTruthy();
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

/*
 * 系统预设：**保存 = 另存为一份自定义副本**，改没改过都一样。
 *
 * 【为什么值得测】这是"系统预设只读"这条旧规矩的替代品：出厂预设必须永远可复原
 * （不能被覆盖），而用户又常常是"「自然」就挺好，先存一份我自己的"。两条都要满足，
 * 唯一的办法就是"改草稿、存副本"。
 *
 * 同时钉住三件事：
 * 1. 副本是一条**新**预设（新 id、`builtin: false`），库里绝不出现系统预设的 id；
 * 2. 草稿切到副本上 —— 否则再按一次保存会又生成一条（那是"每次都新建"，不是保存）；
 * 3. **没编辑过也能存**：按钮是可点的，按下去就该有结果 —— "没动过就不给存"只会让
 *    人以为按钮坏了（旧实现把它做成了空操作）。
 */
test("系统预设：未编辑也能保存为副本，草稿切到副本上", async () => {
    const factory = SYSTEM_VIBRATO_PRESETS.find((preset) => preset.id === "builtin.natural");
    expect(factory, "出厂预设「自然」应存在").toBeTruthy();

    const store = await mountDialog((s) => {
        s.dispatch(setActiveVibratoPreset("builtin.natural"));
    });
    expect(userPresets(store).length, "起手库里不该有自定义预设").toBe(0);

    await clickButton("Save");

    const created = userPresets(store);
    expect(created.length, "没编辑过也应当存出一份副本").toBe(1);
    expect(created[0].id.startsWith("builtin."), "副本必须是用户预设").toBe(false);
    expect(created[0].builtin).toBe(false);
    expect(created[0].name, "名字按显示名预填").toBe("Natural 2");
    expect(created[0].depthCents, "没改过就带着出厂参数").toBe(factory!.depthCents);

    // 再按一次保存：更新的是**同一份**副本（草稿已经切过去了），而不是又造一条。
    await clickButton("Save");
    expect(userPresets(store).length, "第二次保存不该再新建").toBe(1);
});

/*
 * 系统预设：改过再保存 —— 副本带着改动（而不是出厂值）。
 */
test("系统预设：改过再保存，副本带着改动", async () => {
    const store = await mountDialog((s) => {
        s.dispatch(setActiveVibratoPreset("builtin.natural"));
    });

    await typeNumber("Depth", 55);
    await clickButton("Save");

    const saved = userPresets(store);
    expect(saved.length).toBe(1);
    expect(saved[0].depthCents, "副本带着刚才的改动").toBe(55);
    expect(saved[0].name).toBe("Natural 2");
    expect(saved[0].id.startsWith("builtin.")).toBe(false);
});

/*
 * 系统预设：改完直接切走 —— 改动落成副本，而不是被丢掉或覆盖出厂预设。
 *
 * 与"切换预设时把未保存的改动写回库"是同一条规矩（离开这条预设 = 落盘），只是系统
 * 预设落盘的去处是副本。丢掉才是真的糟：界面刚刚还显示着用户改过的值。
 */
test("系统预设：改完切走时，改动落成副本", async () => {
    const store = await mountDialog((s) => {
        s.dispatch(setActiveVibratoPreset("builtin.natural"));
    });

    await typeNumber("Depth", 55);

    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const straightRow = rows.find((row) => row.textContent?.includes("Straight"));
    expect(straightRow, "系统预设行应已渲染").toBeTruthy();
    await act(async () => {
        straightRow!.click();
    });

    const saved = userPresets(store);
    expect(saved.length, "改动应当落成一条副本").toBe(1);
    expect(saved[0].depthCents).toBe(55);
    expect(saved[0].id.startsWith("builtin.")).toBe(false);
});

/*
 * 系统预设 + 「添加颤音」：调完只按「应用」—— 库里不留任何东西。
 *
 * 【为什么单独一条】这正是用户要的"基于系统预设调两个参数然后应用"：草稿里的改动
 * 落到选区上，而预设库保持原样（既不覆盖出厂预设，也不生成副本 —— 生成副本是
 * 「保存」的事）。
 */
test("系统预设：只应用不保存时，库里不留痕迹", async () => {
    const onApply = vi.fn();
    const store = await mountDialog(
        (s) => {
            s.dispatch(setActiveVibratoPreset("builtin.natural"));
        },
        () => undefined,
        { applyTarget: { onApply } },
    );

    await typeNumber("Depth", 55);
    await clickButton("Apply");

    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0].id, "应用的还是系统预设那条（带改动）").toBe("builtin.natural");
    expect(onApply.mock.calls[0][0].depthCents).toBe(55);
    expect(userPresets(store).length, "只应用不该写库").toBe(0);
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
 * 偏斜滑块：参数式形状可调，手绘中必须禁用。
 *
 * 【为什么手绘中要禁用】偏斜只对**参数式形状**有效 —— `sampleCycle` 只在
 * `kind: "shape"` 分支里读它，表波形完全不看这个值。手绘（以及从选区提取）出来的
 * 表就是用户画的那条曲线本身，没有"上升段占比"可言；留着能拖会让用户以为拖了会变，
 * 实际毫无反应。
 *
 * 【为什么反面对照同样重要】只断言"手绘中禁用"会漏掉"把功能整个关掉"这种改法，
 * 所以同一条测试里先钉住三角波下它仍然可调。
 */
test("偏斜滑块：参数式形状可调，进入手绘后禁用", async () => {
    const custom = sanitizeVibratoPreset({
        id: "custom_skew",
        name: "Skew Me",
        cycle: { kind: "shape", shape: "triangle", skew: 0.5 },
    });
    await mountDialog((store) => {
        store.dispatch(upsertVibratoPreset(custom));
        store.dispatch(setActiveVibratoPreset(custom.id));
    });

    // 偏斜是波形分区里的第一个滑块（Radix 把 aria-label 挂在 Root 上，这里按顺序取）。
    const skewThumb = () => document.querySelectorAll<HTMLElement>('[role="slider"]')[0];
    // 偏斜读数是第一个 `.hs-type-mono`。
    const skewReadout = () =>
        [...document.querySelectorAll(".hs-type-mono")].map((el) => el.textContent)[0];
    const step = async () => {
        await act(async () => {
            skewThumb().dispatchEvent(
                new KeyboardEvent("keydown", { key: "ArrowRight", bubbles: true }),
            );
        });
    };

    expect(skewReadout()).toBe("50%");
    // Radix 的禁用标记落在 thumb 上（`data-disabled` 空属性，并移出 tab 序）。
    expect(skewThumb().hasAttribute("data-disabled")).toBe(false);
    await step();
    expect(skewReadout()).toBe("51%");

    // 进入手绘：草稿的波形变成表，偏斜随之失去意义。
    const drawButton = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Draw...",
    );
    await act(async () => {
        drawButton!.click();
    });

    expect(skewThumb().hasAttribute("data-disabled")).toBe(true);
    expect(skewThumb().getAttribute("tabindex")).toBeNull();
    // 键盘步进不再改变读数 —— 是真的禁用了，而不是"能拖但没效果"。
    await step();
    expect(skewReadout()).toBe("51%");
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

/*
 * `initialPresetId`：打开时编辑哪一条。
 *
 * 【为什么它还在】提取出来的那条预设由它带进来 —— 用户刚把一条颤音提成预设，
 * 当然是要接着编辑它，而不是回到"当前使用"的那条重新找一遍。
 */
test("initialPresetId 决定打开时编辑哪一条（优先于当前活动预设）", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_seeded", name: "Seeded", depthCents: 40 });
    await mountDialog(
        (store) => {
            store.dispatch(upsertVibratoPreset(custom));
            // 活动预设**不是**要编辑的那条：窗口应当听 initialPresetId，而不是活动预设。
            store.dispatch(setActiveVibratoPreset("builtin.straight"));
        },
        () => undefined,
        { initialPresetId: custom.id },
    );

    const selected = document.querySelector('[role="option"][data-selected]');
    expect(selected?.textContent, "打开时应选中 initialPresetId 指定的那条").toContain("Seeded");
});

/*
 * 一面：没有选区宿主（「管理预设…」）—— 只有「预设波形」一页，页脚也没有
 * 「应用」「提取」，与合并前的预设管理器一致。
 *
 * 【为什么值得钉】这是"改库"那一面。多画一个点不动的页签、或给一个指向不存在选区的
 * 「应用」，都会让用户以为窗口坏了；而"改库"这件事也不该因为窗口变大了就多出一堆
 * 用不上的按钮。
 */
test("管理预设那一面：不画页签，也没有「应用」「提取」", async () => {
    await mountDialog();

    expect(findButton("Preset waveform"), "没有宿主时不该出现页签").toBeFalsy();
    expect(findButton("Applied to selection")).toBeFalsy();
    expect(findButton("Apply"), "没有宿主时不该有「应用」").toBeFalsy();
    expect(
        findButton("Create vibrato preset from selection"),
        "没有宿主时不该有「从选区提取」",
    ).toBeFalsy();
    // 落在「预设波形」页：周期估算是这一页独有的读数。
    expect(document.body.textContent ?? "", "应落在预设波形页").toContain("cycles");
    // 纯预设库路径的主动作仍是「保存」。
    expect(findButton("Save")).toBeTruthy();
});

/*
 * 另一面：有选区宿主（「添加颤音」）—— 两个页签、页脚多出「应用」与「提取」，
 * 且**直接落在套用页**（那次打开的目的就是"看套上去什么样、然后应用"）。
 *
 * 页签标签用「预设波形 / 套用到选区」把两个问题分开：一个问"这个预设摆多少"，
 * 一个问"套到这段上与原参数线差多少"。
 */
test("添加颤音那一面：两个页签 + 「应用」「提取」都在，且直接落在套用页", async () => {
    await mountDialog(undefined, () => undefined, {
        // 「提取」跟着 `onExtract` 出现：没有它就没有可提取的东西，画一个点不动的
        // 按钮不如不画（宿主总是会给 —— 见 PianoRollPanel 的 applyTarget）。
        applyTarget: { onExtract: async () => null },
    });

    expect(findButton("Preset waveform"), "应有两个页签").toBeTruthy();
    expect(findButton("Applied to selection")).toBeTruthy();
    expect(findButton("Apply"), "有宿主时应有「应用」").toBeTruthy();
    expect(findButton("Create vibrato preset from selection")).toBeTruthy();
});

/*
 * 「应用」把**当前草稿**交给编辑管线 —— 包括还没保存的微调。
 *
 * 【为什么必须测】落盘侧从"按 id 解析预设"改成"吃完整预设对象"就是为了这个：只传 id
 * 的话，用户刚拖出来的深度会被静默丢弃，听到的与得到的不是一回事。
 *
 * 同时钉住另一半：**应用不写库**。预设是"调完再定"的东西，应用只是把这份参数写进
 * 选区；顺手改掉库里的预设是这类工具最恼人的错法。
 */
/*
 * 「应用」把**当前草稿**交给编辑管线 —— 包括还没保存的微调 —— 然后关窗。
 *
 * 【为什么必须测】落盘侧从"按 id 解析预设"改成"吃完整预设对象"就是为了这个：只传 id
 * 的话，用户刚拖出来的深度会被静默丢弃，听到的与得到的不是一回事。
 *
 * 同时钉住另外两半：
 * - **应用不写库**：预设是"调完再定"的东西，应用只是把这份参数写进选区；顺手改掉
 *   库里的预设是这类工具最恼人的错法。
 * - **应用关窗**：它是「添加颤音」这次打开的终点。留着窗口等于把"完成"变成"又一次
 *   操作"，用户还得再找一次关闭。
 */
test("应用：把（含未保存微调的）完整草稿交给编辑管线、不写库、并关窗", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_apply", name: "Mine", depthCents: 40 });
    const onApply = vi.fn();
    const onOpenChange = vi.fn();
    const store = await mountDialog(
        (s) => {
            s.dispatch(upsertVibratoPreset(custom));
            s.dispatch(setActiveVibratoPreset(custom.id));
        },
        onOpenChange,
        { applyTarget: { onApply } },
    );

    // 改深度但不保存。
    const depth = document.querySelector<HTMLInputElement>('input[aria-label="Depth"]');
    expect(depth, "深度输入框应已渲染").toBeTruthy();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(depth, "77");
        depth!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    await clickButton("Apply");

    expect(onApply).toHaveBeenCalledTimes(1);
    expect(onApply.mock.calls[0][0].id).toBe("custom_apply");
    expect(onApply.mock.calls[0][0].depthCents, "本地微调必须传过去").toBe(77);
    expect(onOpenChange, "应用之后应当请求关闭窗口").toHaveBeenCalledWith(false);
    const stored = store
        .getState()
        .session.vibratoPresets.find((preset) => preset.id === "custom_apply");
    expect(stored?.depthCents, "应用不该顺手改库").toBe(40);
});

/*
 * 「从选区提取」：宿主做重活并返回入库后的那条，窗口随即选中它。
 *
 * 【为什么由宿主提取】要读参数帧（选区 + 轨道 + 参数），只有宿主够得着。窗口只负责
 * 把结果显示出来 —— 合并前提取完要开一次管理器，现在同窗，提取 = 列表里多一条并选中。
 */
test("从选区提取：成功后选中新提取的那条", async () => {
    const extracted = sanitizeVibratoPreset({
        id: "custom_extracted",
        name: "From selection",
        depthCents: 55,
    });
    const onExtract = vi.fn(async () => extracted);
    await mountDialog(undefined, () => undefined, {
        applyTarget: { onExtract },
    });

    await clickButton("Create vibrato preset from selection");

    expect(onExtract).toHaveBeenCalledTimes(1);
    /*
     * 断言的是**草稿**而不是列表里的选中行：窗口的职责正是"把草稿切到返回的那条"，
     * 入库是宿主的活儿（真实宿主在 `onExtract` 里 upsert，这里不必假装）。
     */
    const nameInput = document.querySelector<HTMLInputElement>('input[aria-label="Preset name"]');
    expect(nameInput?.value, "应选中刚提取出来的那条").toBe("From selection");
});

test("从选区提取失败：给出行内提示，且不改选中", async () => {
    const onExtract = vi.fn(async () => null);
    await mountDialog(undefined, () => undefined, {
        applyTarget: { onExtract },
    });

    await clickButton("Create vibrato preset from selection");

    expect(document.body.textContent ?? "").toContain("No clear vibrato found in the selection.");
});

/*
 * 拿到选区数据后，套用页签画在同一块画布上（不是另开一张、也不是空白）。
 *
 * 【为什么钉"同一块"】两页共用画布是"切页签不跳高、不闪"的前提；若各自渲染一张，
 * 切页签会重新挂载画布（尺寸动画重来），而且下面那条"预览不落在滚动区"的契约也会
 * 因为多出一块而含糊。
 */
test("套用页签复用同一块画布（有数据时画出来）", async () => {
    await mountDialog(undefined, () => undefined, { applyTarget: {} });

    const canvases = document.querySelectorAll("canvas[role=img]");
    expect(canvases.length, "预览画布应当只有一块").toBe(1);

    await clickButton("Preset waveform");
    expect(document.querySelectorAll("canvas[role=img]").length, "切页签不该再长出一块画布").toBe(
        1,
    );
});

/*
 * 切页签不重置草稿。
 *
 * 【为什么这是合并的关键契约】两页共用同一个草稿：用户正是在"预设波形"里拖出形状、
 * 切到"套用到选区"看它在真实素材上的样子。若切页签把草稿复位，"一边调一边看"就不成立，
 * 合并也就退化成两个窗口并排。
 */
test("切页签不重置草稿（两页共用同一份草稿）", async () => {
    const custom = sanitizeVibratoPreset({ id: "custom_tab", name: "Tabbed", depthCents: 40 });
    await mountDialog(
        (s) => {
            s.dispatch(upsertVibratoPreset(custom));
            s.dispatch(setActiveVibratoPreset(custom.id));
        },
        () => undefined,
        { applyTarget: {} },
    );

    const depth = document.querySelector<HTMLInputElement>('input[aria-label="Depth"]');
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
        setter?.call(depth, "66");
        depth!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    await clickButton("Preset waveform");

    const after = document.querySelector<HTMLInputElement>('input[aria-label="Depth"]');
    expect(Number(after?.value), "切页签后深度应当还是刚才改的值").toBe(66);
});

/*
 * 套用页签的 A/B 试听：两个按钮，文案恒定。
 *
 * 【为什么文案恒定】曾经播放态把按钮换成"停止试听"，按钮宽度随之变化、整行跟着跳。
 * 播放态改由强调色 + `aria-pressed` 表达。
 */
test("套用页签带两个 A/B 试听按钮（文案恒定）", async () => {
    await mountDialog(undefined, () => undefined, { applyTarget: {} });

    expect(findButton("Audition original")).toBeTruthy();
    expect(findButton("Audition result")).toBeTruthy();
    // 预设页签只有一个播放按钮，没有这两个。
    await clickButton("Preset waveform");
    expect(findButton("Audition original")).toBeFalsy();
});

/*
 * 取不到选区数据时：给出占位提示，而不是一块空画布。
 *
 * 空画布与"这段没有数据"在视觉上无法区分，用户会以为窗口坏了。
 */
test("取不到选区数据时显示占位提示（连读数行一起省掉）", async () => {
    await mountDialog(undefined, () => undefined, {
        applyTarget: { loadOriginal: () => Promise.resolve(null) },
    });

    expect(document.body.textContent ?? "").toContain("Select a range to preview the result.");
    /*
     * 画不出曲线时读数行整行不画：那里的 "±N cents" 与「适应」都是"对着一条曲线"
     * 才有意义的动作，留着它们只会把**预设波形**的峰值冒充成选区的幅度。
     */
    expect(document.body.textContent ?? "", "不该留下预设波形的读数").not.toContain("±");
    expect(findButton("Fit"), "没有曲线时不该留一个空转的「适应」").toBeFalsy();
});

/*
 * 取选区数据**失败**（IPC 异常）时也必须收敛到占位提示。
 *
 * 此前 `loadOriginal()` 没有 catch：`then` 永不执行，`originalLoading` 永远停在
 * true —— 页签显示一句永久的「Loading...」，同时抛出 unhandled rejection。
 */
test("取选区数据失败时落到占位提示，而不是永久载入中", async () => {
    await mountDialog(undefined, () => undefined, {
        applyTarget: {
            loadOriginal: () => Promise.reject(new Error("ipc down")),
        },
    });

    expect(document.body.textContent ?? "").toContain("Select a range to preview the result.");
    expect(document.body.textContent ?? "", "失败后不该停在载入中").not.toContain("Loading...");
});

/*
 * 音高全未检测（哨兵 0）：说明"这段没有可加颤音的音高"，而不是画一条直线。
 *
 * 【为什么单独一条】把"没数据"与"这段没有音高"混成同一句话，用户会以为是自己没选对
 * 区域 —— 而这两种情况的处置完全不同（前者去选一段，后者这段本来就没有音高）。
 */
test("音高全未检测时说明原因，而不是画一条直线", async () => {
    await mountDialog(undefined, () => undefined, {
        applyTarget: {
            loadOriginal: () =>
                Promise.resolve({ values: new Array(64).fill(0), framePeriodMs: 5 }),
        },
    });

    expect(document.body.textContent ?? "").toContain(
        "No pitch to apply vibrato to in this range.",
    );
});
