// @vitest-environment jsdom
/*
 * 手绘周期编辑器的右键整体变换契约。
 *
 * 【为什么必须有】这根手势有三处只有真跑起来才会暴露的错法，而且都不会抛错：
 *
 * 1. **右键把原生菜单带出来**。各平台 `contextmenu` 的触发时机不同（Windows 在
 *    `pointerup`，X11/macOS 在 `pointerdown`），只在手势成立后才拦会漏掉"纯右键
 *    单击"那一下 —— 而用户正是想拖一下才按的右键。
 * 2. **右键分支吞掉左键画笔**。两种手势共用一块画布，仲裁写错就是"画不上了"。
 * 3. **对上一帧结果继续变换**（而不是对按下时的快照）。旋转是线性插值，反复插值
 *    会让曲线越拖越平 —— 手感上表现为"拖几下形状就没了"。
 *
 * 换算本身的算术在 `vibratoCycleEdit.test.ts` 里钉；这里只钉"事件进得来、
 * 方向对、不写空草稿、菜单被吃掉"。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { Keybinding } from "../../../features/keybindings/types";
import { makeSineTable } from "../../../features/vibrato/vibratoCycle";
import { I18nProvider } from "../../../i18n/I18nProvider";
import { AppThemeProvider } from "../../../theme/AppThemeProvider";
import { rotateCycleTable } from "./vibratoCycleEdit";
import { VibratoCycleEditor } from "./VibratoCycleEditor";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver；画布的绘制 effect 会构造它。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;
// jsdom 的 2D context 返回 null，绘制提前返回 —— 本文件只关心手势，不关心像素。
(HTMLCanvasElement.prototype.getContext as unknown) ??= () => null;

const ARIA = "周期画布";
/** 画布几何（jsdom 没有排版，得手工给）。宽 640 / 64 格 → 每格 10px。 */
const WIDTH = 640;
const HEIGHT = 120;
/** 峰在 16 格处的正弦表：旋转位移一眼可验。 */
const SINE = makeSineTable(64);
/** 半幅正弦：垂直缩放不会撞上 ±1 的钳制。 */
const HALF_SINE = SINE.map((value) => value * 0.5);

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

async function mountEditor(
    overrides: { table?: number[]; disabled?: boolean; fineAdjustKb?: Keybinding } = {},
) {
    const onChange = vi.fn();
    await act(async () => {
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <VibratoCycleEditor
                        table={overrides.table ?? SINE}
                        disabled={overrides.disabled}
                        fineAdjustKb={overrides.fineAdjustKb}
                        onChange={onChange}
                        smoothLabel="平滑"
                        resetLabel="复位"
                        ariaLabel={ARIA}
                        readoutLabels={{ phase: "相位", scale: "幅度" }}
                        hint="左键绘制 · 右键整体旋转 / 缩放"
                    />
                </I18nProvider>
            </AppThemeProvider>,
        );
    });

    const canvas = host.querySelector(`canvas[aria-label="${ARIA}"]`) as HTMLCanvasElement;
    const container = canvas.parentElement as HTMLDivElement;
    container.getBoundingClientRect = () =>
        ({
            left: 0,
            top: 0,
            right: WIDTH,
            bottom: HEIGHT,
            width: WIDTH,
            height: HEIGHT,
            x: 0,
            y: 0,
            toJSON: () => ({}),
        }) as DOMRect;
    Object.defineProperty(container, "clientWidth", { value: WIDTH, configurable: true });

    return { onChange, canvas, container };
}

/** 指针事件：`pointerId` 在部分环境不可得，组件对缺省是容忍的（见其注释）。 */
function pointerEvent(type: string, init: PointerEventInit): PointerEvent {
    return new PointerEvent(type, { bubbles: true, cancelable: true, pointerId: 7, ...init });
}

async function dispatch(canvas: Element, event: Event) {
    await act(async () => {
        canvas.dispatchEvent(event);
    });
}

async function drag(
    canvas: Element,
    options: {
        from: { x: number; y: number };
        to: { x: number; y: number };
        button?: number;
        modifiers?: PointerEventInit;
    },
) {
    const { from, to, button = 2, modifiers = {} } = options;
    await dispatch(
        canvas,
        pointerEvent("pointerdown", {
            button,
            clientX: from.x,
            clientY: from.y,
            ...modifiers,
        }),
    );
    await dispatch(
        canvas,
        pointerEvent("pointermove", { clientX: to.x, clientY: to.y, ...modifiers }),
    );
    await dispatch(
        canvas,
        pointerEvent("pointerup", { button, clientX: to.x, clientY: to.y, ...modifiers }),
    );
}

/** 峰值所在的格号（旋转断言用 —— 比逐格比较更能说明"图像往哪儿动了"）。 */
function argmax(values: number[]): number {
    let best = 0;
    for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
    return best;
}

test("右键水平拖拽：位移按「一个画布宽 = 一个整周期」旋转", async () => {
    const { onChange, canvas } = await mountEditor();
    // 右移 160px = 1/4 周期 → 16 格。
    await drag(canvas, { from: { x: 100, y: 60 }, to: { x: 260, y: 60 } });

    const next = onChange.mock.calls.at(-1)?.[0] as number[];
    expect(argmax(next)).toBe(32);
    // 原来的峰位（16 格）转过四分之一周期后落到零交叉上。
    expect(next[16]).toBeCloseTo(0, 6);
});

test("右键向左拖拽把图像向左转（方向不反）", async () => {
    const { onChange, canvas } = await mountEditor();
    await drag(canvas, { from: { x: 260, y: 60 }, to: { x: 100, y: 60 } });

    const next = onChange.mock.calls.at(-1)?.[0] as number[];
    expect(argmax(next)).toBe(0); // 16 - 16 = 0
});

test("右键向上拖拽放大整体幅度（形状不变比）", async () => {
    const { onChange, canvas } = await mountEditor({ table: HALF_SINE });
    // 向上拖满半个画布高 → 2^(60/120) = √2 倍。
    await drag(canvas, { from: { x: 320, y: 90 }, to: { x: 320, y: 30 } });

    const next = onChange.mock.calls.at(-1)?.[0] as number[];
    const peak = Math.max(...next.map(Math.abs));
    expect(peak).toBeCloseTo(0.5 * Math.SQRT2, 6);
    // 零交叉仍在原处：垂直缩放绕零线，不搬动相位。
    expect(next[0]).toBeCloseTo(0, 6);
});

test("右键单击（无位移）不写草稿", async () => {
    const { onChange, canvas } = await mountEditor();
    await dispatch(canvas, pointerEvent("pointerdown", { button: 2, clientX: 100, clientY: 60 }));
    await dispatch(canvas, pointerEvent("pointerup", { button: 2, clientX: 102, clientY: 61 }));
    expect(onChange).not.toHaveBeenCalled();
});

test("整段手势基于按下时的快照重算，不是对上一帧结果继续变换", async () => {
    const { onChange, canvas } = await mountEditor();
    await dispatch(canvas, pointerEvent("pointerdown", { button: 2, clientX: 100, clientY: 60 }));
    // 先向右偏 6.5 格、再向左偏 6.5 格。分数位移要走线性插值：对**上一帧结果**继续
    // 插值会留下"插值两次"的痕迹（曲线被磨平），对按下时的快照重算则与一次性旋转
    // 逐位相同。这正是"拖几下形状就没了"那类缺陷的分水岭。
    await dispatch(canvas, pointerEvent("pointermove", { clientX: 165, clientY: 60 }));
    await dispatch(canvas, pointerEvent("pointermove", { clientX: 35, clientY: 60 }));

    const next = onChange.mock.calls.at(-1)?.[0] as number[];
    const oneShot = rotateCycleTable(SINE, -6.5);
    const accumulated = rotateCycleTable(rotateCycleTable(SINE, 6.5), -6.5);
    // 前提：两条路确实不同，否则这个测试什么也没证明（"插值两次"确实磨平了曲线）。
    const maxDiff = Math.max(...SINE.map((_, i) => Math.abs(oneShot[i] - accumulated[i])));
    expect(maxDiff).toBeGreaterThan(0.1);
    for (let i = 0; i < SINE.length; i += 1) expect(next[i]).toBeCloseTo(oneShot[i], 9);
});

test("原生右键菜单被吃掉", async () => {
    const { canvas } = await mountEditor();
    const event = new MouseEvent("contextmenu", { bubbles: true, cancelable: true });
    await dispatch(canvas, event);
    expect(event.defaultPrevented).toBe(true);
});

test("左键仍然绘制（右键分支不吞掉画笔）", async () => {
    const { onChange, canvas } = await mountEditor({ table: new Array(64).fill(0) });
    // x = 5px → 第 0 格；y = 4px = 纵轴内缩处 → 恰好是值 1。
    await dispatch(canvas, pointerEvent("pointerdown", { button: 0, clientX: 5, clientY: 4 }));

    expect(onChange).toHaveBeenCalledTimes(1);
    const next = onChange.mock.calls[0][0] as number[];
    expect(next[0]).toBeCloseTo(1, 6);
});

test("右键拖拽期间显示读数，松手后消失", async () => {
    const { canvas } = await mountEditor();
    await dispatch(canvas, pointerEvent("pointerdown", { button: 2, clientX: 100, clientY: 60 }));
    expect(host.querySelector('[data-testid="vibrato-cycle-readout"]')).toBeNull();

    await dispatch(canvas, pointerEvent("pointermove", { clientX: 260, clientY: 60 }));
    const readout = host.querySelector('[data-testid="vibrato-cycle-readout"]');
    expect(readout?.textContent).toBe("相位 25% · 幅度 100%");

    await dispatch(canvas, pointerEvent("pointerup", { button: 2, clientX: 260, clientY: 60 }));
    expect(host.querySelector('[data-testid="vibrato-cycle-readout"]')).toBeNull();
});

test("精细调整修饰键把位移缩到 1/5", async () => {
    const fineAdjustKb: Keybinding = { key: "shift", shift: true, modifierOnly: true };
    const { onChange, canvas } = await mountEditor({ fineAdjustKb });
    // 同样 160px：未精细是 16 格，精细后 160 × 0.2 = 32px → 3.2 格 → 峰到第 19 格。
    await drag(canvas, {
        from: { x: 100, y: 60 },
        to: { x: 260, y: 60 },
        modifiers: { shiftKey: true },
    });

    const next = onChange.mock.calls.at(-1)?.[0] as number[];
    expect(argmax(next)).toBe(19);
});

test("disabled（系统预设）下右键拖拽不改草稿", async () => {
    const { onChange, canvas } = await mountEditor({ disabled: true });
    await drag(canvas, { from: { x: 100, y: 60 }, to: { x: 260, y: 60 } });
    expect(onChange).not.toHaveBeenCalled();
});

test("提示行常驻渲染", async () => {
    await mountEditor();
    expect(host.textContent).toContain("左键绘制");
});
