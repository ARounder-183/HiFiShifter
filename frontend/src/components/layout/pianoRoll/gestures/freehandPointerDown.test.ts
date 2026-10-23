/**
 * 自由绘制的指针手势契约。
 *
 * 【要钉死什么】这一支曾经在 `710ff764`（把指针按下拆成两条分支）里**整块丢失**：
 * 没有 `pointermove`、没有 `pointerup`、从不调用 `commitStroke`。用户看到的是
 * "按住拖拽只有按下那一刻生效"，而 3567 条测试全绿 —— 因为**没有任何一条测试驱动过
 * 完整的 pointerdown → move → up 序列**。本文件就是那条缺失的测试。
 *
 * 【为什么直接调 `handleFreehandPointerDown` 而不是 `handleCanvasPointerDown`】
 * 后者要装配 90 多个依赖（选区、取数、视口…），测它等于测整个面板。这一支的契约是
 * "拿到一次按下之后，这一笔怎么走、怎么收尾"，前置条件（`strokeRef` 已落首点、
 * `liveEditActiveRef` 已置位）由调用方负责，这里按同一形状准备好即可。
 *
 * 【为什么 rAF 用队列而不是真计时】`processStrokeEvent` 是合帧执行的；真等 rAF 会让
 * 用例依赖时序。队列版让"这一帧处理了没有"变成显式动作。
 */
// @vitest-environment jsdom
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { handleFreehandPointerDown } from "./freehandPointerDown";

type Deps = Parameters<typeof handleFreehandPointerDown>[0];

/** 待执行的帧回调（见文件头：显式 flush，不依赖真计时）。 */
const frames: Array<() => void> = [];

function flushFrames(): void {
    const pending = frames.splice(0, frames.length);
    for (const callback of pending) callback();
}

beforeEach(() => {
    frames.length = 0;
    globalThis.requestAnimationFrame = ((callback: () => void) => {
        frames.push(callback);
        return frames.length;
    }) as unknown as typeof globalThis.requestAnimationFrame;
    globalThis.cancelAnimationFrame = ((id: number) => {
        frames[id - 1] = () => {};
    }) as unknown as typeof globalThis.cancelAnimationFrame;
});

afterEach(() => {
    vi.restoreAllMocks();
});

/** 一次手势所需的全部桩；返回值直接可断言。 */
function makeDeps(overrides: Partial<Record<string, unknown>> = {}) {
    const strokeRef = {
        current: {
            mode: "draw" as "draw" | "restore",
            pointerId: 1,
            param: "pitch" as const,
            points: [{ frame: 280, value: 1 }],
        },
    };
    const liveEditActiveRef = { current: true };
    const liveEditOverrideRef = { current: null };
    const paramViewRef = { current: { startFrame: 0, edit: [0], stride: 1, framePeriodMs: 5 } };
    const vibratoStateRef = { current: null };
    const onEdgeScrolledRef = { current: null as ((clientX: number) => void) | null };
    const commitStroke = vi.fn(async (_points: unknown, _mode: string) => {});
    const applyPostStrokeSmoothing = vi.fn(async (_points: unknown, _mode: string) => {});
    const applyDenseToLiveEdit = vi.fn();
    const requestWaveformRepaint = vi.fn();
    const invalidate = vi.fn();
    const setVibratoDragCaptureActive = vi.fn();
    const onCycleDragDirection = vi.fn();
    const onPitchSnapGestureActiveChange = vi.fn();
    const setActivePointerGestureEnd = vi.fn();
    const clearActivePointerGestureEnd = vi.fn();
    const disposeFineAdjustedPointerState = vi.fn();
    const edgeScroll = { track: vi.fn(), stop: vi.fn() };
    const deps = {
        applyDenseToLiveEdit,
        applyPostStrokeSmoothing,
        clearActivePointerGestureEnd,
        commitStroke,
        createFineAdjustedPointerState: vi.fn(() => ({})),
        disposeFineAdjustedPointerState,
        dragDirection: "free",
        editParam: "pitch",
        edgeScroll,
        fp: 5,
        getFineAdjustedPointerPosition: (
            _state: unknown,
            ev: { clientX: number; clientY: number },
        ) => ({
            clientX: ev.clientX,
            clientY: ev.clientY,
        }),
        installDragDirectionKeyCycler: vi.fn(() => vi.fn()),
        invalidate,
        isDrawMode: true,
        isSnapToggleModifierHeld: () => false,
        liveEditActiveRef,
        liveEditOverrideRef,
        mode: "draw",
        onCycleDragDirection,
        onEdgeScrolledRef,
        onPitchSnapGestureActiveChange,
        paramViewRef,
        penEraserDown: false,
        pointerSec: (clientX: number) => clientX / 100,
        pointerValue: (clientY: number) => clientY / 100,
        requestWaveformRepaint,
        setActivePointerGestureEnd,
        setVibratoDragCaptureActive,
        snapDrawValue: (raw: number) => raw,
        strokeRef,
        value: 1,
        vibratoStateRef,
        ...overrides,
    };
    return {
        deps: deps as unknown as Deps,
        strokeRef,
        liveEditActiveRef,
        commitStroke,
        applyPostStrokeSmoothing,
        applyDenseToLiveEdit,
        requestWaveformRepaint,
        invalidate,
        setVibratoDragCaptureActive,
        onCycleDragDirection,
        onPitchSnapGestureActiveChange,
        setActivePointerGestureEnd,
        clearActivePointerGestureEnd,
        edgeScroll,
    };
}

function canvas(): HTMLCanvasElement {
    const element = document.createElement("canvas");
    document.body.appendChild(element);
    return element;
}

function pointerEvent(
    type: string,
    init: PointerEventInit & { button?: number; buttons?: number },
): PointerEvent {
    return new PointerEvent(type, { bubbles: true, cancelable: true, ...init });
}

/** 按下：与 `handleCanvasPointerDown` 调进本模块时的形状一致。 */
function press(
    deps: Deps,
    init: { button?: number; buttons?: number; clientX?: number; clientY?: number } = {},
): void {
    const target = canvas();
    const event = pointerEvent("pointerdown", {
        button: init.button ?? 0,
        buttons: init.buttons ?? 1,
        clientX: init.clientX ?? 140,
        clientY: init.clientY ?? 100,
        pointerId: 1,
        pointerType: "mouse",
    }) as unknown as Parameters<typeof handleFreehandPointerDown>[1];
    Object.defineProperty(event, "currentTarget", { value: target, configurable: true });
    handleFreehandPointerDown(deps, event);
}

/** 拖拽中的一次移动（窗口级，因为监听挂在 window 上）。 */
function move(clientX: number, clientY: number, buttons = 1): void {
    window.dispatchEvent(
        pointerEvent("pointermove", {
            clientX,
            clientY,
            buttons,
            pointerId: 1,
            pointerType: "mouse",
        }),
    );
}

function up(): void {
    window.dispatchEvent(
        pointerEvent("pointerup", { buttons: 0, pointerId: 1, pointerType: "mouse" }),
    );
}

test("一次拖拽会逐点累积并只提交一次", async () => {
    const h = makeDeps();
    press(h.deps);

    move(150, 110);
    move(160, 120);
    flushFrames();
    up();
    await Promise.resolve();

    // 拖拽过程中的每一次移动都必须写进 live 覆盖层，否则用户看不到线在长。
    // 恰好 2 次 = 两次移动各一次；落笔那一次由调用方（`canvasPointerDown`）写。
    expect(h.applyDenseToLiveEdit).toHaveBeenCalledTimes(2);
    // 松手只提交一次，且提交的是完整轨迹：三个采样 → 帧号 280 / 300 / 320。
    expect(h.commitStroke).toHaveBeenCalledTimes(1);
    const committed = h.commitStroke.mock.calls[0][0] as Array<{ frame: number }>;
    expect(committed.map((point) => point.frame)).toEqual([280, 300, 320]);
    expect(h.applyPostStrokeSmoothing).toHaveBeenCalledTimes(1);
    // 松手后引用必须清空，否则下一次按下会带着上一笔的轨迹。
    expect(h.strokeRef.current).toBeNull();
});

test("同一帧内的移动只改值、不新增点", () => {
    const h = makeDeps();
    press(h.deps);

    // clientX 相同 ⇒ 帧号相同 ⇒ 走 "last.value = v2" 分支。
    move(140, 200);
    flushFrames();

    expect(h.strokeRef.current?.points).toHaveLength(1);
    expect(h.strokeRef.current?.points[0].value).toBe(2);
});

test("pointercancel 回滚：不提交、清引用、复位 live 标志", () => {
    const h = makeDeps();
    press(h.deps);

    move(150, 110);
    flushFrames();
    window.dispatchEvent(
        pointerEvent("pointercancel", { buttons: 0, pointerId: 1, pointerType: "mouse" }),
    );

    expect(h.commitStroke).not.toHaveBeenCalled();
    expect(h.strokeRef.current).toBeNull();
    expect(h.liveEditActiveRef.current).toBe(false);
    expect(h.edgeScroll.stop).toHaveBeenCalled();
});

test("右键走擦除模式，松手同样提交", async () => {
    const h = makeDeps({ mode: "restore" });
    h.strokeRef.current = { ...h.strokeRef.current!, mode: "restore" };
    press(h.deps, { button: 2, buttons: 2 });

    move(150, 110, 2);
    flushFrames();
    up();
    await Promise.resolve();

    expect(h.commitStroke).toHaveBeenCalledTimes(1);
    expect(h.commitStroke.mock.calls[0][1]).toBe("restore");
});

test("拖拽中按右键切换拖拽方向，并吞掉默认行为", () => {
    const h = makeDeps();
    press(h.deps);

    const event = new MouseEvent("mousedown", {
        button: 2,
        buttons: 3,
        bubbles: true,
        cancelable: true,
    });
    window.dispatchEvent(event);

    expect(h.onCycleDragDirection).toHaveBeenCalledWith("draw");
    expect(event.defaultPrevented).toBe(true);
});

test("拖拽中屏蔽上下文菜单", () => {
    const h = makeDeps();
    press(h.deps);

    const event = new Event("contextmenu", { bubbles: true, cancelable: true });
    window.dispatchEvent(event);

    expect(event.defaultPrevented).toBe(true);
});

test("边缘滚屏驱动被跟踪，松手后停止", () => {
    const h = makeDeps();
    press(h.deps);

    move(150, 110);
    expect(h.edgeScroll.track).toHaveBeenCalledWith(150);

    up();
    expect(h.edgeScroll.stop).toHaveBeenCalledTimes(1);
    // 松手后不得再有任何写入到驱动的回调槽位。
    expect(h.deps.onEdgeScrolledRef.current).toBeNull();
});

test("音高类参数的手势开始与结束会通知外部（吸附提示用）", async () => {
    const h = makeDeps();
    // 按下时的置位发生在调用方（`canvasPointerDown`），这里只验收尾。
    press(h.deps);
    up();
    await Promise.resolve();

    expect(h.onPitchSnapGestureActiveChange).toHaveBeenCalledWith(false);
});
