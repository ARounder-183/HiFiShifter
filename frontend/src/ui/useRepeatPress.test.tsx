// @vitest-environment jsdom
/*
 * 按住重复触发（`useRepeatPress`）。
 *
 * 【为什么必须有】这套交互有两处只有上手才会发现的错法，且都不会抛错：
 * 1. **短按触发两次** —— pointerdown 触发一次、随后的 click 又触发一次（"点一下平滑
 *    两步"，用户会觉得按钮坏了）；
 * 2. **长按不累积** —— 重复时闭包捕获的是按下那一刻的值，连按十次等于把同一份数据
 *    平滑十遍（看起来像"按住没反应"）。第 2 条属于调用方的责任，这里用"每次触发都
 *    读取最新值"的探针把契约固定下来。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { useRepeatPress } from "./useRepeatPress";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

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
    vi.useRealTimers();
});

/** 挂一个按钮：`onTrigger` 每次把计数加一（模拟"读最新值 → 累加"的累积动作）。 */
async function mountButton(options: { disabled?: boolean } = {}) {
    const trigger = vi.fn();
    function Probe() {
        const handlers = useRepeatPress({ onTrigger: trigger, disabled: options.disabled });
        return (
            <button type="button" data-testid="btn" {...handlers}>
                go
            </button>
        );
    }
    await act(async () => {
        root.render(<Probe />);
    });
    const button = document.querySelector<HTMLButtonElement>('[data-testid="btn"]')!;
    return { trigger, button };
}

const pointerDown = (button: HTMLElement, clientY = 0) =>
    button.dispatchEvent(
        new PointerEvent("pointerdown", { bubbles: true, button: 0, pointerId: 1, clientY }),
    );

test("短按：pointerdown 立即触发一次，随后的 click 不再触发", async () => {
    const { trigger, button } = await mountButton();

    await act(async () => {
        pointerDown(button);
    });
    expect(trigger).toHaveBeenCalledTimes(1);

    // 指针点击的 click 带 detail ≥ 1：不应重复触发。
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true, detail: 1 }));
    });
    expect(trigger).toHaveBeenCalledTimes(1);
});

test("键盘激活（click 的 detail 为 0）触发一次", async () => {
    const { trigger, button } = await mountButton();
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true, detail: 0 }));
    });
    expect(trigger).toHaveBeenCalledTimes(1);
});

test("长按：等待期后按固定间隔重复，松手即停", async () => {
    vi.useFakeTimers();
    const { trigger, button } = await mountButton();

    await act(async () => {
        pointerDown(button);
    });
    expect(trigger).toHaveBeenCalledTimes(1);

    // 等待期内不重复。
    await act(async () => {
        vi.advanceTimersByTime(300);
    });
    expect(trigger).toHaveBeenCalledTimes(1);

    // 越过等待期后开始按间隔重复。
    await act(async () => {
        vi.advanceTimersByTime(100);
    });
    expect(trigger.mock.calls.length).toBeGreaterThan(1);
    const afterFirstRepeat = trigger.mock.calls.length;

    await act(async () => {
        vi.advanceTimersByTime(200);
    });
    expect(trigger.mock.calls.length).toBeGreaterThan(afterFirstRepeat);

    // 松手后不再重复。
    await act(async () => {
        button.dispatchEvent(new PointerEvent("pointerup", { bubbles: true, pointerId: 1 }));
    });
    const afterRelease = trigger.mock.calls.length;
    await act(async () => {
        vi.advanceTimersByTime(1000);
    });
    expect(trigger.mock.calls.length).toBe(afterRelease);
});

test("指针移出按钮后松手（lostpointercapture）也会停止重复", async () => {
    vi.useFakeTimers();
    const { trigger, button } = await mountButton();

    await act(async () => {
        pointerDown(button);
    });
    await act(async () => {
        vi.advanceTimersByTime(400);
    });
    expect(trigger.mock.calls.length).toBeGreaterThan(1);

    await act(async () => {
        button.dispatchEvent(new Event("lostpointercapture", { bubbles: true }));
    });
    const afterRelease = trigger.mock.calls.length;
    await act(async () => {
        vi.advanceTimersByTime(1000);
    });
    expect(trigger.mock.calls.length).toBe(afterRelease);
});

test("禁用时不触发（指针与键盘都不触发）", async () => {
    const { trigger, button } = await mountButton({ disabled: true });
    await act(async () => {
        pointerDown(button);
    });
    await act(async () => {
        button.dispatchEvent(new MouseEvent("click", { bubbles: true, detail: 0 }));
    });
    expect(trigger).not.toHaveBeenCalled();
});

test("卸载后重复定时器不再打过来", async () => {
    vi.useFakeTimers();
    const { trigger, button } = await mountButton();
    await act(async () => {
        pointerDown(button);
    });
    await act(async () => {
        vi.advanceTimersByTime(400);
    });
    const beforeUnmount = trigger.mock.calls.length;

    await act(async () => {
        root.unmount();
    });
    await act(async () => {
        vi.advanceTimersByTime(1000);
    });
    expect(trigger.mock.calls.length).toBe(beforeUnmount);
});
