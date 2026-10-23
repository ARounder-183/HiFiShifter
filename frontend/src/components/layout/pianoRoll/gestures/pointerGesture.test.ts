/**
 * 指针手势协议的契约。
 *
 * 【要钉死什么】这套协议的每一条都对应过一次真实故障：
 * - 漏装 pointerup ⇒ 从不提交（`710ff764` 的自由绘制）；
 * - 收尾不幂等 ⇒ 同一笔画提交两次；
 * - 不注销额外监听 ⇒ 拖拽结束后右键菜单仍然被吞掉；
 * - 注册 / 注销的函数引用不一致 ⇒ 面板的失焦收尾永远清不掉（见
 *   usePianoRollInteractions 的身份配对）。
 */
// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";

import { startPointerGesture } from "./pointerGesture";

function pointerEvent(type: string, pointerId = 1): PointerEvent {
    return new PointerEvent(type, {
        bubbles: true,
        cancelable: true,
        pointerId,
        pointerType: "mouse",
    });
}

beforeEach(() => {
    document.body.innerHTML = "";
});

test("只接受本指针的移动", () => {
    const onMove = vi.fn();
    startPointerGesture({ pointerId: 1, onMove, onEnd: vi.fn() });
    window.dispatchEvent(pointerEvent("pointermove", 1));
    window.dispatchEvent(pointerEvent("pointermove", 2));
    expect(onMove).toHaveBeenCalledTimes(1);
});

test("松手收尾一次，之后监听全部撤销", () => {
    const onMove = vi.fn();
    const onEnd = vi.fn();
    const gesture = startPointerGesture({ pointerId: 1, onMove, onEnd });
    expect(gesture.active).toBe(true);
    window.dispatchEvent(pointerEvent("pointerup"));
    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(onEnd.mock.calls[0][0]).toBe("up");
    expect(gesture.active).toBe(false);
    // 收尾之后既不再收 move，也不再次收尾。
    window.dispatchEvent(pointerEvent("pointermove"));
    window.dispatchEvent(pointerEvent("pointerup"));
    expect(onMove).not.toHaveBeenCalled();
    expect(onEnd).toHaveBeenCalledTimes(1);
});

test("取消收尾带 cancel 原因，让调用方回滚而不是提交", () => {
    const onEnd = vi.fn();
    startPointerGesture({ pointerId: 1, onMove: vi.fn(), onEnd });
    window.dispatchEvent(pointerEvent("pointercancel"));
    expect(onEnd).toHaveBeenCalledTimes(1);
    expect(onEnd.mock.calls[0][0]).toBe("cancel");
});

test("主动收尾是幂等的（三条收尾路径会竞争）", () => {
    const onEnd = vi.fn();
    const gesture = startPointerGesture({ pointerId: 1, onMove: vi.fn(), onEnd });
    gesture.end("up");
    gesture.end("up");
    window.dispatchEvent(pointerEvent("pointerup"));
    expect(onEnd).toHaveBeenCalledTimes(1);
});

test("额外监听随手势一起安装与注销", () => {
    const onContextMenu = vi.fn((event: Event) => event.preventDefault());
    const gesture = startPointerGesture({
        pointerId: 1,
        onMove: vi.fn(),
        onEnd: vi.fn(),
        extra: [{ type: "contextmenu", listener: onContextMenu, capture: true }],
    });
    window.dispatchEvent(new Event("contextmenu", { cancelable: true }));
    expect(onContextMenu).toHaveBeenCalledTimes(1);
    gesture.end();
    window.dispatchEvent(new Event("contextmenu", { cancelable: true }));
    expect(onContextMenu).toHaveBeenCalledTimes(1);
});

test("注册与注销拿到的是同一个函数引用", () => {
    const registerEnd = vi.fn();
    const clearEnd = vi.fn();
    const gesture = startPointerGesture({
        pointerId: 1,
        onMove: vi.fn(),
        onEnd: vi.fn(),
        registerEnd,
        clearEnd,
    });
    expect(registerEnd).toHaveBeenCalledTimes(1);
    const registered = registerEnd.mock.calls[0][0] as () => void;
    gesture.end();
    expect(clearEnd).toHaveBeenCalledTimes(1);
    // 身份必须相同：面板按引用配对，不一致就永远清不掉。
    expect(clearEnd.mock.calls[0][0]).toBe(registered);
    // 面板调那个引用等价于正常松手收尾。
    expect(gesture.active).toBe(false);
});

test("捕获的指针在收尾时被释放", () => {
    const canvas = document.createElement("canvas");
    document.body.appendChild(canvas);
    const release = vi.fn();
    (canvas as unknown as { releasePointerCapture: () => void }).releasePointerCapture = release;
    const gesture = startPointerGesture({
        pointerId: 7,
        captureTarget: canvas,
        onMove: vi.fn(),
        onEnd: vi.fn(),
    });
    gesture.end();
    expect(release).toHaveBeenCalledWith(7);
});
