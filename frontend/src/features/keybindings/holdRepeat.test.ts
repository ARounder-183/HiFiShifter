import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
    beginHoldRepeat,
    consumeHoldRepeatKeyDown,
    isHoldRepeatActive,
    stopHoldRepeat,
} from "./holdRepeat";

function keyEvent(
    key: string,
    opts: { repeat?: boolean; ctrl?: boolean } = {},
): { ev: KeyboardEvent; preventDefault: ReturnType<typeof vi.fn> } {
    const preventDefault = vi.fn();
    const ev = {
        key,
        repeat: Boolean(opts.repeat),
        ctrlKey: Boolean(opts.ctrl),
        metaKey: false,
        shiftKey: false,
        altKey: false,
        preventDefault,
    } as unknown as KeyboardEvent;
    return { ev, preventDefault };
}

describe("holdRepeat — 长按重复管理器（粘贴同款节奏）", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        stopHoldRepeat();
    });
    afterEach(() => {
        vi.useRealTimers();
        stopHoldRepeat();
    });

    it("按下后先停顿 initialDelay，再按 repeatInterval 固定节奏重复", () => {
        const fire = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fire, {
            initialDelayMs: 400,
            repeatIntervalMs: 50,
        });
        // 首次执行由调用方负责（粘贴流程：先 onPaste 再 begin），管理器不自动触发。
        expect(fire).not.toHaveBeenCalled();
        // 初始延时到点（t=400）只布防 interval，第一拍落在 t=450。
        vi.advanceTimersByTime(400);
        expect(fire).not.toHaveBeenCalled();
        vi.advanceTimersByTime(50);
        expect(fire).toHaveBeenCalledTimes(1);
        vi.advanceTimersByTime(100);
        expect(fire).toHaveBeenCalledTimes(3);
    });

    it("长按进行中：同键 OS 自动重复被吞掉并 preventDefault，不触发 fire", () => {
        const fire = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fire, {
            initialDelayMs: 400,
            repeatIntervalMs: 50,
        });
        const { ev, preventDefault } = keyEvent("t", { repeat: true, ctrl: true });
        expect(consumeHoldRepeatKeyDown(ev)).toBe(true);
        expect(preventDefault).toHaveBeenCalled();
        expect(fire).not.toHaveBeenCalled();
        vi.advanceTimersByTime(400);
        expect(fire).not.toHaveBeenCalled();
        vi.advanceTimersByTime(50);
        expect(fire).toHaveBeenCalledTimes(1);
    });

    it("长按进行中：其它按键的自动重复同样被吞掉", () => {
        const fire = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fire, {
            initialDelayMs: 400,
            repeatIntervalMs: 50,
        });
        const { ev, preventDefault } = keyEvent("x", { repeat: true });
        expect(consumeHoldRepeatKeyDown(ev)).toBe(true);
        // 非主键的重复只吞掉、不阻止默认（避免影响编辑器内其它行为）。
        expect(preventDefault).not.toHaveBeenCalled();
    });

    it("长按进行中：任何非重复按键都视为意图变化，终止长按（粘贴语义）", () => {
        const fire = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fire, {
            initialDelayMs: 100,
            repeatIntervalMs: 50,
        });
        vi.advanceTimersByTime(100);
        expect(fire).not.toHaveBeenCalled();
        vi.advanceTimersByTime(50);
        expect(fire).toHaveBeenCalledTimes(1);
        const { ev } = keyEvent("x");
        expect(consumeHoldRepeatKeyDown(ev)).toBe(false); // 返回 false：新按键照常处理
        expect(isHoldRepeatActive()).toBe(false);
        vi.advanceTimersByTime(500);
        expect(fire).toHaveBeenCalledTimes(1); // 不再重复
    });

    it("stopHoldRepeat 立即终止并清空定时器", () => {
        const fire = vi.fn();
        beginHoldRepeat({ key: "d", ctrl: true }, fire, {
            initialDelayMs: 100,
            repeatIntervalMs: 50,
        });
        expect(isHoldRepeatActive()).toBe(true);
        stopHoldRepeat();
        expect(isHoldRepeatActive()).toBe(false);
        vi.advanceTimersByTime(1000);
        expect(fire).not.toHaveBeenCalled();
    });

    it("无长按进行中时 consume 返回 false，事件照常处理", () => {
        const { ev } = keyEvent("t", { repeat: true, ctrl: true });
        expect(consumeHoldRepeatKeyDown(ev)).toBe(false);
        expect(isHoldRepeatActive()).toBe(false);
    });

    it("重新 begin 会替换旧长按（不叠加定时器）", () => {
        const fireA = vi.fn();
        const fireB = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fireA, {
            initialDelayMs: 100,
            repeatIntervalMs: 50,
        });
        beginHoldRepeat({ key: "d", ctrl: true }, fireB, {
            initialDelayMs: 100,
            repeatIntervalMs: 50,
        });
        vi.advanceTimersByTime(200);
        expect(fireA).not.toHaveBeenCalled();
        expect(fireB).toHaveBeenCalledTimes(2);
    });
});
