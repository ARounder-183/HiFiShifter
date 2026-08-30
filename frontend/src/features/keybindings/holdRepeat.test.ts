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

    it("长按进行中：其它按键的非重复按下视为意图变化，终止长按（粘贴语义）", () => {
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

    it("长按进行中：同键非重复事件不终止长按（布防事件被二次消费时存活）", () => {
        // 回归：同一 keydown 事件会被 App（useKeybindings）与 TimelinePanel
        // （useKeyboardShortcuts）两个 window 捕获监听器依次消费。第一个监听
        // 完成布防后，第二个监听对同一事件调用 consume —— 同键、非重复，
        // 必须放行，否则刚布防的长按被当场杀死（表现为四个动作完全无重复）。
        const fire = vi.fn();
        beginHoldRepeat({ key: "t", ctrl: true }, fire, {
            initialDelayMs: 100,
            repeatIntervalMs: 50,
        });
        const { ev } = keyEvent("t", { ctrl: true }); // 与布防键相同的非重复事件
        expect(consumeHoldRepeatKeyDown(ev)).toBe(false);
        expect(isHoldRepeatActive()).toBe(true); // 布防存活
        vi.advanceTimersByTime(100);
        expect(fire).not.toHaveBeenCalled();
        vi.advanceTimersByTime(50);
        expect(fire).toHaveBeenCalledTimes(1);
        vi.advanceTimersByTime(100);
        expect(fire).toHaveBeenCalledTimes(3);
    });

    it("双监听器同事件消费（顺序 A：全局先布防，时间轴后消费）不杀长按", () => {
        // 模拟 window 捕获监听顺序 [App, Timeline]：
        // 第一个监听执行完整动作（fire + begin），第二个监听对同一事件
        // 再执行 consume —— 布防必须存活并进入重复节奏。
        const fire = vi.fn();
        // 监听 1（App）：无 active → 放行 → 动作 + 布防
        const appListener = (e: KeyboardEvent) => {
            if (consumeHoldRepeatKeyDown(e)) return;
            fire(); // 动作首次执行
            beginHoldRepeat({ key: "t", ctrl: true }, fire, {
                initialDelayMs: 100,
                repeatIntervalMs: 50,
            });
        };
        // 监听 2（Timeline）：对同一事件二次 consume
        const timelineListener = (e: KeyboardEvent) => {
            if (consumeHoldRepeatKeyDown(e)) return;
        };

        const first = keyEvent("t", { ctrl: true });
        appListener(first.ev);
        timelineListener(first.ev); // 同事件、同键、非重复 → 不得终止
        expect(isHoldRepeatActive()).toBe(true);
        expect(fire).toHaveBeenCalledTimes(1); // 仅首次动作，计时器尚未触发

        vi.advanceTimersByTime(100);
        expect(fire).toHaveBeenCalledTimes(1);
        vi.advanceTimersByTime(50);
        expect(fire).toHaveBeenCalledTimes(2); // 计时器节奏照常

        // 长按中的 OS 自动重复：两个监听都吞掉
        const rep = keyEvent("t", { repeat: true, ctrl: true });
        appListener(rep.ev);
        timelineListener(rep.ev);
        expect(fire).toHaveBeenCalledTimes(2);
        vi.advanceTimersByTime(100);
        expect(fire).toHaveBeenCalledTimes(4);
    });

    it("双监听器同事件消费（顺序 B：时间轴先布防，全局后消费）不杀长按（粘贴路径）", () => {
        // 模拟 window 捕获监听顺序 [Timeline, App]（粘贴的布防位置在 Timeline）：
        // 第二个监听（App）对同一事件 consume —— 同键非重复必须放行。
        const fire = vi.fn();
        const timelineListener = (e: KeyboardEvent) => {
            if (consumeHoldRepeatKeyDown(e)) return;
            fire(); // 粘贴首次执行
            beginHoldRepeat({ key: "v", ctrl: true }, fire, {
                initialDelayMs: 100,
                repeatIntervalMs: 50,
            });
        };
        const appListener = (e: KeyboardEvent) => {
            if (consumeHoldRepeatKeyDown(e)) return;
        };

        const first = keyEvent("v", { ctrl: true });
        timelineListener(first.ev);
        appListener(first.ev); // 同事件、同键、非重复 → 不得终止
        expect(isHoldRepeatActive()).toBe(true);
        expect(fire).toHaveBeenCalledTimes(1);

        vi.advanceTimersByTime(150);
        expect(fire).toHaveBeenCalledTimes(2); // 布防存活，节奏照常
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
