/**
 * 预览合帧器：只重放挂起帧，取消路径不得留下待处理工作。
 */
import { expect, test, vi } from "vitest";

import { createRafCoalescedReplay } from "./gesturePreviewRaf";

test("flush 只回放一次挂起帧，重复 flush 不再做 O(数据量) 重算", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextId = 1;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        const id = nextId++;
        callbacks.set(id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => callbacks.delete(id));
    const replay = vi.fn();
    try {
        const coalesced = createRafCoalescedReplay(replay);
        coalesced.schedule();
        coalesced.schedule();
        expect(callbacks.size).toBe(1);
        coalesced.flush();
        expect(replay).toHaveBeenCalledTimes(1);
        coalesced.flush();
        expect(replay).toHaveBeenCalledTimes(1);
        expect(callbacks.size).toBe(0);
    } finally {
        vi.unstubAllGlobals();
    }
});

test("rAF 已执行后 flush 不重放，cancel 丢弃挂起帧", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextId = 1;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        const id = nextId++;
        callbacks.set(id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", (id: number) => callbacks.delete(id));
    const replay = vi.fn();
    try {
        const coalesced = createRafCoalescedReplay(replay);
        coalesced.schedule();
        const [id, callback] = [...callbacks][0];
        callbacks.delete(id);
        callback(0);
        coalesced.flush();
        expect(replay).toHaveBeenCalledTimes(1);

        coalesced.schedule();
        coalesced.cancel();
        coalesced.flush();
        expect(replay).toHaveBeenCalledTimes(1);
    } finally {
        vi.unstubAllGlobals();
    }
});

test("cancel 后已进入回调队列的旧 rAF 不重放，也不撤销新帧", () => {
    const callbacks = new Map<number, FrameRequestCallback>();
    let nextId = 1;
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        const id = nextId++;
        callbacks.set(id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", () => {
        // 故意不删除：模拟浏览器已经把回调投递到任务队列时的竞态。
    });
    const replay = vi.fn();
    try {
        const coalesced = createRafCoalescedReplay(replay);
        coalesced.schedule();
        const stale = [...callbacks.values()][0];
        coalesced.cancel();
        coalesced.schedule();
        stale(0);
        expect(replay).not.toHaveBeenCalled();
        coalesced.flush();
        expect(replay).toHaveBeenCalledTimes(1);
    } finally {
        vi.unstubAllGlobals();
    }
});
