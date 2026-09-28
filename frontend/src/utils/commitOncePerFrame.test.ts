/**
 * 帧合并提交单测。
 *
 * 【本测试要钉死的语义】一段连续手势（滚轮 / 拖拽）里的多次写入必须合并成**每帧一次**，
 * 且提交的是最后一次的值。既不能每个事件提交一次（会抽搐），也不能只在手势结束时提交
 * 一次（拖动过程不跟随）。
 */
import { describe, expect, it } from "vitest";

import { createFrameCommitter, type FrameScheduler } from "./commitOncePerFrame";

/** 手动驱动的帧调度器：测试自己决定"帧什么时候到"。 */
function makeScheduler() {
    const queued = new Map<number, () => void>();
    let nextHandle = 1;
    const scheduler: FrameScheduler = {
        request(callback) {
            const handle = nextHandle++;
            queued.set(handle, callback);
            return handle;
        },
        cancel(handle) {
            queued.delete(handle);
        },
    };
    return {
        scheduler,
        pendingCount: () => queued.size,
        /** 跑掉当前排队的全部回调（模拟一次帧提交；回调里新排的帧留给下一次调用）。 */
        tick() {
            const callbacks = [...queued.values()];
            queued.clear();
            for (const callback of callbacks) callback();
        },
    };
}

describe("createFrameCommitter（帧合并提交）", () => {
    it("同一帧内多次 schedule 只提交一次，且提交最后一个值", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        committer.schedule(1);
        committer.schedule(2);
        committer.schedule(3);
        expect(committed).toEqual([]);
        expect(frames.pendingCount()).toBe(1);

        frames.tick();
        expect(committed).toEqual([3]);
    });

    it("★ 每帧最多一次：连续 50 次写入在 3 帧里只提交 3 次", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        for (let i = 0; i < 20; i += 1) committer.schedule(i);
        frames.tick();
        for (let i = 20; i < 35; i += 1) committer.schedule(i);
        frames.tick();
        for (let i = 35; i < 50; i += 1) committer.schedule(i);
        frames.tick();

        expect(committed).toEqual([19, 34, 49]);
    });

    it("flush 立即提交挂起的值，且不再等待帧", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        committer.schedule(7);
        committer.flush();
        expect(committed).toEqual([7]);
        expect(frames.pendingCount()).toBe(0);

        // 后续帧不得重复提交同一个值。
        frames.tick();
        expect(committed).toEqual([7]);
    });

    it("flush 在无挂起值时不提交（不产生空提交）", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        committer.flush();
        frames.tick();
        expect(committed).toEqual([]);
    });

    it("cancel 丢弃挂起值", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        committer.schedule(5);
        committer.cancel();
        frames.tick();
        expect(committed).toEqual([]);
        expect(committer.isPending()).toBe(false);
    });

    it("★ 提交回调内再次 schedule：不递归提交，改为排下一帧", () => {
        const frames = makeScheduler();
        const committed: number[] = [];
        const committer = createFrameCommitter<number>((value) => {
            committed.push(value);
            if (value < 3) committer.schedule(value + 1);
        }, frames.scheduler);

        committer.schedule(1);
        frames.tick();
        expect(committed).toEqual([1]);
        frames.tick();
        expect(committed).toEqual([1, 2]);
        frames.tick();
        expect(committed).toEqual([1, 2, 3]);
        // 3 不再自我调度，队列清空。
        expect(frames.pendingCount()).toBe(0);
    });

    it("isPending 反映是否有未提交的值", () => {
        const frames = makeScheduler();
        const committer = createFrameCommitter<number>(() => {}, frames.scheduler);
        expect(committer.isPending()).toBe(false);
        committer.schedule(1);
        expect(committer.isPending()).toBe(true);
        frames.tick();
        expect(committer.isPending()).toBe(false);
    });

    it("挂起的值可以是 null / undefined（与「无挂起」区分开）", () => {
        const frames = makeScheduler();
        const committed: Array<number | null> = [];
        const committer = createFrameCommitter<number | null>(
            (value) => committed.push(value),
            frames.scheduler,
        );

        committer.schedule(null);
        expect(committer.isPending()).toBe(true);
        frames.tick();
        expect(committed).toEqual([null]);
    });
});
