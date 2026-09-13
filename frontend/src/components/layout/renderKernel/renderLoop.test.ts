/**
 * 渲染循环（./renderLoop）行为自检。
 *
 * 【主要内容】
 * 1. 同一帧内多次 `invalidate` 只调度一次、只绘制一次；
 * 2. 未标脏时不绘制（按需调度，空闲零 CPU）；
 * 3. `stop` 取消未执行的帧，且停止后不再调度、已取消的帧不绘制；
 * 4. `draw` 内再次标脏会调度下一帧（清理顺序正确）。
 *
 * 【作用】这是"滚动 / 交互变更 → 一帧内合并绘制"的唯一调度器：合并失效会让
 * 高频输入退化为每事件一次全量绘制（正是要消除的卡顿来源）。
 *
 * 【与其他模块的关系】覆盖 `renderLoop.ts`；rAF 由测试注入，不依赖 DOM。
 */

import { describe, expect, it, vi } from "vitest";

import { createRenderLoop } from "./renderLoop";

/** 可手动触发的帧调度桩。 */
function createFrameStub() {
    const frames: FrameRequestCallback[] = [];
    const cancelled: number[] = [];
    return {
        frames,
        cancelled,
        requestFrame: (callback: FrameRequestCallback) => {
            frames.push(callback);
            return frames.length;
        },
        cancelFrame: (handle: number) => {
            cancelled.push(handle);
        },
    };
}

describe("renderLoop", () => {
    it("同一帧内多次 invalidate 只调度一次、只绘制一次", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        loop.invalidate();
        expect(stub.frames).toHaveLength(1);

        stub.frames[0](0);
        expect(draw).toHaveBeenCalledTimes(1);
    });

    it("未标脏时不绘制（start 本身不调度）", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        expect(stub.frames).toHaveLength(0);
        expect(draw).not.toHaveBeenCalled();
        expect(loop.isDirty()).toBe(false);
    });

    it("绘制后脏标记被清除（下一帧不再自动绘制）", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        stub.frames[0](0);
        expect(loop.isDirty()).toBe(false);
        expect(stub.frames).toHaveLength(1);
    });

    it("stop 取消未执行的帧，之后不再调度，已取消的帧不绘制", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        const scheduledHandle = stub.frames.length; // requestFrame 返回 1-based 句柄
        loop.stop();
        expect(stub.cancelled).toEqual([scheduledHandle]);

        loop.invalidate();
        expect(stub.frames).toHaveLength(1);

        // 已取消的帧即使被浏览器调用，也不得在停止后绘制。
        stub.frames[0](0);
        expect(draw).not.toHaveBeenCalled();
    });

    it("draw 内再次标脏会调度下一帧（清理顺序正确）", () => {
        const stub = createFrameStub();
        const draw = vi.fn(() => {
            if (draw.mock.calls.length === 1) loop.invalidate();
        });
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        stub.frames[0](0);
        expect(draw).toHaveBeenCalledTimes(1);
        // 回调内的标脏不应被本次清理吞掉：应已调度第二帧。
        expect(stub.frames).toHaveLength(2);
        stub.frames[1](0);
        expect(draw).toHaveBeenCalledTimes(2);
    });

    /**
     * `flush`：面板的输入路径同步写了 DOM / Canvas2D 时，GL 侧必须在**同一个任务**
     * 里提交，否则同一份视口会被两套图层分两帧呈现。
     */
    it("★ flush 立即绘制，并取消已排队的帧（不留重复绘制）", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        const scheduledHandle = stub.frames.length;
        expect(draw).not.toHaveBeenCalled();

        loop.flush();
        // 同步绘制完成，且那一帧已被取消（本次绘制覆盖了它要做的全部工作）。
        expect(draw).toHaveBeenCalledTimes(1);
        expect(stub.cancelled).toEqual([scheduledHandle]);
        expect(loop.isDirty()).toBe(false);

        // 已取消的帧即便被浏览器调用，也不会重复绘制。
        stub.frames[0](0);
        expect(draw).toHaveBeenCalledTimes(1);
    });

    it("flush 在未标脏时不绘制（可安全放在高频路径上）", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.flush();
        loop.flush();
        expect(draw).not.toHaveBeenCalled();
        expect(stub.frames).toHaveLength(0);
    });

    it("flush 在 stop 之后不绘制（与 invalidate 同一卸载保护）", () => {
        const stub = createFrameStub();
        const draw = vi.fn();
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        loop.stop();
        loop.flush();
        expect(draw).not.toHaveBeenCalled();
    });

    it("flush 内再次标脏会调度下一帧（只吞掉本次要提交的那一帧）", () => {
        const stub = createFrameStub();
        const draw = vi.fn(() => {
            if (draw.mock.calls.length === 1) loop.invalidate();
        });
        const loop = createRenderLoop({
            draw,
            requestFrame: stub.requestFrame,
            cancelFrame: stub.cancelFrame,
        });

        loop.start();
        loop.invalidate();
        loop.flush();
        expect(draw).toHaveBeenCalledTimes(1);
        // 首次标脏那一帧被取消（本次绘制覆盖了它），绘制内的标脏重新排队。
        // 注意 `stub.frames` 记录的是**请求过的帧**，不会因取消而缩短。
        expect(stub.cancelled).toEqual([1]);
        expect(stub.frames).toHaveLength(2);
        stub.frames[1](0);
        expect(draw).toHaveBeenCalledTimes(2);
    });
});
