/**
 * 参数编辑器内核 · 边缘自动滚屏驱动的**重入安全**回归锁。
 *
 * 【为什么单独一个文件】这组用例全部围绕同一条不变式：
 * `rafId !== null` ⟺ **恰好一个**待执行帧。旧实现在 `tick()` 入口就把
 * `rafId` 置空，于是「本帧正在执行」被误读成「空闲」，任何在 `apply()`
 * 内部发生的重入 `track()` 都会再排一帧，而 `tick()` 末尾又排一帧 ——
 * 每帧净增一个无人取消的孤儿回调。症状是拖得越久越卡（第 N 帧要跑 N 次
 * `tick`，每次两回 `getBoundingClientRect()` 强制同步布局）。
 *
 * 【它不做什么】不测手感（速度 / 带宽 / 软启动在 `edgeScrollAxis` 与
 * `valueAxisEdgeScroller`），不测调用方（`canvasPointerDown` 等）。
 *
 * 【违反边界的症状】待执行帧数随帧号线性增长 —— 见第一条用例的注释。
 */
import { describe, expect, it } from "vitest";
import { createEdgeScrollDriver } from "./edgeScrollDriver";
import type { EdgeScrollAxis } from "./edgeScrollAxis";

/**
 * 可数的帧队列：`runOneFrame()` 执行并清空**当前**所有回调，返回执行个数。
 *
 * 用真 `requestAnimationFrame` 测不出泄漏（jsdom 里根本没有），而且我们要断言的
 * 是「队列长度」这个**结构**事实，不是「跑了几毫秒」。
 */
function createFrameQueue() {
    const queue: Array<{ id: number; cb: () => void }> = [];
    let nextId = 1;
    return {
        queue,
        requestFrame: (cb: () => void) => {
            const id = nextId++;
            queue.push({ id, cb });
            return id;
        },
        cancelFrame: (id: number) => {
            const index = queue.findIndex((entry) => entry.id === id);
            if (index >= 0) queue.splice(index, 1);
        },
        runOneFrame: () => {
            const batch = queue.splice(0, queue.length);
            for (const entry of batch) entry.cb();
            return batch.length;
        },
    };
}

function runFrames(
    driverArgs: { reentrant: boolean; stopInsideApply?: boolean },
    frameCount: number,
) {
    const frames = createFrameQueue();
    let clock = 0;
    /*
     * 轴要在驱动构造前就交给驱动，而轴又要能反过来调用驱动（重入 / 自我 stop）
     * —— 用一个小 holder 打断这个构造环，避免 `let driver` + 后赋值。
     */
    const holder: { driver: ReturnType<typeof createEdgeScrollDriver> | null } = { driver: null };
    const applied: number[] = [];
    const axis: EdgeScrollAxis = {
        id: "y",
        policy: { bandPx: 20, maxRatio: 1, rampExponent: 2 },
        canScroll: () => true,
        apply: (_ratio, frameMs) => {
            applied.push(frameMs);
            if (frameMs <= 0) return false;
            if (driverArgs.stopInsideApply === true) holder.driver?.stop();
            // ★ 生产代码：值轴 setCenter → 发布交互视口 → 订阅者同步合成
            //   pointermove → 手势 onMove → 同一个驱动的 track()。
            if (driverArgs.reentrant) holder.driver?.track(100, 5);
            return true;
        },
    };
    const driver = createEdgeScrollDriver({
        getBounds: () => ({ left: 0, right: 200, top: 0, bottom: 100 }),
        axes: [axis],
        now: () => clock,
        requestFrame: frames.requestFrame,
        cancelFrame: frames.cancelFrame,
    });
    holder.driver = driver;
    driver.track(100, 5);
    const pendingPerFrame: number[] = [];
    for (let frame = 0; frame < frameCount; frame += 1) {
        clock += 16;
        frames.runOneFrame();
        pendingPerFrame.push(frames.queue.length);
    }
    return { frames, driver, applied, pendingPerFrame };
}

describe("edgeScrollDriver 重入安全", () => {
    it("★ apply() 内重入 track() 不得让待执行帧数增长", () => {
        const { pendingPerFrame, driver } = runFrames({ reentrant: true }, 6);
        // 修复前实测：[1, 2, 3, 4, 5, 6]（每帧净增一个孤儿回调）。
        expect(pendingPerFrame).toEqual([1, 1, 1, 1, 1, 1]);
        driver.stop();
    });

    it("★ stop() 之后不得留下任何待执行帧", () => {
        const { frames, driver } = runFrames({ reentrant: true }, 6);
        driver.stop();
        expect(frames.queue.length).toBe(0);
    });

    it("stop() 在 apply() 内部被调用时，本帧结束后也不得再排帧", () => {
        const { frames } = runFrames({ reentrant: true, stopInsideApply: true }, 3);
        expect(frames.queue.length).toBe(0);
    });

    it("复位帧位移为 0，之后每帧拿到真实帧时长", () => {
        const { applied, driver } = runFrames({ reentrant: false }, 2);
        // 第 1 帧是复位帧（frameMs = 0，本帧不滚），第 2 帧才是真实帧时长。
        expect(applied).toEqual([0, 16]);
        driver.stop();
    });

    it("两个轴共享同一个 frameMs（第二个轴不能拿到 0）", () => {
        const frames = createFrameQueue();
        let clock = 0;
        const seen: Array<{ id: string; frameMs: number }> = [];
        const makeAxis = (id: "x" | "y"): EdgeScrollAxis => ({
            id,
            policy: { bandPx: 20, maxRatio: 1, rampExponent: 2 },
            canScroll: () => true,
            apply: (_ratio, frameMs) => {
                seen.push({ id, frameMs });
                return true;
            },
        });
        const driver = createEdgeScrollDriver({
            getBounds: () => ({ left: 0, right: 200, top: 0, bottom: 100 }),
            axes: [makeAxis("x"), makeAxis("y")],
            now: () => clock,
            requestFrame: frames.requestFrame,
            cancelFrame: frames.cancelFrame,
        });
        // 指针在右下角：两个轴都落在触发带内。
        driver.track(195, 95);
        // 第 1 帧是复位帧（frameMs = 0），第 2 帧才是真实帧时长。
        clock += 16;
        frames.runOneFrame();
        clock += 16;
        frames.runOneFrame();
        const lastFrame = seen.slice(-2);
        expect(lastFrame.map((entry) => entry.id)).toEqual(["x", "y"]);
        expect(lastFrame.every((entry) => entry.frameMs === 16)).toBe(true);
        driver.stop();
    });
});
