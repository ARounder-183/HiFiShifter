/**
 * 拖拽边缘自动滚屏驱动（./edgeScrollDriver）行为自检。
 *
 * 【本测试守护什么】驱动里有四件**只能靠手拖复现**的事，全部在这里钉住：
 * 1. **指针停在边缘不动也要继续滚**（rAF 循环），而指针离开边缘带就停；
 * 2. **滚动之后回调调用方**，且回调带的是同一个指针位置；
 * 3. **时钟按「距上一帧」推进，且离开带就复位** —— 这是"来回蹭边缘会一跳一跳"
 *    的根因（旧实现按"距上一次 step"计，带外空闲会被补成一帧的大位移）；
 * 4. **帧时长每帧只算一次** —— 两个轴各算一次会让第二个轴永远拿到 0，
 *    症状是"只有横向会滚"。
 *
 * 【与其他模块的关系】驱动只负责时钟、rAF 与分派；比例几何由
 * `edgeScrollAxis.test.ts` 覆盖，值轴积分由 `valueAxisEdgeScroller.test.ts`
 * 覆盖。宿主能力全部注入，不依赖 DOM。
 */
import { describe, expect, it, vi } from "vitest";

import { createEdgeScrollDriver } from "./edgeScrollDriver";
import type { EdgeScrollAxis } from "./edgeScrollAxis";

/** 视口：left=100、right=1100、top=100、bottom=900。 */
const BOUNDS = { left: 100, right: 1100, top: 100, bottom: 900 };
const FRAME_MS = 1000 / 60;
/** 右缘带宽内、离右边界 4px。 */
const NEAR_RIGHT = 1100 - 4;

interface FakeAxis {
    readonly axis: EdgeScrollAxis;
    readonly applied: { ratio: number; frameMs: number }[];
    position: () => number;
    setScrollable: (next: boolean) => void;
}

/**
 * 一个记录调用的假轴：位移 = ratio × frameMs / 1000，单位为「轴自己的单位」。
 *
 * `apply` 在帧时长为 0 时返回 false（真实轴也如此：复位帧不产生位移），
 * 驱动据此继续排帧 —— 这正是要验证的行为之一。
 */
function createFakeAxis(
    id: "x" | "y",
    policy = { bandPx: 32, maxRatio: 1.5, rampExponent: 1 },
): FakeAxis {
    const applied: { ratio: number; frameMs: number }[] = [];
    let position = 0;
    let scrollable = true;
    return {
        axis: {
            id,
            policy,
            canScroll: () => scrollable,
            apply: (ratio, frameMs) => {
                applied.push({ ratio, frameMs });
                if (frameMs <= 0 || ratio === 0) return false;
                position += (ratio * frameMs) / 1000;
                return true;
            },
        },
        applied,
        position: () => position,
        setScrollable: (next: boolean) => {
            scrollable = next;
        },
    };
}

function createHarness(
    axes: readonly EdgeScrollAxis[] = [createFakeAxis("x").axis],
    bounds: { left: number; right: number; top?: number; bottom?: number } = BOUNDS,
) {
    let clockMs = 0;
    let nextHandle = 1;
    const pending = new Map<number, () => void>();
    const onScrolled = vi.fn();
    const driver = createEdgeScrollDriver({
        getBounds: () => bounds,
        axes,
        onScrolled,
        now: () => clockMs,
        requestFrame: (cb) => {
            const handle = nextHandle++;
            pending.set(handle, cb);
            return handle;
        },
        cancelFrame: (handle) => {
            pending.delete(handle);
        },
    });
    return {
        driver,
        onScrolled,
        advanceClock(deltaMs: number) {
            clockMs += deltaMs;
        },
        flushFrame(deltaMs = FRAME_MS) {
            clockMs += deltaMs;
            const callbacks = [...pending.values()];
            pending.clear();
            for (const cb of callbacks) cb();
        },
        pendingFrameCount: () => pending.size,
    };
}

describe("createEdgeScrollDriver · 时钟", () => {
    it("复位后的第一帧不产生位移（frameMs = 0），第二帧才推进", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        expect(axis.applied).toHaveLength(1);
        expect(axis.applied[0].frameMs).toBe(0);
        expect(axis.position()).toBe(0);

        h.flushFrame();
        expect(axis.applied[1].frameMs).toBeCloseTo(FRAME_MS, 9);
        expect(axis.position()).toBeGreaterThan(0);
    });

    it("★ 指针离开带 500ms 后重新进入：第一帧仍为 0，不吃掉带外空闲", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.flushFrame();
        const afterTwoFrames = axis.position();

        // 指针回到视口中央：循环结束，时钟复位。
        h.driver.track(600);
        h.advanceClock(500);
        expect(h.driver.isRunning()).toBe(false);

        // 重新进带：第一帧必须是 0，第二帧才是真实帧时长。
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        expect(axis.position()).toBe(afterTwoFrames);
        h.flushFrame();
        expect(axis.position()).toBeGreaterThan(afterTwoFrames);
    });

    it("带内长时间停顿后的下一帧被钳到 32ms（不补一长段）", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.flushFrame();
        h.flushFrame(500); // 一帧 500ms 的"卡顿"
        expect(axis.applied.at(-1)!.frameMs).toBe(32);
    });

    it("★ 同一帧里两个轴各推进一次（帧时长不重复消费）", () => {
        const x = createFakeAxis("x");
        const y = createFakeAxis("y", { bandPx: 20, maxRatio: 1, rampExponent: 2 });
        const h = createHarness([x.axis, y.axis]);
        // 右下角：两个轴都在带内。
        h.driver.track(NEAR_RIGHT, 896);
        h.flushFrame();
        h.flushFrame();
        expect(x.applied.at(-1)!.frameMs).toBeCloseTo(FRAME_MS, 9);
        expect(y.applied.at(-1)!.frameMs).toBeCloseTo(FRAME_MS, 9);
        expect(x.position()).toBeGreaterThan(0);
        expect(y.position()).toBeGreaterThan(0);
    });
});

describe("createEdgeScrollDriver · rAF 循环", () => {
    it("指针停在边缘不动时持续滚动", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);
        h.flushFrame();
        h.flushFrame();
        const afterSecond = axis.position();
        h.flushFrame();
        expect(axis.position()).toBeGreaterThan(afterSecond);
        expect(h.driver.isRunning()).toBe(true);
    });

    it("指针离开边缘带后循环自行结束", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);
        h.driver.track(600);
        h.flushFrame();
        expect(h.driver.isRunning()).toBe(false);
        expect(h.pendingFrameCount()).toBe(0);
    });

    it("同一帧内多次 track 只保留一个挂起帧", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.driver.track(NEAR_RIGHT);
        h.driver.track(NEAR_RIGHT);
        expect(h.pendingFrameCount()).toBe(1);
    });

    it("不可滚的轴不排帧", () => {
        const axis = createFakeAxis("x");
        axis.setScrollable(false);
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(false);
        expect(h.driver.step(NEAR_RIGHT)).toBe(false);
        expect(axis.applied).toHaveLength(0);
    });

    it("竖直轴缺边界时不参与命中，也不排帧", () => {
        const y = createFakeAxis("y");
        const h = createHarness([y.axis], { left: 100, right: 1100 });
        h.driver.track(600, 896);
        expect(h.driver.isRunning()).toBe(false);
        expect(y.applied).toHaveLength(0);
    });

    it("滚动回调让调用方按滚动后的投影重算（回调次数 = 真正滚动的帧数）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame(); // 复位帧：无位移，不回调
        h.flushFrame();
        h.flushFrame();
        h.flushFrame();
        expect(h.onScrolled).toHaveBeenCalledTimes(3);
        for (const call of h.onScrolled.mock.calls) {
            expect(call[0]).toBe(NEAR_RIGHT);
        }
    });

    it("轴报告「没有位移」时循环结束（到边界不空转）", () => {
        const applied: number[] = [];
        const pinned: EdgeScrollAxis = {
            id: "x",
            policy: { bandPx: 32, maxRatio: 1.5, rampExponent: 1 },
            canScroll: () => true,
            apply: (_ratio, frameMs) => {
                applied.push(frameMs);
                return false;
            },
        };
        const h = createHarness([pinned]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame(); // 复位帧：frameMs 0，apply false → 仍排一帧
        expect(h.driver.isRunning()).toBe(true);
        h.flushFrame(); // 真实帧：apply false → 结束
        expect(h.driver.isRunning()).toBe(false);
    });
});

describe("createEdgeScrollDriver · stop", () => {
    it("取消挂起帧并停止循环", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.flushFrame();
        expect(h.driver.isRunning()).toBe(true);
        const frozen = axis.position();
        h.driver.stop();
        expect(h.driver.isRunning()).toBe(false);
        expect(h.pendingFrameCount()).toBe(0);
        h.flushFrame();
        h.flushFrame();
        expect(axis.position()).toBe(frozen);
    });

    it("幂等：重复 stop 不抛错、不残留挂起帧", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.driver.stop();
        expect(() => h.driver.stop()).not.toThrow();
        expect(h.pendingFrameCount()).toBe(0);
    });

    it("stop 后重新 track 能正常恢复（新手势第一帧仍为 0）", () => {
        const axis = createFakeAxis("x");
        const h = createHarness([axis.axis]);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.flushFrame();
        h.driver.stop();
        const before = axis.position();

        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);
        h.flushFrame();
        expect(axis.position()).toBe(before);
        h.flushFrame();
        expect(axis.position()).toBeGreaterThan(before);
    });

    it("容器未挂载（getBounds 为 null）时不排程、不抛错", () => {
        const driver = createEdgeScrollDriver({
            getBounds: () => null,
            axes: [createFakeAxis("x").axis],
        });
        expect(() => driver.track(NEAR_RIGHT)).not.toThrow();
        expect(driver.isRunning()).toBe(false);
        expect(driver.step(NEAR_RIGHT)).toBe(false);
    });
});
