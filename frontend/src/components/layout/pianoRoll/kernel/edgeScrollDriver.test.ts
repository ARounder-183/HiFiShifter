/**
 * 拖拽边缘自动滚屏驱动（./edgeScrollDriver）行为自检。
 *
 * 【本测试守护什么】驱动里有三件**只能靠手拖复现**的事，全部在这里钉住：
 * 1. **指针停在边缘不动也要继续滚**（rAF 循环），而指针离开边缘带就停 ——
 *    漏掉前者是"画到边缘就画不动了"，漏掉后者是"松手后视图一直滚"；
 * 2. **滚动之后回调调用方**，且回调带的是同一个指针位置（调用方据此按滚动后的
 *    投影重算笔画；不回调就是"视图过去了、线没跟上"的跳变）；
 * 3. **`stop()` 幂等且彻底**：取消挂起帧、清掉指针与时钟。
 *
 * 另外锁住速度语义：步长与**真实间隔**成正比（与事件频率无关），而新手势 / 长间隔
 * 按 1/60 秒起步（避免一步跳上百像素）。
 *
 * 【与其他模块的关系】覆盖 `edgeScrollDriver.ts`；几何本身由
 * `shared/edgeAutoScroll.test.ts` 覆盖，这里只测"驱动"这一层（时钟、rAF 循环、
 * 上界钳制、回调时序）。宿主能力全部注入，不依赖 DOM。
 */

import { describe, expect, it, vi } from "vitest";

import { createEdgeScrollDriver } from "./edgeScrollDriver";

/** 视口：left=100、right=1100（宽 1000）。 */
const BOUNDS = { left: 100, right: 1100 };
const FRAME_MS = 1000 / 60;
/** 速度取 1080 px/秒 ⇒ 60Hz 下每帧满速 18px（与旧手感一致）。 */
const SPEED = 1080;
/** 左缘带宽内、离左边界 4px（比例 28/32 = 0.875）。 */
const NEAR_LEFT = 100 + 4;
/** 右缘带宽内、离右边界 4px。 */
const NEAR_RIGHT = 1100 - 4;

/**
 * 搭一个受控宿主：滚动位置、上界、时间源、rAF 全部可控。
 *
 * rAF 不自动执行 —— 用例显式调用 `flushFrame()` 推进一帧，从而能断言"循环是否
 * 继续"而不受真实帧率影响。
 */
function createHarness(initialScrollLeft = 0, maxScrollLeft = 10_000) {
    let scrollLeft = initialScrollLeft;
    let max = maxScrollLeft;
    let clockMs = 0;
    let nextHandle = 1;
    const pending = new Map<number, () => void>();
    const onScrolled = vi.fn();

    const driver = createEdgeScrollDriver({
        getBounds: () => BOUNDS,
        getScrollLeft: () => scrollLeft,
        setScrollLeft: (next) => {
            scrollLeft = next;
        },
        getMaxScrollLeft: () => max,
        onScrolled,
        maxSpeedPxPerSec: SPEED,
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
        getScrollLeft: () => scrollLeft,
        setMaxScrollLeft: (value: number) => {
            max = value;
        },
        /** 只推进时间（不执行挂起帧）：用于制造"两次 step 之间的真实间隔"。 */
        advanceClock(deltaMs: number) {
            clockMs += deltaMs;
        },
        /** 推进时间并执行当前挂起的那一帧（模拟 rAF 回调）。 */
        flushFrame(deltaMs = FRAME_MS) {
            clockMs += deltaMs;
            const callbacks = [...pending.values()];
            pending.clear();
            for (const cb of callbacks) cb();
        },
        pendingFrameCount: () => pending.size,
    };
}

describe("createEdgeScrollDriver.step（事件驱动，选择工具用）", () => {
    it("指针不在边缘带内 → 不滚、不回调", () => {
        const h = createHarness();
        expect(h.driver.step(600)).toBe(false);
        expect(h.getScrollLeft()).toBe(0);
        expect(h.onScrolled).not.toHaveBeenCalled();
    });

    it("指针在左缘 → 向负方向滚；已在下界时原地不动", () => {
        // 起始位置就是 0（左界）：滚不动，也不该回调。
        const atLeftEdge = createHarness(0);
        expect(atLeftEdge.driver.step(NEAR_LEFT)).toBe(false);
        expect(atLeftEdge.onScrolled).not.toHaveBeenCalled();

        // 从中间位置起滚：负方向生效。
        const h = createHarness(500);
        expect(h.driver.step(NEAR_LEFT)).toBe(true);
        expect(h.getScrollLeft()).toBeLessThan(500);
    });

    it("指针在右缘 → 向正方向滚，并回调同一指针位置", () => {
        const h = createHarness();
        expect(h.driver.step(NEAR_RIGHT)).toBe(true);
        expect(h.getScrollLeft()).toBeGreaterThan(0);
        expect(h.onScrolled).toHaveBeenCalledTimes(1);
        expect(h.onScrolled).toHaveBeenCalledWith(NEAR_RIGHT);
    });

    it("步长与真实间隔成正比（与事件频率无关）", () => {
        /** 第一次 step 建立时钟，隔 `gapMs` 再 step 一次，返回第二次的位移。 */
        const secondStepPx = (gapMs: number): number => {
            const h = createHarness();
            h.driver.step(NEAR_RIGHT);
            const base = h.getScrollLeft();
            h.advanceClock(gapMs);
            h.driver.step(NEAR_RIGHT);
            return h.getScrollLeft() - base;
        };

        const at60HzEvents = secondStepPx(FRAME_MS);
        const at120HzEvents = secondStepPx(FRAME_MS / 2);
        // 同样的墙钟时间必须滚过同样的距离 —— 高轮询率鼠标不再快一倍。
        expect(at120HzEvents).toBeCloseTo(at60HzEvents / 2, 6);
        expect(at120HzEvents * 2).toBeCloseTo(at60HzEvents, 6);
    });

    it("新手势第一步按 1/60 秒起步（长间隔不会跳一大步）", () => {
        const h = createHarness();
        h.driver.step(NEAR_RIGHT); // 第一次：无历史间隔
        const firstStep = h.getScrollLeft();
        expect(firstStep).toBeGreaterThan(0);

        // 隔很久之后的第二次：间隔远超手势窗口 → 仍按 1/60 秒起步，而不是补一大段。
        h.advanceClock(60_000);
        h.driver.step(NEAR_RIGHT);
        expect(h.getScrollLeft() - firstStep).toBeCloseTo(firstStep, 6);
    });

    it("被滚动上界钳住（不越过工程末端）", () => {
        const h = createHarness(0, 5);
        h.driver.step(NEAR_RIGHT);
        expect(h.getScrollLeft()).toBe(5);
        // 已到上界：再 step 不产生位移 → 不回调。
        expect(h.driver.step(NEAR_RIGHT)).toBe(false);
    });
});

describe("createEdgeScrollDriver.track（rAF 驱动，绘制类工具用）", () => {
    it("指针停在边缘不动时持续滚动（停在原地也滚）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);
        // 滚动发生在 rAF 回调里（每帧至多一次位置写入），而不是 track 当场。
        expect(h.getScrollLeft()).toBe(0);

        h.flushFrame();
        const afterFirst = h.getScrollLeft();
        expect(afterFirst).toBeGreaterThan(0);

        // 指针位置**不再变化**，只推进帧 —— 视图必须继续滚。
        h.flushFrame();
        h.flushFrame();
        expect(h.getScrollLeft()).toBeGreaterThan(afterFirst);
        expect(h.driver.isRunning()).toBe(true);
    });

    it("指针离开边缘带后循环自行结束", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);

        h.driver.track(600); // 指针回到视口中央
        h.flushFrame();
        expect(h.driver.isRunning()).toBe(false);
        expect(h.pendingFrameCount()).toBe(0);
    });

    it("同一帧内多次 track 只保留一个挂起帧（不重复排程）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.driver.track(NEAR_RIGHT);
        h.driver.track(NEAR_RIGHT);
        expect(h.pendingFrameCount()).toBe(1);
    });

    it("每帧至多一次位置写入（高采样率数位笔不放大开销）", () => {
        const h = createHarness();
        // 同一帧内 20 次 pointermove：仍只排一个帧回调。
        for (let i = 0; i < 20; i += 1) h.driver.track(NEAR_RIGHT);
        expect(h.pendingFrameCount()).toBe(1);
        h.flushFrame();
        // 一帧只滚一次的量（与单次 track 相同）。
        const oneFrame = h.getScrollLeft();
        const single = createHarness();
        single.driver.track(NEAR_RIGHT);
        single.flushFrame();
        expect(oneFrame).toBeCloseTo(single.getScrollLeft(), 9);
    });

    it("滚动回调让调用方按滚动后的投影重算（回调次数 = 真正滚动的帧数）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame(); // 第 1 帧
        h.flushFrame(); // 第 2 帧
        h.flushFrame(); // 第 3 帧
        expect(h.onScrolled).toHaveBeenCalledTimes(3);
        for (const call of h.onScrolled.mock.calls) {
            expect(call[0]).toBe(NEAR_RIGHT);
        }
    });

    it("到达滚动上界后不再回调，循环也自行结束（不停白读布局）", () => {
        const h = createHarness(0, 5);
        h.driver.track(NEAR_RIGHT);
        h.flushFrame(); // 第 1 帧：滚到上界，本帧确有位移 → 还会再排一帧
        expect(h.getScrollLeft()).toBe(5);
        expect(h.driver.isRunning()).toBe(true);

        h.flushFrame(); // 第 2 帧：已无余量 → 循环结束
        const callsAtLimit = h.onScrolled.mock.calls.length;
        expect(h.driver.isRunning()).toBe(false);

        // 停在边界按住不动：没有位移就不该继续排帧，也不该再回调。
        h.flushFrame();
        h.flushFrame();
        expect(h.onScrolled).toHaveBeenCalledTimes(callsAtLimit);
        expect(h.getScrollLeft()).toBe(5);
    });

    it("已在左界、指针停在左缘：同样不空转", () => {
        const h = createHarness(0);
        h.driver.track(NEAR_LEFT);
        h.flushFrame();
        expect(h.getScrollLeft()).toBe(0);
        expect(h.driver.isRunning()).toBe(false);
    });
});

describe("createEdgeScrollDriver.stop", () => {
    it("取消挂起帧并停止循环（松手后不再滚）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        expect(h.driver.isRunning()).toBe(true);
        expect(h.getScrollLeft()).toBeGreaterThan(0);

        h.driver.stop();
        expect(h.driver.isRunning()).toBe(false);
        expect(h.pendingFrameCount()).toBe(0);

        const frozen = h.getScrollLeft();
        h.flushFrame();
        h.flushFrame();
        expect(h.getScrollLeft()).toBe(frozen);
    });

    it("幂等：重复 stop 不抛错、不残留挂起帧", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.driver.stop();
        expect(() => h.driver.stop()).not.toThrow();
        expect(h.pendingFrameCount()).toBe(0);
    });

    it("stop 后重新 track 能正常恢复（新手势）", () => {
        const h = createHarness();
        h.driver.track(NEAR_RIGHT);
        h.flushFrame();
        h.driver.stop();
        const before = h.getScrollLeft();

        h.driver.track(NEAR_RIGHT);
        expect(h.driver.isRunning()).toBe(true);
        h.flushFrame();
        expect(h.getScrollLeft()).toBeGreaterThan(before);
    });

    it("容器未挂载（getBounds 为 null）时不排程、不抛错", () => {
        const driver = createEdgeScrollDriver({
            getBounds: () => null,
            getScrollLeft: () => 0,
            setScrollLeft: () => {},
            getMaxScrollLeft: () => 1000,
        });
        expect(() => driver.track(NEAR_RIGHT)).not.toThrow();
        expect(driver.isRunning()).toBe(false);
        expect(driver.step(NEAR_RIGHT)).toBe(false);
    });
});
