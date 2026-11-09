/**
 * 参数曲线取数窗口的纯契约。
 *
 * 【要钉死什么】请求窗口可以因为内存预算裁掉两侧余量，但**绝不能裁掉可见区**。
 * 旧实现把窗口硬钳到 200_000 帧，在 4K / 最小缩放下（可见 ~960s）右端只到
 * 可见区中间，后段参数线因此没有数据。
 */
import { expect, test } from "vitest";

import { resolveParamFetchWindow } from "./paramFetchWindow";
import { timeToFrame } from "./utils";

test("远距滚动 + 宽视口：请求窗口必须覆盖可见区右端", () => {
    // 4K 窗口在 4 px/s 下约 960s 可见；模拟已经滚到 5000s。
    const window = resolveParamFetchWindow({
        visibleStartSec: 5000,
        visibleEndSec: 5960,
        framePeriodMs: 5,
    });
    const visibleStartFrame = timeToFrame(5000, 5);
    const visibleEndFrame = timeToFrame(5960, 5);
    expect(window.startFrame).toBeLessThanOrEqual(visibleStartFrame);
    expect(window.startFrame + window.frameCount).toBeGreaterThanOrEqual(visibleEndFrame + 1);
});

test("普通视口仍保留两侧余量", () => {
    const window = resolveParamFetchWindow({
        visibleStartSec: 100,
        visibleEndSec: 110,
        framePeriodMs: 5,
    });
    const visibleStartFrame = timeToFrame(100, 5);
    const visibleEndFrame = timeToFrame(110, 5);
    expect(window.startFrame).toBeLessThan(visibleStartFrame);
    expect(window.startFrame + window.frameCount).toBeGreaterThan(visibleEndFrame + 1);
    // 10s 视口、3 倍窗口 = 6000 帧左右；预算远大于它。
    expect(window.frameCount).toBeGreaterThanOrEqual(6000);
});

test("极端宽视口也不会把 frameCount 钳到可见区以内", () => {
    const window = resolveParamFetchWindow({
        visibleStartSec: 0,
        visibleEndSec: 10_000,
        framePeriodMs: 5,
    });
    const visibleFrames = timeToFrame(10_000, 5) + 1;
    expect(window.frameCount).toBeGreaterThanOrEqual(visibleFrames);
    expect(window.startFrame).toBe(0);
});

test("非法输入不产生负 startFrame / 0 frameCount", () => {
    const window = resolveParamFetchWindow({
        visibleStartSec: Number.NaN,
        visibleEndSec: Number.NaN,
        framePeriodMs: Number.NaN,
    });
    expect(window.startFrame).toBeGreaterThanOrEqual(0);
    expect(window.frameCount).toBeGreaterThan(0);
});
