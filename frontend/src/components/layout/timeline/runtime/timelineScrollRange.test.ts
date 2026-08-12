import {
    resolvePlayheadZoomScrollLeft,
    resolveTimelineScrollRange,
} from "./timelineScrollRange.ts";

let checks = 0;

function assertNear(actual: number, expected: number, label: string): void {
    checks += 1;
    if (Math.abs(actual - expected) > 1e-6) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    checks += 1;
    if (actual !== expected) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

// 工程小于视口：右侧允许滚到工程右端，左侧不允许越过工程起点
{
    const range = resolveTimelineScrollRange({ contentWidth: 500, viewportWidth: 1000 });
    assertEqual(range.minScrollLeft, 0, "min is zero (project start at viewport left edge)");
    assertEqual(
        range.maxScrollLeft,
        500,
        "max reaches project end when project is smaller than viewport",
    );
    assertEqual(range.paddedContentWidth, 1500, "right-only extension covers native scroll range");
}

// 工程大于视口：最大滚动位置连续等于工程宽度（右侧仍可继续延长）
{
    const range = resolveTimelineScrollRange({ contentWidth: 2000, viewportWidth: 1000 });
    assertEqual(range.maxScrollLeft, 2000, "max equals content width when larger");
    assertEqual(range.minScrollLeft, 0, "min stays zero");
    assertEqual(range.paddedContentWidth, 3000, "right extension covers native scroll range");
}

// 工程刚好等于视口
{
    const range = resolveTimelineScrollRange({ contentWidth: 1000, viewportWidth: 1000 });
    assertEqual(range.maxScrollLeft, 1000, "max equals content at exact fit");
    assertEqual(range.minScrollLeft, 0, "min zero at exact fit");
    assertEqual(range.paddedContentWidth, 2000, "right extension at exact fit");
}

// 平滑性：逐步放大时，最大滚动位置不应在 content = viewport 附近出现骤降
{
    const before = resolveTimelineScrollRange({ contentWidth: 999, viewportWidth: 1000 });
    const exact = resolveTimelineScrollRange({ contentWidth: 1000, viewportWidth: 1000 });
    const after = resolveTimelineScrollRange({ contentWidth: 1001, viewportWidth: 1000 });
    assertEqual(before.maxScrollLeft, 999, "max just below viewport");
    assertEqual(exact.maxScrollLeft, 1000, "max at viewport");
    assertEqual(after.maxScrollLeft, 1001, "max just above viewport");
}

// 播放光标在画面内：保持其当前屏幕位置，不校正到鼠标
{
    const scrollLeft = resolvePlayheadZoomScrollLeft({
        playheadSec: 10,
        basePxPerSec: 100,
        baseScrollLeft: 400,
        nextPxPerSec: 110,
        viewportWidth: 1000,
    });
    // 光标当前屏幕 x = 10*100 - 400 = 600，缩放后保持 600
    assertNear(scrollLeft, 10 * 110 - 600, "playhead stays at its current screen position");
}

// 播放光标在画面外（左侧）：校正到画面正中心
{
    const scrollLeft = resolvePlayheadZoomScrollLeft({
        playheadSec: 10,
        basePxPerSec: 100,
        baseScrollLeft: 1200,
        nextPxPerSec: 110,
        viewportWidth: 1000,
    });
    // 光标当前屏幕 x = -200（画面外），缩放后应出现在 x = 500
    assertNear(scrollLeft, 10 * 110 - 500, "off-screen playhead is centered");
}

// 播放光标在画面外（右侧）：同样校正到画面正中心
{
    const scrollLeft = resolvePlayheadZoomScrollLeft({
        playheadSec: 10,
        basePxPerSec: 100,
        baseScrollLeft: -100,
        nextPxPerSec: 110,
        viewportWidth: 1000,
    });
    // 光标当前屏幕 x = 1100（画面外），缩放后应出现在 x = 500
    assertNear(scrollLeft, 10 * 110 - 500, "off-screen playhead on the right is centered");
}

console.log(`timelineScrollRange checks passed (${checks})`);
