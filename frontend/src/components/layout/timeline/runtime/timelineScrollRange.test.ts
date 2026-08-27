import {
    resolveHorizontalWheelZoom,
    resolveCanvasViewportOffset,
    resolvePlayheadZoomScrollLeft,
    resolveTimelineScrollRange,
} from "./timelineScrollRange.ts";
import {
    timelineViewportNativeToState,
    timelineViewportStateToNative,
} from "../../../../utils/timelineViewportSync.ts";

let checks = 0;

function assertNear(actual: number, expected: number, label: string): void {
    checks += 1;
    if (Math.abs(actual - expected) > 1e-6) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

// A viewport-sized canvas renders local coordinates (world - requested
// scroll). Its wrapper lives inside the scrolled content, so the wrapper must
// be placed at the actual native scroll offset. Using only the rounding
// residual leaves the whole canvas near content x=0 and clips appear to jump
// out of view after zooming.
{
    const result = resolveCanvasViewportOffset({
        requestedScrollLeft: 2456.789,
        actualScrollLeft: 2456,
        viewportWidth: 987.654,
    });
    assertNear(
        result.leftPx,
        2456,
        "canvas wrapper follows native scroll position beyond one viewport",
    );
    assertNear(
        result.localScrollLeftPx,
        2456.789,
        "canvas keeps the exact requested local scroll position",
    );
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

// 参数编辑器同步时，播放光标以轨道视图屏幕位置为基准：
// 绘制坐标视口多出的偏移不能把“光标在画面内”误判为左偏。
{
    const offset = 200;
    const scrollLeft = resolvePlayheadZoomScrollLeft({
        playheadSec: 6,
        basePxPerSec: 100,
        baseScrollLeft: 300,
        nextPxPerSec: 110,
        viewportWidth: 1000,
        viewportOffsetPx: offset,
    });
    // 轨道坐标下光标屏幕 x = 6*100 - (300+200) = 100；
    // 缩放后参数编辑器绘制 scrollLeft = 6*110 - 100 - 200 = 360。
    assertNear(scrollLeft, 360, "synced playhead zoom preserves its shared screen position");
}

// 共享滚轮缩放：鼠标锚点保持不动
{
    const zoom = resolveHorizontalWheelZoom({
        factor: 1.1,
        basePxPerSec: 100,
        baseScrollLeft: 0,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 250,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
    });
    assertNear(zoom!.nextPxPerSec, 110, "zoom in scale");
    // 锚点秒 = 250/100 = 2.5，缩放后 scrollLeft = 2.5*110 - 250 = 25
    assertNear(zoom!.nextScrollLeft, 25, "mouse anchor stays fixed");
}

// 共享滚轮缩放：播放光标在画面内时保持其当前位置
{
    const zoom = resolveHorizontalWheelZoom({
        factor: 1.1,
        basePxPerSec: 100,
        baseScrollLeft: 400,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: true,
        playheadSec: 10,
        anchorScreenX: 999,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
    });
    // 光标屏幕 x = 600，缩放后保持 600
    assertNear(zoom!.nextScrollLeft, 10 * 110 - 600, "playhead anchored while visible");
}

// 共享滚轮缩放：播放光标不在画面内时居中
{
    const zoom = resolveHorizontalWheelZoom({
        factor: 0.9,
        basePxPerSec: 100,
        baseScrollLeft: 1200,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: true,
        playheadSec: 10,
        anchorScreenX: 0,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
    });
    assertNear(zoom!.nextScrollLeft, 10 * 90 - 500, "off-screen playhead centered on zoom");
}

// 参数编辑器同步时允许负绘制坐标（最多 -偏移）：
// 接近工程起点放大时不会把结果钳到 0 造成“放大 + 右移”。
{
    const zoom = resolveHorizontalWheelZoom({
        factor: 1.1,
        basePxPerSec: 100,
        baseScrollLeft: -206,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 200,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
        minScrollLeft: -206,
    });
    assertNear(zoom!.nextScrollLeft, -206, "synced param zoom keeps negative drawing offset");
}

// 轨道视图默认仍钳制在 0（不出现负滚动）
{
    const zoom = resolveHorizontalWheelZoom({
        factor: 0.9,
        basePxPerSec: 100,
        baseScrollLeft: 0,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 200,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
    });
    assertNear(zoom!.nextScrollLeft, 0, "track zoom clamps to zero minimum");
}

// 轨道视图端到端：鼠标锚点的“世界秒”必须在缩放前后保持不变。
// 这正是水平缩放时 clip 出现微量偏移的坐标系不变量。
{
    const basePxPerSec = 123.456;
    const baseScrollLeft = 456.789;
    const anchorScreenX = 321.654;
    const totalSec = 100;
    const viewportWidth = 987.654;
    const anchorWorldSec = (baseScrollLeft + anchorScreenX) / basePxPerSec;

    for (const factor of [1.1, 0.9]) {
        const zoom = resolveHorizontalWheelZoom({
            factor,
            basePxPerSec,
            baseScrollLeft,
            totalSec,
            viewportWidth,
            playheadZoomEnabled: false,
            playheadSec: null,
            anchorScreenX,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
        });
        assertNear(
            (zoom!.nextScrollLeft + anchorScreenX) / zoom!.nextPxPerSec,
            anchorWorldSec,
            `mouse anchor world stays fixed through track zoom (factor ${factor})`,
        );
    }
}

// 光标位于同步空白区（世界坐标 < 0）：锚点钳制到工程起点，纯缩放不产生水平移动
{
    const zoomIn = resolveHorizontalWheelZoom({
        factor: 1.1,
        basePxPerSec: 100,
        baseScrollLeft: -206,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 50,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
        minScrollLeft: -206,
        anchorOffsetPx: 206,
    });
    assertNear(zoomIn!.nextScrollLeft, -206, "zoom in at project start keeps the view pure");

    const zoomOut = resolveHorizontalWheelZoom({
        factor: 0.9,
        basePxPerSec: 100,
        baseScrollLeft: -206,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 50,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
        minScrollLeft: -206,
        anchorOffsetPx: 206,
    });
    assertNear(zoomOut!.nextScrollLeft, -206, "zoom out at project start keeps the view pure");
}

// 已滚入工程后（S>0），光标在空白对齐区时：
// 轨道视图左缘对应的世界位置在两个视图中保持固定（纯缩放）。
{
    const baseState = -100;
    const offset = 206;
    const anchorWorld = (baseState + offset) / 100;

    const zoomIn = resolveHorizontalWheelZoom({
        factor: 1.1,
        basePxPerSec: 100,
        baseScrollLeft: baseState,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 50,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
        minScrollLeft: -offset,
        anchorOffsetPx: offset,
    });
    assertNear(
        (zoomIn!.nextScrollLeft + offset) / 110,
        anchorWorld,
        "zoom in keeps the shared anchor world fixed",
    );

    const zoomOut = resolveHorizontalWheelZoom({
        factor: 0.9,
        basePxPerSec: 100,
        baseScrollLeft: baseState,
        totalSec: 10,
        viewportWidth: 1000,
        playheadZoomEnabled: false,
        playheadSec: null,
        anchorScreenX: 50,
        minPxPerSec: 0.5,
        maxPxPerSec: 8000,
        minScrollLeft: -offset,
        anchorOffsetPx: offset,
    });
    assertNear(
        (zoomOut!.nextScrollLeft + offset) / 90,
        anchorWorld,
        "zoom out keeps the shared anchor world fixed",
    );
}

// 端到端：参数编辑器同步缩放的完整换算链（共享函数 → 原生 → 绘制）
// 光标位于工程内容内：光标对应的世界点必须保持固定（纯缩放）。
{
    const offset = 200;
    const state = 500;
    const cursorX = 500;
    const px = 100;
    const anchorWorld = (state + cursorX) / px;

    for (const factor of [1.1, 0.9]) {
        const zoom = resolveHorizontalWheelZoom({
            factor,
            basePxPerSec: px,
            baseScrollLeft: state,
            totalSec: 100,
            viewportWidth: 1000,
            playheadZoomEnabled: false,
            playheadSec: null,
            anchorScreenX: cursorX,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
            minScrollLeft: -offset,
            anchorOffsetPx: offset,
        });
        const native = timelineViewportStateToNative(zoom!.nextScrollLeft, offset);
        const nextState = timelineViewportNativeToState(native, offset);
        assertNear(
            (nextState + cursorX) / zoom!.nextPxPerSec,
            anchorWorld,
            `param cursor world stays fixed (factor ${factor})`,
        );
    }
}

// 端到端：光标位于空白对齐区时，轨道视图左缘对应的世界位置保持固定。
{
    const offset = 200;
    const state = -100;
    const cursorX = 50;
    const px = 100;
    const trackLeftWorld = (state + offset) / px;

    for (const factor of [1.1, 0.9]) {
        const zoom = resolveHorizontalWheelZoom({
            factor,
            basePxPerSec: px,
            baseScrollLeft: state,
            totalSec: 100,
            viewportWidth: 1000,
            playheadZoomEnabled: false,
            playheadSec: null,
            anchorScreenX: cursorX,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
            minScrollLeft: -offset,
            anchorOffsetPx: offset,
        });
        const native = timelineViewportStateToNative(zoom!.nextScrollLeft, offset);
        const nextState = timelineViewportNativeToState(native, offset);
        assertNear(
            (nextState + offset) / zoom!.nextPxPerSec,
            trackLeftWorld,
            `track-left anchor world stays fixed (factor ${factor})`,
        );
    }
}

console.log(`timelineScrollRange checks passed (${checks})`);
