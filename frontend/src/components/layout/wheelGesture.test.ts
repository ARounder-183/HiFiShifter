import { test } from "vitest";

import { getParamEditorWheelAction, getTimelineWheelAction } from "./wheelGesture.ts";

test("components/layout/wheelGesture.test.ts scripted checks", async () => {
    let checks = 0;

    function assertEqual(actual: string, expected: string, label: string): void {
        checks += 1;
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    // 轨道视图：默认绑定（horizontalZoom = None）下，无修饰键的垂直滚轮 = 水平缩放
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: true,
        }),
        "horizontal-zoom",
        "timeline plain wheel zooms",
    );

    // 轨道视图：Shift = 左右平移
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: true,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "horizontal-scroll",
        "timeline shift wheel pans horizontally",
    );

    // 轨道视图：Alt = 垂直滚动，Ctrl = 垂直缩放
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: true,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "vertical-scroll",
        "timeline alt wheel scrolls vertically",
    );
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: true,
            horizontalZoomRequested: false,
        }),
        "vertical-zoom",
        "timeline ctrl wheel zooms vertically",
    );

    // 参数编辑器：无修饰键的垂直滚轮 = 水平缩放（回退行为，不依赖 horizontalZoom 绑定）
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalPanRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "horizontal-zoom",
        "param editor plain wheel zooms",
    );

    // 参数编辑器：Shift = 左右平移
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: true,
            verticalPanRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "horizontal-scroll",
        "param editor shift wheel pans horizontally",
    );

    // 参数编辑器：水平主导的滚轮 = 左右平移
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 120,
            deltaY: 0,
            horizontalScrollRequested: false,
            verticalPanRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "horizontal-scroll",
        "param editor horizontal wheel pans",
    );

    // 参数编辑器：Alt = 垂直平移，Ctrl = 垂直缩放
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalPanRequested: true,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
        }),
        "vertical-pan",
        "param editor alt wheel pans vertically",
    );
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalPanRequested: false,
            verticalZoomRequested: true,
            horizontalZoomRequested: false,
        }),
        "vertical-zoom",
        "param editor ctrl wheel zooms vertically",
    );

    // ── 悬停原生滚动条（modifier.scrollbarZoom 语境）──────────────────
    // 悬停竖直滚动条：无修饰键 = 竖直滚动，覆盖"默认滚轮 = 水平缩放"绑定
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: true,
            scrollbarZone: "vertical",
        }),
        "vertical-scroll",
        "timeline v-scrollbar plain wheel scrolls vertically",
    );
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalPanRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: true,
            scrollbarZone: "vertical",
        }),
        "vertical-pan",
        "param editor v-scrollbar plain wheel pans vertically",
    );
    // 悬停竖直滚动条 + 缩放修饰键 = 竖直缩放
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
            scrollbarZone: "vertical",
            scrollbarZoomRequested: true,
        }),
        "vertical-zoom",
        "timeline v-scrollbar modifier wheel zooms vertically",
    );
    // 悬停水平滚动条：无修饰键 = 水平滚动，修饰键 = 水平缩放
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: true,
            scrollbarZone: "horizontal",
        }),
        "horizontal-scroll",
        "timeline h-scrollbar plain wheel scrolls horizontally",
    );
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
            scrollbarZone: "horizontal",
            scrollbarZoomRequested: true,
        }),
        "horizontal-zoom",
        "timeline h-scrollbar modifier wheel zooms horizontally",
    );
    assertEqual(
        getParamEditorWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalPanRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
            scrollbarZone: "horizontal",
            scrollbarZoomRequested: true,
        }),
        "horizontal-zoom",
        "param editor h-scrollbar modifier wheel zooms horizontally",
    );
    // scrollbarZone 优先于 free-scroll 组合与一切绑定
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: true,
            verticalScrollRequested: true,
            verticalZoomRequested: false,
            horizontalZoomRequested: false,
            scrollbarZone: "vertical",
        }),
        "vertical-scroll",
        "scrollbar zone overrides free-scroll binding combo",
    );
    // scrollbarZone 缺省 / null：行为与旧版完全一致
    assertEqual(
        getTimelineWheelAction({
            deltaX: 0,
            deltaY: 120,
            horizontalScrollRequested: false,
            verticalScrollRequested: false,
            verticalZoomRequested: false,
            horizontalZoomRequested: true,
            scrollbarZone: null,
        }),
        "horizontal-zoom",
        "null scrollbar zone keeps legacy behavior",
    );

    void checks;
});
