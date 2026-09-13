import { test } from "vitest";

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

test("components/layout/timeline/runtime/timelineScrollRange.test.ts scripted checks", async () => {
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

    // 工程小于视口：右侧允许滚到工程右端，左侧不允许越过工程起点
    {
        const range = resolveTimelineScrollRange({ contentWidth: 500, viewportWidth: 1000 });
        assertEqual(range.minScrollLeft, 0, "min is zero (project start at viewport left edge)");
        assertEqual(
            range.maxScrollLeft,
            500,
            "max reaches project end when project is smaller than viewport",
        );
        assertEqual(
            range.paddedContentWidth,
            1500,
            "right-only extension covers native scroll range",
        );
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

    // 【两面板不变量】同步模式下两个面板共享同一个水平位置（= 轨道视图的原生
    // scrollLeft），因此可滚范围必须与「面板自己的视口宽」「同步偏移」都无关。
    //
    // 反例（2026-09-13 修正的缺陷）：参数编辑器曾把偏移做进自己的可滚范围
    // （`[0, 内容宽 + 偏移]`），于是它比轨道视图多出一段「只有自己能滚」的空白——滚到
    // 最右时轨道视图停在内容宽处（工程结尾落在轨道区左缘），参数编辑器还能再滚一个
    // 偏移量（工程结尾落到轨道区左缘以左）。用户看到的就是「工程结尾右侧的额外空白
    // 长度不同」，且两个面板在极端位置上不再对齐。
    {
        const PANEL_WIDTH = 1600;
        const TRACK_HEADER_W = 240;
        const KEYBOARD_W = 80;
        /** 轨道视图（滚动区）宽 = 面板宽 − 轨道头。 */
        const TIMELINE_VIEWPORT = PANEL_WIDTH - TRACK_HEADER_W;
        /** 参数编辑器（滚动区）宽 = 面板宽 − 键盘列。 */
        const PIANO_VIEWPORT = PANEL_WIDTH - KEYBOARD_W;
        /** 同步偏移的定义：两个滚动区左缘之差 = 轨道头宽 − 键盘列宽（恒为正）。 */
        const OFFSET = TRACK_HEADER_W - KEYBOARD_W;
        const CONTENT = 2456.789;

        const timeline = resolveTimelineScrollRange({
            contentWidth: CONTENT,
            viewportWidth: TIMELINE_VIEWPORT,
        });
        const piano = resolveTimelineScrollRange({
            contentWidth: CONTENT,
            viewportWidth: PIANO_VIEWPORT,
        });
        assertNear(
            piano.maxScrollLeft,
            timeline.maxScrollLeft,
            "shared max scroll left is independent of the panel viewport width",
        );
        assertNear(piano.maxScrollLeft, CONTENT, "shared max scroll left equals the content width");

        // 屏幕坐标：轨道区左缘 = 参数编辑器左缘 + 偏移。任意共享位置下，工程结尾
        // 必须落在同一屏幕 x —— 包括各自的最右端。
        const timelineAreaLeft = 1000;
        const pianoAreaLeft = timelineAreaLeft - OFFSET;
        for (const sharedScrollLeft of [0, CONTENT / 2, timeline.maxScrollLeft]) {
            const timelineEndScreenX = timelineAreaLeft + CONTENT - sharedScrollLeft;
            const pianoEndScreenX = pianoAreaLeft + CONTENT - (sharedScrollLeft - OFFSET);
            assertNear(
                pianoEndScreenX,
                timelineEndScreenX,
                `project end stays aligned at shared scroll ${sharedScrollLeft}`,
            );
        }

        // 工程结尾右侧的空白长度也一致（用户可见的判据）。
        const timelineBlank =
            timelineAreaLeft +
            TIMELINE_VIEWPORT -
            (timelineAreaLeft + CONTENT - timeline.maxScrollLeft);
        const pianoBlank =
            pianoAreaLeft +
            PIANO_VIEWPORT -
            (pianoAreaLeft + CONTENT - (piano.maxScrollLeft - OFFSET));
        assertNear(pianoBlank, timelineBlank, "trailing blank right of the project end matches");
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

    // 回归：播放光标在工程末端之后（点击右侧空白区/播放越过最后 clip 都会
    // 出现）时，缩放中心 = 光标真实位置，不得钳回工程末端。
    // 旧逻辑把 playheadSec 钳到 totalSec：中心被“钉”在工程末端，结果
    // scrollLeft 永远比允许的最大滚动位置偏左一个屏幕偏移，且画面内锚定
    // 分支锚定的是错误的内容点（工程末端而非播放光标）。
    {
        // 光标在画面外（右侧）：应居中于真实光标位置，结果被结果级 clamp
        // 收敛到允许的最大滚动位置（= 工程宽度），而不是停在其左侧。
        const zoom = resolveHorizontalWheelZoom({
            factor: 1.1,
            basePxPerSec: 100,
            baseScrollLeft: 2900,
            totalSec: 30,
            viewportWidth: 1000,
            playheadZoomEnabled: true,
            playheadSec: 45,
            anchorScreenX: 0,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
        });
        // 真实中心 45s 居中 → 45*110 - 500 = 4450，超出允许最大 30*110 =
        // 3300 → 钳到 3300（旧逻辑：中心钳到 30s → 画面内锚定 → 3200，
        // 比允许位置偏左 100px 且到不了右端）。
        assertNear(zoom!.nextScrollLeft, 3300, "beyond-end playhead zoom reaches the allowed max");
    }

    {
        // 光标在画面内（低缩放下工程末端的空白区可见）：锚定真实光标的
        // 屏幕位置，世界点保持固定。
        const basePxPerSec = 40;
        const baseScrollLeft = 1000;
        const playheadSec = 45;
        const zoom = resolveHorizontalWheelZoom({
            factor: 1.1,
            basePxPerSec,
            baseScrollLeft,
            totalSec: 30,
            viewportWidth: 1000,
            playheadZoomEnabled: true,
            playheadSec,
            anchorScreenX: 0,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
        });
        // 光标屏幕 x = 45*40 - 1000 = 800（画面内）→ 锚定：45*44 - 800 =
        // 1180，仍在允许范围内（30*44 = 1320）不触发钳制。
        assertNear(
            zoom!.nextScrollLeft,
            1180,
            "beyond-end playhead anchors at its screen position",
        );
        assertNear(
            (zoom!.nextScrollLeft + 800) / zoom!.nextPxPerSec,
            playheadSec,
            "anchored world point is the real playhead, not the clamped project end",
        );
    }

    // 防御：非法播放头（NaN）不产生 NaN 滚动值，按 0 处理。
    {
        const zoom = resolveHorizontalWheelZoom({
            factor: 1.1,
            basePxPerSec: 100,
            baseScrollLeft: 400,
            totalSec: 10,
            viewportWidth: 1000,
            playheadZoomEnabled: true,
            playheadSec: Number.NaN,
            anchorScreenX: 0,
            minPxPerSec: 0.5,
            maxPxPerSec: 8000,
        });
        assertNear(zoom!.nextScrollLeft, 0, "non-finite playhead falls back to zero scroll");
    }

    void checks;
});
