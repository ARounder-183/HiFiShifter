import { test } from "vitest";

import {
    timelineViewportSync,
    timelineViewportNativeToState,
    timelineViewportStateToNative,
} from "./timelineViewportSync.ts";

test("utils/timelineViewportSync.test.ts scripted checks", async () => {
    let checks = 0;

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        checks += 1;
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    timelineViewportSync.reset();

    // seeded 语义：reset 后未播种；任何 setViewport 调用（即使值与模块
    // 默认一致——启动时恰为 scrollLeft=0 / pxPerSec=150 的场景）都视为
    // 拥有方（时间轴）已播种。参数编辑器在 isSeeded() 为 false 时必须
    // 拒绝应用，否则启动瞬间会先把一帧默认视口画出来（"一闪"）。
    {
        assertEqual(timelineViewportSync.isSeeded(), false, "reset clears seeded");
        timelineViewportSync.setViewport({ scrollLeft: 0, pxPerSec: 150 });
        assertEqual(
            timelineViewportSync.isSeeded(),
            true,
            "setViewport seeds even with default values",
        );
    }
    timelineViewportSync.reset();
    {
        timelineViewportSync.setViewport({ scrollLeft: 321 });
        assertEqual(timelineViewportSync.isSeeded(), true, "setViewport seeds");
    }
    timelineViewportSync.reset();
    assertEqual(timelineViewportSync.isSeeded(), false, "reset clears seeded again");

    // Atomic update: zoom and scroll are broadcast together so consumers never
    // observe a "new scroll + old zoom" intermediate state.
    {
        let notifications = 0;
        const unsubscribe = timelineViewportSync.subscribe(() => {
            notifications += 1;
        });
        timelineViewportSync.setViewport({ scrollLeft: 321, pxPerSec: 456 });
        assertEqual(notifications, 1, "setViewport broadcasts exactly once");
        assertEqual(timelineViewportSync.get().scrollLeft, 321, "atomic scrollLeft stored");
        assertEqual(timelineViewportSync.get().pxPerSec, 456, "atomic pxPerSec stored");

        timelineViewportSync.setViewport({ scrollLeft: 321.2, pxPerSec: 456.0000000001 });
        assertEqual(notifications, 1, "sub-threshold atomic update is suppressed");
        unsubscribe();
    }

    // 订阅与推送
    {
        let notifications = 0;
        const unsubscribe = timelineViewportSync.subscribe(() => {
            notifications += 1;
        });
        timelineViewportSync.setViewport({ scrollLeft: 120 });
        timelineViewportSync.setViewport({ pxPerSec: 240 });
        assertEqual(notifications, 2, "subscribe receives both updates");
        assertEqual(timelineViewportSync.get().scrollLeft, 120, "scrollLeft stored");
        assertEqual(timelineViewportSync.get().pxPerSec, 240, "pxPerSec stored");
        unsubscribe();
        timelineViewportSync.setViewport({ scrollLeft: 200 });
        assertEqual(notifications, 2, "unsubscribed listener stops receiving");
    }

    // 小幅度变化被抑制，避免同步环路抖动
    {
        timelineViewportSync.reset();
        timelineViewportSync.setViewport({ scrollLeft: 100 });
        let notifications = 0;
        const unsubscribe = timelineViewportSync.subscribe(() => {
            notifications += 1;
        });
        timelineViewportSync.setViewport({ scrollLeft: 100.2 });
        timelineViewportSync.setViewport({ scrollLeft: 100.6 });
        timelineViewportSync.setViewport({ pxPerSec: 150.0000000001 });
        assertEqual(notifications, 1, "sub-pixel scroll changes are suppressed");
        assertEqual(timelineViewportSync.get().scrollLeft, 100.6, "larger scroll change applied");
        assertEqual(timelineViewportSync.get().pxPerSec, 150, "tiny zoom change suppressed");
        unsubscribe();
    }

    // reset 恢复默认
    {
        timelineViewportSync.reset();
        assertEqual(timelineViewportSync.get().scrollLeft, 0, "reset scrollLeft");
        assertEqual(timelineViewportSync.get().pxPerSec, 150, "reset pxPerSec");
    }

    // 偏移换算：同一轨道坐标在参数编辑器中换算后按全局位置对齐
    {
        const offset = 206;
        const storeScrollLeft = 500;
        const state = timelineViewportNativeToState(storeScrollLeft, offset);
        assertEqual(state, 294, "native -> state subtracts the track-header offset");
        assertEqual(
            timelineViewportStateToNative(state, offset),
            storeScrollLeft,
            "state -> native restores the shared track coordinate",
        );
        // 工程起点（store=0）在参数编辑器中对应负的绘制坐标（工程起点左侧为空白区）
        assertEqual(
            timelineViewportNativeToState(0, offset),
            -offset,
            "project start maps to negative drawing offset for alignment",
        );
    }

    void checks;
});
