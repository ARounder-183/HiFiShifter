import { beforeEach, test } from "vitest";

import {
    beginDockDrag,
    endDockDrag,
    getDockDragState,
    resetDockDragForTests,
    subscribeDockDrag,
    updateDockDrag,
} from "./dockDragStore.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

beforeEach(() => {
    resetDockDragForTests();
});

test("features/dock/dockDragStore.test.ts scripted checks", async () => {
    // ── 初始无拖拽 ──────────────────────────────────────────────
    assertEqual(getDockDragState(), null, "no drag in flight initially");

    // ── 开始拖拽即标记 started ──────────────────────────────────
    //
    // 落点覆盖层完全靠这个标志决定"要不要渲染预览"。早期实现把它写成 false 且
    // 再无处置真，结果是停靠能用、预览从不出现 —— 用户拖拽时看不到任何落点提示。
    // 这是本次修复的核心不变式，因此在这里钉死。
    {
        let notifications = 0;
        const unsubscribe = subscribeDockDrag(() => {
            notifications += 1;
        });

        beginDockDrag({
            mode: "tab",
            formId: "fileBrowser",
            panelId: "fileBrowser",
            pointerX: 100,
            pointerY: 200,
            dockIntent: false,
            floatRect: null,
        });

        const state = getDockDragState();
        assert(state !== null, "state exists right after beginDockDrag");
        assertEqual(state?.started, true, "beginDockDrag marks the drag as started");
        assertEqual(state?.target, null, "no drop target resolved yet");
        assertEqual(state?.mode, "tab", "mode recorded");
        assert(notifications >= 1, "subscribers are notified");

        // ── 更新落点：订阅者再次收到通知 ────────────────────────
        updateDockDrag({
            pointerX: 300,
            pointerY: 400,
            dockIntent: true,
            target: { zoneId: "z2", zone: "left", rect: { x: 0, y: 0, w: 400, h: 300 } },
        });
        const moved = getDockDragState();
        assertEqual(moved?.target?.zoneId, "z2", "target recorded");
        assertEqual(moved?.target?.zone, "left", "zone recorded");
        assertEqual(moved?.started, true, "still started after an update");
        assert(notifications >= 2, "update notifies subscribers");

        unsubscribe();
        endDockDrag();
        assertEqual(getDockDragState(), null, "endDockDrag clears the state");
    }

    // ── 未开始时更新是空操作（不会凭空造出一次拖拽）───────────────
    {
        updateDockDrag({ pointerX: 5, pointerY: 5 });
        assertEqual(getDockDragState(), null, "update without a session is a no-op");
    }

    // ── 浮窗搬运模式同样立即 started ────────────────────────────
    {
        beginDockDrag({
            mode: "float",
            formId: "notebook",
            panelId: "notebook",
            pointerX: 10,
            pointerY: 20,
            dockIntent: false,
            floatRect: { x: 10, y: 20, w: 320, h: 240 },
        });
        const state = getDockDragState();
        assertEqual(state?.started, true, "float drags start immediately too");
        assertEqual(state?.floatRect?.w, 320, "float rect carried into the drag state");
    }
});
