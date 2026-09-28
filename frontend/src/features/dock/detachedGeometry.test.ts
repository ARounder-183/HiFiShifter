/**
 * 独立窗口几何换算自检。
 *
 * 【要钉死的契约】
 * 1. 浮窗 ↔ 独立窗口是**原地**转换：换算一次再换算回来必须回到原值（否则每拆一次
 *    窗口就漂一点，或每关一次浮窗就缩一圈 —— 棘轮效应）。
 * 2. 创建参数用的是**外框**位置 + **客户区**尺寸（见 `floatRectToWindowRect` 的说明）。
 * 3. 换算结果永远落在显示器可达范围内。
 */
import { test } from "vitest";

import {
    clampScreenRectToMonitor,
    floatRectToWindowRect,
    windowRectToFloatRect,
    type MainWindowFrame,
} from "./detachedGeometry";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

/** 主窗口：屏幕 (100, 50) 处，客户区 1280×720，标题栏/边框 8×31。 */
const FRAME: MainWindowFrame = {
    clientOriginX: 108,
    clientOriginY: 81,
    clientWidth: 1280,
    clientHeight: 720,
    frameInsetX: 8,
    frameInsetY: 31,
};

test("features/dock/detachedGeometry.test.ts scripted checks", () => {
    // ── 浮窗 → 独立窗口：外框位置 = 客户区原点 + 浮窗坐标 − 外框厚度 ──
    {
        assertEqual(
            floatRectToWindowRect({ x: 40, y: 60, w: 460, h: 420 }, FRAME),
            { x: 140, y: 110, w: 460, h: 420 },
            "detached window keeps the float in place (outer position, client size)",
        );
    }

    // ── 往返一致（原地转换）────────────────────────────────────
    {
        for (const float of [
            { x: 0, y: 0, w: 460, h: 420 },
            { x: 40, y: 60, w: 460, h: 420 },
            { x: 800, y: 300, w: 420, h: 420 },
            { x: -120, y: -30, w: 300, h: 200 },
        ]) {
            const win = floatRectToWindowRect(float, FRAME);
            // 独立窗口回报的客户区矩形 = 它自己的客户区原点 + 尺寸
            const clientRect = {
                x: win.x + FRAME.frameInsetX,
                y: win.y + FRAME.frameInsetY,
                w: win.w,
                h: win.h,
            };
            assertEqual(
                windowRectToFloatRect(clientRect, FRAME),
                float,
                `round trip is identity for ${JSON.stringify(float)}`,
            );
        }
    }

    // ── 显示器夹紧：整窗可见 / 退化为左上角可达 ──────────────────
    {
        const monitor = { x: 0, y: 0, w: 1920, h: 1080 };
        assertEqual(
            clampScreenRectToMonitor({ x: 500, y: 400, w: 460, h: 420 }, monitor),
            { x: 500, y: 400, w: 460, h: 420 },
            "a window that already fits is untouched",
        );
        assertEqual(
            clampScreenRectToMonitor({ x: 1800, y: 900, w: 460, h: 420 }, monitor),
            { x: 1452, y: 652, w: 460, h: 420 },
            "a window hanging off the bottom-right is pulled back in",
        );
        assertEqual(
            clampScreenRectToMonitor({ x: -300, y: -200, w: 460, h: 420 }, monitor),
            { x: 8, y: 8, w: 460, h: 420 },
            "a window off the top-left is pushed back in",
        );
        // 比显示器还大：取上界（左上角贴边），保证可达而不是被推到负坐标。
        assertEqual(
            clampScreenRectToMonitor({ x: 0, y: 0, w: 2400, h: 1400 }, monitor),
            { x: 8, y: 8, w: 2400, h: 1400 },
            "a window larger than the monitor stays reachable at the margin",
        );
        // 第二显示器（负坐标）同样成立。
        assertEqual(
            clampScreenRectToMonitor(
                { x: -1900, y: 100, w: 400, h: 300 },
                { x: -1920, y: 0, w: 1920, h: 1080 },
            ),
            { x: -1900, y: 100, w: 400, h: 300 },
            "a window on a left-hand monitor is left alone",
        );
    }
});
