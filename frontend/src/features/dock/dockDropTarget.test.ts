import { test } from "vitest";

import {
    clampFloatRect,
    dropPreviewRect,
    dropZoneToSide,
    pickDropTarget,
    pointInRect,
    resolveDropZone,
    snapFloatPosition,
} from "./dockDropTarget.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

const RECT = { x: 100, y: 100, w: 400, h: 300 };

test("features/dock/dockDropTarget.test.ts scripted checks", async () => {
    // ── 包含判定 ────────────────────────────────────────────────
    {
        assertEqual(pointInRect(RECT, 300, 250), true, "inside");
        assertEqual(pointInRect(RECT, 99, 250), false, "just outside left");
        assertEqual(pointInRect(RECT, 500, 400), true, "bottom-right corner is inclusive");
    }

    // ── 中央区 → center ─────────────────────────────────────────
    {
        assertEqual(resolveDropZone(RECT, { x: 300, y: 250 }, 28), "center", "middle is center");
    }

    // ── 四边 → 对应方向 ─────────────────────────────────────────
    {
        assertEqual(resolveDropZone(RECT, { x: 110, y: 250 }, 28), "left", "near left edge");
        assertEqual(resolveDropZone(RECT, { x: 490, y: 250 }, 28), "right", "near right edge");
        assertEqual(resolveDropZone(RECT, { x: 300, y: 110 }, 28), "top", "near top edge");
        assertEqual(resolveDropZone(RECT, { x: 300, y: 390 }, 28), "bottom", "near bottom edge");
    }

    // ── 角落归给更近的那条边 ─────────────────────────────────────
    {
        // 左上角：到左边 4px、到上边 6px → 左边更近
        assertEqual(
            resolveDropZone(RECT, { x: 104, y: 106 }, 28),
            "left",
            "corner picks nearest edge",
        );
    }

    // ── 矩形外 → null ───────────────────────────────────────────
    {
        assertEqual(resolveDropZone(RECT, { x: 0, y: 0 }, 28), null, "outside yields null");
        assertEqual(
            resolveDropZone({ x: 0, y: 0, w: 0, h: 0 }, { x: 0, y: 0 }, 28),
            null,
            "degenerate rect",
        );
    }

    // ── 窄条：感应带被钳制，中央区仍然存在 ────────────────────────
    {
        // 高 20px 的折叠标签条，感应带 28 会被钳到 20*0.4 = 8px。
        const strip = { x: 0, y: 0, w: 600, h: 20 };
        assertEqual(
            resolveDropZone(strip, { x: 300, y: 10 }, 28),
            "center",
            "thin strip keeps a usable centre",
        );
    }

    // ── 多个 Zone：矩形互不重叠，取命中者 ────────────────────────
    {
        const zones = [
            { zoneId: "left", rect: { x: 0, y: 0, w: 500, h: 400 } },
            { zoneId: "right", rect: { x: 504, y: 0, w: 500, h: 400 } },
        ];
        assertEqual(pickDropTarget(zones, { x: 200, y: 200 })?.zoneId, "left", "picks left zone");
        assertEqual(pickDropTarget(zones, { x: 700, y: 200 })?.zoneId, "right", "picks right zone");
        assertEqual(
            pickDropTarget(zones, { x: 502, y: 200 }),
            null,
            "splitter gap matches nothing",
        );
    }

    // ── 重叠时取面积更小者（嵌套容器的兜底）──────────────────────
    {
        const zones = [
            { zoneId: "outer", rect: { x: 0, y: 0, w: 1000, h: 800 } },
            { zoneId: "inner", rect: { x: 100, y: 100, w: 200, h: 200 } },
        ];
        assertEqual(pickDropTarget(zones, { x: 150, y: 150 })?.zoneId, "inner", "smallest wins");
    }

    // ── center/float 不是"某一侧" ───────────────────────────────
    {
        assertEqual(dropZoneToSide("center"), null, "center has no side");
        assertEqual(dropZoneToSide("float"), null, "float has no side");
        assertEqual(dropZoneToSide("left"), "left", "left maps through");
    }

    // ── 预览框与真实落点同源 ────────────────────────────────────
    {
        assertEqual(dropPreviewRect(RECT, "center", 4), RECT, "center preview covers the zone");
        assertEqual(dropPreviewRect(RECT, "float", 4), null, "float has no preview");

        const left = dropPreviewRect(RECT, "left", 4);
        assertEqual([left!.x, left!.w], [100, 198], "left half preview");
        const right = dropPreviewRect(RECT, "right", 4);
        assertEqual([right!.x, right!.w], [302, 198], "right half preview");
        const top = dropPreviewRect(RECT, "top", 4);
        assertEqual([top!.y, top!.h], [100, 148], "top half preview");
        const bottom = dropPreviewRect(RECT, "bottom", 4);
        assertEqual([bottom!.y, bottom!.h], [252, 148], "bottom half preview");
    }

    // ── 浮动窗夹紧：标题栏必须留在视口内 ─────────────────────────
    {
        const viewport = { w: 1200, h: 800 };
        // 完全越界到右下 → 拉回来，标题栏可见
        const pulled = clampFloatRect({ x: 5000, y: 5000, w: 400, h: 300 }, viewport, 26);
        assertEqual(pulled.x <= viewport.w - 48, true, "x pulled into reach");
        assertEqual(pulled.y <= viewport.h - 26, true, "title bar stays visible");

        // 允许向左/上适度溢出，但不能整个消失
        const overLeft = clampFloatRect({ x: -2000, y: -50, w: 400, h: 300 }, viewport, 26);
        assertEqual(overLeft.x >= -(400 - 48), true, "left overhang bounded");
        assertEqual(overLeft.y, 0, "y never goes above the viewport");

        // 尺寸超过视口时被收敛
        const huge = clampFloatRect({ x: 0, y: 0, w: 9999, h: 9999 }, viewport, 26);
        assertEqual([huge.w, huge.h], [1200, 800], "size clamped to viewport");
    }

    // ── 吸附 ───────────────────────────────────────────────────
    {
        const viewport = { w: 1200, h: 800 };
        const others = [{ x: 400, y: 0, w: 300, h: 200 }];

        // 靠近左边缘 → 吸到 0
        assertEqual(
            snapFloatPosition({ x: 5, y: 300, w: 200, h: 200 }, others, viewport, 12),
            { x: 0, y: 300 },
            "snaps to viewport left",
        );

        // 靠近另一个窗体的右边缘 → 贴合
        assertEqual(
            snapFloatPosition({ x: 704, y: 300, w: 200, h: 200 }, others, viewport, 12),
            { x: 700, y: 300 },
            "snaps flush to a sibling's right edge",
        );

        // 阈值 0 = 不吸附
        assertEqual(
            snapFloatPosition({ x: 5, y: 300, w: 200, h: 200 }, others, viewport, 0),
            { x: 5, y: 300 },
            "zero threshold disables snapping",
        );

        // 远离所有吸附线 → 原样返回
        assertEqual(
            snapFloatPosition({ x: 320, y: 310, w: 200, h: 200 }, others, viewport, 12),
            { x: 320, y: 310 },
            "no snap when nothing is close",
        );
    }
});
