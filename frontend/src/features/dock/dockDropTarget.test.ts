import { test } from "vitest";

import {
    buildRootEdgeZones,
    clampFloatRect,
    DOCK_ROOT_ZONE_ID,
    resolveFloatNearRect,
    resolveFloatRect,
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

    // ── 锚点浮窗：位置按当前视口推导 ──────────────────────────────
    //
    // "默认落在右下角"是语义而非坐标：布局创建那一刻量到的窗口尺寸未必是最终值，
    // 写死坐标会让浮窗停在偏高的位置，窗口缩放后也会跑偏。
    {
        const viewport = { w: 1920, h: 1080 };
        const anchored = {
            x: 0,
            y: 0,
            w: 460,
            h: 420,
            anchor: "bottom-right" as const,
            anchorMarginPx: 24,
        };
        assertEqual(
            resolveFloatRect(anchored, viewport),
            { x: 1436, y: 636, w: 460, h: 420 },
            "anchored float sits in the lower-right corner",
        );
        assertEqual(
            resolveFloatRect(anchored, { w: 1200, h: 800 }),
            { x: 716, y: 356, w: 460, h: 420 },
            "and follows the viewport when it changes",
        );
        assertEqual(
            resolveFloatRect(anchored, { w: 300, h: 200 }),
            { x: 24, y: 24, w: 460, h: 420 },
            "degenerate viewport falls back to the margin",
        );
        assertEqual(
            resolveFloatRect({ x: 10, y: 20, w: 300, h: 200 }, viewport),
            { x: 10, y: 20, w: 300, h: 200 },
            "a manually placed float keeps its coordinates",
        );
    }

    // ── 居中锚点：设置类面板（外观设置）的落点 ────────────────────
    //
    // 居中与右下角同样**按视口每帧解析**，所以窗口缩放后它仍在正中；
    // 视口小于窗体时退回边距原点，与右下角锚点的兜底一致（宁可盖住内容，
    // 也不要跑到屏幕外）。
    {
        const centered = {
            x: 0,
            y: 0,
            w: 900,
            h: 640,
            anchor: "center" as const,
        };
        assertEqual(
            resolveFloatRect(centered, { w: 1920, h: 1080 }),
            { x: 510, y: 220, w: 900, h: 640 },
            "centered float sits in the middle of the viewport",
        );
        assertEqual(
            resolveFloatRect(centered, { w: 1280, h: 800 }),
            { x: 190, y: 80, w: 900, h: 640 },
            "and stays centered when the viewport changes",
        );
        assertEqual(
            resolveFloatRect(centered, { w: 400, h: 300 }),
            { x: 24, y: 24, w: 900, h: 640 },
            "degenerate viewport falls back to the margin (same as bottom-right)",
        );
        // 居中锚点下偏移量仍然生效（多个居中浮窗错开用）
        assertEqual(
            resolveFloatRect(
                { ...centered, anchorOffsetX: -100, anchorOffsetY: 40 },
                { w: 1920, h: 1080 },
            ),
            { x: 410, y: 260, w: 900, h: 640 },
            "offset still applies on top of centering",
        );
    }

    // ── 锚点偏移：多个默认浮窗必须错开 ────────────────────────────
    //
    // 记事本（460×420）在右下角，撤销历史（420×420）声明 offsetX = -(460 + 24)：
    // 两者底边对齐、水平间隔恰好一个边距，绝不重叠。
    {
        const viewport = { w: 1920, h: 1080 };
        const notebook = resolveFloatRect(
            {
                x: 0,
                y: 0,
                w: 460,
                h: 420,
                anchor: "bottom-right" as const,
                anchorMarginPx: 24,
                anchorOffsetX: 0,
                anchorOffsetY: 0,
            },
            viewport,
        );
        const undo = resolveFloatRect(
            {
                x: 0,
                y: 0,
                w: 420,
                h: 420,
                anchor: "bottom-right" as const,
                anchorMarginPx: 24,
                anchorOffsetX: -(460 + 24),
                anchorOffsetY: 0,
            },
            viewport,
        );
        assertEqual(undo, { x: 992, y: 636, w: 420, h: 420 }, "offset float sits left");
        assertEqual(
            notebook.x - (undo.x + undo.w),
            24,
            "exactly one margin between the two default floats",
        );
        assertEqual(notebook.y, undo.y, "and they stay bottom-aligned");
        assertEqual(
            resolveFloatRect(
                {
                    x: 0,
                    y: 0,
                    w: 420,
                    h: 420,
                    anchor: "bottom-right" as const,
                    anchorMarginPx: 24,
                    anchorOffsetX: -2000,
                },
                { w: 300, h: 200 },
            ),
            { x: 24, y: 24, w: 420, h: 420 },
            "an offset still clamps to the margin on a degenerate viewport",
        );
    }

    // ── 靠近触发控件打开：落在控件正下方（放不下则翻到上方）──────────
    //
    // 用于"从撤销/重做按钮打开操作记录"：按钮在顶部工具条，面板落在它下方；
    // 靠近视口底部时翻到上方；贴右缘时左移夹紧。
    {
        const viewport = { w: 1280, h: 720 };
        const button = { x: 300, y: 40, w: 24, h: 24 };
        assertEqual(
            resolveFloatNearRect(button, { w: 420, h: 420 }, viewport),
            { x: 300, y: 72, w: 420, h: 420 },
            "a panel opened from a top toolbar button lands just below it",
        );
        // 下方空间不足（按钮在底部）→ 翻到上方。
        assertEqual(
            resolveFloatNearRect({ x: 300, y: 660, w: 24, h: 24 }, { w: 420, h: 420 }, viewport),
            { x: 300, y: 232, w: 420, h: 420 },
            "flips above when there is no room below",
        );
        // 贴右缘：水平夹回视口内（保留 8px 边距）。
        assertEqual(
            resolveFloatNearRect({ x: 1260, y: 40, w: 24, h: 24 }, { w: 420, h: 420 }, viewport),
            { x: 852, y: 72, w: 420, h: 420 },
            "clamps horizontally into the viewport",
        );
        // 比视口还高的面板：贴底，至少让标题条可见（而不是溢出到负坐标）。
        assertEqual(
            resolveFloatNearRect({ x: 100, y: 700, w: 24, h: 24 }, { w: 420, h: 2000 }, viewport),
            { x: 100, y: 8, w: 420, h: 704 },
            "an oversized panel is shrunk and kept inside",
        );
    }

    // ── 根级边缘带（贯通整侧的合成 Zone）────────────────────────
    {
        const root = { x: 0, y: 0, w: 1000, h: 800 };
        const zones = buildRootEdgeZones(root, 28);
        assertEqual(zones.length, 4, "four edge bands");

        // 细带贴根矩形外缘，左右带贯通全高、上下带让出左右两角（互不重叠）。
        assertEqual(zones[0].rect, { x: 0, y: 0, w: 28, h: 800 }, "left band");
        assertEqual(zones[1].rect, { x: 972, y: 0, w: 28, h: 800 }, "right band");
        assertEqual(zones[2].rect, { x: 28, y: 0, w: 944, h: 28 }, "top band");
        assertEqual(zones[3].rect, { x: 28, y: 772, w: 944, h: 28 }, "bottom band");

        // 每条带固定自己的部位、以整个根矩形为预览/提交基准。
        const sides = ["left", "right", "top", "bottom"] as const;
        for (const [index, side] of sides.entries()) {
            assertEqual(zones[index].fixedZone, side, `band ${side} has a fixed zone`);
            assertEqual(zones[index].previewRect, root, `band ${side} previews the root rect`);
            assertEqual(zones[index].zoneId, DOCK_ROOT_ZONE_ID, "all bands share the root id");
        }

        // 角落（两带交界）按"面积最小者优先"取更窄的那条：左上角命中左带。
        assertEqual(
            pickDropTarget(zones, { x: 5, y: 5 })?.fixedZone,
            "left",
            "corner resolves to the taller (smaller-area) band",
        );

        // 细带压过与之重叠的标签组大矩形：贴右缘的指针表达"拆整个停靠区"。
        const withTabset = [{ zoneId: "z1", rect: { x: 0, y: 0, w: 1000, h: 400 } }, ...zones];
        assertEqual(
            pickDropTarget(withTabset, { x: 990, y: 200 })?.fixedZone,
            "right",
            "the root band wins over the tabset at the shared outer edge",
        );
        // 感应带之外的标签组内部仍以标签组为准（它才是面积最小者）。
        assertEqual(
            pickDropTarget(withTabset, { x: 900, y: 200 })?.zoneId,
            "z1",
            "inside the tabset the ordinary zone still wins",
        );

        // 极小停靠区：感应带收缩，中央区不会被四条带吃光。
        const tiny = buildRootEdgeZones({ x: 0, y: 0, w: 40, h: 40 }, 28);
        assertEqual(tiny[0].rect.w, 10, "band shrinks on a tiny root (short side / 4)");
    }
});
