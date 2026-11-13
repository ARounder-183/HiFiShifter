/**
 * 拖动方向的模式表、循环顺序与两个 reducer。
 *
 * 【要钉死什么】
 * 1. 每张模式表都以 `free` 打头（循环从"自由"开始），出厂方向在表内；
 * 2. **没有开关态**：三个方向是同一根轴上的互斥取值，循环走完一圈回到原点，
 *    中途不经过任何"关"；
 * 3. 循环与直设是两条入口（`D` 键 / 拖拽中右键走循环，工具栏菜单走直设），
 *    两者写的是同一份字段；
 * 4. 非法方向被静默丢弃，且不会把界面挪到别的方向。
 */
import { expect, test } from "vitest";

import {
    DEFAULT_DRAG_DIRECTION_BY_TOOL,
    DRAG_DIRECTIONS_BY_TOOL,
    isDragDirectionForTool,
    nextDragDirection,
    resolveDragDirection,
    resolveDragDirectionTool,
} from "./dragDirection";
import { DRAG_DIRECTION_TOOLS } from "./sessionTypes";
import reducer, { cycleDragDirection, setDragDirection } from "./sessionSlice";

const base = reducer(undefined, { type: "@@INIT" });

test("模式表自洽：以自由打头、出厂方向在表内、方向集随工具收窄", () => {
    for (const tool of DRAG_DIRECTION_TOOLS) {
        const order = DRAG_DIRECTIONS_BY_TOOL[tool];
        expect(order[0], tool + " 的循环必须从 free 开始").toBe("free");
        expect(order, tool + " 的出厂方向必须在表内").toContain(
            DEFAULT_DRAG_DIRECTION_BY_TOOL[tool],
        );
    }
    // 绘制 / 颤音没有纵向约束：横向位移本身就是时间推进。
    expect(DRAG_DIRECTIONS_BY_TOOL.draw).toEqual(["free", "x-only"]);
    expect(DRAG_DIRECTIONS_BY_TOOL.vibrato).toEqual(["free", "x-only"]);
    expect(DRAG_DIRECTIONS_BY_TOOL.select).toEqual(["free", "x-only", "y-only"]);
    expect(isDragDirectionForTool("draw", "y-only")).toBe(false);
    expect(isDragDirectionForTool("select", "y-only")).toBe(true);
    expect(isDragDirectionForTool("select", "diagonal")).toBe(false);
});

test("循环：走完一圈回到原点，未知当前值从表头重新开始", () => {
    expect(nextDragDirection("select", "free")).toBe("x-only");
    expect(nextDragDirection("select", "x-only")).toBe("y-only");
    expect(nextDragDirection("select", "y-only")).toBe("free");
    expect(nextDragDirection("draw", "free")).toBe("x-only");
    expect(nextDragDirection("draw", "x-only")).toBe("free");
    // 旧配置里混进了本工具不支持的方向：从表头重来，而不是原地不动。
    expect(nextDragDirection("draw", "y-only")).toBe("free");
});

test("★ 左键循环覆盖全部方向：选择工具三步一圈，绘制工具两步一圈", () => {
    expect(base.selectDragDirection).toBe("y-only");

    const step1 = reducer(base, cycleDragDirection("select"));
    expect(step1.selectDragDirection).toBe("free");
    const step2 = reducer(step1, cycleDragDirection("select"));
    expect(step2.selectDragDirection).toBe("x-only");
    const step3 = reducer(step2, cycleDragDirection("select"));
    expect(step3.selectDragDirection).toBe("y-only");

    // 绘制工具没有仅 Y：两态一圈，不会卡在一个走不到的状态上。
    const draw1 = reducer(base, cycleDragDirection("draw"));
    expect(draw1.drawDragDirection).toBe("x-only");
    const draw2 = reducer(draw1, cycleDragDirection("draw"));
    expect(draw2.drawDragDirection).toBe("free");
});

test("循环只动它自己那一份方向，不碰别的工具、也不碰工具本身", () => {
    const cycled = reducer(base, cycleDragDirection("select"));
    expect(cycled.toolMode).toBe(base.toolMode);
    expect(cycled.drawToolMode).toBe(base.drawToolMode);
    expect(cycled.drawDragDirection).toBe(base.drawDragDirection);
    expect(cycled.lineVibratoDragDirection).toBe(base.lineVibratoDragDirection);

    // 直线与颤音共用一份方向：只动颤音那一份。
    const vibrato = reducer(base, cycleDragDirection("vibrato"));
    expect(vibrato.lineVibratoDragDirection).toBe("x-only");
    expect(vibrato.drawDragDirection).toBe(base.drawDragDirection);
});

test("直设：白名单拦下本工具不支持的方向", () => {
    expect(
        reducer(base, setDragDirection({ tool: "draw", direction: "y-only" })).drawDragDirection,
    ).toBe(base.drawDragDirection);
    expect(
        reducer(base, setDragDirection({ tool: "select", direction: "y-only" }))
            .selectDragDirection,
    ).toBe("y-only");
    expect(
        reducer(base, setDragDirection({ tool: "select", direction: "free" })).selectDragDirection,
    ).toBe("free");
});

test("解析：直线与颤音共用一份方向，选择 / 绘制各读各的", () => {
    const scope = {
        ...base,
        selectDragDirection: "y-only",
        drawDragDirection: "x-only",
        lineVibratoDragDirection: "free",
    } as const;

    expect(resolveDragDirectionTool({ toolMode: "select", drawToolMode: "draw" })).toBe("select");
    expect(resolveDragDirectionTool({ toolMode: "draw", drawToolMode: "draw" })).toBe("draw");
    expect(resolveDragDirectionTool({ toolMode: "draw", drawToolMode: "line" })).toBe("vibrato");
    expect(resolveDragDirectionTool({ toolMode: "draw", drawToolMode: "vibrato" })).toBe("vibrato");

    expect(resolveDragDirection({ ...scope, toolMode: "select", drawToolMode: "draw" })).toBe(
        "y-only",
    );
    expect(resolveDragDirection({ ...scope, toolMode: "draw", drawToolMode: "draw" })).toBe(
        "x-only",
    );
    expect(resolveDragDirection({ ...scope, toolMode: "draw", drawToolMode: "line" })).toBe("free");
    expect(resolveDragDirection({ ...scope, toolMode: "draw", drawToolMode: "vibrato" })).toBe(
        "free",
    );
});
