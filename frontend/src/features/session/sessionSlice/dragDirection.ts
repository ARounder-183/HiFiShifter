/**
 * 拖动方向的两个 reducer（循环 / 直设）。
 *
 * 【主要内容】
 * - `cycleDragDirectionReducer`：工具栏左键、全局 `D` 键、拖拽中右键 / `D` —— 四处入口
 *   是**同一个动作**（依次遍历本工具的全部方向），因此只有一份实现；
 * - `setDragDirectionReducer`：工具栏右键菜单（直接挑一个方向）。
 *
 * 【为什么单独成文件】`sessionSlice.ts` 有**硬性行数预算**（`tests/fileSizeBudget.test.ts`，
 * 只降不升）。这两个 reducer 本来就在主文件里，但"为什么方向集随工具变化""为什么非法值
 * 静默忽略"这类设计记录必须写在代码旁边，塞回主文件会立刻顶破预算。
 *
 * 【为什么没有"开关" reducer】拖动方向是**一根轴上的互斥取值**（自由 / 仅 X / 仅 Y），
 * 不存在与它们正交的开 / 关 —— "自由"本身就是其中一个取值。所以左键是循环而不是开关，
 * 状态里也就不需要第二个布尔（见 `features/session/dragDirection` 的文件头）。
 */
import type { PayloadAction } from "@reduxjs/toolkit";

import { isDragDirectionForTool, nextDragDirection } from "../dragDirection";
import type { DragDirection, DragDirectionTool, DrawDragDirection } from "../sessionTypes";
import type { SessionState } from "./types";

/** 读某个工具当前的方向（三份字段只有存放位置不同）。 */
function readDirection(state: SessionState, tool: DragDirectionTool): DragDirection {
    if (tool === "select") return state.selectDragDirection;
    if (tool === "draw") return state.drawDragDirection;
    return state.lineVibratoDragDirection;
}

/**
 * 写某个工具的方向。
 *
 * 【收窄由调用方保证】绘制 / 颤音的方向字段是 `DrawDragDirection`（只有 `free` /
 * `x-only`），而这里是统一入口 —— 两个调用方给出的值分别来自本工具的模式表
 * （`nextDragDirection`）与已过白名单的 `setDragDirection`，因此不会写进非法值。
 */
function writeDirection(state: SessionState, tool: DragDirectionTool, direction: DragDirection) {
    if (tool === "select") {
        state.selectDragDirection = direction;
    } else if (tool === "draw") {
        state.drawDragDirection = direction as DrawDragDirection;
    } else {
        state.lineVibratoDragDirection = direction as DrawDragDirection;
    }
}

/** 循环到下一个方向（工具栏左键、全局 `D` 键、拖拽中右键 / `D`）。 */
export function cycleDragDirectionReducer(
    state: SessionState,
    action: PayloadAction<DragDirectionTool>,
) {
    const tool = action.payload;
    writeDirection(state, tool, nextDragDirection(tool, readDirection(state, tool)));
}

/**
 * 直接设置（工具栏右键菜单）。
 *
 * 【为什么静默忽略非法值】与 `setAutoScrollMode` / `setRippleScope` 同约定：reducer 是
 * "写入前的最后一道白名单"。非法值只可能来自代码写错或旧测试夹具 —— 抛错会把整个
 * dispatch 链路炸掉，而丢弃只会让界面停在原方向。
 */
export function setDragDirectionReducer(
    state: SessionState,
    action: PayloadAction<{ tool: DragDirectionTool; direction: DragDirection }>,
) {
    const { tool, direction } = action.payload;
    if (!isDragDirectionForTool(tool, direction)) return;
    writeDirection(state, tool, direction);
}
