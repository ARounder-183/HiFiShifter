/**
 * 自动滚屏的两个 reducer（开关 + 模式）。
 *
 * 【主要内容】
 * - `toggleAutoScrollReducer`：开关（工具栏左键、快捷键）。
 * - `setAutoScrollModeReducer`：模式（工具栏右键菜单）。
 *
 * 【为什么单独成文件】
 * 1. 两者是同一族状态的两根正交轴（"要不要滚" × "怎么滚"），放在一起才能一眼看出
 *    它们互不影响；
 * 2. `sessionSlice.ts` 有**硬性行数预算**（`tests/fileSizeBudget.test.ts` 的
 *    `LIMITS`，只降不升，当前只剩 1 行余量）。reducer 的正文不长，但"为什么静默
 *    忽略非法值""为什么模式与开关正交"这类设计记录必须写在代码旁边 —— 塞回主文件
 *    会立刻把预算顶破。
 *
 * 【与其他模块的关系】
 * - 上游：`ActionBar` 的 `AutoScrollButton`（左键 `toggleAutoScroll`、右键
 *   `setAutoScrollMode`）、`persistUiSettings`（落盘）、`loadUiSettings`（水合）。
 * - 下游：`TimelineTransportBridge` / `PianoRollPlayheadBridge` 读这两个字段决定
 *   本帧是否跟随、按哪种语义跟随（见 `utils/autoFollowScroll`）。
 *
 * 【为什么用独立函数而不是 `SliceCaseReducers` 对象】`createSlice` 的 `reducers`
 * 支持"把一个函数直接当某个 reducer 的名字值"（`applyClipSyllableAnalysis` 就是
 * 这么接的）。这种写法保住了 action creator 的载荷类型推导，又不必让主文件认识
 * 本模块的每个细节。
 */
import type { PayloadAction } from "@reduxjs/toolkit";

import { isAutoScrollMode, type AutoScrollMode } from "../../../utils/autoFollowScroll";
import type { SessionState } from "./types";

/** 翻转自动滚屏开关（不触碰模式）。 */
export function toggleAutoScrollReducer(state: SessionState) {
    state.autoScrollEnabled = !state.autoScrollEnabled;
}

/**
 * 设置自动滚屏模式。
 *
 * 【为什么静默忽略非法值】与 `setRippleMode` 同约定：reducer 是"写入前的最后一道
 * 白名单"。非法值只可能来自代码写错或旧测试夹具 —— 抛错会把整个 dispatch 链路炸掉，
 * 而丢弃只会让界面停在原模式。
 *
 * 【为什么不在这里顺手开启自动滚屏】"选模式即开启"是**交互策略**，住在调用方
 * （`ActionBar`）里；reducer 只负责状态本身。把策略塞进 reducer 会让"设置模式"
 * 这个动作产生第二个副作用，而它无法从 action 名字上看出来。
 */
export function setAutoScrollModeReducer(
    state: SessionState,
    action: PayloadAction<AutoScrollMode>,
) {
    if (isAutoScrollMode(action.payload)) {
        state.autoScrollMode = action.payload;
    }
}
