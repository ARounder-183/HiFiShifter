/**
 * 波纹编辑的两个 reducer（开关 + 方向）。
 *
 * 【主要内容】
 * - `toggleRippleEnabledReducer`：开关（工具栏左键）。
 * - `setRippleScopeReducer`：方向（工具栏右键菜单）。
 *
 * 【为什么单独成文件】与 `./autoScroll` 同一条理由：两者是同一族状态的两根正交轴，
 * 放在一起才能一眼看出它们互不影响；而 `sessionSlice.ts` 有**硬性行数预算**
 * （`tests/fileSizeBudget.test.ts`，只降不升），把"为什么拆轴""为什么静默忽略非法值"
 * 这类设计记录写在主文件里会立刻顶破预算。
 *
 * 【为什么不再有 `cycleRippleMode`】旧版左键在三值（off / track / all）之间循环，
 * 于是"从全部轨道关掉"要绕两步，且循环顺序把两个正交维度混成了一条线。
 * 现在左键只做开关 —— 与相邻的"自动滚屏"完全同构，肌肉记忆也一致。
 */
import type { PayloadAction } from "@reduxjs/toolkit";

import { isRippleScope, type RippleScope } from "../rippleMode";
import type { SessionState } from "./types";

/** 翻转波纹编辑开关（不触碰方向）。 */
export function toggleRippleEnabledReducer(state: SessionState) {
    state.rippleEnabled = !state.rippleEnabled;
}

/**
 * 设置波纹编辑方向。
 *
 * 【为什么静默忽略非法值】与 `setAutoScrollMode` 同约定：reducer 是"写入前的最后
 * 一道白名单"。非法值只可能来自代码写错或旧测试夹具 —— 抛错会把整个 dispatch 链路
 * 炸掉，而丢弃只会让界面停在原方向。
 *
 * 【为什么不在这里顺手开启开关】"选方向即开启"是**交互策略**，住在调用方
 * （`ActionBar`）里；reducer 只负责状态本身。
 */
export function setRippleScopeReducer(state: SessionState, action: PayloadAction<RippleScope>) {
    if (isRippleScope(action.payload)) {
        state.rippleScope = action.payload;
    }
}
