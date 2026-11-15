/**
 * 参数编辑器工具族 / 选择子模式的持久化 thunk。
 *
 * 【为什么从 sessionSlice.ts 拆出来】该文件有行数预算，且这几个 thunk 是
 * “状态 reducer + 设置落盘”的组合动作，不是同步 slice 定义。拆开后 reducer
 * 文件只保留状态迁移，工具切换的判重、落盘与 `~` 语义集中在这里。
 */
import { createAsyncThunk } from "@reduxjs/toolkit";
import type { SelectToolMode, ToolMode } from "../sessionTypes";
import type { SessionState } from "../sessionSlice/types";
import { setSelectToolMode, setToolMode } from "../sessionSlice";
import { persistUiSettings } from "./runtimeThunks";
import { toggleSelectToolMode } from "../toolMode";

/**
 * 切换参数编辑器工具族并落盘。
 *
 * 【为什么要一个专门的入口】`setToolMode` 在 App 与参数编辑器面板里共有多个
 * dispatch 点。逐个补一句 `persistUiSettings()`，等于埋下多个“以后新增入口时忘了
 * 补”的坑；工具选择是低频动作（一次点击或一次按键），立即写一次盘的代价可以忽略。
 *
 * 【已经在这个工具上时直接返回】这让“声明式地断言工具”变成免费操作：拖拽中的
 * 预设轮转每次都要确保自己落在正确的工具上（见 `applyVibratoChoice`），若每次都
 * 写一遍盘，一次轮转就会产生两次设置写入。跳过重复项之后，只有真的换了工具才落盘。
 *
 * 【为什么里面要转型】`persistUiSettings` 自己是个 thunk，而 `createAsyncThunk`
 * 默认的 `dispatch` 只认 plain action —— 与 `importThunks` 的
 * `dispatch as unknown as TrackDispatch` 是同一手法。
 */
export const setToolModePersistent = createAsyncThunk<ToolMode, ToolMode>(
    "session/setToolModePersistent",
    (mode, { dispatch, getState }) => {
        // 读的是**派发前**的状态：派发之后它必然等于 `mode`，那样判断永远为真。
        if ((getState() as { session: SessionState }).session.toolMode === mode) return mode;
        dispatch(setToolMode(mode));
        (dispatch as unknown as (action: unknown) => void)(persistUiSettings());
        return mode;
    },
);

/**
 * 写入选择族子模式并落盘。
 *
 * 判重同时看工具族与子模式：从绘制族带着“上次是 block”的记忆按 F7 时，
 * 目标仍是 select + range，必须真的切一次并落到图层不变量。
 */
export const setSelectToolModePersistent = createAsyncThunk<void, SelectToolMode>(
    "session/setSelectToolModePersistent",
    (mode, { dispatch, getState }) => {
        const state = (getState() as { session: SessionState }).session;
        if (state.toolMode === "select" && state.selectToolMode === mode) return;
        dispatch(setSelectToolMode(mode));
        (dispatch as unknown as (action: unknown) => void)(persistUiSettings());
    },
);

/**
 * 选择族子模式切换（`~`）。
 *
 * 当前在选择族：range ↔ block；当前在绘制族：直接进入 select + block。
 * 不读取旧的 `toolMode === "block"`，也不回退到 drawToolMode —— 退出块选择
 * 的唯一去处是 range，返回绘制族由 `mode.toggle`（Tab）负责。
 */
export const toggleSelectToolModePersistent = createAsyncThunk<void, void>(
    "session/toggleSelectToolModePersistent",
    (_, { dispatch, getState }) => {
        const state = (getState() as { session: SessionState }).session;
        const next =
            state.toolMode === "select" ? toggleSelectToolMode(state.selectToolMode) : "block";
        (dispatch as unknown as (action: unknown) => void)(setSelectToolModePersistent(next));
    },
);
