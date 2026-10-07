import { beforeEach, describe, expect, it, vi } from "vitest";

const saveMock = vi.hoisted(() => vi.fn(async () => ({ ok: true })));

/*
 * 只替换落盘那一环，其余导出原样保留 —— 会话切片与它的 thunk 图里还有别的
 * API 具名导出，整份替换会让它们在导入时就变成 undefined。
 */
vi.mock("../../../services/api", async (importOriginal) => {
    const actual = await importOriginal<typeof import("../../../services/api")>();
    return {
        ...actual,
        settingsApi: { ...actual.settingsApi, saveUiSettings: saveMock },
    };
});

import { configureStore } from "@reduxjs/toolkit";

import sessionReducer, { setToolModePersistent } from "../sessionSlice";

const makeStore = () => configureStore({ reducer: { session: sessionReducer } });

/**
 * 工具切换的落盘语义。
 *
 * 【为什么"已经在该工具上"要单独钉】拖拽中的预设轮转每次都无条件断言工具
 * （见 `applyVibratoChoice`）—— 那是为了修掉"从直线工具切回颤音预设后工具却停在
 * 直线"。断言要能随便打，就必须让重复的那次什么都不做；否则一次轮转会写两遍设置。
 */
describe("setToolModePersistent", () => {
    beforeEach(() => {
        saveMock.mockClear();
    });

    it("切换工具：写状态并落盘", async () => {
        const store = makeStore();
        await store.dispatch(setToolModePersistent("line"));
        expect(store.getState().session.toolMode).toBe("line");
        expect(saveMock).toHaveBeenCalledTimes(1);
    });

    it("已经在该工具上：状态不变，且不写盘", async () => {
        const store = makeStore();
        await store.dispatch(setToolModePersistent("line"));
        saveMock.mockClear();

        await store.dispatch(setToolModePersistent("line"));

        expect(store.getState().session.toolMode).toBe("line");
        expect(saveMock).not.toHaveBeenCalled();
    });

    it("绘制类工具会一并同步分组与「上次用的绘制工具」", async () => {
        const store = makeStore();
        await store.dispatch(setToolModePersistent("vibrato"));
        expect(store.getState().session.drawToolMode).toBe("vibrato");
        expect(store.getState().session.toolModeGroup).toBe("draw");

        // 切到选择工具后，绘制工具的记忆留着（Tab 回跳要靠它）。
        await store.dispatch(setToolModePersistent("select"));
        expect(store.getState().session.toolModeGroup).toBe("select");
        expect(store.getState().session.drawToolMode).toBe("vibrato");
    });
});
