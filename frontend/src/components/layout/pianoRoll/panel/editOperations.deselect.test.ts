import { describe, expect, it, vi } from "vitest";
import type { RefObject } from "react";

import type { SessionState } from "../../../../features/session/sessionSlice";
import type { ModelDeps } from "./modelDeps";
import type { ParamSelection } from "../paramSelection";
import { runEditOp } from "./editOperations";

/** 只构造 `deselect` 分支触达的字段，其余以类型断言豁免。 */
function deselectDeps(toolMode: string, selection: ParamSelection | null) {
    const selectionRef = { current: selection } as RefObject<ParamSelection | null>;
    const setSelectionUi = vi.fn();
    const invalidate = vi.fn();
    const deps = {
        rootTrackId: "root-1",
        s: { toolMode } as unknown as SessionState,
        selectionRef,
        setSelectionUi,
        invalidate,
    } as unknown as Parameters<typeof runEditOp>[0];
    return { deps, selectionRef, setSelectionUi, invalidate };
}

describe("runEditOp — deselect 的工具模式守卫", () => {
    // 回归：usePianoRollEditCommands 的 handleEditOp 依赖数组曾漏掉 s.toolMode，
    // 切回 select 工具后闭包里仍是旧值，Ctrl+D/菜单 deselect 静默无操作。
    // 这里钉住 runEditOp 侧的消费契约：守卫读的是**当下**的 toolMode。
    it("绘制工具下 deselect 是空操作", async () => {
        const { deps, selectionRef, setSelectionUi, invalidate } = deselectDeps(
            "draw",
            [{ start: 10, end: 20 }],
        );
        await runEditOp(deps, "deselect");
        expect(selectionRef.current).toEqual([{ start: 10, end: 20 }]);
        expect(setSelectionUi).not.toHaveBeenCalled();
        expect(invalidate).not.toHaveBeenCalled();
    });

    it("选择工具下 deselect 清空选区", async () => {
        const { deps, selectionRef, setSelectionUi, invalidate } = deselectDeps(
            "select",
            [{ start: 10, end: 20 }],
        );
        await runEditOp(deps, "deselect");
        expect(selectionRef.current).toBeNull();
        expect(setSelectionUi).toHaveBeenCalledWith(null);
        expect(invalidate).toHaveBeenCalled();
    });
});
