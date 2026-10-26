// @vitest-environment jsdom
/*
 * handleEditOp 的闭包必须拿到**当下**的 toolMode。
 *
 * 【为什么必须有这条】回归（de6851c5）：handleEditOp 的依赖数组曾漏掉 `s.toolMode`
 * —— 工具切换后 useCallback 的闭包仍持有旧的 `s` 对象，`editOperations` 的 deselect
 * 守卫（`if (s.toolMode !== "select") return`）读到的是闭包里的旧值，表现为
 * "切回 select 工具后 Ctrl+D / 菜单 deselect 静默无操作"。
 *
 * 与 `editOperations.deselect.test.ts` 的分工：那份钉的是 `runEditOp` 消费**当下**
 * 值的契约（守卫本身没问题）；这份钉的是把**当下**值送进闭包的那根线 ——
 * 没有它，把 deps 里的 `s.toolMode` 再删掉一次，测试依然全绿。
 */
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test, vi } from "vitest";

import type { SessionState } from "../../../../features/session/sessionSlice";
import type { ParamSelection } from "../paramSelection";
import { usePianoRollEditCommands } from "./usePianoRollEditCommands";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const mounted: Array<() => Promise<void>> = [];
afterEach(async () => {
    while (mounted.length) await mounted.pop()?.();
});

/** deselect 分支与挂载 effect 只触达这几个字段；其余入参以类型断言豁免。 */
interface HarnessInputs {
    rootTrackId: string;
    selectionRef: { current: ParamSelection | null };
    setSelectionUi: (selection: ParamSelection | null) => void;
    invalidate: () => void;
    clipboardRef: { current: unknown };
}

function Harness({
    toolMode,
    inputs,
    handleEditActionRef,
}: {
    toolMode: string;
    inputs: HarnessInputs;
    handleEditActionRef: { current: (op: string) => void };
}) {
    usePianoRollEditCommands({
        ...inputs,
        s: { toolMode } as unknown as SessionState,
        handleEditActionRef,
        openVibratoDialog: () => {},
    } as unknown as Parameters<typeof usePianoRollEditCommands>[0]);
    return null;
}

test("工具切回 select 后，deselect 读到的是当下的工具模式", async () => {
    const setSelectionUi = vi.fn();
    const invalidate = vi.fn();
    const selectionRef: { current: ParamSelection | null } = {
        current: [{ startFrame: 10, frameCount: 10 }],
    };
    // 生产派发路径正是这枚 ref：hook 在渲染期把它刷成最新的 handleEditOp（见文件头）。
    // 不用 useRef：它不在组件里，RefObject 形状的可变对象就是 hook 需要的全部。
    const handleEditActionRef: { current: (op: string) => void } = { current: () => {} };
    const inputs: HarnessInputs = {
        rootTrackId: "root-1",
        selectionRef,
        setSelectionUi,
        invalidate,
        // 挂载时的剪贴板对齐 effect 会写它（读到 null 即空载荷，无需后端）。
        clipboardRef: { current: null },
    };

    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    const render = (toolMode: string) =>
        act(async () => {
            root.render(
                <Harness
                    toolMode={toolMode}
                    inputs={inputs}
                    handleEditActionRef={handleEditActionRef}
                />,
            );
        });
    mounted.push(async () => {
        await act(async () => root.unmount());
        host.remove();
    });

    await render("draw");
    // 工具切换：闭包必须跟着刷新（deps 数组含 s.toolMode；漏掉它则此处仍是 draw）。
    await render("select");
    await act(async () => {
        handleEditActionRef.current("deselect");
    });

    expect(selectionRef.current).toBeNull();
    expect(setSelectionUi).toHaveBeenCalledWith(null);
    expect(invalidate).toHaveBeenCalled();
});
