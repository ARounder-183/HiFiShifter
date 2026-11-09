// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { BlockToolMenu } from "./BlockToolMenu";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

const labels: Record<string, string> = {
    param_blocks_tool: "块编辑工具",
    param_blocks_short: "块",
    param_blocks_show_overlay: "显示参数块",
};

function model(overrides: Record<string, unknown> = {}) {
    return {
        dispatch: vi.fn(),
        paramBlockToolActive: false,
        paramBlocksVisible: true,
        s: { drawToolMode: "draw" },
        tf: (key: string) => labels[key] ?? key,
        toggleParamBlocksVisible: vi.fn(),
        ...overrides,
    } as unknown as React.ComponentProps<typeof BlockToolMenu>["m"];
}

/*
 * 「块」药丸只剩两件事：进入块编辑工具、显隐块层。
 *
 * 原来的「…」按钮与右键菜单逐项重合，只多出「全选 / 取消选择」，而这两项
 * 都有快捷键与菜单栏入口。同一个动作两套禁用判定与两套文案，只会让用户
 * 追问"这两处为什么不一样"。
 */
test("块药丸只有工具按钮与眼睛，没有「更多块操作」", async () => {
    await act(async () => {
        root.render(<BlockToolMenu m={model()} />);
    });
    const buttons = [...host.querySelectorAll<HTMLButtonElement>("button")];
    expect(buttons).toHaveLength(2);
    expect(host.querySelector('[aria-label="更多块操作"]')).toBeNull();
    expect(host.querySelector('[aria-label="块编辑工具"]')).not.toBeNull();
    expect(host.querySelector('[aria-label="显示参数块"]')).not.toBeNull();
    // 不再挂任何弹出菜单表面。
    expect(document.body.querySelector('[data-hs-context-menu="1"]')).toBeNull();
});

test("点击标签切换块编辑工具", async () => {
    const dispatch = vi.fn();
    await act(async () => {
        root.render(<BlockToolMenu m={model({ dispatch })} />);
    });
    const label = host.querySelector<HTMLButtonElement>('[aria-label="块编辑工具"]');
    if (label === null) throw new Error("tool label missing");
    await act(async () => label.click());
    expect(dispatch).toHaveBeenCalledTimes(1);
});

test("点击眼睛切换块层显隐", async () => {
    const toggle = vi.fn();
    await act(async () => {
        root.render(<BlockToolMenu m={model({ toggleParamBlocksVisible: toggle })} />);
    });
    const eye = host.querySelector<HTMLButtonElement>('[aria-label="显示参数块"]');
    if (eye === null) throw new Error("eye button missing");
    await act(async () => eye.click());
    expect(toggle).toHaveBeenCalledTimes(1);
});
