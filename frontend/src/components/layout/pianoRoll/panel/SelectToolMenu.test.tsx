// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { SelectToolMenu } from "./SelectToolMenu";

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
    select_tool_menu: "选择工具",
    select_tool_range: "参数线选择",
    select_tool_block: "参数块选择",
};

function model(overrides: Record<string, unknown> = {}) {
    return {
        dispatch: vi.fn(),
        s: { toolMode: "draw", selectToolMode: "range" },
        selectToolMenuAnchorRef: { current: null },
        selectToolMenuOpen: false,
        selectToolMenuRef: { current: null },
        setSelectToolMenuOpen: vi.fn(),
        selectToolKb: [{ key: "f7" }],
        blockSelectionKb: [{ key: "__none__" }],
        selectToolButtonTitle: "参数线选择",
        currentSelectToolIcon: <span data-testid="range-icon">range</span>,
        paramBlockToolIcon: <span data-testid="block-icon">block</span>,
        tf: (key: string) => labels[key] ?? key,
        ...overrides,
    } as unknown as React.ComponentProps<typeof SelectToolMenu>["m"];
}

test("选择族按钮没有参数药丸与眼睛，右键打开两个子模式", async () => {
    const setOpen = vi.fn();
    await act(async () => {
        root.render(<SelectToolMenu m={model({ setSelectToolMenuOpen: setOpen })} />);
    });
    expect(host.querySelector(".param-pill")).toBeNull();
    expect(host.querySelector('[aria-label*="显示参数块"]')).toBeNull();
    const button = host.querySelector<HTMLButtonElement>('[aria-label="选择工具: 参数线选择"]');
    if (button === null) throw new Error("select family button missing");
    await act(async () => {
        button.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
    });
    expect(setOpen).toHaveBeenCalledWith(true);
});

test("打开菜单显示 range / block 与真实快捷键，点击 block 派发", async () => {
    const dispatch = vi.fn();
    await act(async () => {
        root.render(
            <SelectToolMenu
                m={model({
                    dispatch,
                    s: { toolMode: "select", selectToolMode: "range" },
                    selectToolMenuOpen: true,
                })}
            />,
        );
    });
    const items = [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")];
    const text = items.map((item) => item.textContent?.trim() ?? "");
    expect(text.some((value) => value.includes("参数线选择"))).toBe(true);
    expect(text.some((value) => value.includes("参数块选择"))).toBe(true);
    expect(text.some((value) => value.includes("F7"))).toBe(true);

    const block = items.find((item) => item.textContent?.includes("参数块选择"));
    if (block === undefined) throw new Error("block menu item missing");
    await act(async () => block.click());
    expect(dispatch).toHaveBeenCalledTimes(1);
});

test("block 激活时选择族按钮仍 active，图标为块图标", async () => {
    await act(async () => {
        root.render(
            <SelectToolMenu
                m={model({
                    s: { toolMode: "select", selectToolMode: "block" },
                    selectToolButtonTitle: "参数块选择",
                    currentSelectToolIcon: <span data-testid="block-icon">block</span>,
                })}
            />,
        );
    });
    const button = host.querySelector<HTMLButtonElement>('[aria-label="选择工具: 参数块选择"]');
    expect(button?.getAttribute("aria-pressed")).toBe("true");
    expect(button?.querySelector('[data-testid="block-icon"]')).not.toBeNull();
});
