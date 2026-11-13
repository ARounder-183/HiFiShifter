// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Theme } from "@radix-ui/themes";
import { ToolFamilyButton } from "./ToolFamilyButton";

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

async function render(props: Partial<React.ComponentProps<typeof ToolFamilyButton>> = {}) {
    const onClick = vi.fn();
    const onContextMenu = vi.fn();
    await act(async () => {
        root.render(
            <Theme>
                <ToolFamilyButton
                    active={false}
                    tooltip="选择工具"
                    icon={<span>icon</span>}
                    onClick={onClick}
                    onContextMenu={onContextMenu}
                    {...props}
                />
            </Theme>,
        );
    });
    return { onClick, onContextMenu };
}

test("工具族按钮渲染图标、tooltip 与小三角", async () => {
    await render({ active: true });
    const button = host.querySelector<HTMLButtonElement>('[aria-label="选择工具"]');
    expect(button).not.toBeNull();
    expect(button?.getAttribute("aria-pressed")).toBe("true");
    // 小三角现在由 `AppIconButton.menuCaret` 统一画（与主工具栏的两个按钮同源）。
    expect(host.querySelector("[data-hs-menu-caret]")).not.toBeNull();
});

test("左键与右键分别回调", async () => {
    const { onClick, onContextMenu } = await render();
    const button = host.querySelector<HTMLButtonElement>('[aria-label="选择工具"]');
    if (button === null) throw new Error("tool button missing");
    await act(async () => button.click());
    expect(onClick).toHaveBeenCalledTimes(1);
    await act(async () => {
        button.dispatchEvent(new MouseEvent("contextmenu", { bubbles: true, cancelable: true }));
    });
    expect(onContextMenu).toHaveBeenCalledTimes(1);
});
