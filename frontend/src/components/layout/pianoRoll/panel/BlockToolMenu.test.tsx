// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Theme } from "@radix-ui/themes";
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
        handleParamBlockMenuAction: vi.fn(),
        paramBlockMenuContext: {
            canSplit: () => false,
            canGroup: false,
            canUngroup: false,
            canDelete: false,
            canRestoreGame: false,
            canRestoreDeleted: false,
            canNormalizePitch: false,
            hasSelection: false,
            isPitch: true,
            isVolume: false,
            isDyn: false,
        },
        paramBlockToolActive: false,
        paramBlocksVisible: true,
        s: { drawToolMode: "draw" },
        tf: (key: string) => labels[key] ?? key,
        toggleParamBlocksVisible: vi.fn(),
        ...overrides,
    } as unknown as React.ComponentProps<typeof BlockToolMenu>["m"];
}

test("块药丸提供工具、眼睛与可发现的溢出菜单", async () => {
    await act(async () => {
        root.render(
            <Theme>
                <BlockToolMenu m={model()} />
            </Theme>,
        );
    });
    expect(host.querySelector('[aria-label="块编辑工具"]')).not.toBeNull();
    expect(host.querySelector('[aria-label="显示参数块"]')).not.toBeNull();
    expect(host.textContent).toContain("…");
});

test("点击标签切换块编辑工具", async () => {
    const dispatch = vi.fn();
    await act(async () => {
        root.render(
            <Theme>
                <BlockToolMenu m={model({ dispatch })} />
            </Theme>,
        );
    });
    const label = host.querySelector<HTMLButtonElement>('[aria-label="块编辑工具"]');
    if (label === null) throw new Error("tool label missing");
    await act(async () => label.click());
    expect(dispatch).toHaveBeenCalledTimes(1);
});

test("点击眼睛切换块层显隐", async () => {
    const toggle = vi.fn();
    await act(async () => {
        root.render(
            <Theme>
                <BlockToolMenu m={model({ toggleParamBlocksVisible: toggle })} />
            </Theme>,
        );
    });
    const eye = host.querySelector<HTMLButtonElement>('[aria-label="显示参数块"]');
    if (eye === null) throw new Error("eye button missing");
    await act(async () => eye.click());
    expect(toggle).toHaveBeenCalledTimes(1);
});
