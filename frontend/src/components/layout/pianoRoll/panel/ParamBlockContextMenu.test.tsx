// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ParamBlockContextMenu } from "./ParamBlockContextMenu";

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
    param_blocks_split_here: "切开",
    param_blocks_merge: "合并",
    param_blocks_delete: "删除",
    param_blocks_restore_game: "音符块还原",
};

test("块菜单包含分段还原", async () => {
    await act(async () => {
        root.render(
            <ParamBlockContextMenu
                x={10}
                y={20}
                selectedCount={2}
                canMerge
                canDelete
                splitAtSec={1}
                selectedDeleted={false}
                onSplit={() => {}}
                onMerge={() => {}}
                onDelete={() => {}}
                onClose={() => {}}
                tf={(key) => labels[key] ?? key}
            />,
        );
    });
    expect(
        [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")].map((item) =>
            item.textContent?.trim(),
        ),
    ).toEqual(["切开", "合并", "音符块还原", "删除"]);
});

test("合并项调用回调", async () => {
    const onMerge = vi.fn();
    await act(async () => {
        root.render(
            <ParamBlockContextMenu
                x={10}
                y={20}
                selectedCount={2}
                canMerge
                canDelete
                splitAtSec={1}
                selectedDeleted={false}
                onSplit={() => {}}
                onMerge={onMerge}
                onDelete={() => {}}
                onClose={() => {}}
                tf={(key) => labels[key] ?? key}
            />,
        );
    });
    const merge = [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")].find(
        (item) => item.textContent?.trim() === "合并",
    );
    if (merge === undefined) throw new Error("merge item missing");
    await act(async () => merge.click());
    expect(onMerge).toHaveBeenCalledTimes(1);
});
