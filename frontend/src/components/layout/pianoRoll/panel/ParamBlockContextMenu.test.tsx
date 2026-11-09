// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ParamBlockContextMenu } from "./ParamBlockContextMenu";

vi.mock("../../../../ui/useMenuShortcut", () => ({
    useMenuShortcut: () => undefined,
}));

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
    param_blocks_group: "编组",
    param_blocks_ungroup: "取消编组",
    param_blocks_restore_game: "还原",
    param_blocks_restore_deleted: "恢复删除的块",
    param_blocks_normalize_pitch: "音高归一化",
    param_blocks_delete: "删除",
};

const baseProps = {
    x: 10,
    y: 20,
    selectedCount: 2,
    canGroup: true,
    canUngroup: false,
    canDelete: true,
    splitAtSec: 1,
    onSplit: () => {},
    onGroup: () => {},
    onUngroup: () => {},
    onDelete: () => {},
    onClose: () => {},
    tf: (key: keyof typeof labels) => labels[key] ?? key,
};

test("块菜单包含编组 / 取消编组 / 还原 / 恢复删除 / 删除", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} canRestoreDeleted />);
    });
    expect(
        [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")].map((item) =>
            item.textContent?.trim(),
        ),
    ).toEqual(["切开", "编组", "取消编组", "还原", "恢复删除的块", "音高归一化", "删除"]);
});

test("编组项调用回调", async () => {
    const onGroup = vi.fn();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} onGroup={onGroup} />);
    });
    const group = [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")].find(
        (item) => item.textContent?.trim() === "编组",
    );
    if (group === undefined) throw new Error("group item missing");
    await act(async () => group.click());
    expect(onGroup).toHaveBeenCalledTimes(1);
});
