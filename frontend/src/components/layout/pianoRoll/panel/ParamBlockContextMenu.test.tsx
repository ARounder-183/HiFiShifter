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
    param_blocks_tool: "块编辑工具",
    param_blocks_split_at_playhead: "在播放头处切开",
    param_blocks_group: "编组",
    param_blocks_ungroup: "取消编组",
    param_blocks_restore_game: "还原",
    param_blocks_restore_deleted: "恢复删除的块",
    param_blocks_normalize_pitch: "音高归一化",
    param_blocks_delete: "删除",
};

const tf = (key: keyof typeof labels): string => labels[key] ?? key;

const baseProps = {
    x: 10,
    y: 20,
    canSplit: true,
    canGroup: true,
    canUngroup: false,
    canDelete: true,
    onSplit: () => {},
    onGroup: () => {},
    onUngroup: () => {},
    onDelete: () => {},
    onClose: () => {},
    tf,
};

function items(): HTMLButtonElement[] {
    return [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")];
}

const expectedLabels = [
    "在播放头处切开",
    "编组",
    "取消编组",
    "还原",
    "恢复删除的块",
    "音高归一化",
    "删除",
];

test("块菜单包含切开 / 编组 / 取消编组 / 还原 / 恢复删除 / 音高归一化 / 删除", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} canRestoreDeleted />);
    });
    expect(items().map((item) => item.textContent?.trim())).toEqual(expectedLabels);
});

/*
 * 每个菜单项**只有一行**文字。
 *
 * 块动作的禁用条件是瞬时状态（选了几个块、是不是编组块、播放头在不在块内），
 * 菜单弹出期间不会再变一次；把它写成第二行解释，菜单就变成解释墙。
 * `description` 只服务"宿主能力"那一类用户改不了的限制（Clip / 轨道区域 /
 * 停靠标签在用）。
 */
test("全部置灰时也不出现第二行解释", async () => {
    await act(async () => {
        root.render(
            <ParamBlockContextMenu
                {...baseProps}
                canSplit={false}
                canGroup={false}
                canUngroup={false}
                canDelete={false}
                canRestoreGame={false}
                canRestoreDeleted={false}
                canNormalizePitch={false}
            />,
        );
    });
    expect(document.querySelectorAll(".hs-menu__description")).toHaveLength(0);
    expect(items().every((item) => item.disabled)).toBe(true);
    expect(items().map((item) => item.textContent?.trim())).toEqual(expectedLabels);
});

test("播放头不在块内时切开项置灰", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} canSplit={false} />);
    });
    expect(items()[0]?.disabled).toBe(true);
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} canSplit />);
    });
    expect(items()[0]?.disabled).toBe(false);
});

test("编组项调用回调", async () => {
    const onGroup = vi.fn();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} onGroup={onGroup} />);
    });
    const group = items().find((item) => item.textContent?.trim() === "编组");
    if (group === undefined) throw new Error("group item missing");
    await act(async () => group.click());
    expect(onGroup).toHaveBeenCalledTimes(1);
});

test("删除项是破坏性项，并单独成段放在最后", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const last = items().at(-1);
    expect(last?.textContent?.trim()).toBe("删除");
    expect(last?.getAttribute("data-danger")).toBe("1");
    // 分隔线：编辑 / 状态恢复 / 破坏性三段之间各一条。
    expect(document.querySelectorAll(".hs-menu__separator").length).toBeGreaterThanOrEqual(2);
});
