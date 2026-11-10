// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ParamBlockContextMenu } from "./ParamBlockContextMenu";
import type { ParamBlockMenuContext } from "../paramBlockMenuModel";

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
    param_blocks_create_from_selection: "从选区新建块",
    param_blocks_group: "编组",
    param_blocks_ungroup: "取消编组",
    param_blocks_restore_game: "还原",
    param_blocks_restore_deleted: "恢复删除的块",
    param_blocks_normalize_pitch: "音高归一化",
    menu_copy: "复制",
    menu_cut: "剪切",
    menu_paste: "粘贴",
    menu_initialize: "初始化",
    menu_transpose_cents: "按指定音分移调...",
    menu_transpose_degrees: "按指定度数移调...",
    menu_set_pitch: "音高设置到...",
    menu_average: "平均化...",
    menu_smooth: "平滑化...",
    menu_add_vibrato: "添加颤音...",
    menu_quantize: "量化...",
    menu_mean_quantize: "均值量化...",
    menu_save_as_pitch_ref: "保存为音高参考块",
    menu_export_midi: "导出为 MIDI",
    menu_convert_volume_to_dyn: "转换为动态",
    menu_convert_dyn_to_volume: "转换为音量",
    menu_select_all: "全选",
    menu_deselect: "取消选择",
    param_blocks_delete: "删除",
};

const tf = (key: string): string => labels[key] ?? key;

const context: ParamBlockMenuContext = {
    canSplit: true,
    canGroup: true,
    canUngroup: false,
    canDelete: true,
    canRestoreGame: true,
    canRestoreDeleted: true,
    canNormalizePitch: true,
    hasSelection: true,
    isPitch: true,
    isVolume: false,
    isDyn: false,
};

const onAction = vi.fn();

const baseProps = {
    x: 10,
    y: 20,
    context,
    onAction,
    onClose: () => {},
    tf,
};

function items(): HTMLButtonElement[] {
    return [...document.querySelectorAll<HTMLButtonElement>(".hs-menu__item")];
}

test("块菜单同时包含块动作与参数动作", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const text = items().map((item) => item.textContent?.trim());
    expect(text).toContain("在播放头处切开");
    expect(text).toContain("从选区新建块");
    expect(text).toContain("初始化");
    expect(text).toContain("平滑化...");
    expect(text).toContain("导出为 MIDI");
    expect(text).toContain("全选");
});

test("参数项按参数类型与选区禁用，而不是消失", async () => {
    await act(async () => {
        root.render(
            <ParamBlockContextMenu
                {...baseProps}
                context={{
                    ...context,
                    hasSelection: false,
                    isPitch: false,
                    canSplit: false,
                    canGroup: false,
                }}
            />,
        );
    });
    const byLabel = (label: string) => items().find((item) => item.textContent?.trim() === label);
    expect(byLabel("复制")?.disabled).toBe(true);
    expect(byLabel("初始化")?.disabled).toBe(true);
    expect(byLabel("粘贴")?.disabled).toBe(false);
    expect(byLabel("全选")?.disabled).toBe(false);
    expect(byLabel("在播放头处切开")?.disabled).toBe(true);
});

test("块动作经统一 action 通道执行", async () => {
    onAction.mockReset();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const group = items().find((item) => item.textContent?.trim() === "编组");
    if (group === undefined) throw new Error("group item missing");
    await act(async () => group.click());
    expect(onAction).toHaveBeenCalledWith("paramBlock", "group");
});

test("参数动作经 edit 通道执行", async () => {
    onAction.mockReset();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const smooth = items().find((item) => item.textContent?.trim() === "平滑化...");
    if (smooth === undefined) throw new Error("smooth item missing");
    await act(async () => smooth.click());
    expect(onAction).toHaveBeenCalledWith("edit", "smooth");
});

test("删除项是破坏性项，并单独成段放在最后", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const last = items().at(-1);
    expect(last?.textContent?.trim()).toBe("删除");
    expect(last?.getAttribute("data-danger")).toBe("1");
    expect(document.querySelectorAll(".hs-menu__separator").length).toBeGreaterThanOrEqual(2);
});
