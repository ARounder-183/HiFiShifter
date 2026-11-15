// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ParamBlockContextMenu } from "./ParamBlockContextMenu";
import type { ParamBlockCommandContext } from "../paramBlockCommandMenu";

const i18nMock = vi.hoisted(() => {
    const labels: Record<string, string> = {
        menu_copy: "复制",
        menu_cut: "剪切",
        menu_paste: "粘贴",
        menu_initialize: "初始化",
        menu_transpose_cents: "按指定音分移调...",
        menu_transpose_degrees: "按指定度数移调...",
        menu_set_pitch: "音高设置到...",
        menu_set_value: "设置到...",
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
        param_blocks_split_at_playhead: "在播放头处分割",
        param_blocks_create_from_selection: "从选区新建参数块",
        param_blocks_group: "编组",
        param_blocks_ungroup: "取消编组",
        param_blocks_restore_game: "还原参数块",
        param_blocks_restore_deleted: "恢复已删除的参数块",
        param_blocks_normalize_pitch: "音高归一化",
        param_blocks_submenu: "参数块",
        param_blocks_menu: "参数块操作",
        param_blocks_delete: "删除参数块",
    };
    return { labels, tf: (key: string): string => labels[key] ?? key };
});

vi.mock("../../../../i18n/I18nProvider", () => ({
    useI18n: () => ({ tf: i18nMock.tf }),
}));

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

const tf = i18nMock.tf;

const context: ParamBlockCommandContext & { isVolume: boolean; isDyn: boolean } = {
    canSplit: () => true,
    canGroup: true,
    canUngroup: true,
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

function itemByLabel(label: string): HTMLButtonElement | undefined {
    return items().find((item) => item.textContent?.trim() === label);
}

async function openBlockCommands(): Promise<void> {
    const trigger = itemByLabel("参数块");
    if (trigger === undefined) throw new Error("block submenu trigger missing");
    await act(async () => trigger.click());
}

test("块菜单顶部是“参数块”子菜单，标准参数命令来自共享编辑菜单", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const topLevel = items().map((item) => item.textContent?.trim());
    expect(topLevel[0]).toBe("参数块");
    // 8 项块命令收在子菜单里，不再平铺到顶层。
    expect(topLevel).not.toContain("在播放头处分割");
    expect(topLevel).toContain("复制");
    expect(topLevel).toContain("初始化");
    expect(topLevel).toContain("平滑化...");
    expect(topLevel).toContain("导出为 MIDI");

    await openBlockCommands();
    const blockText = items().map((item) => item.textContent?.trim());
    for (const label of [
        "在播放头处分割",
        "从选区新建参数块",
        "编组",
        "取消编组",
        "还原参数块",
        "恢复已删除的参数块",
        "音高归一化",
        "删除参数块",
    ]) {
        expect(blockText).toContain(label);
    }
});

test("块菜单隐藏不可用命令，pitch-only 归一化不显示在其它参数上", async () => {
    await act(async () => {
        root.render(
            <ParamBlockContextMenu
                {...baseProps}
                context={{
                    ...context,
                    hasSelection: false,
                    isPitch: false,
                    canSplit: () => false,
                    canGroup: false,
                    canUngroup: false,
                }}
            />,
        );
    });
    await openBlockCommands();
    expect(itemByLabel("在播放头处分割")).toBeUndefined();
    expect(itemByLabel("从选区新建参数块")).toBeUndefined();
    expect(itemByLabel("编组")).toBeUndefined();
    expect(itemByLabel("取消编组")).toBeUndefined();
    expect(itemByLabel("音高归一化")).toBeUndefined();
    // 标准参数命令仍然存在（复制/粘贴不因无选区隐藏）。
    expect(itemByLabel("复制")?.disabled).toBe(false);
    expect(itemByLabel("粘贴")?.disabled).toBe(false);
});

test("块动作经 paramBlock 通道执行", async () => {
    onAction.mockReset();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    await openBlockCommands();
    const group = itemByLabel("编组");
    if (group === undefined) throw new Error("group item missing");
    await act(async () => group.click());
    expect(onAction).toHaveBeenCalledWith("paramBlock", "group");
});

test("参数动作经 edit 通道执行", async () => {
    onAction.mockReset();
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    const smooth = itemByLabel("平滑化...");
    if (smooth === undefined) throw new Error("smooth item missing");
    await act(async () => smooth.click());
    expect(onAction).toHaveBeenCalledWith("edit", "smooth");
});

test("删除项是破坏性项，位于块子菜单最后", async () => {
    await act(async () => {
        root.render(<ParamBlockContextMenu {...baseProps} />);
    });
    await openBlockCommands();
    const submenuItems = [
        ...document.querySelectorAll<HTMLButtonElement>(".hs-menu--submenu .hs-menu__item"),
    ];
    const last = submenuItems.at(-1);
    expect(last?.textContent?.trim()).toBe("删除参数块");
    expect(last?.getAttribute("data-danger")).toBe("1");
});
