/**
 * 参数编辑器「块」药丸。
 *
 * 配置项已迁到「选项 → 参数块设置」；药丸保留进入工具、眼睛显隐与一个紧凑的
 * 动作菜单。动作菜单与右键菜单 / 快捷键共用同一个 hifi:paramBlockEditOp 通道。
 */
import { Box } from "@radix-ui/themes";
import { useState } from "react";
import { setToolModePersistent } from "../../../../features/session/sessionSlice";
import { AppContextMenu, type AppMenuItemSpec } from "../../../../ui";
import { useMenuShortcut } from "../../../../ui/useMenuShortcut";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ParamToolbarPill } from "./toolbarParts";

type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "dispatch"
        | "paramBlockToolActive"
        | "paramBlocksVisible"
        | "s"
        | "tf"
        | "toggleParamBlocksVisible"
        | "selectedParamBlockCount"
        | "canGroupSelectedParamBlocks"
        | "canUngroupSelectedParamBlocks"
        | "canDeleteSelectedParamBlocks"
        | "canRestoreGameParamBlocks"
        | "canRestoreDeletedParamBlocks"
        | "canNormalizeSelectedParamBlock"
    >;
};

export function BlockToolMenu({ m }: Props) {
    const {
        dispatch,
        paramBlockToolActive,
        paramBlocksVisible,
        s,
        tf,
        toggleParamBlocksVisible,
        selectedParamBlockCount,
        canGroupSelectedParamBlocks,
        canUngroupSelectedParamBlocks,
        canDeleteSelectedParamBlocks,
        canRestoreGameParamBlocks,
        canRestoreDeletedParamBlocks,
        canNormalizeSelectedParamBlock,
    } = m;
    const [menu, setMenu] = useState<{ x: number; y: number } | null>(null);
    const splitShortcut = useMenuShortcut("paramBlock.split");
    const groupShortcut = useMenuShortcut("paramBlock.group");
    const ungroupShortcut = useMenuShortcut("paramBlock.ungroup");
    const restoreShortcut = useMenuShortcut("paramBlock.restoreGame");
    const restoreDeletedShortcut = useMenuShortcut("paramBlock.restoreDeleted");
    const normalizeShortcut = useMenuShortcut("paramBlock.normalizePitch");
    const deleteShortcut = useMenuShortcut("paramBlock.delete");
    const selectAllShortcut = useMenuShortcut("paramBlock.selectAll");
    const deselectShortcut = useMenuShortcut("paramBlock.deselect");

    const toggleBlockTool = () => {
        const next = paramBlockToolActive ? s.drawToolMode || "draw" : "block";
        void dispatch(setToolModePersistent(next));
    };
    const run = (op: string) => {
        window.dispatchEvent(new CustomEvent("hifi:paramBlockEditOp", { detail: { op } }));
        setMenu(null);
    };
    const items: AppMenuItemSpec[] = [
        {
            key: "split",
            label: tf("param_blocks_split_here"),
            shortcut: splitShortcut,
            disabled: selectedParamBlockCount !== 1,
            description: tf("param_blocks_split_disabled_reason"),
            onSelect: () => run("split"),
        },
        {
            key: "group",
            label: tf("param_blocks_group"),
            shortcut: groupShortcut,
            disabled: !canGroupSelectedParamBlocks,
            description: tf("param_blocks_group_none"),
            onSelect: () => run("group"),
        },
        {
            key: "ungroup",
            label: tf("param_blocks_ungroup"),
            shortcut: ungroupShortcut,
            disabled: !canUngroupSelectedParamBlocks,
            description: tf("param_blocks_ungroup_none"),
            onSelect: () => run("ungroup"),
        },
        {
            key: "restore",
            label: tf("param_blocks_restore_game"),
            shortcut: restoreShortcut,
            disabled: !canRestoreGameParamBlocks,
            description: tf("param_blocks_restore_disabled_reason"),
            onSelect: () => run("restoreGame"),
        },
        {
            key: "restore-deleted",
            label: tf("param_blocks_restore_deleted"),
            shortcut: restoreDeletedShortcut,
            disabled: !canRestoreDeletedParamBlocks,
            description: tf("param_blocks_restore_deleted_disabled_reason"),
            onSelect: () => run("restoreDeleted"),
        },
        {
            key: "normalize",
            label: tf("param_blocks_normalize_pitch"),
            shortcut: normalizeShortcut,
            disabled: !canNormalizeSelectedParamBlock,
            description: tf("param_blocks_normalize_disabled_reason"),
            onSelect: () => run("normalizePitch"),
        },
        {
            key: "delete",
            label: tf("param_blocks_delete"),
            shortcut: deleteShortcut,
            danger: true,
            disabled: !canDeleteSelectedParamBlocks,
            description: tf("param_blocks_delete_disabled_reason"),
            onSelect: () => run("delete"),
        },
        {
            key: "select-all",
            label: tf("param_blocks_select_all"),
            shortcut: selectAllShortcut,
            separatorBefore: true,
            onSelect: () => run("selectAll"),
        },
        {
            key: "deselect",
            label: tf("param_blocks_deselect"),
            shortcut: deselectShortcut,
            onSelect: () => run("deselect"),
        },
    ];

    return (
        <Box data-hs-context-menu className="flex items-center gap-0.5">
            <ParamToolbarPill
                label={tf("param_blocks_short")}
                labelKind="block-icon"
                labelAriaLabel={tf("param_blocks_tool")}
                labelTooltip={tf("param_blocks_tool")}
                active={paramBlockToolActive}
                onSelect={toggleBlockTool}
                eyeMode={paramBlocksVisible ? "on" : "off"}
                onToggleEye={toggleParamBlocksVisible}
                eyeTooltip={tf("param_blocks_show_overlay")}
                eyeLabel={tf("param_blocks_show_overlay")}
            />
            <button
                type="button"
                aria-label={tf("param_blocks_more_actions")}
                data-tooltip={tf("param_blocks_more_actions")}
                className="px-1 text-qt-text-muted hover:text-qt-text"
                onClick={(event) => {
                    const rect = event.currentTarget.getBoundingClientRect();
                    setMenu({ x: rect.left, y: rect.bottom + 4 });
                }}
            >
                …
            </button>
            {menu !== null ? (
                <AppContextMenu
                    x={menu.x}
                    y={menu.y}
                    items={items}
                    onClose={() => setMenu(null)}
                    ariaLabel={tf("param_blocks_tool")}
                />
            ) : null}
        </Box>
    );
}
