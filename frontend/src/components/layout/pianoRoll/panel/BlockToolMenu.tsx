/**
 * 参数编辑器「块」药丸与溢出菜单。
 *
 * 块画布右键菜单是主入口；工具栏的“…”是同一份描述表的第二个入口，
 * 让块工具里也能发现初始化 / 复制 / 平滑 / 导出等参数命令。
 */
import { Fragment } from "react";
import { Box, DropdownMenu } from "@radix-ui/themes";
import { setToolModePersistent } from "../../../../features/session/sessionSlice";
import { PARAM_BLOCK_MENU_ITEMS } from "../paramBlockMenuModel";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { ParamToolbarPill } from "./toolbarParts";

type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "dispatch"
        | "handleParamBlockMenuAction"
        | "paramBlockMenuContext"
        | "paramBlockToolActive"
        | "paramBlocksVisible"
        | "s"
        | "tf"
        | "toggleParamBlocksVisible"
    >;
};

export function BlockToolMenu({ m }: Props) {
    const {
        dispatch,
        handleParamBlockMenuAction,
        paramBlockMenuContext,
        paramBlockToolActive,
        paramBlocksVisible,
        s,
        tf,
        toggleParamBlocksVisible,
    } = m;

    const toggleBlockTool = () => {
        const next = paramBlockToolActive ? s.drawToolMode || "draw" : "block";
        void dispatch(setToolModePersistent(next));
    };

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
            <DropdownMenu.Root>
                <DropdownMenu.Trigger disabled={!paramBlockToolActive}>
                    <span
                        className="rounded px-1.5 py-0.5 text-qt-xs text-qt-text hover:bg-qt-highlight hover:text-white"
                        aria-label={tf("param_blocks_tool")}
                        data-tooltip={tf("param_blocks_tool")}
                    >
                        …
                    </span>
                </DropdownMenu.Trigger>
                <DropdownMenu.Content variant="soft" color="gray">
                    {PARAM_BLOCK_MENU_ITEMS.map((item) => (
                        <Fragment key={item.key}>
                            {item.separatorBefore === true ? <DropdownMenu.Separator /> : null}
                            <DropdownMenu.Item
                                color={item.danger === true ? "red" : undefined}
                                disabled={!item.enabled(paramBlockMenuContext)}
                                onSelect={() => handleParamBlockMenuAction(item.channel, item.op)}
                            >
                                {tf(item.labelKey)}
                            </DropdownMenu.Item>
                        </Fragment>
                    ))}
                </DropdownMenu.Content>
            </DropdownMenu.Root>
        </Box>
    );
}
