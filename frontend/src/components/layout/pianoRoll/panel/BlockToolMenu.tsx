/**
 * 参数编辑器「块」药丸。
 *
 * 块画布右键菜单是参数块命令的唯一入口；工具栏只保留工具切换与眼睛，
 * 不再渲染“…”菜单或第二个命令描述表。
 */
import { Box } from "@radix-ui/themes";
import { setToolModePersistent } from "../../../../features/session/sessionSlice";
import { toggleParamBlockEditingTool } from "../../../../features/session/toolMode";
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
    >;
};

export function BlockToolMenu({ m }: Props) {
    const { dispatch, paramBlockToolActive, paramBlocksVisible, s, tf, toggleParamBlocksVisible } =
        m;

    const toggleBlockTool = () => {
        void dispatch(
            setToolModePersistent(toggleParamBlockEditingTool(s.toolMode, s.drawToolMode)),
        );
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
        </Box>
    );
}
