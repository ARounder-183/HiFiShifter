/**
 * 参数编辑器「块」药丸。
 *
 * 只保留两件事：进入块编辑工具、显隐块层。块动作**只有**一个入口 ——
 * 画布右键菜单（`ParamBlockContextMenu`），加上可绑定的快捷键。
 *
 * 【为什么删掉原来的「…」菜单】它和右键菜单逐项重合，只多出「全选 / 取消选择」；
 * 而全选有 `paramBlock.selectAll`（默认 Ctrl+A）与菜单栏「编辑 → 参数块」，
 * 取消选择点空白处即可。同一个动作两套 `disabled` 判定、两套文案、两套测试，
 * 换来的"可发现性"抵不过它制造的歧义：用户会问"这两处为什么不一样"。
 */
import { Box } from "@radix-ui/themes";
import { setToolModePersistent } from "../../../../features/session/sessionSlice";
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
        </Box>
    );
}
