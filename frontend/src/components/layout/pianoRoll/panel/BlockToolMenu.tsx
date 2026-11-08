/**
 * 参数编辑器「块」药丸。
 *
 * 配置项已迁到「选项 → 参数块设置」；药丸只保留进入工具与眼睛显隐。
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

    /*
     * 药丸主体进入块工具；再点一次回到进入前的绘制工具。此前只切 `toolMode`，
     * 但 `s.drawToolMode` 在块工具期间仍保存着旧值，用户第二次点击会“没有反应”。
     */
    const toggleBlockTool = () => {
        const next = paramBlockToolActive ? s.drawToolMode || "draw" : "block";
        void dispatch(setToolModePersistent(next));
    };

    return (
        <Box data-hs-context-menu>
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
