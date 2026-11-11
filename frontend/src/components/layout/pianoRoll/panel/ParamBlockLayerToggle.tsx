/**
 * 参数块**图层可见性**开关。
 *
 * 它是视图轴上的独立眼睛，不属于任何工具：块选择是选择族的子模式，而“块层
 * 画不画”只由这个开关决定。关闭图层时 reducer 会把选择子模式收回 range，
 * 因此不会再出现“工具仍开着、图层却被迫弹回可见”的死状态。
 */
import { EyeClosedIcon, EyeOpenIcon } from "@radix-ui/react-icons";
import { AppIconButton } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

type Props = {
    m: Pick<PianoRollPanelModel, "paramBlocksVisible" | "toggleParamBlocksVisible" | "tf">;
};

export function ParamBlockLayerToggle({ m }: Props) {
    const { paramBlocksVisible, toggleParamBlocksVisible, tf } = m;
    const label = tf(paramBlocksVisible ? "param_blocks_layer_hide" : "param_blocks_layer_show");
    return (
        <AppIconButton
            active={paramBlocksVisible}
            emphasis="accent"
            tooltip={label}
            aria-label={label}
            aria-pressed={paramBlocksVisible}
            tabIndex={-1}
            onClick={toggleParamBlocksVisible}
            icon={paramBlocksVisible ? <EyeOpenIcon /> : <EyeClosedIcon />}
        />
    );
}
