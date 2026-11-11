/**
 * 「永远显示参数块」—— 参数块图层的**常显设置**（功能设置，不是状态开关）。
 *
 * 【它不是什么】它不是「当前图层可见吗」的指示灯：块选择模式下图层**总是**可见，
 * 与这个设置无关。所以 active / aria-pressed 读的是**设置值**，不是派生可见性 ——
 * 否则块模式里按钮会显示成「已开启」，而用户点它时画面毫无变化。
 *
 * 【图标为什么不是眼睛】眼睛在本产品里已经表示「副参数叠加」「预设启用」，与参数块
 * 无关。这里用**堆叠的横向分段条**：参数块就是「参数线上被切成一段一段的横条」，
 * 与选择族菜单里的块工具图标（单个圆角矩形 + 两条竖分隔）同族但**不同形**，
 * 全仓无重复（图标清单见 docs/param-blocks.md）。
 */
import { AppIconButton } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";
import { paramBlocksAlwaysShowIcon } from "./toolIcons";

type Props = {
    m: Pick<PianoRollPanelModel, "alwaysShowParamBlocks" | "toggleAlwaysShowParamBlocks" | "tf">;
};

export function ParamBlockAlwaysShowToggle({ m }: Props) {
    const { alwaysShowParamBlocks, toggleAlwaysShowParamBlocks, tf } = m;
    /*
     * 提示词就是按钮名，**不再单列一条 `_tooltip`**。
     *
     * 工具栏上同一排的开关（「音阶高亮」「锁定参数线」）都是这个形态：
     * `AppIconButton` 把 `tooltip` 同时当作 `aria-label`，无文字按钮的可访问名称
     * 就是它。给这一颗按钮写一句独立的长解释，会让它在同一排里显得比邻居"重要
     * 一级"—— 而它只是同一类开关。语义解释（"是设置不是图层指示灯"）属于手册，
     * 见 docs/i18n/USERMANUAL.md 的选择工具一节。
     */
    return (
        <AppIconButton
            active={alwaysShowParamBlocks}
            emphasis="accent"
            tooltip={tf("param_blocks_always_show")}
            /*
             * 显式写 `aria-pressed`（而不是让 AppIconButton 的 `active || undefined`
             * 兜底）：开关的"关"必须表达成 `false`，不是"没有这个属性" ——
             * 读屏器把缺失当成"不是切换按钮"。
             */
            aria-pressed={alwaysShowParamBlocks}
            tabIndex={-1}
            onClick={toggleAlwaysShowParamBlocks}
            icon={paramBlocksAlwaysShowIcon}
        />
    );
}
