/**
 * 工具栏上的**工具图标**：颤音 / 直线 / 吸附半音 / 吸附音阶 / 当前工具。
 *
 * 【为什么是独立的一个模块】这五个图标原本夹在 Pitch Snap 弹窗状态与返回对象之间
 * （原在面板模型里，紧挨着工具按钮的渲染处）。它们是纯 JSX 常量，只依赖"当前画的是什么工具"，与面板的
 * 任何状态、effect、hook 顺序都无关 —— 所以这里**不是 hook，是一个纯函数**：模型的
 * 调用点每帧重建同样的元素树，与拆分前逐字相同（没有引入 memo，因为拆分前也没有）。
 *
 * 【为什么图标要区分】直线与颤音若共用图标，工具栏上就只剩 tooltip 能说明"现在按下
 * 去画的是直线还是颤音"，而 tooltip 需要悬停才看得见。
 */
import { Pencil1Icon } from "@radix-ui/react-icons";
import type { DrawToolMode } from "../../../../features/session/sessionTypes";

/**
 * 生成工具栏要用的五个图标。
 *
 * @param currentDrawTool 当前绘制工具（决定"当前工具"图标取哪一个）。
 */
export function buildToolIcons(currentDrawTool: DrawToolMode) {
    const vibratoToolIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M1.5 7.5C3 7.5 3 3.5 4.5 3.5C6 3.5 6 11.5 7.5 11.5C9 11.5 9 3.5 10.5 3.5C12 3.5 12 7.5 13.5 7.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );

    /**
     * 直线工具的图标：一条斜线。
     *
     * 与颤音工具的波形图区分开 —— 两者若共用图标，工具栏上就只剩 tooltip 能说明
     * "现在按下去画的是直线还是颤音"，而 tooltip 需要悬停才看得见。
     */
    const lineToolIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path d="M2 12L13 3" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
    );

    const pitchSnapSemitoneIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <path
                d="M2.5 12.5H6.5V9.5H10.5V5.5H12.5"
                stroke="currentColor"
                strokeWidth="1.3"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );

    const pitchSnapScaleIcon = (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
        >
            <rect x="2.5" y="10.5" width="4" height="2" rx="0.5" fill="currentColor" />
            <rect x="6.5" y="7.5" width="4" height="2" rx="0.5" fill="currentColor" />
            <rect x="10.5" y="4.5" width="2.5" height="2" rx="0.5" fill="currentColor" />
        </svg>
    );

    const currentDrawToolIcon =
        currentDrawTool === "vibrato" ? (
            vibratoToolIcon
        ) : currentDrawTool === "line" ? (
            lineToolIcon
        ) : (
            <Pencil1Icon />
        );
    return {
        vibratoToolIcon,
        lineToolIcon,
        pitchSnapSemitoneIcon,
        pitchSnapScaleIcon,
        currentDrawToolIcon,
    };
}
