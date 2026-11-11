/**
 * 工具栏上的**工具图标**：选择族 / 绘制族 / 吸附半音 / 吸附音阶。
 *
 * 【为什么是独立的一个模块】这些图标原本夹在 Pitch Snap 弹窗状态与返回对象之间
 * （原在面板模型里，紧挨着工具按钮的渲染处）。它们是纯 JSX 常量，只依赖"当前
 * 画的是什么工具 / 选择族记住的是哪个子模式"，与面板的任何状态、effect、hook
 * 顺序都无关 —— 所以这里**不是 hook，是一个纯函数**。
 *
 * 【为什么图标要区分】选择族的两个子模式、绘制族的三工具若共用图标，工具栏上
 * 就只剩 tooltip 能说明"现在按下去是什么"，而 tooltip 需要悬停才看得见。
 */
import { CursorArrowIcon, Pencil1Icon } from "@radix-ui/react-icons";
import type { DrawToolMode, SelectToolMode } from "../../../../features/session/sessionTypes";

/**
 * 生成工具栏要用的图标。
 *
 * @param currentDrawTool 当前绘制族记忆（决定绘制族按钮图标）。
 * @param selectToolMode 选择族当前 / 上次子模式（决定选择族按钮图标）。
 */
export function buildToolIcons(currentDrawTool: DrawToolMode, selectToolMode: SelectToolMode) {
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

    /**
     * 参数块选择的图标：横向分栏。
     *
     * 与参数组药丸的旧块图标同形，但现在只属于“选择族 → 块选择”这一种子模式，
     * 不再附带眼睛，也不再是第三个工具按钮。
     */
    const paramBlockToolIcon = (
        <svg width="14" height="14" viewBox="0 0 15 15" fill="none" aria-hidden="true">
            <rect
                x="1.5"
                y="3.5"
                width="12"
                height="8"
                rx="1.5"
                stroke="currentColor"
                strokeWidth="1.2"
            />
            <path d="M5.75 4V11" stroke="currentColor" strokeWidth="1.2" />
            <path d="M9.25 4V11" stroke="currentColor" strokeWidth="1.2" />
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
    const currentSelectToolIcon =
        selectToolMode === "block" ? paramBlockToolIcon : <CursorArrowIcon />;
    return {
        vibratoToolIcon,
        lineToolIcon,
        pitchSnapSemitoneIcon,
        pitchSnapScaleIcon,
        currentDrawToolIcon,
        currentSelectToolIcon,
        paramBlockToolIcon,
    };
}
