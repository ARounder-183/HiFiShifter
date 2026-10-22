/**
 * 拖动方向按钮：自由 / 仅横向 / 仅纵向（点击循环切换）。
 *
 * 它决定**拖动参数线时指针位移怎么投影**：自由 = 两个方向都跟随，仅横向 = 只改
 * 时间、仅纵向 = 只改值。三种状态用三段内联 SVG 表达（同一个 15×15 网格上的
 * 双向箭头 / 水平箭头 / 垂直箭头），激活态用主题强调色。
 *
 * 【为什么单独成文件】它是工具栏里唯一"图标随三态各自不同"的按钮——三段 SVG
 * 混在工具栏的行内布局里，会把"这一行有哪些控件"压得看不见。
 */
import {
    formatKeybindingList,
    isNoneBindingList,
} from "../../../../features/keybindings/keybindingsSlice";
import { cycleDragDirection, persistUiSettings } from "../../../../features/session/sessionSlice";
import { AppIconButton } from "../../../../ui";
import type { PianoRollPanelModel } from "./usePianoRollPanelModel";

/** 本文件用到的模型字段（AST 扫描得出）。 */
type Props = {
    m: Pick<
        PianoRollPanelModel,
        | "activeDragDirection"
        | "activeDragDirectionTool"
        | "cycleDragDirectionKb"
        | "dispatch"
        | "tf"
    >;
};

export function DragDirectionButton({ m }: Props) {
    const { activeDragDirection, activeDragDirectionTool, cycleDragDirectionKb, dispatch, tf } = m;

    return (
        <AppIconButton
            active={activeDragDirection !== "free"}
            tooltip={`${tf("drag_direction")}: ${tf(activeDragDirection === "free" ? "drag_direction_free" : activeDragDirection === "x-only" ? "drag_direction_x_only" : "drag_direction_y_only")}${
                isNoneBindingList(cycleDragDirectionKb)
                    ? ""
                    : ` (${formatKeybindingList(cycleDragDirectionKb, "")})`
            }`}
            tabIndex={-1}
            onClick={() => {
                dispatch(cycleDragDirection(activeDragDirectionTool));
                void dispatch(persistUiSettings());
            }}
            icon={
                activeDragDirection === "free" ? (
                    <svg
                        width="15"
                        height="15"
                        viewBox="0 0 15 15"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M3.5 11.5L11.5 3.5M11.5 3.5L8 3.5M11.5 3.5L11.5 7M3.5 11.5L7 11.5M3.5 11.5L3.5 8"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                ) : activeDragDirection === "x-only" ? (
                    <svg
                        width="15"
                        height="15"
                        viewBox="0 0 15 15"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M2 7.5H13M2 7.5L4.5 5M2 7.5L4.5 10M13 7.5L10.5 5M13 7.5L10.5 10"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                ) : (
                    <svg
                        width="15"
                        height="15"
                        viewBox="0 0 15 15"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M7.5 2V13M7.5 2L5 4.5M7.5 2L10 4.5M7.5 13L5 10.5M7.5 13L10 10.5"
                            stroke="currentColor"
                            strokeWidth="1.2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                )
            }
        />
    );
}
