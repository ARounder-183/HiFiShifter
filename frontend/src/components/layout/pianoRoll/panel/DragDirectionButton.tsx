/**
 * 拖动方向按钮：左键循环切换方向（自由 → 仅 X → 仅 Y），右键直接选择。
 *
 * 【它为什么不是"开关 × 模式"】拖动方向是**一根轴上的互斥取值**：自由 / 仅 X / 仅 Y。
 * 三者穷尽且互斥，不存在与它们正交的"开 / 关" —— "自由"本身就是其中一种取值，不是
 * "关"。所以状态里只有一个方向字段、**没有第二个布尔**：左键依次循环（与 `D` 键、拖拽
 * 中右键逐字同义），右键在菜单里直接挑一个。
 *
 * 【高亮仍然要有，但它是派生的】`仅 X` / `仅 Y` 时按钮亮（强调色实心），`自由` 时熄灭 ——
 * 这是"当前是否落在受限取值上"的**渲染结论**，由方向字段当场算出，不是另存一个开关量。
 * 相邻的"自动滚屏""波纹编辑""音高吸附"之所以真有开关，是因为它们的状态模型里确实多
 * 一根正交的轴（要不要跟随 / 要不要吸附）；这里没有那根轴，所以也不存它。
 *
 * 【图标画方向】三段内联 SVG 共用同一个 15×15 网格：双向斜箭头 / 水平双向箭头 /
 * 垂直双向箭头。tooltip 固定是「功能名：方向名」（`拖动方向：仅 X`），它同时是
 * aria-label —— 循环按钮的可访问状态靠这个名字传达，高亮只是同一条信息的视觉复述。
 *
 * 【与兄弟按钮的两处差异】
 * 1. **方向集随工具变化**：绘制 / 直线 / 颤音只有 `自由` / `仅 X`（横向位移本身就是
 *    时间推进，锁掉纵向只剩一条水平线；反过来锁掉横向则画不出东西），选择工具多一个
 *    `仅 Y`。菜单项因此由 `activeDragDirectionTool` 决定，模式表在
 *    `features/session/dragDirection`。
 * 2. **气泡尾巴上的 `(D)`**：左键与 `D` 键在这里是**同一个动作**（都循环），所以这条
 *    提示字面准确；它是工具栏里唯一带快捷键的方向按钮（撤销 / 重做的气泡同样带括号
 *    快捷键）。
 *
 * 【为什么状态住在本组件里】菜单只有"开着吗"一件事，关闭完全由 `AppAnchoredMenu`
 * 的 `onClose` 接管（与 `AutoScrollButton` 同一写法）。放进面板模型只会给那个已到
 * 行数预算的文件多加三行状态，而模型里没有任何其它代码需要知道菜单开着。
 */
import { useRef, useState } from "react";
import { Box } from "@radix-ui/themes";

import {
    formatKeybindingList,
    isNoneBindingList,
} from "../../../../features/keybindings/keybindingsSlice";
import { DRAG_DIRECTIONS_BY_TOOL } from "../../../../features/session/dragDirection";
import {
    cycleDragDirection,
    persistUiSettings,
    setDragDirection,
} from "../../../../features/session/sessionSlice";
import type { DragDirection } from "../../../../features/session/sessionTypes";
import type { MessageKey } from "../../../../i18n/messages";
import { AppAnchoredMenu, AppIconButton } from "../../../../ui";
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

/** 自由：一条双向斜箭头（两个方向都跟随）。 */
export function DragDirectionFreeIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            aria-hidden="true"
            data-hs-drag-direction-icon="free"
        >
            <path
                d="M3.5 11.5L11.5 3.5M11.5 3.5L8 3.5M11.5 3.5L11.5 7M3.5 11.5L7 11.5M3.5 11.5L3.5 8"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 仅 X：水平双向箭头（只改时间，不改值）。 */
export function DragDirectionXOnlyIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            aria-hidden="true"
            data-hs-drag-direction-icon="x-only"
        >
            <path
                d="M2 7.5H13M2 7.5L4.5 5M2 7.5L4.5 10M13 7.5L10.5 5M13 7.5L10.5 10"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 仅 Y：垂直双向箭头（只改值，不改时间）。 */
export function DragDirectionYOnlyIcon() {
    return (
        <svg
            width="15"
            height="15"
            viewBox="0 0 15 15"
            fill="none"
            aria-hidden="true"
            data-hs-drag-direction-icon="y-only"
        >
            <path
                d="M7.5 2V13M7.5 2L5 4.5M7.5 2L10 4.5M7.5 13L5 10.5M7.5 13L10 10.5"
                stroke="currentColor"
                strokeWidth="1.2"
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}

/** 按方向取图标。两处（按钮 / 菜单项）必须同源，否则会出现"按钮是 A、菜单写着 B"。 */
function iconFor(direction: DragDirection) {
    if (direction === "free") return <DragDirectionFreeIcon />;
    return direction === "x-only" ? <DragDirectionXOnlyIcon /> : <DragDirectionYOnlyIcon />;
}

/**
 * 方向名 → 词条。
 *
 * 【为什么是显式映射而不是拼模板】键名里的 `x_only` 与枚举里的 `x-only` 不同形，拼
 * 字符串就得再写一遍下划线替换；而显式映射还能让"少了一种方向"变成类型错误。
 */
const DIRECTION_LABEL_KEY: Record<DragDirection, MessageKey> = {
    free: "drag_direction_free",
    "x-only": "drag_direction_x_only",
    "y-only": "drag_direction_y_only",
};

export function DragDirectionButton({ m }: Props) {
    const { activeDragDirection, activeDragDirectionTool, cycleDragDirectionKb, dispatch, tf } = m;
    const [menuOpen, setMenuOpen] = useState(false);
    const anchorRef = useRef<HTMLDivElement | null>(null);
    const menuRef = useRef<HTMLDivElement | null>(null);

    return (
        // 锚点外壳：菜单贴在这个盒子下方展开（与工具栏上的其它按钮同一种弹出面）。
        <Box ref={anchorRef} data-hs-context-menu>
            <AppIconButton
                /* 高亮**由方向本身派生**（`仅 X` / `仅 Y` 亮，`自由` 不亮）：它不是第二个
                   布尔，而是"这根轴是否处在受限取值上"的渲染结论 —— 状态里没有对应的开关量。 */
                active={activeDragDirection !== "free"}
                // 激活时用主题强调色（与相邻三个开关同款；不带 color 时 Radix 回落强调色）
                emphasis="accent"
                // 固定「功能名：方向名」—— 它同时是 aria-label，所以必须自带功能名。
                tooltip={`${tf("drag_direction")}: ${tf(DIRECTION_LABEL_KEY[activeDragDirection])}${
                    isNoneBindingList(cycleDragDirectionKb)
                        ? ""
                        : ` (${formatKeybindingList(cycleDragDirectionKb, "")})`
                }`}
                tabIndex={-1}
                // 右下角小三角：提示"右键还有方向可选"（与工具栏其它按钮同一个记号）。
                menuCaret
                onClick={() => {
                    dispatch(cycleDragDirection(activeDragDirectionTool));
                    void dispatch(persistUiSettings());
                }}
                onContextMenu={(e) => {
                    e.preventDefault();
                    setMenuOpen(true);
                }}
                icon={iconFor(activeDragDirection)}
            />

            <AppAnchoredMenu
                open={menuOpen}
                anchorRef={anchorRef}
                menuRef={menuRef}
                ariaLabel={tf("drag_direction")}
                onClose={() => setMenuOpen(false)}
                items={DRAG_DIRECTIONS_BY_TOOL[activeDragDirectionTool].map((direction) => ({
                    key: `drag-direction-${direction}`,
                    // 菜单项只说**方向名**：功能名已经在按钮的 tooltip 里。
                    label: tf(DIRECTION_LABEL_KEY[direction]),
                    icon: iconFor(direction),
                    checked: activeDragDirection === direction,
                    // 选方向 = 立即生效：这里没有第二根开关轴，写入方向就是应用它。
                    onSelect: () => {
                        dispatch(setDragDirection({ tool: activeDragDirectionTool, direction }));
                        void dispatch(persistUiSettings());
                    },
                }))}
            />
        </Box>
    );
}
