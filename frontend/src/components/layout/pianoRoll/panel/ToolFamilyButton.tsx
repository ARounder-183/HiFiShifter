/**
 * 参数编辑器工具栏的**工具族按钮**外壳。
 *
 * 选择族与绘制族共用同一个 24px 图标按钮：当前子模式图标 + 右下角小三角
 * （提示“还有同一族的其它模式”）+ 可切换的激活态。左键执行族内默认动作，
 * 右键由调用方打开该族的模式菜单。
 *
 * 【为什么把结构抽出来】选择族与绘制族必须是同构的。此前块编辑被做成一枚
 * 50px 参数药丸，光看工具栏就能读出“它不是工具”；抽出共用按钮后，两个槽位
 * 的视觉、命中区域与无障碍语义由同一段代码保证，不会再漂移。
 *
 * 【小三角现在由 `AppIconButton` 的 `menuCaret` 画】"这枚按钮还有右键菜单"是
 * 交互协议，不是工具族独有的装饰 —— 主工具栏的「自动滚屏」「波纹编辑」用的是
 * 同一个记号。本组件只负责声明它。
 */
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import { AppIconButton } from "../../../../ui";

export interface ToolFamilyButtonProps {
    readonly active: boolean;
    readonly tooltip: string;
    readonly icon: ReactNode;
    readonly onClick: () => void;
    readonly onContextMenu: (event: ReactMouseEvent<HTMLElement>) => void;
}

export function ToolFamilyButton({
    active,
    tooltip,
    icon,
    onClick,
    onContextMenu,
}: ToolFamilyButtonProps) {
    return (
        <AppIconButton
            active={active}
            emphasis="accent"
            tooltip={tooltip}
            tabIndex={-1}
            onClick={onClick}
            onContextMenu={onContextMenu}
            menuCaret
            icon={icon}
        />
    );
}
