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
 */
import type { MouseEvent as ReactMouseEvent, ReactNode } from "react";
import { Box } from "@radix-ui/themes";
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
            icon={
                <Box style={{ position: "relative", width: 15, height: 15 }}>
                    <Box
                        style={{
                            position: "absolute",
                            inset: 0,
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                        }}
                    >
                        {icon}
                    </Box>
                    {/* 右下角小三角：只表达“这一族还有别的模式”，不参与语义。 */}
                    <Box
                        data-testid="tool-family-caret"
                        aria-hidden="true"
                        style={{
                            position: "absolute",
                            right: -1,
                            bottom: -1,
                            width: 6,
                            height: 6,
                            opacity: 0.7,
                        }}
                    >
                        <svg width="6" height="6" viewBox="0 0 6 6" fill="none">
                            <path d="M0 6L6 0V6Z" fill="currentColor" />
                        </svg>
                    </Box>
                </Box>
            }
        />
    );
}
