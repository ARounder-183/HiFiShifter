/**
 * 颤音预设的波形缩略图。
 *
 * 【用途】凡是展示预设名的地方都配上它 —— 工具栏按钮的图标、下拉菜单项、
 * 管理器与弹窗的列表行、拖拽 HUD。用户扫一眼缩略图的形状就能认出预设，
 * 不必逐个读名字或回忆"演歌是哪一条"。
 *
 * 【为什么是 SVG 而不是 canvas】一个列表同时渲染几十个缩略图：SVG polyline
 * 不需要每行一个绘制上下文与 ResizeObserver，路径字符串可按预设缓存；
 * `stroke="currentColor"` 让它直接跟随主题色与选中态颜色。
 *
 * 【定标】按预设自身峰值（`previewScaleCents` 同理），不是固定尺度 —— 选择
 * 场景里"形状可辨"优先于"深度可比"，精确幅度由 tooltip / 读数表达。深度差异
 * 用线宽（1 → 1.6px）作辅助暗示。
 */

import { useMemo } from "react";

import { glyphPath } from "./vibratoDialogLogic";
import type { VibratoPreset } from "../../../features/vibrato/vibratoTypes";

export interface VibratoPresetGlyphProps {
    preset: VibratoPreset;
    /** 视口宽高（CSS 像素）。 */
    width?: number;
    height?: number;
    className?: string;
}

export function VibratoPresetGlyph({
    preset,
    width = 40,
    height = 14,
    className,
}: VibratoPresetGlyphProps) {
    const path = useMemo(() => glyphPath(preset, width, height), [preset, width, height]);
    // 线宽随深度微调：30 分附近的常用预设落在 1.2 上下。
    const strokeWidth = Math.min(1.6, Math.max(1, preset.depthCents / 40));

    return (
        <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            className={className}
            aria-hidden
            // 收缩不下沉：缩略图是行内装饰，flex 行里保持基线稳定。
            style={{ flexShrink: 0, display: "block" }}
        >
            <polyline
                points={path}
                fill="none"
                stroke="currentColor"
                strokeWidth={strokeWidth}
                strokeLinecap="round"
                strokeLinejoin="round"
            />
        </svg>
    );
}
