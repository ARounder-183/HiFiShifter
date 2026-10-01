/**
 * 预览里的「原参数线轮廓」条。
 *
 * 【为什么单独一条带，而不是画进主图】轮廓与颤音差两个数量级：轮廓可以跨十几个
 * 半音，颤音只有几十分。挤在同一条纵轴上时，标尺要么容下轮廓、把颤音压成一条直线
 * （这是修过两轮的缺陷），要么容下颤音、让轮廓整条跑出画面。分成两条带、各用自己
 * 的标尺，两者就都能读；两条带**共享时间轴**，"此刻音高在哪"与"此刻颤音怎么走"
 * 因此仍然可以对着读。
 *
 * 【画什么】原参数线（弱化虚线）与套用后的参数线（强调色）。`existing` 基线
 * （除「直线」外所有预设）下两条几乎重合 —— 那正是"颤音叠在原曲线上"；而
 * `line` / `hold*` / `average` 下能一眼看出预设会把轮廓搬成什么样。
 *
 * 【为什么不做手势】这条带只回答"音高往哪走"，编辑深度 / 相位在主图上做。少一层
 * 交互，也就少一处与主图标尺混淆的可能。
 */

import { useEffect, useRef } from "react";

import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { contourRangeCents } from "./vibratoDialogLogic";
import { strokeFinitePolyline } from "./vibratoPreviewDraw";

export interface VibratoContourStripProps {
    /** 原参数线轮廓（cents，绝对值；断口为 NaN）。 */
    source: readonly number[];
    /** 套用后的参数线轮廓（cents，绝对值；断口为 NaN）。 */
    result: readonly number[];
    height?: number;
    ariaLabel: string;
}

/** 默认高度（CSS 像素）：够看出走向，又不喧宾夺主。 */
const DEFAULT_HEIGHT = 36;

function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

export function VibratoContourStrip({
    source,
    result,
    height = DEFAULT_HEIGHT,
    ariaLabel,
}: VibratoContourStripProps) {
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const containerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const container = containerRef.current;
        if (!canvas || !container) return;

        const draw = () => {
            const width = container.clientWidth;
            if (width <= 0) return;
            const dpr = readDevicePixelRatio();
            canvas.width = Math.round(width * dpr);
            canvas.height = Math.round(height * dpr);
            canvas.style.width = `${width}px`;
            canvas.style.height = `${height}px`;

            const ctx = canvas.getContext("2d");
            if (!ctx) return;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, width, height);

            const accent = tokenColor("--qt-accent", "#6aa9ff");
            const muted = tokenColor("--qt-text-muted", "#8a8a8a");

            // 标尺按两条线一起拟合：标尺必须容得下画出来的东西，否则线会被裁掉。
            const range = contourRangeCents([source, result]);
            const span = Math.max(1e-6, range.max - range.min);
            const reach = height / 2 - 2;
            const toY = (cents: number) =>
                height / 2 - ((cents - (range.min + range.max) / 2) / span) * 2 * reach;

            const count = Math.max(2, Math.max(source.length, result.length));
            const toX = (index: number) => (index / (count - 1)) * width;

            // 原参数线：弱化虚线（与主图的配色语言一致：弱化 = 套用前）。
            ctx.save();
            ctx.strokeStyle = muted;
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);
            strokeFinitePolyline(ctx, source, toX, toY);
            ctx.restore();

            // 套用后的参数线：强调色，盖在虚线上。
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.25;
            ctx.lineJoin = "round";
            strokeFinitePolyline(ctx, result, toX, toY);
        };

        draw();
        // 宽度随停靠面板 / 对话框变化，需要跟着重画（画布不会自己缩放）。
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        return () => observer.disconnect();
    }, [source, result, height]);

    return (
        <div ref={containerRef} style={{ height }}>
            <canvas
                ref={canvasRef}
                role="img"
                aria-label={ariaLabel}
                className="block w-full rounded border border-qt-border bg-qt-panel"
                style={{ height }}
            />
        </div>
    );
}
