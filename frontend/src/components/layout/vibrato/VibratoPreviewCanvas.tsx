/**
 * 颤音预设的波形预览画布。
 *
 * 【为什么必须有】没有它，用户在预设编辑器里是在**盲调**：深度、速率、渐入、
 * 不规则度这几个参数两两耦合，只看数字无法预判听感。这里把
 * `buildVibratoCurve` 的真实输出画出来，参数一改立刻重画。
 *
 * 纵轴按预设**自身**的幅度定标（见 `previewScaleCents`）：按参数值域定标的话，
 * 30 cents 的颤音会是一条直线。真实幅度由旁边的读数表达。
 */

import { useEffect, useRef } from "react";
import { readDevicePixelRatio } from "../../../utils/devicePixelLine";
import { previewScaleCents, type VibratoPreviewSamples } from "./vibratoDialogLogic";

export interface VibratoPreviewCanvasProps {
    samples: VibratoPreviewSamples;
    /** CSS 像素高度。 */
    height?: number;
    ariaLabel?: string;
}

/** 读取语义色令牌；缺失（测试 / 非浏览器环境）时回退到中性色。 */
function tokenColor(name: string, fallback: string): string {
    if (typeof window === "undefined") return fallback;
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

export function VibratoPreviewCanvas({
    samples,
    height = 120,
    ariaLabel,
}: VibratoPreviewCanvasProps) {
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
            const divider = tokenColor("--qt-divider", "#3a3a3a");

            const midY = height / 2;
            const verticalReach = height / 2 - 6;
            const halfCents = previewScaleCents(samples.peakCents);
            const toY = (cents: number) => midY - (cents / halfCents) * verticalReach;

            const count = Math.max(2, samples.wave.length);
            const toX = (index: number) => (index / (count - 1)) * width;

            // 零基线：波形围绕它摆动，视觉上是"音高中心"。
            ctx.strokeStyle = divider;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, midY + 0.5);
            ctx.lineTo(width, midY + 0.5);
            ctx.stroke();

            // 包络带：±envelope 的填充区间，渐入 / 渐强 / 渐出入画就靠它。
            if (samples.envelope.length > 0) {
                ctx.beginPath();
                for (let i = 0; i < count; i += 1) {
                    ctx.lineTo(
                        toX(i),
                        toY(samples.envelope[Math.min(i, samples.envelope.length - 1)]),
                    );
                }
                for (let i = count - 1; i >= 0; i -= 1) {
                    ctx.lineTo(
                        toX(i),
                        toY(-samples.envelope[Math.min(i, samples.envelope.length - 1)]),
                    );
                }
                ctx.closePath();
                ctx.globalAlpha = 0.16;
                ctx.fillStyle = muted;
                ctx.fill();
                ctx.globalAlpha = 1;
            }

            // 波形本体。
            ctx.strokeStyle = accent;
            ctx.lineWidth = 1.5;
            ctx.lineJoin = "round";
            ctx.beginPath();
            for (let i = 0; i < count; i += 1) {
                const x = toX(i);
                const y = toY(samples.wave[i] ?? 0);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };

        draw();

        // 宽度随停靠面板 / 对话框变化，需要跟着重画（画布不会自己缩放）。
        const observer = new ResizeObserver(draw);
        observer.observe(container);
        return () => observer.disconnect();
    }, [samples, height]);

    return (
        <div ref={containerRef} className="w-full" style={{ height }}>
            <canvas
                ref={canvasRef}
                role="img"
                aria-label={ariaLabel}
                className="block w-full"
                style={{ height }}
            />
        </div>
    );
}
