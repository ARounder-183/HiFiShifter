/**
 * BackgroundGrid 组件
 *
 * 负责渲染时间线和钢琴卷帘面板中的垂直网格线。
 * 支持两种渲染模式：
 * - 等间距 CSS 渐变模式（单 tempo 时使用，性能最佳）
 * - Canvas 渲染模式（多 tempo 变速时使用，支持非等间距网格）
 */

import React, { useEffect, useRef } from "react";
import { gridStepBeats } from "./grid";
import type { TempoMap } from "../../../utils/tempoMap";
import { getGridLines } from "../../../utils/tempoMap";

function positiveMod(value: number, mod: number): number {
    if (!Number.isFinite(value) || !Number.isFinite(mod) || mod <= 0) return 0;
    const r = value % mod;
    return (r + mod) % mod;
}

export const BackgroundGrid: React.FC<{
    contentWidth: number;
    contentHeight: number;
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    viewportWidth?: number;
    scrollLeft?: number;
    layerRef?: React.Ref<HTMLDivElement>;
    boundaryRef?: React.Ref<HTMLDivElement>;
    /** Tempo map for variable-tempo grid rendering */
    tempoMap?: TempoMap;
    pxPerSec?: number;
}> = ({
    contentWidth,
    contentHeight,
    pxPerBeat,
    grid,
    beatsPerBar,
    viewportWidth,
    scrollLeft,
    layerRef,
    boundaryRef,
    tempoMap,
    pxPerSec,
}) => {
    const useViewport =
        viewportWidth != null &&
        Number.isFinite(viewportWidth) &&
        scrollLeft != null &&
        Number.isFinite(scrollLeft);

    const hasMultiTempo =
        tempoMap != null &&
        tempoMap.points.length > 1 &&
        pxPerSec != null &&
        pxPerSec > 0;

    const weakStepPx = Math.max(1e-6, pxPerBeat * gridStepBeats(grid));
    const barStepPx = Math.max(1e-6, pxPerBeat * beatsPerBar);

    const width = useViewport
        ? Math.max(1, Math.floor(viewportWidth))
        : contentWidth;
    const height = contentHeight;

    const weakOffsetPx = useViewport
        ? -positiveMod(scrollLeft as number, weakStepPx)
        : 0;
    const barOffsetPx = useViewport
        ? -positiveMod(scrollLeft as number, barStepPx)
        : 0;

    const manualViewportSync =
        useViewport && (layerRef != null || boundaryRef != null);

    const boundaryLeft = useViewport
        ? contentWidth - 1 - (scrollLeft as number)
        : contentWidth - 1;

    const showBoundary =
        Number.isFinite(boundaryLeft) &&
        boundaryLeft >= -2 &&
        boundaryLeft <= width + 2;

    // ── Canvas-based grid for variable tempo ──────────────────────
    const canvasRef = useRef<HTMLCanvasElement | null>(null);

    useEffect(() => {
        if (!hasMultiTempo) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = Math.max(1, window.devicePixelRatio || 1);
        const cw = Math.max(1, Math.floor(width * dpr));
        const ch = Math.max(1, Math.floor(height * dpr));
        if (canvas.width !== cw || canvas.height !== ch) {
            canvas.width = cw;
            canvas.height = ch;
        }
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;

        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, width, height);

        const actualPxPerSec = pxPerSec!;
        const sl = scrollLeft ?? 0;
        const startSec = sl / actualPxPerSec;
        const endSec = (sl + width) / actualPxPerSec;
        const subdivision = gridStepBeats(grid);

        const lines = getGridLines(startSec, endSec, tempoMap!, subdivision);

        // Get CSS variable colors via computed style
        const computedStyle = getComputedStyle(document.documentElement);
        const weakColor =
            computedStyle.getPropertyValue("--qt-graph-grid-weak").trim() ||
            "rgba(255,255,255,0.08)";
        const strongColor =
            computedStyle.getPropertyValue("--qt-graph-grid-strong").trim() ||
            "rgba(255,255,255,0.2)";

        for (const line of lines) {
            const x = line.sec * actualPxPerSec - sl;
            if (x < -1 || x > width + 1) continue;

            if (line.kind === "bar") {
                ctx.strokeStyle = strongColor;
                ctx.lineWidth = 2;
            } else if (line.kind === "beat") {
                ctx.strokeStyle = weakColor;
                ctx.lineWidth = 1;
            } else {
                ctx.strokeStyle = weakColor;
                ctx.lineWidth = 0.5;
            }

            ctx.beginPath();
            const px = Math.round(x) + 0.5;
            ctx.moveTo(px, 0);
            ctx.lineTo(px, height);
            ctx.stroke();
        }
    }, [
        hasMultiTempo,
        width,
        height,
        pxPerSec,
        scrollLeft,
        grid,
        tempoMap,
    ]);

    // Combine layerRef with our internal canvas ref handling for multi-tempo
    const combinedLayerRef = (el: HTMLDivElement | null) => {
        if (typeof layerRef === "function") layerRef(el);
        else if (layerRef && "current" in layerRef)
            (layerRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
    };

    return (
        <>
            {hasMultiTempo ? (
                /* Canvas-based variable-tempo grid */
                <>
                    <canvas
                        ref={canvasRef}
                        className="absolute left-0 top-0 pointer-events-none"
                        style={{
                            width,
                            height,
                            opacity: 0.9,
                        }}
                    />
                    {/* Invisible layer div to satisfy imperative scroll sync from parent */}
                    <div
                        ref={combinedLayerRef}
                        className="absolute left-0 top-0 pointer-events-none"
                        style={{ width: 0, height: 0, display: "none" }}
                    />
                </>
            ) : (
                /* CSS gradient-based constant-tempo grid (original) */
                <div
                    ref={layerRef}
                    className="absolute left-0 top-0 pointer-events-none"
                    style={{
                        width,
                        height,
                        backgroundImage: [
                            "linear-gradient(to right, var(--qt-graph-grid-weak) 1px, transparent 1px)",
                            "linear-gradient(to right, var(--qt-graph-grid-strong) 3px, transparent 3px)",
                        ].join(", "),
                        backgroundSize: [
                            `${weakStepPx}px 100%`,
                            `${barStepPx}px 100%`,
                        ].join(", "),
                        backgroundPosition: useViewport
                            ? manualViewportSync
                                ? undefined
                                : [
                                      `${weakOffsetPx}px 0px`,
                                      `${barOffsetPx}px 0px`,
                                  ].join(", ")
                            : undefined,
                        opacity: 0.9,
                    }}
                />
            )}

            <div
                ref={boundaryRef}
                className="absolute top-0 bottom-0 w-px z-20"
                style={{
                    left: manualViewportSync ? 0 : boundaryLeft,
                    backgroundColor: "var(--qt-highlight)",
                    opacity: manualViewportSync ? 0 : showBoundary ? 0.9 : 0,
                }}
            />
        </>
    );
};
