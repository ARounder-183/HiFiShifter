import React, { useCallback, useEffect, useRef } from "react";
import { gridStepBeats } from "./grid";

/**
 * Grid lines are drawn as SVG paths computed directly from beat positions.
 * Repeating CSS gradients with fractional background sizes can accumulate
 * subpixel rounding drift, which makes the minor grid shift relative to the
 * real beat/snap positions.
 */
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
    lineOpacity?: number;
    showBoundary?: boolean;
    sticky?: boolean;
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
    lineOpacity = 0.9,
    showBoundary = true,
    sticky = false,
}) => {
    const svgRef = useRef<SVGSVGElement | null>(null);

    const useViewport =
        viewportWidth != null &&
        Number.isFinite(viewportWidth) &&
        viewportWidth > 0 &&
        scrollLeft != null &&
        Number.isFinite(scrollLeft);
    const isSticky = sticky && useViewport;

    const weakStepPx = Math.max(1e-6, pxPerBeat * gridStepBeats(grid));
    const barStepPx = Math.max(1e-6, pxPerBeat * Math.max(1, beatsPerBar));

    const width = isSticky ? Math.max(1, Math.floor(viewportWidth as number)) : contentWidth;
    const height = contentHeight;

    const latestRef = useRef({
        weakStepPx,
        barStepPx,
        width,
        height,
        contentWidth,
        viewportWidth:
            viewportWidth != null && Number.isFinite(viewportWidth) && viewportWidth > 0
                ? viewportWidth
                : contentWidth,
        scrollLeft: scrollLeft ?? 0,
        isSticky,
    });
    latestRef.current = {
        weakStepPx,
        barStepPx,
        width,
        height,
        contentWidth,
        viewportWidth:
            viewportWidth != null && Number.isFinite(viewportWidth) && viewportWidth > 0
                ? viewportWidth
                : contentWidth,
        scrollLeft: scrollLeft ?? 0,
        isSticky,
    };

    const draw = useCallback((nextScrollLeft?: number) => {
        const svg = svgRef.current;
        if (!svg) return;
        const paths = svg.querySelectorAll<SVGPathElement>("path");
        if (paths.length < 2) return;

        const latest = latestRef.current;
        const sl = Number.isFinite(nextScrollLeft)
            ? (nextScrollLeft as number)
            : latest.scrollLeft;
        const offset = latest.isSticky ? sl : 0;
        const bufferPx = Math.max(240, latest.viewportWidth * 0.5);
        const visibleStart = latest.isSticky ? 0 : Math.max(0, sl - bufferPx);
        const visibleEnd = latest.isSticky
            ? latest.width
            : Math.min(latest.contentWidth, sl + latest.viewportWidth + bufferPx);

        const buildPath = (stepPx: number): string => {
            if (!Number.isFinite(stepPx) || stepPx <= 0) return "";
            const firstIndex = Math.max(0, Math.floor((visibleStart + offset) / stepPx));
            const lastIndex = Math.max(
                firstIndex,
                Math.ceil((visibleEnd + offset) / stepPx),
            );
            const increment = Math.max(1, Math.ceil(1 / Math.max(1e-9, stepPx)));
            const parts: string[] = [];
            for (let index = firstIndex; index <= lastIndex; index += increment) {
                const x = index * stepPx - offset;
                if (x < -1 || x > latest.width + 1) continue;
                parts.push(`M${x} 0V${latest.height}`);
            }
            return parts.join("");
        };

        paths[0].setAttribute("d", buildPath(latest.weakStepPx));
        paths[1].setAttribute("d", buildPath(latest.barStepPx));
    }, []);

    useEffect(() => {
        draw(scrollLeft);
    }, [
        draw,
        scrollLeft,
        weakStepPx,
        barStepPx,
        width,
        height,
        contentWidth,
        viewportWidth,
        isSticky,
    ]);

    useEffect(() => {
        const el = layerRef && typeof layerRef === "object" ? layerRef.current : null;
        if (!el) return;
        (el as unknown as { __hifiGridRedraw?: (scrollLeft: number) => void }).__hifiGridRedraw =
            draw;
        return () => {
            const current = layerRef && typeof layerRef === "object" ? layerRef.current : null;
            if (current) {
                delete (
                    current as unknown as {
                        __hifiGridRedraw?: (scrollLeft: number) => void;
                    }
                ).__hifiGridRedraw;
            }
        };
    }, [draw, layerRef]);

    const boundaryLeft = isSticky
        ? contentWidth - 1 - (scrollLeft as number)
        : contentWidth - 1;
    const boundaryVisible =
        Number.isFinite(boundaryLeft) && boundaryLeft >= -2 && boundaryLeft <= width + 2;
    const manualViewportSync = isSticky && (layerRef != null || boundaryRef != null);

    return (
        <>
            <div
                ref={layerRef}
                className="absolute left-0 top-0 pointer-events-none"
                style={{ width, height }}
            >
                <svg
                    ref={svgRef}
                    width={width}
                    height={height}
                    className="absolute inset-0"
                    style={{ display: "block" }}
                >
                    <path
                        fill="none"
                        strokeWidth={1}
                        opacity={lineOpacity}
                        style={{ stroke: "var(--qt-graph-grid-weak)" }}
                    />
                    <path
                        fill="none"
                        strokeWidth={2}
                        opacity={lineOpacity}
                        style={{ stroke: "var(--qt-graph-grid-strong)" }}
                    />
                </svg>
            </div>

            <div
                ref={boundaryRef}
                className="absolute top-0 bottom-0 w-px z-20"
                style={{
                    left: manualViewportSync ? 0 : boundaryLeft,
                    backgroundColor: "var(--qt-highlight)",
                    opacity:
                        manualViewportSync || !boundaryVisible ? 0 : showBoundary ? lineOpacity : 0,
                }}
            />
        </>
    );
};
