import React, { useCallback, useEffect, useRef } from "react";
import { explicitGridLinesKey } from "./gridLineKey";
import { resolveGridLineSamplingPlan } from "./gridLineSampling";

/**
 * Grid lines are drawn as SVG paths computed directly from beat positions.
 * Repeating CSS gradients with fractional background sizes can accumulate
 * subpixel rounding drift, which makes the minor grid shift relative to the
 * real beat/snap positions.
 *
 * 提供 `weakLineXs` / `strongLineXs`（内容坐标系 x 像素数组）时，
 * 网格线直接使用这些显式位置（Tempo Map 的不等距网格）。
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
    /** Tempo Map 显式网格线位置（内容坐标 x，升序）。 */
    weakLineXs?: number[] | null;
    strongLineXs?: number[] | null;
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
    weakLineXs = null,
    strongLineXs = null,
}) => {
    const svgRef = useRef<SVGSVGElement | null>(null);

    const useViewport =
        viewportWidth != null &&
        Number.isFinite(viewportWidth) &&
        viewportWidth > 0 &&
        scrollLeft != null &&
        Number.isFinite(scrollLeft);
    const isSticky = sticky && useViewport;

    const useExplicitLines = weakLineXs != null && Array.isArray(weakLineXs);

    const samplingViewportWidth =
        viewportWidth != null && Number.isFinite(viewportWidth) && viewportWidth > 0
            ? viewportWidth
            : contentWidth;
    const samplingPlan = useExplicitLines
        ? { weakStepPx: 0, strongStepPx: 0 }
        : resolveGridLineSamplingPlan({
              pxPerBeat,
              grid,
              beatsPerBar: Math.max(1, Math.round(beatsPerBar)),
              viewportWidth: samplingViewportWidth,
          });

    const width = isSticky ? Math.max(1, Math.floor(viewportWidth as number)) : contentWidth;
    const height = contentHeight;

    const latestRef = useRef({
        weakStepPx: samplingPlan.weakStepPx,
        strongStepPx: samplingPlan.strongStepPx,
        weakLineXs: weakLineXs,
        strongLineXs: strongLineXs,
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
        weakStepPx: samplingPlan.weakStepPx,
        strongStepPx: samplingPlan.strongStepPx,
        weakLineXs,
        strongLineXs,
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

    const lastDrawKeyRef = useRef<string | null>(null);

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

        // 重绘跳过键必须覆盖**全部**网格线位置：拖动 Tempo Map 的中间变化点时，
        // 受影响的是数组中部以该点为锚的整段线（整体平移），而长度与首尾线不变，
        // 任何抽样校验和都会误判“无需重绘”，造成网格跳变/错位（见 gridLineKey.ts）。
        const drawKey = [
            sl,
            latest.weakStepPx,
            latest.strongStepPx,
            explicitGridLinesKey(latest.weakLineXs),
            explicitGridLinesKey(latest.strongLineXs),
            latest.width,
            latest.height,
            latest.contentWidth,
            latest.viewportWidth,
            latest.isSticky,
        ].join("|");
        if (lastDrawKeyRef.current === drawKey) return;
        lastDrawKeyRef.current = drawKey;

        const buildUniformPath = (stepPx: number): string => {
            if (!Number.isFinite(stepPx) || stepPx <= 0) return "";
            const firstIndex = Math.max(0, Math.floor((visibleStart + offset) / stepPx));
            const lastIndex = Math.max(
                firstIndex,
                Math.ceil((visibleEnd + offset) / stepPx),
            );
            const parts: string[] = [];
            for (let index = firstIndex; index <= lastIndex; index += 1) {
                const x = index * stepPx - offset;
                if (x < -1 || x > latest.width + 1) continue;
                parts.push(`M${x} 0V${latest.height}`);
            }
            return parts.join("");
        };

        const buildExplicitPath = (lineXs: number[] | null): string => {
            if (!lineXs || lineXs.length === 0) return "";
            const parts: string[] = [];
            // 二分定位可见范围
            const lo = 0;
            const hi = lineXs.length;
            const lowerBound = (target: number) => {
                let l = lo;
                let h = hi;
                while (l < h) {
                    const mid = (l + h) >> 1;
                    if (lineXs[mid] < target) l = mid + 1;
                    else h = mid;
                }
                return l;
            };
            const start = lowerBound(visibleStart + offset);
            for (let i = start; i < lineXs.length; i += 1) {
                const x = lineXs[i] - offset;
                if (x > latest.width + 1) break;
                if (x < -1) continue;
                parts.push(`M${x} 0V${latest.height}`);
            }
            return parts.join("");
        };

        paths[0].setAttribute(
            "d",
            useExplicitLines
                ? buildExplicitPath(latest.weakLineXs)
                : buildUniformPath(latest.weakStepPx),
        );
        paths[1].setAttribute(
            "d",
            useExplicitLines
                ? buildExplicitPath(latest.strongLineXs)
                : buildUniformPath(latest.strongStepPx),
        );
    }, [useExplicitLines]);

    useEffect(() => {
        draw(scrollLeft);
    }, [
        draw,
        scrollLeft,
        samplingPlan.weakStepPx,
        samplingPlan.strongStepPx,
        weakLineXs,
        strongLineXs,
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
