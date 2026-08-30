import React, { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { explicitGridLinesKey } from "./gridLineKey";
import { resolveGridLineSamplingPlan } from "./gridLineSampling";
import { clearGridRedrawHandler, setGridRedrawHandler } from "./gridRedrawBridge";
import type { TimelineAxis } from "./runtime/timelineAxis";
import type { TimelineTick } from "./runtime/buildTimelineTicks";
import type { TimelineLayer } from "./runtime/timelineFrameCommitter";

/**
 * Grid lines are drawn as SVG paths computed directly from beat positions.
 * Repeating CSS gradients with fractional background sizes can accumulate
 * subpixel rounding drift, which makes the minor grid shift relative to the
 * real beat/snap positions.
 *
 * 提供 `weakLineXs` / `strongLineXs`（内容坐标系 x 像素数组）时，
 * 网格线直接使用这些显式位置（Tempo Map 的不等距网格）。
 */

function resolveRefElement(ref: React.Ref<HTMLDivElement> | undefined): HTMLDivElement | null {
    if (ref == null) return null;
    if (typeof ref === "function") return null;
    return ref.current;
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
    lineOpacity?: number;
    sticky?: boolean;
    /** 网格显示总开关（Snap/Grid 设置）。 */
    visible?: boolean;
    /** 用户配置的最小弱网格线像素间距。 */
    minSpacingPx?: number;
    /** Swing 强度（0-100），仅作用于弱网格线的奇数格。 */
    swingPercent?: number;
    /**
     * Sticky 视口的竖直偏移（内容绝对坐标）。网格线从 -viewportTopPx
     * 处开始可见；垂直滚动时由命令式 draw(scrollLeft, scrollTopPx) 同步。
     */
    viewportTopPx?: number;
    /**
     * 网格在内容绝对坐标中的底部边界。Sticky 层绘制时会把网格裁剪到
     * [0, contentBottomPx]，避免覆盖“添加轨道”等网格区之外的底部内容。
     */
    contentBottomPx?: number;
    /** Tempo Map 显式网格线位置（内容坐标 x，升序）。 */
    weakLineXs?: number[] | null;
    strongLineXs?: number[] | null;
    /**
     * 视口总线：提供时本网格会注册为统一帧提交的图层，由提交器保证与 clip 体
     * / 波形的绘制顺序固定。不提供时仍走 gridRedrawBridge 的命令式调用。
     */
    viewportBus?: {
        getAxis(): TimelineAxis;
        register(layer: TimelineLayer, order: number): () => void;
    };
    /** 在统一帧提交中的绘制顺序，取 LAYER_ORDER 中的值；需与 viewportBus 同传。 */
    layerOrder?: number;
    /**
     * 统一刻度源（推荐）：提供时网格线直接画在刻度的内容坐标上，与标尺严格
     * 同源。不提供时退化到按 pxPerBeat 自行采样——仅供尚未接入 axis 的调用方
     * （参数编辑器）过渡使用。
     */
    ticks?: readonly TimelineTick[] | null;
}> = ({
    contentWidth,
    contentHeight,
    pxPerBeat,
    grid,
    beatsPerBar,
    viewportWidth,
    scrollLeft,
    layerRef,
    lineOpacity = 0.9,
    sticky = false,
    visible = true,
    minSpacingPx,
    swingPercent = 0,
    viewportTopPx = 0,
    contentBottomPx,
    weakLineXs = null,
    strongLineXs = null,
    viewportBus,
    layerOrder,
    ticks = null,
}) => {
    const svgRef = useRef<SVGSVGElement | null>(null);

    // 统一刻度源：拆成弱线/强线两组内容坐标，复用"显式线"绘制路径。
    // Swing 与小节抽取已在 buildTimelineTicks 内完成，这里只负责画。
    const tickLineXs = React.useMemo(() => {
        if (!ticks || ticks.length === 0) return null;
        const weak: number[] = [];
        const strong: number[] = [];
        for (const tick of ticks) {
            (tick.isStrongGridLine ? strong : weak).push(tick.contentPx);
        }
        return { weak, strong };
    }, [ticks]);

    const useViewport =
        viewportWidth != null &&
        Number.isFinite(viewportWidth) &&
        viewportWidth > 0 &&
        scrollLeft != null &&
        Number.isFinite(scrollLeft);
    const isSticky = sticky && useViewport;

    // 统一刻度源优先：网格线与标尺同源。退化路径用调用方显式传入的数组
    // （参数编辑器尚未接入 axis 时的过渡形态）。
    const effectiveWeakXs = weakLineXs ?? tickLineXs?.weak ?? null;
    const effectiveStrongXs = strongLineXs ?? tickLineXs?.strong ?? null;
    const useExplicitLines = effectiveWeakXs != null && Array.isArray(effectiveWeakXs);

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
              minWeakSpacingPx: minSpacingPx,
          });

    const width = isSticky ? Math.max(1, Math.floor(viewportWidth as number)) : contentWidth;
    const height = contentHeight;

    const latestRef = useRef({
        weakStepPx: samplingPlan.weakStepPx,
        strongStepPx: samplingPlan.strongStepPx,
        swingPercent: Math.max(0, Math.min(100, swingPercent)),
        weakLineXs: effectiveWeakXs,
        strongLineXs: effectiveStrongXs,
        width,
        height,
        contentWidth,
        viewportWidth:
            viewportWidth != null && Number.isFinite(viewportWidth) && viewportWidth > 0
                ? viewportWidth
                : contentWidth,
        scrollLeft: scrollLeft ?? 0,
        isSticky,
        lineOpacity,
        viewportTopPx,
        contentBottomPx,
    });

    useLayoutEffect(() => {
        latestRef.current = {
            weakStepPx: samplingPlan.weakStepPx,
            strongStepPx: samplingPlan.strongStepPx,
            swingPercent: Math.max(0, Math.min(100, swingPercent)),
            weakLineXs: effectiveWeakXs,
            strongLineXs: effectiveStrongXs,
            width,
            height,
            contentWidth,
            viewportWidth:
                viewportWidth != null && Number.isFinite(viewportWidth) && viewportWidth > 0
                    ? viewportWidth
                    : contentWidth,
            scrollLeft: scrollLeft ?? 0,
            isSticky,
            lineOpacity,
            viewportTopPx,
            contentBottomPx,
        };
    });

    const lastDrawKeyRef = useRef<string | null>(null);

    useLayoutEffect(() => {
        lastDrawKeyRef.current = null;
    });

    const draw = useCallback(
        (nextScrollLeft?: number, nextViewportTopPx?: number) => {
            const svg = svgRef.current;
            if (!svg) return;
            const paths = svg.querySelectorAll<SVGPathElement>("path");
            if (paths.length < 2) return;

            const latest = latestRef.current;
            const sl = Number.isFinite(nextScrollLeft)
                ? (nextScrollLeft as number)
                : latest.scrollLeft;
            const vpTop = Number.isFinite(nextViewportTopPx)
                ? (nextViewportTopPx as number)
                : latest.viewportTopPx;
            const offset = latest.isSticky ? sl : 0;
            const bufferPx = Math.max(240, latest.viewportWidth * 0.5);
            const visibleStart = latest.isSticky ? 0 : Math.max(0, sl - bufferPx);
            const visibleEnd = latest.isSticky
                ? latest.width
                : Math.min(latest.contentWidth, sl + latest.viewportWidth + bufferPx);

            // Sticky 层绘制时按内容绝对坐标裁剪竖直范围：
            // 网格只覆盖 [0, contentBottomPx]，不得画进底部“添加轨道”行。
            let lineTop = 0;
            let lineBottom = latest.height;
            if (latest.isSticky && Number.isFinite(latest.contentBottomPx)) {
                lineTop = Math.max(0, -vpTop);
                lineBottom = Math.max(
                    lineTop,
                    Math.min(latest.height, (latest.contentBottomPx as number) - vpTop),
                );
            }

            // 重绘跳过键必须覆盖**全部**网格线位置：拖动 Tempo Map 的中间变化点时，
            // 受影响的是数组中部以该点为锚的整段线（整体平移），而长度与首尾线不变，
            // 任何抽样校验和都会误判“无需重绘”，造成网格跳变/错位（见 gridLineKey.ts）。
            // 竖线一根到底、不分段：分段会让每个行边界的端点各自做设备像素
            // 取整，接缝处互相让位，视觉上就是"深浅不一的断线"。
            const ySegments: Array<[number, number]> = [[lineTop, lineBottom]];

            const drawKey = [
                sl,
                vpTop,
                latest.weakStepPx,
                latest.strongStepPx,
                latest.swingPercent,
                explicitGridLinesKey(latest.weakLineXs),
                explicitGridLinesKey(latest.strongLineXs),
                latest.width,
                latest.height,
                latest.contentWidth,
                latest.viewportWidth,
                latest.isSticky,
                latest.lineOpacity,
                latest.contentBottomPx,
                window.devicePixelRatio || 1,
            ].join("|");
            if (lastDrawKeyRef.current === drawKey) return;
            lastDrawKeyRef.current = drawKey;

            if (lineBottom <= lineTop) {
                paths[0].setAttribute("d", "");
                paths[1].setAttribute("d", "");
                return;
            }

            /**
             * 竖线的落笔 x。
             *
             * 线宽与位置都按**设备像素**取整（见下方 stroke-width 说明）：
             * 分数 DPR（Windows 125%/150% 缩放、浏览器缩放）下，1px CSS 线
             * 覆盖 1.25/1.5 物理像素，不同线落在不同亚像素相位上，取整后
             * 有的 1 物理像素、有的 2 物理像素 —— 这就是缩放时"粗细不一"。
             * 先把 x 吸附到设备像素边界，再配以物理像素线宽，任何缩放下
             * 每根弱线都恰好 1 物理像素、强线恰好 2 物理像素。
             * （旧的 +0.5 半像素偏移是 dpr=1 时代与标尺 DOM 盒子对齐用的，
             * 设备像素吸附后不再需要。）
             */
            const dpr = window.devicePixelRatio || 1;
            const deviceSnap = (cssX: number): number => Math.round(cssX * dpr) / dpr;
            const buildUniformPath = (stepPx: number): string => {
                if (!Number.isFinite(stepPx) || stepPx <= 0) return "";
                const firstIndex = Math.max(0, Math.floor((visibleStart + offset) / stepPx));
                const lastIndex = Math.max(firstIndex, Math.ceil((visibleEnd + offset) / stepPx));
                const swingPx =
                    (Math.max(0, Math.min(100, latest.swingPercent)) / 100) * 0.5 * stepPx;
                const parts: string[] = [];
                for (let index = firstIndex; index <= lastIndex; index += 1) {
                    // Swing：奇数网格位置向右偏移（最大半步）。
                    const x = deviceSnap(
                        index * stepPx + (index % 2 === 0 ? 0 : swingPx) - offset,
                    );
                    if (x < -1 || x > latest.width + 1) continue;
                    for (const [segTop, segBottom] of ySegments) {
                        parts.push(`M${x} ${segTop}V${segBottom}`);
                    }
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
                    const x = deviceSnap(lineXs[i] - offset);
                    if (x > latest.width + 1) break;
                    if (x < -1) continue;
                    for (const [segTop, segBottom] of ySegments) {
                        parts.push(`M${x} ${segTop}V${segBottom}`);
                    }
                }
                return parts.join("");
            };

            // 线宽用物理像素整数：配合 deviceSnap，任意缩放下相邻线粗细一致。
            paths[0].setAttribute("stroke-width", String(1 / dpr));
            paths[1].setAttribute("stroke-width", String(2 / dpr));
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
        },
        [useExplicitLines],
    );

    // 浏览器缩放 / 跨屏拖动会改变 devicePixelRatio：线宽按物理像素取整，
    // dpr 变化后必须重绘一次，否则旧的吸附相位会残留。
    useEffect(() => {
        const onResize = () => draw();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [draw]);

    // 绘制必须在 paint 前同步完成（useLayoutEffect）：缩放时网格线的间距
    // 随 pxPerBeat 变化，若走 passive useEffect 会在 DOM 重排后的下一帧才
    // 切换，与 Clip/标尺产生一帧错位。滚动仅影响窗口化（位置为内容坐标，
    // 随原生滚动移动），同样受益于同帧提交。
    useLayoutEffect(() => {
        draw(scrollLeft, viewportTopPx);
    }, [
        draw,
        scrollLeft,
        viewportTopPx,
        samplingPlan.weakStepPx,
        samplingPlan.strongStepPx,
        swingPercent,
        effectiveWeakXs,
        effectiveStrongXs,
        width,
        height,
        contentWidth,
        viewportWidth,
        isSticky,
        lineOpacity,
        contentBottomPx,
    ]);

    // 用 useLayoutEffect 注册命令式重绘句柄：TimelineSurface 挂载后会在
    // 父级 layout effect 中立即用总线快照同步一次，句柄必须已在 paint 前可用。
    useLayoutEffect(() => {
        const el = resolveRefElement(layerRef);
        if (!el) return;
        setGridRedrawHandler(el, draw);
        return () => {
            if (el) {
                clearGridRedrawHandler(el);
            }
        };
    }, [draw, layerRef]);

    // 注册为统一帧提交的图层：滚动 / 缩放时由提交器按固定顺序调用，无需调用
    // 方记得单独通知网格（历史上漏通知会造成网格与 Clip/波形分层）。
    useEffect(() => {
        const bus = viewportBus;
        if (!bus || layerOrder == null) return;
        return bus.register(
            {
                name: `grid-${layerOrder}`,
                paint: (axis) => draw(axis.scrollLeftPx, axis.scrollTopPx),
            },
            layerOrder,
        );
    }, [draw, layerOrder, viewportBus]);

    if (!visible) return null;

    return (
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
                    shapeRendering="crispEdges"
                    style={{ stroke: "var(--qt-graph-grid-weak)" }}
                />
                <path
                    fill="none"
                    strokeWidth={2}
                    opacity={lineOpacity}
                    shapeRendering="crispEdges"
                    style={{ stroke: "var(--qt-graph-grid-strong)" }}
                />
            </svg>
        </div>
    );
};
