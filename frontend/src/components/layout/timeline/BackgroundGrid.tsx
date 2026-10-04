import React, { useCallback, useEffect, useLayoutEffect, useRef } from "react";
import { explicitGridLinesKey } from "./gridLineKey";
import { resolveGridLineSamplingPlan } from "./gridLineSampling";
import { resolveGridDrawViewport } from "./gridDrawViewport";
import { clearGridRedrawHandler, setGridRedrawHandler } from "./gridRedrawBridge";
import type { TimelineAxis } from "../renderKernel/timelineAxis";
import type { TimelineTick } from "./runtime/buildTimelineTicks";
import type { TimelineLayer } from "./runtime/timelineFrameCommitter";
import { subscribeDevicePixelRatio } from "../../../hooks/useDevicePixelRatio";

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
        weakLineXsKey: explicitGridLinesKey(effectiveWeakXs),
        strongLineXsKey: explicitGridLinesKey(effectiveStrongXs),
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
            weakLineXsKey: explicitGridLinesKey(effectiveWeakXs),
            strongLineXsKey: explicitGridLinesKey(effectiveStrongXs),
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
            //
            // 因此键由 `latestRef` 提供（`weakLineXsKey` / `strongLineXsKey`），
            // 与数组一起在渲染期算好——`explicitGridLinesKey` 是 `xs.join(",")`，
            // 对上百个刻度做一次全量字符串拼接，放在 `draw()` 里就是**每帧两次**
            // （网格是注册图层）。它只依赖刻度数组，没必要每帧重算。
            // 竖线一根到底、不分段：分段会让每个行边界的端点各自做设备像素
            // 取整，接缝处互相让位，视觉上就是"深浅不一的断线"。
            const ySegments: Array<[number, number]> = [[lineTop, lineBottom]];

            const drawKey = [
                sl,
                vpTop,
                latest.weakStepPx,
                latest.strongStepPx,
                latest.swingPercent,
                latest.weakLineXsKey,
                latest.strongLineXsKey,
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
             *
             * 【奇数与偶数物理像素宽的线，中心落点不同 —— 这是关键】
             * SVG 的描边以给定 x 为中心、向两侧各展开半个线宽。要恰好覆盖整数列：
             * - 宽 1 物理像素（弱线）：中心必须落在**半格**（k + 0.5）上，覆盖列 k；
             * - 宽 2 物理像素（强线）：中心必须落在**整格**（k）上，覆盖列 k−1 与 k。
             * 若两者都用整格吸附，弱线会横跨相邻两列、各覆盖一半 —— 开抗锯齿时
             * 渲染成两条半亮的线（看起来更糊），开 `crispEdges` 时则由渲染器自行
             * 取整，落点随机、相邻线粗细不一。这正是本组件此前残留的"网格线发虚"。
             */
            const dpr = window.devicePixelRatio || 1;
            /** 中心吸附到设备像素**边界**：供宽 2 物理像素的强线使用。 */
            const deviceSnapBoundary = (cssX: number): number => Math.round(cssX * dpr) / dpr;
            /** 中心吸附到设备像素**半格**：供宽 1 物理像素的弱线使用。 */
            const deviceSnapHalf = (cssX: number): number => (Math.floor(cssX * dpr) + 0.5) / dpr;
            const buildUniformPath = (
                stepPx: number,
                deviceSnap: (cssX: number) => number,
            ): string => {
                if (!Number.isFinite(stepPx) || stepPx <= 0) return "";
                const firstIndex = Math.max(0, Math.floor((visibleStart + offset) / stepPx));
                const lastIndex = Math.max(firstIndex, Math.ceil((visibleEnd + offset) / stepPx));
                const swingPx =
                    (Math.max(0, Math.min(100, latest.swingPercent)) / 100) * 0.5 * stepPx;
                const parts: string[] = [];
                for (let index = firstIndex; index <= lastIndex; index += 1) {
                    // Swing：奇数网格位置向右偏移（最大半步）。
                    const x = deviceSnap(index * stepPx + (index % 2 === 0 ? 0 : swingPx) - offset);
                    if (x < -1 || x > latest.width + 1) continue;
                    for (const [segTop, segBottom] of ySegments) {
                        parts.push(`M${x} ${segTop}V${segBottom}`);
                    }
                }
                return parts.join("");
            };

            const buildExplicitPath = (
                lineXs: number[] | null,
                deviceSnap: (cssX: number) => number,
            ): string => {
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

            // 线宽用物理像素整数，并按奇偶选择对应的中心吸附（见上）。
            paths[0].setAttribute("stroke-width", String(1 / dpr));
            paths[1].setAttribute("stroke-width", String(2 / dpr));
            paths[0].setAttribute(
                "d",
                useExplicitLines
                    ? buildExplicitPath(latest.weakLineXs, deviceSnapHalf)
                    : buildUniformPath(latest.weakStepPx, deviceSnapHalf),
            );
            paths[1].setAttribute(
                "d",
                useExplicitLines
                    ? buildExplicitPath(latest.strongLineXs, deviceSnapBoundary)
                    : buildUniformPath(latest.strongStepPx, deviceSnapBoundary),
            );
        },
        [useExplicitLines],
    );

    // 统一取视口：有总线时一律用总线快照（原生滚动真值），仅无总线时退回
    // React props（量化 scrollLeft，仅供参数编辑器过渡路径）。见
    // gridDrawViewport.ts 顶注——React 重绘若消费量化 scrollLeft，拖拽 Clip
    // 改变 contentWidth 触发重绘时会把网格画在滞后偏移上，且帧提交器按
    // axisEquals 去重后不会再纠正。
    const resolveDrawViewport = useCallback(
        (): { scrollLeftPx: number; scrollTopPx: number } =>
            resolveGridDrawViewport({
                busAxis: viewportBus?.getAxis() ?? null,
                propScrollLeftPx: latestRef.current.scrollLeft,
                propViewportTopPx: latestRef.current.viewportTopPx,
            }),
        [viewportBus],
    );

    // 浏览器缩放 / 跨屏拖动会改变 devicePixelRatio：线宽按物理像素取整，
    // dpr 变化后必须重绘一次，否则旧的吸附相位会残留。
    //
    // 除 `window.resize` 外还必须订阅 `(resolution: N dppx)`：把窗口拖到另一台
    // 缩放率不同的显示器上时，CSS 尺寸不变、`resize` 事件不保证触发，而吸附相位
    // 已经失效 —— 没有这条订阅，网格会停留在旧 dpr 的线宽上。
    useEffect(() => {
        const onResize = () => {
            const vp = resolveDrawViewport();
            draw(vp.scrollLeftPx, vp.scrollTopPx);
        };
        window.addEventListener("resize", onResize);
        const unsubscribeDpr = subscribeDevicePixelRatio(onResize);
        return () => {
            window.removeEventListener("resize", onResize);
            unsubscribeDpr();
        };
    }, [draw, resolveDrawViewport]);

    // 绘制必须在 paint 前同步完成（useLayoutEffect）：缩放时网格线的间距
    // 随 pxPerBeat 变化，若走 passive useEffect 会在 DOM 重排后的下一帧才
    // 切换，与 Clip/标尺产生一帧错位。滚动仅影响窗口化（位置为内容坐标，
    // 随原生滚动移动），同样受益于同帧提交。
    //
    // 水平/竖直偏移经 resolveDrawViewport 取权威视口（总线快照），而非本组
    // 件的 scrollLeft prop——后者是量化提交的 React state，可永久滞后原生
    // 滚动最多一个死区宽度（256px）。数据依赖（ticks/contentWidth/尺寸…）
    // 变化触发的本次重绘必须与总线 paint 输出逐像素一致。
    useLayoutEffect(() => {
        const vp = resolveDrawViewport();
        draw(vp.scrollLeftPx, vp.scrollTopPx);
    }, [
        draw,
        resolveDrawViewport,
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

    // 用 useLayoutEffect 注册命令式重绘句柄：挂载它的父级（PianoRollPanel 的网格层；
    // 旧 TimelineSurface 已随旧渲染路径删除）会在 layout effect 中立即用总线快照同步
    // 一次，句柄必须已在 paint 前可用。
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
