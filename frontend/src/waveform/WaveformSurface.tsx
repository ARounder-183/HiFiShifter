/**
 * 波形画布（WebGL2 主路径 + Canvas2D 回退）的 React 入口。
 *
 * 【主要内容】组装波形绘制所需的视口状态、调度场景与几何构建、管理渲染器
 * 生命周期（含 WebGL 上下文丢失恢复）、并在视口变化时同步重绘。
 *
 * 【作用】时间线与参数编辑器共用同一块波形面，本组件是它们与底层渲染器
 * 之间唯一的适配层：把「行数据 + 坐标投影」翻译成「场景 → 几何 → 顶点」。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelineWaveformSurface` / `PianoRollWaveformSurface` 组装
 *   `WaveformSceneRow[]` 并传入 `TimelineAxis`。
 * - 横向：所有时间↔像素换算由 `timelineAxis.ts` 提供；本组件在总线驱动时用
 *   总线快照派生 axis（`withAxis`），保证与 DOM 内容层同帧。
 * - 下游：`sceneBuilder.ts` → `geometry.ts` → `surfaceRenderer.ts`。
 */

import React from "react";

import { waveformMipmapStore } from "../utils/waveformMipmapStore";
import type { TimelineAxis } from "../components/layout/timeline/runtime/timelineAxis.ts";
import { withAxis } from "../components/layout/timeline/runtime/timelineAxis.ts";
import { buildWaveformGeometry } from "./geometry";
import { buildWaveformScene, type WaveformSceneRow } from "./sceneBuilder";
import {
    Canvas2dWaveformRenderer,
    WebGl2WaveformRenderer,
    type WaveformSurfaceRenderer,
} from "./surfaceRenderer";

export interface WaveformSurfaceProps {
    rows: readonly WaveformSceneRow[];
    widthPx: number;
    heightPx: number;
    /**
     * 统一坐标投影：视口起点、缩放倍率、滚动位置的**唯一来源**。
     * 总线驱动时会被总线快照覆盖 scrollLeftPx / pxPerSec / scrollTopPx。
     */
    axis: TimelineAxis;
    color: string;
    className?: string;
    style?: React.CSSProperties;
    /** 行 topPx 坐标系（内容绝对）到画布的竖直偏移。
     * 仅在无 viewportSource（非总线驱动）时作为回退值；总线驱动时一律以
     * axis.scrollTopPx 为准。 */
    viewportTopPx?: number;
    viewportSource?: {
        getSnapshot(): {
            scrollLeft: number;
            pxPerSec: number;
            viewportWidth: number;
            scrollTopPx?: number;
        };
        subscribe(
            listener: (scrollLeft: number, pxPerSec: number, width: number) => void,
        ): () => void;
    };
}

export const WaveformSurface = React.memo(function WaveformSurface(props: WaveformSurfaceProps) {
    const webglCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const fallbackCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const webglRendererRef = React.useRef<WebGl2WaveformRenderer | null>(null);
    const fallbackRendererRef = React.useRef<Canvas2dWaveformRenderer | null>(null);
    const rafRef = React.useRef<number | null>(null);
    const drawRef = React.useRef<() => void>(() => {});
    const rootRef = React.useRef<HTMLDivElement | null>(null);
    const [rendererKind, setRendererKind] = React.useState<"webgl2" | "canvas2d">("webgl2");

    const invalidate = React.useCallback(() => {
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    const renderWith = React.useCallback(
        (
            renderer: WaveformSurfaceRenderer,
            geometry: ReturnType<typeof buildWaveformGeometry>,
            current: WaveformSurfaceProps,
            widthPx: number,
        ) => {
            renderer.render(
                geometry,
                Math.max(1, widthPx),
                Math.max(1, current.heightPx),
                window.devicePixelRatio || 1,
            );
        },
        [],
    );

    /**
     * 按当前视口重建波形场景与几何并提交渲染。
     *
     * 流程：
     * 1. 取视口快照：总线驱动时用总线快照派生 axis，否则用 props.axis；
     * 2. 由 axis 构建场景（`buildWaveformScene`）与顶点几何；
     * 3. 提交给 WebGL2 渲染器，失败时降级到 Canvas2D。
     *
     * 特殊说明：视口的秒级窗口一律由 axis 派生（禁止 `scrollLeft / pxPerSec`
     * 反算），以保证与 clip 体画布、网格、标尺严格同源。
     */
    const draw = React.useCallback(() => {
        const liveViewport = props.viewportSource?.getSnapshot();
        // 滚动事件在绘制前触发：总线快照优先，保证 sticky 波形面与原生滚动的
        // DOM 内容层在同一帧提交位移（DAW 式无缝滚动）。
        const axis = liveViewport
            ? withAxis(props.axis, {
                  pxPerSec: liveViewport.pxPerSec,
                  scrollLeftPx: liveViewport.scrollLeft,
                  viewportWidthPx: liveViewport.viewportWidth,
                  scrollTopPx: liveViewport.scrollTopPx,
              })
            : props.axis;
        const pxPerSec = axis.pxPerSec;
        const widthPx = axis.viewportWidthPx;
        // 竖直锚点：行坐标是内容绝对值，滚动容器竖直滚动时必须同步平移，
        // 否则波形与 DOM Clip 在竖直方向分层。
        const viewportTopPx = liveViewport ? axis.scrollTopPx : (props.viewportTopPx ?? 0);
        const scene = buildWaveformScene({
            axis,
            widthPx,
            viewportTopPx,
            rows: props.rows,
        });
        const geometry = buildWaveformGeometry({
            scene,
            color: props.color,
            getPeaks: (sourcePath, sampleRate, sourceStartSec, sourceDurationSec) => {
                const level = waveformMipmapStore.selectLevel(
                    Math.max(1, Math.round(sampleRate / Math.max(1e-6, pxPerSec))),
                );
                return waveformMipmapStore.getBestSliceView(
                    sourcePath,
                    level,
                    sourceStartSec,
                    sourceDurationSec,
                );
            },
        });

        if (rendererKind === "webgl2" && webglRendererRef.current) {
            try {
                renderWith(webglRendererRef.current, geometry, props, widthPx);
                return;
            } catch (error) {
                console.warn("[WaveformSurface] WebGL2 render failed; using Canvas 2D", error);
                setRendererKind("canvas2d");
            }
        }
        if (fallbackRendererRef.current) {
            renderWith(fallbackRendererRef.current, geometry, props, widthPx);
        }
    }, [props, renderWith, rendererKind]);

    /**
     * 视觉输入（行数据 / 颜色 / 尺寸 / 缩放）变化时必须在 layout effect
     * 内同步重绘：Clip 拖拽、trim/stretch、行窗口切换都由 React 在同一
     * commit 中更新 Clip DOM 与 Clip 体画布，波形面若再经 rAF 延迟一帧，
     * 就会在编辑手势中相对 Clip“甩出去”。
     * 视口滚动由 timelineViewportBus 的同步订阅负责，这里不重复绘制。
     */
    const visualSignature = React.useMemo(
        () => ({
            rows: props.rows,
            color: props.color,
            widthPx: props.widthPx,
            heightPx: props.heightPx,
            axis: props.axis,
        }),
        [props.rows, props.color, props.widthPx, props.heightPx, props.axis],
    );
    const previousVisualSignatureRef = React.useRef(visualSignature);

    React.useLayoutEffect(() => {
        drawRef.current = draw;
        const busDriven = props.viewportSource != null;
        if (!busDriven || previousVisualSignatureRef.current !== visualSignature) {
            previousVisualSignatureRef.current = visualSignature;
            drawRef.current();
        }
    }, [draw, props.viewportSource, visualSignature]);

    React.useEffect(() => {
        const fallbackCanvas = fallbackCanvasRef.current;
        const webglCanvas = webglCanvasRef.current;
        if (!fallbackCanvas || !webglCanvas) return;
        fallbackRendererRef.current = new Canvas2dWaveformRenderer(fallbackCanvas);
        try {
            webglRendererRef.current = new WebGl2WaveformRenderer(webglCanvas);
            setRendererKind("webgl2");
        } catch (error) {
            console.warn("[WaveformSurface] WebGL2 unavailable; using Canvas 2D", error);
            setRendererKind("canvas2d");
        }
        invalidate();

        const onContextLost = (event: Event) => {
            event.preventDefault();
            setRendererKind("canvas2d");
            invalidate();
        };
        const onContextRestored = () => {
            try {
                webglRendererRef.current?.dispose();
                webglRendererRef.current = new WebGl2WaveformRenderer(webglCanvas);
                setRendererKind("webgl2");
            } catch {
                setRendererKind("canvas2d");
            }
            invalidate();
        };
        webglCanvas.addEventListener("webglcontextlost", onContextLost);
        webglCanvas.addEventListener("webglcontextrestored", onContextRestored);

        return () => {
            webglCanvas.removeEventListener("webglcontextlost", onContextLost);
            webglCanvas.removeEventListener("webglcontextrestored", onContextRestored);
            webglRendererRef.current?.dispose();
            fallbackRendererRef.current?.dispose();
            webglRendererRef.current = null;
            fallbackRendererRef.current = null;
        };
    }, [invalidate]);

    React.useEffect(() => {
        const sourcePaths = Array.from(
            new Set(
                props.rows.flatMap((row) =>
                    row.clips.map((clip) => clip.sourcePath).filter(Boolean),
                ),
            ),
        );
        if (sourcePaths.length > 0) void waveformMipmapStore.batchPreload(sourcePaths);
    }, [props.rows]);

    React.useEffect(() => {
        const needed = new Set(
            props.rows.flatMap((row) => row.clips.map((clip) => clip.sourcePath)),
        );
        return waveformMipmapStore.addListener((sourcePath, status) => {
            if (status === "done" && needed.has(sourcePath)) invalidate();
        });
    }, [invalidate, props.rows]);

    React.useEffect(() => {
        const source = props.viewportSource;
        if (!source) return;
        const apply = () => {
            // 同步绘制：滚动事件在绘制前触发，波形面必须与原生滚动的 DOM
            // 内容层在同一帧内提交位移（DAW 式无缝滚动），禁止 rAF 延迟。
            drawRef.current();
        };
        apply();
        return source.subscribe(() => apply());
    }, [props.viewportSource]);

    React.useEffect(
        () => () => {
            if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
        },
        [],
    );

    const commonStyle: React.CSSProperties = {
        position: "absolute",
        inset: 0,
        pointerEvents: "none",
        ...props.style,
    };
    return (
        <div
            ref={rootRef}
            className={props.className}
            style={commonStyle}
            data-waveform-renderer={rendererKind}
        >
            <canvas
                ref={webglCanvasRef}
                style={{
                    position: "absolute",
                    inset: 0,
                    opacity: rendererKind === "webgl2" ? 1 : 0,
                }}
            />
            <canvas
                ref={fallbackCanvasRef}
                style={{
                    position: "absolute",
                    inset: 0,
                    opacity: rendererKind === "canvas2d" ? 1 : 0,
                }}
            />
        </div>
    );
});
