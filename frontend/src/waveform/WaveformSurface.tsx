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
import { LAYER_ORDER } from "../components/layout/timeline/runtime/timelineFrameCommitter.ts";
import type { TimelineLayer } from "../components/layout/timeline/runtime/timelineFrameCommitter.ts";
import { buildWaveformGeometry, type WaveformVertexSink } from "./geometry";
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
    /**
     * 视口来源（时间线或参数编辑器的总线）。
     * 提供时由总线的统一帧提交器驱动本图层的同步重绘。
     */
    viewportSource?: {
        getAxis(): TimelineAxis;
        register(layer: TimelineLayer, order: number): () => void;
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
    /**
     * 顶点缓冲槽：跨帧复用，稳态零分配。
     *
     * 此前 `buildWaveformGeometry` 每帧末尾 `slice()` 出一份独立副本
     * （实测 703 KB/帧），而顶点在 `render()` 内上传 GPU 之后 CPU 侧就不
     * 再需要——那份拷贝纯属浪费，反而制造了周期性的 GC 压力。
     * 渲染器不得跨 `render()` 边界持有 `geometry.vertices`。
     */
    const vertexSinkRef = React.useRef<WaveformVertexSink>({ buffer: new Float32Array(0) });
    /**
     * 几何缓存的锚点：记录「当前 GPU 上的几何是按什么窗口与缩放构建的」。
     *
     * `rows` / `color` 用引用比较：它们是 React 侧 memo 的产物，引用不变即
     * 内容不变。任一字段变化或视口越出窗口，都必须全量重建。
     */
    const geometryCacheRef = React.useRef<{
        pxPerSec: number;
        widthPx: number;
        heightPx: number;
        rows: readonly WaveformSceneRow[];
        color: string;
        rendererKind: "webgl2" | "canvas2d";
        /** 构建窗口的内容坐标左边界（含余量）。 */
        windowStartPx: number;
        windowEndPx: number;
        windowTopPx: number;
        windowBottomPx: number;
    } | null>(null);
    const [rendererKind, setRendererKind] = React.useState<"webgl2" | "canvas2d">("webgl2");

    // render 期写 ref 镜像（本仓库热路径既有模式，见 TimelineCanvasViewport）。
    // 目的：`draw` 的引用必须稳定——它一旦每渲染都变，下面的 layout effect
    // 就会每帧执行，与视口总线驱动的 paint 重复画一遍。故所有 props 改从
    // ref 读取，`draw` 的依赖里只剩真正会变的东西。
    const propsRef = React.useRef(props);
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    propsRef.current = props;

    const invalidate = React.useCallback(() => {
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
            drawRef.current();
        });
    }, []);

    /**
     * 绘制一帧波形。
     *
     * 流程：
     * 1. 取视口快照（总线驱动时用总线快照，否则用 props.axis）；
     * 2. **复用判定**：缩放、尺寸、行数据、颜色都没变，且视口矩形仍落在
     *    已构建的「窗口 + 余量」之内 → 走 `repaint()`，只更新视口原点，
     *    不重建场景与几何（WebGL 路径退化为一次 uniform 更新 + drawArrays）；
     * 3. 否则按窗口重建场景与几何，再 `render()`。
     *
     * 【坐标系】几何顶点是**窗口局部坐标** = 内容坐标 − 窗口左上角。实现上
     * 不给 `buildWaveformScene` 改签名，而是传一个派生 axis：
     * `scrollLeftPx = windowStartPx`、`viewportWidthPx = windowWidthPx`
     * —— 于是它的 `secToViewportPx()` 恰好产出 `contentPx − windowStart`，
     * 即窗口局部坐标；`viewportStartSec/EndSec` 也自然落在窗口上，裁剪语义
     * 原样成立。竖直方向同理（`viewportTopPx = windowTopPx`）。
     * 屏幕上最终位置 = 局部坐标 − 视口原点，其中
     * `视口原点 = scrollLeft − windowStart`（竖直同理）。
     *
     * 【为什么只对水平方向留余量】竖直方向的行集合由 React 窗口化给出，本身
     * 已带 4 行 overscan；再留竖直余量只会多建不存在的行，所以竖直窗口直接
     * 取行数据的实际覆盖范围。
     *
     * 依赖说明：props 一律经 `propsRef` 读取，**不进依赖**——否则每次父组件
     * 渲染都会换掉本函数引用，让下方 layout effect 每帧触发一次与总线 paint
     * 重复的全量重绘。
     */
    const draw = React.useCallback(() => {
        const props = propsRef.current;
        // 总线驱动时以总线投影为准：滚动事件在绘制前触发，sticky 波形面必须与
        // 原生滚动的 DOM 内容层在同一帧提交位移（DAW 式无缝滚动）。
        const source = props.viewportSource;
        const axis = source ? source.getAxis() : props.axis;
        const pxPerSec = axis.pxPerSec;
        const widthPx = Math.max(1, Math.floor(axis.viewportWidthPx));
        const heightPx = Math.max(1, Math.ceil(props.heightPx));
        const scrollLeftPx = axis.scrollLeftPx;
        // 竖直锚点：行坐标是内容绝对值，滚动容器竖直滚动时必须同步平移，
        // 否则波形与 DOM Clip 在竖直方向分层。
        const scrollTopPx = source ? axis.scrollTopPx : (props.viewportTopPx ?? 0);
        const dpr = window.devicePixelRatio || 1;

        const cache = geometryCacheRef.current;
        const canReuse =
            cache !== null &&
            cache.pxPerSec === pxPerSec &&
            cache.widthPx === widthPx &&
            cache.heightPx === heightPx &&
            cache.rows === props.rows &&
            cache.color === props.color &&
            cache.rendererKind === rendererKind &&
            // 水平：视口必须完整落在已构建的窗口内（两侧各 `marginPx` 可平移）。
            scrollLeftPx >= cache.windowStartPx &&
            scrollLeftPx + widthPx <= cache.windowEndPx &&
            // 竖直：只要求视口**顶边**落在行覆盖范围内。画布比视口高 8 行
            // （overscan），行集合不变时底边必然被覆盖。
            scrollTopPx >= cache.windowTopPx &&
            scrollTopPx <= cache.windowBottomPx;

        if (canReuse && cache !== null) {
            const renderer = cache.rendererKind === "webgl2" ? webglRendererRef.current : null;
            const active = renderer ?? fallbackRendererRef.current;
            if (active !== null) {
                active.repaint(
                    widthPx,
                    heightPx,
                    dpr,
                    scrollLeftPx - cache.windowStartPx,
                    scrollTopPx - cache.windowTopPx,
                );
                return;
            }
        }

        // ── 全量重建 ──────────────────────────────────────────────
        // 余量只给水平方向（WebGL 路径无内存成本；Canvas2D 回退没有顶点缓冲，
        // 平移仍要重放 path，加宽窗口只会白白多建几何，故余量取 0）。
        const marginPx =
            rendererKind === "webgl2"
                ? Math.min(512, Math.max(128, Math.round(widthPx * 0.25)))
                : 0;
        const windowStartPx = scrollLeftPx - marginPx;
        const windowEndPx = scrollLeftPx + widthPx + marginPx;
        // 竖直窗口 = 行数据覆盖的内容范围（行 topPx 是内容绝对坐标、按轨道
        // 顺序升序）。注意**不能**用 `heightPx` 做竖直包含判定：它是
        // `visibleTracks.length * rowHeight`，比视口高 8 行（overscan），
        // 拿它当视口高会得到永远不成立的条件。竖直方向实际上不需要余量——
        // 上游的轨道窗口化已带 4 行 overscan，行集合不变时几何必然覆盖视口。
        let firstRowTopPx = Number.POSITIVE_INFINITY;
        let lastRowTopPx = Number.NEGATIVE_INFINITY;
        for (const row of props.rows) {
            if (row.topPx < firstRowTopPx) firstRowTopPx = row.topPx;
            if (row.topPx > lastRowTopPx) lastRowTopPx = row.topPx;
        }
        let rowHeightPx: number;
        if (props.rows.length >= 2) {
            rowHeightPx = (lastRowTopPx - firstRowTopPx) / (props.rows.length - 1);
        } else if (props.rows.length === 1) {
            rowHeightPx = heightPx;
        } else {
            rowHeightPx = 0;
        }
        const windowTopPx = Number.isFinite(firstRowTopPx) ? firstRowTopPx : scrollTopPx;
        const windowBottomPx = Number.isFinite(lastRowTopPx)
            ? lastRowTopPx + Math.max(rowHeightPx, heightPx)
            : scrollTopPx + heightPx;

        const scene = buildWaveformScene({
            axis: withAxis(axis, {
                pxPerSec,
                scrollLeftPx: windowStartPx,
                scrollTopPx: windowTopPx,
                viewportWidthPx: windowEndPx - windowStartPx,
            }),
            widthPx: windowEndPx - windowStartPx,
            viewportTopPx: windowTopPx,
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
            sink: vertexSinkRef.current,
        });

        const originXPx = scrollLeftPx - windowStartPx;
        const originYPx = scrollTopPx - windowTopPx;
        const commitCache = (renderer: WaveformSurfaceRenderer): void => {
            geometryCacheRef.current = {
                pxPerSec,
                widthPx,
                heightPx,
                rows: props.rows,
                color: props.color,
                rendererKind: renderer.kind,
                windowStartPx,
                windowEndPx,
                windowTopPx,
                windowBottomPx,
            };
        };

        if (rendererKind === "webgl2" && webglRendererRef.current) {
            try {
                webglRendererRef.current.render(
                    geometry,
                    widthPx,
                    heightPx,
                    dpr,
                    originXPx,
                    originYPx,
                );
                commitCache(webglRendererRef.current);
                return;
            } catch (error) {
                console.warn("[WaveformSurface] WebGL2 render failed; using Canvas 2D", error);
                setRendererKind("canvas2d");
            }
        }
        if (fallbackRendererRef.current) {
            fallbackRendererRef.current.render(
                geometry,
                widthPx,
                heightPx,
                dpr,
                originXPx,
                originYPx,
            );
            commitCache(fallbackRendererRef.current);
        }
    }, [rendererKind]);

    /**
     * 视觉输入（行数据 / 颜色 / 尺寸 / 缩放）变化时必须在 layout effect
     * 内同步重绘：Clip 拖拽、trim/stretch、行窗口切换都由 React 在同一
     * commit 中更新 Clip DOM 与 Clip 体画布，波形面若再经 rAF 延迟一帧，
     * 就会在编辑手势中相对 Clip“甩出去”。
     * 视口滚动由 timelineViewportBus 的同步订阅负责，这里不重复绘制。
     *
     * **总线驱动时只把「缩放」留在签名里，滚动剔掉。**
     *
     * 总线驱动时视口取自 `source.getAxis()`，scrollLeft / scrollTop / 视口宽
     * 的变化都会 emit（滚动事件、ResizeObserver，时间线还有每帧对账兜底），
     * 因此它们不必进签名——放进来的话，滚动每帧都是一个新 axis 对象，于是每
     * 帧在总线 paint 之外又全量重绘一次，这正是 P1 要消除的重复绘制。
     *
     * 但**缩放不能剔**：参数编辑器缩放只 `flushSync` 写 state
     * （`PianoRollPanel.tsx:1214`），若锚点恰好让 scrollLeft 不变就不会触发
     * 原生 scroll 事件 → 不 emit → 波形停帧。故单独保留 `pxPerSec`。
     * 非总线驱动时 axis 是唯一视口来源，整体进签名。
     */
    const visualSignature = React.useMemo(
        () => ({
            rows: props.rows,
            color: props.color,
            heightPx: props.heightPx,
            viewportTopPx: props.viewportTopPx,
            pxPerSec: props.axis.pxPerSec,
            axis: props.viewportSource ? null : props.axis,
        }),
        [
            props.rows,
            props.color,
            props.heightPx,
            props.viewportTopPx,
            props.axis,
            props.viewportSource,
        ],
    );
    const previousVisualSignatureRef = React.useRef(visualSignature);

    React.useLayoutEffect(() => {
        drawRef.current = draw;
        const busDriven = props.viewportSource != null;
        if (!busDriven || previousVisualSignatureRef.current !== visualSignature) {
            previousVisualSignatureRef.current = visualSignature;
            drawRef.current();
        }
    }, [draw, props.viewportSource, rendererKind, visualSignature]);

    React.useEffect(() => {
        const fallbackCanvas = fallbackCanvasRef.current;
        const webglCanvas = webglCanvasRef.current;
        if (!fallbackCanvas || !webglCanvas) return;
        // 几何缓存必须随渲染器一起作废：新渲染器的顶点缓冲是空的，而缓存
        // 只认「缩放 / 尺寸 / 行数据 / 窗口」，不知道 GPU 侧已经重置。留着
        // 它会让首帧走 repaint、画出 0 个顶点（波形空白）。
        geometryCacheRef.current = null;
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
            geometryCacheRef.current = null;
            setRendererKind("canvas2d");
            invalidate();
        };
        const onContextRestored = () => {
            try {
                webglRendererRef.current?.dispose();
                webglRendererRef.current = new WebGl2WaveformRenderer(webglCanvas);
                // 上下文丢失后 GPU 侧缓冲随之中断，必须重新上传。
                geometryCacheRef.current = null;
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
            if (status !== "done" || !needed.has(sourcePath)) return;
            // 数据就绪会改变几何结果。「余量窗口」复用判定只看 rows 引用 /
            // axis / 尺寸（见 draw 内 canReuse），不感知峰值数据是否已加载——
            // 若只 invalidate，draw 会命中 canReuse 走 repaint()，把**数据缺失
            // 时**构建的空几何原样重画一遍：表现为打开工程/导入音频并分析完成
            // 后波形仍空白，要等滚动滚出余量或缩放才出现。必须先作废几何缓存
            // 强制全量重建。（7286592a 修的是视口宽度不发布，本处是同一症状
            // 的第二个根源。）
            geometryCacheRef.current = null;
            invalidate();
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
        // 注册到统一帧提交器：由它保证本图层与网格 / clip 体的绘制顺序固定，
        // 且同一帧内重复的视口提交只触发一次重绘。
        return source.register({ name: "waveform", paint: apply }, LAYER_ORDER.waveform);
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
