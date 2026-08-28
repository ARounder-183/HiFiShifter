import React from "react";

import { waveformMipmapStore } from "../utils/waveformMipmapStore";
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
    viewportStartSec: number;
    viewportEndSec: number;
    pxPerSec: number;
    color: string;
    className?: string;
    style?: React.CSSProperties;
    viewportSource?: {
        getSnapshot(): { scrollLeft: number; pxPerSec: number; viewportWidth: number };
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

    const draw = React.useCallback(() => {
        const liveViewport = props.viewportSource?.getSnapshot();
        const pxPerSec = liveViewport?.pxPerSec ?? props.pxPerSec;
        const widthPx = liveViewport?.viewportWidth ?? props.widthPx;
        const viewportStartSec = liveViewport
            ? liveViewport.scrollLeft / Math.max(1e-9, pxPerSec)
            : props.viewportStartSec;
        const viewportEndSec = liveViewport
            ? viewportStartSec + widthPx / Math.max(1e-9, pxPerSec)
            : props.viewportEndSec;
        const scene = buildWaveformScene({
            viewportStartSec,
            viewportEndSec,
            pxPerSec,
            widthPx,
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

    React.useLayoutEffect(() => {
        drawRef.current = draw;
        invalidate();
    }, [draw, invalidate]);

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
