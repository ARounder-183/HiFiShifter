import React from "react";

import { drawTimelineCanvas } from "./runtime/timelineCanvasRenderer";
import { resolveFontFamily } from "./runtime/timelineCanvasStyle";
import type { TimelineCanvasClipModel } from "./runtime/timelineCanvasModel";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";

export const TimelineCanvasViewport: React.FC<{
    width: number;
    height: number;
    model: {
        drawClips: TimelineCanvasClipModel[];
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    };
}> = ({ width, height, model }) => {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const widthRef = React.useRef(width);
    const heightRef = React.useRef(height);
    const modelRef = React.useRef(model);
    const viewportRef = React.useRef(timelineViewportBus.getSnapshot());

    widthRef.current = width;
    heightRef.current = height;
    modelRef.current = model;
    const invalidate = React.useCallback(() => {
        // 同步绘制：滚动事件在绘制前触发，本画布必须与原生滚动的 DOM 内容层
        // 在同一帧内提交位移。任何 rAF 延迟都会让 sticky 画布与 DOM 层分离。
        const canvas = canvasRef.current;
        if (!canvas) return;

        const displayWidth = Math.max(1, Math.ceil(widthRef.current));
        const displayHeight = Math.max(1, Math.ceil(heightRef.current));
        const dpr = window.devicePixelRatio || 1;
        const internalWidth = Math.max(1, Math.floor(displayWidth * dpr));
        const internalHeight = Math.max(1, Math.floor(displayHeight * dpr));

        if (canvas.width !== internalWidth) canvas.width = internalWidth;
        if (canvas.height !== internalHeight) canvas.height = internalHeight;
        canvas.style.width = `${displayWidth}px`;
        canvas.style.height = `${displayHeight}px`;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, displayWidth, displayHeight);
        // 画布内容使用内容绝对坐标（与 DOM 内容层同一坐标系），
        // 由视口偏移统一做水平/竖直平移——两个轴都随滚动同帧提交。
        ctx.translate(-viewportRef.current.scrollLeft, -viewportRef.current.scrollTopPx);
        drawTimelineCanvas(ctx, {
            width: displayWidth,
            height: displayHeight,
            clips: modelRef.current.drawClips,
            fontFamily: resolveFontFamily(),
            activeGroupIds: modelRef.current.activeGroupIds,
            disabledGroupIds: modelRef.current.disabledGroupIds,
        });
    }, []);

    React.useLayoutEffect(() => {
        invalidate();
    }, [height, invalidate, model, width]);

    React.useEffect(() => {
        return timelineViewportBus.subscribe(
            (scrollLeft, pxPerSec, viewportWidth, scrollTopPx, rowHeight) => {
                viewportRef.current = {
                    scrollLeft,
                    pxPerSec,
                    viewportWidth,
                    scrollTopPx,
                    rowHeight,
                    revision: 0,
                };
                invalidate();
            },
        );
    }, [invalidate]);

    return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
};
