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
    const rafRef = React.useRef<number | null>(null);
    const widthRef = React.useRef(width);
    const heightRef = React.useRef(height);
    const modelRef = React.useRef(model);
    const viewportRef = React.useRef(timelineViewportBus.getSnapshot());

    widthRef.current = width;
    heightRef.current = height;
    modelRef.current = model;
    const invalidate = React.useCallback(() => {
        if (rafRef.current != null) return;
        rafRef.current = requestAnimationFrame(() => {
            rafRef.current = null;
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
            ctx.translate(-viewportRef.current.scrollLeft, 0);
            drawTimelineCanvas(ctx, {
                width: displayWidth,
                height: displayHeight,
                clips: modelRef.current.drawClips,
                fontFamily: resolveFontFamily(),
                activeGroupIds: modelRef.current.activeGroupIds,
                disabledGroupIds: modelRef.current.disabledGroupIds,
            });
        });
    }, []);

    React.useLayoutEffect(() => {
        invalidate();
    }, [height, invalidate, model, width]);

    React.useEffect(() => {
        return timelineViewportBus.subscribe((scrollLeft, pxPerSec, viewportWidth) => {
            viewportRef.current = { scrollLeft, pxPerSec, viewportWidth, revision: 0 };
        });
    }, []);

    React.useEffect(() => {
        return () => {
            if (rafRef.current != null) {
                cancelAnimationFrame(rafRef.current);
                rafRef.current = null;
            }
        };
    }, []);

    return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
};
