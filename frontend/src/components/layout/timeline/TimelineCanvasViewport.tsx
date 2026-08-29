import React from "react";

import { drawTimelineCanvas } from "./runtime/timelineCanvasRenderer";
import { resolveFontFamily } from "./runtime/timelineCanvasStyle";
import { rasterize } from "./runtime/canvasRaster";
import type { TimelineCanvasClipModel } from "./runtime/timelineCanvasModel";
import type { TimelineAxis } from "./runtime/timelineAxis";
import { LAYER_ORDER } from "./runtime/timelineFrameCommitter";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";

export const TimelineCanvasViewport: React.FC<{
    width: number;
    height: number;
    model: {
        drawClips: TimelineCanvasClipModel[];
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    };
    /** 轨道横向分界线的可见行窗口；用于让分界线延伸到工程末尾之后。 */
    rowGuides?: {
        startTrackIndex: number;
        rowCount: number;
        rowHeight: number;
        /** 轨道内容底部边界，与网格相同；分界线只画到该边界。 */
        contentBottomPx?: number;
    };
}> = ({ width, height, model, rowGuides }) => {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const widthRef = React.useRef(width);
    const heightRef = React.useRef(height);
    const modelRef = React.useRef(model);
    const rowGuidesRef = React.useRef(rowGuides);

    widthRef.current = width;
    heightRef.current = height;
    modelRef.current = model;
    rowGuidesRef.current = rowGuides;
    /**
     * 按给定投影重绘 clip 体。
     *
     * 流程：统一光栅化 → 按 dpr 设置变换 → 用视口偏移平移内容坐标系 → 绘制。
     *
     * @param axis 视口投影；省略时取总线当前值（供挂载后首次绘制使用）。
     */
    const invalidate = React.useCallback((axis?: TimelineAxis) => {
        // 同步绘制：滚动事件在绘制前触发，本画布必须与原生滚动的 DOM 内容层
        // 在同一帧内提交位移。任何 rAF 延迟都会让 sticky 画布与 DOM 层分离。
        const canvas = canvasRef.current;
        if (!canvas) return;
        const current = axis ?? timelineViewportBus.getAxis();

        // 统一光栅化契约：与波形面共用同一套取整规则，否则两者在半像素 DPR
        // 下会差一整个物理像素。
        const target = rasterize(
            canvas,
            Math.max(1, Math.ceil(widthRef.current)),
            Math.max(1, Math.ceil(heightRef.current)),
            window.devicePixelRatio || 1,
        );

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
        ctx.clearRect(0, 0, target.cssWidthPx, target.cssHeightPx);
        // 画布内容使用内容绝对坐标（与 DOM 内容层同一坐标系），
        // 由视口偏移统一做水平/竖直平移——两个轴都随滚动同帧提交。
        ctx.translate(-current.scrollLeftPx, -current.scrollTopPx);
        drawTimelineCanvas(ctx, {
            width: target.cssWidthPx,
            height: target.cssHeightPx,
            clips: modelRef.current.drawClips,
            fontFamily: resolveFontFamily(),
            activeGroupIds: modelRef.current.activeGroupIds,
            disabledGroupIds: modelRef.current.disabledGroupIds,
            rowGuides: rowGuidesRef.current,
            viewportLeft: current.scrollLeftPx,
            viewportTopPx: current.scrollTopPx,
        });
    }, []);

    React.useLayoutEffect(() => {
        invalidate();
    }, [height, invalidate, model, width]);

    React.useEffect(() => {
        // 注册到统一帧提交器：与网格 / 波形的绘制顺序固定，且同一帧内重复的
        // 视口提交只触发一次重绘。
        return timelineViewportBus.register(
            { name: "clip-body", paint: (axis) => invalidate(axis) },
            LAYER_ORDER.clipBody,
        );
    }, [invalidate]);

    return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
};
