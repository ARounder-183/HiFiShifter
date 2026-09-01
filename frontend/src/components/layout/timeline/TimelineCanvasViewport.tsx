import React from "react";

import { drawTimelineCanvas } from "./runtime/timelineCanvasRenderer";
import { resolveFontFamily } from "./runtime/timelineCanvasStyle";
import { clearCanvasPhysical, rasterize } from "./runtime/canvasRaster";
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
    /** 主题模式：切换时驱动画布当帧按新主题重绘 clip 配色。 */
    darkMode?: boolean;
}> = ({ width, height, model, rowGuides, darkMode }) => {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const widthRef = React.useRef(width);
    const heightRef = React.useRef(height);
    const modelRef = React.useRef(model);
    const rowGuidesRef = React.useRef(rowGuides);
    const darkModeRef = React.useRef(darkMode);

    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    widthRef.current = width;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    heightRef.current = height;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    modelRef.current = model;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    rowGuidesRef.current = rowGuides;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制/事件回调需在同一提交内读取最新值（热路径既有模式）
    darkModeRef.current = darkMode;
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
        // 全物理清屏：round 向上取整时 CSS 尺寸清屏会在底部遗留残影。
        clearCanvasPhysical(ctx, target);
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
            darkMode: darkModeRef.current,
        });
    }, []);

    // `model` 必须留在依赖里：clip 数据变化（拖拽 / trim / 选中）时视口并没
    // 变，总线不会 emit，只有这条路径能让画布跟上。
    // 它不会造成滚动帧的重复绘制——`drawClips` 全部投影在内容坐标系上，配合
    // `TimelinePanel` 的内容轴与稳定 `visibleTrackClipsById`，其**引用**在纯
    // 滚动帧保持不变（见 TimelinePanel 的 contentAxis 注释）。
    React.useLayoutEffect(() => {
        invalidate();
    }, [darkMode, height, invalidate, model, width]);

    React.useEffect(() => {
        // 注册到统一帧提交器：与网格 / 波形的绘制顺序固定，且同一帧内重复的
        // 视口提交只触发一次重绘。
        return timelineViewportBus.register(
            { name: "clip-body", paint: (axis) => invalidate(axis) },
            LAYER_ORDER.clipBody,
        );
    }, [invalidate]);

    React.useEffect(() => {
        // 浏览器缩放 / 跨屏拖动改变 devicePixelRatio：光栅化与设备像素吸附
        // 都依赖 dpr，变化后必须重绘一次。
        const onResize = () => invalidate();
        window.addEventListener("resize", onResize);
        return () => window.removeEventListener("resize", onResize);
    }, [invalidate]);

    return <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />;
};
