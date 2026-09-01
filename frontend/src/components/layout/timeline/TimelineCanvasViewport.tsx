import React from "react";

import { drawTimelineCanvas } from "./runtime/timelineCanvasRenderer";
import { resolveFontFamily } from "./runtime/timelineCanvasStyle";
import { clearCanvasPhysical, rasterize } from "./runtime/canvasRaster";
import type { TimelineCanvasClipModel } from "./runtime/timelineCanvasModel";
import type { TimelineAxis } from "./runtime/timelineAxis";
import { LAYER_ORDER } from "./runtime/timelineFrameCommitter";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import {
    GlClipBodyRenderer,
    isGlClipBodiesEnabled,
    PERF_GL_CLIP_BODIES_KEY,
} from "./runtime/timelineClipGlRenderer.js";

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

    /**
     * GL 块面渲染器（dev 开关，默认关闭）。
     *
     * 开启后 clip 的块面画在**下方**的 GL canvas 上，本组件原有的 2D canvas
     * 只画细节层（旋钮 / 徽标 / 文字 / 淡变 / 吸附三角）并保持在其**上方**，
     * 因此块面先画、细节后画的顺序天然成立。
     */
    const glCanvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const glBodiesRef = React.useRef<GlClipBodyRenderer | null>(null);
    const [glBodiesAvailable, setGlBodiesAvailable] = React.useState(() => isGlClipBodiesEnabled());
    // 用于让"开关变化"立刻生效：localStorage 变化不会触发 React，这里用
    // 一个自增计数驱动重渲染。
    const [glToggleEpoch, setGlToggleEpoch] = React.useState(0);

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
            glBodies: glBodiesRef.current,
            originXPx: current.scrollLeftPx,
            originYPx: current.scrollTopPx,
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

        // dev 开关：PERF 面板切换 localStorage 后派发自定义事件，这里即时
        // 建拆 GL 渲染器。重建后实例缓冲为空，必须重绘一次。
        const onToggle = () => {
            setGlBodiesAvailable(isGlClipBodiesEnabled());
            setGlToggleEpoch((epoch) => epoch + 1);
        };
        window.addEventListener(PERF_GL_CLIP_BODIES_KEY, onToggle);

        return () => {
            window.removeEventListener("resize", onResize);
            window.removeEventListener(PERF_GL_CLIP_BODIES_KEY, onToggle);
        };
    }, [invalidate]);

    // GL 渲染器的生命周期与开关绑定（放独立的 effect，避免与绘制逻辑耦合）。
    React.useEffect(() => {
        const canvas = glCanvasRef.current;
        if (!glBodiesAvailable || !canvas) {
            glBodiesRef.current = null;
            invalidate();
            return;
        }
        try {
            glBodiesRef.current = new GlClipBodyRenderer(canvas);
        } catch (error) {
            // WebGL2 不可用 / 着色器编译失败：静默退回 Canvas2D。
            console.warn("[TimelineCanvasViewport] GL clip bodies unavailable", error);
            glBodiesRef.current = null;
            setGlBodiesAvailable(false);
        }
        invalidate();
        return () => {
            glBodiesRef.current?.dispose();
            glBodiesRef.current = null;
        };
        // glToggleEpoch 只用于触发重建。
    }, [glBodiesAvailable, glToggleEpoch, invalidate]);

    return (
        <>
            {glBodiesAvailable ? (
                <canvas
                    ref={glCanvasRef}
                    className="absolute inset-0 pointer-events-none"
                    data-hs-clip-gl="1"
                />
            ) : null}
            <canvas ref={canvasRef} className="absolute inset-0 pointer-events-none" />
        </>
    );
};
