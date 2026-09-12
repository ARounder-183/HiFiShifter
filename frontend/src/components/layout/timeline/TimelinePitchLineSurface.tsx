import React from "react";

import { useAppSelector } from "../../../app/hooks";
import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { clearCanvasPhysical, rasterize } from "./runtime/canvasRaster";
import type { TimelineAxis } from "./runtime/timelineAxis.js";
import { LAYER_ORDER } from "./runtime/timelineFrameCommitter";
import { drawTrackPitchLines } from "./runtime/timelinePitchLineRenderer";

const EMPTY_CLIPS: readonly ClipInfo[] = [];

/**
 * 时间线轨道音高线面（Pitch Reference Clip 原始音高线）。
 *
 * 【主要内容】把窗口内轨道 / clip 数据与 Redux 音高曲线绘制进 sticky 时间线
 * 的一块共享 Canvas2D 画布，并注册为统一帧提交器的 `pitchLine` 图层。
 *
 * 【作用】音高线此前挂在各 TrackLane 内容层内（每轨一块 canvas + 滚动事件内
 * JS transform 补偿）：原生滚动由合成器线程先行提交，主线程的补偿/重绘必然
 * 晚 ≥1 帧到达，音高线相对波形 / Clip 体明显滞后。本组件把音高线迁入与
 * 波形面 / clip 体画布相同的 sticky 架构：元素位置由浏览器合成器保证，
 * 内容在滚动事件内与其它图层同帧、按固定层序同步重绘。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelineSurface` 传入轨道窗口（与 `TimelineWaveformSurface` 同参，
 *   行波形区几何同源：rowHeight − CLIP_BODY_PADDING_Y − CLIP_HEADER_HEIGHT）。
 * - 横向/纵向：坐标一律走 `TimelineAxis`（绘制用内容绝对坐标 + 视口平移，
 *   与 clip 体画布同法）；时间↔像素换算不在本文件出现。
 * - 下游：`runtime/timelinePitchLineRenderer.ts`（行级绘制）。
 */
export const TimelinePitchLineSurface = React.memo(function TimelinePitchLineSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引：行 topPx 使用内容绝对坐标，
     * 竖直滚动时由总线 scrollTopPx 统一平移（与 DOM 内容层同帧提交）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
}) {
    const { tracks, startTrackIndex, clipsByTrackId, rowHeight, widthPx, heightPx } = props;

    // 与旧 TrackLane 内画布相同的回退数据源：后端 clip_pitch_data 推送的
    // per-clip 音高曲线 / 范围。仅低频变化（分析完成、clip 增删），不会在
    // 滚动帧触发重渲染。
    const clipPitchCurves = useAppSelector((s) => s.session.clipPitchCurves);
    const clipPitchRanges = useAppSelector((s) => s.session.clipPitchRanges);

    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const tracksRef = React.useRef(tracks);
    const startTrackIndexRef = React.useRef(startTrackIndex);
    const clipsByTrackIdRef = React.useRef(clipsByTrackId);
    const rowHeightRef = React.useRef(rowHeight);
    const widthRef = React.useRef(widthPx);
    const heightRef = React.useRef(heightPx);
    const clipPitchCurvesRef = React.useRef(clipPitchCurves);
    const clipPitchRangesRef = React.useRef(clipPitchRanges);

    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    tracksRef.current = tracks;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    startTrackIndexRef.current = startTrackIndex;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    clipsByTrackIdRef.current = clipsByTrackId;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    rowHeightRef.current = rowHeight;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    widthRef.current = widthPx;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    heightRef.current = heightPx;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    clipPitchCurvesRef.current = clipPitchCurves;
    // eslint-disable-next-line react-hooks/refs -- render 期写 ref 镜像：命令式绘制回调需在同一提交内读取最新 props（热路径既有模式，见 TimelineCanvasViewport）
    clipPitchRangesRef.current = clipPitchRanges;

    /**
     * 按给定投影重绘音高线。
     *
     * 流程：统一光栅化 → 按 dpr 设置变换 → 内容绝对坐标平移（水平/竖直都由
     * 视口偏移完成）→ 逐行绘制。必须同步完成（与 clip 体/波形同一帧约束）。
     *
     * @param axis 视口投影；省略时取总线当前值（供挂载后首次绘制使用）。
     */
    const invalidate = React.useCallback((axis?: TimelineAxis) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const current = axis ?? timelineViewportBus.getAxis();

        // 统一光栅化契约：与 clip 体画布 / 波形面同一套取整规则。
        const target = rasterize(
            canvas,
            Math.max(1, Math.ceil(widthRef.current)),
            Math.max(1, Math.ceil(heightRef.current)),
            window.devicePixelRatio || 1,
        );

        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
        clearCanvasPhysical(ctx, target);
        ctx.translate(-current.scrollLeftPx, -current.scrollTopPx);

        const clipsById = clipsByTrackIdRef.current;
        const effectiveRowHeight = rowHeightRef.current;
        for (let i = 0; i < tracksRef.current.length; i++) {
            const track = tracksRef.current[i];
            drawTrackPitchLines({
                ctx,
                axis: current,
                clips: clipsById[track.id] ?? EMPTY_CLIPS,
                rowTopPx: (startTrackIndexRef.current + i) * effectiveRowHeight,
                rowHeight: effectiveRowHeight,
                clipPitchCurves: clipPitchCurvesRef.current,
                clipPitchRanges: clipPitchRangesRef.current,
            });
        }
    }, []);

    // 数据/布局参数必须留在依赖里：clip 数据、音高曲线或行窗口变化时视口
    // 并没变，总线不会 emit，只有这条路径能让画布跟上。不会造成滚动帧的
    // 重复绘制——纯滚动帧这些 props 的引用保持稳定（TimelinePanel memo
    // 产物，与 clip 体画布同保证）。
    React.useLayoutEffect(() => {
        invalidate();
    }, [
        clipPitchCurves,
        clipPitchRanges,
        clipsByTrackId,
        heightPx,
        invalidate,
        rowHeight,
        startTrackIndex,
        tracks,
        widthPx,
    ]);

    React.useEffect(() => {
        // 注册到统一帧提交器：与网格 / clip 体 / 波形的绘制顺序固定，且同一
        // 帧内重复的视口提交只触发一次重绘。
        return timelineViewportBus.register(
            { name: "pitch-line", paint: (axis) => invalidate(axis) },
            LAYER_ORDER.pitchLine,
        );
    }, [invalidate]);

    React.useEffect(() => {
        // 浏览器缩放 / 跨屏拖动改变 devicePixelRatio：光栅化依赖 dpr，变化后
        // 必须重绘一次（与 TimelineCanvasViewport 同法）。
        const onResize = () => invalidate();
        window.addEventListener("resize", onResize);
        return () => {
            window.removeEventListener("resize", onResize);
        };
    }, [invalidate]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
            data-hs-pitch-line="1"
        />
    );
});
