/**
 * 时间线 sticky 可视层：网格 / clip 体画布 / 波形 / 播放头的堆叠容器。
 *
 * 【主要内容】按固定层序堆叠背景网格 → clip 体画布 → 波形面 → 网格覆盖层 →
 * 播放头，并在挂载时用视口总线快照同步一次网格。
 *
 * 【作用】这些图层都不随原生滚动移动（sticky），必须共享同一份视口状态才能
 * 与 DOM 内容层同帧；本组件是它们共用的定位与传参入口。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 传入窗口化的轨道数据与 `TimelineAxis`。
 * - 横向：所有子层的位置/缩放参数一律来自 axis；网格的重绘同步走
 *   `gridRedrawBridge`（P2 将并入统一帧提交器）。
 * - 下游：`BackgroundGrid`、`TimelineCanvasViewport`、`TimelineWaveformSurface`。
 */

import React from "react";

import type { ClipInfo, TrackInfo } from "../../../features/session/sessionTypes";
import { timelineViewportBus } from "../../../utils/timelineViewportBus";
import { BackgroundGrid } from "./BackgroundGrid";
import { invokeGridRedrawHandler } from "./gridRedrawBridge";
import { TimelineCanvasViewport } from "./TimelineCanvasViewport";
import { TimelineWaveformSurface } from "./TimelineWaveformSurface";
import { secToViewportPx, type TimelineAxis } from "./runtime/timelineAxis.js";
import { LAYER_ORDER } from "./runtime/timelineFrameCommitter.js";
import type { TimelineTick } from "./runtime/buildTimelineTicks.js";

export const TimelineSurface = React.memo(function TimelineSurface(props: {
    tracks: readonly TrackInfo[];
    /** 窗口首行的绝对轨道索引（供波形面内容绝对坐标使用）。 */
    startTrackIndex: number;
    clipsByTrackId: Readonly<Record<string, readonly ClipInfo[]>>;
    rowHeight: number;
    widthPx: number;
    heightPx: number;
    topPx: number;
    /** 统一坐标投影：全部子层位置与缩放的唯一来源。 */
    axis: TimelineAxis;
    playheadSec: number;
    clipModel: {
        drawClips: import("./runtime/timelineCanvasModel").TimelineCanvasClipModel[];
        activeGroupIds?: Set<string>;
        disabledGroupIds?: string[];
    };
    /** 主题模式：透传给 clip 体画布，切换时同帧重绘配色。 */
    darkMode?: boolean;
    /** Sticky 背景网格参数（与内容层同坐标系，随滚动同步重绘）。 */
    contentWidth: number;
    pxPerBeat: number;
    grid: string;
    beatsPerBar: number;
    gridVisible: boolean;
    gridMinSpacingPx?: number;
    gridSwingPercent?: number;
    /** 统一刻度源：网格线与标尺刻度同源（由 buildTimelineTicks 生成）。 */
    ticks: readonly TimelineTick[];
    /** 网格内容底部边界（内容绝对坐标 y，通常为最后一条轨道底边）。 */
    gridBottomPx: number;
    gridOverlayLayerRef: React.RefObject<HTMLDivElement | null>;
    /** 播放光标竖直参考线：随滚动/缩放与其它 sticky 层在同一帧移动。 */
    playheadLineRef: React.Ref<HTMLDivElement>;
}) {
    const gridBaseProps = {
        contentWidth: props.contentWidth,
        contentHeight: props.heightPx,
        pxPerBeat: props.pxPerBeat,
        grid: props.grid,
        beatsPerBar: props.beatsPerBar,
        viewportWidth: props.widthPx,
        scrollLeft: props.axis.scrollLeftPx,
        visible: props.gridVisible,
        minSpacingPx: props.gridMinSpacingPx,
        swingPercent: props.gridSwingPercent,
        ticks: props.ticks,
        sticky: true,
        viewportTopPx: 0,
        contentBottomPx: props.gridBottomPx,
    } as const;

    // 挂载时按总线快照立即同步一次网格：恢复滚动位置时 React 的 scrollLeft
    // state 可能仍是 rAF 前值，不能让网格与 Clip 体画布在首帧分叉。
    React.useLayoutEffect(() => {
        const axis = timelineViewportBus.getAxis();
        invokeGridRedrawHandler(
            props.gridOverlayLayerRef.current,
            axis.scrollLeftPx,
            axis.scrollTopPx,
        );
    }, [props.gridOverlayLayerRef, props.gridVisible]);

    // 播放头竖线的水平位置必须与网格 / Clip 体画布同一事实源：总线快照
    // （滚动事件内同步提交的原生滚动真值）。props.axis.scrollLeftPx 是量化
    // 提交的 React state（REACT_SCROLL_STEP_PX 死区，可永久滞后最多 255px），
    // 拖拽 Clip 的逐帧重渲染会用该滞后值重写本元素位置，把命令式同步的正确
    // 位置覆盖回错误值——与 BackgroundGrid 的重绘偏移同根（见
    // gridDrawViewport.ts 顶注）。
    const viewportAxis = timelineViewportBus.getAxis();

    /* eslint-disable react-hooks/refs -- playhead 参考线：ref 属性透传 + 样式按渲染期当前 props 计算（同一提交内跟随播放头更新，既有模式） */
    return (
        <div
            className="sticky left-0 top-0 pointer-events-none"
            style={{ width: props.widthPx, zIndex: 1 }}
        >
            {/* 网格只有叠加层这一份：竖线一根到底、画在波形与半透明 clip 体之上。
                旧的"背景层 + 叠加层"双份绘制会在空泳道里把同一根线叠两遍，
                两层亚像素相位稍有出入就出现 3px 的粗线。 */}
            <div
                className="absolute pointer-events-none"
                style={{
                    top: props.topPx,
                    width: props.widthPx,
                    height: props.heightPx,
                }}
            >
                <TimelineCanvasViewport
                    width={props.widthPx}
                    height={props.heightPx}
                    model={props.clipModel}
                    darkMode={props.darkMode}
                    rowGuides={{
                        startTrackIndex: props.startTrackIndex,
                        rowCount: props.tracks.length,
                        rowHeight: props.rowHeight,
                        contentBottomPx: props.gridBottomPx,
                    }}
                />
            </div>
            <div
                className="absolute pointer-events-none"
                style={{
                    top: props.topPx,
                    width: props.widthPx,
                    height: props.heightPx,
                    // 显式高于 clip body 画布：波形必须绘制在色块之上。
                    zIndex: 2,
                }}
            >
                <TimelineWaveformSurface
                    tracks={props.tracks}
                    startTrackIndex={props.startTrackIndex}
                    clipsByTrackId={props.clipsByTrackId}
                    rowHeight={props.rowHeight}
                    widthPx={props.widthPx}
                    heightPx={props.heightPx}
                    axis={props.axis}
                />
            </div>
            <BackgroundGrid
                {...gridBaseProps}
                layerRef={props.gridOverlayLayerRef}
                // 叠加网格：竖线一根到底（不分段、不跳过 header），与背景层
                // 完全同形，只是绘制顺序在波形之上。
                lineOpacity={1}
                viewportBus={timelineViewportBus}
                layerOrder={LAYER_ORDER.gridOverlay}
            />
            <div
                ref={props.playheadLineRef}
                className="absolute w-px bg-qt-playhead z-20 pointer-events-none"
                style={{
                    top: props.topPx,
                    height: props.heightPx,
                    left: secToViewportPx(viewportAxis, Number(props.playheadSec) || 0),
                }}
            />
        </div>
    );
    /* eslint-enable react-hooks/refs */
});
