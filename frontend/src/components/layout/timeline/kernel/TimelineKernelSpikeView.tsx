/**
 * 时间轴渲染内核 · Spike 宿主视图（React 外壳）
 *
 * 【主要内容】
 * 承载新内核的 React 外壳：提供容器 / 画布 / 滚动条 DOM 节点，把 session 数据以
 * 「镜像 ref」形式喂给命令式宿主（`createTimelineKernelHost`），并在卸载时释放。
 *
 * 【作用】
 * React 只做两件事：提供 DOM 与低频数据；高频滚动 / 渲染全部由宿主在 rAF 内完成，
 * 不经 React。这是设计文档「React 管外壳、runtime 管高频」的最小体现，也是
 * 「不用 sync 机制」的验证载体——宿主不依赖任何随原生滚动移动的 DOM 内容层。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在开关开启时渲染本组件（覆盖时间轴区域）。
 * - 下游：`host/timelineKernelHost` 持有全部 GL 与输入资源。
 * - 数据：`useAppSelector` 读 session（tracks / clips / 网格与标尺参数 / 主题）。
 */

import React from "react";

import { useAppSelector } from "../../../../app/hooks";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import {
    createTimelineKernelHost,
    type TimelineKernelData,
    type TimelineKernelHost,
} from "./host/timelineKernelHost";

export const TimelineKernelSpikeView: React.FC = () => {
    const containerRef = React.useRef<HTMLDivElement | null>(null);
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const hThumbRef = React.useRef<HTMLDivElement | null>(null);
    const vThumbRef = React.useRef<HTMLDivElement | null>(null);
    const [fatal, setFatal] = React.useState<string | null>(null);

    const tracks = useAppSelector((state) => state.session.tracks);
    const clips = useAppSelector((state) => state.session.clips);
    const projectSec = useAppSelector((state) => state.session.projectSec);
    const bpm = useAppSelector((state) => state.session.bpm);
    const beatsPerBar = useAppSelector((state) => state.session.beats);
    const grid = useAppSelector((state) => state.session.grid);
    const primaryTimeUnit = useAppSelector((state) => state.session.primaryTimeUnit);
    const secondaryTimeUnit = useAppSelector((state) => state.session.secondaryTimeUnit);
    const minLabelSpacingPx = useAppSelector((state) => state.session.rulerLabelSpacingPx);
    const tempoMap = useAppSelector((state) => state.session.tempoMap);
    const { mode } = useAppTheme();

    // 数据镜像：渲染期写 ref，宿主在 rAF 内读取（避免宿主订阅 React 状态）。
    const dataRef = React.useRef<TimelineKernelData>({
        tracks,
        clips,
        projectSec,
        darkMode: mode === "dark",
        bpm,
        beatsPerBar,
        grid,
        primaryTimeUnit,
        secondaryTimeUnit,
        minLabelSpacingPx,
        tempoMap,
    });
    // eslint-disable-next-line react-hooks/refs -- 数据镜像：命令式宿主需在 rAF 内读取最新值（既有热路径模式）
    dataRef.current = {
        tracks,
        clips,
        projectSec,
        darkMode: mode === "dark",
        bpm,
        beatsPerBar,
        grid,
        primaryTimeUnit,
        secondaryTimeUnit,
        minLabelSpacingPx,
        tempoMap,
    };

    const hostRef = React.useRef<TimelineKernelHost | null>(null);

    // 挂载：创建宿主（GL 初始化失败时展示回退提示而不是崩溃）。
    React.useEffect(() => {
        const container = containerRef.current;
        const canvas = canvasRef.current;
        const hThumb = hThumbRef.current;
        const vThumb = vThumbRef.current;
        if (!container || !canvas || !hThumb || !vThumb) return;
        let host: TimelineKernelHost | null = null;
        try {
            host = createTimelineKernelHost({
                container,
                canvas,
                hScrollbarThumb: hThumb,
                vScrollbarThumb: vThumb,
                data: () => dataRef.current,
            });
        } catch (error) {
            setFatal(error instanceof Error ? error.message : String(error));
            return;
        }
        hostRef.current = host;
        return () => {
            host?.dispose();
            hostRef.current = null;
        };
    }, []);

    // 低频数据变化 → 场景重建（几何 / 主题 / 网格参数）。
    React.useEffect(() => {
        hostRef.current?.invalidateScene();
    }, [
        tracks,
        clips,
        projectSec,
        bpm,
        beatsPerBar,
        grid,
        primaryTimeUnit,
        secondaryTimeUnit,
        minLabelSpacingPx,
        tempoMap,
        mode,
    ]);

    return (
        <div
            ref={containerRef}
            tabIndex={0}
            data-hs-timeline-kernel="1"
            className="absolute inset-0 z-30 overflow-hidden bg-qt-graph-bg outline-none"
        >
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
            {/* 自绘滚动条：thumb 的几何由宿主每帧写入（见 updateScrollbars）。 */}
            <div className="absolute right-0 top-0 bottom-0 w-2 bg-black/5">
                <div ref={vThumbRef} className="absolute left-0 w-full rounded bg-black/30" />
            </div>
            <div className="absolute bottom-0 left-0 right-0 h-2 bg-black/5">
                <div ref={hThumbRef} className="absolute top-0 h-full rounded bg-black/30" />
            </div>
            {fatal !== null ? (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-red-500">
                    {`时间轴内核不可用：${fatal}`}
                </div>
            ) : null}
        </div>
    );
};
