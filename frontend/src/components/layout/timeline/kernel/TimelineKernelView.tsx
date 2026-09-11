/**
 * 时间轴渲染内核 · React 外壳
 *
 * 【主要内容】
 * 承载新内核的 React 外壳：提供容器 / 画布 / 自绘滚动条 DOM，把 session 数据与
 * 键位绑定以「镜像 ref」形式喂给命令式宿主（`createTimelineKernelHost`），并把
 * 标尺 / 轨道头 / 播放头等**保留为 DOM** 的外部元素交给宿主在 rAF 内同步。
 *
 * 【作用】
 * React 只做三件事：提供 DOM、提供低频数据、接收低频回调（行高 / 缩放 / 场景标脏）；
 * 高频滚动、渲染、DOM 同步全部由宿主在 rAF 内完成，不经 React。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 在开关开启时渲染本组件（替换旧轨道区）。
 * - 下游：`host/timelineKernelHost` 持有全部 GL 与输入资源。
 * - 数据：`useAppSelector` 读 session（tracks / clips / 网格参数 / 主题）与
 *   keybindings（滚轮手势语义）。
 *
 * 【边界（为什么标尺与轨道头不是自绘）】
 * 标尺（含 Tempo Map 行）与左侧轨道头是重交互、低频变化的 DOM 子树：搬进 canvas
 * 等于重写 Tempo 旗帜拖拽、内联编辑、右键菜单与电平表。这里只把「跟随视口」的部分
 * 收敛为每帧一次 transform / scrollTop 写入——成本与图层数无关，且视觉天然与旧实现
 * 一致。轨道区（网格 / clip / 波形 / 交互）才是滚动性能的瓶颈所在，由内核自绘。
 */

import React from "react";

import { useAppSelector } from "../../../../app/hooks";
import { useAppTheme } from "../../../../theme/AppThemeProvider";
import { selectKeybinding } from "../../../../features/keybindings/keybindingsSlice";
import { readDevicePixelRatio, wholeDevicePxLength } from "../../../../utils/devicePixelLine";
import type { ClipInfo } from "../../../../features/session/sessionTypes";
import { createTimelineAxis, type TimelineAxis } from "../runtime/timelineAxis";
import { TimelineWaveformSurface } from "../TimelineWaveformSurface";
import {
    createTimelineKernelHost,
    type TimelineKernelData,
    type TimelineKernelHost,
    type TimelineKernelInteractions,
} from "./host/timelineKernelHost";

export interface TimelineKernelViewProps {
    /** 行高（与左侧轨道头同源；内核不自行维护行高）。 */
    readonly rowHeight: number;
    /** 竖直缩放请求：内核解析手势后回调，由面板写入行高状态。 */
    readonly onRowHeightChange: (rowHeightPx: number) => void;
    /** 初始水平缩放（持久化恢复值），仅用于首次创建滚动内核。 */
    readonly initialPxPerSec: number;
    /** 水平缩放变化（内核为真值源），用于驱动标尺刻度等 React 侧派生量。 */
    readonly onPxPerSecChange: (pxPerSec: number) => void;
    /** 播放头位置读取（工程秒）：取视觉插值后的实时值，避免播放时滞后。 */
    readonly getPlayheadSec: () => number;
    /** 标尺内容层（宿主在 rAF 内写 transform 跟随水平滚动）。 */
    readonly rulerContentRef?: React.MutableRefObject<HTMLElement | null>;
    /** 轨道头滚动容器（宿主在 rAF 内写 scrollTop 跟随纵向滚动）。 */
    readonly trackListScrollerRef?: React.MutableRefObject<HTMLElement | null>;
    /**
     * 标尺播放头竖线（**位于标尺内容层内**）。
     *
     * 宿主写它的 `left`（内容坐标）：标尺内容层已带 translateX(-scrollLeft)，
     * 播放头因此自动跟随滚动；若改写成视口坐标会双重计滚动。
     */
    readonly rulerPlayheadLineRef?: React.MutableRefObject<HTMLElement | null>;
    /** 宿主句柄出口：面板用它把「轨道头滚动」等外部意图转发给内核。 */
    readonly hostRef?: React.MutableRefObject<TimelineKernelHost | null>;
    /**
     * 交互回调（内核只做命中与手势，编辑语义交回面板 / Redux）。
     *
     * 引用须稳定（用 `useCallback`）：内核在创建时取一次，引用抖动不会生效。
     */
    readonly interactions?: TimelineKernelInteractions;
}

export const TimelineKernelView: React.FC<TimelineKernelViewProps> = (props) => {
    const {
        rowHeight,
        onRowHeightChange,
        initialPxPerSec,
        onPxPerSecChange,
        getPlayheadSec,
        rulerContentRef,
        trackListScrollerRef,
        rulerPlayheadLineRef,
        hostRef,
        interactions,
    } = props;

    const containerRef = React.useRef<HTMLDivElement | null>(null);
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const hThumbRef = React.useRef<HTMLDivElement | null>(null);
    const vThumbRef = React.useRef<HTMLDivElement | null>(null);
    const playheadLineRef = React.useRef<HTMLDivElement | null>(null);
    const localHostRef = React.useRef<TimelineKernelHost | null>(null);
    const [fatal, setFatal] = React.useState<string | null>(null);
    /** 宿主是否已创建：波形层依赖内核视口源，必须等宿主就绪后再挂载。 */
    const [hostReady, setHostReady] = React.useState(false);
    /** 可见轨道行窗口（内核低频回调）：波形 scene rows 按行构建。 */
    const [visibleRows, setVisibleRows] = React.useState({ firstRow: 0, rowCount: 0 });
    /** 波形层使用的投影快照（行窗口变化时刷新；滚动由视口源驱动，不经 React）。 */
    const [waveformAxis, setWaveformAxis] = React.useState<TimelineAxis | null>(null);
    const [viewportSize, setViewportSize] = React.useState({ width: 0, height: 0 });

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
    const playheadZoomEnabled = useAppSelector((state) => state.session.playheadZoomEnabled);
    const selectedClipId = useAppSelector((state) => state.session.selectedClipId);
    const multiSelectedClipIds = useAppSelector((state) => state.session.multiSelectedClipIds);
    const horizontalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.horizontalZoom"),
    );
    // 竖直缩放复用 `modifier.pianoRollVerticalZoom`：时间轴与参数编辑器共用
    // 同一个「纵向缩放」语义绑定（旧实现同样如此，见 useTimelineState）。
    const verticalZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.pianoRollVerticalZoom"),
    );
    const scrollHorizontalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollHorizontal"),
    );
    const scrollVerticalKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollVertical"),
    );
    const scrollbarZoomKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.scrollbarZoom"),
    );
    const { mode } = useAppTheme();

    const buildData = (): TimelineKernelData => ({
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
        rowHeight,
        playheadSec: getPlayheadSec(),
        keybindings: {
            horizontalZoom: horizontalZoomKb,
            verticalZoom: verticalZoomKb,
            scrollHorizontal: scrollHorizontalKb,
            scrollVertical: scrollVerticalKb,
            scrollbarZoom: scrollbarZoomKb,
        },
        playheadZoomEnabled,
        initialPxPerSec,
        selectedClipId,
        multiSelectedClipIds,
    });

    // 数据镜像：渲染期写 ref，宿主在 rAF 内读取（避免宿主订阅 React 状态）。
    const dataRef = React.useRef<TimelineKernelData>(buildData());
    // eslint-disable-next-line react-hooks/refs -- 数据镜像：命令式宿主需在 rAF 内读取最新值（既有热路径模式）
    dataRef.current = buildData();

    /**
     * 可见行窗口变化（内核低频回调）：刷新行窗口与波形投影快照。
     *
     * 频率约束：内核只在「行窗口真正变化」时回调（每滚过一行一次），因此这里
     * 的 setState 不会进入滚动热路径。
     */
    const handleVisibleRowsChange = React.useCallback((firstRow: number, rowCount: number) => {
        setVisibleRows((prev) =>
            prev.firstRow === firstRow && prev.rowCount === rowCount
                ? prev
                : { firstRow, rowCount },
        );
        const host = localHostRef.current;
        if (host !== null) setWaveformAxis(host.getAxis());
    }, []);

    // 回调镜像：宿主持有的是稳定函数，函数内部读取最新回调，避免重建宿主。
    const callbacksRef = React.useRef({
        onRowHeightChange,
        onPxPerSecChange,
        getPlayheadSec,
        onVisibleRowsChange: handleVisibleRowsChange,
    });
    // eslint-disable-next-line react-hooks/refs -- 回调镜像：同上
    callbacksRef.current = {
        onRowHeightChange,
        onPxPerSecChange,
        getPlayheadSec,
        onVisibleRowsChange: handleVisibleRowsChange,
    };

    // 交互回调镜像：同上（面板用 useCallback 提供，但引用仍可能在依赖变化时更新）。
    const interactionsRef = React.useRef<TimelineKernelInteractions | undefined>(interactions);
    // eslint-disable-next-line react-hooks/refs -- 回调镜像：同上
    interactionsRef.current = interactions;
    const stableInteractions = React.useMemo<TimelineKernelInteractions>(
        () => ({
            onSeek: (sec, commit) => interactionsRef.current?.onSeek?.(sec, commit),
            onSelectClip: (clipId, additive) =>
                interactionsRef.current?.onSelectClip?.(clipId, additive),
            onDragPreview: (args) => interactionsRef.current?.onDragPreview?.(args),
            onDragCommit: (args) => interactionsRef.current?.onDragCommit?.(args),
            onTrimPreview: (args) => interactionsRef.current?.onTrimPreview?.(args),
            onTrimCommit: (args) => interactionsRef.current?.onTrimCommit?.(args),
            onFadePreview: (args) => interactionsRef.current?.onFadePreview?.(args),
            onFadeCommit: (args) => interactionsRef.current?.onFadeCommit?.(args),
            onBoxSelectPreview: (args) => interactionsRef.current?.onBoxSelectPreview?.(args),
            onBoxSelectCommit: (args) => interactionsRef.current?.onBoxSelectCommit?.(args),
            onContextMenu: (args) => interactionsRef.current?.onContextMenu?.(args),
        }),
        [],
    );

    /**
     * 内核视口源适配器（供波形面订阅）。
     *
     * 引用必须稳定：`WaveformSurface` 在 `props.viewportSource` 变化时会重新注册
     * 图层，引用抖动会导致反复注销 / 注册。
     */
    const kernelViewportSource = React.useMemo(
        () => ({
            getAxis: (): TimelineAxis => {
                const host = localHostRef.current;
                if (host !== null) return host.getAxis();
                // 宿主未就绪：返回占位投影（波形此时尚未挂载，不会被读到）。
                return createTimelineAxis({
                    pxPerSec: 1,
                    scrollLeftPx: 0,
                    scrollTopPx: 0,
                    viewportWidthPx: 1,
                    dpr: 1,
                });
            },
            register: (
                layer: { name: string; paint: (axis: TimelineAxis) => void },
                order: number,
            ) => localHostRef.current?.registerViewportLayer(layer, order) ?? (() => undefined),
        }),
        [],
    );

    /** 波形行数据：只构建可见行窗口内的轨道（每行都会触发 peaks 预加载）。 */
    const waveformTracks = React.useMemo(
        () => tracks.slice(visibleRows.firstRow, visibleRows.firstRow + visibleRows.rowCount),
        [tracks, visibleRows],
    );
    const waveformClipsByTrackId = React.useMemo(() => {
        const map: Record<string, ClipInfo[]> = {};
        for (const track of waveformTracks) map[track.id] = [];
        for (const clip of clips) {
            const list = map[clip.trackId];
            if (list !== undefined) list.push(clip);
        }
        return map;
    }, [waveformTracks, clips]);

    // 视口尺寸：波形画布与内核视口同尺寸（竖直由 axis.scrollTopPx 平移）。
    React.useEffect(() => {
        const container = containerRef.current;
        if (!container) return;
        const measure = () => {
            setViewportSize({ width: container.clientWidth, height: container.clientHeight });
        };
        measure();
        const observer = new ResizeObserver(measure);
        observer.observe(container);
        return () => observer.disconnect();
    }, []);

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
                sync: {
                    rulerContent: rulerContentRef?.current ?? null,
                    trackListScroller: trackListScrollerRef?.current ?? null,
                    playheadLine: playheadLineRef?.current ?? null,
                    rulerPlayheadLine: rulerPlayheadLineRef?.current ?? null,
                },
                onRowHeightChange: (px) => callbacksRef.current.onRowHeightChange(px),
                onZoomChange: (pxPerSec) => callbacksRef.current.onPxPerSecChange(pxPerSec),
                onVisibleRowsChange: (firstRow, rowCount) =>
                    callbacksRef.current.onVisibleRowsChange(firstRow, rowCount),
                interactions: stableInteractions,
            });
        } catch (error) {
            setFatal(error instanceof Error ? error.message : String(error));
            return;
        }
        localHostRef.current = host;
        if (hostRef !== undefined) hostRef.current = host;
        // 宿主就绪后再挂载波形层：它需要内核视口源（见 kernelViewportSource）。
        setWaveformAxis(host.getAxis());
        setHostReady(true);
        return () => {
            host?.dispose();
            localHostRef.current = null;
            if (hostRef !== undefined) hostRef.current = null;
            setHostReady(false);
        };
        // 只在挂载时创建：宿主是长生命周期运行时对象，数据经 dataRef 流入。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // 低频数据变化 → 场景重建（几何 / 主题 / 网格参数 / 行高）。
    React.useEffect(() => {
        localHostRef.current?.invalidateScene();
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
        rowHeight,
    ]);

    return (
        <div
            ref={containerRef}
            tabIndex={0}
            data-hs-timeline-kernel="1"
            className="relative flex-1 overflow-hidden bg-qt-graph-bg outline-none"
        >
            <canvas ref={canvasRef} className="pointer-events-none absolute inset-0" />
            {/* 波形层：独立 WebGL2 画布，由内核视口源驱动（滚动帧只更新 uniform）。
                位于 clip 块面之上、细节层之下（细节层由宿主创建，z-index 2）。 */}
            {hostReady && waveformAxis !== null && visibleRows.rowCount > 0 ? (
                <div className="pointer-events-none absolute inset-0 z-[1]">
                    <TimelineWaveformSurface
                        tracks={waveformTracks}
                        startTrackIndex={visibleRows.firstRow}
                        clipsByTrackId={waveformClipsByTrackId}
                        rowHeight={rowHeight}
                        widthPx={viewportSize.width}
                        heightPx={viewportSize.height}
                        axis={waveformAxis}
                        viewportSource={kernelViewportSource}
                    />
                </div>
            ) : null}
            {/* 轨道区播放头：位置由宿主在 rAF 内写 translateX（视口坐标）；
                线宽按物理像素取整，与旧实现同为恒 1 物理像素。 */}
            <div
                ref={playheadLineRef}
                className="pointer-events-none absolute top-0 bottom-0 z-20 bg-qt-playhead"
                style={{
                    left: 0,
                    width: wholeDevicePxLength(1, readDevicePixelRatio()),
                }}
            />
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
