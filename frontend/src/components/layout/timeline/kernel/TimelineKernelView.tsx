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
import { SnapHighlightLayer } from "../SnapHighlightLayer";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../constants";
import { KernelClipInlineEditor } from "./KernelClipInlineEditor";
import { createTimelineAxis, type TimelineAxis } from "../../renderKernel/timelineAxis";
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
    /**
     * 水平滚动位置的量化提交（每 256px 一次）。
     *
     * 标尺的**刻度范围**由 React 按 `scrollLeft` 计算（`timelineTicks`），内核只写
     * 标尺内容层的 transform 会让刻度停留在初始视口——滚动后刻度消失。
     */
    readonly onScrollLeftCommit?: (scrollLeftPx: number) => void;
    /**
     * 视口宽度回写（尺寸变化时一次）。
     *
     * 标尺的刻度窗口按 `[scrollLeft, scrollLeft + viewportWidth]` 计算；内核模式下
     * 旧的滚动容器不存在，其 ResizeObserver 不会触发——不回写时窗口宽度停在初始值，
     * 标尺只显示得出前面一段刻度。
     */
    readonly onViewportWidthChange?: (widthPx: number) => void;
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
    /**
     * 激活 / 禁用的分组 id（header 控件命中需要）。
     *
     * 分组状态会改变链徽标的**可见性**，而可见性决定后续徽标的 x 偏移——不传会让
     * 命中区与绘制区错位（表现为「点静音点到了链」）。由面板传入：面板是分组语义
     * 的唯一持有者（激活集合由选中态派生）。
     */
    readonly activeGroupIds?: readonly string[];
    readonly disabledGroupIds?: readonly string[];
    /**
     * 静音检测预览区段（`session.silencePreviewSegments`）：clip id → 工程秒区间。
     *
     * 由面板传入（它是检测状态的唯一持有者）；内核只负责画成半透明红色覆盖层。
     */
    readonly silenceSegmentsByClipId?: Readonly<
        Record<string, ReadonlyArray<readonly [number, number]>>
    >;
    /**
     * 是否平铺显示全部 Take（`session.showAllTakes`）。
     *
     * 影响多 Take 的 lane 命中（分界线绘制由波形面与模型各自消费同一份设置）；
     * 缺省 true（与 session 默认值一致）。
     */
    readonly showAllTakes?: boolean;
    /**
     * 内核态行内编辑（重命名 / 增益 / 速率）。
     *
     * 由面板提供：面板持有编辑状态与提交语义（值的格式化与解析是领域知识——
     * 增益是 dB、速率有 `x` / `%` 前缀，不属于输入框）。视图只负责把它定位到
     * clip header 上，并在滚动时随视口更新位置。
     */
    readonly inlineEdit?: {
        readonly clipId: string;
        readonly field: "name" | "gain" | "rate";
        readonly initialValue: string;
        readonly inputMode?: "text" | "decimal";
        readonly onCommit: (value: string) => void;
        readonly onCancel: () => void;
    } | null;
    /**
     * 吸附高亮层的上下文（缺省不渲染该层）。
     *
     * 层内元素用**内容坐标**布局（与旧实现一致：吸附竖线是 `marker.sec × pxPerSec`），
     * 由宿主整层平移跟随视口（见 `TimelineKernelDomSync.snapHighlightContent`）。
     * 旧实现靠原生滚动平移它，内核自绘滚动后必须显式补上这一层平移，否则
     * 吸附高亮要么不显示（挂在旧分支里），要么位置不随滚动。
     *
     * `pxPerSec` / 行高 / 轨道列表都取自 React 侧（与面板同源）：拖拽期间缩放与
     * 行高是低频操作，不需要内核在 rAF 内写。
     */
    readonly snapHighlight?: {
        /** 当前水平缩放（CSS px/秒）。 */
        readonly pxPerSec: number;
        /** 内容层宽度（含右侧虚拟延伸，与旧实现同源）。 */
        readonly contentWidth: number;
        /** 内容层高度（全部轨道总高）。 */
        readonly contentHeight: number;
    };
    /**
     * 素材拖入预览（缺省不渲染；拖入期间由面板给出）。
     *
     * 几何为**内容坐标**（与旧实现一致：`startSec × pxPerSec`），由宿主整层平移
     * 跟随视口。纵向位置按 `rowHeight` 换算（面板没有行高）。
     */
    readonly dropPreview?: {
        /** 内容坐标左缘。 */
        readonly leftPx: number;
        /** 内容坐标宽度（`durationSec × pxPerSec`）。 */
        readonly widthPx: number;
        readonly trackId: string;
        readonly fileName: string;
        readonly contentWidth: number;
        readonly contentHeight: number;
    };
    /**
     * 素材拖入（由面板提供）。
     *
     * 【为什么挂在内核容器上】旧实现的 `onDragOver` / `onDrop` 挂在
     * `TimelineScrollArea` 上，而内核模式下该组件不挂载——拖入会走浏览器默认
     * 行为（打开文件）。处理器内部用 `e.currentTarget` 的 bounds 算落点，因此
     * **必须挂在真正承载时间轴的容器上**，不能挪到含标尺的外层（会整体偏移一个
     * 标尺高度，落点算到错误轨道）。
     */
    readonly onDragOver?: React.DragEventHandler<HTMLDivElement>;
    readonly onDrop?: React.DragEventHandler<HTMLDivElement>;
    /**
     * 拖到**全部轨道之下**（新建轨道）时的幽灵行（缺省不渲染）。
     *
     * 旧实现把它画在最后一条轨道正下方（`clipDropNewTrack` 分支）：一行虚线框 +
     * 待落库 clip 的半透明预览。行位置按 `tracks.length * rowHeight` 换算，因此
     * 这里只需要内容坐标的左缘 / 宽度。
     */
    readonly newTrackDrop?: {
        /** 待落库 clip 的内容坐标几何。 */
        readonly items: readonly {
            readonly key: string;
            readonly leftPx: number;
            readonly widthPx: number;
        }[];
        /** 内容层宽度（用于横向裁剪，与其它内容层同源）。 */
        readonly contentWidth: number;
    } | null;
    /**
     * copy 拖拽的 ghost 预览（缺省不渲染该层）。
     *
     * 【为什么内核需要一个独立图层】移动语义下"乐观位置"就够——内核重建几何时
     * 把 clip 画到新位置即可。但 copy 模式下**原 clip 不动**，没有乐观位置可依赖，
     * 必须由面板给出 ghost。
     *
     * 【坐标系】与 `snapHighlight` 同一机制：层内元素用**内容坐标**布局，宿主在
     * rAF 内整层 `translate(-scrollLeft, -scrollTop)`（见 `TimelineKernelDomSync.ghostContent`）。
     * 背景色由面板算好传入（`color-mix` 的轨道色归一化在面板侧，内核不重复该逻辑）。
     */
    readonly ghost?: {
        readonly items: readonly {
            /** React key。 */
            readonly key: string;
            /** 内容坐标左缘（= 目标起始时间 × pxPerSec）。 */
            readonly leftPx: number;
            /** 内容坐标宽度（= clip 长度 × pxPerSec）。 */
            readonly widthPx: number;
            /** 落点轨道：纵向位置由视图按 `rowHeight` 换算（面板没有行高）。 */
            readonly trackId: string;
            /**
             * 已算好的 CSS 背景（header 条 / body 区）。
             *
             * 由面板给出而不是视图算：轨道色归一化（`normalizedTrackColorCss`）
             * 需要 `darkMode`，那是面板的渲染模式状态；视图重复这套配色逻辑
             * 迟早与真实 clip 分叉。
             */
            readonly headerBackground: string;
            readonly bodyBackground: string;
        }[];
        readonly contentWidth: number;
        readonly contentHeight: number;
    };
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
        snapHighlight,
        ghost,
        dropPreview,
        onDragOver,
        onDrop,
        newTrackDrop,
        inlineEdit,
        activeGroupIds,
        disabledGroupIds,
        silenceSegmentsByClipId,
        showAllTakes,
        onScrollLeftCommit,
        onViewportWidthChange,
    } = props;

    const containerRef = React.useRef<HTMLDivElement | null>(null);
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const hThumbRef = React.useRef<HTMLDivElement | null>(null);
    const vThumbRef = React.useRef<HTMLDivElement | null>(null);
    /** 滚动条轨道容器：承接「点击空白翻页」（原生滚动条的等效交互）。 */
    const hTrackRef = React.useRef<HTMLDivElement | null>(null);
    const vTrackRef = React.useRef<HTMLDivElement | null>(null);
    const playheadLineRef = React.useRef<HTMLDivElement | null>(null);
    /** 吸附高亮内容层容器（宿主在 rAF 内整层平移，跟随视口）。 */
    const snapContentRef = React.useRef<HTMLDivElement | null>(null);
    /** copy ghost 内容层容器（同上）。 */
    const ghostContentRef = React.useRef<HTMLDivElement | null>(null);
    /** 拖入预览内容层容器（同上）。 */
    const dropPreviewContentRef = React.useRef<HTMLDivElement | null>(null);
    /** 新建轨道幽灵行的内容层（同上：常驻容器 + 整层平移）。 */
    const newTrackDropContentRef = React.useRef<HTMLDivElement | null>(null);
    /** 行内编辑浮层根元素（位置与宽度都由宿主在 rAF 内写入）。 */
    const inlineEditorRef = React.useRef<HTMLDivElement | null>(null);
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
    // 淡变形状循环：修饰键 + 单击包络线切换形状（拖动仍是改长度）。
    const fadeShapeCycleKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.fadeShapeCycleClick"),
    );
    // 点击选择的两个修饰键：与旧实现（`TrackLane` / `ClipItem`）同一套绑定。
    // 写死 Ctrl/Shift 会让用户改绑后两种渲染模式的行为分叉。
    const clipMultiSelectToggleKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipMultiSelectToggle"),
    );
    const clipRangeSelectKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.clipRangeSelect"),
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
            fadeShapeCycle: fadeShapeCycleKb,
            clipMultiSelectToggle: clipMultiSelectToggleKb,
            clipRangeSelect: clipRangeSelectKb,
        },
        playheadZoomEnabled,
        initialPxPerSec,
        selectedClipId,
        multiSelectedClipIds,
        activeGroupIds: activeGroupIds ?? [],
        disabledGroupIds: disabledGroupIds ?? [],
        silenceSegmentsByClipId,
        showAllTakes: showAllTakes ?? true,
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
        onScrollLeftCommit,
        onViewportWidthChange,
    });
    // eslint-disable-next-line react-hooks/refs -- 回调镜像：同上
    callbacksRef.current = {
        onRowHeightChange,
        onPxPerSecChange,
        getPlayheadSec,
        onVisibleRowsChange: handleVisibleRowsChange,
        onScrollLeftCommit,
        onViewportWidthChange,
    };

    // 交互回调镜像：同上（面板用 useCallback 提供，但引用仍可能在依赖变化时更新）。
    const interactionsRef = React.useRef<TimelineKernelInteractions | undefined>(interactions);
    // eslint-disable-next-line react-hooks/refs -- 回调镜像：同上
    interactionsRef.current = interactions;
    /**
     * 稳定引用的交互回调集合（内核创建时只取一次，引用抖动不生效）。
     *
     * ⚠️ **这里是手工维护的转发清单**：`TimelineKernelInteractions` 上新增的每一个
     * 回调都必须同步加进来，否则宿主调用的是 `undefined`——表现为「功能完全没反应，
     * 但也没有任何报错」（已在 `onSelectClip` 新增参数与 `onFadeHover` 上各踩一次）。
     * 新增回调时请连同参数一并转发。
     */
    const stableInteractions = React.useMemo<TimelineKernelInteractions>(
        () => ({
            onSeek: (sec, commit, trackId) =>
                interactionsRef.current?.onSeek?.(sec, commit, trackId),
            onSelectClip: (clipId, additive, rangeSelect, clientX) =>
                interactionsRef.current?.onSelectClip?.(clipId, additive, rangeSelect, clientX),
            onDoubleClickClip: (clipId) => interactionsRef.current?.onDoubleClickClip?.(clipId),
            onToggleClipMute: (clipId, nextMuted) =>
                interactionsRef.current?.onToggleClipMute?.(clipId, nextMuted),
            onOpenClipFormant: (clipId, screenX, screenY) =>
                interactionsRef.current?.onOpenClipFormant?.(clipId, screenX, screenY),
            onToggleGroupDisabled: (groupId) =>
                interactionsRef.current?.onToggleGroupDisabled?.(groupId),
            onBadgeEditStart: (clipId, field, screenX, screenY) =>
                interactionsRef.current?.onBadgeEditStart?.(clipId, field, screenX, screenY),
            onClipPointerDownIntercept: (args) =>
                interactionsRef.current?.onClipPointerDownIntercept?.(args) ?? false,
            onGainDragPreview: (args) => interactionsRef.current?.onGainDragPreview?.(args),
            onGainDragCommit: (args) => interactionsRef.current?.onGainDragCommit?.(args),
            onGainReset: (clipId) => interactionsRef.current?.onGainReset?.(clipId),
            onRateBadgeMenu: (clipId, screenX, screenY) =>
                interactionsRef.current?.onRateBadgeMenu?.(clipId, screenX, screenY),
            onRenameClipStart: (clipId, screenX, screenY) =>
                interactionsRef.current?.onRenameClipStart?.(clipId, screenX, screenY),
            onCrossfadeGripPreview: (args) =>
                interactionsRef.current?.onCrossfadeGripPreview?.(args),
            onCrossfadeGripCommit: (args) => interactionsRef.current?.onCrossfadeGripCommit?.(args),
            onSnapOffsetPreview: (args) => interactionsRef.current?.onSnapOffsetPreview?.(args),
            onSnapOffsetCommit: (args) => interactionsRef.current?.onSnapOffsetCommit?.(args),
            onFadeShapeCycle: (clipId, side) =>
                interactionsRef.current?.onFadeShapeCycle?.(clipId, side),
            onCrossfadeCycle: (sides) => interactionsRef.current?.onCrossfadeCycle?.(sides),
            onResetFadeCurvature: (sides) => interactionsRef.current?.onResetFadeCurvature?.(sides),
            onDragPreview: (args) => interactionsRef.current?.onDragPreview?.(args),
            onDragCommit: (args) => interactionsRef.current?.onDragCommit?.(args),
            onTrimPreview: (args) => interactionsRef.current?.onTrimPreview?.(args),
            onTrimCommit: (args) => interactionsRef.current?.onTrimCommit?.(args),
            onFadePreview: (args) => interactionsRef.current?.onFadePreview?.(args),
            onFadeCommit: (args) => interactionsRef.current?.onFadeCommit?.(args),
            onBoxSelectPreview: (args) => interactionsRef.current?.onBoxSelectPreview?.(args),
            onBoxSelectCommit: (args) => interactionsRef.current?.onBoxSelectCommit?.(args),
            onContextMenu: (args) => interactionsRef.current?.onContextMenu?.(args),
            onFadeContextMenu: (request) => interactionsRef.current?.onFadeContextMenu?.(request),
            onFadeHover: (args, clientX, clientY) =>
                interactionsRef.current?.onFadeHover?.(args, clientX, clientY),
            onActivateTake: (clipId, takeId, sec) =>
                interactionsRef.current?.onActivateTake?.(clipId, takeId, sec),
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
                hScrollbarTrack: hTrackRef.current ?? undefined,
                vScrollbarTrack: vTrackRef.current ?? undefined,
                data: () => dataRef.current,
                sync: {
                    rulerContent: rulerContentRef?.current ?? null,
                    trackListScroller: trackListScrollerRef?.current ?? null,
                    playheadLine: playheadLineRef?.current ?? null,
                    rulerPlayheadLine: rulerPlayheadLineRef?.current ?? null,
                    snapHighlightContent: snapContentRef.current,
                    ghostContent: ghostContentRef.current,
                    dropPreviewContent: dropPreviewContentRef.current,
                    newTrackDropContent: newTrackDropContentRef.current,
                },
                onRowHeightChange: (px) => callbacksRef.current.onRowHeightChange(px),
                onZoomChange: (pxPerSec) => callbacksRef.current.onPxPerSecChange(pxPerSec),
                onVisibleRowsChange: (firstRow, rowCount) =>
                    callbacksRef.current.onVisibleRowsChange(firstRow, rowCount),
                interactions: stableInteractions,
                onScrollLeftCommit: (px) => callbacksRef.current.onScrollLeftCommit?.(px),
                onViewportWidthChange: (px) => callbacksRef.current.onViewportWidthChange?.(px),
            });
        } catch (error) {
            setFatal(error instanceof Error ? error.message : String(error));
            return;
        }
        localHostRef.current = host;
        if (hostRef !== undefined) hostRef.current = host;
        // dev-only 调试出口：浏览器里读取内核视口真值（缩放 / 滚动 / 投影），
        // 用于核对「标尺（React 派生量）与网格（内核真值）是否一致」这类问题。
        // 生产构建不挂载（`import.meta.env.DEV` 为 false 时整段被裁剪）。
        if (import.meta.env.DEV) {
            (window as unknown as { __hfsKernel?: TimelineKernelHost }).__hfsKernel = host;
        }
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

    /**
     * 行内编辑浮层：定位到 clip header 上，并随视口更新位置。
     *
     * 位置由宿主在 rAF 内命令式写入（复用视口图层注册）——用 React state 每帧
     * 更新会让输入框在滚动时抖动，甚至因重渲染丢失焦点与已输入内容。
     */
    React.useEffect(() => {
        if (inlineEdit == null) return;
        const host = localHostRef.current;
        if (host == null) return;
        const anchor = host.getClipHeaderAnchor(inlineEdit.clipId, inlineEdit.field);
        if (anchor == null) return;
        const reposition = (axis: TimelineAxis): void => {
            const el = inlineEditorRef.current;
            if (el == null) return;
            // 宽度与位置一起写：把宽度留给 React state 会让「挂载」与「拿到宽度」
            // 分成两次渲染，切换编辑目标时出现"状态已更新但浮层没出现"的空档。
            el.style.width = `${anchor.widthPx}px`;
            el.style.left = `${anchor.contentLeftPx - axis.scrollLeftPx}px`;
            el.style.top = `${anchor.contentTopPx - axis.scrollTopPx}px`;
        };
        // 首帧立即写一次：浮层刚挂载，不能等下一次视口提交才定位。
        reposition(host.getAxis());
        const unregister = host.registerViewportLayer(
            { name: "inline-editor", paint: reposition },
            100,
        );
        return unregister;
    }, [inlineEdit]);

    // 低频数据变化 → 场景重建（几何 / 主题 / 网格参数 / 行高 / **选中态**）。
    //
    // ⚠️ 这个依赖数组是「场景重建」的**唯一触发器**：宿主的绘制走脏标记
    // （`sceneDirty` + renderLoop 的 `dirty`），数据变了但没进这里，画面就**不会
    // 重绘**——表现为「状态明明变了，却看不到任何变化」（点击 clip 选不中的根因）。
    //
    // 因此凡是**影响 clip 外观**的数据都必须列在这里。已知会影响外观的：
    // - `selectedClipId` / `multiSelectedClipIds`：选中描边（白 2px）
    // - `activeGroupIds` / `disabledGroupIds`：编组激活的金色描边
    // - `silenceSegmentsByClipId`：静音检测预览的红色覆盖层
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
        selectedClipId,
        multiSelectedClipIds,
        activeGroupIds,
        disabledGroupIds,
        // 静音检测预览：改变细节层的红色覆盖层（检测对话框实时写入）。
        silenceSegmentsByClipId,
        // 多 Take 平铺开关：改变 lane 分界线。
        showAllTakes,
    ]);

    return (
        <div
            ref={containerRef}
            tabIndex={0}
            data-hs-timeline-kernel="1"
            /* `data-timeline-scroller`：旧实现挂在原生滚动容器上，被
               `measureTimelineViewportOffsetPx()`（参数编辑器同步的左右偏移测量）与
               参数编辑器的 ResizeObserver 当作「时间轴轨道区视口元素」查询。
               内核模式下该容器不存在，不回填这个标记会让同步偏移恒为 0
               （两个面板的网格线无法按同一屏幕位置对齐）。此处语义相同：
               它是轨道区的视口元素（只是滚动由内核自绘而非浏览器维护）。 */
            data-timeline-scroller
            className="relative flex-1 overflow-hidden bg-qt-graph-bg outline-none"
            onDragOver={onDragOver}
            onDrop={onDrop}
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
            {/* 吸附高亮层：内容坐标系（层内元素用 `marker.sec × pxPerSec` 布局），
                由宿主整层 translate 跟随视口。旧实现靠原生滚动平移它，内核自绘
                滚动后必须显式补上，否则该层要么不挂载、要么位置不随滚动。
                注意：transform 会创建 stacking context，因此容器需显式 z-index
                （层内部的 z-[13] 只在容器内生效）。 */}
            {snapHighlight !== undefined ? (
                <div
                    ref={snapContentRef}
                    data-hs-snap-content="1"
                    className="pointer-events-none absolute left-0 top-0 z-[3] overflow-hidden"
                    style={{
                        width: snapHighlight.contentWidth,
                        height: snapHighlight.contentHeight,
                    }}
                >
                    <SnapHighlightLayer
                        pxPerSec={snapHighlight.pxPerSec}
                        rowHeight={rowHeight}
                        tracks={tracks}
                        contentHeight={snapHighlight.contentHeight}
                    />
                </div>
            ) : null}
            {/* copy 拖拽 ghost：内容坐标系，由宿主整层 translate 跟随视口。
                移动语义下内核靠"乐观位置"显示，copy 语义下原 clip 不动，
                必须靠这一层——否则 ⌘+拖拽期间画面毫无反馈。 */}
            {/* 容器**常驻**（无 ghost 时为空）：宿主在挂载时一次性抓取 ref 快照，
                按需挂载会让它在需要平移时仍是 null——表现为"ghost 不随滚动走"。 */}
            <div
                ref={ghostContentRef}
                data-hs-ghost-content="1"
                className="pointer-events-none absolute left-0 top-0 z-[4] overflow-hidden"
                style={{
                    width: ghost?.contentWidth ?? 0,
                    height: ghost?.contentHeight ?? 0,
                }}
            >
                {(ghost?.items ?? []).map((item) => {
                    const trackIndex = tracks.findIndex((t) => t.id === item.trackId);
                    if (trackIndex < 0) return null;
                    return (
                        <div
                            key={item.key}
                            className="absolute opacity-50"
                            style={{
                                left: item.leftPx,
                                top: trackIndex * rowHeight,
                                width: item.widthPx,
                                height: Math.max(1, rowHeight - CLIP_BODY_PADDING_Y),
                            }}
                        >
                            <div
                                className="absolute left-0 right-0 top-0"
                                style={{
                                    height: CLIP_HEADER_HEIGHT,
                                    backgroundColor: item.headerBackground,
                                }}
                            />
                            <div
                                className="absolute left-0 right-0 bottom-0 border border-dashed border-black/40"
                                style={{
                                    top: CLIP_HEADER_HEIGHT,
                                    backgroundColor: item.bodyBackground,
                                }}
                            />
                        </div>
                    );
                })}
            </div>
            {/* 素材拖入预览：与 ghost 同一机制（内容坐标 + 宿主整层平移）。
                容器同样**常驻**——原因见上。 */}
            <div
                ref={dropPreviewContentRef}
                data-hs-drop-preview="1"
                className="pointer-events-none absolute left-0 top-0 z-[5] overflow-hidden"
                style={{
                    width: dropPreview?.contentWidth ?? 0,
                    height: dropPreview?.contentHeight ?? 0,
                }}
            >
                {dropPreview === undefined
                    ? null
                    : (() => {
                          const trackIndex = tracks.findIndex((t) => t.id === dropPreview.trackId);
                          if (trackIndex < 0) return null;
                          return (
                              <div
                                  className="absolute flex items-center overflow-hidden rounded border border-dashed border-qt-accent/70 bg-qt-accent/15 px-1"
                                  style={{
                                      left: dropPreview.leftPx,
                                      top: trackIndex * rowHeight + 8,
                                      width: Math.max(1, dropPreview.widthPx),
                                      height: Math.max(1, rowHeight - 16),
                                  }}
                              >
                                  <span className="truncate text-[10px] text-qt-text">
                                      {dropPreview.fileName}
                                  </span>
                              </div>
                          );
                      })()}
            </div>
            {/* 新建轨道幽灵行：拖到全部轨道之下时出现在最后一行正下方。
                **内容坐标系**（`startSec × pxPerSec`），由宿主整层平移跟随视口——
                与 ghost / dropPreview 同一机制。行位置用 `tracks.length * rowHeight`
                换算（哨兵轨不在 tracks 里，因此不能靠 findIndex 定位）。 */}
            <div
                ref={newTrackDropContentRef}
                data-hs-new-track-content="1"
                className="pointer-events-none absolute left-0 top-0 z-[6] overflow-hidden"
                style={{
                    width: newTrackDrop?.contentWidth ?? 0,
                    // 高度按"全部轨道 + 幽灵行"算：幽灵行在最后一行之下。
                    height: (tracks.length + 1) * rowHeight,
                }}
            >
                {newTrackDrop == null ? null : (
                    <div
                        className="absolute left-0"
                        style={{
                            top: tracks.length * rowHeight,
                            // 虚线框横跨整条轨道（与旧实现 `left-0 right-0` 一致）；
                            // 滚动时由父层整层平移。
                            width: newTrackDrop.contentWidth,
                            height: rowHeight,
                        }}
                    >
                        <div
                            className="absolute inset-0"
                            style={{
                                border: "1px dashed var(--qt-highlight)",
                                backgroundColor:
                                    "color-mix(in oklab, var(--qt-highlight) 10%, transparent)",
                            }}
                        />
                        {newTrackDrop.items.map((item) => (
                            <div
                                key={item.key}
                                className="absolute opacity-60"
                                style={{
                                    left: Math.max(0, item.leftPx),
                                    width: Math.max(1, item.widthPx),
                                    top: 0,
                                    height: Math.max(1, rowHeight - 8),
                                    paddingTop: 8,
                                }}
                            >
                                <div
                                    className="absolute left-0 right-0 top-0 rounded-t-sm"
                                    style={{
                                        height: CLIP_HEADER_HEIGHT,
                                        backgroundColor:
                                            "color-mix(in oklab, var(--qt-highlight) 55%, transparent)",
                                    }}
                                />
                                <div
                                    className="absolute left-0 right-0 bottom-0 rounded-sm border border-dashed border-white/70"
                                    style={{
                                        top: CLIP_HEADER_HEIGHT,
                                        backgroundColor:
                                            "color-mix(in oklab, var(--qt-highlight) 20%, transparent)",
                                    }}
                                />
                            </div>
                        ))}
                    </div>
                )}
            </div>
            {/* 行内编辑浮层（重命名 / 增益 / 速率）：旧实现由 ClipHeader 的 DOM
                输入框承担，内核模式下 clip 是自绘的，必须自备输入框。位置由宿主
                在 rAF 内写入（内容坐标 − 滚动量），避免滚动时抖动或丢焦点。 */}
            {inlineEdit != null ? (
                <KernelClipInlineEditor
                    ref={inlineEditorRef}
                    key={`${inlineEdit.clipId}:${inlineEdit.field}`}
                    initialValue={inlineEdit.initialValue}
                    inputMode={inlineEdit.inputMode}
                    onCommit={inlineEdit.onCommit}
                    onCancel={inlineEdit.onCancel}
                />
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
            {/* 自绘滚动条：thumb 的几何由宿主每帧写入（见 updateScrollbars）。
                样式对齐旧实现（`.custom-scrollbar` 上的原生滚动条）：
                - thumb 取 `--qt-scrollbar-thumb`（浅色 #b4bac7 / 深色 #555555），
                  而不是固定的半透明黑——后者在浅色主题下几乎看不见；
                - 轨道**透明**（旧实现是 `scrollbar-color: … transparent`），
                  加底色会在时间轴上多出一条灰带；
                - 8px 宽 + 胶囊圆角，对应 macOS 的 overlay thin 滚动条
                  （旧实现的原生滚动条不占布局，这里用绝对定位叠加，行为等价）。
                - 外层即**轨道**：承接「点空白翻页」（原生滚动条的等效交互）。
                  宿主的 thumb 处理器 `stopPropagation`，因此到达轨道的按下必然
                  不在 thumb 上，无需二次命中判定。 */}
            <div ref={vTrackRef} className="absolute right-0 top-0 bottom-0 w-2">
                <div
                    ref={vThumbRef}
                    className="absolute left-0 w-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                />
            </div>
            <div ref={hTrackRef} className="absolute bottom-0 left-0 right-0 h-2">
                <div
                    ref={hThumbRef}
                    className="absolute top-0 h-full rounded-full bg-[var(--qt-scrollbar-thumb)]"
                />
            </div>
            {fatal !== null ? (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-red-500">
                    {`时间轴内核不可用：${fatal}`}
                </div>
            ) : null}
        </div>
    );
};
