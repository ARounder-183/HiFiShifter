/**
 * 时间轴渲染内核 · Spike 宿主（命令式运行时）
 *
 * 【主要内容】
 * 把内核各模块装配成一个可运行的宿主：创建 WebGL2 上下文、滚动内核与渲染循环；
 * 绑定 wheel / 中键平移 / 滚动条拖拽 / 键盘输入；在 rAF 内按层序绘制
 * 「网格 → 轨道行分界线 → clip 块面」（一次 draw call）并同步细节层、
 * 播放头、标尺与轨道头；在内容 / 缩放 / 行高变化时重建几何。
 *
 * 【作用】
 * 这是「自绘滚动 + 单 WebGL2」的运行时载体：
 * - 没有任何随原生滚动移动的 DOM **内容**层 → 不需要同帧同步提交，渲染完全由 rAF 调度；
 * - 块面几何以内容坐标常驻 GPU → 滚动帧只更新 `u_viewOrigin`（`repaint`，零重建）；
 * - clip 细节（旋钮 / 徽标 / 标签 / 淡变曲线 / 吸附三角）由 Canvas2D 覆盖层承担，
 *   同样采用「内容坐标 + 窗口平移」策略，滚动在窗口余量内只移动画布、不重绘。
 *
 * 【与其他模块的关系】
 * - 上游：React 外壳（`TimelineKernelView`）提供 DOM 节点与数据镜像。
 * - 复用：`runtime/timelineAxis`（坐标投影）、`runtime/buildTimelineTicks`（刻度）、
 *   `runtime/timelineCanvasModel`（clip 几何）、`runtime/timelineCanvasStyle`（样式 / 字体）、
 *   `runtime/timelineCanvasRenderer`（细节层绘制）、`runtime/timelineScrollRange`（缩放解析）。
 * - 独立性：纯 TS（无 React）；GL 资源、事件监听与细节画布在 `dispose` 中全部释放。
 *
 * 【坐标系约定（评审检查项）】
 * - 块面实例与细节层绘制**都用内容坐标**；前者靠 `u_viewOrigin` uniform 平移，
 *   后者靠画布元素的 `left/top` 平移，两者共用同一个窗口原点，滚动时不会分离。
 * - 标尺播放头位于标尺内容层内（用内容坐标 `left`）；轨道区播放头位于内核视口
 *   容器内（用视口坐标 `translateX`）。混用会导致双重计滚动或粘屏。
 *
 * 【尚未接入】
 * 波形层与命中测试（交互）仍在旧实现中；标尺与左侧轨道头保留 DOM，由本宿主
 * 在 rAF 内同步（见 `syncDom`）。
 */

import type { ClipInfo, TrackInfo } from "../../../../../features/session/sessionTypes";
import {
    isModifierActive,
    isNoneBinding,
} from "../../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../../features/keybindings/types";
import { getTimelineWheelAction, type ScrollbarZone } from "../../../wheelGesture";
import { buildTimelineTicks, type TimelineTick } from "../../runtime/buildTimelineTicks";
import { createTimelineAxis, type TimelineAxis } from "../../runtime/timelineAxis";
import { buildSparseClipRenderModel } from "../../runtime/timelineCanvasModel";
import { drawTimelineCanvas } from "../../runtime/timelineCanvasRenderer";
import { clearCanvasPhysical, rasterize } from "../../runtime/canvasRaster";
import {
    parseRgbaColor,
    type GlClipBodySink,
} from "../../runtime/timelineClipGlRenderer";
import { resolveFontFamily, resolveThemeColor } from "../../runtime/timelineCanvasStyle";
import { resolveHorizontalWheelZoom } from "../../runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../../runtime/timelineZoomBounds";
import {
    CLIP_HEADER_HEIGHT,
    DEFAULT_PX_PER_SEC,
    DEFAULT_ROW_HEIGHT,
    MAX_PX_PER_SEC,
    MAX_ROW_HEIGHT,
    MIN_PX_PER_SEC,
    MIN_ROW_HEIGHT,
} from "../../constants";
import { hitTest, type ClipHitRegion, type HitTestClip } from "../interaction/hitTest";
import {
    resolveDragDelta,
    resolveFadeDrag,
    resolveTargetTrackIndex,
    resolveTrimEdge,
    type FadeSide,
    type TrimEdge,
} from "../interaction/dragGeometry";
import {
    clipIntersectsBox,
    resolveBoxBounds,
    type BoxBounds,
} from "../interaction/boxSelection";
import { createGlCanvas } from "../gl/glContext";
import { CLIP_INSTANCE_FLOATS, writeFlatInstance } from "../gl/instanceLayout";
import { createSdfBoxProgram } from "../gl/sdfBoxProgram";
import { computeScrollbar, scrollDeltaFromThumbDrag } from "../input/scrollbars";
import { normalizeWheelDelta } from "../input/normalizeWheel";
import { createRenderLoop } from "../renderLoop";
import { createClipInstanceBuilder } from "../scene/clipInstances";
import { buildGridInstances } from "../scene/gridInstances";
import type { FlatInstance, Rgba } from "../scene/instanceTypes";
import { createScrollKernel, type TimelineViewportState } from "../scrollKernel";

/** `buildTimelineTicks` 的入参类型（用于让数据镜像的字段类型自动对齐）。 */
type BuildTicksArgs = Parameters<typeof buildTimelineTicks>[0];

/** 宿主渲染所需的数据镜像（由 React 外壳提供，每次 render 更新）。 */
export interface TimelineKernelData {
    readonly tracks: readonly TrackInfo[];
    readonly clips: readonly ClipInfo[];
    /** 工程总时长（秒）。 */
    readonly projectSec: number;
    readonly darkMode: boolean;
    readonly bpm: number;
    readonly beatsPerBar: number;
    readonly grid: BuildTicksArgs["grid"];
    readonly primaryTimeUnit: BuildTicksArgs["primaryUnit"];
    readonly secondaryTimeUnit: BuildTicksArgs["secondaryUnit"];
    readonly minLabelSpacingPx: number;
    readonly tempoMap: BuildTicksArgs["tempoMap"];
    /**
     * 单条轨道高度（CSS px）。
     *
     * 必须与左侧轨道头同源（`constants.ts` 的 rowHeight 状态）：内核独立取值会让
     * 左右两列的行错位——轨道头第 N 行与轨道区第 N 行不在同一水平线上。
     */
    readonly rowHeight: number;
    /** 播放头位置（工程秒），由内核在 rAF 内读取。 */
    readonly playheadSec: number;
    /**
     * 滚轮手势相关的键位绑定（与旧实现同一套 keybinding 体系）。
     *
     * 滚轮语义完全由绑定决定，不能写死修饰键：时间轴默认「无修饰键滚轮 =
     * 水平缩放」（`modifier.horizontalZoom` 是 none-binding，在无修饰键时命中）、
     * Shift = 水平滚动、Alt = 垂直滚动；悬停自绘滚动条时滚轮只作用于该轴，
     * 按住 `modifier.scrollbarZoom`（默认 Alt）改为该轴缩放。用户可在
     * 键位设置里改绑（触摸板 / REAPER / Vegas 预设各不相同），写死会让
     * 「滚轮缩放 / 滚动」在非默认预设下全部失效。
     */
    readonly keybindings: {
        readonly horizontalZoom: Keybinding | null;
        readonly verticalZoom: Keybinding | null;
        readonly scrollHorizontal: Keybinding | null;
        readonly scrollVertical: Keybinding | null;
        readonly scrollbarZoom: Keybinding | null;
    };
    /** 水平缩放是否以播放头为锚点（`playheadZoomEnabled`）。 */
    readonly playheadZoomEnabled: boolean;
    /** 初始水平缩放（CSS px/秒）：仅用于首次创建滚动内核。 */
    readonly initialPxPerSec: number;
    /** 单选焦点 clip（选中样式与旧实现同源，由样式模块消费）。 */
    readonly selectedClipId: string | null;
    /** 多选集合（与 `selectedClipId` 一起决定描边 / 高亮）。 */
    readonly multiSelectedClipIds: readonly string[];
}

/**
 * 与内核视口联动的外部 DOM（全部在 rAF 内命令式更新，**不经 React**）。
 *
 * 【为什么是 DOM 而不是自绘】
 * 标尺（含 Tempo Map 行）与左侧轨道头是重交互、低频变化的 DOM 子树：把它们搬进
 * canvas 等于重写 Tempo 旗帜拖拽、内联编辑、右键菜单与电平表，收益极低。这里只把
 * 它们**跟随内核视口**的部分收敛为「每帧一次 transform / scrollTop 写入」——一次
 * 样式写入的成本与图层数量无关，不构成滚动瓶颈，同时天然与旧实现视觉一致。
 */
export interface TimelineKernelDomSync {
    /** 标尺内容层：写 `translateX(-scrollLeft)` 跟随水平滚动。 */
    readonly rulerContent?: HTMLElement | null;
    /** 轨道头滚动容器：写 `scrollTop` 跟随纵向滚动。 */
    readonly trackListScroller?: HTMLElement | null;
    /** 轨道区播放头竖线：写 `translateX` 跟随视口与播放位置。 */
    readonly playheadLine?: HTMLElement | null;
    /** 标尺播放头竖线（与轨道区播放头同步）。 */
    readonly rulerPlayheadLine?: HTMLElement | null;
}

/** 宿主构造参数。 */
export interface TimelineKernelHostArgs {
    /** 宿主容器（尺寸来源，`ResizeObserver` 目标）。 */
    readonly container: HTMLElement;
    /** 绘制画布（WebGL2）。 */
    readonly canvas: HTMLCanvasElement;
    /** 水平滚动条 thumb 元素。 */
    readonly hScrollbarThumb: HTMLElement;
    /** 竖直滚动条 thumb 元素。 */
    readonly vScrollbarThumb: HTMLElement;
    /** 数据镜像读取函数（每帧调用，避免 React 与 runtime 相互持有）。 */
    readonly data: () => TimelineKernelData;
    /** 需要跟随视口的外部 DOM（标尺 / 轨道头 / 播放头）。 */
    readonly sync?: TimelineKernelDomSync;
    /**
     * 播放头位置读取（工程秒）。
     *
     * 用函数注入而不是值：播放头在播放中逐帧变化，值注入会迫使 React 每帧重渲染；
     * 内核在 rAF 内直接读视觉位置（既有实现同样用视觉插值 ref 驱动）。
     */
    readonly playheadSec?: () => number;
    /**
     * 竖直缩放请求：内核算出新行高后回调，由 React 写入行高状态。
     *
     * 行高的真值源在 React（左侧轨道头与内核必须同源），内核只负责手势解析与
     * 锚点换算，不能自行改行高——否则两侧会各持一个行高并立刻错位。
     */
    readonly onRowHeightChange?: (rowHeightPx: number) => void;
    /**
     * 水平缩放变化：内核为**真值源**，回调用于驱动 React 侧的派生量
     * （标尺刻度 / 内容宽度）。缩放是低频手势（每个滚轮事件一次），
     * 回调频率不会构成渲染压力。
     */
    readonly onZoomChange?: (pxPerSec: number) => void;
    /**
     * 可见轨道行窗口变化（低频：每滚过一行触发一次）。
     *
     * 供需要按「可见行」构建数据的 React 侧图层使用（例如波形面的 scene rows：
     * 每行都要投影一次并触发 peaks 预加载，全量构建代价与轨道数成正比）。
     * 竖直滚动在同一行窗口内不会触发。
     */
    readonly onVisibleRowsChange?: (firstRow: number, rowCount: number) => void;
    /**
     * 水平滚动位置的**量化**提交（每跨过 `SCROLL_COMMIT_STEP_PX` 一次）。
     *
     * 为什么需要它：标尺的刻度由 React 按当前视口范围计算（`timelineTicks` 依赖
     * `scrollLeft` / `pxPerSec`），内核只写标尺内容层的 transform 是不够的——
     * 不更新 state 时标尺的**刻度范围**始终停留在初始视口，滚动后刻度消失。
     * 量化提交（而不是每帧）保证 React 不进滚动热路径，与旧实现的约定一致。
     */
    readonly onScrollLeftCommit?: (scrollLeftPx: number) => void;
    /**
     * 视口宽度变化（尺寸变化时一次）。
     *
     * 与 `onScrollLeftCommit` 同因：标尺的刻度范围由 React 按
     * `[scrollLeft, scrollLeft + viewportWidth]` 计算。旧实现的宽度来自滚动容器的
     * ResizeObserver，内核模式下那个容器不存在——不回写时窗口宽度停在初始值
     * （表现为标尺只显示得出前面一段刻度）。
     */
    readonly onViewportWidthChange?: (widthPx: number) => void;
    /**
     * 交互回调：内核只做**命中与手势**，编辑语义（Redux action / 后端 thunk）
     * 一律交回 React 侧，避免 runtime 直接依赖 store。
     */
    readonly interactions?: TimelineKernelInteractions;
}

/** 内核交互回调集合。 */
export interface TimelineKernelInteractions {
    /**
     * 请求跳转播放头（点击或拖拽空白 / 标尺）。
     *
     * @param sec 目标时间（秒，已钳制到 >= 0）。
     * @param commit true = 单击或手势结束（应提交后端）；false = 拖拽中的预览。
     */
    readonly onSeek?: (sec: number, commit: boolean) => void;
    /**
     * 选中 clip。
     *
     * @param clipId 目标 clip；null 表示清空选择（点击空白时不会触发——空白
     *   点击按旧实现语义只 seek，不清空选择）。
     * @param additive true = 按住多选修饰键（Ctrl / ⌘），应切换而非替换选择。
     */
    readonly onSelectClip?: (clipId: string, additive: boolean) => void;
    /**
     * 拖拽预览（拖拽期间每帧回调，**不经 React 渲染**）。
     *
     * 语义：调用方据此写入**乐观位置**（Redux）。内核不做独立的 ghost 图层——
     * 乐观更新会改变 `clips` 引用，内核在下一帧重建几何时把 clip 画在新位置，
     * 视觉上即"跟着指针走"。这样只有一份位置真值，不会出现 ghost 与实体分叉。
     *
     * @param args 目标位置（已含边界钳制）；`targetTrackId` 为落点轨道。
     */
    readonly onDragPreview?: (args: {
        readonly clipId: string;
        readonly deltaSec: number;
        readonly targetTrackId: string;
    }) => void;
    /**
     * 拖拽结束。
     *
     * @param args 最终位置；`cancelled = true`（Esc / pointercancel / 卸载）
     *   表示调用方应回滚到按下时的位置，而不是提交。
     */
    readonly onDragCommit?: (args: {
        readonly clipId: string;
        readonly deltaSec: number;
        readonly targetTrackId: string;
        readonly cancelled: boolean;
    }) => void;
    /**
     * trim 预览（拖拽左右边缘时每帧回调，已按值去重）。
     *
     * @param args 新的起始时间与长度（已钳制）；`deltaSec` 是实际生效的变化量
     *   （左边缘 = `startSec` 的变化、右边缘 = `lengthSec` 的变化）。
     */
    readonly onTrimPreview?: (args: {
        readonly clipId: string;
        readonly edge: TrimEdge;
        readonly startSec: number;
        readonly lengthSec: number;
        readonly deltaSec: number;
    }) => void;
    /**
     * trim 结束。
     *
     * @param args 最终几何；`cancelled = true` 时调用方应回滚。
     */
    readonly onTrimCommit?: (args: {
        readonly clipId: string;
        readonly edge: TrimEdge;
        readonly startSec: number;
        readonly lengthSec: number;
        readonly cancelled: boolean;
    }) => void;
    /**
     * 淡变角预览（拖拽角部时每帧回调，已按值去重）。
     *
     * @param args 新的淡变长度（已钳制到 `[0, clip 长度]`）。
     */
    readonly onFadePreview?: (args: {
        readonly clipId: string;
        readonly side: FadeSide;
        readonly fadeSec: number;
        readonly deltaSec: number;
    }) => void;
    /**
     * 淡变角结束。
     *
     * @param args 最终长度；`cancelled = true` 时调用方应回滚。
     */
    readonly onFadeCommit?: (args: {
        readonly clipId: string;
        readonly side: FadeSide;
        readonly fadeSec: number;
        readonly cancelled: boolean;
    }) => void;
    /**
     * 框选预览（右键拖拽期间，命中集合变化时才回调）。
     *
     * @param args 框内的 clip id 列表与多选修饰键状态；合并语义（是否保留原选择）
     *   由调用方复用既有 `computeTimelineRectSelection` 决定。
     */
    readonly onBoxSelectPreview?: (args: {
        readonly clipIds: readonly string[];
        readonly additive: boolean;
    }) => void;
    /** 框选结束（`cancelled = true` 时调用方应恢复拖动前的选择）。 */
    readonly onBoxSelectCommit?: (args: {
        readonly clipIds: readonly string[];
        readonly additive: boolean;
        readonly cancelled: boolean;
    }) => void;
    /**
     * 右键菜单请求（未发生框选拖拽的右键单击）。
     *
     * 内核模式下旧的滚动容器不渲染，其 `onContextMenu` 完全不触发——菜单需要由
     * 内核提供**命中信息**，调用方复用既有分支（clip 菜单 / 轨道区菜单）。
     *
     * @param args 指针位置与命中结果；`clipIds` 是指针处该轨道上的**全部** clip
     *   （按 startSec 升序，可能重叠），调用方据此决定是否提供"重叠选择"入口。
     */
    readonly onContextMenu?: (args: {
        readonly clientX: number;
        readonly clientY: number;
        readonly clipIds: readonly string[];
        readonly trackId: string | null;
        /** 指针处的工程时间（秒）：轨道区菜单的"在此处新建"等操作需要它。 */
        readonly sec: number;
    }) => void;
}

/** 宿主句柄。 */
export interface TimelineKernelHost {
    /** 标记场景需要重建（内容 / 缩放 / 主题变化后调用）。 */
    invalidateScene(): void;
    /**
     * 外部请求纵向滚动（左侧轨道头滚动时调用）。
     *
     * 特殊说明：写入后由内核统一钳制并标脏，外部不得自行计算上限
     * （钳制只在 `ScrollKernel` 内做一次）。
     */
    setScrollTop(px: number): void;
    /** 读取当前视口状态（供外部低频读取，例如保存 / 调试）。 */
    getViewport(): { scrollLeft: number; scrollTop: number; pxPerSec: number };
    /**
     * 读取当前坐标投影（内容坐标 ↔ 视口坐标）。
     *
     * 供需要与内核视口同步的**独立画布**图层使用（例如波形面：它自带 WebGL2
     * 上下文与几何缓存，只需要一个与内核同源的 axis 即可零重建跟随滚动）。
     */
    getAxis(): TimelineAxis;
    /**
     * 注册独立画布图层：内核在每次视口提交（滚动 / 缩放 / 重建）后按 order 调用其 paint。
     *
     * 特殊说明：只在**视口变化或几何重建**时调用，播放头移动这类纯 DOM 更新不触发
     * —— 图层自身的复用判定（如波形面的 `canReuse`）会决定是重绘还是只改 uniform。
     *
     * @param layer 图层（name 用于去重，paint 接收当前 axis）。
     * @param order 绘制顺序（小的先画）。
     * @returns 注销函数。
     */
    registerViewportLayer(
        layer: { name: string; paint: (axis: TimelineAxis) => void },
        order: number,
    ): () => void;
    /** 释放全部资源与事件监听。 */
    dispose(): void;
}

/** 横向构建余量（CSS px）：滚动在此范围内不重建几何。 */
const HORIZONTAL_MARGIN_PX = 512;

/** 竖直 overscan 行数：轨道窗口上下各多建若干行。 */
const VERTICAL_OVERSCAN_ROWS = 2;

/** 自绘滚动条厚度（CSS px），与视图中的 thumb 容器一致。 */
const SCROLLBAR_SIZE_PX = 8;

/**
 * 水平滚动向 React 量化提交的步长（CSS px）。
 *
 * 与旧实现的 `REACT_SCROLL_STEP_PX` 取同一量级：步长越小，标尺越跟手，但 React
 * 重渲染越频繁；256px 在"标尺刻度不会明显滞后"与"滚动帧不进 React"之间取平衡。
 */
const SCROLL_COMMIT_STEP_PX = 256;

/** 键盘单步滚动量（CSS px）。 */
const KEYBOARD_STEP_PX = 60;

/** 滚轮缩放的每步倍率（与旧实现一致：向上放大 1.1、向下缩小 0.9）。 */
const WHEEL_ZOOM_IN_FACTOR = 1.1;
const WHEEL_ZOOM_OUT_FACTOR = 0.9;

/**
 * 细节层的 GL 块面接收器：**什么都不画**。
 *
 * 作用：`drawTimelineCanvas` 在收到 `glBodies` 时会跳过 Canvas2D 块面绘制，只画
 * 细节（旋钮 / 徽标 / 文字 / 淡变曲线 / 吸附三角）。内核的块面由 GL 层负责，
 * 因此这里传一个空实现，让细节层专注于细节——两层叠加后与旧实现逐像素等价。
 */
const NOOP_GL_BODY_SINK: GlClipBodySink = {
    render: () => undefined,
};

/** 全局帧率探针的最小接口（`dev/frameProfiler` 通过 globalThis 挂载）。 */
interface FrameProfilerLike {
    recordLayer(name: string, ms: number): void;
    recordCommit(ms: number): void;
}

/**
 * 构建轨道行分界线实例（相邻轨道之间的 1 物理像素横线）。
 *
 * 流程：从 `firstRow` 起逐行产出「行底边」横线，覆盖整行宽度。
 *
 * 特殊说明：
 * - 位置与左侧轨道头的行边框同源（行 i 的底边 = (i+1) × rowHeight），两侧
 *   行线必须在同一水平线上，否则左右两列看起来是错开的；
 * - 线宽按**物理像素**折算并吸附栅格（与网格线同一策略），否则分数 DPR 下
 *   会出现"有的行线粗、有的细"；
 * - 纵坐标用 `(row + 1) × rowHeight` 而不是累加，避免浮点误差逐行累积。
 *
 * @param args 构建参数（行窗口、行高、横向窗口、DPR、颜色）。
 * @returns 行分界线实例数组。
 */
function buildRowBorderInstances(args: {
    readonly firstRow: number;
    readonly rowCount: number;
    readonly rowHeight: number;
    readonly windowLeftPx: number;
    readonly windowWidthPx: number;
    readonly dpr: number;
    readonly rgba: Rgba;
}): FlatInstance[] {
    const out: FlatInstance[] = [];
    const rowHeight = Math.max(1, args.rowHeight);
    const lineHeight = 1 / Math.max(1, args.dpr);
    for (let index = 0; index < args.rowCount; index += 1) {
        const bottomPx = (args.firstRow + index + 1) * rowHeight;
        out.push({
            x: args.windowLeftPx,
            y: Math.round(bottomPx * args.dpr) / args.dpr - lineHeight,
            w: args.windowWidthPx,
            h: lineHeight,
            rgba: args.rgba,
        });
    }
    return out;
}

/**
 * 读取全局帧率探针。
 *
 * 特殊说明：每帧读取而不是创建时缓存——探针可在运行时开关（localStorage + 按钮），
 * 缓存会让"中途开启"失效；未启用时只是一次属性查找。
 *
 * @returns 探针实例；未启用时为 undefined。
 */
function readFrameProfiler(): FrameProfilerLike | undefined {
    return (globalThis as unknown as { __hfsFrameProfiler?: FrameProfilerLike }).__hfsFrameProfiler;
}

/**
 * 数值夹取。
 *
 * @param value 待夹取值。
 * @param min 下界。
 * @param max 上界。
 * @returns 夹取结果；`value` 非有限值时返回 `min`。
 */
function clampNumber(value: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return min;
    return Math.min(max, Math.max(min, value));
}

/** 已解析的主题色缓存（键 = CSS 颜色原值）。 */
const themeRgbaCache = new Map<string, Rgba>();

/**
 * 用浏览器把任意 CSS 颜色归一化为 `rgb()/rgba()`。
 *
 * 特殊说明：借 `getComputedStyle` 而不是自己写解析器——CSS 颜色格式太多
 * （hex 3/4/6/8 位、`rgb()` 空格/逗号两种语法、颜色关键字、`color()` 函数），
 * 浏览器已经实现了全部规则且与主题变量的取值方式一致。
 *
 * @param css CSS 颜色字符串。
 * @returns 归一化后的字符串；浏览器不可用或解析失败时原样返回。
 */
function normalizeCssColor(css: string): string {
    if (typeof document === "undefined") return css;
    const probe = document.createElement("span");
    probe.style.color = css;
    probe.style.display = "none";
    document.body.appendChild(probe);
    const computed = getComputedStyle(probe).color;
    probe.remove();
    return computed.length > 0 ? computed : css;
}

/**
 * 把主题变量解析为 RGBA（支持任意 CSS 颜色格式）。
 *
 * 【为什么不能直接用 parseRgbaColor】它只认 `rgb()/rgba()` 两种写法，其他格式
 * （hex / 颜色关键字）会被解析成**不透明洋红**——这是既有实现的故意设计，用来在
 * 真机上暴露"漏解析"。但自定义主题常用 hex 定义颜色，直接套用会让网格线、行分界线
 * 全部变成刺眼的洋红。这里先借浏览器归一化，再交给同一个解析器，并按值缓存
 * （`getComputedStyle` 会触发样式重算，不能进每帧路径）。
 *
 * @param name 主题变量名（如 `--qt-graph-grid-weak`）。
 * @param fallback 变量缺失时的回退值（必须是 `rgb()/rgba()` 写法）。
 * @returns RGBA 四元组。
 */
function resolveThemeRgba(name: string, fallback: string): Rgba {
    const raw = resolveThemeColor(name, fallback);
    const cached = themeRgbaCache.get(raw);
    if (cached !== undefined) return cached;
    const parsed = parseRgbaColor(normalizeCssColor(raw));
    themeRgbaCache.set(raw, parsed);
    return parsed;
}

/**
 * 判断 DOM 写入值是否需要提交（值去重）。
 *
 * 【为什么用取反写法】去重变量以 `NaN` 表示「从未写入」，而 `NaN > eps` 恒为
 * **false**：直接写 `Math.abs(next - prev) > eps` 会把**首次**写入判定为
 * 「无需写入」而跳过，表现为标尺 / 播放头 / 细节层在第一次交互前完全不动
 * （后续也永远追不上，因为 `last` 一直是 NaN）。取反写法让 NaN 走「需要写入」分支。
 *
 * @param next 本次要写入的值。
 * @param previous 上一次写入的值（NaN = 从未写入）。
 * @param epsilon 视为「无变化」的最大差值。
 * @returns 需要写入时为 true。
 */
function shouldWrite(next: number, previous: number, epsilon = 0.01): boolean {
    return !(Math.abs(next - previous) <= epsilon);
}

/**
 * 创建内核宿主。
 *
 * 流程：
 * 1. 建 GL 上下文与 program（失败抛错，由 React 外壳展示回退提示）；
 * 2. 建滚动内核（内容尺寸 / 轨道数 / 视口高从数据镜像与尺寸现读）；
 * 3. 建渲染循环并订阅滚动变更；
 * 4. 绑定输入（wheel / 滚动条 / 键盘 / 尺寸变化）；
 * 5. 首次标脏。
 *
 * @param args 宿主参数。
 * @returns 宿主句柄；调用方须在卸载时 `dispose()`。
 */
export function createTimelineKernelHost(args: TimelineKernelHostArgs): TimelineKernelHost {
    const { container, canvas, hScrollbarThumb, vScrollbarThumb, data, sync } = args;
    const {
        onRowHeightChange,
        onZoomChange,
        onVisibleRowsChange,
        onScrollLeftCommit,
        onViewportWidthChange,
        interactions,
    } = args;

    const glCanvas = createGlCanvas(canvas);
    if (!glCanvas) throw new Error("WebGL2 不可用");
    const gl = glCanvas.gl;
    const sdfBox = createSdfBoxProgram(gl);

    const readDpr = () => window.devicePixelRatio || 1;
    const clipBuilder = createClipInstanceBuilder();

    // ── 细节层（Canvas2D 覆盖层）─────────────────────────────────────
    // clip 细节（增益旋钮 / 徽标 / 名称与数值标签 / 淡变曲线 / 吸附三角）是逐 clip
    // 的路径与文字绘制：放 GL 需要自建字形图集与曲线三角化，收益与风险不成比例。
    // 这里保留既有 Canvas2D 渲染器（`drawTimelineCanvas`，传空 GL sink 即只画细节），
    // 并把它放在「内容坐标 + 窗口平移」的画布上——滚动在窗口余量内**只更新画布
    // 位置、不重绘**，与 GL 层共用同一套「滚动零重绘」策略。
    const detailCanvas = document.createElement("canvas");
    detailCanvas.style.position = "absolute";
    detailCanvas.style.left = "0px";
    detailCanvas.style.top = "0px";
    detailCanvas.style.pointerEvents = "none";
    // 层序：GL 块面（画布 0）→ 波形（独立画布，由 React 挂载）→ 细节（2）→ 播放头（20）。
    detailCanvas.style.zIndex = "2";
    container.appendChild(detailCanvas);
    const detailCtx = detailCanvas.getContext("2d");

    // ── 框选矩形（DOM 虚线框）──────────────────────────────────────
    // 用 DOM 而不是 canvas 绘制：虚线框是瞬态 chrome，DOM 的 border-dashed 与主题
    // 变量天然一致；它只在拖拽期间可见，且位置每帧只写 4 个属性。
    const boxSelectEl = document.createElement("div");
    boxSelectEl.style.position = "absolute";
    boxSelectEl.style.pointerEvents = "none";
    boxSelectEl.style.display = "none";
    boxSelectEl.style.zIndex = "15";
    boxSelectEl.style.border = "1px dashed var(--qt-highlight, #3b82f6)";
    boxSelectEl.style.backgroundColor = "rgba(59, 130, 246, 0.12)";
    container.appendChild(boxSelectEl);

    /** 细节层窗口（内容坐标）：与 GL 几何窗口一致，滚动在余量内不重绘。 */
    let detailWindowStartX = 0;
    let detailWindowStartY = 0;
    let detailWindowWidth = 0;
    let detailWindowHeight = 0;
    let detailDrawClips: Parameters<typeof drawTimelineCanvas>[1]["clips"] = [];
    let detailRowGuides: Parameters<typeof drawTimelineCanvas>[1]["rowGuides"];
    let lastDetailLeft = Number.NaN;
    let lastDetailTop = Number.NaN;

    /**
     * 重绘细节层（仅在几何重建时调用）。
     *
     * 流程：按窗口尺寸光栅化 → 设 DPR 变换 → 清屏 → 平移到窗口原点 →
     * 用**内容坐标**调用既有 Canvas2D 渲染器（传空 GL sink，只画细节）。
     *
     * 特殊说明：`drawTimelineCanvas` 内部会自行清屏且用 save/restore 包裹，
     * 不会破坏这里设置的 DPR 变换与平移。
     *
     * @returns 无返回值；细节上下文不可用时静默跳过（不影响块面与网格）。
     */
    function redrawDetails(): void {
        if (detailCtx === null) return;
        if (detailWindowWidth <= 0 || detailWindowHeight <= 0) return;
        const target = rasterize(detailCanvas, detailWindowWidth, detailWindowHeight, readDpr());
        detailCtx.setTransform(target.dpr, 0, 0, target.dpr, 0, 0);
        clearCanvasPhysical(detailCtx, target);
        detailCtx.translate(-detailWindowStartX, -detailWindowStartY);
        drawTimelineCanvas(detailCtx, {
            width: target.cssWidthPx,
            height: target.cssHeightPx,
            clips: detailDrawClips,
            fontFamily: resolveFontFamily(),
            rowGuides: detailRowGuides,
            viewportLeft: detailWindowStartX,
            viewportTopPx: detailWindowStartY,
            darkMode: data().darkMode,
            glBodies: NOOP_GL_BODY_SINK,
            originXPx: detailWindowStartX,
            originYPx: detailWindowStartY,
        });
    }

    /**
     * 把细节层平移到当前视口（滚动帧只做这一步，零重绘）。
     *
     * @param view 当前视口状态。
     * @returns 无返回值。
     */
    function positionDetailLayer(view: TimelineViewportState): void {
        const left = detailWindowStartX - view.scrollLeft;
        const top = detailWindowStartY - view.scrollTop;
        if (shouldWrite(left, lastDetailLeft)) {
            lastDetailLeft = left;
            detailCanvas.style.left = `${left}px`;
        }
        if (shouldWrite(top, lastDetailTop)) {
            lastDetailTop = top;
            detailCanvas.style.top = `${top}px`;
        }
    }

    // 尺寸镜像：由 ResizeObserver 维护，供滚动内核与光栅化读取（O(1)，不触发布局）。
    let viewportWidthPx = Math.max(1, container.clientWidth);
    let viewportHeightPx = Math.max(1, container.clientHeight);

    const scroll = createScrollKernel({
        // 初始缩放取 React 侧恢复的持久化值（旧实现从 localStorage 恢复，
        // 内核直接用固定值会让用户每次启动都回到默认缩放）。
        pxPerSec: clampNumber(data().initialPxPerSec, MIN_PX_PER_SEC, MAX_PX_PER_SEC) || DEFAULT_PX_PER_SEC,
        // 行高与左侧轨道头同源（data().rowHeight），否则左右两列行错位。
        rowHeight: Math.max(1, data().rowHeight || DEFAULT_ROW_HEIGHT),
        projectSec: () => Math.max(0, data().projectSec),
        trackCount: () => data().tracks.length,
        viewportHeightPx: () => viewportHeightPx,
        minPxPerSec: MIN_PX_PER_SEC,
        maxPxPerSec: MAX_PX_PER_SEC,
    });

    // ── 场景缓存（内容坐标常驻；滚动在余量内不重建）─────────────────────
    let combinedInstances = new Float32Array(0);
    let combinedCount = 0;
    let sceneDirty = true;
    /** 实例是否已上传 GPU：true 时滚动帧走 `repaint`（零上传）。 */
    let sceneUploaded = false;
    let builtPxPerSec = Number.NaN;
    let builtScrollLeftPx = Number.NaN;
    let builtScrollTopPx = Number.NaN;
    let builtDarkMode = !data().darkMode;
    // 内容变更判定用引用比较（Redux/Immer 不变时引用稳定），避免每帧字符串签名分配。
    let builtClipsRef: readonly ClipInfo[] | null = null;
    let builtTracksRef: readonly TrackInfo[] | null = null;
    let builtProjectSec = -1;
    let builtGrid = "";
    let builtBpm = -1;
    let builtBeatsPerBar = -1;
    let builtRowHeight = -1;
    // 选中态影响 clip 描边 / 高亮样式，变化时必须重建（用引用比较：Immer 未变更
    // 时数组引用稳定）。初值用 undefined 与「未选中（null）」区分。
    let builtSelectedClipId: string | null | undefined = undefined;
    let builtMultiSelectedRef: readonly string[] | null = null;

    /** 组装当前 axis（内容坐标投影）。 */
    function currentAxis(): TimelineAxis {
        const view = scroll.get();
        return createTimelineAxis({
            pxPerSec: view.pxPerSec,
            scrollLeftPx: view.scrollLeft,
            scrollTopPx: view.scrollTop,
            viewportWidthPx,
            dpr: readDpr(),
        });
    }

    /**
     * 场景内容是否变化。
     *
     * 用**引用比较**（Redux/Immer 在未变更时保持引用稳定）而不是字段值比较：
     * clip 拖动 / 编辑只改元素内容、不改数组长度，按长度比较会漏掉重建；
     * 引用比较既准确又零分配（不拼字符串）。
     */
    function sceneContentChanged(d: TimelineKernelData): boolean {
        return (
            builtClipsRef !== d.clips ||
            builtTracksRef !== d.tracks ||
            builtProjectSec !== d.projectSec ||
            builtGrid !== d.grid ||
            builtBpm !== d.bpm ||
            builtBeatsPerBar !== d.beatsPerBar ||
            // 行高变化会改变所有行的内容坐标（分界线 / clip 位置），必须重建。
            builtRowHeight !== d.rowHeight ||
            builtSelectedClipId !== d.selectedClipId ||
            builtMultiSelectedRef !== d.multiSelectedClipIds
        );
    }

    /** 重建网格 + clip 实例（写入同一个合并缓冲：网格在前、clip 在后）。 */
    function rebuildInstances(axis: TimelineAxis): void {
        const d = data();
        const view = scroll.get();
        const windowLeft = Math.max(0, view.scrollLeft - HORIZONTAL_MARGIN_PX);
        const windowWidth = viewportWidthPx + HORIZONTAL_MARGIN_PX * 2;

        // 竖直窗口：只构建可见轨道行（含 overscan）。
        const visibleRowCount = Math.max(1, Math.ceil(viewportHeightPx / view.rowHeight));
        const firstRow = Math.max(
            0,
            Math.floor(view.scrollTop / view.rowHeight) - VERTICAL_OVERSCAN_ROWS,
        );
        const lastRow = Math.min(
            d.tracks.length,
            firstRow + visibleRowCount + VERTICAL_OVERSCAN_ROWS * 2,
        );
        const visibleTracks = d.tracks.slice(firstRow, Math.max(firstRow, lastRow));

        const ticks: TimelineTick[] = buildTimelineTicks({
            axis,
            bpm: d.bpm,
            beatsPerBar: d.beatsPerBar,
            grid: d.grid,
            primaryUnit: d.primaryTimeUnit,
            secondaryUnit: d.secondaryTimeUnit,
            minLabelSpacingPx: d.minLabelSpacingPx,
            tempoMap: d.tempoMap,
        });

        // 配色与旧实现同源（`--qt-graph-grid-weak` / `strong`）：硬编码按主题取
        // 灰会让浅色主题的网格几乎不可见（真实弱线是 rgba(74,88,112,0.3)）。
        const gridInstances = buildGridInstances({
            ticks,
            windowLeftPx: windowLeft,
            windowWidthPx: windowWidth,
            contentBottomPx: d.tracks.length * view.rowHeight,
            dpr: axis.dpr,
            weakRgba: resolveThemeRgba("--qt-graph-grid-weak", "rgba(255,255,255,0.1)"),
            strongRgba: resolveThemeRgba("--qt-graph-grid-strong", "rgba(255,255,255,0.22)"),
        });

        // 轨道行分界线：旧实现由 Canvas2D 逐行画 `--qt-border`；位置必须与左侧
        // 轨道头的行边框一致，否则左右两列看起来是错开的。
        const rowBorders = buildRowBorderInstances({
            firstRow,
            rowCount: Math.max(0, lastRow - firstRow),
            rowHeight: view.rowHeight,
            windowLeftPx: windowLeft,
            windowWidthPx: windowWidth,
            dpr: axis.dpr,
            rgba: resolveThemeRgba("--qt-border", "rgba(148,163,184,0.22)"),
        });

        // clip 几何：复用既有稀疏模型（窗口裁剪 + 内容坐标投影）。
        const clipsByTrackId: Record<string, ClipInfo[]> = {};
        for (const track of visibleTracks) clipsByTrackId[track.id] = [];
        const safePxPerSec = Math.max(1e-9, view.pxPerSec);
        const bufferSec = Math.max(1.5, HORIZONTAL_MARGIN_PX / safePxPerSec);
        const startSec = Math.max(0, view.scrollLeft / safePxPerSec - bufferSec);
        const endSec = (view.scrollLeft + viewportWidthPx) / safePxPerSec + bufferSec;
        for (const clip of d.clips) {
            const list = clipsByTrackId[clip.trackId];
            if (list === undefined) continue;
            if (clip.startSec + clip.lengthSec < startSec || clip.startSec > endSec) continue;
            list.push(clip);
        }

        const model = buildSparseClipRenderModel({
            visibleTracks: visibleTracks.map((track) => ({ id: track.id, color: track.color })),
            startTrackIndex: firstRow,
            // ClipInfo 与 SparseRenderClip 的字段结构兼容（缺省字段由模型内部兜底）。
            visibleTrackClipsById: clipsByTrackId as never,
            axis,
            rowHeight: view.rowHeight,
            // 选中态来自数据镜像：内核不持有选中真值，只消费（与旧实现同一套
            // 样式模块，选中描边 / 高亮因此逐像素一致）。
            selectedClipId: d.selectedClipId,
            multiSelectedClipIds: [...d.multiSelectedClipIds],
            renamingClipId: null,
        });

        const clipResult = clipBuilder.build({
            clips: model.drawClips,
            darkMode: d.darkMode,
            fontFamily: resolveFontFamily(),
            seamColor: d.darkMode ? "rgb(31, 31, 31)" : "rgb(237, 240, 245)",
        });

        // 合并：网格 → 行分界线 → clip（同一 draw call 内的层叠顺序，后者覆盖前者）。
        const gridCount = gridInstances.length;
        const borderCount = rowBorders.length;
        const needed = (gridCount + borderCount + clipResult.count) * CLIP_INSTANCE_FLOATS;
        if (combinedInstances.length < needed) combinedInstances = new Float32Array(needed);
        for (let index = 0; index < gridCount; index += 1) {
            writeFlatInstance(combinedInstances, index, gridInstances[index]);
        }
        for (let index = 0; index < borderCount; index += 1) {
            writeFlatInstance(combinedInstances, gridCount + index, rowBorders[index]);
        }
        combinedInstances.set(
            clipResult.instances.subarray(0, clipResult.count * CLIP_INSTANCE_FLOATS),
            (gridCount + borderCount) * CLIP_INSTANCE_FLOATS,
        );
        combinedCount = gridCount + borderCount + clipResult.count;

        builtPxPerSec = view.pxPerSec;
        builtScrollLeftPx = view.scrollLeft;
        builtScrollTopPx = view.scrollTop;
        builtDarkMode = d.darkMode;
        builtClipsRef = d.clips;
        builtTracksRef = d.tracks;
        builtProjectSec = d.projectSec;
        builtGrid = d.grid;
        builtBpm = d.bpm;
        builtBeatsPerBar = d.beatsPerBar;
        builtRowHeight = d.rowHeight;
        builtSelectedClipId = d.selectedClipId;
        builtMultiSelectedRef = d.multiSelectedClipIds;
        // 细节层窗口：与 GL 几何窗口完全一致（同一批 drawClips、同一内容坐标
        // 范围），因此两层在滚动时一起平移，不会出现细节与块面错位。
        detailWindowStartX = windowLeft;
        detailWindowStartY = firstRow * view.rowHeight;
        detailWindowWidth = windowWidth;
        detailWindowHeight = Math.max(1, (lastRow - firstRow) * view.rowHeight);
        detailDrawClips = model.drawClips;
        detailRowGuides = {
            startTrackIndex: firstRow,
            rowCount: Math.max(0, lastRow - firstRow),
            rowHeight: view.rowHeight,
            contentBottomPx: d.tracks.length * view.rowHeight,
        };
        redrawDetails();
        positionDetailLayer(view);

        // 可见行窗口变化：低频通知 React 侧（波形 scene rows 按行构建）。
        const rowCount = Math.max(0, lastRow - firstRow);
        if (lastVisibleFirstRow !== firstRow || lastVisibleRowCount !== rowCount) {
            lastVisibleFirstRow = firstRow;
            lastVisibleRowCount = rowCount;
            onVisibleRowsChange?.(firstRow, rowCount);
        }

        // 几何已重建 → 需要重新上传 GPU（下一帧走 render 而不是 repaint）。
        sceneUploaded = false;
        sceneDirty = false;
    }

    /**
     * 判断是否需要重建几何。
     *
     * 触发条件（任一成立即重建）：
     * - 显式标脏 / 缩放变化 / 主题变化 / 内容引用变化；
     * - 水平滚动超出横向余量；
     * - **竖直滚动超出 overscan 行**（轨道窗口按 scrollTop 构建，不重建会缺行）。
     *
     * 特殊说明：余量内的纯滚动不重建——这正是"滚动零重绘"的实现点。
     */
    function ensureScene(axis: TimelineAxis): boolean {
        const view = scroll.get();
        const d = data();
        // 行高的真值源在 React（左侧轨道头与内核必须同源）：竖直缩放后把新行高
        // 同步给滚动内核，否则内容高度与竖直钳制仍按旧行高计算。
        scroll.setRowHeight(d.rowHeight);
        const needsRebuild =
            sceneDirty ||
            builtPxPerSec !== view.pxPerSec ||
            builtDarkMode !== d.darkMode ||
            sceneContentChanged(d) ||
            Math.abs(view.scrollLeft - builtScrollLeftPx) > HORIZONTAL_MARGIN_PX ||
            Math.abs(view.scrollTop - builtScrollTopPx) > view.rowHeight * VERTICAL_OVERSCAN_ROWS;
        if (!needsRebuild) return false;
        rebuildInstances(axis);
        return true;
    }

    // 滚动条 DOM 写入去重：几何取整后未变化时不写 style（避免每帧触发样式重算）。
    let lastHorizontalThumbKey = "";
    let lastVerticalThumbKey = "";

    /** 更新自绘滚动条 thumb 的几何（每帧一次，值变化才写 DOM）。 */
    function updateScrollbars(): void {
        const view = scroll.get();
        const horizontal = computeScrollbar({
            contentSizePx: view.pxPerSec * Math.max(0, data().projectSec),
            viewportSizePx: viewportWidthPx,
            scrollPx: view.scrollLeft,
            maxScrollPx: scroll.maxScrollLeft(),
        });
        const horizontalKey = `${horizontal.scrollable ? 1 : 0}|${Math.round(
            horizontal.thumbLengthPx,
        )}|${Math.round(horizontal.thumbStartPx)}`;
        if (horizontalKey !== lastHorizontalThumbKey) {
            lastHorizontalThumbKey = horizontalKey;
            hScrollbarThumb.style.width = `${Math.max(0, horizontal.thumbLengthPx)}px`;
            hScrollbarThumb.style.transform = `translateX(${Math.max(0, horizontal.thumbStartPx)}px)`;
            hScrollbarThumb.style.display = horizontal.scrollable ? "block" : "none";
        }

        const vertical = computeScrollbar({
            contentSizePx: data().tracks.length * view.rowHeight,
            viewportSizePx: viewportHeightPx,
            scrollPx: view.scrollTop,
            maxScrollPx: scroll.maxScrollTop(),
        });
        const verticalKey = `${vertical.scrollable ? 1 : 0}|${Math.round(
            vertical.thumbLengthPx,
        )}|${Math.round(vertical.thumbStartPx)}`;
        if (verticalKey !== lastVerticalThumbKey) {
            lastVerticalThumbKey = verticalKey;
            vScrollbarThumb.style.height = `${Math.max(0, vertical.thumbLengthPx)}px`;
            vScrollbarThumb.style.transform = `translateY(${Math.max(0, vertical.thumbStartPx)}px)`;
            vScrollbarThumb.style.display = vertical.scrollable ? "block" : "none";
        }
    }

    // ── 外部 DOM 同步（标尺 / 轨道头 / 播放头）─────────────────────────
    // 全部在 rAF 内命令式写入并做值去重：一次样式写入的成本与内容规模无关，
    // 但重复写同一个值会白白触发样式重算，因此只在跨过阈值时写。
    let lastRulerTranslateX = Number.NaN;
    let lastTrackListScrollTop = Number.NaN;
    let lastPlayheadViewportX = Number.NaN;
    let lastPlayheadContentX = Number.NaN;
    /** 上一次绘制的视口（引用比较：`ScrollKernel.get()` 的引用在未变化时稳定）。 */
    let lastDrawnView: TimelineViewportState | null = null;
    /** 上一次绘制的播放头位置（秒），用于判断是否需要继续自驱动。 */
    let lastDrawnPlayheadSec = Number.NaN;
    /** 上一次通知 React 的可见行窗口（去重：同窗口不重复回调）。 */
    let lastVisibleFirstRow = -1;
    let lastVisibleRowCount = -1;
    /** 上一次量化提交给 React 的水平滚动位置（NaN = 从未提交）。 */
    let lastCommittedScrollLeft = Number.NaN;

    /**
     * 独立画布图层（波形等）：内核在视口提交后按 order 调用其 paint。
     *
     * 这些图层自带渲染上下文与几何缓存，只需要一个与内核同源的 axis——把它们
     * 并入内核 GL 上下文等于重写其几何与峰值管线，收益与风险都不成比例。
     */
    const viewportLayers: Array<{
        name: string;
        paint: (axis: TimelineAxis) => void;
        order: number;
    }> = [];

    // ── 命中索引（内容变化时重建）───────────────────────────────────
    // 命中必须覆盖**全部** clip，而不是几何窗口内的可见子集——窗口外的 clip
    // 在滚动回来之前仍需能被点击命中。索引按轨道分桶、桶内按 startSec 升序
    // （命中测试用二分定位候选）。
    let hitClipsByTrack = new Map<string, HitTestClip[]>();
    let hitTracks: readonly TrackInfo[] = [];
    let hitClipsRef: readonly ClipInfo[] | null = null;
    let hitTracksRef: readonly TrackInfo[] | null = null;

    /** 重建命中索引（按 clip / 轨道引用变化判定，避免滚动重建时白做）。 */
    function rebuildHitIndex(): void {
        const d = data();
        const map = new Map<string, HitTestClip[]>();
        for (const track of d.tracks) map.set(track.id, []);
        for (const clip of d.clips) {
            const list = map.get(clip.trackId);
            if (list === undefined) continue;
            list.push({
                id: clip.id,
                trackId: clip.trackId,
                startSec: clip.startSec,
                lengthSec: clip.lengthSec,
            });
        }
        for (const list of map.values()) list.sort((a, b) => a.startSec - b.startSec);
        hitClipsByTrack = map;
        hitTracks = d.tracks;
        hitClipsRef = d.clips;
        hitTracksRef = d.tracks;
    }

    /** 按需刷新命中索引（引用比较：Redux/Immer 未变更时引用稳定）。 */
    function ensureHitIndex(): void {
        const d = data();
        if (hitClipsRef !== d.clips || hitTracksRef !== d.tracks) rebuildHitIndex();
    }

    /**
     * 把内核视口同步到外部 DOM。
     *
     * 流程：
     * 1. 标尺内容层：`translateX(-scrollLeft)`（标尺刻度以内容坐标布局，靠平移跟随）；
     * 2. 轨道头滚动容器：写 `scrollTop`（与内核纵向滚动一致，保证左右行对齐）；
     * 3. 播放头竖线（轨道区 + 标尺）：`playheadSec × pxPerSec − scrollLeft`，
     *    即**视口坐标**（元素位于视口容器内，不随内容滚动）。
     *
     * 特殊说明：轨道头写入 `scrollTop` 会触发其 scroll 事件，若外部把该事件无条件
     * 回灌内核就会形成"内核 → DOM → 内核"的循环。两侧都按 0.5px 容差短路
     * （见 `setScrollTop` 与调用方），因此循环在第一次写入后即收敛。
     *
     * @param view 当前视口状态。
     */
    function syncDom(view: TimelineViewportState): void {
        if (sync === undefined) return;

        const ruler = sync.rulerContent;
        if (ruler != null) {
            const translateX = -view.scrollLeft;
            if (shouldWrite(translateX, lastRulerTranslateX)) {
                lastRulerTranslateX = translateX;
                ruler.style.transform = `translateX(${translateX}px)`;
            }
        }

        const trackList = sync.trackListScroller;
        if (trackList != null) {
            // 容差 0.5px：与调用方的回灌判定同量级，避免「内核 → DOM → 内核」循环。
            if (shouldWrite(view.scrollTop, lastTrackListScrollTop, 0.5)) {
                lastTrackListScrollTop = view.scrollTop;
                trackList.scrollTop = view.scrollTop;
            }
        }

        // 播放头：两个元素的**定位坐标系不同**，不能共用同一个值。
        // - 标尺播放头位于标尺内容层内 → 用内容坐标 `left`（随内容层的
        //   translateX(-scrollLeft) 自动跟随滚动）；
        // - 轨道区播放头位于内核视口容器内 → 用视口坐标 `translateX`
        //   （滚动时必须重算，否则会粘在屏幕上不跟内容走）。
        const playheadContentX = data().playheadSec * view.pxPerSec;
        const playheadViewportX = playheadContentX - view.scrollLeft;
        if (
            shouldWrite(playheadContentX, lastPlayheadContentX) ||
            shouldWrite(playheadViewportX, lastPlayheadViewportX)
        ) {
            lastPlayheadContentX = playheadContentX;
            lastPlayheadViewportX = playheadViewportX;
            // 设备像素吸附：分数 DPR 下不吸附会让线宽在 1↔2 物理像素间跳动。
            const dpr = readDpr();
            const snappedContentX = Math.round(playheadContentX * dpr) / dpr;
            if (sync.rulerPlayheadLine != null) {
                sync.rulerPlayheadLine.style.left = `${snappedContentX}px`;
            }
            if (sync.playheadLine != null) {
                sync.playheadLine.style.transform = `translateX(${snappedContentX - view.scrollLeft}px)`;
            }
        }
    }

    /**
     * 绘制一帧：清屏 → 网格 + clip（一次 draw call）→ 文字。
     *
     * 特殊说明（性能关键）：滚动帧走 `repaint`——实例缓冲已在 GPU 上，只更新
     * `u_viewOrigin` uniform 后重发 draw call，**不上传任何顶点数据**；只有几何
     * 重建后的第一帧才 `render`（上传）。这是"滚动零重绘"的最后一环。
     *
     * 帧耗时经全局帧率探针（`__hfsFrameProfiler`，见 dev/frameProfiler）上报为
     * `kernel-draw` 图层，便于与既有实现同面板对比；探针未启用时只有一次属性查找。
     */
    function draw(): void {
        const profiler = readFrameProfiler();
        const startMs = profiler === undefined ? 0 : performance.now();
        const view = scroll.get();
        const axis = currentAxis();
        const rebuilt = ensureScene(axis);
        const viewChanged = lastDrawnView !== view;
        lastDrawnView = view;

        // GL 只在「视口变化」或「几何重建」时提交：播放头移动这类纯 DOM 更新
        // 不需要重绘 canvas（播放中的每帧成本因此退化为几次样式写入）。
        if (viewChanged || rebuilt) {
            const target = glCanvas!.resize(viewportWidthPx, viewportHeightPx, readDpr());
            glCanvas!.clear();
            if (combinedCount > 0) {
                if (sceneUploaded) {
                    sdfBox.repaint(target, view.scrollLeft, view.scrollTop);
                } else {
                    sdfBox.render(
                        combinedInstances,
                        combinedCount,
                        target,
                        view.scrollLeft,
                        view.scrollTop,
                    );
                    sceneUploaded = true;
                }
            }
            // 独立画布图层（波形）：与 GL 同帧提交，避免两层在滚动时分离。
            // 它们内部自行判定重绘 / 复用（波形的平移帧只更新 uniform）。
            for (const layer of viewportLayers) layer.paint(axis);
        }

        updateScrollbars();
        // 细节层与 GL 层同帧平移：两层共用同一个内容坐标窗口，滚动时一起移动。
        positionDetailLayer(view);
        syncDom(view);

        // 水平滚动量化提交：标尺的**刻度范围**由 React 按 scrollLeft 计算，
        // 只写内容层 transform 会让刻度停留在初始视口（滚动后刻度消失）。
        if (onScrollLeftCommit !== undefined) {
            if (shouldWrite(view.scrollLeft, lastCommittedScrollLeft, SCROLL_COMMIT_STEP_PX)) {
                lastCommittedScrollLeft = view.scrollLeft;
                onScrollLeftCommit(view.scrollLeft);
            }
        }

        const playheadSec = data().playheadSec;
        const playheadMoved = playheadSec !== lastDrawnPlayheadSec;
        lastDrawnPlayheadSec = playheadSec;
        // 播放中播放头逐帧移动：继续自驱动；停止后位置不再变化，循环自然收敛。
        if (playheadMoved) loop.invalidate();

        if (profiler !== undefined) {
            const elapsedMs = performance.now() - startMs;
            // 内核只有一层绘制，图层耗时即整次提交耗时。
            profiler.recordLayer("kernel-draw", elapsedMs);
            profiler.recordCommit(elapsedMs);
        }
    }

    const loop = createRenderLoop({ draw });
    loop.start();
    const unsubscribeScroll = scroll.subscribe(() => loop.invalidate());

    // ── 输入：wheel（滚动 / 缩放，规则与旧实现同源）──────────────────
    /**
     * 判断指针是否悬停在自绘滚动条上。
     *
     * 规则：竖直条占右侧 `SCROLLBAR_SIZE_PX` 宽，水平条占底部同样高；右下角
     * 交叠处归竖直（与既有 `nativeScrollbarZoneAt` 的判定顺序一致）。
     *
     * @param clientX 指针视口坐标 X。
     * @param clientY 指针视口坐标 Y。
     * @param rect 宿主容器的视口矩形。
     * @returns 命中的滚动条轴；未命中为 null。
     */
    function scrollbarZoneAt(clientX: number, clientY: number, rect: DOMRect): ScrollbarZone | null {
        if (
            clientX < rect.left ||
            clientX > rect.right ||
            clientY < rect.top ||
            clientY > rect.bottom
        ) {
            return null;
        }
        if (clientX > rect.right - SCROLLBAR_SIZE_PX) return "vertical";
        if (clientY > rect.bottom - SCROLLBAR_SIZE_PX) return "horizontal";
        return null;
    }

    /**
     * 解析并执行滚轮手势。
     *
     * 流程（与 `TimelineScrollArea` 的旧实现逐条对齐）：
     * 1. 悬停滚动条 → 滚轮只归属该轴（无修饰键 = 该轴滚动，`scrollbarZoom` = 该轴缩放），
     *    优先于一切全局绑定；
     * 2. 否则按 keybinding 判定四类请求（none-binding 在**无修饰键**时命中）；
     * 3. `getTimelineWheelAction` 把请求 + delta 主轴映射为具体动作；
     * 4. 滚动类动作直接改视口；缩放类动作走 `resolveHorizontalWheelZoom`
     *    （含播放头锚点与上下限）或行高缩放（锚点 = 指针下的行位置不变）。
     *
     * @param event 原生 wheel 事件。
     * @returns 无返回值。
     */
    function onWheel(event: WheelEvent): void {
        const rect = container.getBoundingClientRect();
        const d = data();
        const kb = d.keybindings;
        const noModifierPressed =
            !event.ctrlKey && !event.metaKey && !event.altKey && !event.shiftKey;
        const isWheelBindingRequested = (binding: Keybinding | null): boolean => {
            // 未提供绑定的动作 = 未请求；none-binding 表示"无修饰键滚轮"。
            if (binding == null) return false;
            if (isNoneBinding(binding)) return noModifierPressed;
            return isModifierActive(binding, event);
        };

        const scrollbarZone = scrollbarZoneAt(event.clientX, event.clientY, rect);
        const scrollbarZoomRequested =
            scrollbarZone != null &&
            kb.scrollbarZoom != null &&
            !isNoneBinding(kb.scrollbarZoom) &&
            isModifierActive(kb.scrollbarZoom, event);

        const horizontalScrollRequested =
            scrollbarZone === "horizontal" && !scrollbarZoomRequested
                ? true
                : scrollbarZone == null && isWheelBindingRequested(kb.scrollHorizontal);
        const verticalScrollRequested =
            scrollbarZone === "vertical" && !scrollbarZoomRequested
                ? true
                : scrollbarZone == null && isWheelBindingRequested(kb.scrollVertical);
        const horizontalZoomRequested =
            scrollbarZone == null
                ? isWheelBindingRequested(kb.horizontalZoom)
                : scrollbarZone === "horizontal" && scrollbarZoomRequested;
        const verticalZoomRequested =
            scrollbarZone == null
                ? isWheelBindingRequested(kb.verticalZoom)
                : scrollbarZone === "vertical" && scrollbarZoomRequested;

        const action = getTimelineWheelAction({
            deltaX: event.deltaX,
            deltaY: event.deltaY,
            horizontalScrollRequested,
            verticalScrollRequested,
            verticalZoomRequested,
            horizontalZoomRequested,
        });

        const ctx = { lineHeightPx: 16, pageHeightPx: Math.max(1, rect.height) };
        const deltaY = normalizeWheelDelta(event.deltaY, event.deltaMode, ctx);
        const deltaX = normalizeWheelDelta(event.deltaX, event.deltaMode, ctx);
        const view = scroll.get();

        if (action === "free-scroll") {
            event.preventDefault();
            scroll.setScrollLeft(view.scrollLeft + deltaX);
            scroll.setScrollTop(view.scrollTop + deltaY);
            return;
        }

        if (action === "horizontal-scroll") {
            event.preventDefault();
            // 触摸板横向双指用 deltaX；鼠标滚轮只有 deltaY，回退到它。
            const horizontalDelta = Math.abs(event.deltaX) > 0.5 ? deltaX : deltaY;
            scroll.setScrollLeft(view.scrollLeft + horizontalDelta);
            return;
        }

        if (action === "vertical-scroll") {
            event.preventDefault();
            scroll.setScrollTop(view.scrollTop + deltaY);
            return;
        }

        if (action === "native") {
            // 内核没有原生滚动容器，"native" 等价于双轴自由滚动。
            event.preventDefault();
            scroll.setScrollLeft(view.scrollLeft + deltaX);
            scroll.setScrollTop(view.scrollTop + deltaY);
            return;
        }

        event.preventDefault();
        const factor = event.deltaY < 0 ? WHEEL_ZOOM_IN_FACTOR : WHEEL_ZOOM_OUT_FACTOR;

        if (action === "vertical-zoom") {
            const baseRowHeight = Math.max(1, view.rowHeight);
            const pointerY = clampNumber(event.clientY - rect.top, 0, Math.max(1, rect.height));
            // 锚点：指针下的行位置（行单位）保持不变。
            const rowUnitAtPointer = (view.scrollTop + pointerY) / baseRowHeight;
            const nextRowHeight = Math.round(
                clampNumber(baseRowHeight * factor, MIN_ROW_HEIGHT, MAX_ROW_HEIGHT),
            );
            if (nextRowHeight === baseRowHeight) return;
            onRowHeightChange?.(nextRowHeight);
            scroll.setScrollTop(rowUnitAtPointer * nextRowHeight - pointerY);
            return;
        }

        const totalSec = Math.max(0, d.projectSec);
        const minPxPerSec = resolveTimelineMinPxPerSec({
            baseMinPxPerSec: MIN_PX_PER_SEC,
            projectSec: totalSec,
            viewportWidthPx,
        });
        const zoom = resolveHorizontalWheelZoom({
            factor,
            basePxPerSec: view.pxPerSec,
            baseScrollLeft: view.scrollLeft,
            totalSec,
            viewportWidth: viewportWidthPx,
            playheadZoomEnabled: d.playheadZoomEnabled,
            playheadSec: d.playheadSec,
            anchorScreenX: event.clientX - rect.left,
            minPxPerSec,
            maxPxPerSec: MAX_PX_PER_SEC,
        });
        if (zoom === null) return;
        scroll.setZoom(zoom.nextPxPerSec, event.clientX - rect.left);
        scroll.setScrollLeft(zoom.nextScrollLeft);
        onZoomChange?.(zoom.nextPxPerSec);
    }
    container.addEventListener("wheel", onWheel, { passive: false });

    // ── 输入：中键平移 ───────────────────────────────────────────────
    // 与旧实现一致：仅鼠标中键（pointerType === "mouse"）、按下期间抓取指针，
    // 位移按 1:1 反向映射到滚动位置（抓取内容而不是"推视口"的手感）。
    let panPointerId: number | null = null;
    let panStartX = 0;
    let panStartY = 0;
    let panStartScrollLeft = 0;
    let panStartScrollTop = 0;

    // ── 输入：左键手势（选中 / seek）─────────────────────────────────
    /**
     * 手势状态（同一时刻只有一个）。
     *
     * - `pending-select`：按下时命中 clip，等待「抬起（选中）」或「超过位移阈值
     *   （拖拽）」二选一——用阈值区分点击与拖拽是 DAW 的通用手感，避免手抖把
     *   一次点击判成拖拽。
     * - `seek`：按下时命中空白，立即 seek 并在拖拽中持续 seek。
     */
    type Gesture =
        | { kind: "none" }
        | {
              kind: "pending-select";
              startClientX: number;
              startClientY: number;
              clipId: string;
              /** 命中分区：决定超过阈值后升级为拖拽（body/header）还是 trim（边缘）。 */
              region: ClipHitRegion;
              /** 按下时的内容坐标（用于换算拖拽位移）。 */
              startContentX: number;
              startContentY: number;
              /** 按下时 clip 的几何（换算 delta 与跨轨的基准）。 */
              originStartSec: number;
              originTrackId: string;
              lengthSec: number;
              /** 按下时的淡变长度（按 region 取对应侧；非淡变角为 0）。 */
              originFadeSec: number;
          }
        | {
              kind: "clip-fade";
              clipId: string;
              side: FadeSide;
              startContentX: number;
              originFadeSec: number;
              lengthSec: number;
              lastDeltaSec: number;
              lastFadeSec: number;
          }
        | {
              kind: "box-select";
              startClientX: number;
              startClientY: number;
              startContentX: number;
              startContentY: number;
              additive: boolean;
              /** 是否已超过阈值（未超过时不显示框，避免右键单击闪一下）。 */
              active: boolean;
              /** 最近一次命中的 clip 集合（去重：同集合不重复回调）。 */
              lastClipIds: readonly string[];
          }
        | {
              kind: "clip-trim";
              clipId: string;
              edge: TrimEdge;
              startContentX: number;
              originStartSec: number;
              originLengthSec: number;
              /** 最近一次派发的实际变化量（去重）。 */
              lastDeltaSec: number;
              /** 最近一次派发的新几何（收尾时直接提交，避免重算）。 */
              lastStartSec: number;
              lastLengthSec: number;
          }
        | {
              kind: "clip-drag";
              clipId: string;
              startContentX: number;
              startContentY: number;
              originStartSec: number;
              originTrackId: string;
              lengthSec: number;
              /** 最近一次派发的预览值（去重：同值不重复回调，避免空重建）。 */
              lastDeltaSec: number;
              lastTargetTrackId: string;
          }
        | { kind: "seek" };
    let gesture: Gesture = { kind: "none" };

    /** 点击 / 拖拽的位移阈值（CSS px）。 */
    const DRAG_THRESHOLD_PX = 4;

    /** trim 允许的最小 clip 长度（秒）：再短会难以命中与选中。 */
    const MIN_CLIP_LENGTH_SEC = 0.05;

    /** 框选阈值（CSS px）：与旧实现 `TIMELINE_SELECTION_DRAG_THRESHOLD_PX` 一致。 */
    const BOX_SELECT_THRESHOLD_PX = 5;

    /**
     * 是否需要抑制下一次 `contextmenu`。
     *
     * 右键按下即进入「待框选」，若不做抑制，框选松手时浏览器会补发 contextmenu
     * 弹出右键菜单（与刚完成的框选操作冲突）。
     */
    let suppressNextContextMenu = false;

    /**
     * 把视口内坐标换算为命中结果（内容坐标下的轨道 + clip）。
     *
     * @param clientX 指针视口坐标 X。
     * @param clientY 指针视口坐标 Y。
     * @returns 命中结果。
     */
    function hitAt(clientX: number, clientY: number): ReturnType<typeof hitTest> {
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        ensureHitIndex();
        return hitTest({
            contentX: view.scrollLeft + (clientX - rect.left),
            contentY: view.scrollTop + (clientY - rect.top),
            pxPerSec: view.pxPerSec,
            rowHeight: view.rowHeight,
            tracks: hitTracks,
            clipsByTrack: hitClipsByTrack,
            headerHeightPx: CLIP_HEADER_HEIGHT,
        });
    }

    /**
     * 把视口内坐标换算为工程时间。
     *
     * @param clientX 指针视口坐标 X。
     * @returns 工程时间（秒，已钳制到 >= 0）。
     */
    function secAt(clientX: number): number {
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        return Math.max(
            0,
            (view.scrollLeft + (clientX - rect.left)) / Math.max(1e-9, view.pxPerSec),
        );
    }

    /** 判断事件目标是否为可编辑元素（输入框内的中键不应触发平移）。 */
    function isEditableTarget(target: EventTarget | null): boolean {
        const element = target as HTMLElement | null;
        if (element == null) return false;
        const tag = element.tagName;
        return (
            tag === "INPUT" ||
            tag === "TEXTAREA" ||
            tag === "SELECT" ||
            element.isContentEditable === true
        );
    }

    /**
     * 指针按下总入口：按按钮分派到中键平移 / 左键手势。
     *
     * 特殊说明：可编辑元素（输入框等）内的按下必须放行——重命名输入框的文本
     * 光标与选区操作不能被时间轴手势吞掉。
     */
    function onPointerDown(event: PointerEvent): void {
        if (isEditableTarget(event.target)) return;
        if (event.button === 1) {
            if (event.pointerType !== "mouse") return;
            startMiddlePan(event);
            return;
        }
        if (event.button === 2) {
            // 右键按下进入「待框选」：未超过阈值时让 contextmenu 正常弹菜单，
            // 超过阈值才真正开始框选（与旧实现一致）。
            const rect = container.getBoundingClientRect();
            const view = scroll.get();
            gesture = {
                kind: "box-select",
                startClientX: event.clientX,
                startClientY: event.clientY,
                startContentX: view.scrollLeft + (event.clientX - rect.left),
                startContentY: view.scrollTop + (event.clientY - rect.top),
                additive: event.ctrlKey || event.metaKey,
                active: false,
                lastClipIds: [],
            };
            return;
        }
        if (event.button !== 0) return;
        startPrimaryGesture(event);
    }

    /**
     * 收集落在框内的 clip id（内容坐标矩形相交）。
     *
     * 特殊说明：遍历**全部** clip 而不是几何窗口内的子集——窗口外的 clip 同样
     * 可能落在框内（框可以拖出可视区）。
     *
     * @param box 规范化后的内容坐标矩形。
     * @param view 当前视口。
     * @returns 命中的 clip id 列表。
     */
    function collectClipsInBox(box: BoxBounds, view: TimelineViewportState): string[] {
        ensureHitIndex();
        const trackIndexById = new Map<string, number>();
        for (let index = 0; index < hitTracks.length; index += 1) {
            trackIndexById.set(hitTracks[index].id, index);
        }
        const result: string[] = [];
        for (const clip of data().clips) {
            const trackIndex = trackIndexById.get(clip.trackId);
            if (trackIndex === undefined) continue;
            if (
                clipIntersectsBox({
                    box,
                    clipStartSec: clip.startSec,
                    clipLengthSec: clip.lengthSec,
                    trackIndex,
                    pxPerSec: view.pxPerSec,
                    rowHeight: view.rowHeight,
                })
            ) {
                result.push(clip.id);
            }
        }
        return result;
    }

    /**
     * 驱动框选（右键拖拽期间每帧调用）。
     *
     * 流程：阈值判定（未超过不显示框）→ 规范化矩形 → 更新 DOM 框（视口坐标）
     * → 收集命中集合 → 按集合去重后回调。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyBoxSelect(event: PointerEvent): void {
        if (gesture.kind !== "box-select") return;
        const dx = event.clientX - gesture.startClientX;
        const dy = event.clientY - gesture.startClientY;
        if (!gesture.active) {
            if (dx * dx + dy * dy < BOX_SELECT_THRESHOLD_PX * BOX_SELECT_THRESHOLD_PX) return;
            gesture.active = true;
            // 拖拽已成立：抑制随后的 contextmenu（否则松手会弹右键菜单）。
            suppressNextContextMenu = true;
        }
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const box = resolveBoxBounds(
            gesture.startContentX,
            gesture.startContentY,
            view.scrollLeft + (event.clientX - rect.left),
            view.scrollTop + (event.clientY - rect.top),
        );
        boxSelectEl.style.display = "block";
        boxSelectEl.style.left = `${box.left - view.scrollLeft}px`;
        boxSelectEl.style.top = `${box.top - view.scrollTop}px`;
        boxSelectEl.style.width = `${Math.max(0, box.right - box.left)}px`;
        boxSelectEl.style.height = `${Math.max(0, box.bottom - box.top)}px`;

        const ids = collectClipsInBox(box, view);
        if (ids.length === gesture.lastClipIds.length) {
            let same = true;
            for (let index = 0; index < ids.length; index += 1) {
                if (ids[index] !== gesture.lastClipIds[index]) {
                    same = false;
                    break;
                }
            }
            if (same) return;
        }
        gesture.lastClipIds = ids;
        interactions?.onBoxSelectPreview?.({ clipIds: ids, additive: gesture.additive });
    }

    /** 左键按下：命中 clip → 待选中/待拖拽；命中空白 → 立即 seek 并进入拖拽 seek。 */
    function startPrimaryGesture(event: PointerEvent): void {
        const hit = hitAt(event.clientX, event.clientY);
        if (hit.kind === "clip") {
            const rect = container.getBoundingClientRect();
            const view = scroll.get();
            // 淡变角需要当前的淡变长度：命中索引只带几何字段，这里按 region
            // 决定取哪一侧，避免把手势升级时的分支判断散落到多处。
            const fadeSide: FadeSide | null =
                hit.region === "fade-in-corner"
                    ? "in"
                    : hit.region === "fade-out-corner"
                      ? "out"
                      : null;
            const clipInfo =
                fadeSide === null
                    ? undefined
                    : data().clips.find((item) => item.id === hit.clip.id);
            gesture = {
                kind: "pending-select",
                startClientX: event.clientX,
                startClientY: event.clientY,
                clipId: hit.clip.id,
                region: hit.region,
                startContentX: view.scrollLeft + (event.clientX - rect.left),
                startContentY: view.scrollTop + (event.clientY - rect.top),
                originStartSec: hit.clip.startSec,
                originTrackId: hit.clip.trackId,
                lengthSec: hit.clip.lengthSec,
                originFadeSec:
                    fadeSide === "in"
                        ? (clipInfo?.fadeInSec ?? 0)
                        : fadeSide === "out"
                          ? (clipInfo?.fadeOutSec ?? 0)
                          : 0,
            };
        } else {
            gesture = { kind: "seek" };
            interactions?.onSeek?.(hit.sec, true);
        }
        try {
            container.setPointerCapture(event.pointerId);
        } catch {
            // 捕获失败不影响手势（window 上的 move 监听仍会收到事件）。
        }
    }

    /**
     * 悬停光标：按命中分区设置（取值与旧实现一致）。
     *
     * 特殊说明：只在**无手势**时更新——手势期间的光标由 `onGesturePointerMove`
     * 按手势类型设置，否则普通移动会把手势光标覆盖掉（表现为"拖拽时又变回箭头"）。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function updateHoverCursor(event: PointerEvent): void {
        if (gesture.kind !== "none" || panPointerId !== null) return;
        const hit = hitAt(event.clientX, event.clientY);
        let cursor = "default";
        if (hit.kind === "clip") {
            switch (hit.region) {
                case "left-edge":
                case "right-edge":
                    cursor = "ew-resize";
                    break;
                case "fade-in-corner":
                    cursor = "nwse-resize";
                    break;
                case "fade-out-corner":
                    cursor = "nesw-resize";
                    break;
                default:
                    cursor = "grab";
                    break;
            }
        }
        if (container.style.cursor !== cursor) container.style.cursor = cursor;
    }

    /** 左键手势的移动处理（中键平移由 onPanPointerMove 单独负责）。 */
    function onGesturePointerMove(event: PointerEvent): void {
        // 手势光标（与旧实现取值一致）：拖拽 grab→grabbing、trim/fade 用 resize、
        // 框选用 crosshair。
        if (gesture.kind === "clip-drag") {
            container.style.cursor = "grabbing";
        } else if (gesture.kind === "clip-trim") {
            container.style.cursor = "ew-resize";
        } else if (gesture.kind === "clip-fade") {
            container.style.cursor = gesture.side === "in" ? "nwse-resize" : "nesw-resize";
        } else if (gesture.kind === "box-select") {
            container.style.cursor = "crosshair";
        }
        if (gesture.kind === "seek") {
            interactions?.onSeek?.(secAt(event.clientX), false);
            return;
        }
        if (gesture.kind === "box-select") {
            applyBoxSelect(event);
            return;
        }
        if (gesture.kind === "pending-select") {
            const dx = event.clientX - gesture.startClientX;
            const dy = event.clientY - gesture.startClientY;
            if (dx * dx + dy * dy < DRAG_THRESHOLD_PX * DRAG_THRESHOLD_PX) return;
            // 超过阈值：按按下时的命中分区升级——淡变角 → fade、边缘 → trim、
            // 其余 → 拖拽移动。淡变角与边缘在水平方向重叠，命中层已按竖直方向
            // 切分（见 hitTest 的 ClipHitRegion 注释），这里只需按 region 分派。
            if (gesture.region === "fade-in-corner" || gesture.region === "fade-out-corner") {
                gesture = {
                    kind: "clip-fade",
                    clipId: gesture.clipId,
                    side: gesture.region === "fade-in-corner" ? "in" : "out",
                    startContentX: gesture.startContentX,
                    originFadeSec: gesture.originFadeSec,
                    lengthSec: gesture.lengthSec,
                    lastDeltaSec: Number.NaN,
                    lastFadeSec: gesture.originFadeSec,
                };
                applyFadePreview(event);
                return;
            }
            const isEdge = gesture.region === "left-edge" || gesture.region === "right-edge";
            if (isEdge) {
                gesture = {
                    kind: "clip-trim",
                    clipId: gesture.clipId,
                    edge: gesture.region === "left-edge" ? "left" : "right",
                    startContentX: gesture.startContentX,
                    originStartSec: gesture.originStartSec,
                    originLengthSec: gesture.lengthSec,
                    lastDeltaSec: Number.NaN,
                    lastStartSec: gesture.originStartSec,
                    lastLengthSec: gesture.lengthSec,
                };
                applyTrimPreview(event);
                return;
            }
            gesture = {
                kind: "clip-drag",
                clipId: gesture.clipId,
                startContentX: gesture.startContentX,
                startContentY: gesture.startContentY,
                originStartSec: gesture.originStartSec,
                originTrackId: gesture.originTrackId,
                lengthSec: gesture.lengthSec,
                lastDeltaSec: 0,
                lastTargetTrackId: gesture.originTrackId,
            };
            applyDragPreview(event);
            return;
        }
        if (gesture.kind === "clip-trim") {
            applyTrimPreview(event);
            return;
        }
        if (gesture.kind === "clip-fade") {
            applyFadePreview(event);
            return;
        }
        if (gesture.kind === "clip-drag") {
            applyDragPreview(event);
        }
    }

    /**
     * 计算并派发一次 trim 预览（拖拽边缘期间每帧调用）。
     *
     * 流程：内容坐标位移 → `resolveTrimEdge`（新起始时间 + 新长度 + 边界/最小长度
     * 钳制）→ 与上次值比较去重 → 回调。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyTrimPreview(event: PointerEvent): void {
        if (gesture.kind !== "clip-trim") return;
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const contentX = view.scrollLeft + (event.clientX - rect.left);
        const result = resolveTrimEdge({
            edge: gesture.edge,
            deltaContentXPx: contentX - gesture.startContentX,
            pxPerSec: view.pxPerSec,
            startSec: gesture.originStartSec,
            lengthSec: gesture.originLengthSec,
            projectSec: Math.max(0, data().projectSec),
            minLengthSec: MIN_CLIP_LENGTH_SEC,
        });
        if (result.deltaSec === gesture.lastDeltaSec) return;
        gesture.lastDeltaSec = result.deltaSec;
        gesture.lastStartSec = result.startSec;
        gesture.lastLengthSec = result.lengthSec;
        interactions?.onTrimPreview?.({
            clipId: gesture.clipId,
            edge: gesture.edge,
            startSec: result.startSec,
            lengthSec: result.lengthSec,
            deltaSec: result.deltaSec,
        });
    }

    /**
     * 计算并派发一次淡变角预览（拖拽角部期间每帧调用）。
     *
     * 流程：内容坐标位移 → `resolveFadeDrag`（方向按侧翻转 + 钳制到 clip 长度）
     * → 与上次值比较去重 → 回调。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyFadePreview(event: PointerEvent): void {
        if (gesture.kind !== "clip-fade") return;
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const contentX = view.scrollLeft + (event.clientX - rect.left);
        const result = resolveFadeDrag({
            side: gesture.side,
            deltaContentXPx: contentX - gesture.startContentX,
            pxPerSec: view.pxPerSec,
            currentSec: gesture.originFadeSec,
            lengthSec: gesture.lengthSec,
        });
        if (result.deltaSec === gesture.lastDeltaSec) return;
        gesture.lastDeltaSec = result.deltaSec;
        gesture.lastFadeSec = result.fadeSec;
        interactions?.onFadePreview?.({
            clipId: gesture.clipId,
            side: gesture.side,
            fadeSec: result.fadeSec,
            deltaSec: result.deltaSec,
        });
    }

    /**
     * 计算并派发一次拖拽预览（拖拽期间每帧调用）。
     *
     * 流程：内容坐标位移 → `resolveDragDelta`（时间位移 + 边界钳制）→
     * `resolveTargetTrackIndex`（落点轨道）→ 与上次值比较去重 → 回调。
     *
     * 特殊说明：去重是必要的——调用方每次预览都写 Redux，而 Redux 变更会触发
     * 内核重建几何；不去重时指针静止也会持续重建。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyDragPreview(event: PointerEvent): void {
        if (gesture.kind !== "clip-drag") return;
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const d = data();
        const contentX = view.scrollLeft + (event.clientX - rect.left);
        const contentY = view.scrollTop + (event.clientY - rect.top);
        const delta = resolveDragDelta({
            deltaContentXPx: contentX - gesture.startContentX,
            pxPerSec: view.pxPerSec,
            startSec: gesture.originStartSec,
            lengthSec: gesture.lengthSec,
            projectSec: Math.max(0, d.projectSec),
        });
        const trackIndex = resolveTargetTrackIndex(contentY, view.rowHeight, d.tracks.length);
        const targetTrackId =
            trackIndex >= 0
                ? (d.tracks[trackIndex]?.id ?? gesture.originTrackId)
                : gesture.originTrackId;
        if (delta.deltaSec === gesture.lastDeltaSec && targetTrackId === gesture.lastTargetTrackId) {
            return;
        }
        gesture.lastDeltaSec = delta.deltaSec;
        gesture.lastTargetTrackId = targetTrackId;
        interactions?.onDragPreview?.({
            clipId: gesture.clipId,
            deltaSec: delta.deltaSec,
            targetTrackId,
        });
    }

    /**
     * 左键手势收尾。
     *
     * @param event 指针事件。
     * @param cancelled true = 取消（pointercancel / 卸载）：拖拽应回滚而不提交。
     */
    function onGesturePointerUp(event: PointerEvent, cancelled = false): void {
        if (gesture.kind === "clip-drag") {
            interactions?.onDragCommit?.({
                clipId: gesture.clipId,
                deltaSec: gesture.lastDeltaSec,
                targetTrackId: gesture.lastTargetTrackId,
                cancelled,
            });
        }
        if (gesture.kind === "clip-trim") {
            interactions?.onTrimCommit?.({
                clipId: gesture.clipId,
                edge: gesture.edge,
                startSec: gesture.lastStartSec,
                lengthSec: gesture.lastLengthSec,
                cancelled,
            });
        }
        if (gesture.kind === "clip-fade") {
            interactions?.onFadeCommit?.({
                clipId: gesture.clipId,
                side: gesture.side,
                fadeSec: gesture.lastFadeSec,
                cancelled,
            });
        }
        if (gesture.kind === "box-select") {
            // 未超过阈值（右键单击）不提交：交给 contextmenu 弹菜单。
            if (gesture.active) {
                interactions?.onBoxSelectCommit?.({
                    clipIds: gesture.lastClipIds,
                    additive: gesture.additive,
                    cancelled,
                });
            }
            boxSelectEl.style.display = "none";
        }
        if (gesture.kind === "pending-select" && !cancelled) {
            interactions?.onSelectClip?.(gesture.clipId, event.ctrlKey || event.metaKey);
        }
        if (gesture.kind !== "none") {
            try {
                container.releasePointerCapture(event.pointerId);
            } catch {
                // 已释放 / 未捕获：忽略。
            }
            gesture = { kind: "none" };
            // 交回悬停逻辑（下一次 pointermove 会按命中分区重设）。
            container.style.cursor = "";
        }
    }

    /**
     * 指针取消（系统抢占 / 触摸中断 / 浏览器手势接管）：按取消处理。
     *
     * 与 `pointerup` 分开注册的原因：取消路径必须让调用方**回滚**乐观位置，
     * 而正常抬起要提交——两者语义相反，共用一个入口会漏掉取消分支。
     */
    function onGesturePointerCancel(event: PointerEvent): void {
        onGesturePointerUp(event, true);
    }

    /** 中键平移的按下处理（抓取式：记录起点与当时的滚动位置）。 */
    function startMiddlePan(event: PointerEvent): void {
        event.preventDefault();
        const view = scroll.get();
        panPointerId = event.pointerId;
        panStartX = event.clientX;
        panStartY = event.clientY;
        panStartScrollLeft = view.scrollLeft;
        panStartScrollTop = view.scrollTop;
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none";
        try {
            container.setPointerCapture(event.pointerId);
        } catch {
            // 指针捕获失败不影响手势（window 上的 move 监听仍会收到事件）。
        }
    }

    function onPanPointerMove(event: PointerEvent): void {
        if (panPointerId === null || event.pointerId !== panPointerId) return;
        scroll.setScrollLeft(panStartScrollLeft - (event.clientX - panStartX));
        scroll.setScrollTop(panStartScrollTop - (event.clientY - panStartY));
    }

    function endPan(event?: PointerEvent): void {
        if (panPointerId === null) return;
        if (event !== undefined && event.pointerId !== panPointerId) return;
        try {
            container.releasePointerCapture(panPointerId);
        } catch {
            // 已释放 / 未捕获：忽略。
        }
        panPointerId = null;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
    }

    /** 中键的 auxclick（抬起）与浏览器"自动滚动"：内核自绘滚动，必须全部吞掉。 */
    function onAuxClick(event: MouseEvent): void {
        if (event.button === 1) event.preventDefault();
    }

    /**
     * 框选拖拽结束后抑制一次右键菜单。
     *
     * 右键按下即进入「待框选」，若不做抑制，框选松手时浏览器补发的 contextmenu
     * 会弹出右键菜单（与刚完成的框选操作冲突）。
     */
    function onContextMenu(event: MouseEvent): void {
        if (suppressNextContextMenu) {
            event.preventDefault();
            suppressNextContextMenu = false;
            return;
        }
        if (interactions?.onContextMenu === undefined) return;
        event.preventDefault();
        const hit = hitAt(event.clientX, event.clientY);
        const trackId = hit.kind === "clip" ? hit.clip.trackId : hit.trackId;
        interactions.onContextMenu({
            clientX: event.clientX,
            clientY: event.clientY,
            clipIds: trackId === null ? [] : clipsAtPointer(trackId, hit.sec),
            trackId,
            sec: hit.sec,
        });
    }

    /**
     * 收集某轨道上覆盖指定时间点的全部 clip（按 startSec 升序）。
     *
     * 特殊说明：用**闭区间**（`sec <= 右端`）而不是命中测试的半开区间——右键菜单
     * 要列出"指针处有哪些 clip"，相邻紧贴的两个在交界处都该出现（用户可能想选
     * 被压住的那个）。
     *
     * @param trackId 轨道 id。
     * @param sec 工程时间（秒）。
     * @returns clip id 列表。
     */
    function clipsAtPointer(trackId: string, sec: number): string[] {
        ensureHitIndex();
        const list = hitClipsByTrack.get(trackId) ?? [];
        const result: string[] = [];
        for (const clip of list) {
            if (sec >= clip.startSec && sec <= clip.startSec + clip.lengthSec) {
                result.push(clip.id);
            }
        }
        return result;
    }

    /** 指针离开轨道区：清掉自定义光标，交回默认值。 */
    function onPointerLeave(): void {
        if (gesture.kind === "none" && panPointerId === null) container.style.cursor = "";
    }

    container.addEventListener("pointerdown", onPointerDown);
    container.addEventListener("auxclick", onAuxClick);
    container.addEventListener("contextmenu", onContextMenu);
    container.addEventListener("pointermove", updateHoverCursor);
    container.addEventListener("pointerleave", onPointerLeave);
    window.addEventListener("pointermove", onPanPointerMove);
    window.addEventListener("pointermove", onGesturePointerMove);
    window.addEventListener("pointerup", endPan);
    window.addEventListener("pointercancel", endPan);
    window.addEventListener("pointerup", onGesturePointerUp);
    window.addEventListener("pointercancel", onGesturePointerCancel);

    // ── 输入：滚动条拖拽 ─────────────────────────────────────────────
    let dragAxis: "x" | "y" | null = null;
    let dragStartPointer = 0;
    let dragStartScroll = 0;

    /** 造一个 thumb 拖拽的按下处理器。 */
    function makeThumbPointerDown(axis: "x" | "y") {
        return (event: PointerEvent) => {
            event.preventDefault();
            event.stopPropagation();
            dragAxis = axis;
            dragStartPointer = axis === "x" ? event.clientX : event.clientY;
            dragStartScroll = axis === "x" ? scroll.get().scrollLeft : scroll.get().scrollTop;
            (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
        };
    }

    /** 把拖拽位移换算为滚动位置增量并提交。 */
    function applyThumbDrag(pointerDeltaPx: number): void {
        const view = scroll.get();
        if (dragAxis === "x") {
            const geometry = computeScrollbar({
                contentSizePx: view.pxPerSec * Math.max(0, data().projectSec),
                viewportSizePx: viewportWidthPx,
                scrollPx: dragStartScroll,
                maxScrollPx: scroll.maxScrollLeft(),
            });
            scroll.setScrollLeft(
                dragStartScroll +
                    scrollDeltaFromThumbDrag(pointerDeltaPx, geometry, scroll.maxScrollLeft()),
            );
            return;
        }
        const geometry = computeScrollbar({
            contentSizePx: data().tracks.length * view.rowHeight,
            viewportSizePx: viewportHeightPx,
            scrollPx: dragStartScroll,
            maxScrollPx: scroll.maxScrollTop(),
        });
        scroll.setScrollTop(
            dragStartScroll +
                scrollDeltaFromThumbDrag(pointerDeltaPx, geometry, scroll.maxScrollTop()),
        );
    }

    function onPointerMove(event: PointerEvent): void {
        if (dragAxis === null) return;
        applyThumbDrag((dragAxis === "x" ? event.clientX : event.clientY) - dragStartPointer);
    }

    function onPointerUp(): void {
        dragAxis = null;
    }

    const onHDown = makeThumbPointerDown("x");
    const onVDown = makeThumbPointerDown("y");
    hScrollbarThumb.addEventListener("pointerdown", onHDown);
    vScrollbarThumb.addEventListener("pointerdown", onVDown);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);

    // ── 输入：键盘滚动 ───────────────────────────────────────────────
    function onKeyDown(event: KeyboardEvent): void {
        const view = scroll.get();
        switch (event.key) {
            case "PageDown":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft + viewportWidthPx * 0.9);
                break;
            case "PageUp":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft - viewportWidthPx * 0.9);
                break;
            case "Home":
                event.preventDefault();
                scroll.setScrollLeft(0);
                break;
            case "End":
                event.preventDefault();
                scroll.setScrollLeft(scroll.maxScrollLeft());
                break;
            case "ArrowLeft":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft - KEYBOARD_STEP_PX);
                break;
            case "ArrowRight":
                event.preventDefault();
                scroll.setScrollLeft(view.scrollLeft + KEYBOARD_STEP_PX);
                break;
            case "Escape": {
                // 拖拽 / trim 中按 Esc = 取消：回调 cancelled 让调用方回滚乐观值。
                if (gesture.kind === "clip-drag") {
                    event.preventDefault();
                    interactions?.onDragCommit?.({
                        clipId: gesture.clipId,
                        deltaSec: gesture.lastDeltaSec,
                        targetTrackId: gesture.lastTargetTrackId,
                        cancelled: true,
                    });
                    gesture = { kind: "none" };
                    break;
                }
                if (gesture.kind === "clip-trim") {
                    event.preventDefault();
                    interactions?.onTrimCommit?.({
                        clipId: gesture.clipId,
                        edge: gesture.edge,
                        startSec: gesture.lastStartSec,
                        lengthSec: gesture.lastLengthSec,
                        cancelled: true,
                    });
                    gesture = { kind: "none" };
                    break;
                }
                if (gesture.kind === "clip-fade") {
                    event.preventDefault();
                    interactions?.onFadeCommit?.({
                        clipId: gesture.clipId,
                        side: gesture.side,
                        fadeSec: gesture.lastFadeSec,
                        cancelled: true,
                    });
                    gesture = { kind: "none" };
                }
                break;
            }
            default:
                break;
        }
    }
    container.addEventListener("keydown", onKeyDown);

    // ── 尺寸与 DPR 变化 ──────────────────────────────────────────────
    const resizeObserver = new ResizeObserver((entries) => {
        let widthChanged = false;
        for (const entry of entries) {
            const nextWidth = Math.max(1, entry.contentRect.width);
            if (nextWidth !== viewportWidthPx) widthChanged = true;
            viewportWidthPx = nextWidth;
            viewportHeightPx = Math.max(1, entry.contentRect.height);
        }
        // 宽度变化会改变标尺的刻度窗口（React 侧按它算 ticks）。
        if (widthChanged) onViewportWidthChange?.(viewportWidthPx);
        // 外部边界变化后必须重新钳制（见 ScrollKernel 的约束 1）。
        scroll.reclamp();
        sceneDirty = true;
        loop.invalidate();
    });
    resizeObserver.observe(container);
    // 初始宽度回写：ResizeObserver 的首次回调是异步的，而标尺在首帧就需要正确的
    // 刻度窗口（否则首屏标尺只画得出前一段）。
    onViewportWidthChange?.(viewportWidthPx);

    function onWindowResize(): void {
        sceneDirty = true;
        loop.invalidate();
    }
    window.addEventListener("resize", onWindowResize);

    // 首次标脏（尺寸与数据就绪后绘制第一帧）。
    loop.invalidate();

    return {
        invalidateScene() {
            sceneDirty = true;
            loop.invalidate();
        },

        setScrollTop(px: number) {
            // 由 ScrollKernel 统一钳制；值未变化时不会通知订阅者（因此不会空重绘）。
            scroll.setScrollTop(px);
        },

        getViewport() {
            const view = scroll.get();
            return {
                scrollLeft: view.scrollLeft,
                scrollTop: view.scrollTop,
                pxPerSec: view.pxPerSec,
            };
        },

        getAxis() {
            return currentAxis();
        },

        registerViewportLayer(layer, order) {
            viewportLayers.push({ name: layer.name, paint: layer.paint, order });
            viewportLayers.sort((a, b) => a.order - b.order);
            return () => {
                const index = viewportLayers.findIndex((item) => item.name === layer.name);
                if (index >= 0) viewportLayers.splice(index, 1);
            };
        },

        dispose() {
            unsubscribeScroll();
            loop.stop();
            resizeObserver.disconnect();
            container.removeEventListener("wheel", onWheel);
            container.removeEventListener("keydown", onKeyDown);
            container.removeEventListener("pointerdown", onPointerDown);
            container.removeEventListener("auxclick", onAuxClick);
            container.removeEventListener("contextmenu", onContextMenu);
            container.removeEventListener("pointermove", updateHoverCursor);
            container.removeEventListener("pointerleave", onPointerLeave);
            boxSelectEl.remove();
            window.removeEventListener("pointermove", onGesturePointerMove);
            window.removeEventListener("pointerup", onGesturePointerUp);
            window.removeEventListener("pointercancel", onGesturePointerCancel);
            hScrollbarThumb.removeEventListener("pointerdown", onHDown);
            vScrollbarThumb.removeEventListener("pointerdown", onVDown);
            window.removeEventListener("pointermove", onPointerMove);
            window.removeEventListener("pointerup", onPointerUp);
            window.removeEventListener("pointermove", onPanPointerMove);
            window.removeEventListener("pointerup", endPan);
            window.removeEventListener("pointercancel", endPan);
            window.removeEventListener("resize", onWindowResize);
            // 平移中途卸载：光标 / 选择态是写在 body 上的全局状态，必须复原。
            endPan();
            detailCanvas.remove();
            sdfBox.dispose();
            glCanvas.dispose();
        },
    };
}
