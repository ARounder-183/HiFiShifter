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
import { parseRgbaColor, type GlClipBodySink } from "../../runtime/timelineClipGlRenderer";
import {
    buildTimelineClipVisualStyle,
    resolveFontFamily,
    resolveThemeColor,
} from "../../runtime/timelineCanvasStyle";
import { hitClipHeaderControl, type ClipHeaderControl } from "../interaction/clipHeaderControls";
import { isFadeShapeCycleModifierHeld } from "../../fadeShapeCycle";
import { noteFadeLinePointerDown } from "../../hooks/fadeLineClickGesture";
import { effectiveFadeSec, hitClipFadeTarget } from "../interaction/fadeTargets";
import type { FadeContextSide } from "../../FadeContextMenu";
import { hitOverlapControl } from "../interaction/overlapControls";
import { resolveHorizontalWheelZoom } from "../../runtime/timelineScrollRange";
import { resolveTimelineMinPxPerSec } from "../../runtime/timelineZoomBounds";
import {
    CLIP_HEADER_HEIGHT,
    DEFAULT_PX_PER_SEC,
    DEFAULT_ROW_HEIGHT,
    MAX_PX_PER_SEC,
    MAX_ROW_HEIGHT,
    MIN_CLIP_LENGTH_SEC,
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
import { clipIntersectsBox, resolveBoxBounds, type BoxBounds } from "../interaction/boxSelection";
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
        /**
         * 淡变形状循环（`modifier.fadeShapeCycleClick`）。
         *
         * 修饰键 + 单击包络线 = 循环切换形状并重置曲率；拖动超过阈值仍是改长度。
         */
        readonly fadeShapeCycle: Keybinding | null;
    };
    /** 水平缩放是否以播放头为锚点（`playheadZoomEnabled`）。 */
    readonly playheadZoomEnabled: boolean;
    /** 初始水平缩放（CSS px/秒）：仅用于首次创建滚动内核。 */
    readonly initialPxPerSec: number;
    /** 单选焦点 clip（选中样式与旧实现同源，由样式模块消费）。 */
    readonly selectedClipId: string | null;
    /** 多选集合（与 `selectedClipId` 一起决定描边 / 高亮）。 */
    readonly multiSelectedClipIds: readonly string[];
    /**
     * 激活 / 禁用的分组 id。
     *
     * 供 header 控件级命中构造 `buildTimelineClipVisualStyle`：分组的激活 / 禁用
     * 状态会改变链徽标的可见性与配色，而**可见性决定偏移量**（静音徽标的 x 依赖
     * 链徽标是否显示）——不传会让命中区与绘制区错位。
     */
    readonly activeGroupIds: readonly string[];
    readonly disabledGroupIds: readonly string[];
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
    /**
     * 吸附高亮内容层（`SnapHighlightLayer` 的容器）：整层写
     * `translate(-scrollLeft, -scrollTop)`。
     *
     * 【为什么是"内容层 + 整层平移"】吸附竖线在组件内部用**内容坐标**布局
     * （`marker.sec × pxPerSec`，与旧实现一致——旧实现把它放在原生滚动的内容层里
     * 靠滚动平移）。内核自绘滚动后没有内容层，这里补一个：层内布局不变，滚动只写
     * 一次 transform，因此拖拽期间每帧重渲染的高亮层不会因滚动而重排。
     */
    readonly snapHighlightContent?: HTMLElement | null;
    /**
     * copy 拖拽 ghost 的内容层：整层写 `translate(-scrollLeft, -scrollTop)`。
     *
     * 与 `snapHighlightContent` 同一机制（内容坐标 + 整层平移）——ghost 的位置由
     * 面板按内容坐标算好，滚动时只平移容器，**层内不重排**。copy 模式下原 clip
     * 不动，没有"乐观位置"可依赖，因此必须有这一层。
     */
    readonly ghostContent?: HTMLElement | null;
    /**
     * 素材拖入预览的内容层：与 `ghostContent` 同一机制（内容坐标 + 整层平移）。
     *
     * 旧实现把它渲染在 `TrackLane` 内（内核模式下不挂载），几何用内容坐标
     * （`startSec × pxPerSec`），因此迁过来后同样需要这一层。
     */
    readonly dropPreviewContent?: HTMLElement | null;
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
/**
 * 拖拽期间的修饰键快照。
 *
 * 形状与 `isModifierActive` 的 `event` 参数一致，因此调用方可以把它直接喂给
 * `isModifierActive(kb, modifiers)`，无需伪造 DOM 事件。
 */
export interface KernelDragModifiers {
    readonly ctrlKey: boolean;
    readonly shiftKey: boolean;
    readonly altKey: boolean;
    readonly metaKey: boolean;
}

export interface TimelineKernelInteractions {
    /**
     * 请求跳转播放头（点击或拖拽空白 / 标尺）。
     *
     * @param sec 目标时间（秒，已钳制到 >= 0）。
     * @param commit true = 单击或手势结束（应提交后端）；false = 拖拽中的预览。
     * @param trackId 点击空白时指针所在轨道（拖拽预览帧与标尺来源不带）。
     *   供面板实现「空白点击」的完整语义：清空 clip 选中 + 按
     *   `允许时间轴点击切换轨道` 切换当前轨道（旧实现 `TimelinePanel` 的
     *   pointerdown 捕获分支）。**不参与 seek 本身**。
     */
    readonly onSeek?: (sec: number, commit: boolean, trackId?: string | null) => void;
    /**
     * 选中 clip。
     *
     * @param clipId 目标 clip；null 表示清空选择（点击空白走 `onSeek` 的
     *   `trackId` 参数，不经这里）。
     * @param additive true = 按住多选修饰键（Ctrl / ⌘），应切换而非替换选择。
     */
    readonly onSelectClip?: (clipId: string, additive: boolean) => void;
    /**
     * clip 左键按下拦截（在**任何**内核手势判定之前回调）。
     *
     * 【用途】把「修饰键 + 竖直拖 = 调音高」这类**由面板自持状态机**的手势整体
     * 交回面板——它复用旧实现的 `useClipPitchDrag`（自带 window 监听、参数帧预览、
     * undo group 与收尾），内核只负责识别「按在了 clip 上」并提供几何。
     *
     * 特殊说明：返回 true 表示已接管，内核**不得**再启动选中 / 拖拽 / 控件分派。
     *
     * @param args 命中 clip、指针几何、修饰键快照与内核容器（供接管方做 pointer
     *   capture 或坐标换算）。
     * @returns true = 面板已接管本次按下。
     */
    readonly onClipPointerDownIntercept?: (args: {
        readonly clipId: string;
        readonly clientX: number;
        readonly clientY: number;
        readonly pointerId: number;
        readonly modifiers: KernelDragModifiers;
        readonly container: HTMLElement;
    }) => boolean;
    /**
     * 双击 clip（第二次按下命中，且两次之间未发生拖拽）。
     *
     * 旧实现语义：请求参数编辑器按 clip 起止范围创建选区，并把交互焦点切到
     * 参数编辑器（见 `ClipItem` 的 `hifi:editOp/selectClipParamRange`）。
     * 内核只负责识别手势，事件派发与焦点切换由面板完成。
     */
    readonly onDoubleClickClip?: (clipId: string) => void;
    /**
     * 切换 clip 静音（单击 header 的静音徽标）。
     *
     * @param clipId 目标 clip。
     * @param nextMuted 目标状态（取反后的值，与旧实现一致：旧实现传 `!clip.muted`）。
     */
    readonly onToggleClipMute?: (clipId: string, nextMuted: boolean) => void;
    /**
     * 打开共振峰工具窗口（单击 header 的 F 徽标）。
     *
     * @param clipId 目标 clip。
     * @param screenX 浮窗锚点的视口坐标 x（旧实现取按钮右缘 + 12）。
     * @param screenY 浮窗锚点的视口坐标 y（旧实现取按钮上缘）。
     */
    readonly onOpenClipFormant?: (clipId: string, screenX: number, screenY: number) => void;
    /**
     * 临时禁用 / 启用编组的联动编辑（单击 header 的锁链徽标）。
     *
     * 特殊说明：作用于**整个编组**（不是单个 clip）——面板复用既有的
     * `toggleGroupDisabled`，与旧实现同一份语义。
     *
     * @param groupId 目标编组。
     */
    readonly onToggleGroupDisabled?: (groupId: string) => void;
    /**
     * 开始行内编辑 clip 的增益 / 速率（单击对应标签或增益旋钮）。
     *
     * @param clipId 目标 clip。
     * @param field 编辑字段。
     * @param screenX 输入框锚点的视口坐标。
     * @param screenY 输入框锚点的视口坐标。
     */
    readonly onBadgeEditStart?: (
        clipId: string,
        field: "gain" | "rate",
        screenX: number,
        screenY: number,
    ) => void;
    /**
     * 音量旋钮拖拽预览（按住旋钮上下拖动时每帧回调，已按值去重）。
     *
     * 【为什么只给位移】dB 换算（`CLIP_GAIN_DRAG_DB_PER_PX`）、精细调整修饰键、
     * ±12dB 钳制与批量应用都在面板侧——与 `onDragPreview` 同一架构约定。
     *
     * @param args `deltaYPx` 是相对按下点的竖直位移（向下为正，未乘任何系数）。
     */
    readonly onGainDragPreview?: (args: {
        readonly clipId: string;
        readonly deltaYPx: number;
        readonly modifiers: KernelDragModifiers;
    }) => void;
    /**
     * 音量旋钮拖拽结束。
     *
     * @param args `changed = false` 表示未越起手阈值的单击——调用方**不应**写后端；
     *   `cancelled = true`（Esc / pointercancel）时应回滚到按下时的增益。
     */
    readonly onGainDragCommit?: (args: {
        readonly clipId: string;
        readonly changed: boolean;
        readonly cancelled: boolean;
    }) => void;
    /**
     * 双击音量旋钮 = 恢复 0 dB（手册：「双击恢复为 0 dB」）。
     *
     * @param clipId 目标 clip。
     */
    readonly onGainReset?: (clipId: string) => void;
    /**
     * 打开速率高级编辑（**右键**速率标签）。
     *
     * @param clipId 目标 clip。
     * @param screenX 菜单锚点的视口坐标。
     * @param screenY 菜单锚点的视口坐标。
     */
    readonly onRateBadgeMenu?: (clipId: string, screenX: number, screenY: number) => void;
    /**
     * 请求进入 clip 重命名（**双击名称区**）。
     *
     * 与 `onDoubleClickClip`（双击其他区域 → 参数编辑器选区）互斥：名称区优先。
     * 旧实现里名称区的处理器会 `stopPropagation`，两者天然不会同时触发。
     *
     * @param clipId 目标 clip。
     * @param screenX 输入框锚点的视口坐标。
     * @param screenY 输入框锚点的视口坐标。
     */
    readonly onRenameClipStart?: (clipId: string, screenX: number, screenY: number) => void;
    /**
     * 交叉点抓手拖拽预览（拖拽期间每帧，**不经 React 渲染**）。
     *
     * 语义：同时把前一个 clip 的右缘与后一个 clip 的左缘按同一位移移动，重叠长度
     * 不变（旧实现 `crossfade_edges`）。调用方据此写入两个 clip 的乐观几何。
     *
     * @param args 两侧 clip 与本次位移（秒）。
     */
    readonly onCrossfadeGripPreview?: (args: {
        readonly earlierClipId: string;
        readonly laterClipId: string;
        readonly deltaSec: number;
    }) => void;
    /**
     * 交叉点抓手收尾。
     *
     * @param args 两侧 clip、最终位移与是否取消（Esc / pointercancel）。
     */
    readonly onCrossfadeGripCommit?: (args: {
        readonly earlierClipId: string;
        readonly laterClipId: string;
        readonly deltaSec: number;
        readonly cancelled: boolean;
    }) => void;
    /**
     * 淡变形状循环（循环修饰键 + 单击包络线，未拖动）。
     *
     * @param clipId 目标 clip。
     * @param side 淡入还是淡出。
     */
    readonly onFadeShapeCycle?: (clipId: string, side: "in" | "out") => void;
    /**
     * 交叉点抓手上的循环点击：同时切换两侧的形状。
     *
     * @param sides 两侧（前一个 clip 的淡出 + 后一个 clip 的淡入）。
     */
    readonly onCrossfadeCycle?: (sides: Array<{ clipId: string; isOut: boolean }>) => void;
    /**
     * 重置淡变曲率（双击包络线本体 / 交叉点抓手）。
     *
     * @param sides 需要重置的侧。
     */
    readonly onResetFadeCurvature?: (sides: Array<{ clipId: string; isOut: boolean }>) => void;
    /**
     * 拖拽预览（拖拽期间每帧回调，**不经 React 渲染**）。
     *
     * 语义：调用方据此写入**乐观位置**（Redux）。移动语义下内核不做独立的 ghost
     * 图层——乐观更新会改变 `clips` 引用，内核在下一帧重建几何时把 clip 画在新位置，
     * 视觉上即"跟着指针走"。这样只有一份位置真值，不会出现 ghost 与实体分叉。
     *
     * **copy 模式例外**：原 clip 不动，因此没有"乐观位置"可依赖，必须由调用方给出
     * ghost（见 `TimelineKernelView` 的 `ghost` prop）。判定归调用方（键位绑定在
     * 面板侧），内核只透传 `modifiers` 快照。
     *
     * @param args 目标位置（已含边界钳制）；`targetTrackId` 为落点轨道。
     */
    readonly onDragPreview?: (args: {
        readonly clipId: string;
        readonly deltaSec: number;
        readonly targetTrackId: string;
        /**
         * 修饰键快照。
         *
         * 【为什么由内核给、而不是面板自己读事件】拖拽期间没有 DOM 事件穿过面板，
         * 面板拿不到当前修饰键。内核只做**快照透传**——「哪个键是复制」由键位绑定
         * 决定（`resolveClipDragCopyMode`），那是面板侧的配置，内核不解释。
         */
        readonly modifiers: KernelDragModifiers;
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
        /** 收尾时的修饰键快照（见 `onDragPreview`）。 */
        readonly modifiers: KernelDragModifiers;
    }) => void;
    /**
     * trim 预览（拖拽左右边缘时每帧回调，已按值去重）。
     *
     * @param args 新的起始时间与长度（已钳制）；`deltaSec` 是实际生效的变化量
     *   （左边缘 = `startSec` 的变化、右边缘 = `lengthSec` 的变化）；
     *   `modifiers` 是当帧修饰键快照（供面板做「免吸附」这类按键语义——内核
     *   不解释按键含义，见 `onDragPreview` 的同款约定）。
     */
    readonly onTrimPreview?: (args: {
        readonly clipId: string;
        readonly edge: TrimEdge;
        readonly startSec: number;
        readonly lengthSec: number;
        readonly deltaSec: number;
        readonly modifiers: KernelDragModifiers;
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
     * 吸附偏移拖拽预览（拖拽期间每次位移回调，已按值去重）。
     *
     * 【为什么只给几何】吸附规则（网格 / 其他 clip 边缘与偏移 / 播放光标）与
     * 高亮发布都在面板侧（`snapTimelineDetailed`）。内核不复制这套规则——与
     * `onDragPreview` 同一架构约定：内核只做手势，语义交回面板。
     *
     * @param args `rawOffsetSec` 是**未吸附**的目标偏移，可能为负或超过 clip
     *             长度；调用方负责吸附与钳制后再写入。`modifiers` 为当帧修饰键
     *             快照（免吸附判定用）。
     */
    readonly onSnapOffsetPreview?: (args: {
        readonly clipId: string;
        readonly rawOffsetSec: number;
        readonly modifiers: KernelDragModifiers;
    }) => void;
    /**
     * 吸附偏移拖拽结束。
     *
     * @param args `cancelled = true` 时调用方应回滚；`changed = false` 表示
     *             零位移单击——调用方**不应**写后端（与旧实现同一语义：单击不
     *             产生 undo 步）。最终值由调用方从自身状态读取（拖拽期间已乐观
     *             收敛），内核不代传，避免两份"最终值"。
     */
    readonly onSnapOffsetCommit?: (args: {
        readonly clipId: string;
        readonly cancelled: boolean;
        readonly changed: boolean;
    }) => void;
    /**
     * 淡变角预览（拖拽角部时每帧回调，已按值去重）。
     *
     * @param args 新的淡变长度（已钳制到 `[0, clip 长度]`）；`modifiers` 为当帧
     *   修饰键快照（免吸附判定用）。
     */
    readonly onFadePreview?: (args: {
        readonly clipId: string;
        readonly side: FadeSide;
        readonly fadeSec: number;
        readonly deltaSec: number;
        readonly modifiers: KernelDragModifiers;
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
    /**
     * 淡变包络 / 交叉点抓手的**专属右键菜单**请求。
     *
     * 【为什么单独一条通道】旧实现里淡变菜单由三层 DOM 命中块
     * （`ClipItem` 角部、`FadeHitLayer`、`OverlapEditLayer`）各自发起，它们在内核
     * 模式下都不挂载；而菜单宿主（`FadeContextMenuHost`）挂在面板上、经全局总线
     * 接收请求——因此内核只负责命中解析与载荷构造，不直接弹菜单。
     *
     * @param request `primary` 是右键命中的那一侧包络；交叉点抓手时
     *   `primary` = 前一个 clip 的淡出、`secondary` = 后一个 clip 的淡入
     *   （与旧实现 `crossfadeSides.out / .in` 的列序一致）。
     */
    readonly onFadeContextMenu?: (request: {
        readonly clientX: number;
        readonly clientY: number;
        readonly primary: FadeContextSide;
        readonly secondary: FadeContextSide | null;
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
    /**
     * 外部请求横向滚动（自动滚屏 / 聚焦播放光标 / 参数编辑器同步）。
     *
     * 特殊说明：与 `setScrollTop` 同一约定——钳制只在 `ScrollKernel` 内做一次；
     * 调用方若需要「浏览器实际接受的值」语义，请在写入后回读 `getViewport()`。
     */
    setScrollLeft(px: number): void;
    /**
     * 原子地设置缩放与横向滚动位置（键盘缩放 / 视图同步）。
     *
     * 特殊说明：两个字段必须一次提交，否则会出现「用旧上限钳制新缩放」的中间态
     * （缩放放大时横向位置会被错误地卡在旧上限）。pxPerSec 真正变化时会回调
     * `onZoomChange`，调用方无需自己同步 React 侧派生量。
     *
     * @param next 需要覆盖的字段；未给出的沿用当前值。
     * @returns 提交后的真值（已钳制）。
     */
    setViewport(next: { pxPerSec?: number; scrollLeft?: number }): {
        scrollLeft: number;
        pxPerSec: number;
    };
    /**
     * 读取宿主容器的视口矩形（client 坐标）。
     *
     * 供「clientX/clientY → 工程时间 / 轨道」的换算使用（素材拖入落点、右键命中）。
     * 旧实现对应的是原生 scroller 的 `getBoundingClientRect()`；自绘滚动后容器
     * 仍是同一个 DOM（内核容器），但滚动量不再由浏览器维护，必须与
     * `getViewport()` 的 scrollLeft/Top 配套使用。
     *
     * 特殊说明：本方法会触发布局读取，只允许在**低频事件**（拖放 / 右键）中调用，
     * 不得进入每帧路径。
     *
     * @returns 容器矩形；容器已卸载时为 null。
     */
    getContainerRect(): DOMRect | null;
    /**
     * 读取当前视口状态（供外部低频读取，例如保存 / 调试）。
     *
     * `viewportWidth` / `viewportHeight` 是 `ResizeObserver` 缓存的量测值
     * （O(1)、不触发布局），供外部换算「视口宽度」（自动滚屏 / 缩放锚点 /
     * 拖入落点）——**不要**改用 `getContainerRect()`，那会在每帧路径上强制
     * 样式重算。
     */
    getViewport(): {
        scrollLeft: number;
        scrollTop: number;
        pxPerSec: number;
        viewportWidth: number;
        viewportHeight: number;
    };
    /**
     * 读取当前坐标投影（内容坐标 ↔ 视口坐标）。
     *
     * 供需要与内核视口同步的**独立画布**图层使用（例如波形面：它自带 WebGL2
     * 上下文与几何缓存，只需要一个与内核同源的 axis 即可零重建跟随滚动）。
     */
    getAxis(): TimelineAxis;
    /**
     * 读取 clip header 上某控件的锚点（**内容坐标** + 建议宽度）。
     *
     * 供内核态行内编辑浮层定位输入框。返回内容坐标而不是屏幕坐标：浮层需要随
     * 滚动更新位置，只有拿到内容坐标才能与视口量（`scrollLeft/Top`）相减重算。
     *
     * @param clipId 目标 clip。
     * @param field 编辑字段（决定锚点落在名称区还是标签处）。
     * @returns 锚点；clip 或轨道不存在时为 null。
     */
    getClipHeaderAnchor(
        clipId: string,
        field: "name" | "gain" | "rate",
    ): {
        readonly contentLeftPx: number;
        readonly contentTopPx: number;
        readonly widthPx: number;
    } | null;
    /**
     * 调试用：报告某屏幕坐标的命中结果与 header 控件判定。
     *
     * 【为什么放进公开句柄】内核是自绘的，命中区无法从 DOM 观察。排查「点不准」
     * 类问题（命中区与绘制区错位、控件分派未触发）必须能读到几何判定的中间结果，
     * 否则只能靠反复试点击猜坐标。配合 `window.__hfsKernel` 使用。
     *
     * @param clientX 屏幕坐标 x。@param clientY 屏幕坐标 y。
     * @returns 命中分区、clip 局部坐标与控件判定（无命中时相应字段缺省）。
     */
    debugHitAt(
        clientX: number,
        clientY: number,
    ): {
        readonly kind: string;
        readonly region?: string;
        readonly control?: string;
        readonly clipId?: string;
        readonly localX?: number;
        readonly localY?: number;
        readonly sec?: number;
    };
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
    /**
     * 细节层用的编组状态（与几何同一次重建时更新）。
     *
     * 【为什么必须传】`drawTimelineCanvas` 需要它才能画出：
     * - 编组激活的**深金外圈描边**（`activeGroupIds`）；
     * - 锁链徽标的「已禁用联动」配色（`disabledGroupIds`）。
     * 不传时描边与徽标状态永远停在"未激活 / 未禁用"——表现为「点了锁链没有任何反馈」。
     */
    let detailActiveGroupIds: ReadonlySet<string> = new Set<string>();
    let detailDisabledGroupIds: readonly string[] = [];
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
            // 编组状态：描边与锁链徽标配色都由它决定（见 `detailActiveGroupIds`）。
            activeGroupIds: detailActiveGroupIds as Set<string>,
            disabledGroupIds: [...detailDisabledGroupIds],
            // 块面归 GL（细节层在波形之上，画块面会盖住波形）；这里只补那一圈描边。
            groupOutlineOverGl: true,
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
        pxPerSec:
            clampNumber(data().initialPxPerSec, MIN_PX_PER_SEC, MAX_PX_PER_SEC) ||
            DEFAULT_PX_PER_SEC,
        // 行高与左侧轨道头同源（data().rowHeight），否则左右两列行错位。
        rowHeight: Math.max(1, data().rowHeight || DEFAULT_ROW_HEIGHT),
        projectSec: () => Math.max(0, data().projectSec),
        trackCount: () => data().tracks.length,
        viewportHeightPx: () => viewportHeightPx,
        // 下限必须与滚轮缩放解析（resolveHorizontalWheelZoom 的 minPxPerSec）
        // **同源且动态**：工程短于视口时解析会允许缩到 0.5，若内核仍按固定常量
        // 钳制，就会出现「标尺能缩得更小而网格缩不动」。
        minPxPerSec: () =>
            resolveTimelineMinPxPerSec({
                baseMinPxPerSec: MIN_PX_PER_SEC,
                projectSec: data().projectSec,
                viewportWidthPx,
            }),
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
            // 编组状态参与两件事：overlay 展开（激活编组的成员一并进 DOM 覆盖层）
            // 与样式（锁链徽标配色）。漏传会让"点锁链禁用联动"没有任何视觉反馈。
            disabledGroupIds: [...d.disabledGroupIds],
        });

        const clipResult = clipBuilder.build({
            clips: model.drawClips,
            darkMode: d.darkMode,
            fontFamily: resolveFontFamily(),
            seamColor: d.darkMode ? "rgb(31, 31, 31)" : "rgb(237, 240, 245)",
            // 编组状态进 GL 样式：激活编组的块面描边/徽标配色与旧实现同源。
            activeGroupIds: model.activeGroupIds,
            disabledGroupIds: [...d.disabledGroupIds],
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
        detailActiveGroupIds = model.activeGroupIds;
        detailDisabledGroupIds = d.disabledGroupIds;
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
    /** 吸附高亮内容层的整层变换（字符串去重：同时含两轴）。 */
    let lastSnapTransform = "";
    /** copy ghost 内容层的整层变换（去重方式同上）。 */
    let lastGhostTransform = "";
    /** 拖入预览内容层的整层变换（去重方式同上）。 */
    let lastDropPreviewTransform = "";
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
                // 吸附偏移：**`hitTest` 会消费它**（SnapOffset 手柄的命中区左缘
                // 跟随该值），与下方纯透传的业务字段不同，必须随几何一起更新。
                snapOffsetSec: clip.snapOffsetSec,
                // 以下字段 `hitTest` 本身不消费，只做透传：header 控件级命中需要
                // 它们构造 `buildTimelineClipVisualStyle`（与绘制端同一份样式）。
                muted: clip.muted,
                gain: clip.gain,
                playbackRate: clip.playbackRate,
                name: clip.name,
                groupId: clip.groupId,
                isMidiClip: clip.midiNoteCount != null,
                fadeInSec: clip.fadeInSec,
                autoFadeInSec: clip.autoFadeInSec,
                fadeInShape: clip.fadeInShape,
                fadeInDir: clip.fadeInDir,
                fadeOutSec: clip.fadeOutSec,
                autoFadeOutSec: clip.autoFadeOutSec,
                fadeOutShape: clip.fadeOutShape,
                fadeOutDir: clip.fadeOutDir,
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

        // 吸附高亮内容层：整层平移（内容坐标 → 视口坐标）。
        // 与细节层同一策略——层内元素全用内容坐标布局，滚动只写一次 transform。
        // 用字符串去重（同时含两轴，天然规避 NaN 初值问题）。
        const snapContent = sync.snapHighlightContent;
        if (snapContent != null) {
            const transform = `translate(${-view.scrollLeft}px, ${-view.scrollTop}px)`;
            if (transform !== lastSnapTransform) {
                lastSnapTransform = transform;
                snapContent.style.transform = transform;
            }
        }

        // copy ghost 内容层：与吸附高亮同一机制（内容坐标 + 整层平移）。
        const ghostContent = sync.ghostContent;
        if (ghostContent != null) {
            const transform = `translate(${-view.scrollLeft}px, ${-view.scrollTop}px)`;
            if (transform !== lastGhostTransform) {
                lastGhostTransform = transform;
                ghostContent.style.transform = transform;
            }
        }

        // 拖入预览内容层：同上。
        const dropPreviewContent = sync.dropPreviewContent;
        if (dropPreviewContent != null) {
            const transform = `translate(${-view.scrollLeft}px, ${-view.scrollTop}px)`;
            if (transform !== lastDropPreviewTransform) {
                lastDropPreviewTransform = transform;
                dropPreviewContent.style.transform = transform;
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
    function scrollbarZoneAt(
        clientX: number,
        clientY: number,
        rect: DOMRect,
    ): ScrollbarZone | null {
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
        // 用**实际生效值**回调 React：下限随工程长度变化，钳制可能改变请求值；
        // 直接回传请求值会让标尺（React 侧派生量）与网格（内核真值）分叉。
        const appliedPxPerSec = scroll.setZoom(zoom.nextPxPerSec, event.clientX - rect.left);
        scroll.setScrollLeft(zoom.nextScrollLeft);
        onZoomChange?.(appliedPxPerSec);
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
              /** 按下时的吸附偏移（秒；非 snap 手柄为 0）。 */
              originSnapOffsetSec: number;
              /**
               * 交叉点抓手的前一个 clip id。
               *
               * 仅 `region === "crossfade-grip"` 时有值：命中结果只给出后一个
               * clip（重叠区里二分的自然结果），而抓手需要同时操作两侧。
               */
              partnerClipId?: string;
              /**
               * 按下时循环修饰键是否按住（仅淡变控件与交叉点抓手会置位）。
               *
               * 收尾时若仍未升级（= 未超过拖拽阈值）→ 视为「循环点击」而不是选中，
               * 与旧实现的「延后判定用户意图」一致：拖动 = 改长度，未拖动 = 循环。
               */
              cycleHeld?: boolean;
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
              kind: "crossfade-grip";
              /** 前一个 clip：右缘随拖拽移动。 */
              earlierClipId: string;
              /** 后一个 clip：左缘随拖拽移动。 */
              laterClipId: string;
              startContentX: number;
              /** 上一次预览的位移（去重：相同位移不重复派发）。 */
              lastDeltaSec: number;
          }
        | {
              kind: "snap-offset-drag";
              clipId: string;
              startContentX: number;
              /** 按下时的吸附偏移（秒）。 */
              originOffsetSec: number;
              /** clip 长度（面板用它钳制；内核只给几何）。 */
              lengthSec: number;
              /**
               * 最近一次派发的**未吸附**偏移（去重）。
               *
               * 初值 `NaN`：收尾时据此判断是否发生过真实位移——零位移单击
               * 不应写后端（与旧实现 `checkpointed` 标志同一语义）。
               */
              lastOffsetSec: number;
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
        | {
              kind: "gain-drag";
              clipId: string;
              startClientX: number;
              startClientY: number;
              /** 最近一次派发的竖直位移（CSS px，向下为正）。 */
              lastDeltaY: number;
              /**
               * 是否已越过起手阈值。
               *
               * 旧实现的音量旋钮有 3px 起手阈值：阈值内的按下/抬起是**单击**，
               * 不产生任何编辑（只有拖动才改增益）。收尾据此决定是否提交。
               */
              started: boolean;
          }
        | { kind: "seek" };
    let gesture: Gesture = { kind: "none" };

    /** 点击 / 拖拽的位移阈值（CSS px）。 */
    const DRAG_THRESHOLD_PX = 4;

    /**
     * 音量旋钮的起手阈值（CSS px）。
     *
     * 与旧实现 `ClipHeader` 一致（3px）：阈值内的按下/抬起是单击，不产生任何编辑。
     */
    const GAIN_DRAG_THRESHOLD_PX = 3;

    /**
     * 双击判定参数（与旧实现 `ClipItem` 完全一致）。
     *
     * 用手动判定而不是原生 `dblclick`：clip 的 `pointerdown` 会 `preventDefault`，
     * 原生 dblclick 不可靠；且第一次按下后若发生拖拽，待定记录必须失效，否则
     * "拖拽后落点回位"会被误判成双击。
     */
    const DOUBLE_CLICK_MOVE_PX = 6;
    const DOUBLE_CLICK_INTERVAL_MS = 500;

    /** 上一次左键按下（clip 命中时记录，用于双击判定）。 */
    let lastClipPress: {
        pointerId: number;
        clientX: number;
        clientY: number;
        time: number;
    } | null = null;

    /** trim 允许的最小 clip 长度（秒）：再短会难以命中与选中。 */

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
     * 「待框选」手势期间被抑制的 contextmenu 位置。
     *
     * 松手时若框选**未成立**（右键单击）则在该位置补发菜单；若框选成立则丢弃。
     */
    let pendingContextMenu: { clientX: number; clientY: number } | null = null;

    /**
     * 把视口内坐标换算为命中结果（内容坐标下的轨道 + clip）。
     *
     * @param clientX 指针视口坐标 X。
     * @param clientY 指针视口坐标 Y。
     * @returns 命中结果。
     */
    function hitAt(clientX: number, clientY: number): KernelHit {
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        ensureHitIndex();
        const contentX = view.scrollLeft + (clientX - rect.left);
        const contentY = view.scrollTop + (clientY - rect.top);
        const hit = hitTest({
            contentX,
            contentY,
            pxPerSec: view.pxPerSec,
            rowHeight: view.rowHeight,
            tracks: hitTracks,
            clipsByTrack: hitClipsByTrack,
            headerHeightPx: CLIP_HEADER_HEIGHT,
        });
        if (hit.kind !== "clip") return hit;

        // ── 重叠区按位置改写 ──
        // 二分取到的是「最后一个 startSec <= sec」的 clip，在重叠区里永远是**后
        // 一个**；前一个 clip 的右缘与淡出控件因此完全不可达。这里按旧实现
        // `OverlapEditLayer` 的位置规则改写命中结果（见 overlapControls 文件头）。
        const track = hitTracks[hit.trackIndex];
        if (track === undefined) return hit;
        const trackClips = hitClipsByTrack.get(track.id);
        if (trackClips === undefined || trackClips.length < 2) return hit;
        const overlap = hitOverlapControl({
            clips: trackClips,
            contentX,
            localY: hit.localY,
            pxPerSec: view.pxPerSec,
            rowHeight: view.rowHeight,
        });
        if (overlap === null) {
            // ── 淡变包络线 / 区域边缘竖线（非重叠情形）──
            // 旧实现由 `ClipItem` 内的 `FadeHitLayer` 提供「画线即控件」，沿包络线
            // 任意位置都能抓住调长度；内核原先只有角部小方块能抓，长淡变的曲线
            // 中段完全抓不到。
            //
            // 只在 `body` 分区检查：header 有自己的控件；clip 边缘与角部已在
            // `hitTest` 内判过——旧实现的优先级是「clip 边缘 > 淡变边缘线 > 包络线」，
            // 把淡变放在边缘之后正好吻合。
            if (hit.region === "body") {
                const fade = hitClipFadeTarget({
                    clip: hit.clip,
                    clipLeftPx: hit.clip.startSec * view.pxPerSec,
                    clipWidthPx: Math.max(1, hit.clip.lengthSec * view.pxPerSec),
                    contentX,
                    localY: hit.localY,
                    pxPerSec: view.pxPerSec,
                    rowHeight: view.rowHeight,
                });
                if (fade !== null) {
                    return {
                        kind: "clip",
                        clip: hit.clip,
                        region: fade.side === "out" ? "fade-out-corner" : "fade-in-corner",
                        sec: hit.sec,
                        trackIndex: hit.trackIndex,
                        localX: hit.localX,
                        localY: hit.localY,
                        fadeIsLine: fade.kind === "line",
                    };
                }
            }
            return hit;
        }

        const target = trackClips.find((item) => item.id === overlap.clipId);
        if (target === undefined) return hit;
        const region: ClipHitRegion =
            overlap.kind === "clip-left-edge"
                ? "left-edge"
                : overlap.kind === "clip-right-edge"
                  ? "right-edge"
                  : overlap.kind === "crossfade-grip"
                    ? "crossfade-grip"
                    : overlap.fadeSide === "out"
                      ? "fade-out-corner"
                      : "fade-in-corner";
        // 说明：淡变命中（`kind === "fade"`）映射到既有角部区域，复用同一条
        // `clip-fade` 手势——包络线拖拽与角部拖拽在旧实现里是同一个语义（调长度），
        // 只是抓取位置不同。
        return {
            kind: "clip",
            clip: target,
            region,
            sec: hit.sec,
            trackIndex: hit.trackIndex,
            localX: contentX - target.startSec * view.pxPerSec,
            localY: hit.localY,
            partnerClipId: overlap.partnerClipId,
            fadeIsLine: overlap.fadeIsLine,
        };
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
        // 新的指针交互开始：上一次右键交互遗留的「吞掉下一次 contextmenu」标记
        // 必须失效，否则会误吞这一次交互真正需要的菜单。
        suppressNextContextMenu = false;
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
    /**
     * 内核命中结果。
     *
     * `hitTest` 的原始结果 + 重叠区解析补充的字段（`partnerClipId` 只在交叉点
     * 抓手上出现：抓手需要同时知道两侧的 clip）。
     */
    type KernelHit = ReturnType<typeof hitTest> & {
        readonly partnerClipId?: string;
        /**
         * 淡变命中是否为包络线本体（`false` = 区域边缘竖线）。
         *
         * 双击重置曲率只对本体生效（旧实现 `zone.line` / `isLine` 同义）。
         */
        readonly fadeIsLine?: boolean;
    };

    /**
     * 判定指针落在 clip header 的哪个控件上。
     *
     * 流程：按 clip 几何与业务字段构造 `buildTimelineClipVisualStyle`（**与绘制端
     * 同一个函数、同一组参数**）→ 交给 `hitClipHeaderControl` 做控件级命中。
     *
     * 特殊说明：必须传 `activeGroupIds` / `disabledGroupIds` / `isMidiClip`——它们
     * 改变 header 控件的**可见性**，而可见性决定偏移量（静音徽标的 x 依赖链徽标
     * 是否显示）。漏传会让命中区整体偏移，表现为"点静音点到了链"。
     *
     * @param hit `hitTest` 的 clip 命中结果（已含 `localX` / `localY`）。
     * @returns 命中的控件；非 header 分区或未命中控件时为 null。
     */
    /**
     * clip 视觉样式的返回类型。
     *
     * 从函数推导而不是手写接口：字段会随绘制端演进而增删，手写副本迟早漏字段，
     * 而漏掉的往往正是命中端需要的那一个。
     */
    type ClipHeaderStyle = ReturnType<typeof buildTimelineClipVisualStyle>;

    function buildHeaderStyle(clip: HitTestClip, clipWidthPx: number): ClipHeaderStyle {
        const d = data();
        const track = d.tracks.find((item) => item.id === clip.trackId);
        return buildTimelineClipVisualStyle({
            widthPx: clipWidthPx,
            trackColor: track?.color,
            selected: d.selectedClipId === clip.id || d.multiSelectedClipIds.includes(clip.id),
            muted: clip.muted === true,
            gain: clip.gain ?? 0,
            playbackRate: clip.playbackRate ?? 1,
            name: clip.name ?? "",
            fontFamily: resolveFontFamily(),
            isPitchAdjustment: clip.isMidiClip === true,
            groupId: clip.groupId,
            isGroupActive: clip.groupId != null && d.activeGroupIds.includes(clip.groupId),
            isGroupDisabled: clip.groupId != null && d.disabledGroupIds.includes(clip.groupId),
            darkMode: d.darkMode,
        });
    }

    function resolveHeaderControl(hit: {
        readonly clip: HitTestClip;
        readonly region: ClipHitRegion;
        readonly localX: number;
        readonly localY: number;
    }): ClipHeaderControl {
        if (hit.region !== "header") return null;
        const view = scroll.get();
        const clipWidthPx = Math.max(1, hit.clip.lengthSec * view.pxPerSec);
        const style = buildHeaderStyle(hit.clip, clipWidthPx);
        return hitClipHeaderControl({
            localX: hit.localX,
            localY: hit.localY,
            clipWidthPx,
            style,
            headerHeightPx: CLIP_HEADER_HEIGHT,
        });
    }

    /**
     * 把视口内坐标换算为屏幕坐标（供浮层 / 输入框锚点使用）。
     *
     * @param clientX 指针视口坐标 x。@param clientY 指针视口坐标 y。
     * @returns 与指针同位置的屏幕坐标（浮层需要的是 `clientX/Y`，直接透传）。
     */
    function screenAnchor(clientX: number, clientY: number): { x: number; y: number } {
        return { x: clientX, y: clientY };
    }

    function startPrimaryGesture(event: PointerEvent): void {
        const hit = hitAt(event.clientX, event.clientY);
        if (hit.kind === "clip") {
            // 面板可在此整体接管（例如 `Alt + Shift` 竖直拖 = 调音高，复用旧实现的
            // 状态机）。返回 true 时内核不启动任何自己的手势。
            if (
                interactions?.onClipPointerDownIntercept?.({
                    clipId: hit.clip.id,
                    clientX: event.clientX,
                    clientY: event.clientY,
                    pointerId: event.pointerId,
                    modifiers: dragModifiersOf(event),
                    container,
                }) === true
            ) {
                return;
            }
            // 双击判定（参数与旧实现 `ClipItem` 一致）：命中后请求参数编辑器按
            // clip 起止范围创建选区，**不**进入选中 / 拖拽手势（旧实现同样在
            // 第二次按下时拦截并 return）。第一次按下只记待定，若随后发生拖拽，
            // 待定记录会在手势升级时失效（见 onGesturePointerMove）。
            const previousPress = lastClipPress;
            const isDoubleClick =
                previousPress != null &&
                previousPress.pointerId === event.pointerId &&
                Math.abs(previousPress.clientX - event.clientX) <= DOUBLE_CLICK_MOVE_PX &&
                Math.abs(previousPress.clientY - event.clientY) <= DOUBLE_CLICK_MOVE_PX &&
                performance.now() - previousPress.time <= DOUBLE_CLICK_INTERVAL_MS;
            lastClipPress = isDoubleClick
                ? null
                : {
                      pointerId: event.pointerId,
                      clientX: event.clientX,
                      clientY: event.clientY,
                      time: performance.now(),
                  };
            // ── 淡变包络线 / 交叉点抓手的双击 = 重置曲率 ──
            // 用旧实现同一套判定（时间窗 + 目标键，见 `fadeLineClickGesture`）：
            // 位置无关——包络线很细，用户两次点击很难落在同一像素上，按位置判双击
            // 会经常失效。循环修饰键按住时循环优先，双击让位（与旧实现一致）。
            const fadeCycleHeld = isFadeShapeCycleModifierHeld(
                data().keybindings.fadeShapeCycle,
                event,
            );
            if (
                hit.region === "fade-in-corner" ||
                hit.region === "fade-out-corner" ||
                hit.region === "crossfade-grip"
            ) {
                const isGrip = hit.region === "crossfade-grip";
                const pressKey = isGrip
                    ? `${hit.clip.id}:${hit.partnerClipId ?? ""}:grip`
                    : `${hit.clip.id}:${hit.region}`;
                if (!fadeCycleHeld && noteFadeLinePointerDown(pressKey) === "double") {
                    // 抓手 = 同时重置两侧；包络线**本体** = 只重置该侧；
                    // 区域边缘竖线不参与（旧实现的 `zone.line` 门槛）。
                    const sides = isGrip
                        ? hit.partnerClipId === undefined
                            ? null
                            : [
                                  { clipId: hit.partnerClipId, isOut: true },
                                  { clipId: hit.clip.id, isOut: false },
                              ]
                        : hit.fadeIsLine === true
                          ? [{ clipId: hit.clip.id, isOut: hit.region === "fade-out-corner" }]
                          : null;
                    if (sides !== null) {
                        event.preventDefault();
                        interactions?.onResetFadeCurvature?.(sides);
                        return;
                    }
                }
            }

            if (isDoubleClick) {
                // 名称区双击 → 重命名；旋钮双击 → 重置 0 dB；增益 / 速率徽标双击 →
                // 行内编辑；其他区域双击 → 参数编辑器选区。
                // （旧实现里名称区处理器会 stopPropagation，两者天然互斥。）
                //
                // 阻止默认动作：否则浏览器会把焦点移到被点击的容器上，而重命名输入框
                // 是在本次 pointerdown 内同步挂载的——刚聚焦就被夺走焦点，其 onBlur
                // 会把这次编辑当作"点开又点走"立刻取消。
                event.preventDefault();
                const anchor = screenAnchor(event.clientX, event.clientY);
                const control = resolveHeaderControl(hit);
                if (control === "name") {
                    interactions?.onRenameClipStart?.(hit.clip.id, anchor.x, anchor.y);
                } else if (control === "gain-knob") {
                    // 双击旋钮 = 恢复 0 dB（手册：「双击恢复为 0 dB」）。
                    interactions?.onGainReset?.(hit.clip.id);
                } else if (control === "gain-label" || control === "rate-label") {
                    // 双击徽标 = 原地输入数值（手册：「双击徽标即可在原位置输入数值」）。
                    interactions?.onBadgeEditStart?.(
                        hit.clip.id,
                        control === "gain-label" ? "gain" : "rate",
                        anchor.x,
                        anchor.y,
                    );
                } else {
                    interactions?.onDoubleClickClip?.(hit.clip.id);
                }
                return;
            }

            // ── header 控件级分派 ──
            // 位于双击判定之后：双击优先级更高（名称区双击进入重命名，而不是
            // 触发一次"开始编辑"再被双击覆盖）。
            const control = resolveHeaderControl(hit);
            if (control !== null) {
                // 阻止默认动作（焦点转移）：行内编辑输入框是在本次 pointerdown 内
                // 同步挂载并自动聚焦的，浏览器随后把焦点移到容器会让它立刻失焦，
                // 表现为"点了标签但输入框一闪即消"。`name` 不消费（要继续走选中 /
                // 拖拽），因此只对真正被消费的控件生效。
                if (control !== "name") event.preventDefault();
                const anchor = screenAnchor(event.clientX, event.clientY);
                if (control === "mute") {
                    interactions?.onToggleClipMute?.(hit.clip.id, hit.clip.muted !== true);
                    return;
                }
                if (control === "formant") {
                    interactions?.onOpenClipFormant?.(hit.clip.id, anchor.x, anchor.y);
                    return;
                }
                if (control === "gain-knob") {
                    // 音量旋钮 = **拖动调值**（旧实现 3px 起手阈值），双击重置 0 dB。
                    // 单击不做任何事（手册：「音量旋钮依然可以直接上下拖动，双击恢复为
                    // 0 dB」；数值输入在**徽标**上）。
                    //
                    // 双击不需要在这里判定：旋钮属于 clip 命中，双击记录由上方
                    // `lastClipPress` 统一维护，双击分支（`isDoubleClick`）已按
                    // `control === "gain-knob"` 派发 `onGainReset`。这里若再判一次，
                    // 读到的会是**本次**按下（上方刚写过），永远判成双击。
                    gesture = {
                        kind: "gain-drag",
                        clipId: hit.clip.id,
                        startClientX: event.clientX,
                        startClientY: event.clientY,
                        lastDeltaY: 0,
                        started: false,
                    };
                    return;
                }
                // 增益 / 速率徽标：单击不进入编辑（手册：「双击徽标即可在原位置输入
                // 数值」），因此这里只放行——继续往下走选中 / 拖拽，双击在双击分支处理。
                if (control === "chain") {
                    // 锁链徽标：临时禁用 / 启用该编组的联动编辑（旧实现
                    // `ClipHeader` 的 `onToggleGroupDisabled`，作用于整个组而非单个 clip）。
                    const groupId = data().clips.find((item) => item.id === hit.clip.id)?.groupId;
                    if (groupId != null && groupId !== "") {
                        interactions?.onToggleGroupDisabled?.(groupId);
                        return;
                    }
                    // 无组可切换（理论不可达：徽标只在有组时可见）→ 落到选中 / 拖拽。
                }
                // `name`：单击仍走选中 / 拖拽（只有双击才重命名），继续往下走。
            }

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
                partnerClipId: hit.partnerClipId,
                // 循环修饰键只在淡变控件 / 抓手上起作用（其他分区的单击语义不变）。
                cycleHeld:
                    fadeCycleHeld &&
                    (hit.region === "fade-in-corner" ||
                        hit.region === "fade-out-corner" ||
                        hit.region === "crossfade-grip"),
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
                // 吸附偏移直接取命中索引里的值（与 hitTest 判定手柄位置用的是
                // 同一份数据，不另查一次 `data().clips`——两份来源迟早漂移）。
                originSnapOffsetSec:
                    hit.region === "snap-offset-handle"
                        ? Math.max(0, Number(hit.clip.snapOffsetSec) || 0)
                        : 0,
            };
        } else {
            gesture = { kind: "seek" };
            // 空白按下：连同**指针所在轨道**一起交给面板——「清空选中 + 按设置
            // 切换当前轨道」是旧实现 pointerdown 捕获分支的语义，不能在面板侧
            // 从 sec 反推（轨道要靠 clientY 换算）。
            interactions?.onSeek?.(hit.sec, true, hit.trackId);
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
        // 默认分区（body / header）**不设抓取光标**。
        //
        // 旧实现的 clip 本体（`ClipItem` 根元素）与 header（名称 / 徽标）都没有
        // cursor 类，只有边缘与淡变手柄设了 resize —— 继承下来就是 default。
        // 给整块 clip 挂 `grab` 会让「可拖拽」这一个提示盖过其它更具体的语义
        // （header 上的静音 / 共振峰按钮、名称、增益速率标签各有自己的交互），
        // 既与旧实现不一致，也让用户看不出哪些位置是可点的控件。
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
                case "snap-offset-handle":
                    // 与旧实现的握把一致（`ClipItem` 的 SnapOffset 命中区）。
                    cursor = "ew-resize";
                    break;
                default:
                    // body / header / 重叠区控件：保持 default（见上方说明）。
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
        } else if (gesture.kind === "crossfade-grip") {
            // 抓手是水平移动双方边缘，光标与旧实现的 crossfade_edges 一致。
            container.style.cursor = "ew-resize";
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
            // 发生拖拽 → 双击待定记录失效：否则"拖拽后落点回位"（第二次按下
            // 恰好落在首次按下附近）会被误判成双击（旧实现同样在拖拽分支清空）。
            lastClipPress = null;
            // 交叉点抓手最先判：它在重叠区里优先级最高（旧实现给抓手显式
            // zIndex 400，注释写明「应高于所有淡入淡出/边缘控件」）。
            if (gesture.region === "crossfade-grip") {
                if (gesture.partnerClipId === undefined) {
                    // 理论上不会发生（region 由重叠解析连同 partner 一起给出）；
                    // 真出现就放弃这次手势，而不是拿单个 clip 去移动双方边缘。
                    gesture = { kind: "none" };
                    return;
                }
                gesture = {
                    kind: "crossfade-grip",
                    earlierClipId: gesture.partnerClipId,
                    laterClipId: gesture.clipId,
                    startContentX: gesture.startContentX,
                    lastDeltaSec: Number.NaN,
                };
                applyCrossfadeGripPreview(event);
                return;
            }
            // SnapOffset 手柄：优先级仅次于交叉点抓手（旧实现 z-400 抓手 >
            // z-70 手柄 > z-65 淡变角 > z-60 边缘）。
            if (gesture.region === "snap-offset-handle") {
                gesture = {
                    kind: "snap-offset-drag",
                    clipId: gesture.clipId,
                    startContentX: gesture.startContentX,
                    originOffsetSec: gesture.originSnapOffsetSec,
                    lengthSec: gesture.lengthSec,
                    lastOffsetSec: Number.NaN,
                };
                applySnapOffsetPreview(event);
                return;
            }
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
        if (gesture.kind === "gain-drag") {
            applyGainPreview(event);
            return;
        }
        if (gesture.kind === "crossfade-grip") {
            applyCrossfadeGripPreview(event);
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
            modifiers: dragModifiersOf(event),
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
    /**
     * 交叉点抓手的拖拽预览。
     *
     * 语义（与旧实现 `crossfade_edges` 一致）：拖动抓手 = 同时把前一个 clip 的
     * 右缘与后一个 clip 的左缘按**同一位移**移动，因此重叠长度不变（手动 /
     * 自动淡变长度都不受影响）。
     *
     * 内核只给位移，具体几何（谁的长度加减多少）由面板按预览原点换算——与
     * trim / fade 同一条分工。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyCrossfadeGripPreview(event: PointerEvent): void {
        if (gesture.kind !== "crossfade-grip") return;
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const contentX = view.scrollLeft + (event.clientX - rect.left);
        const deltaSec = (contentX - gesture.startContentX) / Math.max(1e-9, view.pxPerSec);
        if (deltaSec === gesture.lastDeltaSec) return;
        gesture.lastDeltaSec = deltaSec;
        interactions?.onCrossfadeGripPreview?.({
            earlierClipId: gesture.earlierClipId,
            laterClipId: gesture.laterClipId,
            deltaSec,
        });
    }

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
            modifiers: dragModifiersOf(event),
        });
    }

    /**
     * 计算并派发一次音量旋钮拖拽预览（拖拽期间每帧调用）。
     *
     * 流程：先判起手阈值（3px，与旧实现一致）→ 取**相对按下点**的竖直位移 →
     * 与上次值比较去重 → 回调。
     *
     * 特殊说明：内核只给竖直位移（CSS px）——dB 换算（`CLIP_GAIN_DRAG_DB_PER_PX`）、
     * 精细调整修饰键、±12dB 钳制与批量应用都在面板侧（与「内核只做几何」的既有分工一致）。
     *
     * @param event 指针事件。
     * @returns 无返回值。
     */
    function applyGainPreview(event: PointerEvent): void {
        if (gesture.kind !== "gain-drag") return;
        if (!gesture.started) {
            const dx = event.clientX - gesture.startClientX;
            const dy = event.clientY - gesture.startClientY;
            if (dx * dx + dy * dy < GAIN_DRAG_THRESHOLD_PX * GAIN_DRAG_THRESHOLD_PX) return;
            gesture.started = true;
        }
        const deltaY = event.clientY - gesture.startClientY;
        if (deltaY === gesture.lastDeltaY) return;
        gesture.lastDeltaY = deltaY;
        interactions?.onGainDragPreview?.({
            clipId: gesture.clipId,
            deltaYPx: deltaY,
            modifiers: dragModifiersOf(event),
        });
    }

    /**
     * 计算并派发一次吸附偏移预览（拖拽期间每帧调用）。
     *
     * 与旧实现同源：位移按 `pxPerSec` 换算为秒，叠加在按下时的偏移上。**不在此处
     * 吸附、也不钳制**——吸附引擎与边界都在面板侧（见 `onSnapOffsetPreview`）。
     *
     * @param event 指针事件。
     */
    function applySnapOffsetPreview(event: PointerEvent): void {
        if (gesture.kind !== "snap-offset-drag") return;
        const rect = container.getBoundingClientRect();
        const view = scroll.get();
        const contentX = view.scrollLeft + (event.clientX - rect.left);
        const deltaSec = (contentX - gesture.startContentX) / Math.max(1e-9, view.pxPerSec);
        const rawOffsetSec = gesture.originOffsetSec + deltaSec;
        if (rawOffsetSec === gesture.lastOffsetSec) return;
        gesture.lastOffsetSec = rawOffsetSec;
        interactions?.onSnapOffsetPreview?.({
            clipId: gesture.clipId,
            rawOffsetSec,
            modifiers: dragModifiersOf(event),
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
    /**
     * 取修饰键快照（指针 / 键盘事件都带这四个字段）。
     *
     * @param event 事件。
     * @returns 快照，形状与 `isModifierActive` 的 `event` 参数一致。
     */
    function dragModifiersOf(event: {
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey: boolean;
    }): KernelDragModifiers {
        return {
            ctrlKey: event.ctrlKey,
            shiftKey: event.shiftKey,
            altKey: event.altKey,
            metaKey: event.metaKey,
        };
    }

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
        if (
            delta.deltaSec === gesture.lastDeltaSec &&
            targetTrackId === gesture.lastTargetTrackId
        ) {
            return;
        }
        gesture.lastDeltaSec = delta.deltaSec;
        gesture.lastTargetTrackId = targetTrackId;
        interactions?.onDragPreview?.({
            clipId: gesture.clipId,
            deltaSec: delta.deltaSec,
            targetTrackId,
            modifiers: dragModifiersOf(event),
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
                modifiers: dragModifiersOf(event),
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
        if (gesture.kind === "gain-drag") {
            interactions?.onGainDragCommit?.({
                clipId: gesture.clipId,
                // `changed = false` 表示未越起手阈值的单击——调用方不应提交
                // （旧实现同样只在真正拖动后才写增益）。
                changed: gesture.started,
                cancelled,
            });
        }
        if (gesture.kind === "crossfade-grip") {
            interactions?.onCrossfadeGripCommit?.({
                earlierClipId: gesture.earlierClipId,
                laterClipId: gesture.laterClipId,
                deltaSec: gesture.lastDeltaSec,
                cancelled,
            });
        }
        if (gesture.kind === "snap-offset-drag") {
            interactions?.onSnapOffsetCommit?.({
                clipId: gesture.clipId,
                cancelled,
                // 零位移单击：`lastOffsetSec` 从未被赋值（NaN）→ 未发生真实位移。
                // 与旧实现一致：单击不写后端、不产生 undo 步。
                changed: Number.isFinite(gesture.lastOffsetSec),
            });
        }
        if (gesture.kind === "box-select") {
            const boxActive = gesture.active;
            // 未超过阈值（右键单击）不提交：交给 contextmenu 弹菜单。
            if (boxActive) {
                interactions?.onBoxSelectCommit?.({
                    clipIds: gesture.lastClipIds,
                    additive: gesture.additive,
                    cancelled,
                });
            }
            boxSelectEl.style.display = "none";
            // 菜单延迟到松手判定（见 `onContextMenu` 的说明）：
            // - 框选成立 → 丢弃按下时被抑制的菜单，并吞掉紧随的那一次 contextmenu
            //   （兼容「松手后才触发 contextmenu」的平台）；
            // - 框选未成立（右键单击）→ 在按下位置补发菜单（旧实现同一语义）。
            const pending = pendingContextMenu;
            pendingContextMenu = null;
            if (boxActive) {
                suppressNextContextMenu = true;
            } else if (pending !== null && !cancelled) {
                suppressNextContextMenu = true;
                dispatchContextMenuAt(pending.clientX, pending.clientY);
            }
        }
        if (gesture.kind === "pending-select" && !cancelled) {
            if (gesture.region === "snap-offset-handle") {
                // 手柄上的单击**不改选中**：旧实现的握把在 pointerdown 里
                // `stopPropagation`，点击不会落到 clip 上——只有拖动才有意义。
            } else if (gesture.cycleHeld === true) {
                // 循环修饰键 + 未超过拖拽阈值（= 单击）→ 切换淡变形状。
                // 不派发选中：循环点击是编辑操作，不是选择操作（旧实现同样不选中）。
                if (gesture.region === "crossfade-grip") {
                    if (gesture.partnerClipId !== undefined) {
                        interactions?.onCrossfadeCycle?.([
                            { clipId: gesture.partnerClipId, isOut: true },
                            { clipId: gesture.clipId, isOut: false },
                        ]);
                    }
                } else {
                    interactions?.onFadeShapeCycle?.(
                        gesture.clipId,
                        gesture.region === "fade-out-corner" ? "out" : "in",
                    );
                }
            } else {
                interactions?.onSelectClip?.(gesture.clipId, event.ctrlKey || event.metaKey);
            }
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
        // 右键交互（待框选）尚未结束：菜单要等**松手**才决定是否弹出。
        //
        // 【为什么不能在这里直接弹】macOS / 部分 Chromium 平台在右键**按下**时就
        // 触发 contextmenu，而框选是否成立要等指针移动超过阈值才知道。按下即弹的
        // 后果是：框选确实完成了，但菜单也已经挂在画面上（且它的 backdrop 会吞掉
        // 下一次左键——表现为"框选之后拖不动 clip"）。旧实现同样把右键单击的菜单
        // 延迟到松手重放（见 `useTimelineSelectionRect`）。
        if (gesture.kind === "box-select") {
            event.preventDefault();
            event.stopPropagation();
            pendingContextMenu = { clientX: event.clientX, clientY: event.clientY };
            return;
        }
        if (suppressNextContextMenu) {
            event.preventDefault();
            event.stopPropagation();
            suppressNextContextMenu = false;
            return;
        }
        event.preventDefault();
        dispatchContextMenuAt(event.clientX, event.clientY);
    }

    /**
     * 在指定屏幕位置派发通用右键菜单（命中解析 + 速率标签优先）。
     *
     * 流程：命中测试 → 速率角标优先（旧实现同样在速率角标上拦截右键）→
     * 收集指针处的 clip 列表 → 交给面板的 `onContextMenu`。
     *
     * 特殊说明：被「待框选」手势抑制的 contextmenu 会在松手且**未发生框选**时
     * 由本函数补发（位置取按下时的指针位置），语义与右键单击一致。
     *
     * @param clientX 指针视口坐标 X。@param clientY 指针视口坐标 Y。
     */
    function dispatchContextMenuAt(clientX: number, clientY: number): void {
        const hit = hitAt(clientX, clientY);
        // 淡变包络 / 交叉点抓手右键 → **淡变专属菜单**（曲率、形状、长度、重置…），
        // 优先于 clip 通用菜单。旧实现由 `FadeHitLayer` / `OverlapEditLayer`
        // 这两层 DOM 命中块拦截，内核模式下它们不挂载——必须在这里补。
        if (
            hit.kind === "clip" &&
            interactions?.onFadeContextMenu !== undefined &&
            (hit.region === "fade-in-corner" ||
                hit.region === "fade-out-corner" ||
                hit.region === "crossfade-grip")
        ) {
            const isGrip = hit.region === "crossfade-grip";
            // 抓手 = 双侧：primary 是**前一个** clip 的淡出、secondary 是后一个的淡入
            //（与旧实现 `crossfadeSides.out / .in` 的列序一致）。
            const primary = isGrip
                ? hit.partnerClipId === undefined
                    ? null
                    : fadeSideAt(hit.partnerClipId, true)
                : fadeSideAt(hit.clip.id, hit.region === "fade-out-corner");
            const secondary = isGrip ? fadeSideAt(hit.clip.id, false) : null;
            if (primary !== null) {
                interactions.onFadeContextMenu({ clientX, clientY, primary, secondary });
                return;
            }
        }
        // 速率标签右键 → 速率高级编辑（BPM 换算对话框），优先于通用右键菜单
        // （旧实现同样在速率角标上拦截右键）。其余位置仍走通用菜单。
        if (
            hit.kind === "clip" &&
            interactions?.onRateBadgeMenu !== undefined &&
            resolveHeaderControl(hit) === "rate-label"
        ) {
            interactions.onRateBadgeMenu(hit.clip.id, clientX, clientY);
            return;
        }
        if (interactions?.onContextMenu === undefined) return;
        const trackId = hit.kind === "clip" ? hit.clip.trackId : hit.trackId;
        interactions.onContextMenu({
            clientX,
            clientY,
            clipIds: trackId === null ? [] : clipsAtPointer(trackId, hit.sec),
            trackId,
            sec: hit.sec,
        });
    }

    /**
     * 构造淡变专属菜单的单侧载荷（形状 / 方向 / 生效长度）。
     *
     * 特殊说明：长度取**生效值**（自动交叉淡化 > 0 时覆盖手动值），与绘制端和旧实现
     * `OverlapEditLayer.effectiveFadeInSec` 同一规则——菜单里显示的必须是画面上真正
     * 生效的那个长度，否则会出现「菜单写 0.8s、包络线却是 1.2s」。
     *
     * @param clipId 目标 clip。
     * @param isOut true = 淡出侧，false = 淡入侧。
     * @returns 载荷；clip 已不存在时为 null（调用方退化为不弹菜单）。
     */
    function fadeSideAt(clipId: string, isOut: boolean): FadeContextSide | null {
        const clip = data().clips.find((item) => item.id === clipId);
        if (clip === undefined) return null;
        return {
            clipId,
            isOut,
            shape: (isOut ? clip.fadeOutShape : clip.fadeInShape) ?? 0,
            dir: (isOut ? clip.fadeOutDir : clip.fadeInDir) ?? 0,
            lengthSec: effectiveFadeSec(
                isOut ? clip.fadeOutSec : clip.fadeInSec,
                isOut ? clip.autoFadeOutSec : clip.autoFadeInSec,
            ),
        };
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
                        modifiers: dragModifiersOf(event),
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
                    break;
                }
                if (gesture.kind === "gain-drag") {
                    // 音量旋钮拖拽中按 Esc：回滚到按下时的增益（未起手时无副作用）。
                    event.preventDefault();
                    interactions?.onGainDragCommit?.({
                        clipId: gesture.clipId,
                        changed: gesture.started,
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

        setScrollLeft(px: number) {
            scroll.setScrollLeft(px);
        },

        setViewport(next: { pxPerSec?: number; scrollLeft?: number }) {
            const beforePxPerSec = scroll.get().pxPerSec;
            const applied = scroll.setViewport(next);
            // 缩放真值源在内核：变化时必须回调 React 侧派生量（标尺刻度 / 内容宽度），
            // 否则「网格已缩放、标尺还在旧刻度」。
            if (applied.pxPerSec !== beforePxPerSec) onZoomChange?.(applied.pxPerSec);
            return { scrollLeft: applied.scrollLeft, pxPerSec: applied.pxPerSec };
        },

        getContainerRect() {
            return container.getBoundingClientRect();
        },

        getViewport() {
            const view = scroll.get();
            return {
                scrollLeft: view.scrollLeft,
                scrollTop: view.scrollTop,
                pxPerSec: view.pxPerSec,
                viewportWidth: viewportWidthPx,
                viewportHeight: viewportHeightPx,
            };
        },

        getAxis() {
            return currentAxis();
        },

        getClipHeaderAnchor(clipId, field) {
            const d = data();
            const clip = d.clips.find((item) => item.id === clipId);
            if (clip === undefined) return null;
            const rowIndex = d.tracks.findIndex((item) => item.id === clip.trackId);
            if (rowIndex < 0) return null;
            const view = scroll.get();
            const clipWidthPx = Math.max(1, clip.lengthSec * view.pxPerSec);
            const style = buildHeaderStyle(
                {
                    id: clip.id,
                    trackId: clip.trackId,
                    startSec: clip.startSec,
                    lengthSec: clip.lengthSec,
                    muted: clip.muted,
                    gain: clip.gain,
                    playbackRate: clip.playbackRate,
                    name: clip.name,
                    groupId: clip.groupId,
                    isMidiClip: clip.midiNoteCount != null,
                },
                clipWidthPx,
            );
            const clipLeftPx = clip.startSec * view.pxPerSec;
            const rowTopPx = rowIndex * view.rowHeight;
            if (field === "name") {
                const left = clipLeftPx + style.leadingControlsWidth;
                const available =
                    clipWidthPx - style.leadingControlsWidth - style.trailingReservePx;
                return {
                    contentLeftPx: left,
                    contentTopPx: rowTopPx,
                    widthPx: Math.max(60, Math.min(240, available)),
                };
            }
            if (field === "gain") {
                return {
                    contentLeftPx: clipLeftPx + clipWidthPx - style.gainLabelWidth - 6,
                    contentTopPx: rowTopPx,
                    widthPx: 72,
                };
            }
            return {
                contentLeftPx:
                    clipLeftPx + clipWidthPx - style.gainLabelWidth - style.rateLabelWidth - 14,
                contentTopPx: rowTopPx,
                widthPx: 72,
            };
        },

        debugHitAt(clientX: number, clientY: number) {
            const hit = hitAt(clientX, clientY);
            if (hit.kind !== "clip") {
                return { kind: hit.kind, sec: hit.sec };
            }
            return {
                kind: hit.kind,
                region: hit.region,
                control: resolveHeaderControl(hit) ?? undefined,
                clipId: hit.clip.id,
                localX: hit.localX,
                localY: hit.localY,
                sec: hit.sec,
            };
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
