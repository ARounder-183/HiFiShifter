/**
 * 自动滚屏（播放跟随）的几何与模式判定 —— 纯函数，无 DOM、无 React。
 *
 * 【主要内容】
 * - 滚动式：`computeAutoFollowScrollLeft`（把光标钉在视口左缘固定偏移处）；
 * - 翻页式：`computePageScrollLeft`（越界才整页翻，页内静止）；
 * - 统一入口：`resolveAutoScrollTarget`（模式分派 + 变化阈值，两个播放头桥接组件共用）；
 * - 模式取值与归一化：`AUTO_SCROLL_MODES` / `normalizeAutoScrollMode`。
 *
 * 【作用】把「本帧要不要滚、滚到哪里」从渲染代码里剥离出来：两个面板（时间轴 /
 * 参数编辑器）只需把结果写进各自的视口载体，不必各自实现一遍跟随语义。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelineTransportBridge` / `PianoRollPlayheadBridge` 的逐帧回调；
 * - 下游：`TimelineViewportAccess.setScrollLeft` / `PianoRollKernelHost.setScrollLeft`；
 * - 持久化：模式经 `SessionState.autoScrollMode` → `UiSettings.autoScrollMode` 落盘；
 * - 无关：拖拽边缘自动滚屏（`components/layout/shared/edgeAutoScroll.ts`）—— 那是
 *   手势驱动，与本模块的「播放跟随」是两条独立的线。
 */

export const AUTO_FOLLOW_PLAYHEAD_OFFSET_PX = 12;

/** 自动滚屏模式。scroll = 滚动式（实时跟随）；page = 翻页式（越界翻一页）。 */
export const AUTO_SCROLL_MODES = ["scroll", "page"] as const;
export type AutoScrollMode = (typeof AUTO_SCROLL_MODES)[number];

/**
 * 出厂默认：滚动式。
 *
 * 【为什么默认是它】引入翻页式之前的行为就是滚动式；默认值必须与升级前**逐值一致**，
 * 否则用户会在某次更新后发现"什么都没改，播放时视图的走法却变了"。
 */
export const DEFAULT_AUTO_SCROLL_MODE: AutoScrollMode = "scroll";

/**
 * 滚动位置变化判定容差（CSS px）。
 *
 * 两个桥接组件此前各自硬编码 0.5；抽到一处后，"多小的位移不值得写"只有一份定义。
 * 它同时是「钳制后不抖动」的唯一保证：翻页式在工程右端会被钳到与当前值相同的
 * 目标，正是这条守卫让后续帧返回 null（见 computePageScrollLeft 的说明）。
 */
export const AUTO_SCROLL_EPSILON_PX = 0.5;

/** 取值是否合法（用于 reducer 的静默忽略与设置水合的白名单）。 */
export function isAutoScrollMode(value: unknown): value is AutoScrollMode {
    return (AUTO_SCROLL_MODES as readonly unknown[]).includes(value);
}

/**
 * 归一化持久化里的模式值。
 *
 * 【为什么非法值一律回落而不是抛错】设置来自磁盘 / 插件形态 / 手改配置，脏值必须
 * 让应用照常启动；回落"滚动式"等价于"没设过"。
 */
export function normalizeAutoScrollMode(value: unknown): AutoScrollMode {
    return isAutoScrollMode(value) ? value : DEFAULT_AUTO_SCROLL_MODE;
}

/**
 * 滚动式自动滚屏的目标位置：把光标钉在视口左缘固定偏移（默认 12px）处。
 *
 * 【为什么不是"越界才动"】这是本功能引入翻页式**之前**就有的语义，也是"自动滚屏
 * 抽搐"那次修复之后的形状：视图在播放全程连续平移，光标始终停在左缘附近。
 * 改成"越界才跟"是另一个功能，见 `computePageScrollLeft`。
 *
 * 【上限为什么是"工程宽 − 视口宽"】工程右端到达画面右缘后停止平移（DAW 通用行为）。
 * 注意它**不是**内核的 `maxScrollLeft()`（那里等于工程宽度）：内核允许视口越出工程
 * 右缘，而播放跟随不该主动把画面拖出工程。
 */
export function computeAutoFollowScrollLeft(args: {
    playheadSec: number;
    pxPerSec: number;
    viewportWidth: number;
    contentWidth: number;
    offsetPx?: number;
}): number {
    const {
        playheadSec,
        pxPerSec,
        viewportWidth,
        contentWidth,
        offsetPx = AUTO_FOLLOW_PLAYHEAD_OFFSET_PX,
    } = args;

    const playheadX = Math.max(0, playheadSec) * Math.max(0, pxPerSec);
    // 播放跟随语义：工程右端到达画面右缘后停止平移（DAW 通用行为），
    // 因此上限仍是“工程宽 − 视口宽”。
    const maxScrollLeft = Math.max(0, Math.max(0, contentWidth) - Math.max(0, viewportWidth));
    const target = Math.max(0, playheadX - Math.max(0, offsetPx));
    return Math.min(maxScrollLeft, target);
}

/**
 * “聚焦播放光标”的水平滚动位置。
 *
 * 与播放跟随不同：聚焦必须把光标带到视口内固定偏移处，即使光标接近或
 * 位于工程末尾也要成立。自 resolveTimelineScrollRange 引入“只向右延长”
 * 的滚动范围后，原生滚动上限 = 工程宽度（右侧虚拟余量允许视口越出
 * 工程边界），因此这里的钳制上限取工程宽度本身；若沿用旧的
 * “工程宽 − 视口宽”上限，接近工程末尾的光标会被卡在画面右缘、
 * 无法正确展示。
 */
export function computeFocusCursorScrollLeft(args: {
    playheadSec: number;
    pxPerSec: number;
    /** 工程（内容）总宽度，单位 px。 */
    contentWidth: number;
    offsetPx?: number;
}): number {
    const { playheadSec, pxPerSec, contentWidth } = args;
    const offsetPx = args.offsetPx ?? AUTO_FOLLOW_PLAYHEAD_OFFSET_PX;

    const playheadX = Math.max(0, playheadSec) * Math.max(0, pxPerSec);
    // 新滚动模型：最大滚动位置 = 工程宽度。
    const maxScrollLeft = Math.max(0, contentWidth);
    const target = Math.max(0, playheadX - Math.max(0, offsetPx));
    return Math.min(maxScrollLeft, target);
}

/**
 * 翻页式自动滚屏的目标位置。
 *
 * 【流程】
 * 1. 光标仍在当前页 `[scrollLeft, scrollLeft + 视口宽)` 内 → 返回 null（本帧不滚）；
 * 2. 否则以**当前视口**为页基准求光标所在页的页首：
 *    `target = scrollLeft + floor((playheadX − scrollLeft) / 视口宽) × 视口宽`；
 * 3. 钳到 `[0, max(0, 工程宽 − 视口宽)]`（与滚动式同一业务上限：工程右端贴住视口右缘后停）。
 *
 * 【为什么用 floor 而不是"±一页"】±一页在"光标被 seek 到很远"或"用户手动滚走"时会
 * 产生**逐帧翻页的级联**（一帧翻一页，直到追上）。floor 一次就把任意距离的越界拉回
 * 光标所在的那一页，是"翻页"而不是"快速滚动"。
 *
 * 【为什么以当前视口为页基准（而不是工程原点的绝对页网格）】用户手动滚到 250px 之后，
 * 他眼前的"下一页"应当从 250 起算；用绝对页网格会产生一个只有 750px 的"半页"，
 * 与"翻页"的直觉不符。
 *
 * 【特殊说明】钳制只可能发生在工程右端：此时光标可能仍停在页外，但 target === 当前值，
 * 统一入口的容差守卫会让后续帧返回 null —— 不会每帧重复写入、也不会抖动。
 *
 * @returns 目标滚动位置；光标在页内 / 无法判定时为 null。
 */
export function computePageScrollLeft(args: {
    playheadSec: number;
    pxPerSec: number;
    viewportWidth: number;
    scrollLeft: number;
    contentWidth: number;
}): number | null {
    const { playheadSec, pxPerSec, scrollLeft, contentWidth } = args;
    const pageWidth = Math.max(0, args.viewportWidth);
    // 视口未量测 / 宿主未挂载：无从判断"一页"有多宽，本帧不滚。
    if (!(pageWidth > 0)) return null;
    if (!Number.isFinite(playheadSec) || !Number.isFinite(pxPerSec)) return null;
    if (!Number.isFinite(scrollLeft) || !Number.isFinite(contentWidth)) return null;

    const playheadX = Math.max(0, playheadSec) * Math.max(0, pxPerSec);
    const left = Math.max(0, scrollLeft);
    // 光标在页内 → 一动不动（"只有在光标超出可见范围以后"）。
    if (playheadX >= left && playheadX < left + pageWidth) return null;

    const pageIndex = Math.floor((playheadX - left) / pageWidth);
    const target = left + pageIndex * pageWidth;
    // 与滚动式同一上限：工程右端到达视口右缘后停止平移。
    const maxScrollLeft = Math.max(0, Math.max(0, contentWidth) - pageWidth);
    return Math.min(maxScrollLeft, Math.max(0, target));
}

/**
 * 本帧应写入的滚动位置（两种模式的唯一入口）。
 *
 * 【流程】按 mode 求目标 → 非有限值丢弃 → |目标 − 当前| ≤ 容差 视为无需写入。
 *
 * 【为什么把"要不要写"收进来】此前两个桥接组件各写一遍 `Math.abs(...) > 0.5`；
 * 抽到一处后，"两种模式共用同一条变化阈值"成为结构性事实，而不是两处巧合。
 *
 * @returns 需要写入的目标位置；本帧不滚动时为 null。
 */
export function resolveAutoScrollTarget(args: {
    mode: AutoScrollMode;
    playheadSec: number;
    pxPerSec: number;
    viewportWidth: number;
    scrollLeft: number;
    contentWidth: number;
}): number | null {
    const current = Number.isFinite(args.scrollLeft) ? Math.max(0, args.scrollLeft) : 0;
    const target =
        args.mode === "page"
            ? computePageScrollLeft(args)
            : computeAutoFollowScrollLeft({
                  playheadSec: args.playheadSec,
                  pxPerSec: args.pxPerSec,
                  viewportWidth: args.viewportWidth,
                  contentWidth: args.contentWidth,
              });
    if (target === null || !Number.isFinite(target)) return null;
    return Math.abs(target - current) > AUTO_SCROLL_EPSILON_PX ? target : null;
}
