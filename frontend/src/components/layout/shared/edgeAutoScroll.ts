/**
 * 拖拽边缘自动滚屏 · 几何（纯函数，跨内核共享）
 *
 * 【主要内容】
 * 1. `resolveEdgeScrollDeltaPx()`：指针位置 + 视口边界 + 本帧时长 + 最大速度
 *    → 本帧滚动步长；
 * 2. `edgeScrollMaxLeftPx()`：视口能滚到的最右位置（`scrollLeft` 上界）；
 * 3. 两个与速度无关的常量：边缘带宽、比例上限。
 *
 * 【为什么单独成模块】
 * 时间轴内核（`timeline/kernel/input/dragAutoScroll.ts`）与参数编辑器内核
 *（`pianoRoll/kernel/dragArithmetic.ts` 的调用方）都需要这段几何，此前两边各写了
 * 一份**实现**。两份实现意味着调手感时要改两处，而漏改一处的表现是"两个面板的
 * 滚屏行为不一样"—— 没有任何理由让同一个公式存在两个副本。
 *
 * 本模块只依赖算术，不依赖 DOM / React / 任一内核，因此两边都可以直接依赖它，
 * 不构成内核之间的耦合。
 *
 * 【为什么速度是入参而不是常量】
 * 两个面板的**最大速度**是有意不同的：时间轴 720 px/秒（clip 更宽，全览时一个 clip
 * 可能横跨半个视口，滚太快难以精确落点），参数编辑器 1080 px/秒
 *（= 它原有的"每帧 18px" @60Hz，曲线编辑需要更快地跨过工程）。共享的是**公式**，
 * 手感参数各归其主 —— 把速度也统一成同一个值会改变其中一个面板的既有手感，
 * 那不是本次修复的目的。
 *
 * 【为什么按"每秒速度"而不是"每帧步长"】
 * 逐帧步长会让滚屏速度随刷新率（或事件频率）变化：60Hz 与 120Hz 差一倍；
 * 而按"每 pointermove 事件滚 N 像素"的旧实现更糟 —— 1000Hz 轮询率的鼠标会把
 * 18px/事件放大到 18000 px/秒，同一个手势在不同鼠标上快慢相差一个数量级。
 * 这里以 px/秒 表达、乘以本帧实际时长，速度与帧率、事件频率都解耦。
 *
 * 【维护说明】调整手感**只改调用方传入的速度常量**，不要改公式结构 —— 带宽内
 * 线性加速、比例上限 1.5、"步长与帧时长成正比"三点都有单测钉住。
 */

/** 边缘触发带宽（CSS px）：指针进入距视口边缘这么近时开始自动滚屏。 */
export const EDGE_SCROLL_BAND_PX = 32;

/** 指针与视口的横向边界（`getBoundingClientRect()` 的左右缘）。 */
export interface EdgeScrollBounds {
    readonly left: number;
    readonly right: number;
}

/** 比例上限：指针拖出视口后不再继续加速（否则越拖越快，难以停在想看的位置）。 */
export const EDGE_SCROLL_MAX_RATIO = 1.5;

/** 本帧时长的合法区间（毫秒）。上限的理由见 `resolveEdgeScrollDeltaPx`。 */
const MIN_FRAME_MS = 1;
const MAX_FRAME_MS = 100;

/** 默认帧时长（60Hz），用于非法帧时长的回退。 */
const DEFAULT_FRAME_MS = 1000 / 60;

/** 数值夹取。 */
function clampNumber(value: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return min;
    return Math.min(max, Math.max(min, value));
}

/**
 * 把「指针相对视口边缘的位置」解析为**本帧**的滚动步长。
 *
 * 流程：
 * 1. 指针落在左/右边缘带内 → 按"进入带宽的比例"线性加速（上限 1.5）；
 * 2. 比例 × 最大速度 → 每秒速度；
 * 3. 每秒速度 × 本帧时长（秒）→ 本帧步长。
 *
 * 特殊说明 1：**方向符号**——左缘为负（内容左移 = 看到更早的时间）、右缘为正，
 * 与 `scrollLeft` 增大方向一致，调用方直接相加即可。
 *
 * 特殊说明 2：非有限输入一律返回 0。滚动位置被 NaN 污染后无法恢复，而"这一帧
 * 不滚"只是让自动滚屏晚一帧启动。
 *
 * 特殊说明 3：帧时长非法（0 / 负 / NaN）时按 1/60 秒回退，而不是返回 0 ——
 * 宁可步长略不精确，也不要让自动滚屏在个别脏帧上完全停住。
 *
 * 特殊说明 4：帧时长**上限 100ms**（约 10fps）。长于它的间隔通常是"标签页被挂起
 * 后恢复"这类非真实帧，按真实时长补步会一次性跳过很远的位置；夹到 100ms 相当于
 * 「补一帧、不补一段」。
 *
 * @param args.clientX 指针的视口坐标 X。
 * @param args.leftPx 视口左边界（`getBoundingClientRect().left`）。
 * @param args.rightPx 视口右边界（`getBoundingClientRect().right`）。
 * @param args.frameMs 本帧时长（毫秒）；非法值回退 1/60 秒，上限 100ms。
 * @param args.maxSpeedPxPerSec 该面板的最大滚屏速度（CSS px/秒），见模块头说明。
 * @returns 本帧滚动步长（CSS px，左负右正）；不在边缘带内时为 0。
 */
export function resolveEdgeScrollDeltaPx(args: {
    readonly clientX: number;
    readonly leftPx: number;
    readonly rightPx: number;
    readonly frameMs: number;
    readonly maxSpeedPxPerSec: number;
}): number {
    const { clientX, leftPx, rightPx, maxSpeedPxPerSec } = args;
    if (!Number.isFinite(clientX) || !Number.isFinite(leftPx) || !Number.isFinite(rightPx)) {
        return 0;
    }
    if (!Number.isFinite(maxSpeedPxPerSec) || maxSpeedPxPerSec <= 0) return 0;
    // 视口宽为 0 / 反向（布局未就绪）：没有可用的边缘语义。
    if (!(rightPx > leftPx)) return 0;

    let ratio = 0;
    if (clientX < leftPx + EDGE_SCROLL_BAND_PX) {
        ratio = -clampNumber(
            (leftPx + EDGE_SCROLL_BAND_PX - clientX) / EDGE_SCROLL_BAND_PX,
            0,
            EDGE_SCROLL_MAX_RATIO,
        );
    } else if (clientX > rightPx - EDGE_SCROLL_BAND_PX) {
        ratio = clampNumber(
            (clientX - (rightPx - EDGE_SCROLL_BAND_PX)) / EDGE_SCROLL_BAND_PX,
            0,
            EDGE_SCROLL_MAX_RATIO,
        );
    }
    if (ratio === 0) return 0;

    const frameMs =
        Number.isFinite(args.frameMs) && args.frameMs > 0
            ? clampNumber(args.frameMs, MIN_FRAME_MS, MAX_FRAME_MS)
            : DEFAULT_FRAME_MS;
    const perSecond = ratio * maxSpeedPxPerSec;
    return (perSecond * frameMs) / 1000;
}

/**
 * 视口能滚到的**最右**位置（`scrollLeft` 的上界）。
 *
 * 内容宽度由「工程末帧 × 每帧像素」给出（内容不可能比工程更长）；同步时间轴时
 * 内容层整体右移一个偏移量，上界随之平移。
 *
 * 特殊说明 1：**结果不为负**。上界为负意味着"连内容起点都滚不到"，那是无意义的
 * 状态，写入它会让原生 `scrollLeft` 落到非法区间。
 *
 * 特殊说明 2：任一输入非有限 → 返回 0（视为不可滚）。自动滚屏停住比把视口写到
 * 未定义位置安全得多。
 *
 * 特殊说明 3：**不减视口宽**。`scrollLeft` 的上界是内容宽，而不是「内容宽 − 视口宽」
 * —— 两个内核的 `ScrollKernel` 都把上界定义成**内容宽**（原生容器比视口宽出一整屏，
 * 视口只是窗口）。曾经这里减去视口宽，于是自动滚屏的右界比内核真值小了一整个视口：
 * 指针停在右缘时每帧写 `max`、内核回写更大的值，视图在两处往复 —— 即"闪现"。
 * `viewportWidthPx` 现在只用于"内容比视口还窄时归零"。
 *
 * 【偏移的符号】`nativeOffsetPx` 按「原生 = 绘制 + 偏移」投影（与
 * `timelineViewportStateToNative` 同一约定）：原生上界 = 内容宽，故**绘制**上界 =
 * 内容宽 − 偏移。写成 `+ 偏移` 会让右界超出内核真值一个偏移量，触发同一类往复。
 *
 * @param args.pxPerSec 当前横向缩放（像素/秒）。
 * @param args.framePeriodMs 帧周期（毫秒）。
 * @param args.maxFrame 工程末帧（帧号，半开区间右端）。
 * @param args.viewportWidthPx 视口宽度（`clientWidth`），仅用于"内容比视口窄时归零"。
 * @param args.nativeOffsetPx 内容层偏移（同步时间轴时非 0，否则 0）。
 * @returns `scrollLeft` 的上界（CSS px，≥ 0）。
 */
export function edgeScrollMaxLeftPx(args: {
    readonly pxPerSec: number;
    readonly framePeriodMs: number;
    readonly maxFrame: number;
    readonly viewportWidthPx: number;
    readonly nativeOffsetPx: number;
}): number {
    const { pxPerSec, framePeriodMs, maxFrame, viewportWidthPx, nativeOffsetPx } = args;
    if (
        !Number.isFinite(pxPerSec) ||
        !Number.isFinite(framePeriodMs) ||
        !Number.isFinite(maxFrame) ||
        !Number.isFinite(viewportWidthPx) ||
        !Number.isFinite(nativeOffsetPx)
    ) {
        return 0;
    }
    // 负缩放 / 负帧周期没有物理意义，按 0 处理而不是让内容宽度变成负数。
    const pxPerFrame = (Math.max(0, pxPerSec) * Math.max(0, framePeriodMs)) / 1000;
    const contentWidthPx = Math.max(0, maxFrame) * pxPerFrame;
    // 内容比视口还窄（或缩放尚未就绪）→ 没有可滚余地，上界归 0。
    // 其余情况：`scrollLeft` 上界 = 内容宽（与两个内核 ScrollKernel 同口径，见
    // 函数文档的特殊说明 3），再按 `nativeOffsetPx` 投影到调用方所在的坐标系。
    if (contentWidthPx <= Math.max(0, viewportWidthPx)) return 0;
    return Math.max(0, contentWidthPx - nativeOffsetPx);
}
