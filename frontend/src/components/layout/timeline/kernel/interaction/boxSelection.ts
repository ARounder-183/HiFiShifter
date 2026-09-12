/**
 * 时间轴渲染内核 · 框选几何
 *
 * 【主要内容】
 * 提供框选矩形的规范化（拖拽方向无关）与「clip 是否落在框内」的判定。
 *
 * 【作用】
 * 内核的轨道区没有 DOM 内容层，框选必须由几何计算得出。本模块只做**几何**：
 * 选择的合并语义（是否保留原有选择、主修饰键切换）复用既有
 * `useTimelineSelectionRect.computeTimelineRectSelection`——两处各写一份合并规则
 * 会让「按修饰键框选」的行为在两种渲染模式下分叉。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 的 `box-select` 手势在每帧调用。
 * - 下游：回调把命中的 clip id 列表交给 React 侧合并与提交。
 * - 独立性：纯函数，无 DOM / React 依赖，可直接单测。
 *
 * 【设计约束】
 * 1. 矩形一律**规范化**（`left <= right`、`top <= bottom`）：用户可能从右下往
 *    左上拖，若各处自行比较起止点，判定会散落成四种方向分支。
 * 2. 相交判定用**闭区间**（相切即算命中）：与拖拽选择的手感一致——框刚好压住
 *    clip 边缘时用户期望它被选中。
 */

/** 内容坐标下的矩形边界。 */
export interface BoxBounds {
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}

/**
 * 由起止点构造规范化矩形。
 *
 * @param startContentX 起点内容坐标 x。
 * @param startContentY 起点内容坐标 y。
 * @param currentContentX 当前点内容坐标 x。
 * @param currentContentY 当前点内容坐标 y。
 * @returns 规范化后的矩形（left <= right、top <= bottom）。
 */
export function resolveBoxBounds(
    startContentX: number,
    startContentY: number,
    currentContentX: number,
    currentContentY: number,
): BoxBounds {
    const x1 = Number.isFinite(startContentX) ? startContentX : 0;
    const y1 = Number.isFinite(startContentY) ? startContentY : 0;
    const x2 = Number.isFinite(currentContentX) ? currentContentX : 0;
    const y2 = Number.isFinite(currentContentY) ? currentContentY : 0;
    return {
        left: Math.min(x1, x2),
        right: Math.max(x1, x2),
        top: Math.min(y1, y2),
        bottom: Math.max(y1, y2),
    };
}

/** clip 的矩形相交判定参数。 */
export interface ClipBoxIntersectArgs {
    readonly box: BoxBounds;
    readonly clipStartSec: number;
    readonly clipLengthSec: number;
    /** clip 所在轨道下标。 */
    readonly trackIndex: number;
    readonly pxPerSec: number;
    readonly rowHeight: number;
}

/**
 * 判定 clip 的矩形是否与框相交（闭区间，相切即命中）。
 *
 * @param args 判定参数。
 * @returns 相交时为 true。
 */
export function clipIntersectsBox(args: ClipBoxIntersectArgs): boolean {
    const pxPerSec = Number.isFinite(args.pxPerSec) && args.pxPerSec > 0 ? args.pxPerSec : 0;
    if (pxPerSec <= 0 || args.trackIndex < 0) return false;
    const rowHeight = Number.isFinite(args.rowHeight) && args.rowHeight > 0 ? args.rowHeight : 1;
    const clipLeft = args.clipStartSec * pxPerSec;
    const clipRight = (args.clipStartSec + Math.max(0, args.clipLengthSec)) * pxPerSec;
    const clipTop = args.trackIndex * rowHeight;
    const clipBottom = clipTop + rowHeight;
    if (clipRight < args.box.left || clipLeft > args.box.right) return false;
    if (clipBottom < args.box.top || clipTop > args.box.bottom) return false;
    return true;
}
