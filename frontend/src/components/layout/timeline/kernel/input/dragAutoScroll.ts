/**
 * 时间轴渲染内核 · 拖拽边缘自动滚屏
 *
 * 【主要内容】
 * 把「拖拽期间指针停在视口边缘」解析为**每帧的水平滚屏步长**，并驱动
 * `ScrollKernel` 滚屏：
 * - `resolveDragEdgeScroll()` 纯函数：指针位置 + 容器边界 + 每帧毫秒 → 步长（CSS px）；
 * - `DRAG_EDGE_SCROLL_BAND_PX` / `DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC` 常量。
 *
 * 【作用：为什么必须有它】
 * 内核自绘滚动没有原生滚动容器，浏览器**不会**在拖拽靠近边缘时自动滚屏。
 * 没有这一层时，clip 一旦被拖到指针所能到达的视口边缘就再也无法继续右移——
 * 「使劲向右拖会被卡住」的第二个成因（第一个是已移除的 `projectSec` 自指上界，
 * 见 `interaction/dragGeometry` 的说明）。设计文档
 * `docs/plans/2026-09-11-timeline-unified-render-kernel-design.md` 早已写明
 * 「拖拽到边缘时驱动 ScrollKernel 自动滚动」，但一直未实现。
 *
 * 【为什么按"每秒速度"而不是"每帧步长"】
 * 逐帧步长会让滚屏速度随刷新率变化（60Hz 与 120Hz 差一倍），且掉帧时明显变慢。
 * 这里以 px/秒 表达、乘以本帧实际时长，速度与帧率解耦。
 *
 * 【为什么步长上限带 1.5 倍（而不是 1.0）】
 * 指针被 pointer capture 拖到视口外时距离会超出带宽，比例被夹到 1.5 后**不再增长**
 * ——刻意留出"拖出窗外仍能持续滚屏"的余量，同时避免越拖越快难以停下。
 *
 * 【与其他模块的关系】
 * - 上游：宿主 `host/timelineKernelHost` 在 `clip-drag` / `clip-trim` /
 *   `clip-fade` 等手势的拖拽帧里调用（见 `applyDragPreview` 调用点）。
 * - 下游：宿主把步长交给 `ScrollKernel.setScrollLeft`（钳制在那里统一做）。
 * - 复用：几何公式与参数编辑器内核的 `pianoRoll/kernel/dragArithmetic`
 *   `edgeAutoScrollDeltaPx` **同一套语义**（同样的带宽 32px 与 1.5 比例上限）；
 *   此处按秒表达速度，因此保留了独立的常量与函数，而不是跨模块强耦合。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测（本模块的测试即回归锁）。
 *
 * 【维护说明】若调整手感，**只改常量**，不要改公式结构——比例上限 1.5 与
 * "带宽内线性加速"两点都有单测钉住。
 */

/** 边缘触发带宽（CSS px）：指针进入距视口边缘这么近时开始自动滚屏。 */
export const DRAG_EDGE_SCROLL_BAND_PX = 32;

/**
 * 自动滚屏的最大速度（CSS px/秒）。
 *
 * 取值参考：约等于"60Hz 下每帧 12px"，比参数编辑器内核的 18px/帧 略保守
 * ——时间轴的 clip 更宽（全览时一个 clip 可能横跨半个视口），滚太快难以精确落点。
 */
export const DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC = 720;

/** 比例上限：指针拖出视口后不再继续加速（见文件头说明）。 */
const DRAG_EDGE_SCROLL_MAX_RATIO = 1.5;

/** 数值夹取。 */
function clampNumber(value: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return min;
    return Math.min(max, Math.max(min, value));
}

/**
 * 把「指针相对视口边缘的位置」解析为**本帧**的水平滚屏步长。
 *
 * 流程：
 * 1. 指针落在左/右边缘带内 → 按"进入带宽的比例"线性加速（上限 1.5）；
 * 2. 比例 × 最大速度 → 每秒速度；
 * 3. 每秒速度 × 本帧时长（秒）→ 本帧步长。
 *
 * 特殊说明 1：**方向符号**——左缘为负（内容左移，看到更早的时间）、右缘为正，
 * 与 `scrollLeft` 增大方向一致，调用方直接相加即可。
 *
 * 特殊说明 2：步长与**本帧时长**成正比（而不是固定值），因此掉帧时不会"滚得变慢"，
 * 高刷屏也不会"滚得变快"。
 *
 * 特殊说明 3：非有限输入一律返回 0（拖拽热路径上出现 NaN 会让滚动位置被污染，
 * 且难以定位）。帧时长非法时按 1/60 秒回退，而不是返回 0——宁可步长略不精确，
 * 也不要让自动滚屏在个别脏帧上完全停住。
 *
 * 特殊说明 4：帧时长**上限 100ms**（约 10fps）。长于它的间隔通常是"标签页被挂起
 * 后恢复"这类非真实帧，按真实时长补步会一次性跳过很远的位置；夹到 100ms 相当于
 * 「补一帧、不补一段」。
 *
 * @param args.clientX 指针的视口坐标 X。
 * @param args.leftPx 视口左边界（`getBoundingClientRect().left`）。
 * @param args.rightPx 视口右边界（`getBoundingClientRect().right`）。
 * @param args.frameMs 本帧时长（毫秒）；非法值回退 1/60 秒，上限 100ms。
 * @returns 本帧滚屏步长（CSS px，左负右正）；不在边缘带内时为 0。
 */
export function resolveDragEdgeScroll(args: {
    readonly clientX: number;
    readonly leftPx: number;
    readonly rightPx: number;
    readonly frameMs: number;
}): number {
    const { clientX, leftPx, rightPx } = args;
    if (!Number.isFinite(clientX) || !Number.isFinite(leftPx) || !Number.isFinite(rightPx)) {
        return 0;
    }
    // 视口宽为 0 / 反向（布局未就绪）：没有可用的边缘语义。
    if (!(rightPx > leftPx)) return 0;

    let ratio = 0;
    if (clientX < leftPx + DRAG_EDGE_SCROLL_BAND_PX) {
        ratio = -clampNumber(
            (leftPx + DRAG_EDGE_SCROLL_BAND_PX - clientX) / DRAG_EDGE_SCROLL_BAND_PX,
            0,
            DRAG_EDGE_SCROLL_MAX_RATIO,
        );
    } else if (clientX > rightPx - DRAG_EDGE_SCROLL_BAND_PX) {
        ratio = clampNumber(
            (clientX - (rightPx - DRAG_EDGE_SCROLL_BAND_PX)) / DRAG_EDGE_SCROLL_BAND_PX,
            0,
            DRAG_EDGE_SCROLL_MAX_RATIO,
        );
    }
    if (ratio === 0) return 0;

    const frameMs =
        Number.isFinite(args.frameMs) && args.frameMs > 0
            ? clampNumber(args.frameMs, 1, 100)
            : 1000 / 60;
    const perSecond = ratio * DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC;
    return (perSecond * frameMs) / 1000;
}

/**
 * 判断某个手势种类是否应该在拖拽到边缘时自动滚屏。
 *
 * 【为什么要限定种类】只有"位置/长度随指针横向移动"的手势才需要滚屏：
 * `clip-drag`（移动）、`clip-trim`（改长度）、`clip-fade`（改淡变）、
 * `snap-offset-drag`（改吸附偏移）——它们的语义都与"更远的时间位置"有关。
 * `gain-drag`（增益只跟纵向有关）与 `crossfade-grip`（同轨相邻边缘，横向极短）
 * 不滚屏，否则会在用户只想调增益时把视口带走。
 *
 * 特殊说明：`box-select`（框选）在时间轴由右键触发，其边缘滚屏由框选自身处理，
 * 不在本函数范围内。
 *
 * @param kind 手势种类（与 `interaction/moveDispatch` 的联合一致）。
 * @returns 需要边缘自动滚屏时为 true。
 */
export function shouldAutoScrollForGesture(kind: string): boolean {
    return (
        kind === "clip-drag" ||
        kind === "clip-trim" ||
        kind === "clip-fade" ||
        kind === "snap-offset-drag"
    );
}
