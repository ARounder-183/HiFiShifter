/**
 * 时间轴渲染内核 · 拖拽边缘自动滚屏
 *
 * 【主要内容】
 * 把「拖拽期间指针停在视口边缘」解析为**每帧的水平滚屏步长**，并驱动
 * `ScrollKernel` 滚屏：
 * - `resolveDragEdgeScroll()`：指针位置 + 容器边界 + 每帧毫秒 → 步长（CSS px）；
 * - `DRAG_EDGE_SCROLL_BAND_PX` / `DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC` 常量；
 * - `shouldAutoScrollForGesture()`：哪些手势需要自动滚屏。
 *
 * 【作用：为什么必须有它】
 * 内核自绘滚动没有原生滚动容器，浏览器**不会**在拖拽靠近边缘时自动滚屏。
 * 没有这一层时，clip 一旦被拖到指针所能到达的视口边缘就再也无法继续右移——
 * 「使劲向右拖会被卡住」的第二个成因（第一个是已移除的 `projectSec` 自指上界，
 * 见 `interaction/dragGeometry` 的说明）。设计文档
 * `docs/plans/2026-09-11-timeline-unified-render-kernel-design.md` 早已写明
 * 「拖拽到边缘时驱动 ScrollKernel 自动滚动」，但一直未实现。
 *
 * 【几何现在与参数编辑器共用】
 * 步长公式、边缘带宽与比例上限都搬到了
 * `components/layout/shared/edgeAutoScroll.ts` —— 此前两个内核各写一份**实现**，
 * 是同一个公式的两个副本。共享之后公式只有一个定义处。
 * 两边保留各自的速度常量是**有意的**：时间轴 720 px/秒（clip 更宽，滚太快难以
 * 精确落点），参数编辑器 1080 px/秒（曲线编辑需要更快跨过工程）。
 * 本模块的 `DRAG_EDGE_SCROLL_*` 名字不变，时间轴侧的调用点与单测因此零改动。
 *
 * 【与其他模块的关系】
 * - 上游：宿主 `host/timelineKernelHost` 在 `clip-drag` / `clip-trim` /
 *   `clip-fade` 等手势的拖拽帧里调用（见 `applyDragPreview` 调用点）。
 * - 下游：宿主把步长交给 `ScrollKernel.setScrollLeft`（钳制在那里统一做）。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测（本模块的测试即回归锁）。
 */

import { EDGE_SCROLL_BAND_PX, resolveEdgeScrollDeltaPx } from "../../../shared/edgeAutoScroll";

/** 边缘触发带宽（CSS px）：指针进入距视口边缘这么近时开始自动滚屏。 */
export const DRAG_EDGE_SCROLL_BAND_PX = EDGE_SCROLL_BAND_PX;

/**
 * 自动滚屏的最大速度（CSS px/秒）。
 *
 * 取值参考：约等于"60Hz 下每帧 12px"，比参数编辑器内核的 18px/帧 略保守
 * ——时间轴的 clip 更宽（全览时一个 clip 可能横跨半个视口），滚太快难以精确落点。
 */
export const DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC = 720;

/**
 * 把「指针相对视口边缘的位置」解析为**本帧**的水平滚屏步长。
 *
 * 完整语义（带宽内线性加速、比例上限 1.5、方向符号、帧时长钳制与回退）见
 * `shared/edgeAutoScroll.ts` 的 `resolveEdgeScrollDeltaPx` —— 本函数是它在时间轴
 * 侧的入口，只额外绑定本面板的最大速度。
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
    return resolveEdgeScrollDeltaPx({
        ...args,
        maxSpeedPxPerSec: DRAG_EDGE_SCROLL_MAX_SPEED_PX_PER_SEC,
    });
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
