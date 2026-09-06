/**
 * BackgroundGrid 命令式重绘的视口来源决策。
 *
 * 【为什么需要这个决策】
 * 网格（sticky 层）有两条重绘路径：
 * - 视口总线 paint：`paint: (axis) => draw(axis.scrollLeftPx, axis.scrollTopPx)`
 *   —— 总线快照在滚动事件内同步提交，等于原生 scroller 的实际偏移（DOM 真值）。
 * - React 重绘 effect：数据依赖（ticks / contentWidth / 尺寸…）变化时触发。
 *
 * React 提交的 `scrollLeft` state 是**量化值**：`useTimelineState` 用
 * `REACT_SCROLL_STEP_PX`（256px）死区经 rAF 提交，专为窗口化/裁剪设计（下游
 * 窗口一律带 256px 缓冲）。它可**永久滞后**原生滚动 0–255px——小幅滚动后
 * 死区不再放行提交，且每帧对账只对账"总线 ↔ 原生 scroller"（两者一致即跳过），
 * 不会补提交 React state。
 *
 * 因此 React 重绘 effect **绝不能**拿量化 state 当绘制偏移：拖拽 Clip 会逐帧
 * 改变 `contentWidth` 等数据依赖，一旦触发该路径用滞后偏移重绘，帧提交器会因
 * 总线投影未变而按 `axisEquals` 去重跳过纠正，对账循环也发现不了——网格就
 * 停留在错误偏移上，直到下一次滚动/缩放。表现为"拖拽 Clip 后网格偏移"。
 *
 * 契约：提供了视口总线时，绘制偏移**一律**取总线快照（水平与竖直都是——竖直
 * 裁剪 `contentBottomPx` 也按同一快照折算，保证 React 重绘与总线 paint 输出
 * **逐像素一致**）；仅当无总线（参数编辑器过渡路径）或快照非法时才退回 props。
 */
export interface GridDrawViewport {
    /** 水平绘制偏移（内容坐标原点相对视口左缘）。 */
    scrollLeftPx: number;
    /** 竖直绘制偏移（sticky 视口的竖直平移）。 */
    scrollTopPx: number;
}

/** 视口快照的最小形态（TimelineAxis / 总线 getAxis() 均满足）。 */
export type GridViewportSnapshot = {
    scrollLeftPx: number;
    scrollTopPx: number;
};

export function resolveGridDrawViewport(args: {
    /** 权威视口快照（视口总线）；缺省/非法时退回 props。 */
    busAxis: GridViewportSnapshot | null | undefined;
    /** React 提交的量化 scrollLeft（仅无总线时兜底）。 */
    propScrollLeftPx: number | null | undefined;
    /** React 提交的竖直偏移（仅无总线时兜底）。 */
    propViewportTopPx: number | null | undefined;
}): GridDrawViewport {
    const bus = args.busAxis;
    if (
        bus &&
        Number.isFinite(bus.scrollLeftPx) &&
        Number.isFinite(bus.scrollTopPx)
    ) {
        return { scrollLeftPx: bus.scrollLeftPx, scrollTopPx: bus.scrollTopPx };
    }
    return {
        scrollLeftPx: Number.isFinite(args.propScrollLeftPx)
            ? (args.propScrollLeftPx as number)
            : 0,
        scrollTopPx: Number.isFinite(args.propViewportTopPx)
            ? (args.propViewportTopPx as number)
            : 0,
    };
}
