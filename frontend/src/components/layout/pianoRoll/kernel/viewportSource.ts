/**
 * 参数编辑器 · 面板侧渲染投影的视口来源。
 *
 * 【主要内容】
 * 决定面板自己的绘制（Canvas2D 细节层、曲线图层的**可见段选择**、波形广播、标尺播放头）
 * 用哪一份 `{pxPerSec, scrollLeft}`。
 *
 * 【为什么不能直接用面板的 refs（这是一个真实缺陷的根因）】
 * 面板的 `pxPerSecRef` / `scrollLeftRef` 在**渲染期**就被同步成 React state 的新值
 * （为了别处 emit 读到最新值）。而 React 允许"渲染但尚未提交"（并发渲染可被更高优先级
 * 更新打断并丢弃），此时内核仍是旧值、DOM 仍是旧值——只有 ref 提前变了。于是任何一次
 * 绘制都会拿到一对"混合视口"：面板的 Canvas2D / 曲线可见段按新视口画，而宿主 GL
 * （网格 / 键盘 / 曲线 / 播放头）与 DOM（标尺平移 / 网格层）按内核的旧视口画。
 * 表现就是"这些线偏移了"。
 *
 * 时间轴侧早已用同一思路解决（`useTimelineState.livePxPerSec`：逐帧发布一律取**内核
 * 真值**，不取渲染期 ref）。本函数把这套约定搬到参数编辑器：只要宿存在，渲染投影一律
 * 以内核视口为准；只有宿主尚未创建（挂载期）时才退回 refs。
 *
 * 【同一个约定必须也覆盖**交互**路径（一个真实缺陷的根因）】
 * 本函数此前只被渲染路径使用，而**指针 → 拍**的换算（框选起点/终点、选区边缘命中、
 * 标尺 seek）仍在交互 hook 里自建轴、直接读 `scrollLeftRef`。由于 `scrollLeftRef` 在
 * 渲染期被 256px 量化的 React state 回写，滚动之后它最多滞后内核 255px，于是：
 * 画面上的选区块按内核绘制、而框选换算按滞后的 ref 计算——用户划定区域的落点与鼠标
 * 划过的区域相差同一距离（"选区偏移"）。交互侧现已改为走同一份内核真值
 * （`usePianoRollInteractions` 的 `getViewportTruth`），本文件因此成为两类路径
 * **共同的**视口来源契约。
 *
 * 特殊说明：内核视口对外是**绘制坐标**（原生值减同步偏移），与面板各图层的口径一致，
 * 因此这里不需要再做任何换算。
 */
export interface PanelRenderViewportArgs {
    /** 内核视口（`getViewport()`；未挂载时为 null）。 */
    kernelView: { pxPerSec: number; scrollLeft: number } | null;
    /** 渲染期同步的横向缩放（兜底）。 */
    refPxPerSec: number;
    /** 渲染期同步的横向位置（**绘制坐标**，兜底）。 */
    refScrollLeftPx: number;
}

/** 渲染投影的视口（绘制坐标）。 */
export interface PanelRenderViewport {
    pxPerSec: number;
    scrollLeftPx: number;
}

/**
 * 解析面板渲染投影应使用的视口。
 *
 * 流程：内核视口存在且合法 → 取内核；否则退回渲染期 refs。
 *
 * @param args 见 `PanelRenderViewportArgs`。
 * @returns 绘制坐标下的 `{pxPerSec, scrollLeftPx}`。
 */
export function resolvePanelRenderViewport(args: PanelRenderViewportArgs): PanelRenderViewport {
    const view = args.kernelView;
    if (view !== null && Number.isFinite(view.pxPerSec) && Number.isFinite(view.scrollLeft)) {
        return { pxPerSec: view.pxPerSec, scrollLeftPx: view.scrollLeft };
    }
    return {
        pxPerSec: Number.isFinite(args.refPxPerSec) ? args.refPxPerSec : 0,
        scrollLeftPx: Number.isFinite(args.refScrollLeftPx) ? args.refScrollLeftPx : 0,
    };
}
