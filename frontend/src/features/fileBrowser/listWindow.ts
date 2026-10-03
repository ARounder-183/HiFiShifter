/**
 * 列表窗口计算（纯函数）。
 *
 * 【要解决的问题】文件浏览器把 `displayEntries` 全量渲染。一个有两万文件的目录
 * 意味着两万个行组件、十几万个 DOM 节点 —— 首次挂载要数秒，且整机（不只面板）
 * 都跟着卡。窗口化只渲染"视口 + 上下各一段缓冲"，DOM 规模与目录大小脱钩。
 *
 * 【为什么单独成模块】窗口的边界算术（首行、末行、总高度、偏移）是纯算术，与
 * Redux / DOM 无关；放在组件里就只能靠"渲染整个面板"来验证。这里可以直接单测
 * 边界：空列表、滚动到末尾、视口尚未测量（高度为 0）、行高为 0 的初始帧。
 */

/**
 * 视口上下额外渲染的行数。
 *
 * 【为什么需要】滚动是逐帧到达的：只渲染"当前可见"的话，快速滚动时会看到空白。
 * 8 行 ≈ 半屏，足以覆盖一帧内的滚动距离，同时把 DOM 规模压在几十行。
 */
export const LIST_OVERSCAN = 8;

/**
 * 视口高度尚未测量出来时使用的估值（px）。
 *
 * 【为什么不用 0】高度为 0 时"可见行数"也是 0，首帧就只会渲染 overscan 行 ——
 * 面板会出现一次可见的空窗。给一个接近真实面板高度的估值，首帧即可正常铺满；
 * 真正的测量在下一帧到达，窗口随之收敛。
 */
export const FALLBACK_VIEWPORT_PX = 400;

export interface ListWindow {
    /** 首个渲染行的下标（含）。 */
    first: number;
    /** 末个渲染行的下标（**不含**）。 */
    last: number;
    /** 撑起滚动条的全量高度。 */
    totalHeight: number;
    /** 窗口内容相对容器顶部的偏移（用于 `translateY`）。 */
    offsetTop: number;
}

export interface ListWindowInput {
    /** 列表总行数。 */
    total: number;
    /** 行高（px）。行高不一时窗口化不适用，调用方需保证等行高。 */
    rowHeight: number;
    scrollTop: number;
    viewportHeight: number;
    overscan?: number;
}

/**
 * 算出当前应当渲染的行区间。
 *
 * 约定：`last` 不含；空列表 / 非法行高返回空窗口（而不是把全部行当成窗口 ——
 * 那正是要避免的）。
 */
export function computeListWindow({
    total,
    rowHeight,
    scrollTop,
    viewportHeight,
    overscan = LIST_OVERSCAN,
}: ListWindowInput): ListWindow {
    if (total <= 0) return { first: 0, last: 0, totalHeight: 0, offsetTop: 0 };
    if (!(rowHeight > 0)) {
        // 行高还没测出来（首帧）。此时无法换算下标，退回"全量渲染"会让两万行
        // 立刻上屏 —— 宁可先渲染一屏，等测量到达再收敛。
        return {
            first: 0,
            last: Math.min(total, overscan * 2),
            totalHeight: 0,
            offsetTop: 0,
        };
    }

    const height = viewportHeight > 0 ? viewportHeight : FALLBACK_VIEWPORT_PX;
    const safeScrollTop = Math.max(0, scrollTop);
    const firstVisible = Math.floor(safeScrollTop / rowHeight);
    const visibleCount = Math.ceil(height / rowHeight);
    /*
     * `first` 必须夹到 `total - 1` 以内。
     *
     * 滚动位置**可以**超出内容高度：过滤 / 搜索把列表从两万行缩到三行时，容器的
     * `scrollTop` 仍是旧的大值（浏览器要到下一次布局才把它收回来）。不夹的话
     * `first` 会算出四万多，窗口落在列表之外 —— 列表整屏空白，而这是用户
     * "刚打完搜索词"的那一刻，看起来就像面板坏了。
     */
    const first = Math.max(0, Math.min(total - 1, firstVisible - overscan));
    const last = Math.min(total, firstVisible + visibleCount + overscan);

    return {
        first,
        last: Math.max(last, first + 1),
        totalHeight: total * rowHeight,
        offsetTop: first * rowHeight,
    };
}
