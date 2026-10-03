import { useEffect, useRef, useState, type RefObject } from "react";

import { TOOLBAR_MAX_TIER, nextToolbarTier } from "./toolbarOverflow";

/**
 * 测量"自然宽度"时把行临时撑到的宽度（px）。
 *
 * 只要大于任何可能的内容宽度即可：目的是让所有子项都回到**不受挤压**的自然尺寸。
 */
const NATURAL_MEASURE_WIDTH_PX = 10000;

/**
 * 量出一行工具栏的「可见宽度」与「内容自然宽度」。
 *
 * # 为什么必须量"与当前宽度无关"的自然宽度
 * 工具栏里有个**可伸缩**的元素（平滑度滑块 `flex: 1 1 0; minWidth: 16`）：可用宽度
 * 一变，它就被挤窄，于是按当前布局量到的"内容宽度"也跟着变（实测 1600px 时 1202、
 * 1150px 时 1134、1060px 时 1058）。这样的量值**不能**当判据：
 * - 同一层级在两次测量里数值不同 ⇒ 被误判成"内容变了"，缓存反复失效、层级反复归零；
 * - 恢复判据拿到的"上一级宽度"会随拖拽漂移 ⇒ 固定点不存在，拖拽时来回切换。
 *
 * # 做法
 * 1. 把行宽临时撑大（`NATURAL_MEASURE_WIDTH_PX`），让所有子项回到自然尺寸 ——
 *    滑块回到 120px 上限，不再随可用宽度伸缩；
 * 2. 关掉带 `flex-grow` 的子项：否则"填充剩余空间"的那个子项会吃掉余量，
 *    量到的是可用宽度而不是内容宽度；
 * 3. 内容需求 = Σ max(子项 `scrollWidth`, `clientWidth`) + 间距。
 *
 * 两步都只改行的内联样式、同一帧内还原，浏览器不会绘制中间态。
 *
 * 前提：工具栏内的文字**不折行**（容器带 `whitespace-nowrap`）。否则中文标签会被
 * 压成逐字折行、"内容宽度"随之消失，量到的永远是"放得下"。
 */
function measureRow(row: HTMLElement): { available: number; needed: number } {
    const available = row.clientWidth;
    const children = Array.from(row.children).filter(
        (child): child is HTMLElement => child instanceof HTMLElement,
    );
    if (children.length === 0) return { available, needed: available };

    const gap = Number.parseFloat(getComputedStyle(row).columnGap) || 0;

    const previousWidth = row.style.width;
    row.style.width = `${NATURAL_MEASURE_WIDTH_PX}px`;

    const grown: Array<{ el: HTMLElement; previous: string }> = [];
    for (const child of children) {
        if ((Number.parseFloat(getComputedStyle(child).flexGrow) || 0) > 0) {
            grown.push({ el: child, previous: child.style.flexGrow });
            child.style.flexGrow = "0";
        }
    }

    let needed = 0;
    children.forEach((child, index) => {
        if (index > 0) needed += gap;
        needed += Math.max(child.scrollWidth, child.clientWidth);
    });

    for (const { el, previous } of grown) el.style.flexGrow = previous;
    row.style.width = previousWidth;

    return { available, needed };
}

/**
 * 观察工具栏行容器，返回当前应隐藏到第几级。
 *
 * # 触发时机（两条路，缺一不可）
 * - `ResizeObserver`：容器尺寸变化（拖拽分栏 / 浮窗拉伸）——**不会**触发 React 渲染，
 *   所以必须自己重测。
 * - `MutationObserver`：行内**内容**变化（参数增删、语系切换、门禁生效）——
 *   行的边框盒尺寸不变，`ResizeObserver` 看不到，只有 DOM 变动能捕捉到。
 *
 * # 为什么不每次渲染都测
 * 读尺寸会强制一次同步布局；本面板渲染频繁，每次都读等于把布局拖进渲染路径。
 *
 * # 不在观察回调里同步 setState
 * `ResizeObserver` 回调里同步 setState 会形成"每帧投递"的永久循环
 * （`PianoRollPanel` 里已有同类事故的注释记录）。这里统一用
 * `requestAnimationFrame` 合帧，同帧内多次触发只测一次。
 *
 * @param rowRef 工具栏行容器的 ref
 * @param maxTier 最大隐藏层级
 * @returns 当前隐藏层级；`0` = 全部显示
 */
export function useToolbarOverflowTier(
    rowRef: RefObject<HTMLElement | null>,
    maxTier: number = TOOLBAR_MAX_TIER,
): number {
    const [tier, setTier] = useState(0);

    /**
     * 各层级**实测**的内容自然宽度。恢复判据用的是"少隐藏一级"时的实测值
     * （见 `nextToolbarTier`），因此需要把走过每一级时量到的宽度留下来。
     */
    const neededByTierRef = useRef(new Map<number, number>());

    useEffect(() => {
        let frame = 0;

        const measure = () => {
            frame = 0;
            const row = rowRef.current;
            if (!row) return;

            const { available, needed } = measureRow(row);
            const cache = neededByTierRef.current;

            // 同一层级两次量到的宽度不一致 ⇒ 内容变了（参数增删 / 语系切换）。
            // 已缓存各级的宽度随之作废；退回 0 级重新走一遍，避免拿过期宽度做恢复判据。
            const cached = cache.get(tier);
            if (cached !== undefined && Math.abs(cached - needed) > 1) {
                cache.clear();
                if (tier !== 0) {
                    setTier(0);
                    return;
                }
            }

            cache.set(tier, needed);
            const next = nextToolbarTier({
                currentTier: tier,
                maxTier,
                available,
                needed,
                neededAtLowerTier: tier > 0 ? cache.get(tier - 1) : undefined,
            });
            if (next !== tier) setTier(next);
        };

        const schedule = () => {
            if (frame !== 0) return;
            frame = requestAnimationFrame(measure);
        };

        const cleanups: Array<() => void> = [];
        const row = rowRef.current;
        if (row) {
            const resizeObserver = new ResizeObserver(schedule);
            resizeObserver.observe(row);
            cleanups.push(() => resizeObserver.disconnect());

            const mutationObserver = new MutationObserver(schedule);
            mutationObserver.observe(row, {
                childList: true,
                subtree: true,
                characterData: true,
            });
            cleanups.push(() => mutationObserver.disconnect());
        }

        // 首帧（或层级变化后）：DOM 已提交，可以量到真实尺寸。
        // 层级变化会重跑本 effect，因此"隐藏一级 → 再测一轮"的收敛链是显式接上的，
        // 不依赖"自身渲染引发的 DOM 变动恰好被观察到"。
        schedule();

        return () => {
            if (frame !== 0) cancelAnimationFrame(frame);
            for (const cleanup of cleanups) cleanup();
        };
    }, [rowRef, maxTier, tier]);

    return tier;
}
