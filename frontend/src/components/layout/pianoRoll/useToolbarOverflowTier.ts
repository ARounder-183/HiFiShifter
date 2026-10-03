import { useEffect, useRef, useState, type RefObject } from "react";

import { TOOLBAR_MAX_TIER, nextToolbarTier } from "./toolbarOverflow";

/**
 * 量出一行工具栏的「可见宽度」与「内容**最小**宽度」。
 *
 * # 为什么量"最小"而不是"自然"宽度
 * 工具栏里的平滑度滑块是**故意设计成先让步**的（`flex: 1 1 0` + 16px 下限，见
 * `PianoRollPanel` 的既有注释）：横向不足时它先变窄，真的压无可压才轮到隐藏其它项。
 * 若按"自然宽度"判断，滑块永远轮不到收缩 —— 一窄就直接隐藏，等于把它的伸缩逻辑
 * 废掉（用户报告："伸缩逻辑几乎完全失效"）。
 *
 * 因此判据取"所有可压缩项都压到下限时的宽度"：可用宽度在这个值以上时靠**收缩**
 * 消化（滑块变窄），跌破它才按优先级隐藏。
 *
 * # 为什么这个量值与可用宽度无关
 * 做法是把行宽压到 0，并临时清掉子树上所有 `min-width: 0`（那是"允许压到 0"的
 * 声明，会让容器塌成 0、量不到内容下限）。此时行的 `scrollWidth` 就是内容再也压不
 * 下去的宽度 —— 它只取决于**当前显示了哪些项**，不随可用宽度漂移。
 * 这一点是判据能有稳定不动点（不来回切换）的前提。
 *
 * 只改内联样式、同一帧内还原，浏览器不会绘制中间态。
 *
 * 前提：工具栏内的文字**不折行**（容器带 `whitespace-nowrap`）。否则中文标签会被
 * 压成逐字折行、"内容宽度"随之消失，量到的永远是"放得下"。
 */
function measureRow(row: HTMLElement): { available: number; needed: number } {
    const available = row.clientWidth;

    const restore: Array<() => void> = [];
    const candidates: HTMLElement[] = [row];
    for (const el of Array.from(row.querySelectorAll<HTMLElement>("*"))) candidates.push(el);
    for (const el of candidates) {
        const computed = getComputedStyle(el);
        // `min-width: 0` 是"允许压到 0"的声明：留着它容器会塌成 0、量不到内容下限。
        // 而 `min-width: 16px` 这类**真实下限**必须保留，否则会把滑块量成可以压到 0。
        if (computed.minWidth === "0px") {
            const previous = el.style.minWidth;
            el.style.minWidth = "auto";
            restore.push(() => {
                el.style.minWidth = previous;
            });
        }
        // 同时关掉 `flex-grow`：否则行宽压到 0 时，行会被自己的 min-content 撑住
        // （实测 1218 > 自然宽度 1202），量出来的根本不是下限。
        if ((Number.parseFloat(computed.flexGrow) || 0) > 0) {
            const previous = el.style.flexGrow;
            el.style.flexGrow = "0";
            restore.push(() => {
                el.style.flexGrow = previous;
            });
        }
    }

    const previousWidth = row.style.width;
    row.style.width = "0px";
    const needed = row.scrollWidth;
    row.style.width = previousWidth;

    for (const undo of restore) undo();

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
