import { useEffect, useRef, useState, type RefObject } from "react";

import { TOOLBAR_MAX_TIER, nextToolbarTier, type ToolbarRowMeasurement } from "./toolbarOverflow";

/**
 * 量出一行工具栏的「可见宽度」与「内容自然宽度」。
 *
 * # 为什么不能用 `row.scrollWidth` 当内容需求
 * 行里有一个**填充剩余空间**的子项（`flex: 1 1 auto`，见 `PianoRollPanel` 的
 * `参数编辑器` 组）。它会把余量吃光，于是 `scrollWidth` 恒等于 `clientWidth` ——
 * 无论内容多少都量不出差别，判据永远认为"刚好放得下"，也就永远不会隐藏、
 * 更永远不会把隐藏的东西放回来。
 *
 * # 做法
 * 1. 测量期间把带 `flex-grow` 的子项临时置 0（同一帧内还原，浏览器不会绘制中间态），
 *    让每个子项都回到自己的自然宽度；
 * 2. 内容需求 = Σ max(子项 `scrollWidth`, `clientWidth`) + 间距：
 *    被压缩的子项由 `scrollWidth` 给出**内容**宽度，未被压缩的两者相等。
 *
 * 这样得到的 `needed` 与"当前隐藏到第几级"无关地反映真实需求，配合
 * `nextToolbarTier` 的滞后带，隐藏与恢复才都能收敛。
 */
function measureRow(row: HTMLElement): ToolbarRowMeasurement {
    const available = row.clientWidth;
    const children = Array.from(row.children).filter(
        (child): child is HTMLElement => child instanceof HTMLElement,
    );
    if (children.length === 0) return { available, needed: available };

    const gap = Number.parseFloat(getComputedStyle(row).columnGap) || 0;

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
 * @param rows 待观察的行容器 ref 列表（**调用方需保证数组引用稳定**，如 `useMemo`）
 * @param maxTier 最大隐藏层级
 * @returns 当前隐藏层级；`0` = 全部显示
 */
export function useToolbarOverflowTier(
    rows: readonly RefObject<HTMLElement | null>[],
    maxTier: number = TOOLBAR_MAX_TIER,
): number {
    const [tier, setTier] = useState(0);
    /** 最近一次由观察器安装的测量函数；层级变化后用它再测一轮，保证收敛。 */
    const measureRef = useRef<() => void>(() => {});

    useEffect(() => {
        let frame = 0;

        const measure = () => {
            frame = 0;
            const measured: ToolbarRowMeasurement[] = [];
            for (const row of rows) {
                const el = row.current;
                if (el) measured.push(measureRow(el));
            }
            if (measured.length === 0) return;
            // 函数式更新：effect 因此不必依赖 `tier`，也就不会因层级变化重订阅。
            setTier((current) => {
                const next = nextToolbarTier({ currentTier: current, maxTier, rows: measured });
                return next === current ? current : next;
            });
        };
        measureRef.current = measure;

        const schedule = () => {
            if (frame !== 0) return;
            frame = requestAnimationFrame(measure);
        };

        const cleanups: Array<() => void> = [];
        for (const row of rows) {
            const el = row.current;
            if (!el) continue;

            const resizeObserver = new ResizeObserver(schedule);
            resizeObserver.observe(el);
            cleanups.push(() => resizeObserver.disconnect());

            const mutationObserver = new MutationObserver(schedule);
            mutationObserver.observe(el, {
                childList: true,
                subtree: true,
                characterData: true,
            });
            cleanups.push(() => mutationObserver.disconnect());
        }

        // 首帧：此时 DOM 已提交，可以量到真实尺寸。
        schedule();

        return () => {
            if (frame !== 0) cancelAnimationFrame(frame);
            for (const cleanup of cleanups) cleanup();
            measureRef.current = () => {};
        };
    }, [rows, maxTier]);

    // 每次层级变化后再测一轮。
    //
    // 【为什么不能只靠 MutationObserver】隐藏一级会让内容变窄，于是可能还挤
    // （需要继续隐藏）或已有多余空间（需要把上一级放回来）—— 这个"再决定一次"
    // 必须发生，否则层级会停在半路：实测只依赖 DOM 变动观察时，收窄后回不到原位，
    // 每次改宽只回落一级。这里显式接上这一环。
    //
    // 收敛性由滞后带保证：`nextToolbarTier` 的回落余量大于单级最大隐藏量，
    // 因此不会出现"升级 ↔ 降级"自我维持的来回切换。
    useEffect(() => {
        measureRef.current();
    }, [tier]);

    return tier;
}
