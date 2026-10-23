import { useEffect, useState, type RefObject } from "react";

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";
import { chooseTier, layoutSignature, measureTierFits } from "./toolbarGauge";

/**
 * 观察工具栏行容器，返回当前应隐藏到第几级。
 *
 * # 一次决策，而不是逐帧收敛
 *
 * 每次内容或宽度变化时，在**真实行**上于一个同步任务里逐级问"放不放得下"
 * （见 `measureTierFits`），再一次性算出目标层级。浏览器只在任务结束时绘制，
 * 因此用户看不到"先撑满再收回去"的中间态，也不会看到被让位的药丸闪一下。
 *
 * # 触发时机（两条路，缺一不可）
 * - `ResizeObserver`：容器尺寸变化（拖拽分栏 / 浮窗拉伸）——不会触发 React 渲染；
 * - `MutationObserver`：行内**内容**变化（参数增删、语系切换、药丸多出 chevron）。
 *
 * # 为什么先用签名过滤
 * 平滑度百分比在拖滑块时每帧都改文本；不过滤就等于把一次强制布局塞进拖拽热路径。
 * 签名只包含影响宽度的部分（见 `layoutSignature`），不变就直接返回。
 *
 * # 不在观察回调里同步 setState
 * `ResizeObserver` 回调里同步 setState 会形成"每帧投递"的永久循环。这里统一用
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

    useEffect(() => {
        let frame = 0;
        /** 上一次测过的"布局签名"，见 `layoutSignature`。 */
        let lastSignature: string | null = null;
        let lastAvailable = -1;

        const measure = () => {
            frame = 0;
            const row = rowRef.current;
            if (!row) return;
            const available = row.clientWidth;
            // 0 = 面板当前不可见（停泊中的标签页 / 浮窗收起）。此时量不到真实宽度，
            // 保留上一次的层级，等真的显示出来再测 —— 不能把它当成"窄到要全藏"。
            if (available <= 0) return;

            const signature = layoutSignature(row);
            if (signature === lastSignature && available === lastAvailable) return;
            lastSignature = signature;
            lastAvailable = available;

            // 测量在真实行上完成，并把行留在选中的那一级（见 `measureTierFits`）。
            const fitsByTier = measureTierFits(row, maxTier);
            if (fitsByTier.length === 0) return;
            const next = chooseTier(fitsByTier);
            setTier((previous) => (previous === next ? previous : next));
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
            mutationObserver.observe(row, { childList: true, subtree: true, characterData: true });
            cleanups.push(() => mutationObserver.disconnect());
        }

        // 首帧：DOM 已提交，可以量到真实尺寸。
        schedule();

        return () => {
            if (frame !== 0) cancelAnimationFrame(frame);
            for (const cleanup of cleanups) cleanup();
        };
        // 刻意**不**依赖 `tier`：层级变化只改样式，不改变内容宽度，再测一遍是白跑；
        // 旧实现把 `tier` 放进依赖里，于是每降一级都要再等一帧，形成可见的阶梯。
    }, [rowRef, maxTier]);

    return tier;
}
