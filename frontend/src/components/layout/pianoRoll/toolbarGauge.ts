/**
 * 工具栏「放不放得下」的判据：在**真实行**上、一个同步任务里逐级试一遍。
 *
 * # 为什么需要它
 * 旧实现把"测量"和"呈现"绑在同一个 state 上：内容一变就把层级**重置为 0**，再指望
 * 逐帧收敛（每次只肯藏一级）。用户看到的就是"先撑成完整长度、再一级一级收回去"——
 * 最多 9 帧、约 150ms 的可见抽动，且期间被让位的药丸会重新点亮（副参数曲线跟着闪）。
 *
 * # 为什么不再用"不可见的克隆"
 *
 * 上一版把行克隆到视口外，撑到 `max-content` 再读 `scrollWidth` 当"这一级需要多宽"。
 * 两条都错，且实测差 ~100px：
 *
 * 1. `max-content` 对 `flex: 1 1 0` 的子项（平滑度滑块）与 `full-width` 的下拉给出 0，
 *    判据因此永远觉得"还放得下"；
 * 2. 克隆与真实行的布局并不等价（实测同一宽度下左 / 右簇的分配差 10px），
 *    于是量到的是"另一个页面"的答案。
 *
 * 现在直接量真实行：**所有层级在同一个同步任务里试完**。浏览器只在任务结束时绘制，
 * 因此中间态一次都不会出现在屏幕上 —— 这正是原设计想要的性质，而且不需要克隆。
 *
 * # 判据：溢出，而不是"内容宽度"
 *
 * 把可用宽度摆好之后问：**它溢出吗？** 两处都要看：
 *
 * 1. 行本身 `scrollWidth > clientWidth` —— 右簇（`min-width: auto`）放不下；
 * 2. **任一直接子项** `scrollWidth > clientWidth` —— 左簇（`min-width: 0`）被压到
 *    内容宽度以下。这一条是上一版完全看不见的那一半，也正是"控件互相重叠"的直接成因。
 *
 * 两者合起来 = "用户会看到重叠或裁切"，这才是"放不下"的定义。
 */

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";

/** 克隆与真实 DOM 共用的显隐钩子。 */
export const HIDE_ATTR = "data-hide-at";

/**
 * 把某个层级应用到一组槽位：达到 `hideAt` 的项隐藏，其余恢复成"不占盒子的包装"。
 *
 * `display: contents` 让包装元素自己不参与 flex 布局，子元素直接成为祖父的 flex 项 ——
 * 与 `ToolbarSlot` 的呈现方式一致。
 */
function applyTier(slots: readonly HTMLElement[], tier: number): void {
    for (const element of slots) {
        const hideAt = Number(element.dataset.hideAt);
        element.style.display = Number.isFinite(hideAt) && tier >= hideAt ? "none" : "contents";
    }
}

/** 元素是否被压到自己的内容宽度以下（= 内部正在互相重叠 / 被裁掉）。 */
function isSqueezed(element: HTMLElement): boolean {
    return element.scrollWidth > element.clientWidth + 1;
}

/** 这一级在当前可用宽度下放不放得下（见模块头的两条判据）。 */
function fits(row: HTMLElement): boolean {
    if (isSqueezed(row)) return false;
    return !Array.from(row.children).some(
        (child) => child instanceof HTMLElement && isSqueezed(child),
    );
}

/**
 * 量出 `tier = 0..maxTier` 各自放不放得下，并把行**留在选中的那一级**上。
 *
 * 返回的表单调不降（藏得越多越容易放下）。量不出来时返回空数组（调用方保持现状）。
 *
 * 【为什么它自己写回最终层级】测量期间行被逐级改过 `display`。函数返回时若停在
 * "最后试过的那一级"，在 React 提交新状态之前的那一帧就会显示错层级。这里直接写回
 * 选中的那一级，于是**从测量到呈现没有中间态**，与是否走 React 无关。
 *
 * @param row 工具栏行容器（真实 DOM）
 * @param maxTier 最大隐藏层级
 */
export function measureTierFits(row: HTMLElement, maxTier = TOOLBAR_MAX_TIER): boolean[] {
    if (typeof document === "undefined") return [];
    const slots = Array.from(row.querySelectorAll<HTMLElement>(`[${HIDE_ATTR}]`));
    if (slots.length === 0) return [];
    const fitsByTier: boolean[] = [];
    try {
        for (let tier = 0; tier <= maxTier; tier += 1) {
            applyTier(slots, tier);
            fitsByTier.push(fits(row));
        }
    } finally {
        applyTier(slots, chooseTier(fitsByTier));
    }
    return fitsByTier;
}

/**
 * 在"哪一级放得下"的表里选出**最小**的、放得下的那一级；全放不下时停在最大级。
 *
 * 纯函数：同一张表 ⇒ 同一结果。压缩与恢复因此走的是同一条判据，不存在"进入用
 * 阈值 A、退出用 A−H"那种两条路径排版不一致的问题。
 */
export function chooseTier(fitsByTier: readonly boolean[]): number {
    if (fitsByTier.length === 0) return 0;
    const firstFitting = fitsByTier.indexOf(true);
    return firstFitting >= 0 ? firstFitting : fitsByTier.length - 1;
}

/**
 * 内容"布局签名"：只包含会影响宽度的部分。
 *
 * 【为什么要它】MutationObserver 会为平滑度百分比的每次文本变化回调（拖滑块时每帧一次）。
 * 若每次都跑一遍克隆测量，就等于把强制布局塞进拖拽热路径。签名不变 ⇒ 直接返回。
 *
 * 取舍：对带 `data-toolbar-volatile` 的槽位只记长度（"100% → 99%"不触发，但
 * "9% → 10%" 会）；其余槽位记全文（"PIT → FRM" 必须触发），并附上节点数量
 * （药丸多出一个 chevron 段时文本不变、结构变了）。
 */
export function layoutSignature(row: HTMLElement): string {
    return Array.from(row.querySelectorAll<HTMLElement>(`[${HIDE_ATTR}]`))
        .map((element) => {
            const text = element.textContent ?? "";
            const volatile = element.querySelector("[data-toolbar-volatile]") !== null;
            const structural = element.querySelectorAll("*").length;
            return `${element.dataset.hideAt}|${volatile ? text.length : text}|${structural}`;
        })
        .join("~");
}
