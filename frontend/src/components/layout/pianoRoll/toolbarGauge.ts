/**
 * 工具栏「看不见的影子测量」：一次量出**每个隐藏层级**的内容宽度。
 *
 * # 为什么需要它
 * 旧实现把"测量"和"呈现"绑在同一个 state 上：内容一变就把层级**重置为 0**，再指望
 * 逐帧收敛（每次只肯藏一级）。用户看到的就是"先撑成完整长度、再一级一级收回去"——
 * 最多 9 帧、约 150ms 的可见抽动，且期间被让位的药丸会重新点亮（副参数曲线跟着闪）。
 *
 * 根因是结构性的：**测量必须改变被测量的对象**。只要"量第 N 级"意味着"真的渲染到
 * 第 N 级"，就必然要经过中间态。
 *
 * # 做法
 * 把测量搬到**不可见的克隆**上：克隆里所有可让位项都仍在 DOM 中（真实 DOM 里它们也
 * 一直挂着，只由 `data-hide-at` 控制显隐），于是可以在同一个任务里把克隆逐级隐藏、
 * 逐级读 `scrollWidth`，拿到完整的宽度表。调用方据此一次决策 —— 真实 DOM 只变一次，
 * 浏览器一次都没画过中间态。
 *
 * # 为什么用 max-content 容器
 * 真实行里的项会被 flex 压缩，量到的是"当前被压扁后的宽度"。容器给 `max-content` 之后
 * 没有任何收缩压力，量到的才是"自然宽度"——这才是判据需要的值。
 */

import { TOOLBAR_MAX_TIER } from "./toolbarOverflow";

/** 克隆里控制显隐的钩子。真实 DOM 与克隆共用同一套属性。 */
export const HIDE_ATTR = "data-hide-at";

/**
 * 把某个层级应用到一棵子树：达到 `hideAt` 的项隐藏，其余恢复成"不占盒子的包装"。
 *
 * `display: contents` 让包装元素自己不参与 flex 布局，子元素直接成为祖父的 flex 项 ——
 * 与真实 DOM 的呈现方式一致（`ToolbarSlot` 用的也是它）。
 */
function applyTier(root: HTMLElement, tier: number): void {
    for (const element of Array.from(root.querySelectorAll<HTMLElement>(`[${HIDE_ATTR}]`))) {
        const hideAt = Number(element.dataset.hideAt);
        element.style.display = Number.isFinite(hideAt) && tier >= hideAt ? "none" : "contents";
    }
}

/**
 * 量出 `tier = 0..maxTier` 各自的内容宽度（CSS px，含行的左右内边距）。
 *
 * 返回的表**单调不增**：藏得越多只会更窄。量不出来时返回空数组（调用方保持现状）。
 */
export function measureTierWidths(row: HTMLElement, maxTier = TOOLBAR_MAX_TIER): number[] {
    if (typeof document === "undefined") return [];
    const stage = document.createElement("div");
    stage.setAttribute("aria-hidden", "true");
    // 视口外 + 不绘制：既不闪、也不产生滚动条。宽度交给内容。
    stage.style.cssText =
        "position:fixed;left:-10000px;top:0;visibility:hidden;pointer-events:none;width:max-content;";
    const clone = row.cloneNode(true) as HTMLElement;
    // 去掉会与外层重复的身份与全局钩子：id 会造成重复标识，data-tooltip 会挂上
    // 真实的 Tooltip 监听器（克隆是临时节点，不该参与任何交互）。
    const candidates: HTMLElement[] = [clone, ...Array.from(clone.querySelectorAll("*"))].filter(
        (node): node is HTMLElement => node instanceof HTMLElement,
    );
    for (const element of candidates) {
        element.removeAttribute("id");
        element.removeAttribute("for");
        element.removeAttribute("aria-describedby");
        element.removeAttribute("data-tooltip");
        element.removeAttribute("data-hs-context-menu");
    }
    stage.appendChild(clone);
    document.body.appendChild(stage);
    const widths: number[] = [];
    try {
        for (let tier = 0; tier <= maxTier; tier += 1) {
            applyTier(clone, tier);
            widths.push(clone.scrollWidth);
        }
    } finally {
        stage.remove();
    }
    return widths;
}

/**
 * 在宽度表里选出**最小**的、放得下的层级。
 *
 * 全是纯函数：同一张表 + 同一宽度 ⇒ 同一结果。压缩与恢复因此走的是同一条判据，
 * 不存在"进入用阈值 A、退出用 A−H"那种两条路径排版不一致的问题。
 */
export function chooseTier(widths: readonly number[], available: number): number {
    if (widths.length === 0) return 0;
    if (!(available > 0)) return widths.length - 1;
    for (let tier = 0; tier < widths.length; tier += 1) {
        const needed = widths[tier];
        if (Number.isFinite(needed) && needed <= available) return tier;
    }
    return widths.length - 1;
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
