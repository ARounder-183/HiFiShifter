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
 * # 三条必须守住的约束
 *
 * 1. **判据只能有一份**：显隐规则来自 `toolbarOverflow.slotDisplay`，与渲染侧的
 *    `ToolbarSlot` 是同一个函数。各写一份迟早会分叉，而分叉的表现是"量到的不是用户
 *    看到的那一页"。
 * 2. **预算必须固定**：可用宽度在测量开始时取一次，全程用同一个数。若在测量途中
 *    重新读 `row.clientWidth`，一张表就横跨了多个宽度，对不上任何一帧。
 * 3. **内容不得依赖层级**：见 `toolbarOverflow` 模块头的铁律。测量只负责"试显隐"，
 *    一旦某一级会改变内容（例如把控件从 DOM 摘掉），判据就变成自反馈并开始振荡。
 *
 * # 判据：溢出，而不是"内容宽度"
 *
 * 把可用宽度摆好之后问：**它溢出吗？** 两处都要看：
 *
 * 1. 行本身 `scrollWidth > 可用宽度` —— 右簇（`min-width: auto`）放不下；
 * 2. **任一直接子项** `scrollWidth > clientWidth` —— 左簇（`min-width: 0`）被压到
 *    内容宽度以下。这一条是上一版完全看不见的那一半，也正是"控件互相重叠"的直接成因。
 *
 * 两者合起来 = "用户会看到重叠或裁切"，这才是"放不下"的定义。
 */

import { TOOLBAR_MAX_TIER, slotDisplay, type SlotRange } from "./toolbarOverflow";

/** 槽位"从第几级起隐藏"的属性（渲染与测量共用）。 */
export const HIDE_AT_ATTR = "data-hide-at";

/** 槽位"从第几级起才显示"的属性（「⋯」按钮用）。 */
export const SHOW_AT_ATTR = "data-show-at";

const SLOT_SELECTOR = `[${HIDE_AT_ATTR}], [${SHOW_AT_ATTR}]`;

/** 读一个 `data-*` 数值属性：缺失或非数值都当成"没写"。 */
function numberOrNull(raw: string | undefined): number | null {
    if (raw === undefined) return null;
    const value = Number(raw);
    return Number.isFinite(value) ? value : null;
}

/** 从 DOM 读出一个槽位的显隐区间（与 `ToolbarSlot` 写下的属性一一对应）。 */
export function readSlotRange(element: HTMLElement): SlotRange {
    return {
        hideAt: numberOrNull(element.dataset.hideAt),
        showAt: numberOrNull(element.dataset.showAt),
    };
}

/**
 * 把某个层级应用到一组槽位。
 *
 * 这是**测量侧**的渲染：与 `ToolbarSlot` 走同一个 `slotDisplay`，因此"试"与"呈现"
 * 不会分叉。副作用只碰 `style.display`，不碰内容。
 */
function applyTier(slots: readonly HTMLElement[], tier: number): void {
    for (const element of slots) {
        element.style.display = slotDisplay(readSlotRange(element), tier);
    }
}

/** 元素是否被压到自己的内容宽度以下（= 内部正在互相重叠 / 被裁掉）。 */
function isSqueezed(element: HTMLElement): boolean {
    return element.scrollWidth > element.clientWidth + 1;
}

/** 这一级在给定可用宽度下放不放得下（见模块头的两条判据）。 */
function fits(row: HTMLElement, available: number): boolean {
    if (row.scrollWidth > available + 1) return false;
    return !Array.from(row.children).some(
        (child) => child instanceof HTMLElement && isSqueezed(child),
    );
}

/**
 * 量出 `tier = 0..maxTier` 各自放不放得下，并把行**留在要呈现的那一级**上。
 *
 * 返回的表单调不降（藏得越多越容易放下），且在第一个"放得下"的层级处**截断** ——
 * `chooseTier` 取的就是第一个 true，后面的答案不影响结论，再试只是白跑布局
 * （常见情况：第 0 级就放得下，于是 1 次布局而不是 12 次）。
 *
 * 【为什么它自己写回最终层级】测量期间行被逐级改过 `display`。函数返回时若停在
 * "最后试过的那一级"，在 React 提交新状态之前的那一帧就会显示错层级。这里直接写回
 * 选中的那一级，于是**从测量到呈现没有中间态**，与是否走 React 无关。
 *
 * 【为什么失败要回到 `currentTier`】半张表的答案（例如只试到第 2 级）不是任何一帧
 * 的真实排版；若把它写回 DOM，React 会认为 `style` 与上次渲染一致而**永不修正**，
 * 工具栏就永久停在一个错层级上。
 *
 * @param row 工具栏行容器（真实 DOM）
 * @param maxTier 最大隐藏层级
 * @param currentTier 进入测量时行所在的层级（失败时的回退目标）
 */
export function measureTierFits(
    row: HTMLElement,
    maxTier = TOOLBAR_MAX_TIER,
    currentTier = 0,
): boolean[] {
    if (typeof document === "undefined") return [];
    const slots = Array.from(row.querySelectorAll<HTMLElement>(SLOT_SELECTOR));
    if (slots.length === 0) return [];

    // 预算取一次，全程不变（见模块头第 2 条）。0 = 面板当前不可见，量不到真实宽度。
    const available = row.clientWidth;
    if (available <= 0) return [];

    const fitsByTier: boolean[] = [];
    /** 结束时应当留在 DOM 上的层级；`null` = 测量未完成，回退到 `currentTier`。 */
    let landing: number | null = null;
    try {
        for (let tier = 0; tier <= maxTier; tier += 1) {
            applyTier(slots, tier);
            // 行宽在测量途中变了（行被内容撑开 / 祖先出现滚动条）：这张表横跨了多个
            // 宽度，对不上任何一帧。作废，让调用方下一帧重来。
            if (row.clientWidth !== available) return [];
            const ok = fits(row, available);
            fitsByTier.push(ok);
            if (ok) break;
        }
        landing = chooseTier(fitsByTier);
    } finally {
        applyTier(slots, landing ?? currentTier);
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
 * 若每次都跑一遍测量，就等于把强制布局塞进拖拽热路径。签名不变 ⇒ 直接返回。
 *
 * 取舍：对带 `data-toolbar-volatile` 的槽位只记长度（"100% → 99%"不触发，但
 * "9% → 10%" 会）；其余槽位记全文（"PIT → FRM" 必须触发），并附上节点数量
 * （药丸多出一个 chevron 段时文本不变、结构变了）。
 *
 * 另外两条**必须**在签名里，否则"该重测时不重测"：
 * - 槽位的显隐区间（`hideAt/showAt`）：第 10 级才出现的「⋯」按钮由 `showAt` 表达，
 *   区间变了就是另一套判据；
 * - 行的直接子节点数：行一级的条件子节点（对话框）挂载会改变可用宽度，但不改任何
 *   槽位的文本。
 */
export function layoutSignature(row: HTMLElement): string {
    const slots = Array.from(row.querySelectorAll<HTMLElement>(SLOT_SELECTOR)).map((element) => {
        const text = element.textContent ?? "";
        const volatile = element.querySelector("[data-toolbar-volatile]") !== null;
        const structural = element.querySelectorAll("*").length;
        const range = `${element.dataset.hideAt ?? ""}/${element.dataset.showAt ?? ""}`;
        return `${range}|${volatile ? text.length : text}|${structural}`;
    });
    return Array.from(row.children).length + ":" + slots.join("~");
}
