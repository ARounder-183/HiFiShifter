/**
 * 菜单滚动的 **DOM 侧**判据：找滚动容器、读 CSS 令牌、把某一项滚进可视窗口。
 *
 * 【为什么与 `menuScrollModel` 分家】那边是纯数字进出，可以在 node 里逐值断言；
 * 这边每一次调用都要碰真实布局，jsdom 里返回值全是 0。混在一起会让"这一层能测
 * 什么"变得含糊 —— 而含糊的单测最后都退化成"断言 undefined"。
 *
 * 【为什么与 `MenuScrollScope` 分家】`react-refresh` 要求组件文件只导出组件：
 * 混着导出工具函数会让热更新在改动的那一刻整块重挂载，菜单这种带手势状态的组件
 * 一重挂载就丢按下态。这条规则在本仓是 error，不是风格建议。
 */
import {
    MENU_SCROLL_EPSILON_PX,
    clampMenuScrollTop,
    menuScrollDeltaToReveal,
} from "./menuScrollModel";

/** 一个元素是否按 CSS 自己滚动。 */
function isScrollContainer(element: HTMLElement): boolean {
    if (typeof getComputedStyle !== "function") return false;
    const overflowY = getComputedStyle(element).overflowY;
    return overflowY === "auto" || overflowY === "scroll";
}

/**
 * 找内容所在的滚动容器。
 *
 * 【为什么"最近的 .hs-menu 不可滚"就返回 null，而不是继续往上找】`--no-scroll`
 * 的壳是 `overflow: visible`。继续往上会命中**菜单之外的**滚动区（面板、时间轴），
 * 于是"这张菜单不滚动"会被误判成"它在滚整块面板"—— 提示行随后会随面板滚动亮起。
 * 菜单的滚动范围止于菜单自己。
 */
export function menuScrollContainerOf(start: Element | null): HTMLElement | null {
    let node: Element | null = start instanceof Element ? start.parentElement : null;
    while (node instanceof HTMLElement) {
        if (node.classList.contains("hs-menu")) return isScrollContainer(node) ? node : null;
        if (isScrollContainer(node)) return node;
        node = node.parentElement;
    }
    return null;
}

/**
 * 读一个 CSS 长度令牌（px）。
 *
 * 读不到时返回 0 —— jsdom 里恒为 0，调用侧必须能接受这个退化值（所有依赖它的
 * 计算都要在 0 下仍然给出正数或 0，不能给出负数或 NaN）。
 */
export function readMenuCssPixels(element: Element | null, name: string): number {
    if (!(element instanceof Element) || typeof getComputedStyle !== "function") return 0;
    const raw = getComputedStyle(element).getPropertyValue(name).trim();
    const value = Number.parseFloat(raw);
    return Number.isFinite(value) ? value : 0;
}

/** 提示行占掉的高度（桌面 18px / 触摸 28px，见 `--qt-menu-scroll-hint-h`）。 */
export function menuScrollHintHeightOf(container: Element | null): number {
    return readMenuCssPixels(container, "--qt-menu-scroll-hint-h");
}

/** 一个菜单项的高度（见 `--qt-menu-item-h`）。 */
export function menuItemHeightOf(container: Element | null): number {
    return readMenuCssPixels(container, "--qt-menu-item-h");
}

/**
 * 把某一项滚进「视口减去两端提示行」的窗口，并让它**完整可见**。
 *
 * 【为什么不用 `scrollIntoView({ block: "nearest" })`】它只知道"客户端可见区域"，
 * 不知道那里盖着一条 18px 的提示行 —— 于是键盘走到最后一项时，项**技术上可见**、
 * **实际看不见**（被下方渐变吃掉）。
 *
 * 【为什么两端都预留提示行】滚动会改变"哪条提示行可见"，若只按当前可见的那一侧
 * 预留，就会出现"滚上去 → 上方提示行出现 → 又盖住刚露出的那一项 → 再滚"的振荡。
 * 只要菜单**可滚**，两端就都预留：判据与滚动位置无关，因此一次就收敛。
 */
export function revealMenuItem(container: HTMLElement | null, item: HTMLElement | null): void {
    if (!container || !item) return;
    if (typeof container.getBoundingClientRect !== "function") return;
    const containerRect = container.getBoundingClientRect();
    const itemRect = item.getBoundingClientRect();
    // 未布局（jsdom / display:none）：矩形全零，任何计算都没有意义。
    if (itemRect.width === 0 && itemRect.height === 0) return;
    const maxScroll = Math.max(0, container.scrollHeight - container.clientHeight);
    const inset = maxScroll > MENU_SCROLL_EPSILON_PX ? menuScrollHintHeightOf(container) : 0;
    // `clientTop` 就是上边框宽度：sticky 提示行贴的是**内边距盒**的上缘，
    // 而 `getBoundingClientRect` 报的是边框盒，两者差这一条边。
    const borderTop = container.clientTop;
    const delta = menuScrollDeltaToReveal({
        itemTop: itemRect.top - containerRect.top,
        itemBottom: itemRect.bottom - containerRect.top,
        viewTop: borderTop,
        viewBottom: borderTop + container.clientHeight,
        insetTop: inset,
        insetBottom: inset,
    });
    if (delta === 0) return;
    container.scrollTop = clampMenuScrollTop(container.scrollTop + delta, maxScroll);
}
