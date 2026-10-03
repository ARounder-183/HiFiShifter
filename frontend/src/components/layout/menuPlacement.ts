/**
 * 下拉菜单的可用高度（**永远向下展开**）。
 *
 * 【为什么不能向上翻转】参数编辑器本身就是一个停靠窗口，它的上方没有额外的展示
 * 区域：把菜单翻到按钮上方，只会盖住本面板自己的工具栏、甚至越过面板边界跑到别的
 * 面板上。菜单属于本面板，因此只在**面板内**向下铺开 —— 高度取"按钮下方到面板底边"
 * 的剩余空间，再压一个"不超过面板高度一半"的上限，超出的部分交给滚动条。
 *
 * 【为什么不按视口算】面板可以停靠在窗口的任意高度，按 `window.innerHeight` 算出的
 * 空间可能远大于面板本身，菜单于是会伸到面板之外。
 */

/** 菜单与容器底边之间保留的间距（CSS 像素）。 */
export const MENU_BOUNDARY_MARGIN = 8;

/** 菜单高度占容器高度的上限。 */
export const MENU_MAX_CONTAINER_FRACTION = 0.5;

/**
 * 由锚点与容器边界求菜单的最大高度。
 *
 * 不提供"向上展开"的选项：见文件头。返回 0 表示容器内确实没有空间（此时菜单只有
 * 一条滚动条的高度，但仍在容器内，不会跑到外面）。
 */
export function resolveMenuMaxHeight(input: {
    /** 锚点（按钮）在视口中的下边界。 */
    anchorBottom: number;
    /** 容器（面板）在视口中的上边界。 */
    containerTop: number;
    /** 容器（面板）在视口中的下边界。 */
    containerBottom: number;
    margin?: number;
    maxFraction?: number;
}): number {
    const margin = input.margin ?? MENU_BOUNDARY_MARGIN;
    const maxFraction = input.maxFraction ?? MENU_MAX_CONTAINER_FRACTION;
    const spaceBelow = input.containerBottom - input.anchorBottom - margin;
    const cap = Math.max(0, input.containerBottom - input.containerTop) * maxFraction;
    return Math.max(0, Math.floor(Math.min(spaceBelow, cap)));
}
