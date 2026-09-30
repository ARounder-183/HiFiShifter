/**
 * 下拉菜单在视口内的展开方向与最大高度。
 *
 * 【为什么需要】菜单是绝对定位的浮层，可用高度取决于**锚点在视口里的位置**：面板
 * 可以停靠在窗口的任意高度，只按 `vh` 给一个固定上限，当锚点靠下时菜单就会伸出
 * 窗口底部（用户看到的就是"菜单比窗口还长"）。这里按锚点上下两侧的实际空间决定
 * 展开方向，并把高度钳进那一侧的空间里。
 *
 * 【为什么还要 50vh 上限】即便下方空间充足，一个铺满半屏的预设列表也难用 ——
 * 高度上限让菜单保持"一眼扫完"的尺寸，超出部分交给滚动条。
 */

/** 菜单与视口边缘之间保留的间距（CSS 像素）。 */
export const MENU_VIEWPORT_MARGIN = 8;

/** 菜单高度占视口高度的上限。 */
export const MENU_MAX_VIEWPORT_FRACTION = 0.5;

export interface MenuPlacement {
    /** 展开方向：`below` = 锚点下方（默认），`above` = 锚点上方（翻转）。 */
    side: "below" | "above";
    /** 该方向上的可用最大高度（CSS 像素，已扣掉边距并应用 50vh 上限）。 */
    maxHeight: number;
}

/**
 * 由锚点位置求菜单的展开方向与最大高度。
 *
 * 规则：**哪一侧空间大就朝哪一侧**（空间相等时朝下）。这样面板停靠在窗口底部时
 * 菜单会自动向上展开，而不是硬撑出窗口。
 */
export function resolveMenuPlacement(input: {
    /** 锚点（按钮）在视口中的上边界。 */
    anchorTop: number;
    /** 锚点（按钮）在视口中的下边界。 */
    anchorBottom: number;
    /** 视口高度（`window.innerHeight`）。 */
    viewportHeight: number;
    margin?: number;
    maxFraction?: number;
}): MenuPlacement {
    const margin = input.margin ?? MENU_VIEWPORT_MARGIN;
    const maxFraction = input.maxFraction ?? MENU_MAX_VIEWPORT_FRACTION;
    const below = input.viewportHeight - input.anchorBottom - margin;
    const above = input.anchorTop - margin;
    const side: MenuPlacement["side"] = below >= above ? "below" : "above";
    const space = side === "below" ? below : above;
    const cap = input.viewportHeight * maxFraction;
    return { side, maxHeight: Math.max(0, Math.floor(Math.min(space, cap))) };
}
