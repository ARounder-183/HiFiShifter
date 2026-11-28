/**
 * 菜单滚动与定位的**纯函数模型**。
 *
 * 【为什么单独成模块】菜单的滚动与定位有三处消费者：壳（`MenuScrollScope` 的
 * 提示行可见性）、键盘（`useMenuKeyboard` 的焦点跟随、`AppContextMenu` 的
 * `activeIndex` 跟随）、以及 rAF 滚动引擎。三处各写一遍判据必然分叉 ——
 * 最典型的症状是"提示行说还能往下滚，但键盘走到末项时又停在屏幕外"。
 *
 * 【为什么这里不碰 DOM】判据全部是数字进出。真实布局量在调用侧取得，因此这一
 * 层可以在 node 里逐值断言，不需要 jsdom 去伪造 `scrollHeight`（jsdom 里它恒为
 * 0，任何基于它的断言都会退化成"断言 undefined"）。
 */
import { clampAxisPosition } from "../components/appTooltipPosition";

/**
 * 「装得下」的容差：小于 1px 的溢出视为不滚动。
 *
 * 【为什么必须有】本仓踩过两次"滚不动的滚动条"（`index.css` 的对话框内容框、
 * `Slider` 的装饰层）。菜单里同样有 Radix 的绝对定位隐藏 `<input>`：内容其实
 * 没超，`scrollHeight` 却多出 1–2px。没有容差时菜单底部会**常驻**一条永远点不
 * 动的提示行 —— 那比没有提示更糟：它承诺了一个不存在的内容。
 */
export const MENU_SCROLL_EPSILON_PX = 1;

/** 菜单当前能往哪个方向滚。 */
export interface MenuScrollState {
    canUp: boolean;
    canDown: boolean;
}

/**
 * 由三个布局数字判出两个方向提示的可见性。
 *
 * 【为什么"装不下"是前置判据而不是逐方向判】溢出小于容差时两个方向都必须假：
 * 若按 `scrollTop > EPS` 逐条算，一张刚好装下、但被隐藏 input 撑出 1px 的菜单
 * 会在滚动后显示上方提示，而实际上它一格都滚不动。
 */
export function computeMenuScrollState(input: {
    scrollTop: number;
    clientHeight: number;
    scrollHeight: number;
}): MenuScrollState {
    const max = Math.max(0, input.scrollHeight - input.clientHeight);
    if (max <= MENU_SCROLL_EPSILON_PX) return { canUp: false, canDown: false };
    return {
        canUp: input.scrollTop > MENU_SCROLL_EPSILON_PX,
        canDown: input.scrollTop < max - MENU_SCROLL_EPSILON_PX,
    };
}

/**
 * 把目标 `scrollTop` 夹进可滚范围。
 *
 * 【为什么不用 `Math.min(Math.max(...))` 内联】滚动引擎每一帧都要用同一个口径；
 * 内联三处就等于三个"边界怎么算"的答案。
 */
export function clampMenuScrollTop(next: number, maxScroll: number): number {
    if (!Number.isFinite(next)) return 0;
    return Math.min(Math.max(0, next), Math.max(0, maxScroll));
}

/**
 * 需要额外滚动多少，才能让某一项完整落在「视口减去两端提示行」的窗口里。
 *
 * 返回 0 表示已经完整可见；负数表示要往上滚；正数表示要往下滚。
 *
 * 【为什么不直接用 `scrollIntoView({ block: "nearest" })`】它只知道"客户端可见
 * 区域"，不知道那里盖着一条 18px 的提示行 —— 于是键盘走到最后一项时，项**技术上
 * 可见**（在 clientHeight 之内）、**实际看不见**（被下方渐变吃掉）。这正是
 * 不变量 I5 要挡的东西。
 */
export function menuScrollDeltaToReveal(input: {
    /** 项相对滚动容器**视口顶端**的坐标（`rect.top - containerRect.top`）。 */
    itemTop: number;
    itemBottom: number;
    /** 滚动容器的可视窗口（通常 `0` 与 `clientHeight`）。 */
    viewTop: number;
    viewBottom: number;
    /** 上方 / 下方提示行占掉的高度（提示行不可见时传 0）。 */
    insetTop: number;
    insetBottom: number;
}): number {
    const top = input.viewTop + Math.max(0, input.insetTop);
    const bottom = input.viewBottom - Math.max(0, input.insetBottom);
    // 【先判上方】项比窗口还高时两个条件会同时成立；此时应当"露出顶部"，
    // 因为标签与图标在最上面，用户需要先读到"这是哪一项"。
    if (input.itemTop < top) return input.itemTop - top;
    if (input.itemBottom > bottom) return input.itemBottom - bottom;
    return 0;
}

/**
 * 点击提示行时翻过多少**像素**。
 *
 * `max(3×项高, 一页 − 两行)`：矮窗口里 `clientHeight - 2×itemHeight` 会退化成
 * 很小甚至负数，而"翻一页"至少要能翻过三项 —— 否则点一下几乎不动，用户会以为
 * 提示行没响应。
 */
export function computeMenuPageStep(input: { clientHeight: number; itemHeight: number }): number {
    const item = Math.max(1, input.itemHeight);
    return Math.max(3 * item, input.clientHeight - 2 * item);
}

/**
 * `PageUp` / `PageDown` 一次跨过多少**项**。
 *
 * 【为什么不复用 `computeMenuPageStep`】那个返回像素，而 PageUp/PageDown 要移动
 * 的是"当前项"，单位是项。两个量在同一个视口里恰好接近，但换算成项之后**必须
 * 至少 1**，否则键盘会一步都走不动。
 */
export function computeMenuPageItemCount(input: {
    clientHeight: number;
    itemHeight: number;
}): number {
    const item = Math.max(1, input.itemHeight);
    return Math.max(1, Math.floor(input.clientHeight / item));
}

/**
 * 菜单的落点。
 *
 * 【为什么要翻转而不是只夹紧】右键菜单此前只做夹紧：锚点在指针上，菜单往右下
 * 展开，于是**菜单的第一行永远盖住用户刚点的那个位置**。装得下时应当翻到指针的
 * 左上（原生菜单就是这样）；装不下时才退回夹紧。
 */
export interface MenuPlacementInput {
    /** 锚点（指针 / 控件下缘）的视口坐标。 */
    anchorX: number;
    anchorY: number;
    /** 菜单实测尺寸。 */
    width: number;
    height: number;
    viewportWidth: number;
    viewportHeight: number;
    edgeGap: number;
}

export interface MenuPlacement {
    left: number;
    top: number;
    flippedX: boolean;
    flippedY: boolean;
}

/**
 * 每轴独立地：① 装得下锚点之后 → 贴锚点；② 否则装得下之前 → 翻到前面；
 * ③ 否则 → 交给 `clampAxisPosition`。
 *
 * 【为什么 ③ 一定落回 `clampAxisPosition`】两条轴都装不下时（窗口比菜单还小），
 * 翻转没有任何好处，而夹紧至少保证菜单左上角可见 —— 那是唯一能滚到全部内容的
 * 起点。保持与迁移前逐值相同，是为了让"装不下"这条退化路径**没有新行为**。
 */
export function placeMenuWithinViewport(input: MenuPlacementInput): MenuPlacement {
    const fitsAfterX = input.anchorX + input.width <= input.viewportWidth - input.edgeGap;
    const fitsAfterY = input.anchorY + input.height <= input.viewportHeight - input.edgeGap;
    const fitsBeforeX = input.anchorX - input.width >= input.edgeGap;
    const fitsBeforeY = input.anchorY - input.height >= input.edgeGap;
    const flippedX = !fitsAfterX && fitsBeforeX;
    const flippedY = !fitsAfterY && fitsBeforeY;
    return {
        left: flippedX
            ? input.anchorX - input.width
            : clampAxisPosition(input.anchorX, input.width, input.viewportWidth, 0, input.edgeGap),
        top: flippedY
            ? input.anchorY - input.height
            : clampAxisPosition(
                  input.anchorY,
                  input.height,
                  input.viewportHeight,
                  0,
                  input.edgeGap,
              ),
        flippedX,
        flippedY,
    };
}

/** 缓出三次曲线：菜单滚动是"被推出去"而不是"匀速走"，收尾必须减速。 */
export function easeOutCubic(t: number): number {
    const clamped = Math.min(1, Math.max(0, t));
    return 1 - (1 - clamped) ** 3;
}

/**
 * 提示行自动滚动的**动**参数（延迟与速度）。
 *
 * 【为什么这些数在 TS 里而不在 CSS 令牌里】它们由 rAF 的数学直接消费。放进 CSS
 * 就只能靠 `getComputedStyle` 读自定义属性 —— 那在 jsdom 里恒为空串，测试会退化
 * 成"断言 undefined"。分工是：**CSS 决定长什么样，TS 决定动多快**（尺寸 18px、
 * 颜色、缓动曲线都在 CSS）。
 *
 * 【为什么分 pointer / touch 两档】触摸没有悬停：一旦开始滚就必须**立刻**到巡航
 * 速度（没有指针可以"慢慢移过去"），翻页也必须瞬时（140ms 的补间在手指底下读
 * 起来像卡住）。
 */
export const MENU_SCROLL_TUNING = {
    pointer: {
        /** 悬停多久才开始滚：短于这个值会让"划过"变成误滚。 */
        hoverDelayMs: 180,
        /** 悬停巡航速度（px/s）。 */
        hoverSpeed: 220,
        /** 速度从 40% 爬到 100% 的时长（ms）。 */
        hoverRampMs: 220,
        /** 按住多久从"翻一页"转成"持续滚动"（ms）。 */
        pressDelayMs: 420,
        /** 按住时的巡航速度（px/s，匀速）。 */
        pressSpeed: 900,
        /** 点击翻一页的补间时长（ms）。 */
        pageTweenMs: 140,
    },
    touch: {
        hoverDelayMs: 0,
        hoverSpeed: 320,
        hoverRampMs: 0,
        pressDelayMs: 420,
        pressSpeed: 900,
        pageTweenMs: 0,
    },
} as const;

export type MenuScrollPointerKind = keyof typeof MENU_SCROLL_TUNING;

/** 由指针事件类型挑档；鼠标 / 笔走 pointer 档，其余（触摸）走 touch 档。 */
export function menuScrollPointerKind(pointerType: string | undefined): MenuScrollPointerKind {
    return pointerType === "touch" ? "touch" : "pointer";
}
