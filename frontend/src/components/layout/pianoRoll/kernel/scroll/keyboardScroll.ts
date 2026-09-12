/**
 * 参数编辑器内核 · 竖向键盘滚动目标解析。
 *
 * 【主要内容】
 * 把 PageUp / PageDown / Home / End 四个按键解析成目标竖向滚动位置。
 *
 * 【作用：为什么必须由内核自己实现】
 * 内核模式下原生 scroller 是**被动镜像**，其 `scroll` 事件会被当作回声忽略
 * （宿主每帧把内核真值写回 DOM，事件不带来源、无法与用户输入区分）。
 * 旧实现里这四个键是靠**浏览器的原生滚动**生效的——忽略回声后
 * 原生滚动的那次变化就没人采纳了，键盘翻页会**彻底失效**。因此必须像时间轴内核
 * 已经做的那样（`timelineKernelHost` 的 `onKeyDown`），把键盘滚动收进内核。
 *
 * 【翻页量为什么是「视口高 − 20」而不是「视口高」】
 * 这是 Chromium 自身的翻页量（`kPageOverlap = 20`，留 20px 重叠便于连续阅读）。
 * 旧实现走原生滚动，实测（5 组视口高度，误差 0）就是该式：
 * 视口高 823 → 步进 803；623 → 603；323 → 303；423 → 403；1023 → 1003。
 * 用整屏会与迁移前的位移不一致，属手感变更，不在本次迁移范围。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 的 keydown 监听调用本模块。
 * - 下游：目标值交给 `ScrollKernel.setScrollTop`（钳制在那里统一做）。
 * - 独立性：纯函数，不依赖 DOM / React。
 */

/** 与 Chromium 原生翻页量对齐的视口重叠量（CSS px）。 */
export const PAGE_SCROLL_OVERLAP_PX = 20;

/** 内核竖向位置的上限（与 `PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX` 同值）。 */
export interface KeyboardScrollArgs {
    /** 按键名（`KeyboardEvent.key`，大小写不敏感）。 */
    readonly key: string;
    /** 当前竖向位置（CSS px）。 */
    readonly scrollTopPx: number;
    /** 视口高度（CSS px），决定翻页量。 */
    readonly viewportHeightPx: number;
    /** 竖向滚动上限（CSS px）。 */
    readonly maxScrollTopPx: number;
}

/**
 * 解析一个按键对应的竖向滚动目标。
 *
 * 流程：归一化按键名 → 按四个键分别给出目标（Home 取 0、End 取上限、
 * PageUp/PageDown 按「视口高 − 20」整页位移）→ **不做钳制**，交由内核统一处理。
 *
 * 特殊说明 1：**不钳制**是有意的。内核是唯一做钳制的地方（`ScrollKernel` 的强制
 * 约束），这里再夹一次会出现两份上限来源；`End` 直接用传入的上限是为了与内核同源。
 *
 * 特殊说明 2：翻页量取 `max(0, 视口高 − 20)`。视口极矮（< 20px）时不得产生负步进，
 * 否则 PageDown 会往**回**翻。
 *
 * 特殊说明 3：不处理的按键返回 `null`（调用方据此不 `preventDefault`，让按键继续
 * 走既有的编辑 / 快捷键路径）。
 *
 * @param args 按键、当前位置、视口高与上限。
 * @returns 目标竖向位置；该键不归竖向滚动时返回 null。
 */
export function resolveKeyboardScrollTarget(args: KeyboardScrollArgs): number | null {
    const key = typeof args.key === "string" ? args.key.toLowerCase() : "";
    const pageStep = Math.max(0, args.viewportHeightPx - PAGE_SCROLL_OVERLAP_PX);
    switch (key) {
        case "pagedown":
            return args.scrollTopPx + pageStep;
        case "pageup":
            return args.scrollTopPx - pageStep;
        case "home":
            return 0;
        case "end":
            return args.maxScrollTopPx;
        default:
            return null;
    }
}
