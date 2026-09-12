/**
 * 渲染内核 · 竖向键盘滚动目标解析（时间轴与参数编辑器**共用**）
 *
 * 【主要内容】
 * 把 PageUp / PageDown / Home / End 四个按键解析成目标竖向滚动位置。
 *
 * 【作用：为什么必须由内核自己实现】
 * 内核模式下原生 scroller 是**被动镜像**，其 `scroll` 事件会被当作回声忽略
 * （宿主每帧把内核真值写回 DOM，事件不带来源，无法与用户输入区分）。
 * 旧实现里这四个键是靠**浏览器的原生滚动**生效的——忽略回声后
 * 原生滚动的那次变化就没人采纳了，键盘翻页会**彻底失效**。因此必须把键盘滚动
 * 收进内核：参数编辑器内核与时间轴内核都监听这四个键，共用本模块的解析。
 *
 * 【翻页量为什么是「视口高 − 20」而不是「视口高」】
 * 这是 Chromium 自身的翻页量（`kPageOverlap = 20`，留 20px 重叠便于连续阅读）。
 * 参数编辑器旧实现走原生滚动，实测（5 组视口高度，误差 0）就是该式：
 * 视口高 823 → 步进 803；623 → 603；323 → 303；423 → 403；1023 → 1003。
 * 用整屏会与迁移前的位移不一致，属手感变更，不在迁移范围。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRollKernelHost` 与 `timelineKernelHost` 的 keydown 监听都调用它。
 * - 下游：目标值交给各自的 `ScrollKernel.setScrollTop`（钳制在那里统一做）。
 * - 独立性：纯函数，不依赖 DOM / React。因此放在共享的 `renderKernel` 目录，
 *   而不是任一面板目录下——否则 `timeline` 要反向依赖 `pianoRoll`，形成横向耦合。
 */

/** 与 Chromium 原生翻页量对齐的视口重叠量（CSS px）。 */
export const PAGE_SCROLL_OVERLAP_PX = 20;

/**
 * 翻页量的最小视口占比（Blink `ScrollableArea::kMinFractionToStepWhenPaging`）。
 *
 * 【为什么需要它——只减 20 在矮视口下与原生的位移不一致】
 * 原生翻页量实际是 `max(视口高 − 20, 视口高 × 0.875)`，不是单纯的减法。两者在
 * 高视口下重合（823 → 803），但在**矮视口**下由这一项接管：
 * 视口高 151 → 实测 132，而 `151 − 20 = 131`（差 1px）。
 * 实测依据（全部来自真实应用，非合成探针）：
 * - 时间轴 legacy：视口高 151 → 步进 **132**
 * - 参数编辑器 legacy：823 → 803、623 → 603、423 → 403、323 → 303、1023 → 1003
 * 六组数据全部满足 `max(H − 20, floor(0.875H))`，且仅靠 `H − 20` 在 151 时失配。
 *
 * 时间轴的轨道头视口只有 151px 高（面板高度受限），因此这个分支**实际会被走到**，
 * 不是理论边界。
 */
export const PAGE_SCROLL_MIN_FRACTION = 0.875;

/** 竖向键盘滚动的解析入参。 */
export interface KeyboardScrollArgs {
    /** 按键名（`KeyboardEvent.key`，大小写不敏感）。 */
    readonly key: string;
    /** 当前竖向位置（CSS px）。 */
    readonly scrollTopPx: number;
    /** 视口高度（CSS px），决定翻页量。 */
    readonly viewportHeightPx: number;
    /** 竖向滚动上限（CSS px；`End` 直接取它，与内核同源）。 */
    readonly maxScrollTopPx: number;
}

/**
 * 计算与 Chromium 原生一致的整页位移。
 *
 * 流程：`max(视口高 − 20, floor(视口高 × 0.875))` → 夹到非负。
 *
 * 特殊说明 1：两项都不可省。减 20 项在高视口下更大（823 → 803 > 720），占比项在
 * 矮视口下更大（151 → 132 > 131），原生取的正是两者的**较大值**（见常量说明与
 * `keyboardScroll.test.ts` 的实测基线）。
 *
 * 特殊说明 2：夹非负是必要的——视口极矮（< 20px）时减法项为负，若不夹会让 PageDown
 * 往**回**翻。取整只对占比项做（与原生一致：减 20 项本身是整数语义）。
 *
 * @param viewportHeightPx 视口高度（CSS px）。
 * @returns 整页位移（≥ 0）。
 */
export function resolvePageScrollStepPx(viewportHeightPx: number): number {
    if (!Number.isFinite(viewportHeightPx)) return 0;
    return Math.max(
        0,
        Math.max(
            viewportHeightPx - PAGE_SCROLL_OVERLAP_PX,
            Math.floor(viewportHeightPx * PAGE_SCROLL_MIN_FRACTION),
        ),
    );
}

/**
 * 解析一个按键对应的竖向滚动目标。
 *
 * 流程：归一化按键名 → 按四个键分别给出目标（Home 取 0、End 取上限、
 * PageUp/PageDown 按整页位移，见 `resolvePageScrollStepPx`）→ **不做钳制**，
 * 交由内核统一处理。
 *
 * 特殊说明 1：**不钳制**是有意的。内核是唯一做钳制的地方（`ScrollKernel` 的强制
 * 约束），这里再夹一次会出现两份上限来源；`End` 直接用传入的上限是为了与内核同源。
 *
 * 特殊说明 2：整页位移见 `resolvePageScrollStepPx`（含"为什么不是单纯减 20"的
 * 实测依据）。视口极矮时该函数已保证非负步进。
 *
 * 特殊说明 3：不处理的按键返回 `null`（调用方据此不 `preventDefault`，让按键继续
 * 走既有的编辑 / 快捷键路径）。
 *
 * @param args 按键、当前位置、视口高与上限。
 * @returns 目标竖向位置；该键不归竖向滚动时返回 null。
 */
export function resolveKeyboardScrollTarget(args: KeyboardScrollArgs): number | null {
    const key = typeof args.key === "string" ? args.key.toLowerCase() : "";
    const pageStep = resolvePageScrollStepPx(args.viewportHeightPx);
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
