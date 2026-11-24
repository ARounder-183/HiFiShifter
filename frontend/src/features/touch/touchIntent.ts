// 触摸意图：这一下触摸到底是什么意思。
//
// 【为什么必须有这一层】两块核心画布（时间轴内核 / 参数编辑器）里，
// `"touch"` 这个词此前出现 **0 次** —— 所有触摸适配都停在外壳层（顶栏、抽屉、
// 菜单），一进画布就回到桌面语义：滚轮滚、右键弹菜单、拖空白处框选。
// 于是手机上「滚动」这件事根本没有实现，用户只能去拖那根 8px 的滑块。
//
// 本模块把「按下的是什么 + 几根手指 + 当前工具」压成一个**唯一**的判定表，
// 两块画布共用。它是纯函数，因此这张表可以被单测钉死，也可以被两个画布
// 同时信任 —— 而不是各写一份"大概是这样"的判断。
//
// 判定表（唯一一份）：
//
//   | 手指数 | 命中               | 工具   | 意图  |
//   | ------ | ------------------ | ------ | ----- |
//   | 2      | 任意               | 任意   | pinch |
//   | 1      | empty              | 任意   | pan   |
//   | 1      | ruler              | 任意   | edit  |
//   | 1      | 可编辑对象         | 任意   | edit  |
//
// 【为什么 empty → pan 是关键】这是移动端"画布可滚动"的唯一实现方式。桌面靠
// 滚轮，触摸只能靠"在空白处拖"。而"空白"正好是命中测试返回 null 的地方，
// 因此不需要任何新几何 —— 判定表读的就是既有的命中结果。
//
// 【为什么 ruler 不 pan】标尺的唯一语义就是播放头；让它可滚动会把"想移动
// 播放头"变成"滚了一下页面"，而这两件事在手指上没有区别，只能靠命中区分。

/** 一次触摸可能产生的意图。 */
export type TouchIntent = "edit" | "pan" | "pinch" | "menu";

/** 命中目标（两块画布的交集，取粗粒度即可 —— 意图只区分"能不能编辑"）。 */
export type TouchHitKind =
    | "clip-body"
    | "clip-edge"
    | "fade-handle"
    | "curve"
    | "param-block"
    | "selection-edge"
    | "ruler"
    | "empty";

/** 当前工具。与 `ToolMode` 同形，但只关心"是不是剃刀"。 */
export type TouchTool = "select" | "draw" | "line" | "vibrato" | "razor";

export interface TouchIntentInput {
    hit: TouchHitKind;
    /** 当前按下的手指数（1 或 2）。 */
    pointerCount: number;
    tool: TouchTool;
    /**
     * 该画布是否启用了触摸手势层（能力位 `input.two_finger_gesture`）。
     *
     * 【为什么要显式传】桌面上也装了同一套监听器（浏览器里也能收到
     * `pointerType === "touch"` 的合成事件，例如触摸屏笔记本）。能力位为假时
     * 必须**永远**返回 `edit`，让桌面行为逐字节不变。
     */
    gestureEnabled: boolean;
}

/** 命中目标是否"可以被单指编辑"。 */
export function isEditableHit(hit: TouchHitKind): boolean {
    return hit !== "empty";
}

/**
 * 解析一次触摸的意图。
 *
 * 优先级：手势层关闭 → 恒 `edit`；双指 → `pinch`（无论命中什么，包括落在
 * Clip 上 —— 用户按下第二根手指就是在说"我要缩放，不要编辑"）；其余按命中判。
 */
export function resolveTouchIntent(input: TouchIntentInput): TouchIntent {
    if (!input.gestureEnabled) return "edit";
    if (input.pointerCount >= 2) return "pinch";
    // 标尺永远归播放头：让它可滚会把"移动播放头"变成"滚了一下页面"。
    if (input.hit === "ruler") return "edit";
    if (input.hit === "empty") return "pan";
    return "edit";
}

/**
 * 第二根手指落下时，第一根手指正在进行的手势该怎么处理。
 *
 * 【为什么必须是"取消"而不是"忽略第二根手指"】用户按下第二根手指的意图是
 * 缩放/平移；若第一根手指的编辑继续，画面会一边被拖动编辑、一边被缩放 ——
 * 且松手时会**提交**那次本来要被放弃的编辑。取消是唯一不产生副作用的处理。
 */
export function shouldCancelEditForPinch(input: TouchIntentInput): boolean {
    return input.gestureEnabled && input.pointerCount >= 2;
}
