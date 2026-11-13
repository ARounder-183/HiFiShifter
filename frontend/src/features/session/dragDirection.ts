/**
 * 拖动方向的模式表与切换规则 —— 工具栏「拖动方向」按钮的唯一事实来源。
 *
 * 【主要内容】
 * - `DRAG_DIRECTIONS_BY_TOOL`：每个工具**实际可用**的方向；顺序即左键 / `D` 键的循环顺序；
 * - `DEFAULT_DRAG_DIRECTION_BY_TOOL`：出厂方向（`initialState` 直接取用）；
 * - `nextDragDirection`：循环到下一个方向（工具栏左键、全局 `D` 键、拖拽中右键 / `D`）；
 * - `isDragDirectionForTool`：写入前的白名单（reducer 与设置水合共用这一份）；
 * - `resolveDragDirectionTool` / `resolveDragDirection`：当前工具读哪份方向字段。
 *
 * 【为什么状态里**没有**一个布尔】拖动方向不是"一个功能 + 一个模式"，而是**一根轴上的
 * 互斥取值**：自由 / 仅 X / 仅 Y。三个方向穷尽且互斥，不存在与它们正交的"开 / 关" ——
 * "自由"本身就是其中一种取值，不是"关"。因此状态只有一个方向字段；左键依次循环（与
 * `D` 键逐字同义），右键直接挑一个。按钮在 `仅 X` / `仅 Y` 时高亮，那是由方向当场派生的
 * 渲染结论（见 `panel/DragDirectionButton`），不是第二个状态。
 *
 * 【对比自动滚屏】那边的"关"是另一根轴（要不要跟随），与模式正交，所以按钮有开关态、
 * 菜单只列模式。两者的形状差异来自**状态模型的差异**，不是风格漂移。
 *
 * 【为什么方向集随工具变化】`y-only` 只在选择工具下有意义：绘制 / 直线 / 颤音工具的
 * 横向位移本身就是"时间推进"，锁掉纵向只剩一条水平线（这正是 `x-only` 的语义），
 * 反过来锁掉横向则会在同一帧反复写同一个值、画不出任何东西。所以那两个工具只有
 * `free` / `x-only` 两种，菜单也就只有两项 —— 这是工具能力差异，不是遗漏。
 *
 * 【与其他模块的关系】
 * - 上游：`sessionSlice/dragDirection`（两个 reducer）、`sessionSlice/initialState`
 *   （出厂值）、`sessionSlice/extraReducers/runtime`（设置水合的白名单）。
 * - 下游：`panel/DragDirectionButton`（图标 / 菜单 / tooltip）、
 *   `usePianoRollPanelModel`（把当前工具解析成方向与作用域，喂给手势层）。
 */
import type {
    DragDirection,
    DragDirectionTool,
    DrawDragDirection,
    DrawToolMode,
    ToolMode,
} from "./sessionTypes";

/** 每个工具实际可用的方向（顺序即 `D` 键的循环顺序）。 */
export const DRAG_DIRECTIONS_BY_TOOL: Record<DragDirectionTool, readonly DragDirection[]> = {
    select: ["free", "x-only", "y-only"],
    draw: ["free", "x-only"],
    vibrato: ["free", "x-only"],
};

/**
 * 出厂方向（`initialState` 直接取这三项，避免两处各写一遍字面量）。
 *
 * 【为什么是 `as const satisfies` 而不是 `Record` 标注】标注会把字面量**拓宽**成
 * `DragDirection`，于是 `initialState` 里 `drawDragDirection`（类型是更窄的
 * `DrawDragDirection`）不得不加一次断言；`satisfies` 既校验"键齐全、值合法"，又保住字面量。
 */
export const DEFAULT_DRAG_DIRECTION_BY_TOOL = {
    // 参数线拖拽最常见的动作是上下挪一条线，而不是横向挪时间。
    select: "y-only",
    draw: "free",
    vibrato: "free",
} as const satisfies Record<DragDirectionTool, DragDirection>;

/**
 * 该工具下这个方向是否合法（reducer 与设置水合共用的唯一白名单）。
 *
 * 【为什么分三重签名】绘制 / 颤音的状态字段是更窄的 `DrawDragDirection`，而校验本身
 * 走的是同一张表。按作用域分别收窄，调用方拿到 `"free" | "x-only"` 就能直接赋给字段，
 * 不必再补一次断言（断言会把"其实写进了 `y-only`"这类错误藏到运行时）。
 */
export function isDragDirectionForTool(tool: "select", value: unknown): value is DragDirection;
export function isDragDirectionForTool(
    tool: "draw" | "vibrato",
    value: unknown,
): value is DrawDragDirection;
export function isDragDirectionForTool(
    tool: DragDirectionTool,
    value: unknown,
): value is DragDirection;
export function isDragDirectionForTool(tool: DragDirectionTool, value: unknown): boolean {
    return DRAG_DIRECTIONS_BY_TOOL[tool].includes(value as DragDirection);
}

/** 循环到下一个方向（`D` 键 / 拖拽中右键，走到末尾回卷）。 */
export function nextDragDirection(tool: DragDirectionTool, current: DragDirection): DragDirection {
    const order = DRAG_DIRECTIONS_BY_TOOL[tool];
    const index = order.indexOf(current);
    // 当前值不在本工具的模式表里（旧配置 / 手改文件）时从表头重新开始，
    // 而不是停在原地 —— 那会让 `D` 键看起来失灵。
    return order[(index + 1) % order.length];
}

/** 解析"当前工具读哪份方向"所需的字段（`SessionState` 结构上满足它）。 */
type DragDirectionScope = {
    toolMode: ToolMode;
    drawToolMode: DrawToolMode;
    selectDragDirection: DragDirection;
    drawDragDirection: DrawDragDirection;
    lineVibratoDragDirection: DrawDragDirection;
};

/** 当前工具对应哪份方向字段（工具栏菜单与手势层读的都是它）。 */
export function resolveDragDirectionTool(
    s: Pick<DragDirectionScope, "toolMode" | "drawToolMode">,
): DragDirectionTool {
    if (s.toolMode === "select") return "select";
    return s.drawToolMode === "draw" ? "draw" : "vibrato";
}

/**
 * 当前生效的拖动方向。
 *
 * 直线与颤音共用同一份方向（`lineVibratoDragDirection`）—— 它们是同一个
 * "起点 → 终点"手势，拆成两份只会让"在直线下调过、切到颤音又不算"成为新的困惑。
 */
export function resolveDragDirection(s: DragDirectionScope): DragDirection {
    const tool = resolveDragDirectionTool(s);
    if (tool === "select") return s.selectDragDirection;
    return tool === "draw" ? s.drawDragDirection : s.lineVibratoDragDirection;
}
