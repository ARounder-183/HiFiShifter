/**
 * 参数编辑器内核 · 手势命中测试（纯函数）
 *
 * 【主要内容】
 * 把参数编辑器里"指针落在哪个可交互目标上"的判定抽成纯函数：
 * 曲线邻域、选区左右边缘、选区主体。判定只依赖指针位置与**序列化的几何输入**
 * （值→y 投影、选区边界像素坐标、曲线采样数组），不接触 DOM / React。
 *
 * 【作用】
 * 本工程的参数编辑器交互此前集中在一个 3,875 行的 hook
 * （`usePianoRollInteractions`）里，命中判定与手势状态机混在一起：既无法单测，
 * 也无法在迁移中复用。按 `timeline/kernel/interaction/` 的既有做法抽成纯模块后，
 * 每条判定都有单测，hook 只负责"取几何量 → 问命中 → 分发手势"。
 *
 * 【与其他模块的关系】
 * - 上游：`usePianoRollInteractions` 在 pointerdown / pointermove 时调用。
 * - 下游：手势状态机据命中结果进入「拖动选区 / 拉伸边缘 / 拖动曲线 / 画线」分支。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React，可在 node 环境完整单测。
 *
 * 【设计约束】
 * 1. **命中半径与投影必须同源**。曲线邻域判定用 `valueToY` 把值换算为 y 再比距离，
 *    不能用"值差 × 某个系数"——后者在缩放/换参数后与看到的曲线分离。
 * 2. **pitch 的 +0.5 偏移必须施加**。`render.ts` 绘制 pitch 曲线时对 MIDI 值加 0.5
 *    使曲线居于琴键中心（与 `curvePoints` 同源）；命中测试若不加，指针要偏离曲线
 *    半个键高才能命中。该偏移在本工程里出现过多次不一致，故在此显式集中。
 * 3. **非有限输入一律判定为不命中**。`Math.abs(NaN - x) <= w` 恒为 `false`，看似
 *    已是安全侧，但依赖这个隐式行为会让"为什么没命中"难以排查；显式拒绝更可靠。
 */

/** 曲线邻域命中半径（CSS px）。与 `usePianoRollInteractions` 的既有阈值一致。 */
export const CURVE_HIT_RADIUS_PX = 10;

/** 选区左右边缘的命中带宽（CSS px）。 */
export const SELECTION_EDGE_HIT_PX = 8;

/**
 * 命中测试使用的值→视口 y 投影。
 *
 * 与绘制侧同一约定：接收参数值，返回视口坐标 y（CSS px）。
 */
export type ValueToY = (value: number) => number;

/**
 * 取指针横坐标所在帧的曲线值（不含"靠近曲线"的邻域判定）。
 *
 * 流程：秒 → 帧号（`sec × 1000 / framePeriodMs`，向下取整并 clamp 到 ≥0）
 * → 采样下标（`(frame − startFrame) / stride`，四舍五入）→ 取数组值。
 *
 * 特殊说明 1：下标用**四舍五入**而不是取整——采样点是离散的，取整会让指针在
 * 两个采样点中间时偏向左侧，与绘制路径（相邻点直线连接）产生半格的系统性偏移。
 *
 * 特殊说明 2：越界、非有限值、非法帧周期一律返回 `null` 而不是 `0`。返回 `0` 会
 * 让调用方把"没有数据"误当成"曲线在 0 处"，进而在空白区域弹出浮窗。
 *
 * @param args 查询参数。
 * @returns 该帧的曲线值；不可用时为 `null`。
 */
export function curveValueAtPointerFrame(args: {
    /** 指针位置的秒数（时间轴坐标）。 */
    readonly sec: number;
    /** 首个采样值对应的帧号。 */
    readonly startFrame: number;
    /** 采样步长（帧）。 */
    readonly stride: number;
    /** 每帧时长（毫秒）。 */
    readonly framePeriodMs: number;
    /** 采样值数组（通常是 `edit`）。 */
    readonly values: readonly number[];
}): number | null {
    const { sec, startFrame, stride, framePeriodMs, values } = args;
    if (!Number.isFinite(sec)) return null;
    if (!Number.isFinite(framePeriodMs) || framePeriodMs <= 0) return null;
    if (!Array.isArray(values) || values.length === 0) return null;

    const frame = Math.max(0, Math.floor((sec * 1000) / framePeriodMs));
    const step = Math.max(1, Math.floor(stride));
    const idx = Math.round((frame - startFrame) / step);
    if (idx < 0 || idx >= values.length) return null;
    const value = Number(values[idx]);
    return Number.isFinite(value) ? value : null;
}

/**
 * 判定指针是否落在参数线的命中邻域内。
 *
 * 流程：取指针的视口 y（`pointerY` 直接用，或由 `pointerValue` 经 `valueToY` 换算）
 * → 把曲线值经 **pitch 的 +0.5 偏移** 后换算为 y → 比较纵向距离是否小于
 * `CURVE_HIT_RADIUS_PX`。
 *
 * 特殊说明 1：只比较**纵向**距离，不比横向。曲线在横向上处处存在，横向筛选由
 * 调用方（"指针所在帧的值"）完成。
 *
 * 特殊说明 2：`pointerY` 与 `pointerValue` 二选一。前者用于调用方已经拿到画布局部
 * y 的场景（省一次 getBoundingClientRect），后者用于只有参数值的场景。
 *
 * @param args 判定参数。
 * @returns 是否命中。
 */
export function isPointerNearCurve(args: {
    /** 指针的画布局部 y（CSS px）。与 `pointerValue` 二选一，优先。 */
    readonly pointerY?: number;
    /** 指针处的参数值。与 `pointerY` 二选一。 */
    readonly pointerValue?: number;
    /** 参数名（决定是否施加 pitch 的 +0.5 偏移）。 */
    readonly param: string;
    /** 值 → 视口 y 投影（须与绘制侧同源）。 */
    readonly valueToY: ValueToY;
    /** 指针所在帧的曲线值。 */
    readonly curveValue: number;
}): boolean {
    const { pointerY, pointerValue, param, valueToY, curveValue } = args;
    if (!Number.isFinite(curveValue)) return false;

    let y: number;
    if (pointerY !== undefined) {
        if (!Number.isFinite(pointerY)) return false;
        y = pointerY;
    } else if (pointerValue !== undefined) {
        if (!Number.isFinite(pointerValue)) return false;
        y = valueToY(pointerValue);
    } else {
        return false;
    }

    // 与 render.ts 的绘制偏移同源：pitch 曲线画在 N 键中心。
    const mapped = param === "pitch" ? curveValue + 0.5 : curveValue;
    const curveY = valueToY(mapped);
    if (!Number.isFinite(curveY)) return false;
    return Math.abs(y - curveY) < CURVE_HIT_RADIUS_PX;
}

/**
 * 判定指针落在选区的哪条边缘上。
 *
 * 流程：归一左右边界（调用方可能传入未排序的 beat 投影结果）→ 依次比较到左右
 * 边界的距离是否 ≤ `SELECTION_EDGE_HIT_PX`。
 *
 * 特殊说明 1：**左缘优先**。当选区窄于两倍命中带宽时，所有位置都落在某条边的
 * 带内；固定取左缘保证"向左拉伸"始终可达，且行为确定（不随距离抖动）。
 *
 * 特殊说明 2：边界比较用 `<=`（闭区间），让"正好拖到边缘"这一帧仍然可命中——
 * 指针事件的坐标本来就离散，开区间会让边界那一列像素失效。
 *
 * @param args 判定参数。
 * @returns `"left"` / `"right"` / `null`（未命中边缘）。
 */
export function hitTestSelectionEdge(args: {
    /** 选区左边界（画布局部 x，CSS px）。 */
    readonly leftXPx: number;
    /** 选区右边界（画布局部 x，CSS px）。 */
    readonly rightXPx: number;
    /** 指针的画布局部 x。 */
    readonly localXPx: number;
}): "left" | "right" | null {
    const { leftXPx, rightXPx, localXPx } = args;
    if (!Number.isFinite(localXPx)) return null;
    if (!Number.isFinite(leftXPx) || !Number.isFinite(rightXPx)) return null;
    const left = Math.min(leftXPx, rightXPx);
    const right = Math.max(leftXPx, rightXPx);
    if (Math.abs(localXPx - left) <= SELECTION_EDGE_HIT_PX) return "left";
    if (Math.abs(localXPx - right) <= SELECTION_EDGE_HIT_PX) return "right";
    return null;
}

/**
 * 判定指针是否落在选区**主体**内（用于拖动整个选区）。
 *
 * 流程：归一左右边界 → 判断指针 x 是否在闭区间内 → 与"靠近曲线"取交集。
 *
 * 【为什么必须与"靠近曲线"取交集】只判选区区间会让**整个选区块变成拖拽热区**：
 * 用户在选区内画线/擦除时，pointerdown 会先命中"拖动选区"，画线功能直接失效。
 * 加上曲线邻域后，只有"按在曲线上"才算拖动，其余位置透传给绘制手势。
 *
 * @param args 判定参数。
 * @returns 是否可拖动选区主体。
 */
export function hitTestSelectionBody(args: {
    readonly leftXPx: number;
    readonly rightXPx: number;
    readonly localXPx: number;
    /** 指针是否落在曲线邻域内（由 `isPointerNearCurve` 得出）。 */
    readonly nearCurve: boolean;
}): boolean {
    const { leftXPx, rightXPx, localXPx, nearCurve } = args;
    if (!nearCurve) return false;
    if (!Number.isFinite(localXPx)) return false;
    if (!Number.isFinite(leftXPx) || !Number.isFinite(rightXPx)) return false;
    const left = Math.min(leftXPx, rightXPx);
    const right = Math.max(leftXPx, rightXPx);
    return localXPx >= left && localXPx <= right;
}
