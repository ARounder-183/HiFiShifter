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
 * 允许边缘拖拽所需的**最小选区宽度**（CSS px）。
 *
 * 【为什么需要】随机单击参数编辑器会留下一个**零宽**选区：框选路径在 pointerdown
 * 时就写入一段 `startBeat === endBeat` 的选区（为了立刻抹掉上一个选区的框，见
 * `usePianoRollInteractions` 的框选分支）。这种选区在画面上没有任何可见区域，两条
 * "边缘"也重合在同一条线上，但边缘命中带（±8px）照样命中 —— 用户会在单击处看到
 * `ew-resize` 光标，却找不到任何可以拖的选区，于是"光标变了但画面没东西可拖"。
 *
 * 阈值取 2px：低于它时填充只剩一条发丝线、两条边框几乎重合，"拖的是左缘还是右缘"
 * 已无从分辨，边缘拖拽没有可兑现的语义。**零宽 / 亚像素选区一律不提供边缘交互**
 * （光标与手势都不变），用户就不会在空选区上看到可拖的提示。
 */
export const SELECTION_EDGE_MIN_WIDTH_PX = 2;

/**
 * 命中测试使用的值→视口 y 投影。
 *
 * 与绘制侧同一约定：接收参数值，返回视口坐标 y（CSS px）。
 */
export type ValueToY = (value: number) => number;

/**
 * 取指针处**参数线（渲染折线）上的点**：像素 y + 参数值（不含"靠近曲线"的判定）。
 *
 * 【为什么必须沿折线插值，而不是取"指针所在帧的那个采样值"】参数线是**连续的**：
 * `projectCurvePoints` 把相邻采样点用直线连起来，两点之间处处有线段。若只取最近
 * 采样点的值去比纵向距离，指针落在两个采样点中间时量到的是最近**端点**的 y ——
 * 陡峭段上这个距离可以远超命中半径，于是"光标明明压在线上却抓不住参数线"。
 * 插值到线段上之后，指针只要贴着线就能命中，与画面所见一致。
 *
 * 流程：秒 → **连续**采样坐标 `t = (sec×1000/framePeriodMs − startFrame) / stride`
 * → 取相邻两点 `[lo, hi]` 与段内比例 `frac` → 两端的 y 按 `frac` 线性插值
 * （值同样按 `frac` 插值，供浮窗显示）。
 *
 * 特殊说明 1：`t` **不取整**（这正是本函数存在的理由），只在取相邻点时 `floor`。
 *
 * 特殊说明 2：y 在**像素域**插值（先各自投影再插值），与绘制侧"连两个投影后的
 * 端点"逐像素一致；`valueToY` 带 clamp，先插值再投影会在轴范围外差一点。
 *
 * 特殊说明 3：越界、非有限值、非法帧周期一律返回 `null` 而不是某个值。返回 0 会
 * 让调用方把"没有数据"误当成"曲线在 0 处"，进而在空白区域弹出浮窗。
 *
 * @param args 查询参数。
 * @returns 线上的点；不可用时为 `null`。
 */
export function curvePointAtPointer(args: {
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
    /** 参数名（决定是否施加 pitch 的 +0.5 偏移，与绘制同源）。 */
    readonly param: string;
    /** 值 → 视口 y 投影（须与绘制侧同源）。 */
    readonly valueToY: ValueToY;
}): { y: number; value: number } | null {
    const { sec, startFrame, stride, framePeriodMs, values, param, valueToY } = args;
    if (!Number.isFinite(sec)) return null;
    if (!Number.isFinite(framePeriodMs) || framePeriodMs <= 0) return null;
    if (!Array.isArray(values) || values.length === 0) return null;
    if (!Number.isFinite(startFrame)) return null;

    const step = Math.max(1, Math.floor(stride));
    const t = ((sec * 1000) / framePeriodMs - startFrame) / step;
    if (!Number.isFinite(t)) return null;
    // 折线只存在于首末采样点之间：出界即"指针不在曲线上"。
    if (t < 0 || t > values.length - 1) return null;

    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, values.length - 1);
    const frac = t - lo;
    const valueLow = Number(values[lo]);
    const valueHigh = Number(values[hi]);
    if (!Number.isFinite(valueLow) || !Number.isFinite(valueHigh)) return null;

    const offset = param === "pitch" ? 0.5 : 0;
    const yLow = valueToY(valueLow + offset);
    const yHigh = valueToY(valueHigh + offset);
    if (!Number.isFinite(yLow) || !Number.isFinite(yHigh)) return null;

    return {
        y: yLow + (yHigh - yLow) * frac,
        value: valueLow + (valueHigh - valueLow) * frac,
    };
}

/**
 * 判定指针是否落在参数线的命中邻域内。
 *
 * 流程：比较指针 y 与**参数线在指针 x 处的 y**（由 {@link curvePointAtPointer} 沿
 * 折线插值给出）的纵向距离是否小于 `CURVE_HIT_RADIUS_PX`。
 *
 * 特殊说明 1：只比较**纵向**距离，不比横向。曲线在横向上处处存在；横向筛选（指针
 * 是否落在选区内 / 数据窗口内）由 `curvePointAtPointer` 与调用方完成。
 *
 * 特殊说明 2：投影与 pitch 的 +0.5 偏移**不在这里**，而在 `curvePointAtPointer`
 * 里一次算完 —— 参数线的位置只该有一个定义处，否则"浮窗显示的值"与"命中判定用的
 * 位置"会各自漂移（本模块文件头约束 1 的同一教训）。
 *
 * @param args.pointerY 指针的画布局部 y（CSS px）。
 * @param args.curveY 参数线在指针 x 处的 y（沿折线插值）。
 * @returns 是否命中。
 */
export function isPointerNearCurve(args: {
    readonly pointerY: number;
    readonly curveY: number;
}): boolean {
    const { pointerY, curveY } = args;
    if (!Number.isFinite(pointerY) || !Number.isFinite(curveY)) return false;
    return Math.abs(pointerY - curveY) < CURVE_HIT_RADIUS_PX;
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
 * 特殊说明 3：**窄于 {@link SELECTION_EDGE_MIN_WIDTH_PX} 的选区一律不命中**。
 * 零宽 / 亚像素选区在画面上没有可见区域（随机单击就会留下一个），却会因为两条
 * 边缘重合而始终落在命中带内，表现为"光标变成可拉伸、但看不到任何选区"。
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
    // 不可见的选区（零宽 / 亚像素）没有可抓的边缘：见 SELECTION_EDGE_MIN_WIDTH_PX。
    if (right - left < SELECTION_EDGE_MIN_WIDTH_PX) return null;
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
