/**
 * 折线抽稀（纯函数）
 *
 * 【主要内容】
 * 把一条按时间单调递增的曲线点序列，按**设备像素列**抽稀到肉眼可分辨的密度：
 * 每个物理像素列最多保留 2 个点（该列的 y 最小值与最大值），并携带每个点在
 * **原始序列**中的累积弧长。
 *
 * 【作用（解决什么问题）】
 * 最小缩放下参数编辑器视口 1864 CSS px 覆盖 466 秒，而曲线按 200 点/秒采样
 * （`usePianoRollData.ts` 的 `stride = 1`）：
 *
 * | 量 | 值 |
 * |---|---|
 * | 可见采样点 | ~93,200 |
 * | 物理像素列（dpr 2） | 3,728 |
 * | **每列点数** | **25** |
 *
 * 同列的点渲染在**同一个像素**上，视觉完全冗余，但下游要为每个点付钱：
 * Canvas2D 是逐点 `lineTo`（实测净 30.3 ms/帧），GL 是逐点展开 9 个顶点
 * （实测 213,595 点 → 1,922,315 顶点 → 32 ms/帧）。按列抽稀把成本从
 * **随点数**变成**随像素列**，两条路径同时受益。
 *
 * 【为什么不改成后端降采样（stride > 1）】
 * `selectionEditData.ts` 的零 IPC 快路径要求 `stride === 1`——**编辑**依赖全分辨率
 * 数据。抽稀只能发生在渲染层，取数必须保持全分辨率。
 *
 * 【精度契约（"保持一定的精度"的可验证形式）】
 * 1. **包络**：每列的 y 极值都保留。曲线在同列内折返时，只留首末点会把极值削平，
 *    表现为 pitch 曲线的八度跳变被磨圆。这是抽稀唯一真正会丢信息的地方，故必须保住。
 * 2. **弧长**：`along` 取自**原始**序列。虚线相位按弧长推进（见
 *    `polylineGeometry` 的文件头），若按抽稀后的折线重新累加，弧长变短会让
 *    虚线图案随缩放漂移。
 * 3. **端点**：整条曲线的首末点始终保留，否则曲线两端被截短。
 * 4. **顺序**：结果 x 单调不减，折线不会自交或倒退。
 *
 * 【与其他模块的关系】
 * - 上游：`pianoRoll/kernel/scene/curvePoints` 的投影结果（视口坐标，x 单调不减）。
 * - 下游：`polylineGeometry.buildPolylineVertices`（点 → 三角形）。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

import type { PolylinePoint } from "./polylineGeometry";

/**
 * 每个设备像素列保留的点数上限。
 *
 * 【为什么是 2 而不是 1】列内保留 y 的 min 与 max：曲线在同一列内上下折返时，
 * 只留一个点会把振幅压平。2 个点既能保住包络，又把每列成本钉死为常数。
 */
const POINTS_PER_COLUMN = 2;

/** 抽稀参数。 */
export interface DecimatePolylineArgs {
    /** 视口坐标点序列，`x` 必须**单调不减**（曲线投影的天然性质）。 */
    readonly points: readonly PolylinePoint[];
    /** 设备像素比。 */
    readonly dpr: number;
    /** 视口宽度（CSS px），用于算设备像素列数。 */
    readonly viewportWidthPx: number;
}

/** 抽稀结果。 */
export interface DecimatePolylineResult {
    /** 结果点序列；未抽稀时是**入参的同一引用**。 */
    readonly points: readonly PolylinePoint[];
    /**
     * 每个结果点在原始序列中的累积弧长；未抽稀时为 `null`。
     *
     * 特殊说明：未抽稀时返回 `null` 而不是现算的数组——那种情况下调用方本来就
     * 要按顺序累加，预先算一遍等于白付一次 O(n)。抽稀时弧长必须随点一起带出，
     * 因为结果已无法从自身重算出与原始一致的弧长。
     */
    readonly along: readonly number[] | null;
    /** 是否真的发生了抽稀。 */
    readonly decimated: boolean;
}

/**
 * 判断点序列是否可以安全抽稀。
 *
 * 流程：校验 dpr / 视口宽有限且为正 → 点数至少 2 → 逐点有限 → `x` 单调不减。
 *
 * 特殊说明：任一条不满足都返回 `false`（原样返回）。这是**安全侧**设计：
 * 抽稀是纯优化，任何不确定的情形下放弃优化都比产出更差的结果好。
 * `x` 单调是分列算法的前提——它依赖"处理到第 i 个点时，其所在列及之前的列
 * 都已封闭"。若 x 回退，回退段会被错误并入前一列。
 *
 * @param points 点序列。
 * @param dpr 设备像素比。
 * @param viewportWidthPx 视口宽度（CSS px）。
 * @returns 是否可以抽稀。
 */
function canDecimate(
    points: readonly PolylinePoint[],
    dpr: number,
    viewportWidthPx: number,
): boolean {
    if (!(dpr > 0) || !Number.isFinite(dpr)) return false;
    if (!(viewportWidthPx > 0) || !Number.isFinite(viewportWidthPx)) return false;
    if (points.length < 2) return false;
    let prevX = Number.NEGATIVE_INFINITY;
    for (const p of points) {
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return false;
        if (p.x < prevX) return false;
        prevX = p.x;
    }
    return true;
}

/**
 * 按设备像素列抽稀曲线点序列。
 *
 * 流程：
 * 1. 校验（`canDecimate`）；不通过则原样返回，`decimated = false`；
 * 2. 算设备像素列数 `cols = ceil(viewportWidthPx × dpr)`，以及"可分辨预算"
 *    `cols × POINTS_PER_COLUMN`；点数未超预算时原样返回（常见的中高缩放路径）；
 * 3. 预扫一遍原始序列的累积弧长（`O(n)`）；
 * 4. 按列归并：维护当前列的 y 最小/最大下标，遇到新列时把这两点按**下标顺序**
 *    推入结果（保证 x 单调）；
 * 5. 结尾封闭最后一列，并**强制保留首末点**（端点在视觉上比极值更重要）；
 * 6. 若结果反而不少于原始点数（预算边界上的退化情形），原样返回。
 *
 * 特殊说明 1：`x < 0` 的点（视口左缘外）归入第 0 列。调用方传的是可见点序列，
 * 正常情况下不会有负 x；万一有，并入第 0 列只是让边界略保守，不会丢信息。
 *
 * 特殊说明 2：结果是**新数组**，但其中的点对象是从入参**原样引用**的（不复制），
 * 这样调用方可以用坐标反查原始下标（单测就依赖这一点）。
 *
 * @param args 抽稀参数。
 * @returns 抽稀结果。
 */
export function decimatePolylinePoints(args: DecimatePolylineArgs): DecimatePolylineResult {
    const { points, dpr, viewportWidthPx } = args;
    const passThrough: DecimatePolylineResult = { points, along: null, decimated: false };
    if (!canDecimate(points, dpr, viewportWidthPx)) return passThrough;

    const cols = Math.max(1, Math.ceil(viewportWidthPx * dpr));
    const budget = cols * POINTS_PER_COLUMN;
    // 常见路径：点数本就在可分辨范围内，直接跳过（不复制数组、不算弧长）。
    if (points.length <= budget) return passThrough;

    // 原始累积弧长（虚线相位要的是"沿真实曲线的距离"，不是抽稀后折线的长度）。
    const along: number[] = new Array(points.length);
    along[0] = 0;
    for (let i = 1; i < points.length; i += 1) {
        along[i] =
            along[i - 1] + Math.hypot(points[i].x - points[i - 1].x, points[i].y - points[i - 1].y);
    }

    const outPoints: PolylinePoint[] = [];
    const outAlong: number[] = [];
    const push = (index: number): void => {
        outPoints.push(points[index]);
        outAlong.push(along[index]);
    };

    /** 当前列的 y 极值下标。 */
    let minIdx = 0;
    let maxIdx = 0;
    let currentCol = Math.min(cols - 1, Math.max(0, Math.floor(points[0].x * dpr)));

    /** 封闭当前列：按下标顺序推入极值（保证 x 单调不减）。 */
    const closeColumn = (): void => {
        if (minIdx === maxIdx) {
            push(minIdx);
            return;
        }
        const first = Math.min(minIdx, maxIdx);
        const second = Math.max(minIdx, maxIdx);
        push(first);
        push(second);
    };

    for (let i = 1; i < points.length; i += 1) {
        const col = Math.min(cols - 1, Math.max(0, Math.floor(points[i].x * dpr)));
        if (col !== currentCol) {
            closeColumn();
            currentCol = col;
            minIdx = i;
            maxIdx = i;
            continue;
        }
        if (points[i].y < points[minIdx].y) minIdx = i;
        if (points[i].y > points[maxIdx].y) maxIdx = i;
    }
    closeColumn();

    // 端点保真：首末点在视觉上比单列极值更重要（曲线不能两端被截短）。
    if (outPoints[0] !== points[0]) {
        outPoints.unshift(points[0]);
        outAlong.unshift(along[0]);
    }
    const lastIndex = points.length - 1;
    if (outPoints[outPoints.length - 1] !== points[lastIndex]) {
        outPoints.push(points[lastIndex]);
        outAlong.push(along[lastIndex]);
    }

    // 退化保护：结果不少于原始点数时放弃（否则"优化"反而增加下游成本）。
    if (outPoints.length >= points.length) return passThrough;

    return { points: outPoints, along: outAlong, decimated: true };
}
