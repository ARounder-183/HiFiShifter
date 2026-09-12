/**
 * 折线几何构建（纯函数）
 *
 * 【主要内容】
 * 把一串内容坐标点展开为可交给 WebGL2 绘制的三角形顶点缓冲，顶点携带**到中心线
 * 的有符号距离**与**沿线的累积弧长**，供片元着色器做亚像素抗锯齿与虚线相位。
 *
 * 【作用】
 * 参数编辑器的曲线图层一直走 Canvas2D 的 `stroke()`。搬到 GL 后要自行承担
 * `stroke()` 原本代劳的三件事：**连接方式**（miter/bevel）、**亚像素抗锯齿**
 * （线宽是 1.8 / 2.6 / 3.2 这类非整数值）、**虚线相位**（按弧长推进）。
 * 本模块只做几何与标量计算，不接触 GL / DOM，因此可以完整单测。
 *
 * 【顶点布局】每个顶点 4 个 float：
 * ```
 * [x, y, along, across]
 *   x, y   —— 内容坐标（CSS px）
 *   along  —— 沿折线的累积弧长（CSS px），供虚线相位
 *   across —— 到中心线的有符号距离（CSS px），供片元计算覆盖率
 * ```
 * 特殊说明：`across` 的范围是 `±(线宽/2 + aaPadPx)`，比几何边缘**更宽**——
 * 片元需要在外侧留有像素来承载覆盖率衰减。因此着色器的覆盖率阈值必须由调用方
 * 以独立 uniform 传入（`lineWidth/2`），**不能**从 `max|across|` 反推。
 *
 * 【为什么不把顶点展开到精确外沿】
 * 若把每个顶点沿法线推出 `线宽/2`，那么在拐角处"两条边的外沿交点"与"中心线到
 * 该交点的垂直距离"不再相等，同一个顶点无法同时满足相邻两段——线宽会随拐角
 * 变化。携带距离场则让着色器对每个片元独立求解覆盖率，这也是能同时表达
 * "非整数线宽 + 抗锯齿 + 虚线"的唯一方式。
 *
 * 【与其他模块的关系】
 * - 上游：曲线的投影结果（可见点序列，内容坐标）。
 * - 下游：`polylineProgram`（顶点属性 → 着色器）。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

/** 每个顶点占用的 float 数（布局见文件头）。 */
export const POLYLINE_FLOATS_PER_VERTEX = 4;

/** 默认的 AA 余量（CSS px）：约为一个设备像素的一半，足够承载覆盖率衰减。 */
export const DEFAULT_AA_PAD_PX = 0.5;

/** 输入点。 */
export interface PolylinePoint {
    readonly x: number;
    readonly y: number;
}

/** 构建参数。 */
export interface PolylineGeometryArgs {
    /** 内容坐标点序列（至少 2 个有效点）。 */
    readonly points: readonly PolylinePoint[];
    /** 线宽（CSS px），必须为正。可为分数（Canvas2D 路径就用了 1.8 / 2.6 等）。 */
    readonly lineWidth: number;
    /**
     * miter 比例上限。
     *
     * 语义与 Canvas2D / PDF 一致：`比例 = miter长度 / 半线宽 = 1 / cos(转角/2)`。
     * 超过上限时该拐角退化为 bevel（不生成尖角顶点）。
     */
    readonly miterLimit: number;
    /** AA 余量（CSS px）；缺省 `DEFAULT_AA_PAD_PX`，传 0 得到纯几何外沿。 */
    readonly aaPadPx?: number;
}

/** 内部使用：一个已归一化的段方向与法线。 */
interface Segment {
    readonly ux: number;
    readonly uy: number;
    /** 左法线。 */
    readonly nx: number;
    readonly ny: number;
}

/** 判断点是否全部有限且线宽 / miter 合法。 */
function isArgsValid(args: PolylineGeometryArgs): boolean {
    if (!(args.lineWidth > 0) || !Number.isFinite(args.lineWidth)) return false;
    if (!(args.miterLimit > 0) || !Number.isFinite(args.miterLimit)) return false;
    const pad = args.aaPadPx ?? DEFAULT_AA_PAD_PX;
    if (!Number.isFinite(pad) || pad < 0) return false;
    for (const p of args.points) {
        if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return false;
    }
    return true;
}

/**
 * 去掉相邻的重复点（距离小于 1e-9 视为重复）。
 *
 * 【为什么必须去重】重复点会让段的长度为零，归一化方向时产生 `0/0 = NaN`，
 * 而 NaN 一旦写进顶点缓冲会让**整层几何失效**（不是少画一段，而是全部消失）。
 *
 * @param points 原始点序列。
 * @returns 去重后的点序列。
 */
function dedupeConsecutive(points: readonly PolylinePoint[]): PolylinePoint[] {
    const out: PolylinePoint[] = [];
    for (const p of points) {
        const last = out[out.length - 1];
        if (last !== undefined && Math.hypot(p.x - last.x, p.y - last.y) < 1e-9) continue;
        out.push(p);
    }
    return out;
}

/**
 * 构建折线顶点缓冲。
 *
 * 流程：
 * 1. 校验参数 → 去重相邻重复点 → 少于 2 点则返回空；
 * 2. 逐段求单位方向与左法线；
 * 3. 逐段产出 2 个三角形（6 顶点），`along` 为累积弧长、`across` 为 `±(半线宽+pad)`；
 * 4. 内部顶点上，若 `miter比例 <= miterLimit` 则额外产出一个**尖角三角形**
 *    把顶点推到两条偏移线的交点；否则留空（视觉上即 bevel）。
 *
 * 特殊说明 1：尖角三角形也把 `across` 记为 `±(半线宽+pad)`——尖点落在两条偏移线
 * 的交点上，到各自中心线的垂直距离仍是半线宽。这正是"距离场"编码的必然结果，
 * 也是为什么 miter 的效果只体现在**位置**上。
 *
 * 特殊说明 2：`along` 从**传入序列的第一个点**开始累加（为 0）。调用方必须传
 * 可见点序列而不是整条曲线，否则虚线相位会与 Canvas2D 不一致（Canvas2D 的相位
 * 从子路径起点算起，而曲线是"从首个可见点开始"的单个子路径）。
 *
 * @param args 构建参数。
 * @returns 顶点缓冲（每 4 个 float 一个顶点）；参数非法或无可绘制段时为空。
 */
export function buildPolylineVertices(args: PolylineGeometryArgs): Float32Array {
    if (!isArgsValid(args)) return new Float32Array(0);
    const pts = dedupeConsecutive(args.points);
    if (pts.length < 2) return new Float32Array(0);

    const pad = args.aaPadPx ?? DEFAULT_AA_PAD_PX;
    const offset = args.lineWidth / 2 + pad;
    const segments: Segment[] = [];
    for (let i = 0; i + 1 < pts.length; i += 1) {
        const a = pts[i];
        const b = pts[i + 1];
        const len = Math.hypot(b.x - a.x, b.y - a.y);
        // 去重后理论上不会出现零长度，这里仍兜底以防浮点残差
        if (!(len > 1e-12)) continue;
        const ux = (b.x - a.x) / len;
        const uy = (b.y - a.y) / len;
        segments.push({ ux, uy, nx: -uy, ny: ux });
    }
    if (segments.length === 0) return new Float32Array(0);

    /** 每段的累积起点弧长。 */
    const segStart: number[] = [];
    let acc = 0;
    for (let i = 0; i < segments.length; i += 1) {
        segStart.push(acc);
        acc += Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
    }

    // 缓冲容量：每段 6 顶点；每个内部顶点若走 miter 再加 3 顶点。
    const maxVerts = segments.length * 6 + Math.max(0, segments.length - 1) * 3;
    const buf = new Float32Array(maxVerts * POLYLINE_FLOATS_PER_VERTEX);
    let vi = 0;
    const push = (x: number, y: number, along: number, across: number): void => {
        const base = vi * POLYLINE_FLOATS_PER_VERTEX;
        buf[base] = x;
        buf[base + 1] = y;
        buf[base + 2] = along;
        buf[base + 3] = across;
        vi += 1;
    };

    for (let i = 0; i < segments.length; i += 1) {
        const s = segments[i];
        const a = pts[i];
        const b = pts[i + 1];
        const a0 = segStart[i];
        const segLen = Math.hypot(b.x - a.x, b.y - a.y);
        const a1 = a0 + segLen;
        // 起点两侧
        const axL = a.x + s.nx * offset;
        const ayL = a.y + s.ny * offset;
        const axR = a.x - s.nx * offset;
        const ayR = a.y - s.ny * offset;
        // 终点两侧
        const bxL = b.x + s.nx * offset;
        const byL = b.y + s.ny * offset;
        const bxR = b.x - s.nx * offset;
        const byR = b.y - s.ny * offset;

        // 两个三角形（逆时针无关紧要：关闭背面剔除，且混合与顺序无关）
        push(axL, ayL, a0, offset);
        push(axR, ayR, a0, -offset);
        push(bxL, byL, a1, offset);

        push(axR, ayR, a0, -offset);
        push(bxR, byR, a1, -offset);
        push(bxL, byL, a1, offset);

        // 内部顶点：尝试 miter 连接
        if (i + 1 >= segments.length) continue;
        const next = segments[i + 1];
        const sx = s.nx + next.nx;
        const sy = s.ny + next.ny;
        const m2 = sx * sx + sy * sy;
        // 两段几乎反向（180° 折返）时 miter 无解，直接留 bevel
        if (m2 < 1e-12) continue;
        // miter 向量：长度 = offset / cos(转角/2)
        const mx = (sx * 2 * offset) / m2;
        const my = (sy * 2 * offset) / m2;
        const ratio = Math.hypot(mx, my) / offset;
        if (!(ratio <= args.miterLimit)) continue;
        const apexX = b.x + mx;
        const apexY = b.y + my;
        // 尖角三角形覆盖两侧：左半（沿 s 的法线侧）与右半
        push(bxL, byL, a1, offset);
        push(bxR, byR, a1, -offset);
        push(apexX, apexY, a1, offset);
        push(bxR, byR, a1, -offset);
        push(apexX, apexY, a1, offset);
        push(apexX, apexY, a1, -offset);
    }

    return buf.subarray(0, vi * POLYLINE_FLOATS_PER_VERTEX);
}
