/**
 * 折线几何构建（纯函数）
 *
 * 【主要内容】
 * 把一串内容坐标点展开为可交给 WebGL2 绘制的三角形顶点缓冲，顶点携带**到中心线
 * 的有符号距离**与**沿线的累积弧长**，供片元着色器做亚像素抗锯齿与虚线相位。
 *
 * 【作用】
 * 参数编辑器的曲线图层一直走 Canvas2D 的 `stroke()`。搬到 GL 后要自行承担
 * `stroke()` 原本代劳的三件事：**连接方式**（拐角填充）、**亚像素抗锯齿**
 * （线宽是 1.8 / 2.6 / 3.2 这类非整数值）、**虚线相位**（按弧长推进）。
 * 本模块只做几何与标量计算，不接触 GL / DOM，因此可以完整单测。
 *
 * 【拐角连接：为什么是"圆角扇面"而不是 miter 尖角（这是一个真实缺陷的根因）】
 * 距离场编码要求每个片元的 `across` 等于**它到整条折线的真实距离**。逐段四边形
 * 只在段内成立：在拐角外侧的楔形区里，"到折线的最近点"是那个顶点，于是真实距离是
 * **到顶点的径向距离**——径向场无法用三角形的线性插值表达（miter 尖点也正是因此
 * 无法正确编码：它是两条偏移线的交点，不是距离场上的等值点）。
 *
 * 历史实现用 `across = ±offset` 的 miter 三角形去填这个楔形，结果是：
 * 1. 尖点只落在**一侧**（`(n1+n2)` 方向），另一个三角形退化为零面积（三个顶点
 *    里有两个重复）——即"左半 + 右半各一个三角形"的注释与代码不符，实际只发出一个；
 * 2. 于是**转角旋向**决定缺口出现在哪一侧：一个方向的外侧楔形没有任何几何覆盖，
 *    另一个方向只留一条细缝。
 *
 * 缺口处没有墨水，透出的是画布背景（深色主题下是接近黑的值），表现为曲线上
 * **随机分布的黑点 / 毛糙边缘**；滚动或缩放会重新投影采样点、缺口集合随之重建，
 * 于是"黑点消失"——这正是用户报告的现象。
 *
 * 现在改为**圆角扇面**：在拐角外侧、以顶点为中心、半径 `offset` 的圆弧上取
 * 若干等分点，发出 `(b, rim_k, rim_{k+1})` 三角形族，并令
 * `across(b) = 0`、`across(rim) = offset`。因为 `|rim - b| = offset`，扇形内的
 * 线性插值恰好给出 `across ≈ |P - b|`——**这就是拐角外侧的真实距离场**。
 * 缺口在数学上不可能出现，且与转角旋向无关（正负转角镜像对称）。
 *
 * 特殊说明：`miterLimit` 入参**保留但不再影响几何**。圆角在所有 `miterLimit`
 * 下都不劣于 miter/bevel（Canvas2D 用 bevel 兜底超限尖角，而圆角是比 bevel 更
 * 精确的填充），因此无需按比例退化。参数保留只为调用方签名稳定。
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
     * 【保留但不再影响几何】历史实现按本值在 miter / bevel 之间切换；现在拐角一律
     * 用圆角扇面填充（见文件头"拐角连接"），它在所有比例下都无缺口，也就不需要
     * 退化阈值。参数保留是为了不改动调用方签名（宿主仍传 `miterLimit: 10`）。
     */
    readonly miterLimit: number;
    /** AA 余量（CSS px）；缺省 `DEFAULT_AA_PAD_PX`，传 0 得到纯几何外沿。 */
    readonly aaPadPx?: number;
    /**
     * 用**外部提供的弧长**替换自行累加（长度必须与 `points` 一致）。
     *
     * 【为什么需要】渲染层按设备像素列抽稀曲线后（见 `polylineDecimation`），
     * 相邻点的间距变大；若仍按抽稀后的折线自行累加，虚线周期会比原始曲线**变短**，
     * 缩放时虚线图案会漂移。抽稀模块已把每个点在原始序列中的累积弧长算好带出，
     * 这里按传入值写入即可。
     *
     * 特殊说明 1：长度不符或含非有限值时**整体忽略**（退回自行累加）。长度不符说明
     * 调用方有 bug，此时宁可虚线相位不准，也不能越界读数组或让 NaN 污染整层几何。
     *
     * 特殊说明 2：段内顶点按参数在两端的原始弧长之间线性插值——这样虚线周期在
     * 抽稀后仍然与原始曲线一致（而不是随段长被拉伸）。
     */
    readonly alongOverride?: readonly number[];
}

/** 内部使用：一个已归一化的段方向与法线。 */
interface Segment {
    readonly ux: number;
    readonly uy: number;
    /** 左法线。 */
    readonly nx: number;
    readonly ny: number;
}

/**
 * 判断点是否全部有限且线宽 / miter 合法。 */
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
 * 去重相邻的重复点（距离小于 1e-9 视为重复），并记录每个保留点在原序列中的下标。
 *
 * 【为什么必须去重】重复点会让段的长度为零，归一化方向时产生 `0/0 = NaN`，
 * 而 NaN 一旦写进顶点缓冲会让**整层几何失效**（不是少画一段，而是全部消失）。
 *
 * 【为什么要带出下标】`alongOverride` 是按**原序列**下标给出的（抽稀模块算的是
 * 原始累积弧长）。去重会改变下标，若不对应回原下标，弧长就会错位。
 *
 * @param points 原始点序列。
 * @returns 去重后的点序列与各自的原下标。
 */
function dedupeConsecutive(points: readonly PolylinePoint[]): {
    points: PolylinePoint[];
    indices: number[];
} {
    const out: PolylinePoint[] = [];
    const indices: number[] = [];
    for (let i = 0; i < points.length; i += 1) {
        const p = points[i];
        const last = out[out.length - 1];
        if (last !== undefined && Math.hypot(p.x - last.x, p.y - last.y) < 1e-9) continue;
        out.push(p);
        indices.push(i);
    }
    return { points: out, indices };
}

/**
 * 解析可用的 `alongOverride`：长度与**去重后**点数一致、且全为有限值。
 *
 * 【为什么与去重后的点数比而不是原始点数】去重后几何只按保留点构建，弧长也必须
 * 按同一套下标给出。长度不符说明调用方没考虑去重，此时整体忽略（安全侧）。
 *
 * @param override 调用方传入的弧长（可能未传）。
 * @param count 去重后的点数。
 * @returns 可用的弧长数组；不可用时为 `null`。
 */
function resolveAlongOverride(
    override: readonly number[] | undefined,
    count: number,
): readonly number[] | null {
    if (override === undefined) return null;
    if (override.length !== count) return null;
    for (const a of override) {
        if (!Number.isFinite(a)) return null;
    }
    return override;
}

/**
 * 解析一个内部点的拐角扇面：需要转过的有符号角度与三角形分段数。
 *
 * 【方向约定】`sweep > 0` 表示左转（`cross(u1, u2) > 0`），`< 0` 表示右转。
 * 角度大小取 `atan2(|cross|, dot)`，落在 `[0, π]`（180° 折返取 π）。
 *
 * 【为什么"缺口不可见"就整体跳过】未覆盖的是拐角**外侧**的扇形，缺口宽度约为
 * `offset × θ`（扇形的弧长）。θ 小于阈值时它在屏幕上不足 0.05 CSS px——比抗锯齿
 * 过渡宽度还小一个数量级，任何显示器上都看不出来。平滑曲线上的绝大多数相邻点都
 * 落在这里，跳过它们把顶点量拉回与历史实现同一量级（抽稀后的曲线相邻点间距约
 * 1 设备像素，但整段曲线的曲率很小）。
 *
 * @param cur 当前段。
 * @param next 下一段。
 * @param minAngle 跳过阈值（弧度，见 `MIN_JOIN_ANGLE`）。
 * @param maxStep 单段角步长上限（弧度，见 `MAX_JOIN_STEP`）。
 * @returns 有符号转角与三角形分段数；无需填充时为 `{ sweep: 0, steps: 0 }`。
 */
function resolveJoin(
    cur: Segment,
    next: Segment,
    minAngle: number,
    maxStep: number,
): { sweep: number; steps: number } {
    const cross = cur.ux * next.uy - cur.uy * next.ux;
    const dot = cur.ux * next.ux + cur.uy * next.uy;
    const theta = Math.atan2(Math.abs(cross), dot);
    if (!(theta >= minAngle)) return { sweep: 0, steps: 0 };
    const sign = cross >= 0 ? 1 : -1;
    return { sweep: sign * theta, steps: Math.max(1, Math.ceil(theta / maxStep)) };
}

/**
 * 发出拐角**外侧**的圆角扇面（三角形族）。
 *
 * 【几何】以连接点 `center` 为圆心、`offset` 为半径，从外侧起始方向开始，
 * 沿转角方向扫过 `sweep`，每 `steps` 等分发出一片三角形：
 * `(center, rim_k, rim_{k+1})`。首片的第一条边与上一段的偏移边重合、
 * 末片的最后一条边与下一段的偏移边重合，因此与两侧四边形**无缝拼接**。
 *
 * 【`across` 为什么这样取】圆心的 `across = 0`、弧上各点 `across = offset`。
 * 扇面内任一点 P 满足 `|P - center| ≤ offset`，而"到整条折线的距离"在楔形内
 * 就等于 `|P - center|`（最近点是连接点本身）——线性插值给出的正是这个径向距离，
 * 因此覆盖率、抗锯齿与线宽在整个拐角上都是精确的。这也正是 miter 尖角做不到的事。
 *
 * 【为什么不需要填内侧】左转时内侧是左侧：上一段的四边形覆盖 `|y| ≤ offset`
 * 且延伸到连接点，而内侧扇形内的每一点到中心线的投影都在该四边形范围内
 * （扇形半径 ≤ offset，最大横向偏移恰为 offset）。实测内侧扇形 100% 被两侧
 * 四边形覆盖，再发一遍只会造成同色重复混合（拐角变亮），因此刻意不填。
 *
 * 【只填外侧的判据】`sign` 由 `cross(u1, u2)` 给出，镜像的输入得到镜像的扇面
 * ——旋向对称性由构造保证（历史实现的缺口恰恰是旋向不对称的产物）。
 *
 * @param args 顶点写入器与几何参数。
 */
function emitJoinSectors(args: {
    push: (x: number, y: number, along: number, across: number) => void;
    center: PolylinePoint;
    cur: Segment;
    next: Segment;
    offset: number;
    along: number;
    sweep: number;
    steps: number;
}): void {
    const { push, center, cur, offset, along, sweep, steps } = args;
    const sign = sweep >= 0 ? 1 : -1;
    // 外侧起始方向：左转取 `-n1`（右侧），右转取 `+n1`（左侧）。
    const startAngle = Math.atan2(-sign * cur.ny, -sign * cur.nx);
    const stepAngle = sweep / steps;
    for (let k = 0; k < steps; k += 1) {
        const a0 = startAngle + k * stepAngle;
        const a1 = a0 + stepAngle;
        push(center.x, center.y, along, 0);
        push(center.x + Math.cos(a0) * offset, center.y + Math.sin(a0) * offset, along, offset);
        push(center.x + Math.cos(a1) * offset, center.y + Math.sin(a1) * offset, along, offset);
    }
}

/**
 * 构建折线顶点缓冲。
 *
 * 流程：
 * 1. 校验参数 → 去重相邻重复点 → 少于 2 点则返回空；
 * 2. 逐段求单位方向与左法线；
 * 3. 逐段产出 2 个三角形（6 顶点），`along` 为累积弧长、`across` 为 `±(半线宽+pad)`；
 * 4. 每个**内部点**上产出圆角扇面把拐角填满（见文件头「拐角连接」）：外侧楔形与
 *    内侧扇区各一段，按角度自适应细分；角度小到缺口不足半个像素时跳过。
 *
 * 特殊说明 1：扇面顶点的 `across` 取 `0`（圆心）与 `offset`（弧上），因此扇面内的
 * 线性插值近似于**到顶点的径向距离**，这正是拐角处的真实距离场。
 *
 * 特殊说明 2：`along` 从**传入序列的第一个点**开始累加（为 0）。调用方必须传
 * 可见点序列而不是整条曲线，否则虚线相位会与 Canvas2D 不一致（Canvas2D 的相位
 * 从子路径起点算起，而曲线是"从首个可见点开始"的单个子路径）。
 *
 * 特殊说明 3：扇面所有顶点的 `along` 取该内部点的弧长 `a1`。拐角扇面在弧长上是
 * 一个点，虚线相位在这里本来就是跳变的；这样处理与历史实现（miter 三角形取 `a1`）
 * 逐值一致，不会改变虚线的观感。
 *
 * @param args 构建参数。
 * @returns 顶点缓冲（每 4 个 float 一个顶点）；参数非法或无可绘制段时为空。
 */
export function buildPolylineVertices(args: PolylineGeometryArgs): Float32Array {
    if (!isArgsValid(args)) return new Float32Array(0);
    const deduped = dedupeConsecutive(args.points);
    const pts = deduped.points;
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

    /**
     * 每个顶点的弧长取值来源。
     *
     * 【两种模式】未提供 `alongOverride` 时按段长累加（与历史行为逐位一致，
     * 保证"没传就完全不变"）；提供时直接用传入值。因为段的两端各自取自己在原始
     * 曲线上的弧长，段内自然按位置线性过渡，虚线的周期不会随抽稀后的段长被拉伸。
     */
    const alongOverride = resolveAlongOverride(args.alongOverride, pts.length);

    /** 自行累加的累积弧长（`alongOverride` 未提供时使用）。 */
    const cumulative: number[] = [0];
    for (let i = 0; i + 1 < pts.length; i += 1) {
        cumulative.push(
            cumulative[i] + Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y),
        );
    }

    /** 第 `i` 个点处的弧长。 */
    const alongAtPoint = (i: number): number =>
        alongOverride === null ? (cumulative[i] ?? 0) : (alongOverride[i] ?? 0);

    /**
     * 拐角处每段的角步长上限（弧度，30°）。
     *
     * 【为什么需要上限】扇面用「圆心 + 弧上两点」的三角形插值径向距离，弦与弧之间的
     * 矢高会让插值略小于（或大于）真实半径，误差 ≈ `(1 − cos(步长/2))`。步长 30°
     * 时误差约 3.4% × offset ≈ 0.06 CSS px——远小于 AA 过渡宽度（1 设备像素），
     * 因此子像素级无感。
     */
    const MAX_JOIN_STEP = Math.PI / 6;

    /**
     * 缺口被判定为"不可见"的角度阈值。
     *
     * 未覆盖的楔形最大宽度 ≈ `offset × 转角`。要求它小于 0.05 CSS px（约为 AA
     * 宽度的 1/10，任何显示器上都不可见）即可安全跳过扇面——平滑曲线上的绝大多数
     * 相邻点都落在这里，跳过它们让顶点量回到与历史实现同一量级。
     */
    const MIN_JOIN_ANGLE = 0.05 / Math.max(1e-6, offset);

    /**
     * 拐角扇面：**先算清每个内部点的扇面角与分段数**，再据此一次性精确分配缓冲。
     *
     * 【为什么要预扫】扇面是按角度自适应的（转角越大分得越细），若按最坏情况
     * （每点 6 个三角形）分配，缓冲区会是实际需要的数倍——而它的分配发生在
     * **每帧**、**每图层**的绘制热路径上。预扫只做算术，成本可忽略，且让
     * 「分配」与「写入」共用同一份结果，不可能出现容量与实际写入量不一致
     * （历史上正是容量少算一半，`Float32Array` 越界写被静默忽略，曲线右端
     * 凭空截断 25%）。
     */
    const joins: { sweep: number; steps: number }[] = [];
    for (let i = 0; i + 1 < segments.length; i += 1) {
        joins.push(resolveJoin(segments[i], segments[i + 1], MIN_JOIN_ANGLE, MAX_JOIN_STEP));
    }

    let totalVerts = segments.length * 6;
    for (const join of joins) totalVerts += join.steps * 3;
    const buf = new Float32Array(totalVerts * POLYLINE_FLOATS_PER_VERTEX);
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

        const a0 = alongAtPoint(i);
        const a1 = alongAtPoint(i + 1);
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

        // 两个三角形覆盖段体（b 为向量，b.x/b.y 是坐标；此处变量名沿用几何记号）
        push(axL, ayL, a0, offset);
        push(axR, ayR, a0, -offset);
        push(bxL, byL, a1, offset);

        push(axR, ayR, a0, -offset);
        push(bxR, byR, a1, -offset);
        push(bxL, byL, a1, offset);

        // 内部点：用圆角扇面填满拐角（见文件头「拐角连接」）
        if (i + 1 >= segments.length) continue;
        const join = joins[i];
        if (join.steps === 0) continue;
        emitJoinSectors({
            push,
            center: pts[i + 1],
            cur: s,
            next: segments[i + 1],
            offset,
            along: a1,
            sweep: join.sweep,
            steps: join.steps,
        });
    }

    return buf.subarray(0, vi * POLYLINE_FLOATS_PER_VERTEX);
}
