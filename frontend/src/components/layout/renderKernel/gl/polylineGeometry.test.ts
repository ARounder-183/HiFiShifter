/**
 * 折线几何构建（./polylineGeometry）行为自检。
 *
 * 【主要内容】
 * 1. 每条线段产出 2 个三角形（6 个顶点），顶点携带**到中心线的有符号距离**
 *    （`across`）与沿线的累积弧长（`along`），而不是被展开到精确外沿——后续要靠
 *    片元着色器做亚像素抗锯齿与虚线相位；
 * 2. 拐角用 **miter 连接**（Canvas2D 默认 `lineJoin: "miter"`，本工程从未设置过
 *    `lineJoin`）；miter 比例超限时必须退化为 bevel，否则尖角会甩出长刺；
 * 3. 几何外沿 = `半线宽 + aaPadPx`：不预留 AA 余量时最外圈没有像素可承载覆盖率
 *    衰减，线会比 Canvas2D 更细；
 * 4. 少于 2 个点、非有限坐标、零/负线宽都必须返回空而不是抛错或产出 NaN
 *    （NaN 会让整个顶点缓冲失效、整层消失）。
 *
 * 【测试设计说明：miter 断言为什么看**位置**而不是 `across`】
 * 距离编码下，miter 尖点的 `across` 仍是 `±半线宽`——因为尖点落在两条偏移线的
 * 交点上，而两条偏移线到各自中心线的垂直距离都是 `半线宽`。所以「顶点被推远了」
 * 这件事只体现在**位置**上。断言 `across` 会在两种连接方式下都通过，等于没测。
 *
 * 曾经还写错过用例本身：用 (0,0)-(100,0.5)-(200,0) 当"尖角"，它的转角实际只有
 * 0.57°、miter 比例 ≈1.000，**永远走不到 bevel 分支**。因此用例一律按**转角**
 * 构造，并用 `miter比例 = 1/cos(转角/2)` 这个解析值来挑选 limit。
 */
import { describe, expect, it } from "vitest";

import { buildPolylineVertices, POLYLINE_FLOATS_PER_VERTEX } from "./polylineGeometry";

/** 取出所有顶点的 `across` 分量。 */
function acrossOf(buf: Float32Array): number[] {
    const out: number[] = [];
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        out.push(buf[i + 3]);
    }
    return out;
}

/** 取出所有顶点的 `along` 分量。 */
function alongOf(buf: Float32Array): number[] {
    const out: number[] = [];
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        out.push(buf[i + 2]);
    }
    return out;
}

/**
 * 判定缓冲中是否存在位于 `(x, y)` 的顶点（容差 `tol`）。
 *
 * 【为什么直接找顶点而不是测"最远距离"】测最远距离会被**同一段另一端的顶点**
 * 干扰（段长 10 ≫ 半线宽 5），于是连接方式之间的差异都得到 10，断言失效；
 * 收窄邻域又会把大转角下远端的顶点排除掉，产生假失败。
 * 节点位置的解析值是已知的，直接按位置查找既无歧义也无阈值调参。
 */
function hasVertexAt(buf: Float32Array, x: number, y: number, tol = 1e-6): boolean {
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        if (Math.abs(buf[i] - x) <= tol && Math.abs(buf[i + 1] - y) <= tol) return true;
    }
    return false;
}

/**
 * 取出顶点缓冲里的三角形（每 3 个顶点一组，含各自的 `across`）。
 *
 * 用于把几何**光栅化**成覆盖率，从而验证"真实描边轮廓内的每个像素都有墨水"。
 * 这是唯一能抓住"拐角缺口"的不变量：缺口是**没有几何**，位置类断言看不见它。
 */
function trianglesOf(buf: Float32Array): { x: number; y: number; across: number }[][] {
    const tris: { x: number; y: number; across: number }[][] = [];
    const verts: { x: number; y: number; across: number }[] = [];
    for (let i = 0; i < buf.length; i += POLYLINE_FLOATS_PER_VERTEX) {
        verts.push({ x: buf[i], y: buf[i + 1], across: buf[i + 3] });
    }
    for (let i = 0; i + 2 < verts.length; i += 3) {
        tris.push([verts[i], verts[i + 1], verts[i + 2]]);
    }
    return tris;
}

/**
 * 对几何做点采样光栅化：返回 `(px, py)` 是否落在某个三角形内**且**该处的插值
 * `across` 在 `halfWidth` 之内（即着色器会给出非零覆盖率）。
 *
 * 这正是片元着色器的覆盖率判据（`|across| <= halfWidth`）在 CPU 侧的等价物。
 */
function rasterCovered(
    tris: { x: number; y: number; across: number }[][],
    px: number,
    py: number,
    halfWidth: number,
): boolean {
    for (const [A, B, C] of tris) {
        const d = (B.y - C.y) * (A.x - C.x) + (C.x - B.x) * (A.y - C.y);
        if (Math.abs(d) < 1e-12) continue;
        const w1 = ((B.y - C.y) * (px - C.x) + (C.x - B.x) * (py - C.y)) / d;
        const w2 = ((C.y - A.y) * (px - C.x) + (A.x - C.x) * (py - C.y)) / d;
        const w3 = 1 - w1 - w2;
        if (w1 < -1e-9 || w2 < -1e-9 || w3 < -1e-9) continue;
        const across = w1 * A.across + w2 * B.across + w3 * C.across;
        if (Math.abs(across) <= halfWidth) return true;
    }
    return false;
}

/** 点到线段的距离。 */
function distToSegment(
    px: number,
    py: number,
    a: { x: number; y: number },
    b: { x: number; y: number },
): number {
    const vx = b.x - a.x;
    const vy = b.y - a.y;
    const len2 = vx * vx + vy * vy;
    let t = len2 > 0 ? ((px - a.x) * vx + (py - a.y) * vy) / len2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (a.x + t * vx), py - (a.y + t * vy));
}

/** 点到折线的距离（真实描边轮廓 = 距离 <= 半线宽）。 */
function distToPolyline(px: number, py: number, pts: { x: number; y: number }[]): number {
    let best = Number.POSITIVE_INFINITY;
    for (let i = 0; i + 1 < pts.length; i += 1) {
        best = Math.min(best, distToSegment(px, py, pts[i], pts[i + 1]));
    }
    return best;
}

/**
 * 统计"真实描边内部却没有任何覆盖率"的采样点数量（即**缺口**）。
 *
 * 采样范围刻意排除两端 `半线宽 + 0.5` 的**平头端帽**区域：本渲染器不画端帽
 * （Canvas2D 的默认 `lineCap: "butt"` 同样如此），端帽附近真实轮廓与几何的差异
 * 属于设计而非缺陷。同理排除真实轮廓最外 0.6px（抗锯齿过渡带，覆盖率本就近零）。
 */
function countGaps(
    buf: Float32Array,
    pts: { x: number; y: number }[],
    lineWidth: number,
): number {
    const halfWidth = lineWidth / 2;
    const tris = trianglesOf(buf);
    let blanks = 0;
    let minX = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const p of pts) {
        minX = Math.min(minX, p.x - lineWidth);
        maxX = Math.max(maxX, p.x + lineWidth);
        minY = Math.min(minY, p.y - lineWidth);
        maxY = Math.max(maxY, p.y + lineWidth);
    }
    const first = pts[0];
    const last = pts[pts.length - 1];
    for (let y = minY; y <= maxY; y += 0.1) {
        for (let x = minX; x <= maxX; x += 0.1) {
            const d = distToPolyline(x, y, pts);
            if (d > halfWidth - 0.6) continue;
            if (Math.hypot(x - first.x, y - first.y) < halfWidth + 0.5) continue;
            if (Math.hypot(x - last.x, y - last.y) < halfWidth + 0.5) continue;
            if (!rasterCovered(tris, x, y, halfWidth)) blanks += 1;
        }
    }
    return blanks;
}

describe("buildPolylineVertices", () => {
    it("单段产出 2 个三角形（6 顶点）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBe(6 * POLYLINE_FLOATS_PER_VERTEX);
    });

    it("几何外沿 = 半线宽 + AA 余量（aaPadPx 可显式关闭）", () => {
        // aaPadPx = 0：纯几何语义，外沿恰好是半线宽
        const bare = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 4,
            miterLimit: 10,
            aaPadPx: 0,
        });
        const bareAcross = acrossOf(bare);
        expect(Math.min(...bareAcross)).toBeCloseTo(-2, 9);
        expect(Math.max(...bareAcross)).toBeCloseTo(2, 9);

        // 默认：外扩到 半线宽 + pad，给片元的覆盖率衰减留出采样空间。
        // 不这样做，最外圈像素拿不到部分覆盖，线会比 Canvas2D 细。
        const padded = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 4,
            miterLimit: 10,
        });
        const paddedAcross = acrossOf(padded);
        expect(Math.max(...paddedAcross)).toBeCloseTo(2.5, 9); // 2 + 0.5
        expect(Math.min(...paddedAcross)).toBeCloseTo(-2.5, 9);

        // 注意：`across` 记录的是**几何**距离（含 pad），而着色器的覆盖率阈值取
        // lineWidth/2。因此 `u_halfWidth` 必须作为独立 uniform 传入，
        // 不能从几何的 max|across| 反推。
    });

    it("沿线的累积弧长写入 along（供虚线相位使用）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 3, y: 4 },
                { x: 3, y: 14 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        const along = alongOf(out);
        // 第一段长 5（3-4-5 直角三角形），第二段长 10 -> 总 15
        expect(Math.min(...along)).toBeCloseTo(0, 9);
        expect(Math.max(...along)).toBeCloseTo(15, 9);
    });

    it("拐角用圆角扇面填满：外侧有顶点、且镜像旋向对称", () => {
        // 【为什么不再断言 miter 尖点】拐角连接已从「miter 尖角 / bevel」改为
        // 「圆角扇面」（见 polylineGeometry 文件头的「拐角连接」）：历史实现在拐角
        // **外侧**留下未被覆盖的楔形，表现为曲线上的黑点（透出画布背景）。扇面把
        // 外侧填满，因此本用例改断言"外侧确实有几何"与"旋向镜像对称"——后者是
        // 历史缺陷的直接指纹（同一个 90° 转角，左转有缺口、右转没有）。
        const rightAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10, y: 10 },
        ];
        const mirrorAngle = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10, y: -10 },
        ];
        const lineWidth = 10;
        const offset = lineWidth / 2; // aaPadPx: 0 时几何外沿即半线宽

        const left = buildPolylineVertices({
            points: rightAngle,
            lineWidth,
            miterLimit: 10,
            aaPadPx: 0,
        });
        const right = buildPolylineVertices({
            points: mirrorAngle,
            lineWidth,
            miterLimit: 10,
            aaPadPx: 0,
        });

        // 左转的外侧是右侧：扇面必须触及 (10 + offset, 0) 方向。
        expect(hasVertexAt(left, 10 + offset, 0)).toBe(true);
        // 右转的外侧是左侧：镜像位置必须同样存在。
        expect(hasVertexAt(right, 10 - offset, 0)).toBe(true);

        // 顶点数必须逐值相等（镜像输入 → 镜像几何，不允许一侧多填/少填）。
        expect(right.length).toBe(left.length);

        // 且相对连接点镜像：把右侧缓冲的 y 取反后，顶点集合应与左侧一致。
        const key = (x: number, y: number) => `${x.toFixed(6)},${y.toFixed(6)}`;
        const leftSet = new Set<string>();
        for (let i = 0; i < left.length; i += POLYLINE_FLOATS_PER_VERTEX) {
            leftSet.add(key(left[i], left[i + 1]));
        }
        for (let i = 0; i < right.length; i += POLYLINE_FLOATS_PER_VERTEX) {
            // 连接点在 y = 0，绕该点镜像即 y → -y
            expect(leftSet.has(key(right[i], -right[i + 1]))).toBe(true);
        }
    });

    it("圆角扇面在 miterLimit 变化下不变（无退化阈值）", () => {
        const points = [
            { x: 0, y: 0 },
            { x: 10, y: 0 },
            { x: 10, y: 10 },
        ];
        const tight = buildPolylineVertices({ points, lineWidth: 10, miterLimit: 1, aaPadPx: 0 });
        const loose = buildPolylineVertices({ points, lineWidth: 10, miterLimit: 10, aaPadPx: 0 });
        expect(Array.from(loose)).toEqual(Array.from(tight));
    });

    /**
     * 覆盖率不变量：**真实描边轮廓内的每个采样点都必须有墨水**（缺陷回归）。
     *
     * 【为什么必须有这条用例】历史实现在拐角外侧留下未覆盖的楔形：那里的像素没有
     * 任何几何，于是透出画布背景（深色主题下接近黑），用户看到的是"曲线上随机的
     * 黑点 / 毛糙边缘"。缺口是**没有几何**，位置类断言（顶点是否存在）与标量类
     * 断言（`across` 取值）都看不见它——只有把顶点缓冲真正光栅化、再与真实描边
     * 轮廓比对才能抓住。
     *
     * 【为什么用多个转角与两种旋向】历史缺口是**旋向不对称**的（右转无缺口、
     * 左转有），只测一个方向会漏掉一半。这里正负各取三个转角，任一方向出现缺口
     * 都会失败。
     */
    it("拐角外侧无缺口：真实描边内部处处有覆盖率（旋向镜像对称）", () => {
        const lineWidth = 2.6;
        const turnsDeg = [90, 120, 165];
        for (const deg of turnsDeg) {
            const rad = (deg * Math.PI) / 180;
            for (const sign of [1, -1]) {
                const pts = [
                    { x: 0, y: 0 },
                    { x: 40, y: 0 },
                    {
                        x: 40 + Math.cos(sign * rad) * 40,
                        y: Math.sin(sign * rad) * 40,
                    },
                ];
                const buf = buildPolylineVertices({
                    points: pts,
                    lineWidth,
                    miterLimit: 10,
                });
                expect(buf.length).toBeGreaterThan(0);
                expect(countGaps(buf, pts, lineWidth)).toBe(0);
            }
        }
    });

    it("锯齿折线（抽稀曲线的典型形态）同样无缺口", () => {
        // 抽稀会把同列的 y 极值配对，产生大量接近 180° 的折返——正是缺口最宽的场景。
        const pts: { x: number; y: number }[] = [];
        for (let i = 0; i < 40; i += 1) {
            pts.push({ x: i * 2, y: i % 2 === 0 ? -6 : 6 });
        }
        const buf = buildPolylineVertices({ points: pts, lineWidth: 2.6, miterLimit: 10 });
        expect(buf.length).toBeGreaterThan(0);
        expect(countGaps(buf, pts, 2.6)).toBe(0);
    });

    it("少于 2 个点返回空数组", () => {
        expect(buildPolylineVertices({ points: [], lineWidth: 2, miterLimit: 10 }).length).toBe(0);
        expect(
            buildPolylineVertices({
                points: [{ x: 1, y: 1 }],
                lineWidth: 2,
                miterLimit: 10,
            }).length,
        ).toBe(0);
    });

    it("非有限坐标 / 非法线宽返回空数组（防 NaN 污染顶点缓冲）", () => {
        const bad = [
            {
                points: [
                    { x: 0, y: 0 },
                    { x: Number.NaN, y: 1 },
                ],
                lineWidth: 2,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: Number.POSITIVE_INFINITY },
                ],
                lineWidth: 2,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                ],
                lineWidth: 0,
                miterLimit: 10,
            },
            {
                points: [
                    { x: 0, y: 0 },
                    { x: 1, y: 1 },
                ],
                lineWidth: -3,
                miterLimit: 10,
            },
        ];
        for (const args of bad) {
            expect(buildPolylineVertices(args).length).toBe(0);
        }
    });

    it("重合点不产生 NaN（相邻点相同时跳过该段）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 5, y: 5 },
                { x: 5, y: 5 },
                { x: 15, y: 5 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBeGreaterThan(0);
        for (let i = 0; i < out.length; i += 1) {
            expect(Number.isFinite(out[i])).toBe(true);
        }
    });

    it("全部点重合时返回空数组（没有可绘制的段）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 5, y: 5 },
                { x: 5, y: 5 },
                { x: 5, y: 5 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        });
        expect(out.length).toBe(0);
    });

    /**
     * `alongOverride`：抽稀后仍按**原始**弧长推进虚线相位。
     *
     * 【为什么需要这个入口】渲染层会按设备像素列抽稀（见 `polylineDecimation`），
     * 抽稀后相邻点的间距变大，若用折线自身重新累加弧长，虚线周期会比原始曲线**变短**，
     * 表现为缩放时虚线图案漂移。抽稀模块已经把每个点在原始序列中的弧长算好带出来，
     * 这里只需支持按传入值写入 `along`。
     *
     * 【为什么不是"传一个起始偏移"】抽稀丢掉的是**中间**的点，弧长跳变发生在
     * 每一对相邻结果点之间，不是一个全局偏移能表达的。
     */
    it("alongOverride：along 按传入的弧长写入，不再自行累加", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
                { x: 20, y: 0 },
            ],
            lineWidth: 2,
            miterLimit: 10,
            // 模拟"中间被抽掉了 990px 的点"：第 3 个点的真实弧长是 1000，不是 20。
            alongOverride: [0, 10, 1000],
        });
        const along = alongOf(out);
        expect(Math.min(...along)).toBe(0);
        expect(Math.max(...along)).toBe(1000);
        // 折线自身累加只会得到 20；必须出现 1000 才说明用了传入值。
        expect(along).not.toContain(20);
    });

    it("alongOverride：段内的中间值按参数在两原始弧长之间线性插值", () => {
        // 单段、两端弧长为 0 与 100：段内任意顶点的 along 必须落在 [0, 100]，
        // 且两端的顶点分别取到 0 与 100（否则虚线的周期会随段长被拉伸）。
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 2,
            miterLimit: 10,
            alongOverride: [0, 100],
        });
        const along = alongOf(out);
        expect(Math.min(...along)).toBe(0);
        expect(Math.max(...along)).toBe(100);
        for (const a of along) {
            expect(a).toBeGreaterThanOrEqual(0);
            expect(a).toBeLessThanOrEqual(100);
        }
    });

    it("alongOverride 长度与点数不符时忽略它，退回自行累加（安全侧）", () => {
        // 长度不匹配说明调用方有 bug；此时宁可虚线相位不准，也不能越界读数组。
        for (const bad of [[0], [0, 1, 2, 3], []]) {
            const out = buildPolylineVertices({
                points: [
                    { x: 0, y: 0 },
                    { x: 10, y: 0 },
                ],
                lineWidth: 2,
                miterLimit: 10,
                alongOverride: bad,
            });
            const along = alongOf(out);
            expect(Math.min(...along)).toBe(0);
            expect(Math.max(...along)).toBeCloseTo(10, 9);
        }
    });

    it("alongOverride 含非有限值时忽略它（防 NaN 污染整层几何）", () => {
        const out = buildPolylineVertices({
            points: [
                { x: 0, y: 0 },
                { x: 10, y: 0 },
            ],
            lineWidth: 2,
            miterLimit: 10,
            alongOverride: [0, Number.NaN],
        });
        for (let i = 0; i < out.length; i += 1) {
            expect(Number.isFinite(out[i])).toBe(true);
        }
    });

    it("alongOverride 未传时行为与原来完全一致（自行累加）", () => {
        const args = {
            points: [
                { x: 0, y: 0 },
                { x: 3, y: 4 },
                { x: 3, y: 14 },
            ],
            lineWidth: 2,
            miterLimit: 10,
        };
        const withoutIt = buildPolylineVertices(args);
        const withUndefined = buildPolylineVertices({ ...args, alongOverride: undefined });
        expect(Array.from(withUndefined)).toEqual(Array.from(withoutIt));
        // 5 + 10 = 15
        expect(Math.max(...alongOf(withoutIt))).toBeCloseTo(15, 9);
    });
});

/**
 * 缓冲容量必须覆盖**全部**顶点（回归：曲线右端被硬截断）。
 *
 * 【缺陷现象】历史实现按「每段 6 顶点 + 每个内部顶点 3 顶点」估算容量，但 miter
 * 分支实际 push **6** 个顶点（尖角要覆盖两侧：左半 + 右半各一个三角形）。
 * `Float32Array` 的越界写入会被**静默忽略**，于是第 ~75% 之后的顶点全部丢失——
 * 表现为曲线在右侧被硬截断（实测 20s 曲线只画到 ~15s）。
 *
 * 【现在的保证更强】容量不再"估算"而是**预扫精确计算**（见 `resolveJoin`）：先按
 * 每个内部点的实际扇面分段数求和，再分配缓冲。因此容量与实际写入量在构造上恒等，
 * 不可能再出现越界。这里用大点数 + 大量急转角压这个边界。
 *
 * 【判据】末点对应的顶点必须存在。若缓冲不够，末点会被丢弃。
 */
describe("buildPolylineVertices 缓冲容量", () => {
    it("大量点 + 大量急转角时，末点顶点不会被丢弃", () => {
        // 4000 点、每点 x 递增 0.1、y 做大幅折返（保证每个内部顶点都要发扇面）。
        const n = 4000;
        const points = Array.from({ length: n }, (_, i) => ({
            x: i * 0.1,
            y: 100 + (i % 2 === 0 ? -20 : 20),
        }));
        const out = buildPolylineVertices({ points, lineWidth: 2.6, miterLimit: 10 });

        // 找到缓冲中的最大 x —— 必须覆盖最后一个点。
        let maxX = -Infinity;
        for (let i = 0; i < out.length; i += POLYLINE_FLOATS_PER_VERTEX) {
            if (out[i] > maxX) maxX = out[i];
        }
        const lastX = points[points.length - 1].x;
        // 几何外扩（半线宽+AA pad）后应略大于末点 x，绝不能小于。
        expect(maxX).toBeGreaterThanOrEqual(lastX);
    });

    it("无任何顶点被静默丢弃：缓冲长度恰好等于实际写入量的 4 倍", () => {
        // `subarray(0, vi * 4)` 的语义是"顶点数 == vi"。若容量比 vi 小，越界写被忽略，
        // length 会小于真实需要且末端坐标缺失——用末点 x 的存在性即可证伪。
        const n = 1500;
        const points = Array.from({ length: n }, (_, i) => ({
            x: i * 0.5,
            y: 50 + ((i * 37) % 11) * 10,
        }));
        const out = buildPolylineVertices({ points, lineWidth: 2, miterLimit: 10 });
        expect(out.length % POLYLINE_FLOATS_PER_VERTEX).toBe(0);
        let maxX = -Infinity;
        for (let i = 0; i < out.length; i += POLYLINE_FLOATS_PER_VERTEX) {
            if (out[i] > maxX) maxX = out[i];
        }
        expect(maxX).toBeGreaterThanOrEqual(points[points.length - 1].x);
    });
});
