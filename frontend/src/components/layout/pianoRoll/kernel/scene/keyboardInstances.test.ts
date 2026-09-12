/**
 * 参数编辑器内核 · 钢琴键盘轴实例构建（纯函数）单测。
 *
 * 【本测试守护什么】
 * 1. 可见半音区间是**左闭右开**（`< endMidi`）——与网格层的闭区间不同，这处
 *    不对称是既有行为，写错会多/少一个键；
 * 2. 键体高度的 1px 下限（缩放到极小时键盘不能出现空洞）；
 * 3. **覆盖率修正**的正确性：GL 的实例矩形是硬边、不做抗锯齿，而 Canvas2D 的
 *    `fillRect`/`stroke` 会按面积覆盖率把边界像素混合。因此构建器必须把落在
 *    设备像素内部/设备行内部的边界，拆成"对齐到整像素 + alpha 乘以覆盖率"的
 *    矩形。写错的表现是边界少一层墨或多一层墨，只差 1 个设备像素、极难归因
 *    （实测曾导致每个八度的 C 分隔线上方少一整行）。
 * 4. 黑键右缘渐变按设备列精确展开（不是固定段数近似）。
 *
 * 【等价性如何验证】末尾的 describe 复刻 `render.ts` 的行循环，并比较
 * **逐设备行的墨量场**（面积口径：Σ 宽 × 行内纵向重叠 × alpha）。选这个判据
 * 是因为它没有"矩形归属到哪个键"的歧义——相邻键共享边界，任何基于位置或中心
 * 的归属都会在边界处产生 ±1 行的误差，而墨量场是渲染的真实契约。
 */
import { describe, expect, it } from "vitest";

import { buildKeyboardInstances, type KeyboardInstance } from "./keyboardInstances";

const WHITE = [1, 1, 1, 1] as const;
// 刻意不用纯黑：黑键的**边界列** alpha 会被覆盖率衰减，若 RGB 与渐变同为
// (0,0,0) 就无法用颜色区分「键体边界列」与「渐变段」。
const BLACK = [0.2, 0.2, 0.2, 1] as const;
const GRADIENT = [0, 0, 0, 0.4] as const;
const C_SEP = [1, 0, 0, 1] as const;
const SEP = [0, 1, 0, 1] as const;
const BORDER = [0, 0, 1, 0.5] as const;

/**
 * 判定是否为分隔线段。
 *
 * 特殊说明：分隔线按**覆盖率**展开——亚像素线会被拆成多个设备行矩形、每行 alpha
 * 乘以覆盖率。因此不能用 `rgba === C_SEP` 做指纹比较（alpha 已被缩放），
 * 改为比较 RGB 三通道（色相不变）。
 */
const isCLine = (i: { rgba: readonly number[] }) =>
    i.rgba[0] === C_SEP[0] && i.rgba[1] === C_SEP[1] && i.rgba[2] === C_SEP[2];
const isWeakLine = (i: { rgba: readonly number[] }) =>
    i.rgba[0] === SEP[0] && i.rgba[1] === SEP[1] && i.rgba[2] === SEP[2];
/**
 * 判定是否为键体矩形（白键或黑键）。
 *
 * 特殊说明：键体可能被拆成多个矩形（边界列覆盖率修正），因此不能用"一个键一个
 * 矩形"来断言；按 RGB 识别后还需按 y 聚合成"每个键的水平覆盖范围"。
 */
const isWhiteBody = (i: { rgba: readonly number[] }) =>
    i.rgba[0] === WHITE[0] && i.rgba[1] === WHITE[1] && i.rgba[2] === WHITE[2];
const isBlackBody = (i: { rgba: readonly number[] }) =>
    i.rgba[0] === BLACK[0] && i.rgba[1] === BLACK[1] && i.rgba[2] === BLACK[2];

/** 把同一键（相同 y）的键体矩形聚合为水平覆盖范围。 */
const bodyExtent = (group: { x: number; w: number }[]) => {
    // `+ 0` 把 -0 归一为 +0：`Math.min(-0, 0)` 返回 -0，会让 `toBe(0)` 失败。
    const left = Math.min(...group.map((r) => r.x)) + 0;
    const right = Math.max(...group.map((r) => r.x + r.w)) + 0;
    return { left, right, width: right - left };
};

/** 判定是否为黑键渐变段（黑色 RGB、alpha 在 (0,1) 开区间）。 */
const isGradientSeg = (i: { rgba: readonly number[] }) =>
    i.rgba[0] === 0 && i.rgba[1] === 0 && i.rgba[2] === 0 && i.rgba[3] > 0 && i.rgba[3] < 1;

/** 取一组覆盖率矩形的「覆盖率加权重心 y」。 */
const coverageCentroid = (group: { y: number; h: number; rgba: readonly number[] }[]) => {
    const cov = group.reduce((sum, i) => sum + i.rgba[3], 0);
    return group.reduce((sum, i) => sum + (i.y + i.h / 2) * i.rgba[3], 0) / cov;
};

/** 覆盖展开后每个矩形恒为 1 个设备像素高（dpr = 2 时为 0.5 CSS px）。 */
const DEVICE_ROW_H = 0.5;

/** 取一组覆盖率矩形的「总覆盖高度（CSS px）」。 */
const coverageHeight = (group: { rgba: readonly number[] }[]) =>
    group.reduce((sum, i) => sum + i.rgba[3], 0) * DEVICE_ROW_H;

/**
 * 把实例按「相同颜色 + 纵向连续」合并成**逻辑图元**。
 *
 * 【为什么需要】覆盖率修正会把一个键体拆成多个矩形（上面/:中间/下面；黑键还会
 * 因右缘落在设备列内部再拆左右），因此"一个键 = 一个矩形"不再成立。按颜色把
 * 纵向连续的片段合并，即可还原出"每个键"的逻辑范围，同时不受拆分数量影响。
 *
 * @param items 实例序列。
 * @returns 合并后的逻辑图元（含纵向范围与水平范围）。
 */
/**
 * 把键体实例按「相同颜色 + 纵向重叠」聚类成**每个键**。
 *
 * 【为什么不按"纵向相邻"合并】覆盖率修正会把部分覆盖的边界带**对齐到设备行**，
 * 相邻两个键的边界带因此可能落在同一行上、彼此重叠。用"首尾相接"判定会把它们
 * 错误地合成一组。改为按**纵向重叠**聚类，可以正确区分不同键。
 *
 * @param items 实例序列。
 * @returns 每个键的实例集合与其纵向范围。
 */
function clusterKeyBodies(items: KeyboardInstance[]) {
    const groups: KeyboardInstance[][] = [];
    for (const item of [...items].sort((l, r) => l.y - r.y)) {
        const hit = groups.find((g) =>
            g.some(
                (other) =>
                    other.rgba[0] === item.rgba[0] &&
                    other.rgba[1] === item.rgba[1] &&
                    other.rgba[2] === item.rgba[2] &&
                    item.y < other.y + other.h - 1e-9 &&
                    other.y < item.y + item.h - 1e-9,
            ),
        );
        if (hit) hit.push(item);
        else groups.push([item]);
    }
    return groups.map((group) => {
        const yTop = Math.min(...group.map((g) => g.y));
        const yBottom = Math.max(...group.map((g) => g.y + g.h));
        const xLeft = Math.min(...group.map((g) => g.x));
        const xRight = Math.max(...group.map((g) => g.x + g.w));
        return { group, yTop, yBottom, xLeft, xRight };
    });
}

/** 值 → y 的线性投影桩（值 100 在顶部、0 在底部）。 */
const makeValueToY = () => (midi: number, h: number) => ((100 - midi) / 100) * h;

const isBlackKey = (midi: number) => {
    const pc = ((midi % 12) + 12) % 12;
    return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
};

/** 造一份构建参数。 */
function makeArgs(overrides: Partial<Parameters<typeof buildKeyboardInstances>[0]> = {}) {
    return {
        view: { center: 50, span: 4 },
        absMin: 36,
        absMax: 96,
        heightPx: 100,
        axisWidthPx: 56,
        dpr: 2,
        valueToY: makeValueToY(),
        isBlackKey,
        whiteKeyRgba: WHITE,
        blackKeyRgba: BLACK,
        blackKeyGradientRgba: GRADIENT,
        cSeparatorRgba: C_SEP,
        keySeparatorRgba: SEP,
        axisBorderRgba: BORDER,
        ...overrides,
    };
}

describe("buildKeyboardInstances", () => {
    it("可见半音区间左闭右开（与 render.ts 的 < endMidi 一致）", () => {
        // span 4 / center 50 → min 48, max 52 → 键 48,49,50,51（不含 52）
        const items = buildKeyboardInstances(makeArgs());
        // 每个键一条分隔线；亚像素弱线会展开成多个设备行矩形，故按"总覆盖高度"
        // 而不是"实例个数"来断言。48 是 C（1 CSS px 强线），49/50/51 是弱线
        // （各 0.5 CSS px）：总覆盖高度 = 1 + 0.5 × 3 = 2.5 CSS px。
        const separators = items.filter((i) => isWeakLine(i) || isCLine(i));
        expect(coverageHeight(separators)).toBeCloseTo(2.5, 9);
        // 每个键一条分隔线，4 个键
        // 键体会被覆盖率拆成多个矩形，合并回"每个键"再计数。
        const bodyKeys = clusterKeyBodies(items.filter((i) => isWhiteBody(i) || isBlackBody(i)));
        expect(bodyKeys.length).toBe(4);
    });

    it("白键铺满轴宽，黑键只占 72%", () => {
        const items = buildKeyboardInstances(makeArgs());
        // 覆盖率修正会把一个键拆成多个矩形，先合并回"逻辑图元"再比较范围。
        const merged = clusterKeyBodies(items);
        const whiteKeys = merged.filter((m) => isWhiteBody(m.group[0]));
        const blackKeys = merged.filter((m) => isBlackBody(m.group[0]));
        expect(whiteKeys.length).toBeGreaterThan(0);
        expect(blackKeys.length).toBeGreaterThan(0);
        for (const key of whiteKeys) {
            expect(key.xLeft + 0).toBe(0);
            // 轴宽 56 是整数、dpr=2 -> 右缘恰在设备像素上，不产生边界列。
            expect(key.xRight).toBe(56);
        }
        for (const key of blackKeys) {
            expect(key.xLeft + 0).toBe(0);
            const { width } = bodyExtent(key.group);
            // 黑键宽 0.72×56 = 40.32，右缘落在设备列内部，故聚合范围会向上取到完整
            // 设备列（dpr=2 时 40.5）——聚合宽度略大于标称是**预期**的，真实的 40.32
            // 体现在最右列的 alpha 覆盖率上（该列 alpha 按 0.64 衰减）。
            expect(width).toBeGreaterThanOrEqual(56 * 0.72 - 1e-9);
            expect(width).toBeLessThanOrEqual(56 * 0.72 + 1 / 2 + 1e-9);
        }
    });

    it("键体高度有 1px 下限（缩放到极小时不出现空洞）", () => {
        // span 极大 -> 每键投影高度远小于 1
        const items = buildKeyboardInstances(makeArgs({ view: { center: 66, span: 60 } }));
        // 只对**键体**断言：分隔线本就是 0.5/1px 的细线，不受该下限约束。
        // 覆盖率拆分会产出高度不足 1px 的**边界带**（它们用 alpha 表达覆盖率，
        // 高度固定为 1 个设备行），因此不能用"每个矩形的高度"断言 1px 下限。
        // 正确的不变量是「每个键的**覆盖率加权高度**」>= 1：加权高度 = Σ(h × alpha)
        // 还原了该键实际覆盖的纵向像素量。
        const merged = clusterKeyBodies(items.filter((i) => isWhiteBody(i) || isBlackBody(i)));
        expect(merged.length).toBeGreaterThan(0);
        for (const body of merged) {
            const covered = body.group.reduce((sum, r) => sum + r.h * r.rgba[3], 0);
            expect(covered).toBeGreaterThanOrEqual(1 - 1e-9);
        }
    });

    it("分隔线：C 用 1px 强线 + C 色，其余 0.5px + 弱色", () => {
        // 48..52 含 48（C）
        const items = buildKeyboardInstances(makeArgs());
        const cLines = items.filter(isCLine);
        const weakLines = items.filter(isWeakLine);
        // C 强线：1 CSS px = 2 设备行，各自满覆盖（alpha 不衰减）。
        expect(coverageHeight(cLines)).toBeCloseTo(2 * DEVICE_ROW_H, 9);
        expect(cLines.every((i) => i.rgba[3] === 1)).toBe(true);
        // 弱线：0.5 CSS px = 1 设备 px，跨 2 行各半覆盖 —— 每键 2 段、alpha 减半。
        expect(weakLines.length).toBe(3 * 2);
        for (const line of weakLines) {
            expect(line.h).toBeCloseTo(DEVICE_ROW_H, 9);
            expect(line.rgba[3]).toBeCloseTo(0.5, 9); // 覆盖率 0.5
        }
        expect(coverageHeight(weakLines)).toBeCloseTo(3 * DEVICE_ROW_H, 9);
    });

    it("分隔线的覆盖率加权重心 == 描边中心（stroke 与矩形的语义差）", () => {
        const dpr = 2;
        const items = buildKeyboardInstances(makeArgs({ dpr }));
        const iter = makeValueToY();
        // 复算第 48 键的 top：top = min(valueToY(48), valueToY(49))
        const top48 = Math.min(iter(48, 100), iter(49, 100));
        const expectedCenter = top48 + 0.5;
        const cLines = items.filter(isCLine);
        expect(cLines.length).toBeGreaterThan(0);
        // 用覆盖率加权重心而非单段"上缘 + 半宽"：前者同时守住展开后的对称性。
        expect(coverageCentroid(cLines)).toBeCloseTo(expectedCenter, 12);
    });

    it("黑键渐变：逐设备列展开、沿 x 递增、覆盖完整渐变区", () => {
        const items = buildKeyboardInstances(makeArgs());
        // 渐变段：黑色 RGB 且 alpha 在 (0,1) 开区间。必须显式限定 RGB——覆盖率
        // 展开让弱分隔线、键体边界带也落入"半透明"区间（灰色 RGB），不限定会混入。
        const segs = items.filter(isGradientSeg);
        expect(segs.length).toBeGreaterThan(1);

        // 按**纵向带**（y + h）分组：纵向覆盖率修正会把同一黑键的渐变拆成多条
        // y 带（上边界带 / 整段 / 下边界带），每条带内各自沿 x 逐列展开。
        const byBand = new Map<string, typeof segs>();
        for (const segment of segs) {
            const key = `${segment.y.toFixed(6)}|${segment.h.toFixed(6)}`;
            const list = byBand.get(key) ?? [];
            list.push(segment);
            byBand.set(key, list);
        }
        expect(byBand.size).toBeGreaterThan(0);

        const dpr = 2;
        const startX = 56 * 0.62;
        const endX = 56 * 0.72;
        for (const list of byBand.values()) {
            const sorted = [...list].sort((a, b) => a.x - b.x);
            // 沿 x 不透明度递增（渐变从左端透明到右端最深）。
            //
            // 特殊说明：**末列除外**——渐变区右缘落在设备列内部，末列只被部分覆盖，
            // 其 alpha 还乘了覆盖率（实测 0.249 < 前一列的 0.359）。这不是错误：
            // Canvas2D 的渐变矩形同样在 40.32 处截止，末列确实只被覆盖 64%。
            const interior = sorted.slice(0, -1);
            for (let i = 1; i < interior.length; i += 1) {
                expect(interior[i].rgba[3]).toBeGreaterThan(interior[i - 1].rgba[3]);
            }
            // 逐设备像素列展开：每列恰好 1 个设备列宽，列间首尾相接（无缝隙、无重叠）。
            // 不能用"宽度之和 == 渐变区宽"断言——边界列只被部分覆盖，但仍各占一个
            // 完整设备列矩形，故宽度之和略大于渐变区宽（实测 6 vs 5.6）；覆盖率体现
            // 在各列的 alpha 上，不在宽度上。
            for (const segment of sorted) {
                expect(segment.w).toBeCloseTo(1 / dpr, 9);
            }
            for (let i = 1; i < sorted.length; i += 1) {
                expect(sorted[i].x).toBeCloseTo(sorted[i - 1].x + sorted[i - 1].w, 9);
            }
            // 起止边界：覆盖被渐变区触及的第一列与最后一列
            expect(sorted[0].x).toBeCloseTo(Math.floor(startX * dpr) / dpr, 9);
            const last = sorted[sorted.length - 1];
            expect(last.x + last.w).toBeCloseTo(Math.ceil(endX * dpr) / dpr, 9);
            // 末端列接近峰值（渐变在右端最深）
            expect(last.rgba[3]).toBeGreaterThan(GRADIENT[3] * 0.4);
        }
    });

    it("渐变峰值不透明度等于传入的渐变 alpha", () => {
        const items = buildKeyboardInstances(makeArgs());
        const segs = items.filter(isGradientSeg);
        const maxAlpha = Math.max(...segs.map((s) => s.rgba[3]));
        // 最后一段取 t = (n-0.5)/n < 1，故峰值略小于传入 alpha 但接近
        expect(maxAlpha).toBeLessThanOrEqual(0.4 + 1e-9);
        expect(maxAlpha).toBeGreaterThan(0.4 * 0.85);
    });

    it("非有限 span / 非法 dpr 返回空数组（防死循环与 NaN）", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(buildKeyboardInstances(makeArgs({ view: { center: 50, span: bad } }))).toEqual(
                [],
            );
        }
        expect(buildKeyboardInstances(makeArgs({ dpr: 0 }))).toEqual([]);
    });

    it("值域被钳制到绝对范围内（span 超过全域时用全域）", () => {
        const items = buildKeyboardInstances(makeArgs({ view: { center: 66, span: 1000 } }));
        const bodyKeys = clusterKeyBodies(items.filter((i) => isWhiteBody(i) || isBlackBody(i)));
        // 36..96 左闭右开 → 60 个键（覆盖拆分的多个矩形合并为一个键）
        expect(bodyKeys.length).toBe(60);
    });
});

/**
 * 与 Canvas2D 路径的**逐值等价**守护。
 *
 * 【为什么必须有这一组】上面的用例只证明"构建器符合我对 render.ts 的理解"。
 * 这里原地复刻 `render.ts:428-496` 的键循环（不改动原文件），在
 * center × span × 高度 的组合空间上逐键比对：键体几何、分隔线几何与颜色判定。
 * 任何一处口径差异立刻失败——包括最容易写错的「左闭右开」区间与
 * 「stroke 中心 → 矩形上缘」换算。
 */
describe("与 render.ts 键循环逐值等价", () => {
    const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
    const proj = (midi: number, h: number) => ((100 - midi) / 100) * h;

    function legacyKeys(center: number, spanRaw: number, h: number, w: number) {
        const absMin = 36;
        const absMax = 96;
        const span = clamp(spanRaw, 1e-6, absMax - absMin);
        const min = clamp(center - span / 2, absMin, absMax - span);
        const max = min + span;
        const startMidi = clamp(Math.floor(min), absMin, absMax);
        const endMidi = clamp(Math.ceil(max), absMin, absMax);
        const out: {
            black: boolean;
            top: number;
            keyH: number;
            bodyW: number;
            lineW: number;
            lineCenterY: number;
            isC: boolean;
        }[] = [];
        for (let midi = startMidi; midi < endMidi; midi += 1) {
            const y0 = proj(midi, h);
            const y1 = proj(midi + 1, h);
            const top = Math.min(y0, y1);
            const keyH = Math.max(1, Math.max(y0, y1) - top);
            const black = isBlackKey(midi);
            const pc = ((midi % 12) + 12) % 12;
            out.push({
                black,
                top,
                keyH,
                bodyW: black ? w * 0.72 : w,
                lineW: pc === 0 ? 1 : 0.5,
                lineCenterY: top + 0.5,
                isC: pc === 0,
            });
        }
        return out;
    }

    it("center × span × 高度组合下键体与分隔线几何完全一致", () => {
        for (const center of [36, 48, 60, 72, 96, 66.5]) {
            for (const span of [1e-6, 6, 24, 60]) {
                for (const height of [400, 823]) {
                    const legacy = legacyKeys(center, span, height, 56);
                    const built = buildKeyboardInstances(
                        makeArgs({
                            view: { center, span },
                            heightPx: height,
                        }),
                    );
                    // 键体：比较**逐设备行的墨量场**（面积口径）——这是渲染的真实契约，
                    // 且没有"矩形归属到哪个键"的歧义（归属判定在边界处天生含糊：
                    // 相邻键共享边界，边界带会同时压到两键的名义区间上）。
                    //
                    // 量纲必须是「面积」而不是「纵向高度之和」：黑键被横向拆成
                    // "键体(宽 0.72w) + 右边界列(宽 0.5，alpha 0.64)"两片，若对每片的
                    // 纵向重叠都累加一次高度，同一行会被计两次（实测 0.82 vs 期望 0.5）。
                    // 正确做法是每行累加 Σ(宽 × 行内纵向重叠 × alpha)，与 legacy 的
                    // `fillRect` 面积一致。
                    const bodies = built.filter((i) => isWhiteBody(i) || isBlackBody(i));
                    const rowHeight = 1 / 2;
                    const firstRow = Math.floor(Math.min(...legacy.map((k) => k.top)) * 2);
                    const lastRow = Math.ceil(Math.max(...legacy.map((k) => k.top + k.keyH)) * 2);
                    for (let row = firstRow; row <= lastRow; row += 1) {
                        const rowTop = row * rowHeight;
                        const rowBottom = rowTop + rowHeight;
                        const expected = legacy.reduce(
                            (sum, k) =>
                                sum +
                                k.bodyW *
                                    Math.max(
                                        0,
                                        Math.min(rowBottom, k.top + k.keyH) -
                                            Math.max(rowTop, k.top),
                                    ),
                            0,
                        );
                        const actual = bodies.reduce(
                            (sum, r) =>
                                sum +
                                r.w *
                                    r.rgba[3] *
                                    Math.max(
                                        0,
                                        Math.min(rowBottom, r.y + r.h) - Math.max(rowTop, r.y),
                                    ),
                            0,
                        );
                        // 失败时给出参数上下文，便于定位是哪一组（纯数值差异很难归因）
                        expect(
                            actual,
                            `center=${center} span=${span} height=${height} row=${row}`,
                        ).toBeCloseTo(expected, 9);
                    }
                    // 每个键的**最左**片段必然从 0 开始（覆盖率修正只拆分右边界列，
                    // 不移动左缘）。不能断言"每个矩形 x 都为 0"——黑键的右边界列
                    // 是独立矩形（x = 0.72w），那是刻意拆分出来的。
                    for (const k of legacy) {
                        const tone = k.black ? BLACK : WHITE;
                        const own = bodies.filter(
                            (r) => r.rgba[0] === tone[0] && r.rgba[1] === tone[1],
                        );
                        // 该颜色所有片段的最小 x 必为 0
                        const minX = Math.min(...own.map((r) => r.x));
                        expect(minX + 0).toBe(0);
                    }
                    // 分隔线：覆盖率展开后每条线变成 1~2 个相邻设备行矩形。
                    //
                    // 判据同样用**覆盖率场**而不是"把矩形归属到某条线"：把线段按设备
                    // 行采样，比较每行的墨量 = Σ(线宽 × alpha)。这与键体用的是同一套
                    // 口径，且不需要任何归属阈值。
                    //
                    // 期望侧：legacy 的每条线覆盖其 [center − w/2, center + w/2] 区间，
                    // 逐行取交集长度即该行的墨量。
                    const lines = built.filter((i) => isWeakLine(i) || isCLine(i));
                    const sepFirst = Math.floor(
                        Math.min(...legacy.map((k) => k.lineCenterY - k.lineW / 2)) * 2,
                    );
                    const sepLast = Math.ceil(
                        Math.max(...legacy.map((k) => k.lineCenterY + k.lineW / 2)) * 2,
                    );
                    for (let row = sepFirst; row <= sepLast; row += 1) {
                        const rowTop = row * rowHeight;
                        const rowBottom = rowTop + rowHeight;
                        const expected = legacy.reduce(
                            (sum, k) =>
                                sum +
                                Math.max(
                                    0,
                                    Math.min(rowBottom, k.lineCenterY + k.lineW / 2) -
                                        Math.max(rowTop, k.lineCenterY - k.lineW / 2),
                                ),
                            0,
                        );
                        const actual = lines.reduce(
                            (sum, r) =>
                                sum +
                                r.rgba[3] *
                                    Math.max(
                                        0,
                                        Math.min(rowBottom, r.y + r.h) - Math.max(rowTop, r.y),
                                    ),
                            0,
                        );
                        expect(
                            actual,
                            `separator row center=${center} span=${span} height=${height} row=${row}`,
                        ).toBeCloseTo(expected, 9);
                    }
                }
            }
        }
    });
});
