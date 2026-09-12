/**
 * 参数编辑器内核 · 网格实例构建（纯函数）单测。
 *
 * 【本测试守护什么】像素级复刻 `render.ts:671-821` 的网格语义：取值域的
 * **整数半音 / 步进**、强弱线的判定阈值、以及两者**不同的**半像素取向
 * （弱线 +0.5/dpr，强线不加）。取向写错不会报错，只会让网格整体偏移半个
 * 设备像素——表现为"发虚"，很难归因。
 *
 * 【为什么断言 w 与 h 分开】横线的 `w` 是横向范围、`h` 是线厚，两者写反会画出
 * 一条通高的竖条而不是一条线。这类转置错误在数值上都"有限且合理"，只有分开
 * 断言才能暴露。
 */
import { describe, expect, it } from "vitest";

import { buildPitchGridInstances, buildValueGridInstances } from "./gridInstances";

const RED = [1, 0, 0, 1] as const;
const BLUE = [0, 0, 1, 1] as const;
const WHITE = [1, 1, 1, 1] as const;
const BLACK = [0, 0, 0, 1] as const;

/** 值 → y 的线性投影桩（值 100 在顶部、0 在底部），便于手算期望。 */
const makeValueToY = () => (v: number, h: number) => ((100 - v) / 100) * h;

describe("buildPitchGridInstances", () => {
    it("按整数半音逐行产出，范围含端点（与 render.ts 的 <= 一致）", () => {
        const items = buildPitchGridInstances({
            view: { center: 50, span: 10 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        });
        // span 10 / center 50 → min 45, max 55 → 45..55 共 11 行
        expect(items.length).toBe(11);
        expect(items[0].value).toBe(45);
        expect(items[items.length - 1].value).toBe(55);
    });

    it("pc === 0（C）用 colorC，其余用 colorOther", () => {
        const items = buildPitchGridInstances({
            view: { center: 48, span: 4 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        });
        // 46..50 → 只有 48 是 C
        const cRows = items.filter((i) => i.rgba === RED);
        expect(cRows.length).toBe(1);
        expect(cRows[0].value).toBe(48);
    });

    it("横线几何：x=0、w=视口宽、h=线厚（易错点：宽与厚不可互换）", () => {
        const items = buildPitchGridInstances({
            view: { center: 50, span: 2 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        });
        for (const item of items) {
            expect(item.x).toBe(0);
            expect(item.w).toBe(800); // 横向范围
            expect(item.h).toBeCloseTo(0.5, 9); // 1/dpr = 线厚
            // y 是矩形**上缘**：弱线厚 1 设备像素且中心对齐到 k+0.5 设备像素，
            // 故上缘恰落在整数设备像素上（2*y 为整数）。这是「描边中心 → 上缘」
            // 换算的结果；直接用中心值会让整条线下移半个线厚（曾实际发生）。
            expect(Math.abs((item.y * 2) % 1)).toBeCloseTo(0, 9);
        }
    });

    it("值域被钳制到绝对值域内（span 大于全域时用全域）", () => {
        const items = buildPitchGridInstances({
            view: { center: 66, span: 1000 },
            absMin: 36,
            absMax: 96,
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        });
        // span 被钳到 60 → 36..96 共 61 行
        expect(items.length).toBe(61);
        expect(items[0].value).toBe(36);
        expect(items[60].value).toBe(96);
    });

    it("非有限 span / center 返回空数组（防死循环与 NaN）", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(
                buildPitchGridInstances({
                    view: { center: 50, span: bad },
                    absMin: 36,
                    absMax: 96,
                    heightPx: 100,
                    viewportWidthPx: 800,
                    dpr: 1,
                    valueToY: makeValueToY(),
                    colorC: RED,
                    colorOther: BLUE,
                }),
            ).toEqual([]);
        }
    });

    it("非法 dpr / 高度返回空数组", () => {
        const base = {
            view: { center: 50, span: 10 },
            absMin: 36,
            absMax: 96,
            viewportWidthPx: 800,
            valueToY: makeValueToY(),
            colorC: RED,
            colorOther: BLUE,
        } as const;
        expect(buildPitchGridInstances({ ...base, heightPx: 100, dpr: 0 })).toEqual([]);
        expect(buildPitchGridInstances({ ...base, heightPx: Number.NaN, dpr: 1 })).toEqual([]);
    });
});

describe("buildValueGridInstances", () => {
    it("cents 参数按 100 步进、%1200 为强线", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        // -150..150 步进 100 → -100, 0, 100（0 为强线）
        expect(items.map((i) => Math.round(i.value)).sort((a, b) => a - b)).toEqual([-100, 0, 100]);
        const strong = items.filter((i) => i.rgba === WHITE);
        expect(strong.length).toBe(1);
        expect(strong[0].value).toBe(0);
    });

    it("degrees 参数按 1 步进、%7 为强线", () => {
        const items = buildValueGridInstances({
            kind: "degrees",
            view: { center: 0, span: 7 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        // -3.5..3.5 步进 1 → -3..3 共 7 行；强线只有 0
        expect(items.length).toBe(7);
        const strong = items.filter((i) => i.rgba === WHITE);
        expect(strong.length).toBe(1);
        expect(strong[0].value).toBe(0);
    });

    it("formant 参数按 50 步进、%600 为强线", () => {
        const items = buildValueGridInstances({
            kind: "formantCents",
            view: { center: 0, span: 200 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        // -100..100 步进 50 → -100,-50,0,50,100
        expect(items.length).toBe(5);
        expect(items.filter((i) => i.rgba === WHITE).length).toBe(1);
    });

    it("强线不加半像素、弱线加半像素（两种取向都必须保留）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        expect(items.length).toBeGreaterThan(0);
        // 换算成矩形上缘后，两种取向都落在整数设备像素上——因为它们各自的
        // 中心（弱线 k+0.5、强线 k）减去半个线厚（弱线 0.5、强线 1 设备像素）
        // 都得到整数。因此这里断言的等价性质是"上缘在设备像素栅格上"。
        for (const item of items) {
            const frac = Math.abs((item.y * 2) % 1);
            expect(frac).toBeCloseTo(0, 9);
        }
    });

    it("强线线厚是弱线的两倍（2/dpr vs 1/dpr）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 300 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 2,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        for (const item of items) {
            const expected = item.rgba === WHITE ? 1 : 0.5; // 2/dpr vs 1/dpr
            expect(item.h).toBeCloseTo(expected, 9);
            expect(item.w).toBe(800); // 横向范围不受强弱影响
        }
    });

    it("span 非正时退化为单行（与 render.ts 的 1e-6 下限一致，不是空数组）", () => {
        const items = buildValueGridInstances({
            kind: "cents",
            view: { center: 0, span: 0 },
            heightPx: 100,
            viewportWidthPx: 800,
            dpr: 1,
            valueToY: makeValueToY(),
            strongRgba: WHITE,
            weakRgba: BLACK,
        });
        // render.ts:743 用 Math.max(1e-6, view.span)：span=0 时域退化为 [0,0]，
        // 循环仍会产出 v=0 这一行。**不是**空数组——写成空数组会让实现与既有
        // 渲染分叉（少一条零线）。
        expect(items.length).toBe(1);
        expect(items[0].value).toBeCloseTo(0, 9);
    });

    it("span 为 NaN / Infinity 时返回空数组（防死循环）", () => {
        // render.ts 的 `Math.max(1e-6, NaN)` 仍是 NaN，`for` 上界为 Infinity 时
        // 循环永不终止 —— 原实现没有这层保护，因为调用方保证 span 有限。
        // 几何构建跑在渲染热路径上，一旦死循环整个面板会卡死，故必须显式防御。
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(
                buildValueGridInstances({
                    kind: "cents",
                    view: { center: 0, span: bad },
                    heightPx: 100,
                    viewportWidthPx: 800,
                    dpr: 1,
                    valueToY: makeValueToY(),
                    strongRgba: WHITE,
                    weakRgba: BLACK,
                }),
            ).toEqual([]);
        }
    });
});

/**
 * 与 Canvas2D 路径的**逐值等价**守护。
 *
 * 【为什么必须有这一组】上面那些用例只证明"构建器符合我对 render.ts 的理解"；
 * 一旦理解有偏差，测试会自信地通过而实际渲染已经错位。这里原地复刻
 * `render.ts` 的行循环（不改动原文件），在 center × span × dpr 的组合空间上
 * 逐行比对行数、y 与强弱判定——任何一处口径差异都会立刻失败。
 *
 * 【覆盖的边界】span 取 `1e-6`（下限退化）、6（半音级）、24（默认）、60（全域）；
 * dpr 取 1 / 1.25 / 2 / 3（含**分数 DPR**，两种半像素取向在这里才会分叉）。
 */
describe("与 render.ts 行循环逐值等价", () => {
    const clamp = (v: number, lo: number, hi: number) => Math.min(Math.max(v, lo), hi);
    const proj = (v: number, h: number) => ((100 - v) / 100) * h;
    const hairline = (y: number, dpr: number) => (Math.round(y * dpr) + 0.5) / dpr;
    const snap = (y: number, dpr: number) => Math.round(y * dpr) / dpr;

    /** 复刻 render.ts:672-740 的音高行循环。 */
    function legacyPitchRows(center: number, spanRaw: number, h: number, dpr: number) {
        const absMin = 36;
        const absMax = 96;
        const span = clamp(spanRaw, 1e-6, absMax - absMin);
        const min = clamp(center - span / 2, absMin, absMax - span);
        const max = min + span;
        const startMidi = clamp(Math.floor(min), absMin, absMax);
        const endMidi = clamp(Math.ceil(max), absMin, absMax);
        const rows: { midi: number; y: number }[] = [];
        for (let midi = startMidi; midi <= endMidi; midi += 1) {
            rows.push({ midi, y: hairline(proj(midi + 0.5, h), dpr) });
        }
        return rows;
    }

    /** 复刻 render.ts:741-766 的 cents 行循环。 */
    function legacyCentsRows(center: number, spanRaw: number, h: number, dpr: number) {
        const span = Math.max(1e-6, spanRaw);
        const vMin = center - span / 2;
        const vMax = center + span / 2;
        const step = 100;
        const start = Math.ceil(vMin / step) * step;
        const rows: { v: number; y: number; strong: boolean }[] = [];
        for (let v = start; v <= vMax + step * 0.01; v += step) {
            const strong = Math.round(v) % 1200 === 0;
            rows.push({ v, y: strong ? snap(proj(v, h), dpr) : hairline(proj(v, h), dpr), strong });
        }
        return rows;
    }

    it("音高网格：center × span × dpr 组合下 y 与行数完全一致", () => {
        for (const center of [36, 48, 66, 96, 72.5]) {
            for (const span of [1e-6, 6, 24, 60]) {
                for (const dpr of [1, 1.25, 2, 3]) {
                    const legacy = legacyPitchRows(center, span, 100, dpr);
                    const built = buildPitchGridInstances({
                        view: { center, span },
                        absMin: 36,
                        absMax: 96,
                        heightPx: 100,
                        viewportWidthPx: 800,
                        dpr,
                        valueToY: proj,
                        colorC: RED,
                        colorOther: BLUE,
                    });
                    expect(built.length).toBe(legacy.length);
                    for (let i = 0; i < built.length; i += 1) {
                        expect(built[i].value).toBe(legacy[i].midi);
                        // legacy 给出的是**描边中心**，构建器给出**矩形上缘**：
                        // 两者相差半个线厚，比较时换算回去。
                        expect(built[i].y + built[i].h / 2).toBeCloseTo(legacy[i].y, 12);
                    }
                }
            }
        }
    });

    it("cents 网格：center × span × dpr 组合下 y、强弱判定与行数完全一致", () => {
        for (const center of [-1200, -50, 0, 37.5, 600, 2400]) {
            for (const span of [1e-6, 100, 300, 2400]) {
                for (const dpr of [1, 1.25, 2, 3]) {
                    const legacy = legacyCentsRows(center, span, 100, dpr);
                    const built = buildValueGridInstances({
                        kind: "cents",
                        view: { center, span },
                        heightPx: 100,
                        viewportWidthPx: 800,
                        dpr,
                        valueToY: proj,
                        strongRgba: WHITE,
                        weakRgba: BLACK,
                    });
                    expect(built.length).toBe(legacy.length);
                    for (let i = 0; i < built.length; i += 1) {
                        expect(built[i].value).toBeCloseTo(legacy[i].v, 9);
                        // 同上：上缘 + 半厚 == 描边中心。
                        expect(built[i].y + built[i].h / 2).toBeCloseTo(legacy[i].y, 12);
                        expect(built[i].rgba === WHITE).toBe(legacy[i].strong);
                    }
                }
            }
        }
    });
});
