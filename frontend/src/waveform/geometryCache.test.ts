/**
 * 波形几何复用判定的单测。
 *
 * 【为什么这段判定必须有单测】它是波形性能的命门：判定为真时一帧的成本是
 * 「一次 uniform 更新 + drawArrays」，与 clip 数、像素列数**完全无关**；判定为假
 * 时每帧都要重建场景与几何（400 clip / 全览实测 ≈ 10.8 ms）。两者差一个数量级，
 * 而判定的正确性又同时决定了「波形会不会缺内容」——只能靠用例钉住。
 *
 * 【覆盖两类失效】
 * 1. **性能失效**（不该重建时重建）：本文件用「仅 scrollLeft 变化且在窗口内 →
 *    必须复用」直接钉住。这正是时间轴平移帧的主路径。
 * 2. **正确性失效**（不该复用时复用）：视口越出构建窗口、rows 引用变化、缩放 /
 *    尺寸 / dpr / 颜色 / 后端 / 幅度映射及其修订号变化，以及调用方没有承诺
 *    `rowsCoverViewport` 时——都必须拒绝复用，否则会把缺内容或错内容的旧几何
 *    平移上去（历史上表现为「波形消失、随滚动又恢复」）。
 */
import { describe, expect, it } from "vitest";

import {
    canReuseGeometry,
    type WaveformGeometryAnchor,
    type WaveformReuseQuery,
} from "./geometryCache";
import type { WaveformSceneRow } from "./sceneBuilder";

/** 视口宽（CSS px）。 */
const VIEW_W = 1600;
/** 视口高（CSS px）。 */
const VIEW_H = 900;
/** 构建窗口两侧的水平余量（对应 draw() 的 `marginPx`）。 */
const MARGIN = 512;
/** 锚点与查询共用的视口左缘（内容坐标）。 */
const SCROLL_LEFT = 10_000;
/** 竖直方向的最大可复用 scrollTop：几何底端 − 视口高。 */
const MAX_SCROLL_TOP = 2000;
/**
 * 几何实际覆盖的内容底端（各行波形带底边的最大值，与 draw() 现在写入锚点
 * 的口径一致）：= MAX_SCROLL_TOP + VIEW_H，使「最大可复用 scrollTop 的视口
 * 底边」恰好贴着几何底端 —— 边界用例（贴边复用 / 越界 1px 重建）有干净的
 * 数字。
 */
const GEOMETRY_BOTTOM_PX = MAX_SCROLL_TOP + VIEW_H;

/**
 * 造一行最小可用的场景行。
 *
 * 判定只看 `rows` 的**引用**（React 侧 memo 的产物，引用不变即内容不变），
 * 因此每次调用都返回全新数组，用来模拟「行数据换了」。
 *
 * 波形带必须覆盖整个视口：竖直复用判定要求视口**完整**落在几何覆盖内
 * （底边 ≤ 几何底端），行覆盖不足时任何 scrollTop 都会拒绝复用。
 */
function rows(): WaveformSceneRow[] {
    return [
        {
            topPx: 0,
            waveformTopPx: 0,
            waveformHeightPx: GEOMETRY_BOTTOM_PX,
            clips: [],
        },
    ];
}

/** 构建锚点：1600×900 视口、dpr=1、水平两侧各留 512px 余量。 */
function anchor(overrides: Partial<WaveformGeometryAnchor> = {}): WaveformGeometryAnchor {
    return {
        pxPerSec: 40,
        widthPx: VIEW_W,
        heightPx: VIEW_H,
        dpr: 1,
        rows: rows(),
        color: "#8fa3bf",
        rendererKind: "webgl2",
        amplitudeMap: undefined,
        amplitudeRevision: 0,
        windowStartPx: SCROLL_LEFT - MARGIN,
        windowEndPx: SCROLL_LEFT + VIEW_W + MARGIN,
        windowTopPx: 0,
        windowBottomPx: GEOMETRY_BOTTOM_PX,
        ...overrides,
    };
}

/**
 * 构建查询：缺省即「与给定锚点完全一致的那一帧」。
 *
 * 必须从**同一个锚点**派生默认值——否则 `rows` 会是另一个数组引用，判定按引用
 * 比较必然为假，用例会假失败。
 */
function query(
    a: WaveformGeometryAnchor,
    overrides: Partial<WaveformReuseQuery> = {},
): WaveformReuseQuery {
    return {
        pxPerSec: a.pxPerSec,
        widthPx: a.widthPx,
        heightPx: a.heightPx,
        dpr: a.dpr,
        rows: a.rows,
        color: a.color,
        rendererKind: a.rendererKind,
        amplitudeMap: a.amplitudeMap,
        amplitudeRevision: a.amplitudeRevision,
        scrollLeftPx: SCROLL_LEFT,
        scrollTopPx: 0,
        rowsCoverViewport: true,
        ...overrides,
    };
}

describe("canReuseGeometry", () => {
    it("无锚点（首帧）时必须重建", () => {
        expect(canReuseGeometry(null, query(anchor()))).toBe(false);
    });

    it("★ 仅 scrollLeft 变化且视口仍在窗口内 → 复用（时间轴平移帧的主路径）", () => {
        const a = anchor();
        // 余量 512px，取 ±500 保守。
        for (const scrollLeftPx of [
            SCROLL_LEFT - 500,
            SCROLL_LEFT - 1,
            SCROLL_LEFT,
            SCROLL_LEFT + 1,
            SCROLL_LEFT + 500,
        ]) {
            expect(canReuseGeometry(a, query(a, { scrollLeftPx }))).toBe(true);
        }
    });

    it("★ 仅 scrollTop 变化且视口完整落在几何覆盖内 → 复用（竖直平移帧）", () => {
        const a = anchor();
        // 2000 是最大可复用值：视口底边 = 2000 + 900 = 2900 = 几何底端（贴边含）。
        for (const scrollTopPx of [0, 400, 1999, 2000]) {
            expect(canReuseGeometry(a, query(a, { scrollTopPx }))).toBe(true);
        }
    });

    it("★ 视口底边越出几何实际覆盖 → 必须重建（快速竖直平移的空白条）", () => {
        const a = anchor();
        // 几何只画到 windowBottomPx = 2900：scrollTop=2001 时视口底边 = 2901，
        // 越出的 1px 没有任何几何可画 —— 旧判定只查顶边会复用，把空白条原样
        // 平移上去（快速竖直平移时视口底部肉眼可见）。
        expect(canReuseGeometry(a, query(a, { scrollTopPx: 2001 }))).toBe(false);
        // 贴边（视口底边恰等于几何底端）仍可复用：判定是闭区间。
        expect(canReuseGeometry(a, query(a, { scrollTopPx: 2000 }))).toBe(true);
    });

    it("视口越出窗口左边界 → 必须重建", () => {
        const a = anchor();
        // windowStartPx = 9488：视口左缘正好贴上窗口边时仍可复用。
        expect(canReuseGeometry(a, query(a, { scrollLeftPx: SCROLL_LEFT - MARGIN }))).toBe(true);
        expect(canReuseGeometry(a, query(a, { scrollLeftPx: SCROLL_LEFT - MARGIN - 1 }))).toBe(
            false,
        );
    });

    it("视口越出窗口右边界 → 必须重建", () => {
        const a = anchor();
        // windowEndPx = SCROLL_LEFT + VIEW_W + MARGIN：
        // 视口右缘 = scrollLeft + VIEW_W 不得超过它。
        expect(canReuseGeometry(a, query(a, { scrollLeftPx: SCROLL_LEFT + MARGIN }))).toBe(true);
        expect(canReuseGeometry(a, query(a, { scrollLeftPx: SCROLL_LEFT + MARGIN + 1 }))).toBe(
            false,
        );
    });

    it("视口顶边越出行覆盖范围 → 必须重建", () => {
        const a = anchor();
        // -1：顶边高于窗口顶。2001：底边越出几何底端（见上面的空白条用例），
        // 旧行为里它由「顶边 ≤ windowBottomPx」拦下，现由底边覆盖判定拦下，
        // 结果一致。
        expect(canReuseGeometry(a, query(a, { scrollTopPx: -1 }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { scrollTopPx: 2001 }))).toBe(false);
    });

    it("★ 调用方未承诺 rowsCoverViewport 时一律重建（保守默认）", () => {
        const a = anchor();
        // 锚点、视口、窗口全都匹配，唯一差别是这个承诺。
        expect(canReuseGeometry(a, query(a, { rowsCoverViewport: false }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { rowsCoverViewport: true }))).toBe(true);
    });

    it("rows 引用变化（行窗口切换 / clip 编辑）→ 必须重建", () => {
        const a = anchor();
        expect(canReuseGeometry(a, query(a, { rows: rows() }))).toBe(false);
    });

    it("缩放 / 尺寸 / dpr 任一变化 → 必须重建", () => {
        const a = anchor();
        expect(canReuseGeometry(a, query(a, { pxPerSec: 80 }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { widthPx: VIEW_W + 1 }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { heightPx: 901 }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { dpr: 2 }))).toBe(false);
    });

    it("颜色 / 渲染后端变化 → 必须重建", () => {
        const a = anchor();
        expect(canReuseGeometry(a, query(a, { color: "#ffffff" }))).toBe(false);
        expect(canReuseGeometry(a, query(a, { rendererKind: "canvas2d" }))).toBe(false);
    });

    it("幅度映射引用变化 → 必须重建（参数编辑器换面板语义）", () => {
        const map = (value: number, gain: number) => value * gain;
        const other = (value: number, gain: number) => value * gain * 2;
        // 锚点无映射、查询有映射：必须重建。
        expect(canReuseGeometry(anchor(), query(anchor(), { amplitudeMap: map }))).toBe(false);
        // 映射换成另一个引用：同样必须重建。
        expect(
            canReuseGeometry(
                anchor({ amplitudeMap: map }),
                query(anchor({ amplitudeMap: map }), { amplitudeMap: other }),
            ),
        ).toBe(false);
    });

    it("★ 幅度映射引用不变但修订号变化 → 必须重建（响度映射的延迟取值）", () => {
        const map = (value: number, gain: number) => value * gain;
        const a = anchor({ amplitudeMap: map, amplitudeRevision: 3 });
        expect(canReuseGeometry(a, query(a, { amplitudeRevision: 3 }))).toBe(true);
        expect(canReuseGeometry(a, query(a, { amplitudeRevision: 4 }))).toBe(false);
    });

    it("判定是纯函数：同一输入重复调用结果稳定，且不修改入参", () => {
        const a = anchor();
        const q = query(a, { scrollLeftPx: SCROLL_LEFT + 100 });
        const before = JSON.stringify({ ...a, rows: a.rows.length });
        for (let i = 0; i < 3; i += 1) expect(canReuseGeometry(a, q)).toBe(true);
        expect(JSON.stringify({ ...a, rows: a.rows.length })).toBe(before);
    });
});
