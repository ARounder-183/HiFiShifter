/**
 * 参数编辑器内核 · GL 场景层视口解析与内容签名单测。
 *
 * 【本测试要钉住的核心不变量】
 * 1. **视口的竖向真值来自内核**（`scrollTop`），而不是面板写下的快照；
 * 2. **内容签名必须随内核竖向位置变化**——这是「网格冻住」的回归测试：
 *    签名取自快照时，竖向滚动（不进 React）不改变签名，几何永远停在旧位置。
 *    实测（1920×1200，pitch span=24）：内核中心 72 → 79.29（等价内容位移 250px），
 *    GL 画布上的横线中心逐像素完全相同，而旧实现 Canvas2D 路径同条件移动了 10px。
 * 3. 无变化时签名稳定（不得无谓重建几何——每帧重建会吃掉滚动帧预算）。
 */
import { describe, expect, it } from "vitest";

import { gridGeometrySignature, keyboardGeometrySignature, resolveLiveGridView } from "./gridView";

/** 实测基线（1920×1200 @dpr2，pitch 参数）。 */
const PITCH = { absMin: 36, absMax: 96, span: 24 };

describe("resolveLiveGridView（视口竖向真值来自内核）", () => {
    it("center 由内核 scrollTop 反算，而非沿用快照值", () => {
        // 内核 scrollTop = 533.333 时实测 center = 72（基线）。
        const rest = resolveLiveGridView({ ...PITCH, scrollTop: 533.3333333333333 });
        expect(rest.center).toBeCloseTo(72, 4);
        expect(rest.span).toBe(24);

        // 同一份 span/边界，只把内核位置往上滚 → center 必须随之变大。
        const moved = resolveLiveGridView({ ...PITCH, scrollTop: 429.65 });
        expect(moved.center).toBeGreaterThan(rest.center);
    });

    it("span 原样透传（缩放由面板决定，内核不参与）", () => {
        expect(resolveLiveGridView({ ...PITCH, span: 7.5, scrollTop: 800 }).span).toBe(7.5);
    });

    it("两端越界被钳制到可动中心域，不产生 NaN", () => {
        const top = resolveLiveGridView({ ...PITCH, scrollTop: -1e6 });
        const bottom = resolveLiveGridView({ ...PITCH, scrollTop: 1e6 });
        for (const view of [top, bottom]) {
            expect(Number.isFinite(view.center)).toBe(true);
            expect(view.center).toBeGreaterThanOrEqual(PITCH.absMin + PITCH.span / 2);
            expect(view.center).toBeLessThanOrEqual(PITCH.absMax - PITCH.span / 2);
        }
        // 方向：scrollTop 越大 → center 越小（与 verticalScrollMapping 同向）。
        expect(top.center).toBeGreaterThan(bottom.center);
    });
});

describe("gridGeometrySignature（网格几何内容签名）", () => {
    const metrics = {
        kind: "pitch" as const,
        ...PITCH,
        viewportWidthPx: 1864,
        viewportHeightPx: 823,
        dpr: 2,
        strongRgba: [1, 1, 1, 1] as const,
        weakRgba: [0, 0, 0, 1] as const,
    };
    const sigAt = (scrollTop: number) =>
        gridGeometrySignature({
            ...metrics,
            view: resolveLiveGridView({ ...PITCH, scrollTop }),
        });

    it("★ 快照 span/边界不变、只有内核位置变化时，签名必须变化", () => {
        // 这条就是「网格冻住」的回归：旧签名含的是面板 render 期写下的
        // spec.view.center，竖向滚动不进 React → 快照不变 → 签名不变 →
        // 几何不重建 → 屏幕上的横线一动不动。
        expect(sigAt(429.65)).not.toBe(sigAt(533.3333333333333));
    });

    it("无变化时签名稳定（不得无谓重建几何）", () => {
        expect(sigAt(600)).toBe(sigAt(600));
    });

    it("视口尺寸与 dpr 参与签名（几何按物理像素取向）", () => {
        const base = sigAt(600);
        const taller = gridGeometrySignature({
            ...metrics,
            viewportHeightPx: 900,
            view: resolveLiveGridView({ ...PITCH, scrollTop: 600 }),
        });
        expect(taller).not.toBe(base);
    });

    it("颜色参与签名（主题切换必须重建几何）", () => {
        const base = sigAt(600);
        const other = gridGeometrySignature({
            ...metrics,
            strongRgba: [0.5, 0.5, 0.5, 1],
            view: resolveLiveGridView({ ...PITCH, scrollTop: 600 }),
        });
        expect(other).not.toBe(base);
    });
});

describe("keyboardGeometrySignature（键盘 / 数值轴几何签名）", () => {
    const metrics = {
        kind: "pitch" as const,
        ...PITCH,
        viewportHeightPx: 823,
        axisWidthPx: 56,
        dpr: 2,
        strongRgba: [1, 1, 1, 1] as const,
        weakRgba: [0, 0, 0, 1] as const,
    };

    it("★ 内核位置变化时签名变化（键盘同样不得冻住）", () => {
        const a = keyboardGeometrySignature({
            ...metrics,
            view: resolveLiveGridView({ ...PITCH, scrollTop: 429.65 }),
        });
        const b = keyboardGeometrySignature({
            ...metrics,
            view: resolveLiveGridView({ ...PITCH, scrollTop: 533.3333333333333 }),
        });
        expect(a).not.toBe(b);
    });

    it("非 pitch 参数没有键盘 → 空签名（GL 据此清空几何）", () => {
        expect(
            keyboardGeometrySignature({
                ...metrics,
                kind: "cents",
                view: resolveLiveGridView({ ...PITCH, scrollTop: 600 }),
            }),
        ).toBe("");
    });

    it("键盘配色参与签名（切主题后键体颜色必须更新）", () => {
        const base = keyboardGeometrySignature({
            ...metrics,
            view: resolveLiveGridView({ ...PITCH, scrollTop: 600 }),
        });
        const themed = keyboardGeometrySignature({
            ...metrics,
            whiteKeyRgba: [0.9, 0.9, 0.9, 1],
            view: resolveLiveGridView({ ...PITCH, scrollTop: 600 }),
        });
        expect(themed).not.toBe(base);
    });
});
