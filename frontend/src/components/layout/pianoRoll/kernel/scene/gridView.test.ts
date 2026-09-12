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

import {
    gridGeometrySignature,
    keyboardGeometrySignature,
    resolveLiveGridView,
    resolveLiveSpan,
} from "./gridView";

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

    it("★ span 变化必须改变签名（竖向缩放回归：否则只画旧值域、留空白带）", () => {
        // 【为什么单列一条】`center` 的同类不变量已由上一条覆盖，但 `span` 走的是
        // **另一条**数据路径：竖向缩放由面板改 `pitchViewRef.current.span` 并
        // `invalidate()`，**不触发 React 渲染**（见 `setPitchView`）。因此
        // `spec.view.span` 这个 render 期快照会停在旧值上，而几何按实时 span 枚举
        // 半音位置 → 只覆盖旧窗口，屏幕留下空白带。
        //
        // 实测（dpr 2，钢琴键区 Alt+滚轮连滚 3 次）：实时 span 24 → 42.5，而
        // `gridSpec.view.span` 与 `gridInstanceCount` 都**没变**（24 / 26）；
        // 旧实现同手势最大空白带 398px。
        //
        // 判据：span 变了，签名就必须变（否则几何不重建）。
        const wider = gridGeometrySignature({
            ...metrics,
            view: resolveLiveGridView({ ...PITCH, span: 42.5, scrollTop: 600 }),
        });
        expect(wider).not.toBe(sigAt(600));
    });

    it("★ resolveLiveGridView 必须原样透传传入的 span（调用方给实时值才有效）", () => {
        // 这条钉住"实时"的定义：解析出的 span 恒等于入参。宿主若误传快照值，
        // 上面那条签名不变量虽然成立，实际喂进去的仍是旧值。
        const stale = resolveLiveGridView({ ...PITCH, span: 24, scrollTop: 600 });
        const live = resolveLiveGridView({ ...PITCH, span: 42.5, scrollTop: 600 });
        expect(live.span).toBe(42.5);
        expect(stale.span).toBe(24);
        expect(live).not.toEqual(stale);
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

/**
 * 「实时跨度」取值判定。
 *
 * 【为什么必须单独测这一层】竖向缩放的缺陷**不在**纯函数的算术里，而在宿主的
 * **接线**：`liveGridView` 曾把 `span: spec.view.span`（render 期快照）喂进来。
 * 只测 `resolveLiveGridView` 的透传性质**测不出这个 bug**——实测把宿主改回快照
 * 取值后，本文件其余 12 项断言全部照旧通过。
 *
 * 因此把"取哪个 span"抽成 `resolveLiveSpan` 并在此钉住优先级，宿主的接线也就
 * 有了可判别的守护。
 */
describe("resolveLiveSpan（实时跨度优先级）", () => {
    it("★ 镜像可用时取镜像值（实时），忽略快照", () => {
        // 实测场景：竖向缩放后实时 span 24 → 42.5，而快照仍是 24。
        // 取快照 → 几何只按 24 枚举半音 → 屏幕留空白带。
        expect(resolveLiveSpan({ domainSpan: 42.5, snapshotSpan: 24 })).toBe(42.5);
    });

    it("★ 镜像不可用（NaN）时退回快照值，而不是把 NaN 传下去", () => {
        // NaN 会让整层实例属性失效、几何消失——比"少画几行"严重得多。
        expect(resolveLiveSpan({ domainSpan: Number.NaN, snapshotSpan: 24 })).toBe(24);
    });

    it("★ 镜像非正时同样退回快照（0 / 负数不是合法跨度）", () => {
        expect(resolveLiveSpan({ domainSpan: 0, snapshotSpan: 24 })).toBe(24);
        expect(resolveLiveSpan({ domainSpan: -5, snapshotSpan: 24 })).toBe(24);
    });

    it("两者都不可用时原样返回镜像值（由调用方处理，不在这里造数）", () => {
        // 刻意不返回某个"默认跨度"：那会掩盖上游的值域错误，让问题更难归因。
        expect(resolveLiveSpan({ domainSpan: Number.NaN, snapshotSpan: Number.NaN })).toBeNaN();
        expect(resolveLiveSpan({ domainSpan: 0, snapshotSpan: 0 })).toBe(0);
    });

    it("镜像与快照一致时结果一致（常见情形不改变行为）", () => {
        expect(resolveLiveSpan({ domainSpan: 24, snapshotSpan: 24 })).toBe(24);
    });
});
