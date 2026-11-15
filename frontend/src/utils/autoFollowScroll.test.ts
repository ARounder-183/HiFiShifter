/**
 * 自动滚屏的几何与模式判定。
 *
 * 【要钉死什么】两种模式的语义差、翻页式的边界（含"一次到位"与工程右端钳制）、
 * 以及两者共用同一条变化阈值。这里**不需要 jsdom**：本模块是纯算术，
 * 判定逻辑一旦正确，两个桥接组件剩下的只是"把数字写进载体"一行调用。
 */
import { describe, expect, it } from "vitest";

import {
    AUTO_SCROLL_EPSILON_PX,
    computeAutoFollowScrollLeft,
    computePageScrollLeft,
    normalizeAutoScrollMode,
    resolveAutoScrollTarget,
} from "./autoFollowScroll";

describe("normalizeAutoScrollMode", () => {
    it("接受两个合法模式", () => {
        expect(normalizeAutoScrollMode("scroll")).toBe("scroll");
        expect(normalizeAutoScrollMode("page")).toBe("page");
    });

    it("其余一切（undefined / null / 数字 / 拼错）回落到滚动式", () => {
        for (const value of [undefined, null, "", "pagee", "PAGE", 0, 1, {}, []]) {
            expect(normalizeAutoScrollMode(value)).toBe("scroll");
        }
    });
});

describe("computePageScrollLeft", () => {
    const base = { pxPerSec: 100, viewportWidth: 1000, contentWidth: 60_000 };
    const at = (playheadSec: number, scrollLeft: number) =>
        computePageScrollLeft({ ...base, playheadSec, scrollLeft });

    it("光标在页内 → null（静止）", () => {
        expect(at(5, 0)).toBeNull();
        expect(at(9.999, 0)).toBeNull();
    });

    it("恰好在右缘（不含）→ 翻一页，光标落在页首", () => {
        expect(at(10, 0)).toBe(1000);
        expect(at(10.005, 0)).toBe(1000);
    });

    it("一次到位：seek 到远处只翻一次，不是逐帧级联", () => {
        expect(at(35, 0)).toBe(3000);
    });

    it("以当前视口为页基准：手动滚到 250 后整页位移", () => {
        expect(at(12.51, 250)).toBe(1250);
    });

    it("光标在视口左侧之外 → 目标 < 当前值且光标落回页内", () => {
        expect(at(5, 2000)).toBe(0);
        expect(at(19.99, 2000)).toBe(1000);
    });

    it("工程右端钳制到 工程宽 − 视口宽", () => {
        expect(
            computePageScrollLeft({ ...base, contentWidth: 2500, playheadSec: 25, scrollLeft: 0 }),
        ).toBe(1500);
    });

    it("工程比视口还短 → 钳回 0（页宽仍按视口算）", () => {
        // 光标 x = 1200 已越出 [0, 1000)；工程只有 600 宽，所以上限 maxScrollLeft = 0。
        expect(
            computePageScrollLeft({ ...base, contentWidth: 600, playheadSec: 12, scrollLeft: 0 }),
        ).toBe(0);
        // 光标仍在页内时一律不动（哪怕工程比视口短）。
        expect(
            computePageScrollLeft({ ...base, contentWidth: 600, playheadSec: 6, scrollLeft: 0 }),
        ).toBeNull();
    });

    it("视口宽为 0 / 非有限输入 → null", () => {
        expect(
            computePageScrollLeft({ ...base, viewportWidth: 0, playheadSec: 99, scrollLeft: 0 }),
        ).toBeNull();
        expect(
            computePageScrollLeft({ ...base, playheadSec: Number.NaN, scrollLeft: 0 }),
        ).toBeNull();
        expect(
            computePageScrollLeft({ ...base, playheadSec: 99, scrollLeft: Number.NaN }),
        ).toBeNull();
    });

    it("幂等：把返回值当作新的 scrollLeft 再算 → null", () => {
        for (const [sec, left] of [
            [12.51, 250],
            [35, 0],
            [19.99, 2000],
        ] as const) {
            const next = at(sec, left);
            expect(next).not.toBeNull();
            expect(at(sec, next!)).toBeNull();
        }
    });
});

describe("resolveAutoScrollTarget", () => {
    const base = {
        playheadSec: 12.5,
        pxPerSec: 100,
        viewportWidth: 1000,
        scrollLeft: 0,
        contentWidth: 60_000,
    };

    it("滚动式与 computeAutoFollowScrollLeft 同值", () => {
        expect(resolveAutoScrollTarget({ ...base, mode: "scroll" })).toBe(
            computeAutoFollowScrollLeft(base),
        );
    });

    it("滚动式在容差内返回 null", () => {
        const next = computeAutoFollowScrollLeft(base);
        expect(
            resolveAutoScrollTarget({
                ...base,
                mode: "scroll",
                scrollLeft: next + AUTO_SCROLL_EPSILON_PX / 2,
            }),
        ).toBeNull();
    });

    it("翻页式委托给 computePageScrollLeft", () => {
        // 光标 x = 12.5 × 100 = 1250，视口 [0, 1000) → 翻到页首 1000。
        expect(resolveAutoScrollTarget({ ...base, mode: "page" })).toBe(1000);
        // 光标在页内（[500, 1500)）→ 不滚。
        expect(resolveAutoScrollTarget({ ...base, mode: "page", scrollLeft: 500 })).toBeNull();
    });

    it("翻页式在工程右端被钳住之后不再重复写入", () => {
        // 光标停在页外，但钳制目标与当前值相同 —— 这正是"钳制后不抖动"的那条守卫。
        const atEnd = { ...base, mode: "page" as const, contentWidth: 2500, scrollLeft: 1500 };
        expect(computePageScrollLeft({ ...atEnd, playheadSec: 25 })).toBe(1500);
        expect(resolveAutoScrollTarget({ ...atEnd, playheadSec: 25 })).toBeNull();
    });

    it("★ 频率对比：同样 1 秒 60 帧，翻页式只写 1 次，滚动式写满 60 次", () => {
        const sweep = (mode: "scroll" | "page") => {
            let scrollLeft = 0;
            let writes = 0;
            for (let frame = 0; frame <= 60; frame += 1) {
                const target = resolveAutoScrollTarget({
                    ...base,
                    mode,
                    playheadSec: (frame / 60) * 12, // 12 秒 × 100px/s = 1200px，跨过一页
                    scrollLeft,
                });
                if (target !== null) {
                    writes += 1;
                    scrollLeft = target;
                }
            }
            return writes;
        };
        // 这一条就是"更新频率远低于滚动式"的量化形式：不是判定频率，是**写入**频率。
        expect(sweep("page")).toBe(1);
        expect(sweep("scroll")).toBe(60);
    });
});
