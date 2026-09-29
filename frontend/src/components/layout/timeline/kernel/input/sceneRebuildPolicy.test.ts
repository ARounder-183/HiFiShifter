/**
 * 场景（GPU 几何）重建判定（./sceneRebuildPolicy）行为自检。
 *
 * 【主要内容】
 * 1. 行高变化（**内核值**）必须重建 —— 行高改变所有行的内容坐标；
 * 2. **React 镜像行高变化不得单独触发重建** —— 判据必须与几何构建所用行高同源，
 *    否则"内核已换行高、几何还是旧行高"的那一帧逃过重建，画布按旧行高绘制一帧
 *    （竖直缩放抽动的直接机制）；
 * 3. 余量内的纯滚动不重建（保留"滚动零重建"的既有语义）；
 * 4. 超出余量必须重建（横向绝对像素、纵向按行数）；
 * 5. 标脏 / 缩放 / 主题 / 内容引用变化各自都能单独触发重建。
 *
 * 【作用】把"竖直缩放抽动"的根因钉成回归用例：任何把行高判据改回 React 镜像的改动
 * 都会让第 2 条变红。
 *
 * 【与其他模块的关系】覆盖 `sceneRebuildPolicy.ts`；不依赖 DOM / React / GL。
 */

import { describe, expect, it } from "vitest";

import { shouldRebuildScene, type SceneRebuildInputs } from "./sceneRebuildPolicy";

/** 基准入参：一切"未变化"，行高 96、位置 0、无余量漂移。 */
function base(overrides: Partial<SceneRebuildInputs> = {}): SceneRebuildInputs {
    return {
        sceneDirty: false,
        builtPxPerSec: 100,
        viewPxPerSec: 100,
        builtDarkMode: false,
        darkMode: false,
        contentChanged: false,
        builtScrollLeftPx: 0,
        viewScrollLeft: 0,
        builtScrollTopPx: 0,
        viewScrollTop: 0,
        viewRowHeight: 96,
        builtRowHeight: 96,
        horizontalMarginPx: 512,
        verticalOverscanRows: 2,
        ...overrides,
    };
}

describe("shouldRebuildScene", () => {
    it("一切未变化时不重建", () => {
        expect(shouldRebuildScene(base())).toBe(false);
    });

    it("内核行高变化必须重建", () => {
        // 几何是用内核行高构建的：88 → 96 时所有行的内容坐标都变。
        expect(shouldRebuildScene(base({ viewRowHeight: 96, builtRowHeight: 88 }))).toBe(true);
    });

    it("行高判据只认内核值：镜像行高变化不单独触发重建", () => {
        // 这一帧的真实处境：内核行高已换成 96，而几何仍是 88 建出来的。
        // 判据必须看内核值 ⇒ 必须重建。若有人把判据改回 React 镜像（两者相等），
        // 本用例会变红 —— 那正是竖直缩放抽动的复现条件。
        const kernelLeads = base({ viewRowHeight: 96, builtRowHeight: 88 });
        expect(shouldRebuildScene(kernelLeads)).toBe(true);

        // 反向：镜像滞后（96）而内核与几何都还是 88 —— 行高没有真变化，不重建。
        const mirrorLeads = base({ viewRowHeight: 88, builtRowHeight: 88 });
        expect(shouldRebuildScene(mirrorLeads)).toBe(false);
    });

    it("余量内的纯滚动不重建（滚动零重建语义）", () => {
        expect(
            shouldRebuildScene(
                base({
                    builtScrollLeftPx: 0,
                    viewScrollLeft: 512,
                    builtScrollTopPx: 0,
                    viewScrollTop: 96 * 2,
                }),
            ),
        ).toBe(false);
    });

    it("横向超出绝对像素余量必须重建", () => {
        expect(shouldRebuildScene(base({ builtScrollLeftPx: 0, viewScrollLeft: 512.5 }))).toBe(
            true,
        );
    });

    it("纵向超出按行数换算的余量必须重建", () => {
        // 余量 = 2 行 × 96px = 192px；193px 必须重建。
        expect(shouldRebuildScene(base({ builtScrollTopPx: 0, viewScrollTop: 193 }))).toBe(true);
        // 行高变小后同样的位移不再越界（余量随行高换算）：80px × 2 行 = 160px。
        expect(
            shouldRebuildScene(
                base({
                    viewRowHeight: 80,
                    builtRowHeight: 80,
                    builtScrollTopPx: 0,
                    viewScrollTop: 160,
                }),
            ),
        ).toBe(false);
    });

    it("标脏 / 缩放 / 主题 / 内容引用各自都能单独触发重建", () => {
        expect(shouldRebuildScene(base({ sceneDirty: true }))).toBe(true);
        expect(shouldRebuildScene(base({ viewPxPerSec: 120 }))).toBe(true);
        expect(shouldRebuildScene(base({ darkMode: true }))).toBe(true);
        expect(shouldRebuildScene(base({ contentChanged: true }))).toBe(true);
    });

    it("首次调用（哨兵初值）必须重建", () => {
        // 宿主的初值是 NaN / -1 哨兵：与任何真实值都不等，因此首帧必然重建。
        expect(
            shouldRebuildScene(
                base({
                    builtPxPerSec: Number.NaN,
                    builtScrollLeftPx: Number.NaN,
                    builtScrollTopPx: Number.NaN,
                    builtRowHeight: -1,
                }),
            ),
        ).toBe(true);
    });
});
