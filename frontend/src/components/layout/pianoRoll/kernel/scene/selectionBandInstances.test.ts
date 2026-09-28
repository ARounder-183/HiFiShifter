/**
 * 选区块实例构建（`scene/selectionBandInstances`）行为自检。
 *
 * 【钉住的不变量】
 * 1. **整高**：填充矩形必须覆盖整个视口高度。选区块是"时间上的区间"，纵向没有
 *    范围概念；若高度算错（例如用了内容高或半高），选区内的一部分曲线就不会有
 *    底色——而那看起来像"选区只覆盖了一半"，不是崩溃，很难归因。
 * 2. **每条边各一个 1 CSS px 的实心矩形**：这里是逐条复刻 Canvas2D 的
 *    `strokeRect(x0 + 0.5, 0.5, w - 1, h - 1)`（路径展开后四条边恰好各占 1px），
 *    而不是画一个描边——GL 的平面实例没有"描边"概念。
 * 3. **不可见的段整体跳过**：视口两侧之外的段不产生几何（长工程的大选区在最小
 *    缩放下可能有上万像素宽，白白上传没有意义）。
 * 4. **非有限 / 零宽段防御**：不产生几何，也不产出 NaN（NaN 会让整层实例失效）。
 */
import { test } from "vitest";

import { createTimelineAxis } from "../../../renderKernel/timelineAxis";
import { buildSelectionBandInstances } from "./selectionBandInstances.js";

test("components/layout/pianoRoll/kernel/scene/selectionBandInstances.test.ts scripted checks", () => {
    function assert(condition: boolean, message: string): void {
        if (!condition) throw new Error(message);
    }

    function assertClose(actual: number, expected: number, tol: number, message: string): void {
        if (!(Math.abs(actual - expected) <= tol)) {
            throw new Error(`${message}: expected ${expected} (±${tol}), received ${actual}`);
        }
    }

    const fill = [0.4, 0.8, 1, 0.08] as const;
    const border = [0.4, 0.8, 1, 0.3] as const;

    // 100 px/秒、无滚动：秒 = 视口 x / 100。
    const axis = createTimelineAxis({ pxPerSec: 100, scrollLeftPx: 0, viewportWidthPx: 800 });

    // ── 单段：填充 + 四条边 = 5 个实例 ────────────────────────────────
    const one = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [{ startSec: 1, endSec: 3 }],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(one.length === 5, `单段应产出 5 个实例，实得 ${one.length}`);

    const [fillInst] = one;
    assertClose(fillInst.x, 100, 1e-9, "填充左缘");
    assertClose(fillInst.w, 200, 1e-9, "填充宽度");
    assertClose(fillInst.y, 0, 1e-9, "填充上缘");
    assertClose(fillInst.h, 400, 1e-9, "填充必须整高");
    assert(fillInst.rgba === fill, "填充色必须原样传递（引用即可）");

    // 四条边：上下横条各 1px 高，左右竖条各 1px 宽。
    const borders = one.slice(1);
    assert(borders.length === 4, "每条边一个实例");
    for (const b of borders) {
        assert(b.rgba === border, "边框色必须原样传递");
    }
    // 上边
    assertClose(borders[0].y, 0, 1e-9, "上边框 y");
    assertClose(borders[0].h, 1, 1e-9, "上边框厚度");
    assertClose(borders[0].w, 200, 1e-9, "上边框宽度");
    // 下边
    assertClose(borders[1].y, 399, 1e-9, "下边框 y");
    assertClose(borders[1].h, 1, 1e-9, "下边框厚度");
    // 左边
    assertClose(borders[2].x, 100, 1e-9, "左边框 x");
    assertClose(borders[2].w, 1, 1e-9, "左边框厚度");
    assertClose(borders[2].h, 400, 1e-9, "左边框必须整高");
    // 右边
    assertClose(borders[3].x, 299, 1e-9, "右边框 x");
    assertClose(borders[3].w, 1, 1e-9, "右边框厚度");

    // ── 多段：逐段产出（断层处不画）────────────────────────────────────
    const multi = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [
            { startSec: 1, endSec: 2 },
            { startSec: 4, endSec: 5 },
        ],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(multi.length === 10, `两段应产出 10 个实例，实得 ${multi.length}`);
    // 断层处（200..400）不得有任何填充
    for (const inst of multi) {
        if (inst.rgba !== fill) continue;
        assert(!(inst.x >= 200 && inst.x < 400), `断层处出现了填充实例（x=${inst.x}）`);
    }

    // ── 起止颠倒自动归一 ─────────────────────────────────────────────
    const reversed = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [{ startSec: 3, endSec: 1 }],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(reversed.length === 5, "起止颠倒的段应当照常产出");
    assertClose(reversed[0].x, 100, 1e-9, "归一化后的填充左缘");

    // ── 视口外的段整体跳过 ───────────────────────────────────────────
    const offscreen = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [
            { startSec: 100, endSec: 200 }, // 全在右侧之外
            { startSec: -50, endSec: -10 }, // 全在左侧之外
        ],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(offscreen.length === 0, `视口外的段不应产出几何，实得 ${offscreen.length}`);

    // 跨视口边界的段仍要画（只有完全在外的才跳过）
    const partial = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [{ startSec: -5, endSec: 2 }],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(partial.length === 5, "部分可见的段必须绘制");

    // ── 非法输入防御：不产出几何、不含 NaN ──────────────────────────
    // 注意：起止颠倒（如 {6,5}）**不是**非法输入——它自动归一为 [5,6] 的合法区间，
    // 已在上面单独验证。这里只放真正应当被丢弃的段。
    const bad = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 400,
        spansSec: [
            { startSec: Number.NaN, endSec: 2 },
            { startSec: 1, endSec: Number.POSITIVE_INFINITY },
            { startSec: 5, endSec: 5 },
            { startSec: Number.NEGATIVE_INFINITY, endSec: Number.POSITIVE_INFINITY },
        ],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(bad.length === 0, `非法 / 零宽段不应产出几何，实得 ${bad.length}`);

    const zeroHeight = buildSelectionBandInstances({
        axis,
        viewportHeightPx: 0,
        spansSec: [{ startSec: 1, endSec: 3 }],
        fillRgba: fill,
        borderRgba: border,
    });
    assert(zeroHeight.length === 0, "视口高为 0 时不产出几何");

    for (const inst of one) {
        for (const v of [inst.x, inst.y, inst.w, inst.h]) {
            assert(Number.isFinite(v), `实例含非有限值：${v}`);
        }
    }
});
