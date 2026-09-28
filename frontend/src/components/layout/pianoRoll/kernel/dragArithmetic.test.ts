/**
 * 参数编辑器内核 · 拖拽算术（纯函数）行为自检。
 *
 * 【本测试守护什么】
 * 1. **选区帧区间 → 采样下标的换算只有一份实现**。这段算术此前在
 *    `usePianoRollInteractions` 里被抄了 **3 遍**（选区拉伸的 `buildDense`、
 *    起点快照、morph 叠加层构造），任一处改动都可能让三条路径对"选区覆盖哪些采样点"
 *    产生分歧——而它们必须完全一致，否则拉伸预览与最终提交会错位。
 * 2. **末帧取 `frameCount - 1`**：选区是半开区间 `[startFrame, start + count)`，
 *    而采样下标是闭区间，右端必须退一帧。不退会把选区外的一帧拉进窗口。
 * 3. **下标必须 clamp 到 `[0, len-1]` 且保证 `startIdx <= endIdx`**。曲线数据的
 *    覆盖窗口由后端按请求给出，选区可能超出它；不 clamp 会 slice 出空数组，
 *    让整个拉伸/形变功能静默失效（不报错，只是没反应）。
 * 4. **`stride` 非法时的安全侧**：`stride = 0` 会让下标除零得 Infinity → clamp 后
 *    落到 len-1，看似"无害"但语义完全错误；显式归一为 1。
 *
 * 【已删除的用例】改造前这里还测 `selectionFrameRange` / `beatToFrameDelta` /
 * `frameDeltaToBeat`（拍 ↔ 帧的双向换算）。选区改用帧制后这三个函数整体消失：
 * 选区给过来的就是帧，拖拽位移也直接是帧位移，换算层没有存在理由。它们的覆盖
 * （负起点夹取、退化段）移到了 `paramSelection.test.ts` 的 `selectionToFrameRanges`。
 */
import { describe, expect, it } from "vitest";

import { edgeAutoScrollDeltaPx, frameToIndex, selectionIndexRange } from "./dragArithmetic";

/**
 * 断言非空并收窄类型。
 *
 * 【为什么需要】`expect(x).not.toBeNull()` 是运行期断言，TypeScript 不会据此收窄
 * 类型；后续访问 `x.startIdx` 仍报 "possibly null"。用一个显式抛错的辅助函数
 * 才能在保留断言语义的同时通过类型检查。
 */
function assertNotNull<T>(value: T | null): T {
    if (value === null) throw new Error("expected non-null");
    return value;
}

/** 造一个采样段：起点帧、步长、采样数。 */
function makeParamView(startFrame: number, stride: number, len: number) {
    return { startFrame, stride, edit: new Array(len).fill(0) };
}

describe("selectionIndexRange", () => {
    it("末帧取 startFrame + frameCount - 1（半开区间的右端退一帧）", () => {
        const pv = makeParamView(0, 1, 1000);
        const range = selectionIndexRange({ startFrame: 150, frameCount: 170, paramView: pv });
        expect(range?.startIdx).toBe(150);
        expect(range?.endIdx).toBe(319);
    });

    it("零长选区（单击留下的那一段）只覆盖起点那一帧", () => {
        const pv = makeParamView(0, 1, 1000);
        const range = selectionIndexRange({ startFrame: 42, frameCount: 0, paramView: pv });
        expect(range?.startIdx).toBe(42);
        expect(range?.endIdx).toBe(42);
    });

    it("startFrame 非 0 时下标相对起点折算，并 clamp 到数据窗口内", () => {
        // 数据从帧 100 开始，共 50 个采样。选区落在窗口左外侧。
        const pv = makeParamView(100, 1, 50);
        const range = selectionIndexRange({ startFrame: 0, frameCount: 20, paramView: pv });
        // 帧 0..19 → 相对下标 -100..-81 → clamp 到 0。
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(0);
    });

    it("选区超出数据窗口右缘时下标 clamp 到末位（不产生空切片）", () => {
        const pv = makeParamView(0, 1, 10);
        const range = selectionIndexRange({ startFrame: 0, frameCount: 10_000, paramView: pv });
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(9);
    });

    it("stride > 1 时下标按 stride 折算（四舍五入）", () => {
        const pv = makeParamView(0, 4, 100);
        // 帧 0..20（21 帧），stride 4 → 下标 0..5。
        const range = selectionIndexRange({ startFrame: 0, frameCount: 21, paramView: pv });
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(5);
    });

    it("数据为空 / 非法输入返回 null（不抛异常、不返回空切片）", () => {
        const empty = { startFrame: 0, stride: 1, edit: [] as number[] };
        expect(selectionIndexRange({ startFrame: 0, frameCount: 5, paramView: empty })).toBeNull();
        expect(
            selectionIndexRange({
                startFrame: Number.NaN,
                frameCount: 5,
                paramView: makeParamView(0, 1, 10),
            }),
        ).toBeNull();
        expect(
            selectionIndexRange({
                startFrame: 0,
                frameCount: Number.NaN,
                paramView: makeParamView(0, 1, 10),
            }),
        ).toBeNull();
    });

    it("stride = 0 归一为 1（不产生除零）", () => {
        const pv = makeParamView(0, 0, 100);
        const range = assertNotNull(
            selectionIndexRange({ startFrame: 0, frameCount: 21, paramView: pv }),
        );
        expect(Number.isFinite(range.startIdx)).toBe(true);
        expect(Number.isFinite(range.endIdx)).toBe(true);
        // 归一为 1 时下标就等于帧号本身（0..20）。
        expect(range.startIdx).toBe(0);
        expect(range.endIdx).toBe(20);
    });
});

describe("frameToIndex", () => {
    it("相对起点折算并四舍五入（与绘制路径的直线连接一致）", () => {
        expect(frameToIndex({ frame: 110, startFrame: 100, stride: 1 })).toBe(10);
        // 帧 105、stride 10 → 0.5 → 四舍五入为 1（不是 0）
        expect(frameToIndex({ frame: 105, startFrame: 100, stride: 10 })).toBe(1);
        expect(frameToIndex({ frame: 104, startFrame: 100, stride: 10 })).toBe(0);
    });

    it("stride 非法归一为 1；结果非有限时返回 null", () => {
        expect(frameToIndex({ frame: 10, startFrame: 0, stride: 0 })).toBe(10);
        expect(frameToIndex({ frame: Number.NaN, startFrame: 0, stride: 1 })).toBeNull();
        expect(
            frameToIndex({ frame: 10, startFrame: 0, stride: Number.POSITIVE_INFINITY }),
        ).toBeNull();
    });
});

/**
 * 边缘自动滚动（框选拖到画布边缘时自动平移视图）。
 *
 * 【为什么抽出来】这段算术把"指针离边缘多远"映射为"每帧滚动多少像素"，
 * 内含三处魔法数（边缘带宽 32px、单帧最大步长 18px、距离比例上限 1.5）与一次
 * `scrollLeft` 的二次钳制。它是纯算术，但此前埋在 pointermove 闭包里，
 * 无法单测，改动风险只能靠手拖复现。
 */
describe("edgeAutoScrollDeltaPx", () => {
    const view = { leftPx: 100, rightPx: 900 };

    it("远离边缘 → 0", () => {
        expect(edgeAutoScrollDeltaPx({ clientX: 500, ...view })).toBe(0);
        expect(edgeAutoScrollDeltaPx({ clientX: 132, ...view })).toBe(0);
        expect(edgeAutoScrollDeltaPx({ clientX: 868, ...view })).toBe(0);
    });

    it("靠近左缘 → 负向滚动，越靠越快；到边缘时比例恰为 1", () => {
        const half = edgeAutoScrollDeltaPx({ clientX: 116, ...view }); // 带内一半
        const atEdge = edgeAutoScrollDeltaPx({ clientX: 100, ...view }); // 恰在边界
        expect(half).toBeCloseTo(-9, 9); // 0.5 × 18
        expect(atEdge).toBeCloseTo(-18, 9); // 1.0 × 18，不是 27
        expect(atEdge).toBeLessThan(half);
    });

    it("靠近右缘 → 正向滚动，对称于左缘", () => {
        expect(edgeAutoScrollDeltaPx({ clientX: 884, ...view })).toBeCloseTo(
            -edgeAutoScrollDeltaPx({ clientX: 116, ...view }),
            9,
        );
        expect(edgeAutoScrollDeltaPx({ clientX: 900, ...view })).toBeCloseTo(18, 9);
    });

    it("超出视口（指针已被 capture 到外面）仍取满速，不越界", () => {
        expect(edgeAutoScrollDeltaPx({ clientX: 0, ...view })).toBeCloseTo(-27, 9);
        expect(edgeAutoScrollDeltaPx({ clientX: 2000, ...view })).toBeCloseTo(27, 9);
    });

    it("非有限输入返回 0", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(edgeAutoScrollDeltaPx({ clientX: bad, ...view })).toBe(0);
            expect(edgeAutoScrollDeltaPx({ clientX: 100, leftPx: bad, rightPx: 900 })).toBe(0);
        }
    });
});
