/**
 * 参数编辑器内核 · 拖拽算术（纯函数）行为自检。
 *
 * 【本测试守护什么】
 * 1. **选区 beat → 帧 → 采样下标的换算只有一份实现**。这段算术此前在
 *    `usePianoRollInteractions` 里被抄了 **3 遍**（选区拉伸的 `buildDense`、
 *    起点快照、morph 叠加层构造），任一处改动都可能让三条路径对"选区覆盖哪些采样点"
 *    产生分歧——而它们必须完全一致，否则拉伸预览与最终提交会错位。
 * 2. **两端取整方向不同是刻意的**：起点 `floor`、终点 `ceil`。都取 floor 会漏掉
 *    右端不足一帧的部分，都取 ceil 会把左端拉进选区外的一帧。这是"选区覆盖"的
 *    语义要求，不是随手写的。
 * 3. **下标必须 clamp 到 `[0, len-1]` 且保证 `startIdx <= endIdx`**。曲线数据的
 *    覆盖窗口由后端按请求给出，选区可能超出它；不 clamp 会 slice 出空数组，
 *    让整个拉伸/形变功能静默失效（不报错，只是没反应）。
 * 4. **`stride` 与 `framePeriodMs` 非法时的安全侧**：`stride = 0` 会让下标除零得
 *    Infinity → clamp 后落到 len-1，看似"无害"但语义完全错误；显式归一为 1。
 */
import { describe, expect, it } from "vitest";

import {
    beatToFrameDelta,
    edgeAutoScrollDeltaPx,
    frameDeltaToBeat,
    frameToIndex,
    selectionFrameRange,
    selectionIndexRange,
} from "./dragArithmetic";

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
    it("beat 起点取 floor、终点取 ceil（两端方向不同是刻意的）", () => {
        // secPerBeat = 0.5、fp = 5ms → 100 帧/beat。
        // 选区 [1.5, 3.2] beat → 起点帧 150、终点帧 320。
        const pv = makeParamView(0, 1, 1000);
        const range = selectionIndexRange({
            aBeat: 1.5,
            bBeat: 3.2,
            secPerBeat: 0.5,
            framePeriodMs: 5,
            paramView: pv,
        });
        expect(range?.startFrame).toBe(150);
        expect(range?.endFrame).toBe(320);
        expect(range?.startIdx).toBe(150);
        expect(range?.endIdx).toBe(320);
    });

    it("起点 floor / 终点 ceil 在非整帧边界上体现（不漏不多）", () => {
        // fp = 7ms（非整除）：选区 beat 让两端落在帧之间。
        // 起点 0.1s → 帧 14.28… → floor 14；终点 0.2s → 帧 28.57… → ceil 29。
        const pv = makeParamView(0, 1, 1000);
        const range = selectionIndexRange({
            aBeat: 0.1,
            bBeat: 0.2,
            secPerBeat: 1,
            framePeriodMs: 7,
            paramView: pv,
        });
        expect(range?.startFrame).toBe(14);
        expect(range?.endFrame).toBe(29);
    });

    it("startFrame 非 0 时下标相对起点折算，并 clamp 到数据窗口内", () => {
        // 数据从帧 100 开始，共 50 个采样。选区落在窗口左外侧。
        const pv = makeParamView(100, 1, 50);
        const range = selectionIndexRange({
            aBeat: 0,
            bBeat: 0.1,
            secPerBeat: 1,
            framePeriodMs: 5,
            paramView: pv,
        });
        // 帧 0..20 → 相对下标 -100..-80 → clamp 到 0。
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(0);
    });

    it("选区超出数据窗口右缘时下标 clamp 到末位（不产生空切片）", () => {
        const pv = makeParamView(0, 1, 10);
        const range = selectionIndexRange({
            aBeat: 0,
            bBeat: 100,
            secPerBeat: 1,
            framePeriodMs: 5,
            paramView: pv,
        });
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(9);
    });

    it("stride > 1 时下标按 stride 折算", () => {
        const pv = makeParamView(0, 4, 100);
        const range = selectionIndexRange({
            aBeat: 0,
            bBeat: 0.1,
            secPerBeat: 1,
            framePeriodMs: 5,
            paramView: pv,
        });
        // 帧 0..20，stride 4 → 下标 0..5。
        expect(range?.startIdx).toBe(0);
        expect(range?.endIdx).toBe(5);
    });

    it("beat 次序颠倒时自动归一（不依赖调用方排序）", () => {
        const pv = makeParamView(0, 1, 1000);
        const swapped = selectionIndexRange({
            aBeat: 3.2,
            bBeat: 1.5,
            secPerBeat: 0.5,
            framePeriodMs: 5,
            paramView: pv,
        });
        expect(swapped?.startFrame).toBe(150);
        expect(swapped?.endFrame).toBe(320);
    });

    it("数据为空 / 非法输入返回 null（不抛异常、不返回空切片）", () => {
        const empty = { startFrame: 0, stride: 1, edit: [] as number[] };
        expect(
            selectionIndexRange({
                aBeat: 0,
                bBeat: 1,
                secPerBeat: 1,
                framePeriodMs: 5,
                paramView: empty,
            }),
        ).toBeNull();
        expect(
            selectionIndexRange({
                aBeat: Number.NaN,
                bBeat: 1,
                secPerBeat: 1,
                framePeriodMs: 5,
                paramView: makeParamView(0, 1, 10),
            }),
        ).toBeNull();
        expect(
            selectionIndexRange({
                aBeat: 0,
                bBeat: 1,
                secPerBeat: 0, // 非法
                framePeriodMs: 5,
                paramView: makeParamView(0, 1, 10),
            }),
        ).toBeNull();
    });

    it("stride = 0 归一为 1（不产生除零）", () => {
        const pv = makeParamView(0, 0, 100);
        const range = assertNotNull(
            selectionIndexRange({
                aBeat: 0,
                bBeat: 0.1,
                secPerBeat: 1,
                framePeriodMs: 5,
                paramView: pv,
            }),
        );
        expect(Number.isFinite(range.startIdx)).toBe(true);
        expect(Number.isFinite(range.endIdx)).toBe(true);
        // 归一为 1 时下标就等于帧号本身（0..20）。
        expect(range.startIdx).toBe(0);
        expect(range.endIdx).toBe(20);
    });

    it("framePeriodMs 为 0 归一为 1e-6（与既有 Math.max(1e-6, fp) 一致）", () => {
        const pv = makeParamView(0, 1, 10);
        const range = assertNotNull(
            selectionIndexRange({
                aBeat: 0,
                bBeat: 1,
                secPerBeat: 1,
                framePeriodMs: 0,
                paramView: pv,
            }),
        );
        // fp → 1e-6 时帧号极大，全部 clamp 到末位。
        expect(range.endIdx).toBe(9);
    });
});

describe("selectionFrameRange", () => {
    it("只算帧范围（起点 floor / 终点 ceil），不做下标折算", () => {
        const range = assertNotNull(
            selectionFrameRange({
                aBeat: 1.5,
                bBeat: 3.2,
                secPerBeat: 0.5,
                framePeriodMs: 5,
            }),
        );
        expect(range.startFrame).toBe(150);
        expect(range.endFrame).toBe(320);
    });

    it("起点为负时 clamp 到 0；终点不小于起点", () => {
        const range = assertNotNull(
            selectionFrameRange({
                aBeat: -5,
                bBeat: -1,
                secPerBeat: 1,
                framePeriodMs: 5,
            }),
        );
        expect(range.startFrame).toBe(0);
        expect(range.endFrame).toBe(0);
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
 * 拖拽增量换算（曲线拖动路径）。
 *
 * 【为什么单独测】曲线拖动用「帧」记录位移（`lastFrameDelta`），而选区位置用
 * 「beat」表示，两者之间**来回换算**：`beat → 帧` 用 `round`，`帧 → beat` 不再取整。
 * 这个往返在 hook 里出现 3 次（拖动预览、拖动提交、形变路径），任一处取整方式不同
 * 都会让"松手后选区跳一下"——偏差只有半格，肉眼难察但用户按多次会累积。
 */
describe("beatToFrameDelta / frameDeltaToBeat", () => {
    it("beatDelta → 帧：乘以秒每拍与每毫秒帧数后四舍五入", () => {
        // secPerBeat = 0.5、fp = 5ms → 100 帧/beat。
        expect(beatToFrameDelta({ beatDelta: 1, secPerBeat: 0.5, framePeriodMs: 5 })).toBe(100);
        // 0.004 beat → 0.4 帧 → round 0
        expect(beatToFrameDelta({ beatDelta: 0.004, secPerBeat: 0.5, framePeriodMs: 5 })).toBe(0);
        // 0.006 beat → 0.6 帧 → round 1
        expect(beatToFrameDelta({ beatDelta: 0.006, secPerBeat: 0.5, framePeriodMs: 5 })).toBe(1);
    });

    it("帧 → beatDelta：不取整（保留亚帧精度，否则往返会丢信息）", () => {
        // 100 帧 → 1 beat
        expect(frameDeltaToBeat({ frameDelta: 100, framePeriodMs: 5, secPerBeat: 0.5 })).toBeCloseTo(
            1,
            12,
        );
        // 1 帧 → 0.01 beat（不取整才留得住）
        expect(frameDeltaToBeat({ frameDelta: 1, framePeriodMs: 5, secPerBeat: 0.5 })).toBeCloseTo(
            0.01,
            12,
        );
    });

    it("往返一致：帧 → beat → 帧 对整数帧是恒等", () => {
        for (const frameDelta of [-250, -1, 0, 1, 37, 250]) {
            const beat = frameDeltaToBeat({
                frameDelta,
                framePeriodMs: 5,
                secPerBeat: 0.5,
            });
            const back = beatToFrameDelta({
                beatDelta: beat,
                secPerBeat: 0.5,
                framePeriodMs: 5,
            });
            expect(back).toBe(frameDelta);
        }
    });

    it("非法输入返回 0（拖拽路径对 0 是安全的空操作）", () => {
        // 与既有内联实现一致：`Math.round(NaN)` 是 NaN，会让选区位置变成 NaN
        // 并一路传播到绘制；显式返回 0 更安全。
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
            expect(
                beatToFrameDelta({ beatDelta: bad, secPerBeat: 0.5, framePeriodMs: 5 }),
            ).toBe(0);
            expect(
                beatToFrameDelta({ beatDelta: 1, secPerBeat: bad, framePeriodMs: 5 }),
            ).toBe(0);
            expect(
                beatToFrameDelta({ beatDelta: 1, secPerBeat: 0.5, framePeriodMs: bad }),
            ).toBe(0);
            expect(
                frameDeltaToBeat({ frameDelta: bad, framePeriodMs: 5, secPerBeat: 0.5 }),
            ).toBe(0);
            expect(
                frameDeltaToBeat({ frameDelta: 1, framePeriodMs: bad, secPerBeat: 0.5 }),
            ).toBe(0);
        }
    });

    it("framePeriodMs 为 0 归一为 1e-6（不产生 Infinity）", () => {
        const v = beatToFrameDelta({ beatDelta: 1, secPerBeat: 0.5, framePeriodMs: 0 });
        expect(Number.isFinite(v)).toBe(true);
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
