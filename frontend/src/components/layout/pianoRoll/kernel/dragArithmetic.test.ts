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

import { selectionFrameRange, selectionIndexRange, frameToIndex } from "./dragArithmetic";

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
