/**
 * 参数编辑器内核 · 手势命中测试（纯函数）行为自检。
 *
 * 【本测试守护什么】
 * 1. **曲线邻域判定**：悬停浮窗与"拖动已有曲线"依赖"指针是否落在参数线 10px 以内"。
 *    这个 10px 是**绘制坐标**下的距离，须与 `valueToY` 同一投影，否则缩放后命中区
 *    会与看到的曲线分离。
 * 2. **pitch 的 +0.5 偏移**：`render.ts` 绘制 pitch 曲线时把 MIDI 值加 0.5（使曲线
 *    居于琴键中心，见 `curvePoints`）。命中测试必须施加**同一偏移**，否则指针要偏离
 *    曲线半个键高才能命中。修好之前这里差 0.5 个 MIDI 值 ≈ 视口高度 / 音域。
 * 3. **选区边缘命中**：拉伸选区需要 8px 的边缘带；左右两条边都要能命中，且
 *    选区退化为零宽时不应命中"两条边"之外的任何位置。
 * 4. **选区拖拽命中**：只有在选区内**且**靠近曲线时才可拖——否则整个选区块会变成
 *    拖拽热区，用户在选区内画线会被误判为拖动。
 * 5. **非有限指针坐标一律不命中**：NaN 参与比较恒为 false，看似"安全"，但会把
 *    `Math.abs(NaN - x) <= w` 这类判定静默变成不命中；显式拒绝比隐式行为更可靠。
 *
 * 【为什么抽成纯模块】本工程的参数编辑器交互集中在一个 3,875 行的 hook 里，命中
 * 判定与手势状态机混在一起，既无法单测也无法复用。判定本身只依赖「指针位置 +
 * 序列化的几何输入」，因此按 `timeline/kernel/interaction/` 的既有做法抽为纯函数。
 */
import { describe, expect, it } from "vitest";

import {
    CURVE_HIT_RADIUS_PX,
    SELECTION_EDGE_HIT_PX,
    curveValueAtPointerFrame,
    hitTestSelectionBody,
    hitTestSelectionEdge,
    isPointerNearCurve,
} from "./gestureHitTest";

/**
 * 造一个线性的值→y 投影：`span` 个值铺满 `heightPx`。
 *
 * 方向与真实实现一致（值越大 y 越小，屏幕 y 轴向下）。
 */
function makeValueToY(center: number, span: number, heightPx: number) {
    return (value: number): number => {
        const top = center + span / 2;
        return ((top - value) / span) * heightPx;
    };
}

describe("curveValueAtPointerFrame", () => {
    it("按帧周期与 stride 定位采样下标", () => {
        // fp=5ms → 200 帧/秒；startFrame=0，stride=1 → 第 1 秒对应 idx 200。
        const edit = new Array(1000).fill(0).map((_, i) => i);
        const value = curveValueAtPointerFrame({
            sec: 1,
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: edit,
        });
        expect(value).toBe(200);
    });

    it("stride > 1 时下标按 stride 折算", () => {
        const edit = new Array(200).fill(0).map((_, i) => i * 10);
        // 第 1 秒 = 帧 200；startFrame=0、stride=2 → idx = 100 → 值 1000。
        const value = curveValueAtPointerFrame({
            sec: 1,
            startFrame: 0,
            stride: 2,
            framePeriodMs: 5,
            values: edit,
        });
        expect(value).toBe(1000);
    });

    it("startFrame 非 0 时下标相对起点折算", () => {
        const edit = new Array(100).fill(7);
        // 帧 200、起点 100、stride 1 → idx 100（越界，应为 null）。
        expect(
            curveValueAtPointerFrame({
                sec: 1,
                startFrame: 100,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
            }),
        ).toBeNull();
        // 帧 100 对应 idx 0。
        expect(
            curveValueAtPointerFrame({
                sec: 0.5,
                startFrame: 100,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
            }),
        ).toBe(7);
    });

    it("秒为负 / 越界 / 值为非有限时返回 null", () => {
        const edit = [1, 2, 3];
        expect(
            curveValueAtPointerFrame({
                sec: -1,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
            }),
        ).toBe(1); // 帧号被 clamp 到 0
        expect(
            curveValueAtPointerFrame({
                sec: 999,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
            }),
        ).toBeNull();
        expect(
            curveValueAtPointerFrame({
                sec: Number.NaN,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
            }),
        ).toBeNull();
        expect(
            curveValueAtPointerFrame({
                sec: 0,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 0, // 非法帧周期
                values: edit,
            }),
        ).toBeNull();
    });

    it("空数组 / 非数组返回 null（不抛异常）", () => {
        expect(
            curveValueAtPointerFrame({
                sec: 0,
                startFrame: 0,
                stride: 1,
                framePeriodMs: 5,
                values: [],
            }),
        ).toBeNull();
    });
});

describe("isPointerNearCurve", () => {
    const valueToY = makeValueToY(72, 24, 480);

    it("指针落在曲线 10px 内 → 命中；超出 → 不命中", () => {
        const curveY = valueToY(72); // 中心 = 240

        // 正好在曲线上
        expect(
            isPointerNearCurve({
                pointerY: curveY,
                param: "cents",
                valueToY,
                curveValue: 72,
            }),
        ).toBe(true);
        // 边界内侧（9px）
        expect(
            isPointerNearCurve({
                pointerY: curveY + CURVE_HIT_RADIUS_PX - 1,
                param: "cents",
                valueToY,
                curveValue: 72,
            }),
        ).toBe(true);
        // 边界外侧（11px）
        expect(
            isPointerNearCurve({
                pointerY: curveY + CURVE_HIT_RADIUS_PX + 1,
                param: "cents",
                valueToY,
                curveValue: 72,
            }),
        ).toBe(false);
        // 两种指针坐标都不给 → 不命中（不是默认命中）
        expect(isPointerNearCurve({ param: "cents", valueToY, curveValue: 72 })).toBe(false);
    });

    it("pitch 参数施加 +0.5 偏移（曲线画在键中心）", () => {
        // 曲线值是 MIDI 60；绘制时加 0.5 → 60.5。指针在 60.5 对应的 y 上应命中。
        const drawnY = valueToY(60.5);
        expect(
            isPointerNearCurve({
                pointerY: drawnY,
                param: "pitch",
                valueToY,
                curveValue: 60,
            }),
        ).toBe(true);
        // 若忘了 +0.5，就会拿 60 的 y 去比：两者相差 0.5 个 MIDI 值。
        const undrawnY = valueToY(60);
        const gapPx = Math.abs(undrawnY - drawnY);
        // 该 fixture 下 0.5 个 MIDI = 10px，正好在边界上 —— 说明这个偏移不是可忽略的小量。
        expect(gapPx).toBeCloseTo(10, 6);
    });

    it("非 pitch 参数不施加偏移", () => {
        expect(
            isPointerNearCurve({
                pointerY: valueToY(5),
                param: "formant_shift",
                valueToY,
                curveValue: 5,
            }),
        ).toBe(true);
    });

    it("指针 y 或曲线值非有限 → 不命中", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
            expect(
                isPointerNearCurve({
                    pointerY: bad,
                    param: "pitch",
                    valueToY,
                    curveValue: 60,
                }),
            ).toBe(false);
            expect(
                isPointerNearCurve({
                    pointerY: 100,
                    param: "pitch",
                    valueToY,
                    curveValue: bad,
                }),
            ).toBe(false);
        }
    });

    it("两种传参方式等价：pointerY 与 pointerValue 二选一", () => {
        // pointerValue 是"参数值"，内部经 valueToY 换算；对非 pitch 两者应一致。
        const v = 70;
        const y = valueToY(v);
        expect(
            isPointerNearCurve({ pointerY: y, param: "cents", valueToY, curveValue: v }),
        ).toBe(true);
        expect(
            isPointerNearCurve({ pointerValue: v, param: "cents", valueToY, curveValue: v }),
        ).toBe(true);
    });
});

describe("hitTestSelectionEdge", () => {
    const args = { leftXPx: 100, rightXPx: 300 };

    it("左右边缘 8px 内命中，中间不命中", () => {
        expect(hitTestSelectionEdge({ ...args, localXPx: 100 })).toBe("left");
        expect(hitTestSelectionEdge({ ...args, localXPx: 100 - SELECTION_EDGE_HIT_PX })).toBe(
            "left",
        );
        expect(hitTestSelectionEdge({ ...args, localXPx: 100 + SELECTION_EDGE_HIT_PX })).toBe(
            "left",
        );
        expect(hitTestSelectionEdge({ ...args, localXPx: 200 })).toBeNull();
        expect(hitTestSelectionEdge({ ...args, localXPx: 300 })).toBe("right");
        expect(hitTestSelectionEdge({ ...args, localXPx: 300 + SELECTION_EDGE_HIT_PX })).toBe(
            "right",
        );
        expect(hitTestSelectionEdge({ ...args, localXPx: 300 - SELECTION_EDGE_HIT_PX })).toBe(
            "right",
        );
    });

    it("稍超边界不命中", () => {
        expect(hitTestSelectionEdge({ ...args, localXPx: 100 - SELECTION_EDGE_HIT_PX - 1 })).toBeNull();
        expect(hitTestSelectionEdge({ ...args, localXPx: 300 + SELECTION_EDGE_HIT_PX + 1 })).toBeNull();
    });

    it("左右边重叠时左侧优先（与既有实现逐字一致）", () => {
        // 选区宽度 4px < 2×8px：所有位置都落在某条边的带内。
        // 既有实现是 `hitLeft ? "left" : hitRight ? "right" : null`，即左缘优先；
        // 固定取左缘可保证"向左拉伸"始终可达，且行为确定（不随距离抖动）。
        const narrow = { leftXPx: 100, rightXPx: 104 };
        expect(hitTestSelectionEdge({ ...narrow, localXPx: 100 })).toBe("left");
        expect(hitTestSelectionEdge({ ...narrow, localXPx: 102 })).toBe("left");
        // 两条边的带**都**覆盖 106（距左 6、距右 2）→ 仍取左缘，不是取更近的那条。
        expect(hitTestSelectionEdge({ ...narrow, localXPx: 106 })).toBe("left");
        // 超出左缘带（距左 9 > 8）但仍在右缘带内（距右 5）→ 这时才轮到右缘。
        expect(hitTestSelectionEdge({ ...narrow, localXPx: 109 })).toBe("right");
        // 两条带都不覆盖 → 不命中。
        expect(hitTestSelectionEdge({ ...narrow, localXPx: 113 })).toBeNull();
    });

    it("边界次序颠倒时自动归一（不依赖调用方排序）", () => {
        expect(
            hitTestSelectionEdge({ leftXPx: 300, rightXPx: 100, localXPx: 100 }),
        ).toBe("left");
        expect(
            hitTestSelectionEdge({ leftXPx: 300, rightXPx: 100, localXPx: 300 }),
        ).toBe("right");
    });

    it("非有限输入不命中", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(hitTestSelectionEdge({ ...args, localXPx: bad })).toBeNull();
            expect(hitTestSelectionEdge({ leftXPx: bad, rightXPx: 300, localXPx: 100 })).toBeNull();
        }
    });
});

describe("hitTestSelectionBody", () => {
    const args = { leftXPx: 100, rightXPx: 300 };

    it("选区内且靠近曲线才命中（否则选区块会变成拖拽热区）", () => {
        expect(hitTestSelectionBody({ ...args, localXPx: 200, nearCurve: true })).toBe(true);
        expect(hitTestSelectionBody({ ...args, localXPx: 200, nearCurve: false })).toBe(false);
    });

    it("选区外即使靠近曲线也不命中", () => {
        expect(hitTestSelectionBody({ ...args, localXPx: 99, nearCurve: true })).toBe(false);
        expect(hitTestSelectionBody({ ...args, localXPx: 301, nearCurve: true })).toBe(false);
    });

    it("边界是闭区间（贴着边缘也算在选区内）", () => {
        expect(hitTestSelectionBody({ ...args, localXPx: 100, nearCurve: true })).toBe(true);
        expect(hitTestSelectionBody({ ...args, localXPx: 300, nearCurve: true })).toBe(true);
    });

    it("非有限输入不命中", () => {
        expect(
            hitTestSelectionBody({ ...args, localXPx: Number.NaN, nearCurve: true }),
        ).toBe(false);
    });
});
