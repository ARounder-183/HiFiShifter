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
    SELECTION_EDGE_MIN_WIDTH_PX,
    curvePointAtPointer,
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

describe("curvePointAtPointer", () => {
    const valueToY = makeValueToY(0, 100, 100); // 1 个值 = 1px，便于核对

    it("正好落在采样点上 → 该点的值与 y", () => {
        const edit = [10, 20, 30];
        const point = curvePointAtPointer({
            sec: 0.005, // 帧 1（fp=5ms）
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: edit,
            param: "cents",
            valueToY,
        });
        expect(point?.value).toBe(20);
        expect(point?.y).toBe(valueToY(20));
    });

    it("**落在两个采样点之间 → 沿折线插值**（这是本函数存在的理由）", () => {
        // 两个采样点：帧 0 值 0、帧 1 值 40（陡峭段；本 fixture 下 1 个值 = 1px）。
        const edit = [0, 40];
        const point = curvePointAtPointer({
            sec: 0.0025, // 帧 0.5 —— 正中间
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: edit,
            param: "cents",
            valueToY,
        });
        expect(point?.value).toBeCloseTo(20, 9);
        // y 取两端点的中点：若只取"最近采样点的值"（帧 0 → 值 0），量到的距离是
        // 半个跨度 —— 这里 20px，是命中半径的两倍，"光标压在线上却抓不住"。
        expect(point?.y).toBeCloseTo((valueToY(0) + valueToY(40)) / 2, 9);
        expect(Math.abs((point?.y ?? 0) - valueToY(0))).toBeGreaterThan(CURVE_HIT_RADIUS_PX);
    });

    it("stride > 1 时插值发生在**采样索引**空间", () => {
        // 采样点每 2 帧一个：帧 0 值 0、帧 2 值 40。
        const edit = [0, 40];
        const point = curvePointAtPointer({
            sec: 0.005, // 帧 1 → 采样坐标 0.5
            startFrame: 0,
            stride: 2,
            framePeriodMs: 5,
            values: edit,
            param: "cents",
            valueToY,
        });
        expect(point?.value).toBeCloseTo(20, 9);
    });

    it("startFrame 非 0 时相对起点折算；范围外返回 null", () => {
        const edit = new Array(100).fill(7);
        // 帧 100 = 起点 → idx 0。
        expect(
            curvePointAtPointer({
                sec: 0.5,
                startFrame: 100,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
                param: "cents",
                valueToY,
            })?.value,
        ).toBe(7);
        // 起点之前 → null（折线不存在于首采样点之前）。
        expect(
            curvePointAtPointer({
                sec: 0,
                startFrame: 100,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
                param: "cents",
                valueToY,
            }),
        ).toBeNull();
        // 末采样点之后 → null。
        expect(
            curvePointAtPointer({
                sec: 1,
                startFrame: 100,
                stride: 1,
                framePeriodMs: 5,
                values: edit,
                param: "cents",
                valueToY,
            }),
        ).toBeNull();
    });

    it("pitch 施加 +0.5 偏移（曲线画在键中心，与绘制同源）", () => {
        const point = curvePointAtPointer({
            sec: 0,
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: [60],
            param: "pitch",
            valueToY,
        });
        expect(point?.y).toBe(valueToY(60.5));
        // 值本身不加偏移（浮窗显示的是参数值）。
        expect(point?.value).toBe(60);
    });

    it("y 在**像素域**插值：投影带 clamp 时与「先插值再投影」不同", () => {
        // 投影把值夹到 [0, 20]（真实轴在范围外就是这个行为）。
        const clamped = (v: number) => Math.min(20, Math.max(0, v)) * 10;
        const point = curvePointAtPointer({
            sec: 0.0025, // 两端点正中间
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: [0, 100],
            param: "cents",
            valueToY: clamped,
        });
        // 两端点 y = 0 与 200 → 中点 100（与绘制折线逐像素一致）。
        expect(point?.y).toBeCloseTo(100, 9);
        // "先插值再投影"会得到 clamp(50)=20 → 200，与画面不符。
        expect(clamped(point?.value ?? 0)).toBe(200);
    });

    it("非法输入返回 null（不抛异常、不返回 0）", () => {
        const base = {
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: [1, 2, 3],
            param: "cents",
            valueToY,
        } as const;
        expect(curvePointAtPointer({ ...base, sec: Number.NaN })).toBeNull();
        expect(curvePointAtPointer({ ...base, sec: 0, framePeriodMs: 0 })).toBeNull();
        expect(curvePointAtPointer({ ...base, sec: 0, values: [] })).toBeNull();
        expect(curvePointAtPointer({ ...base, sec: 0, startFrame: Number.NaN })).toBeNull();
    });
});

describe("isPointerNearCurve", () => {
    it("指针落在曲线 10px 内 → 命中；超出 → 不命中", () => {
        const curveY = 240;
        expect(isPointerNearCurve({ pointerY: curveY, curveY })).toBe(true);
        // 边界内侧（9px）
        expect(isPointerNearCurve({ pointerY: curveY + CURVE_HIT_RADIUS_PX - 1, curveY })).toBe(
            true,
        );
        // 边界外侧（11px）
        expect(isPointerNearCurve({ pointerY: curveY + CURVE_HIT_RADIUS_PX + 1, curveY })).toBe(
            false,
        );
    });

    it("指针 y 或曲线 y 非有限 → 不命中（不是默认命中）", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
            expect(isPointerNearCurve({ pointerY: bad, curveY: 100 })).toBe(false);
            expect(isPointerNearCurve({ pointerY: 100, curveY: bad })).toBe(false);
        }
    });

    it("与折线插值串起来：中间位置也能命中（端到端形式）", () => {
        const valueToY = makeValueToY(0, 100, 100);
        const edit = [0, 40];
        const pointerY = (valueToY(0) + valueToY(40)) / 2; // 线上正中间
        const point = curvePointAtPointer({
            sec: 0.0025,
            startFrame: 0,
            stride: 1,
            framePeriodMs: 5,
            values: edit,
            param: "cents",
            valueToY,
        });
        expect(isPointerNearCurve({ pointerY, curveY: point?.y ?? Number.NaN })).toBe(true);
        // 若误用"最近采样点的值"（帧 0 → 值 0），就会漏掉这一次命中。
        expect(isPointerNearCurve({ pointerY, curveY: valueToY(0) })).toBe(false);
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
        expect(
            hitTestSelectionEdge({ ...args, localXPx: 100 - SELECTION_EDGE_HIT_PX - 1 }),
        ).toBeNull();
        expect(
            hitTestSelectionEdge({ ...args, localXPx: 300 + SELECTION_EDGE_HIT_PX + 1 }),
        ).toBeNull();
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
        expect(hitTestSelectionEdge({ leftXPx: 300, rightXPx: 100, localXPx: 100 })).toBe("left");
        expect(hitTestSelectionEdge({ leftXPx: 300, rightXPx: 100, localXPx: 300 })).toBe("right");
    });

    it("非有限输入不命中", () => {
        for (const bad of [Number.NaN, Number.POSITIVE_INFINITY]) {
            expect(hitTestSelectionEdge({ ...args, localXPx: bad })).toBeNull();
            expect(hitTestSelectionEdge({ leftXPx: bad, rightXPx: 300, localXPx: 100 })).toBeNull();
        }
    });

    it("零宽 / 亚像素选区不命中（随机单击留下的不可见选区）", () => {
        // 单击会在点击位置留下一段 startBeat === endBeat 的选区：两条边界投影到
        // 同一个 x，若不做宽度门槛，命中带内的任意位置都会命中"左缘"，用户就会在
        // 什么都看不到的地方看到 ew-resize 光标。
        const degenerate = { leftXPx: 200, rightXPx: 200 };
        for (const localXPx of [200, 192, 208, 195, 205]) {
            expect(hitTestSelectionEdge({ ...degenerate, localXPx })).toBeNull();
        }
        // 亚像素（取样点落在同一设备像素内）同样视为不可见。
        expect(hitTestSelectionEdge({ leftXPx: 200, rightXPx: 200.4, localXPx: 200 })).toBeNull();
        // 门槛之下的极窄选区（1px）同样不给边缘交互。
        expect(hitTestSelectionEdge({ leftXPx: 200, rightXPx: 201, localXPx: 200 })).toBeNull();
        // 恰好达到最小可见宽度 → 恢复命中（与既有"左缘优先"规则一致）。
        expect(
            hitTestSelectionEdge({
                leftXPx: 200,
                rightXPx: 200 + SELECTION_EDGE_MIN_WIDTH_PX,
                localXPx: 200,
            }),
        ).toBe("left");
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
        expect(hitTestSelectionBody({ ...args, localXPx: Number.NaN, nearCurve: true })).toBe(
            false,
        );
    });
});
