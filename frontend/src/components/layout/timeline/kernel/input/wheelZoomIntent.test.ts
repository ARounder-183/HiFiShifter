/**
 * 滚轮缩放意图判定（./wheelZoomIntent）行为自检。
 *
 * 【主要内容】
 * 1. 方向约定：负增量 = 放大、正 = 缩小（与旧实现一致）；
 * 2. **回归**：`deltaY = 0` 的纯横向手势不再被判成缩小；
 * 3. **回归**：precision touchpad 的变号小幅噪声**完全不缩放**（旧实现在此处
 *    逐事件翻转方向，用户报告为"Windows 上 ctrl+滚轮剧烈抖动"）；
 * 4. 死区不钝化真实输入：鼠标滚轮一格（±100）恰好一步；
 * 5. 真实反向立刻生效（不被上一次手势的残余吃掉）；
 * 6. 非法输入不产生 NaN、不污染累积状态。
 *
 * 【作用】这是"ctrl+滚轮抖动"的**回归锁**。抖动不是几何或渲染问题，而是
 * 方向判定问题：旧规则 `factor = deltaY < 0 ? 1.1 : 0.9` 只看符号，于是
 * （a）`deltaY = 0` 落入缩小分支、（b）任何变号噪声都翻转方向。两者都在这
 * 条测试里被钉死。
 *
 * 【与其他模块的关系】覆盖 `wheelZoomIntent.ts`；不依赖 DOM 与 React
 * （node 环境的 vitest 可直接驱动）。
 */

import { describe, expect, it } from "vitest";

import {
    createWheelZoomAccumulator,
    resolveWheelZoomStep,
    WHEEL_ZOOM_DEADZONE_PX,
    type WheelZoomAccumulator,
} from "./wheelZoomIntent";

/** 把一串事件喂进解析器，返回每个事件产生的方向。 */
function drive(events: Array<{ deltaX?: number; deltaY?: number }>, deadzonePx?: number): number[] {
    let accumulator: WheelZoomAccumulator = createWheelZoomAccumulator();
    const directions: number[] = [];
    for (const event of events) {
        const step = resolveWheelZoomStep({
            accumulator,
            deltaX: event.deltaX ?? 0,
            deltaY: event.deltaY ?? 0,
            deadzonePx,
        });
        accumulator = step.accumulator;
        directions.push(step.direction);
    }
    return directions;
}

describe("resolveWheelZoomStep", () => {
    it("滚轮向上（负增量）= 放大、向下 = 缩小（方向约定与旧实现一致）", () => {
        expect(drive([{ deltaY: -100 }])).toEqual([-1]);
        expect(drive([{ deltaY: 100 }])).toEqual([1]);
    });

    it("鼠标滚轮一格（±100 / ±120）恰好一步，不被死区钝化", () => {
        expect(drive([{ deltaY: -100 }])).toEqual([-1]);
        expect(drive([{ deltaY: 120 }])).toEqual([1]);
        expect(drive([{ deltaY: -120 }, { deltaY: -100 }])).toEqual([-1, -1]);
    });

    it("【回归】deltaY = 0 的纯横向手势按横向符号缩放，不再被判成缩小", () => {
        // 旧实现：deltaY = 0 → 0 < 0 为假 → factor = 0.9 → 稳定地"缩小"。
        // 横向负增量（向右滑 = 缩小方向的相反）应判为放大。
        expect(drive([{ deltaX: -120, deltaY: 0 }])).toEqual([-1]);
        expect(drive([{ deltaX: 120, deltaY: 0 }])).toEqual([1]);
    });

    it("【回归】deltaY = 0 且完全无增量时不产生任何缩放", () => {
        expect(drive([{ deltaX: 0, deltaY: 0 }])).toEqual([0]);
    });

    it("【回归】变号的小幅噪声完全不缩放（旧实现逐事件翻转方向 = 抖动）", () => {
        // 实测的 precision touchpad ctrl+捏合增量序列（Windows）。
        const noise = [
            { deltaY: -2 },
            { deltaY: 3 },
            { deltaY: -1.5 },
            { deltaY: 1.8 },
            { deltaY: -0.4 },
            { deltaY: 0.4 },
            { deltaY: -3 },
            { deltaY: 2 },
        ];
        expect(drive(noise)).toEqual(noise.map(() => 0));
    });

    it("【回归】横向噪声同样不缩放", () => {
        expect(
            drive([
                { deltaX: -3, deltaY: 0 },
                { deltaX: 4, deltaY: 0 },
                { deltaX: -2, deltaY: 0 },
                { deltaX: 5, deltaY: 0 },
            ]),
        ).toEqual([0, 0, 0, 0]);
    });

    it("同向小幅增量会累积到死区后才出手（慢速捏合仍有响应）", () => {
        const directions = drive([{ deltaY: -3 }, { deltaY: -3 }, { deltaY: -3 }, { deltaY: -3 }]);
        // 前两次累积 6px（< 8）不出手，第三次越过死区。
        expect(directions.slice(0, 2)).toEqual([0, 0]);
        expect(directions[2]).toBe(-1);
    });

    it("单调同向滚动不出现反向步（抖动的直接判据）", () => {
        const directions = drive(Array.from({ length: 12 }, () => ({ deltaY: -1.2 })));
        expect(directions).not.toContain(1);
    });

    it("真实反向立刻生效，不被上一次的残余吃掉", () => {
        // 先积累 -6（未到死区），再反向 +100：必须立刻缩小而不是继续累积。
        const directions = drive([{ deltaY: -6 }, { deltaY: 100 }]);
        expect(directions[0]).toBe(0);
        expect(directions[1]).toBe(1);
    });

    it("大增量饱和在死区，不留下会立刻触发下一步的余额", () => {
        // 旧式"减去死区"会留下 ~92 的余额，紧接一次微小噪声就会再触发一整步。
        const directions = drive([{ deltaY: 1000 }, { deltaY: -0.5 }, { deltaY: 0.5 }]);
        expect(directions).toEqual([1, 0, 0]);
    });

    it("非法增量不产生 NaN、不污染累积（NaN / Infinity 归零）", () => {
        const step = resolveWheelZoomStep({
            accumulator: createWheelZoomAccumulator(),
            deltaX: Number.NaN,
            deltaY: Number.POSITIVE_INFINITY,
        });
        expect(step.direction).toBe(0);
        expect(Number.isFinite(step.accumulator.pending)).toBe(true);
        expect(step.accumulator.pending).toBe(0);
    });

    it("非法死区回退默认值", () => {
        expect(drive([{ deltaY: -100 }], 0)).toEqual([-1]);
        expect(drive([{ deltaY: -100 }], Number.NaN)).toEqual([-1]);
    });

    it("不修改传入的累积状态（返回新对象）", () => {
        const accumulator = createWheelZoomAccumulator();
        const before = accumulator.pending;
        const step = resolveWheelZoomStep({ accumulator, deltaX: 0, deltaY: -100 });
        expect(accumulator.pending).toBe(before);
        expect(step.accumulator).not.toBe(accumulator);
    });

    it("死区常量在实测噪声上界与真实滚轮下界之间", () => {
        // 噪声实测上界约 ±3，滚轮一格 ±100：死区必须落在这两者之间。
        expect(WHEEL_ZOOM_DEADZONE_PX).toBeGreaterThan(3);
        expect(WHEEL_ZOOM_DEADZONE_PX).toBeLessThan(100);
    });
});
