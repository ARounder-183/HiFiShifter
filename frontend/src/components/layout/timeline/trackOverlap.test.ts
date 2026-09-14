/**
 * 时间轴 · clip 前置重叠计算的单测。
 *
 * 【为什么单测这个函数】它从 `TrackLane.tsx` 拆出（旧轨道组件已随旧渲染路径
 * 删除），但**内核仍在用**：内核视图 `TimelineKernelView` 自行挂载的
 * `TimelineWaveformSurface` 依赖它。拆分类改动必须有用例固定其行为，否则"拆错了"
 * 只会表现为波形重叠区的颜色异常，极难归因。
 */
import { describe, expect, it } from "vitest";

import { computeLeadingOverlapSecByClipId } from "./trackOverlap";
import type { ClipInfo } from "../../../features/session/sessionTypes";

/** 造一个测试用 clip（只填本函数用到的字段）。 */
function clip(id: string, startSec: number, lengthSec: number): ClipInfo {
    return { id, startSec, lengthSec } as ClipInfo;
}

describe("computeLeadingOverlapSecByClipId", () => {
    it("单个 clip 无重叠", () => {
        expect(computeLeadingOverlapSecByClipId([clip("a", 0, 5)])).toEqual({ a: 0 });
    });

    it("完全不重叠时全为 0", () => {
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 2), clip("b", 5, 2)]);
        expect(r).toEqual({ a: 0, b: 0 });
    });

    it("★ 后一个 clip 覆盖前一个的尾部时，只算其左前导重叠段", () => {
        // a: [0,5)，b: [3,8) → b 的前导重叠 = 5 - 3 = 2
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 5), clip("b", 3, 5)]);
        expect(r.a).toBe(0);
        expect(r.b).toBeCloseTo(2, 9);
    });

    it("★ 重叠取「最远的那个前序 clip 末端」（多个前序时取 max）", () => {
        // a: [0,4)，b: [0,3)，c: [2,6) → c 的前导重叠 = max(4,3) - 2 = 2
        const r = computeLeadingOverlapSecByClipId([
            clip("a", 0, 4),
            clip("b", 0, 3),
            clip("c", 2, 4),
        ]);
        expect(r.c).toBeCloseTo(2, 9);
    });

    it("渲染顺序按 startSec 升序、同起点按 id 字典序（决定谁算「前序」）", () => {
        // 同起点时 id 字典序小的在前：因此对 "b" 而言 "a" 是前序 → b 的前导重叠 = 3
        const r = computeLeadingOverlapSecByClipId([clip("b", 0, 3), clip("a", 0, 3)]);
        expect(r.b).toBeCloseTo(3, 9);
        expect(r.a).toBe(0);
    });

    it("结果不含负数（不重叠时为 0 而非负值）", () => {
        const r = computeLeadingOverlapSecByClipId([clip("a", 0, 1), clip("b", 0, 1)]);
        for (const v of Object.values(r)) expect(v).toBeGreaterThanOrEqual(0);
    });

    it("空输入返回空对象（不抛异常）", () => {
        expect(computeLeadingOverlapSecByClipId([])).toEqual({});
    });
});
