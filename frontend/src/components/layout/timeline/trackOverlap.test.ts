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

    /**
     * 随机对拍：新的单遍扫描实现 ≡ 改前的 O(n²) 回扫实现。
     *
     * 【为什么必须对拍】新实现把「对每个前序取 max of min(clipEnd, end_j)」化简为
     * 「min(clipEnd, max_j end_j)」（利用 min 对第二参单调）。这是纯代数化简，
     * 但重叠区的颜色完全由它决定，一旦推错只会表现为"重叠色带宽度不对"，极难
     * 归因。因此用改前的实现作为参照，对确定性伪随机输入逐值比对。
     *
     * 输入刻意覆盖：同起点（触发 id 字典序分支）、相邻首尾相接（触发 1e-9 容差）、
     * 完全包含、完全不相交、零长度。
     */
    describe("随机对拍 ≡ 改前的 O(n²) 实现", () => {
        /** 改前的实现：对每个 clip 回扫全部前序。仅作参照，不参与生产。 */
        function referenceCompute(clips: ClipInfo[]): Record<string, number> {
            const compare = (a: ClipInfo, b: ClipInfo): number => {
                const d = (a.startSec ?? 0) - (b.startSec ?? 0);
                if (Math.abs(d) > 1e-9) return d;
                return String(a.id).localeCompare(String(b.id));
            };
            const sorted = [...clips].sort(compare);
            const out: Record<string, number> = {};
            for (let i = 0; i < sorted.length; i += 1) {
                const clip = sorted[i];
                const clipStart = clip.startSec;
                const clipEnd = clip.startSec + clip.lengthSec;
                let leadingOverlapEnd = clipStart;
                for (let j = 0; j < i; j += 1) {
                    const other = sorted[j];
                    const overlapEnd = Math.min(clipEnd, other.startSec + other.lengthSec);
                    if (overlapEnd <= clipStart + 1e-9) continue;
                    if (overlapEnd > leadingOverlapEnd) leadingOverlapEnd = overlapEnd;
                }
                out[clip.id] = Math.max(0, leadingOverlapEnd - clipStart);
            }
            return out;
        }

        /** 确定性伪随机（回归必须可复现，禁用 Math.random）。 */
        function createRng(seed: number): () => number {
            let state = seed >>> 0;
            return () => {
                state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
                return state / 0x1_0000_0000;
            };
        }

        /** 生成一批 clip：起点量化到 0.5s 网格（制造同起点），长度含 0 与相邻值。 */
        function buildClips(rng: () => number, count: number): ClipInfo[] {
            const out: ClipInfo[] = [];
            for (let i = 0; i < count; i += 1) {
                const startSec = Math.floor(rng() * 12) * 0.5;
                const lengthSec = Math.floor(rng() * 4) * 0.5;
                out.push(clip(`c${String(i).padStart(3, "0")}`, startSec, lengthSec));
            }
            return out;
        }

        it("20 组随机输入逐值相等", () => {
            for (let seed = 1; seed <= 20; seed += 1) {
                const rng = createRng(seed);
                const clips = buildClips(rng, 3 + Math.floor(rng() * 40));
                const actual = computeLeadingOverlapSecByClipId(clips);
                const expected = referenceCompute(clips);
                expect(Object.keys(actual).sort()).toEqual(Object.keys(expected).sort());
                for (const id of Object.keys(expected)) {
                    expect(actual[id]).toBeCloseTo(expected[id] as number, 12);
                }
            }
        });
    });
});
