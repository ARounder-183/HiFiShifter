/**
 * 起手选区收敛判定（./primeSelection）行为自检。
 *
 * 【主要内容】
 * 1. 未按选择类修饰键 + 多选集合含多个成员 ⇒ 必须收敛（缺陷回归锁）；
 * 2. 未按选择类修饰键 + 被按 clip 不在集合内 ⇒ 必须收敛；
 * 3. 按了多选 / 范围选择键 ⇒ 一律不收敛（用户明确要保留集合）；
 * 4. 空集合 / 单成员且命中 ⇒ 不收敛（避免同手势重复写选区）；
 * 5. 边界：集合大小与命中判定组合的完整真值表。
 *
 * 【作用】这是「拖一个 clip，右边的 clip 也跟着动」的**根因回归锁**。
 * 参与拖拽的集合来自 `multiSelectedClipIds`；导入 / 分割 / 粘贴 / 全选会把它填成
 * 多个，而拖拽不经过"抬起选中"，集合永不自愈 —— 于是拖谁都带上别人。
 *
 * 【与其他模块的关系】覆盖 `primeSelection.ts`；不依赖 DOM 与 React。
 */

import { describe, expect, it } from "vitest";

import { shouldPrimeSelectionOnPress } from "./primeSelection";

/** 便捷构造：默认"未按选择类修饰键"。 */
function args(
    multiSelectionSize: number,
    clipInMultiSelection: boolean,
    shouldPrimeSelection = true,
) {
    return { shouldPrimeSelection, clipInMultiSelection, multiSelectionSize };
}

describe("shouldPrimeSelectionOnPress", () => {
    it("【回归】无修饰键 + 多选集合有多个成员 ⇒ 收敛（否则拖拽会带走整组）", () => {
        // 现场：导入 2 个音频块后 multiSelectedClipIds = [c1, c2]，
        // 此后直接拖 c1 会让 c2（右边那个）一起移动，且永不自愈。
        expect(shouldPrimeSelectionOnPress(args(2, true))).toBe(true);
        expect(shouldPrimeSelectionOnPress(args(3, true))).toBe(true);
    });

    it("【回归】无修饰键 + 被按 clip 不在多选集合内 ⇒ 收敛", () => {
        expect(shouldPrimeSelectionOnPress(args(2, false))).toBe(true);
        expect(shouldPrimeSelectionOnPress(args(5, false))).toBe(true);
    });

    it("按下多选 / 范围选择修饰键 ⇒ 不收敛（用户明确要保留集合）", () => {
        // 多选切换或范围选择进行中：集合是用户意图，不能被起手清掉。
        expect(shouldPrimeSelectionOnPress(args(3, true, false))).toBe(false);
        expect(shouldPrimeSelectionOnPress(args(3, false, false))).toBe(false);
        // 即便集合只有 1 个成员，带修饰键时也不该走收敛路径。
        expect(shouldPrimeSelectionOnPress(args(1, true, false))).toBe(false);
    });

    it("空集合 ⇒ 不收敛（单选权威在 selectedClipId，避免同手势重复写选区）", () => {
        expect(shouldPrimeSelectionOnPress(args(0, false))).toBe(false);
        expect(shouldPrimeSelectionOnPress(args(0, true))).toBe(false);
    });

    it("单成员且命中的就是它 ⇒ 不收敛（已是目标状态，幂等）", () => {
        expect(shouldPrimeSelectionOnPress(args(1, true))).toBe(false);
    });

    it("单成员但命中的不是它 ⇒ 收敛", () => {
        expect(shouldPrimeSelectionOnPress(args(1, false))).toBe(true);
    });

    it("真值表：多选集合大小 × 命中 的完整覆盖（无遗漏组合）", () => {
        const table: Array<[number, boolean, boolean]> = [
            // [size, clipInMultiSelection, expected]
            [0, false, false],
            [0, true, false],
            [1, false, true],
            [1, true, false],
            [2, false, true],
            [2, true, true],
            [9, false, true],
            [9, true, true],
        ];
        for (const [size, inSelection, expected] of table) {
            expect(
                shouldPrimeSelectionOnPress(args(size, inSelection)),
                `size=${size} inSelection=${inSelection}`,
            ).toBe(expected);
        }
    });

    it("负数集合大小不抛错（脏值防御：按空集合处理）", () => {
        expect(() => shouldPrimeSelectionOnPress(args(-1, true))).not.toThrow();
        expect(shouldPrimeSelectionOnPress(args(-1, true))).toBe(false);
    });
});
