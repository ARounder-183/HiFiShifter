/**
 * 拖拽起手选区收敛判定（./primeSelection）行为自检。
 *
 * 【主要内容】
 * 1. **缺陷锁**：动作驱动填充的多选集合（导入 / 粘贴 / 分割…）在拖拽起手时必须收敛
 *    —— 否则"拖一个 clip"会带上右邻（用户报告的症状）；
 * 2. **功能锁**：用户显式多选（框选 / 修饰键点击 / 范围选择）后拖拽**不得**收敛
 *    —— 整组移动是既有功能，误收敛会让它静默消失；
 * 3. 按住多选切换键（Ctrl/⌘，同时是复制拖拽键）不收敛 —— 复制要整组复制；
 * 4. **不依赖 Shift**：Shift 是 click 类型的范围选择键兼 drag 类型的免吸附键，
 *    拖拽路径只认后者，因此判定结果与 Shift 无关（这条同时锁住 Shift+拖拽
 *    不再带回陈旧邻块）；
 * 5. 边界：空集合 / 单成员 / 锚点不在集合内。
 *
 * 【作用】这是「拖一个 clip，右边的 clip 也跟着动」的**根因回归锁**，同时也是
 * "多选整组拖动"不被误伤的**功能锁** —— 两条都必须有，缺一条就会有一天把
 * 其中一侧改坏而无人发现。
 *
 * 【与其他模块的关系】覆盖 `primeSelection.ts`；不依赖 DOM 与 React。
 */

import { describe, expect, it } from "vitest";

import {
    shouldCollapseStaleSelectionOnDrag,
    type CollapseStaleSelectionArgs,
} from "./primeSelection";

/** 便捷构造：默认"动作驱动填充的集合 + 锚点在内 + 无修饰键"。 */
function stale(overrides: Partial<CollapseStaleSelectionArgs> = {}): boolean {
    return shouldCollapseStaleSelectionOnDrag({
        multiSelectToggleActive: false,
        selectionIntentional: false,
        multiSelectionSize: 2,
        anchorInMultiSelection: true,
        ...overrides,
    });
}

describe("shouldCollapseStaleSelectionOnDrag", () => {
    it("【缺陷锁】动作驱动填充的集合 + 锚点在内 ⇒ 收敛（否则右邻会一起动）", () => {
        // 现场：一次导入 3 个音频文件后集合 = [c1,c2,c3]（导入把全部新 clip 设为选中），
        // 用户随手抓 c1 拖 —— 修复前 c2/c3 跟着走。
        expect(stale()).toBe(true);
        expect(stale({ multiSelectionSize: 3 })).toBe(true);
        expect(stale({ multiSelectionSize: 99 })).toBe(true);
    });

    it("【功能锁】用户显式多选的集合 ⇒ 不收敛（整组移动是既有功能）", () => {
        // 框选 / ⌘ 逐个点选 A、B 后拖 A ⇒ 两个都该动。
        // 若这里误判为"需要收敛"，"多选整组拖动"会静默消失。
        expect(stale({ selectionIntentional: true })).toBe(false);
        expect(stale({ selectionIntentional: true, multiSelectionSize: 5 })).toBe(false);
    });

    it("按住多选切换键（Ctrl/⌘）⇒ 不收敛（该键同时是复制拖拽键，要整组复制）", () => {
        expect(stale({ multiSelectToggleActive: true })).toBe(false);
        expect(stale({ multiSelectToggleActive: true, selectionIntentional: true })).toBe(false);
    });

    it("集合为空 / 只有一个成员 ⇒ 不收敛（参与者本就只有它）", () => {
        expect(stale({ multiSelectionSize: 0 })).toBe(false);
        expect(stale({ multiSelectionSize: 1 })).toBe(false);
        // 单成员时即便是动作填充也无意义（没有"别人"会被带上）。
        expect(stale({ multiSelectionSize: 1, anchorInMultiSelection: false })).toBe(false);
    });

    it("锚点不在集合内 ⇒ 不收敛（参与者解析已只取锚点，无需额外动作）", () => {
        expect(stale({ anchorInMultiSelection: false })).toBe(false);
    });

    it("【Shift 无关】判定不接受也不依赖范围选择键（Shift 是 click 类型键）", () => {
        // 函数签名里**没有** rangeSelect 维度：拖拽只认 drag 类型的免吸附语义，
        // 后者不影响选区。这条断言把"签名里不该出现 Shift"钉住 ——
        // 修复前正是因 Shift 走了 click 语义的判定，导致 Shift+拖拽带回陈旧邻块。
        const keys = Object.keys({
            multiSelectToggleActive: false,
            selectionIntentional: false,
            multiSelectionSize: 2,
            anchorInMultiSelection: true,
        });
        expect(keys).not.toContain("rangeSelectActive");
        expect(keys).not.toContain("shiftKey");
        // 无论集合多大、锚点是否在内，只要来源是显式选择就不收敛（与 Shift 无关）。
        expect(stale({ selectionIntentional: true, multiSelectionSize: 50 })).toBe(false);
    });

    it("真值表：多选切换键 × 来源 × 集合大小 × 锚点在内 的完整覆盖", () => {
        const table: Array<[boolean, boolean, number, boolean, boolean]> = [
            // [toggle, intentional, size, anchorIn, expected]
            [false, false, 2, true, true], // ← 缺陷场景
            [false, false, 3, true, true],
            [false, true, 2, true, false], // ← 既有功能
            [false, true, 8, true, false],
            [true, false, 2, true, false], // 复制拖拽
            [true, true, 2, true, false],
            [false, false, 0, false, false],
            [false, false, 1, true, false],
            [false, false, 2, false, false],
            [false, true, 2, false, false],
        ];
        for (const [toggle, intentional, size, anchorIn, expected] of table) {
            expect(
                shouldCollapseStaleSelectionOnDrag({
                    multiSelectToggleActive: toggle,
                    selectionIntentional: intentional,
                    multiSelectionSize: size,
                    anchorInMultiSelection: anchorIn,
                }),
                `toggle=${toggle} intentional=${intentional} size=${size} anchorIn=${anchorIn}`,
            ).toBe(expected);
        }
    });

    it("负数集合大小按空集合处理（脏值不抛错）", () => {
        expect(() => stale({ multiSelectionSize: -1 })).not.toThrow();
        expect(stale({ multiSelectionSize: -1 })).toBe(false);
    });
});
