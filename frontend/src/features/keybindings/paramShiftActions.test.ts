import { describe, expect, it } from "vitest";

import { resolveParamShiftIntent } from "./paramShiftActions";
import type { ActionId } from "./types";

describe("paramShiftActions — 参数线平移快捷键解析", () => {
    it("音频块范围：方向与幅度档位一一对应（方向不能被 Large/Small 后缀干扰）", () => {
        // 回归：id 以 Large/Small 结尾，endsWith("Down") 会把「大幅下移」
        // 误判为上移 —— 方向判定必须用 includes("Down")。
        const cases: Array<[ActionId, boolean, "coarse" | "fine" | "normal"]> = [
            ["pianoRoll.shiftParamUp", true, "normal"],
            ["pianoRoll.shiftParamDown", false, "normal"],
            ["pianoRoll.shiftParamUpLarge", true, "coarse"],
            ["pianoRoll.shiftParamDownLarge", false, "coarse"],
            ["pianoRoll.shiftParamUpSmall", true, "fine"],
            ["pianoRoll.shiftParamDownSmall", false, "fine"],
        ];
        for (const [id, isUp, magnitude] of cases) {
            const intent = resolveParamShiftIntent(id);
            expect(intent).not.toBeNull();
            expect(intent?.isUp).toBe(isUp);
            expect(intent?.magnitude).toBe(magnitude);
            expect(intent?.selectionOp).toBeNull();
        }
    });

    it("选择范围：同样解析方向与幅度，并给出 hifi:editOp 的 op 名", () => {
        const cases: Array<
            [
                ActionId,
                boolean,
                "coarse" | "fine" | "normal",
                "shiftParamUpSelection" | "shiftParamDownSelection",
            ]
        > = [
            ["pianoRoll.shiftParamUpSelection", true, "normal", "shiftParamUpSelection"],
            ["pianoRoll.shiftParamDownSelection", false, "normal", "shiftParamDownSelection"],
            ["pianoRoll.shiftParamUpSelectionLarge", true, "coarse", "shiftParamUpSelection"],
            ["pianoRoll.shiftParamDownSelectionLarge", false, "coarse", "shiftParamDownSelection"],
            ["pianoRoll.shiftParamUpSelectionSmall", true, "fine", "shiftParamUpSelection"],
            ["pianoRoll.shiftParamDownSelectionSmall", false, "fine", "shiftParamDownSelection"],
        ];
        for (const [id, isUp, magnitude, op] of cases) {
            const intent = resolveParamShiftIntent(id);
            expect(intent).not.toBeNull();
            expect(intent?.isUp).toBe(isUp);
            expect(intent?.magnitude).toBe(magnitude);
            expect(intent?.selectionOp).toBe(op);
        }
    });

    it("非平移类 actionId 返回 null", () => {
        expect(resolveParamShiftIntent("clip.paste")).toBeNull();
        expect(resolveParamShiftIntent("edit.undo")).toBeNull();
        expect(resolveParamShiftIntent("pianoRoll.copy")).toBeNull();
    });
});
