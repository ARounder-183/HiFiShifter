import { describe, expect, it } from "vitest";
import {
    isBlockSelectionActive,
    isRangeToolActive,
    resolveToolScope,
    toggleSelectToolMode,
} from "./toolMode";

describe("tool family derivation", () => {
    it("toggles only the selection submode", () => {
        expect(toggleSelectToolMode("range")).toBe("block");
        expect(toggleSelectToolMode("block")).toBe("range");
    });

    it("activates block only when the selection family owns the interaction", () => {
        expect(isBlockSelectionActive("select", "block")).toBe(true);
        expect(isBlockSelectionActive("select", "range")).toBe(false);
        expect(isBlockSelectionActive("draw", "block")).toBe(false);
        expect(isBlockSelectionActive("line", "block")).toBe(false);
    });

    /**
     * 四点编辑控件、可拖选区光标、选区边缘拉伸提示只属于 range 子模式。
     * 块模式继承工具族身份，但**不继承**这些 affordance —— Alt 在块模式里是
     * 「新建块 / 在点击处分割」，按住它不该弹出四个控制点。
     */
    it("keeps the range affordances out of block mode", () => {
        expect(isRangeToolActive("select", "range")).toBe(true);
        expect(isRangeToolActive("select", "block")).toBe(false);
        expect(isRangeToolActive("draw", "range")).toBe(false);
        expect(isRangeToolActive("draw", "block")).toBe(false);
        expect(isRangeToolActive("line", "range")).toBe(false);
        expect(isRangeToolActive("vibrato", "block")).toBe(false);
    });

    it("is the exact complement of block selection inside the selection family", () => {
        for (const mode of ["select", "draw", "line", "vibrato"] as const) {
            for (const sub of ["range", "block"] as const) {
                expect(isRangeToolActive(mode, sub)).toBe(
                    mode === "select" && !isBlockSelectionActive(mode, sub),
                );
            }
        }
    });

    it("folds the selection submode into the command-routing scope", () => {
        expect(resolveToolScope("select", "block")).toBe("block");
        expect(resolveToolScope("select", "range")).toBe("select");
        expect(resolveToolScope("line", "block")).toBe("line");
        expect(resolveToolScope("vibrato", "range")).toBe("vibrato");
    });
});
