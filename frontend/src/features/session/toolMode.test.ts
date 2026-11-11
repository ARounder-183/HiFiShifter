import { describe, expect, it } from "vitest";
import { isBlockSelectionActive, resolveToolScope, toggleSelectToolMode } from "./toolMode";

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

    it("folds the selection submode into the command-routing scope", () => {
        expect(resolveToolScope("select", "block")).toBe("block");
        expect(resolveToolScope("select", "range")).toBe("select");
        expect(resolveToolScope("line", "block")).toBe("line");
        expect(resolveToolScope("vibrato", "range")).toBe("vibrato");
    });
});
