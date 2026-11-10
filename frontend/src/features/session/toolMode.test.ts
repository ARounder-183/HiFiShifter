import { describe, expect, it } from "vitest";
import { toggleParamBlockEditingTool } from "./toolMode";

describe("toggleParamBlockEditingTool", () => {
    it("returns to the previous draw tool", () => {
        expect(toggleParamBlockEditingTool("block", "draw")).toBe("draw");
        expect(toggleParamBlockEditingTool("block", "line")).toBe("line");
        expect(toggleParamBlockEditingTool("block", "vibrato")).toBe("vibrato");
        expect(toggleParamBlockEditingTool("draw", "line")).toBe("block");
        expect(toggleParamBlockEditingTool("select", "vibrato")).toBe("block");
    });
});
