import { describe, expect, it } from "vitest";
import { toggleParamBlockEditingTool } from "./paramBlockMarquee";

describe("toggleParamBlockEditingTool", () => {
    it("toggles between the block tool and the previous draw tool", () => {
        expect(toggleParamBlockEditingTool("block")).toBe("draw");
        expect(toggleParamBlockEditingTool("draw")).toBe("block");
        expect(toggleParamBlockEditingTool("select")).toBe("block");
    });
});
