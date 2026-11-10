import { describe, expect, it } from "vitest";
import { PARAM_BLOCK_MENU_ITEMS } from "./paramBlockMenuModel";

describe("paramBlockMenuModel shortcut labels", () => {
    it("uses the real block-tool actions for edit commands", () => {
        const action = (key: string) =>
            PARAM_BLOCK_MENU_ITEMS.find((item) => item.key === key)?.shortcutAction;
        expect(action("copy")).toBe("pianoRoll.copy");
        expect(action("cut")).toBe("pianoRoll.cut");
        expect(action("paste")).toBe("pianoRoll.paste");
        expect(action("select-all")).toBe("paramBlock.selectAll");
        expect(action("deselect")).toBe("paramBlock.deselect");
    });
});
