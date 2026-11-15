import { describe, expect, it } from "vitest";
import {
    PARAM_BLOCK_COMMANDS,
    type ParamBlockCommandContext,
    type ParamBlockCommandItem,
} from "./paramBlockCommandMenu";
import { PARAM_BLOCK_ACTION_TO_OP } from "../../../features/keybindings/focusRouting";

function context(overrides: Partial<ParamBlockCommandContext> = {}): ParamBlockCommandContext {
    return {
        canSplit: () => true,
        canGroup: true,
        canUngroup: false,
        canDelete: true,
        canRestoreGame: false,
        canRestoreDeleted: false,
        canNormalizePitch: true,
        hasSelection: true,
        isPitch: true,
        ...overrides,
    };
}

const visible = (ctx: ParamBlockCommandContext): ParamBlockCommandItem[] =>
    PARAM_BLOCK_COMMANDS.filter((item) => item.visible?.(ctx) ?? true);

const visibleKeys = (ctx: ParamBlockCommandContext): string[] =>
    visible(ctx).map((item) => item.key);

const item = (key: string): ParamBlockCommandItem => {
    const found = PARAM_BLOCK_COMMANDS.find((candidate) => candidate.key === key);
    if (found === undefined) throw new Error("missing menu item " + key);
    return found;
};

describe("paramBlockCommandMenu", () => {
    it("only describes the eight block-specific commands", () => {
        expect(PARAM_BLOCK_COMMANDS.map((entry) => entry.key)).toEqual([
            "split",
            "create-from-selection",
            "group",
            "ungroup",
            "restore-game",
            "restore-deleted",
            "normalize-pitch",
            "delete",
        ]);
    });

    it("maps every routed block action to the same op", () => {
        for (const entry of PARAM_BLOCK_COMMANDS) {
            if (entry.shortcutAction === undefined) continue;
            expect(PARAM_BLOCK_ACTION_TO_OP[entry.shortcutAction]).toBe(entry.op);
        }
    });

    it("hides unavailable block commands instead of showing them disabled", () => {
        const keys = visibleKeys(
            context({
                canSplit: () => false,
                canGroup: false,
                canUngroup: false,
                canDelete: false,
                canRestoreGame: false,
                canRestoreDeleted: false,
                canNormalizePitch: false,
                hasSelection: false,
            }),
        );
        expect(keys).toEqual([]);
    });

    it("shows normalize-pitch only for a pitch parameter with exactly one selected block", () => {
        expect(visibleKeys(context())).toContain("normalize-pitch");
        expect(visibleKeys(context({ isPitch: false }))).not.toContain("normalize-pitch");
        expect(visibleKeys(context({ canNormalizePitch: false }))).not.toContain("normalize-pitch");
    });

    it("keeps delete destructive and last", () => {
        const deleteItem = item("delete");
        expect(deleteItem.danger).toBe(true);
        expect(deleteItem.separatorBefore).toBe(true);
        expect(PARAM_BLOCK_COMMANDS.at(-1)?.key).toBe("delete");
    });
});
