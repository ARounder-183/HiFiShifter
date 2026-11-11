import { describe, expect, it } from "vitest";
import {
    PARAM_BLOCK_MENU_ITEMS,
    type ParamBlockMenuContext,
    type ParamBlockMenuItem,
} from "./paramBlockMenuModel";
import {
    ACTION_TO_EDIT_OP,
    PARAM_BLOCK_ACTION_TO_OP,
} from "../../../features/keybindings/focusRouting";

function context(overrides: Partial<ParamBlockMenuContext> = {}): ParamBlockMenuContext {
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
        isVolume: false,
        isDyn: false,
        ...overrides,
    };
}

const visible = (ctx: ParamBlockMenuContext): ParamBlockMenuItem[] =>
    PARAM_BLOCK_MENU_ITEMS.filter((item) => item.visible?.(ctx) ?? true);

const visibleKeys = (ctx: ParamBlockMenuContext): string[] =>
    visible(ctx).map((item) => item.key);

const item = (key: string): ParamBlockMenuItem => {
    const found = PARAM_BLOCK_MENU_ITEMS.find((candidate) => candidate.key === key);
    if (found === undefined) throw new Error("missing menu item " + key);
    return found;
};

describe("paramBlockMenuModel layout", () => {
    it("keeps the flat part at 10 items or fewer for a typical block selection", () => {
        const flat = visible(context()).filter(
            (entry) =>
                entry.key !== "delete" &&
                (entry.group === "block" ||
                    entry.group === "selection" ||
                    entry.group === "clipboard"),
        );
        expect(flat.map((entry) => entry.key)).toEqual([
            "split",
            "create-from-selection",
            "group",
            "normalize-pitch",
            "select-all",
            "deselect",
            "copy",
            "cut",
            "paste",
        ]);
        expect(flat.length).toBeLessThanOrEqual(10);
    });

    it("moves transform and file IO commands into the submenu groups", () => {
        const groups = new Set(
            PARAM_BLOCK_MENU_ITEMS.filter((entry) =>
                ["initialize", "transpose-cents", "save-as-pitch-ref", "export-midi"].includes(
                    entry.key,
                ),
            ).map((entry) => entry.group),
        );
        expect(groups).toEqual(new Set(["transform", "io"]));
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
                hasSelection: false,
            }),
        );
        expect(keys).not.toContain("split");
        expect(keys).not.toContain("create-from-selection");
        expect(keys).not.toContain("group");
        expect(keys).not.toContain("ungroup");
        expect(keys).not.toContain("delete");
        expect(keys).not.toContain("deselect");
    });

    it("hides pitch-only and parameter-conversion commands for other params", () => {
        const keys = visibleKeys(context({ isPitch: false }));
        expect(keys).not.toContain("transpose-cents");
        expect(keys).not.toContain("transpose-degrees");
        expect(keys).not.toContain("add-vibrato");
        expect(keys).not.toContain("save-as-pitch-ref");
        expect(keys).not.toContain("export-midi");
        expect(keys).toContain("set-pitch");

        const volumeKeys = visibleKeys(context({ isPitch: false, isVolume: true }));
        expect(volumeKeys).toContain("convert-volume-to-dyn");
        expect(volumeKeys).not.toContain("convert-dyn-to-volume");

        const dynKeys = visibleKeys(context({ isPitch: false, isDyn: true }));
        expect(dynKeys).toContain("convert-dyn-to-volume");
        expect(dynKeys).not.toContain("convert-volume-to-dyn");
    });
});

describe("paramBlockMenuModel shortcut labels", () => {
    it("uses the real block-tool actions for edit commands", () => {
        expect(item("copy").shortcutAction).toBe("pianoRoll.copy");
        expect(item("cut").shortcutAction).toBe("pianoRoll.cut");
        expect(item("paste").shortcutAction).toBe("pianoRoll.paste");
        expect(item("select-all").shortcutAction).toBe("paramBlock.selectAll");
        expect(item("deselect").shortcutAction).toBe("paramBlock.deselect");
    });

    it("maps every edit dialog command to its real edit.* action", () => {
        const expected = {
            initialize: "edit.initialize",
            "transpose-cents": "edit.transposeCents",
            "transpose-degrees": "edit.transposeDegrees",
            "set-pitch": "edit.setPitch",
            average: "edit.average",
            smooth: "edit.smooth",
            "add-vibrato": "edit.addVibrato",
            quantize: "edit.quantize",
            "mean-quantize": "edit.meanQuantize",
        } as const;
        for (const [key, action] of Object.entries(expected)) {
            expect(item(key).shortcutAction).toBe(action);
        }
    });

    it("maps every routed param-block action to a menu item", () => {
        for (const [action, op] of Object.entries(PARAM_BLOCK_ACTION_TO_OP)) {
            const match = PARAM_BLOCK_MENU_ITEMS.find((entry) => entry.op === op);
            expect(match, action + " -> " + op).toBeDefined();
            expect(match?.shortcutAction).toBe(action);
        }
    });

    it("maps the routed edit ops used by this menu", () => {
        for (const [action, op] of Object.entries(ACTION_TO_EDIT_OP)) {
            if (!action.startsWith("pianoRoll.")) continue;
            const match = PARAM_BLOCK_MENU_ITEMS.find(
                (entry) => entry.channel === "edit" && entry.op === op,
            );
            expect(match, action + " -> " + op).toBeDefined();
            expect(match?.shortcutAction).toBe(action);
        }
    });

    it("uses the pitch or generic label for the shared set-value command", () => {
        const entry = item("set-pitch");
        expect(typeof entry.labelKey).toBe("function");
        if (typeof entry.labelKey === "function") {
            expect(entry.labelKey(context())).toBe("menu_set_pitch");
            expect(entry.labelKey(context({ isPitch: false }))).toBe("menu_set_value");
        }
    });
});
