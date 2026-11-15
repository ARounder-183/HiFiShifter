import { describe, expect, it } from "vitest";
import type { KeybindingMap } from "../../../features/keybindings/types";
import {
    resolveParamBlockCreateShape,
    resolveParamBlockModifierSnapshot,
} from "./paramBlockModifierPolicy";

const keybindings = {
    "modifier.paramBlockCreate": [{ key: "a", modifierOnly: true, alt: true }],
    "modifier.paramBlockCreateFlat": [{ key: "a", modifierOnly: true, alt: true, ctrl: true }],
    "modifier.paramBlockSplitAtClick": [{ key: "a", modifierOnly: true, alt: true }],
    "modifier.clipNoSnap": [{ key: "a", modifierOnly: true, shift: true }],
} as unknown as KeybindingMap;

const event = (
    overrides: Partial<{
        ctrlKey: boolean;
        shiftKey: boolean;
        altKey: boolean;
        metaKey: boolean;
    }> = {},
) => ({
    ctrlKey: false,
    shiftKey: false,
    altKey: false,
    metaKey: false,
    ...overrides,
});

describe("paramBlockModifierPolicy", () => {
    it("switches normal -> flat when Ctrl is added to Alt, and back when released", () => {
        const normal = resolveParamBlockModifierSnapshot({
            event: event({ altKey: true }),
            keybindings,
        });
        expect(normal).toMatchObject({ createBlock: true, createFlat: false });
        let shape = resolveParamBlockCreateShape({ previous: "normal", snapshot: normal });
        expect(shape).toBe("normal");
        const flat = resolveParamBlockModifierSnapshot({
            event: event({ altKey: true, ctrlKey: true }),
            keybindings,
        });
        shape = resolveParamBlockCreateShape({ previous: shape, snapshot: flat });
        expect(shape).toBe("flat");
        const released = resolveParamBlockModifierSnapshot({
            event: event({ altKey: true }),
            keybindings,
        });
        shape = resolveParamBlockCreateShape({ previous: shape, snapshot: released });
        expect(shape).toBe("normal");
    });

    it("keeps the last valid shape when all create modifiers are released", () => {
        const none = resolveParamBlockModifierSnapshot({ event: event(), keybindings });
        expect(resolveParamBlockCreateShape({ previous: "flat", snapshot: none })).toBe("flat");
    });

    it("honors custom modifier combinations and gives flat priority", () => {
        const custom = {
            "modifier.paramBlockCreate": [{ key: "a", modifierOnly: true, shift: true }],
            "modifier.paramBlockCreateFlat": [
                { key: "a", modifierOnly: true, shift: true, ctrl: true },
            ],
            "modifier.paramBlockSplitAtClick": [{ key: "a", modifierOnly: true, alt: true }],
            "modifier.clipNoSnap": [{ key: "a", modifierOnly: true, meta: true }],
        } as unknown as KeybindingMap;
        const snapshot = resolveParamBlockModifierSnapshot({
            event: event({ shiftKey: true, ctrlKey: true }),
            keybindings: custom,
        });
        expect(snapshot.createBlock).toBe(true);
        expect(snapshot.createFlat).toBe(true);
        expect(resolveParamBlockCreateShape({ previous: "normal", snapshot })).toBe("flat");
    });
});
