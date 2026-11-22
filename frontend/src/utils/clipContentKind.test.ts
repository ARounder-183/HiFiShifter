import { describe, expect, it } from "vitest";

import { resolveClipContentKind } from "./clipContentKind";

describe("resolveClipContentKind", () => {
    it("prefers the explicit role over legacy note counts", () => {
        expect(
            resolveClipContentKind({
                midiRole: "midi_take",
                midiNoteCount: 4,
                midiTakeNoteCount: 4,
            }),
        ).toBe("midi_take");
        expect(
            resolveClipContentKind({
                midiRole: "pitch_reference",
                midiNoteCount: 4,
                midiTakeNoteCount: 4,
            }),
        ).toBe("pitch_reference");
    });

    it("migrates legacy note-only payloads to pitch reference", () => {
        expect(resolveClipContentKind({ midiNoteCount: 4 })).toBe("pitch_reference");
    });

    it("recognizes new MIDI Take payloads without a role", () => {
        expect(resolveClipContentKind({ midiTakeNoteCount: 4 })).toBe("midi_take");
    });

    it("treats an empty clip as audio", () => {
        expect(resolveClipContentKind({})).toBe("audio");
    });
});
