import { describe, expect, it } from "vitest";

import { buildMidiNoteOccurrences } from "./midiNoteGeometry";

const note = (startSec: number, endSec: number, pitch = 60) => ({
    startSec,
    endSec,
    note: pitch,
    velocity: 100,
    channel: 0,
});

describe("buildMidiNoteOccurrences", () => {
    it("projects notes through the non-loop source window", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(1, 2)],
            clipLengthSec: 2,
            playbackRate: 1,
            reversed: false,
            loopEnabled: false,
            sourceStartSec: 1,
            sourceEndSec: 3,
        });
        expect(out).toHaveLength(1);
        expect(out[0]!.startSec).toBeCloseTo(0);
        expect(out[0]!.endSec).toBeCloseTo(1);
    });

    it("mirrors notes inside the non-loop reverse window", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(2, 3)],
            clipLengthSec: 2,
            playbackRate: 1,
            reversed: true,
            loopEnabled: false,
            sourceStartSec: 0,
            sourceEndSec: 4,
        });
        expect(out).toHaveLength(1);
        expect(out[0]!.startSec).toBeCloseTo(1);
        expect(out[0]!.endSec).toBeCloseTo(2);
    });

    it("repeats every note once per loop cycle", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(0, 0.5)],
            clipLengthSec: 3,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            sourceStartSec: 0,
            sourceEndSec: 1,
            loopCycle: { cycleSec: 1, fwdAnchorSec: 0, revAnchorEndSec: 1 },
        });
        expect(out.map((item) => item.startSec)).toEqual([0, 1, 2]);
    });

    it("anchors reverse loop notes at the reverse edge", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(3, 4)],
            clipLengthSec: 5,
            playbackRate: 1,
            reversed: true,
            loopEnabled: true,
            sourceStartSec: 0,
            sourceEndSec: 4,
            loopCycle: { cycleSec: 4, fwdAnchorSec: 0, revAnchorEndSec: 4 },
        });
        expect(out.map((item) => item.startSec)).toEqual([0, 4]);
    });

    it("keeps a note that crosses the cycle boundary continuous", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(0.75, 1.25)],
            clipLengthSec: 3,
            playbackRate: 1,
            reversed: false,
            loopEnabled: true,
            sourceStartSec: 0,
            sourceEndSec: 1,
            loopCycle: { cycleSec: 1, fwdAnchorSec: 0, revAnchorEndSec: 1 },
        });
        expect(out[0]!.startSec).toBeCloseTo(0.75);
        expect(out[0]!.endSec).toBeCloseTo(1.25);
        expect(out[1]!.startSec).toBeCloseTo(1.75);
    });

    it("scales cycle and note positions by playback rate", () => {
        const out = buildMidiNoteOccurrences({
            notes: [note(0, 0.5)],
            clipLengthSec: 2,
            playbackRate: 2,
            reversed: false,
            loopEnabled: true,
            sourceStartSec: 0,
            sourceEndSec: 1,
            loopCycle: { cycleSec: 1, fwdAnchorSec: 0, revAnchorEndSec: 1 },
        });
        expect(out.map((item) => item.startSec)).toEqual([0, 0.5, 1, 1.5]);
    });
});
