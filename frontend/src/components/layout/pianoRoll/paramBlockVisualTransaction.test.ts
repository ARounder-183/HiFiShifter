import { describe, expect, it } from "vitest";
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import {
    advanceParamBlockVisualTransaction,
    paramBlockSpecsMatchGeometry,
    sameParamBlockEpoch,
    shouldUseParamBlockVisualTransaction,
    type ParamBlockEpoch,
    type ParamBlockVisualTransaction,
} from "./paramBlockVisualTransaction";

const epoch: ParamBlockEpoch = {
    rootTrackId: "root",
    layoutKey: "layout-a",
    editRevision: 3,
    historyKey: "0:0",
    framePeriodMs: 5,
};

function spec(startSec = 1, endSec = 2, center = 60): PianoRollParamBlockSpec {
    return {
        param: "pitch",
        valueMin: 0,
        valueMax: 127,
        blocks: [
            {
                id: "a",
                startSec,
                endSec,
                averageValue: center,
                frameCenterValue: center,
                selected: true,
                grouped: false,
            },
        ],
        borderRgba: [0, 0, 0, 1],
        selectedRgba: [1, 1, 1, 1],
        groupedRgba: [1, 0, 0, 1],
    };
}

function tx(overrides: Partial<ParamBlockVisualTransaction> = {}): ParamBlockVisualTransaction {
    return {
        id: 1,
        epoch,
        kind: "resize",
        spec: spec(),
        phase: "awaiting-authority",
        startedAt: 0,
        ...overrides,
    };
}

describe("paramBlockVisualTransaction", () => {
    it("rejects a transaction from another epoch", () => {
        expect(
            shouldUseParamBlockVisualTransaction(tx(), { ...epoch, layoutKey: "layout-b" }),
        ).toBe(false);
        expect(sameParamBlockEpoch(epoch, { ...epoch })).toBe(true);
    });

    it("releases only when canonical geometry matches and authority advanced", () => {
        expect(
            advanceParamBlockVisualTransaction({
                tx: tx(),
                now: 10,
                canonicalSpec: spec(),
                currentEpoch: epoch,
                authorityRevision: 2,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: true,
            }),
        ).toBe("release");
        expect(
            advanceParamBlockVisualTransaction({
                tx: tx(),
                now: 10,
                canonicalSpec: spec(1, 2, 61),
                currentEpoch: epoch,
                authorityRevision: 2,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: true,
            }),
        ).toBe("keep");
    });

    it("expires a waiting transaction after the hard timeout", () => {
        expect(
            advanceParamBlockVisualTransaction({
                tx: tx(),
                now: 1201,
                canonicalSpec: null,
                currentEpoch: epoch,
                authorityRevision: 1,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: false,
            }),
        ).toBe("expire");
    });

    it("does not expire an active gesture, but epoch mismatch releases immediately", () => {
        expect(
            advanceParamBlockVisualTransaction({
                tx: tx({ phase: "gesture" }),
                now: 5000,
                canonicalSpec: null,
                currentEpoch: epoch,
                authorityRevision: 1,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: false,
            }),
        ).toBe("keep");
        expect(
            advanceParamBlockVisualTransaction({
                tx: tx({ phase: "gesture" }),
                now: 5000,
                canonicalSpec: null,
                currentEpoch: { ...epoch, historyKey: "1:0" },
                authorityRevision: 1,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: false,
            }),
        ).toBe("release");
    });

    it("compares only observable geometry, not presentation fields", () => {
        const left = spec();
        const right = spec();
        (right.blocks[0] as { selected: boolean }).selected = false;
        expect(paramBlockSpecsMatchGeometry(left, right)).toBe(true);
        expect(paramBlockSpecsMatchGeometry(left, spec(1, 2, 61))).toBe(false);
    });
});
