import { describe, expect, it } from "vitest";
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import {
    advanceParamBlockVisualTransaction,
    isParamBlockVisualEpochCurrent,
    paramBlockSpecsMatchGeometry,
    sameParamBlockEpoch,
    sameParamBlockEpochLayout,
    shouldUseParamBlockVisualTransaction,
    type ParamBlockEpoch,
    type ParamBlockVisualTransaction,
} from "./paramBlockVisualTransaction";

const epoch: ParamBlockEpoch = {
    rootTrackId: "root",
    param: "pitch",
    layoutKey: "layout-a",
    editRevision: 3,
    externalHistoryToken: "0:0",
    commitToken: "0",
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
        ownedEditRevision: null,
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
                currentEpoch: { ...epoch, externalHistoryToken: "1:0" },
                authorityRevision: 1,
                authorityRevisionAtStart: 1,
                committedSegmentsPresent: false,
            }),
        ).toBe("release");
    });

    it("keeps drag layout valid across a self commit but not an external jump", () => {
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, commitToken: "1" })).toBe(true);
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, externalHistoryToken: "1:0" })).toBe(
            false,
        );
    });

    it("rejects a transaction when the parameter changes", () => {
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, param: "volume" })).toBe(false);
        expect(
            shouldUseParamBlockVisualTransaction(tx(), { ...epoch, param: "volume" }),
        ).toBe(false);
    });

    it("accepts base and owned edit revisions, but not a third-party edit", () => {
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: epoch,
                ownedEditRevision: null,
            }),
        ).toBe(true);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch, editRevision: 4 },
                ownedEditRevision: 4,
            }),
        ).toBe(true);
        expect(
            isParamBlockVisualEpochCurrent({
                epoch,
                current: { ...epoch, editRevision: 5 },
                ownedEditRevision: 4,
            }),
        ).toBe(false);
    });

    it("compares only observable geometry, not presentation fields", () => {
        const left = spec();
        const right = spec();
        (right.blocks[0] as { selected: boolean }).selected = false;
        expect(paramBlockSpecsMatchGeometry(left, right)).toBe(true);
        expect(paramBlockSpecsMatchGeometry(left, spec(1, 2, 61))).toBe(false);
    });
});
