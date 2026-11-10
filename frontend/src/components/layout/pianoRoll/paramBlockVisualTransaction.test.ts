import { describe, expect, it } from "vitest";
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";
import {
    isParamBlockVisualEpochCurrent,
    sameParamBlockEpochLayout,
    sameParamBlockExternalEpoch,
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

function spec(): PianoRollParamBlockSpec {
    return {
        param: "pitch",
        valueMin: 0,
        valueMax: 127,
        blocks: [],
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
    it("rejects a transaction from another layout or parameter", () => {
        expect(
            shouldUseParamBlockVisualTransaction(tx(), { ...epoch, layoutKey: "layout-b" }),
        ).toBe(false);
        expect(shouldUseParamBlockVisualTransaction(tx(), { ...epoch, param: "volume" })).toBe(
            false,
        );
        expect(shouldUseParamBlockVisualTransaction(tx(), epoch)).toBe(true);
    });

    it("keeps drag layout valid across a self commit but not an external jump", () => {
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, commitToken: "1" })).toBe(true);
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, editRevision: 4 })).toBe(true);
        expect(sameParamBlockEpochLayout(epoch, { ...epoch, externalHistoryToken: "1:0" })).toBe(
            false,
        );
        expect(sameParamBlockExternalEpoch(epoch, { ...epoch, param: "volume" })).toBe(false);
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
});
