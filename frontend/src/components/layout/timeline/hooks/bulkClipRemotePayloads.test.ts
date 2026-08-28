import { test } from "vitest";

import {
    buildBulkClipStateUpdates,
    buildDuplicateClipsBulkPayload,
} from "./bulkClipRemotePayloads.js";

test("components/layout/timeline/hooks/bulkClipRemotePayloads.test.ts scripted checks", async () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    const updates = buildBulkClipStateUpdates({
        clipIds: ["a", "b"],
        changesById: new Map([
            ["a", { gain: 1.2 }],
            ["b", { muted: true, fadeInSec: 0.4 }],
        ]),
    });

    assertDeepEqual(
        updates,
        [
            { clipId: "a", gain: 1.2 },
            { clipId: "b", muted: true, fadeInSec: 0.4 },
        ],
        "bulk state payload",
    );

    const duplicatePayload = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["a", "b"],
        deltaSec: 1.5,
        copyLinkedParams: true,
        applyAutoCrossfade: true,
        trackMode: { kind: "offset_tracks", offset: 1 },
    });

    assertDeepEqual(
        duplicatePayload,
        {
            sourceClipIds: ["a", "b"],
            deltaSec: 1.5,
            copyLinkedParams: true,
            applyAutoCrossfade: true,
            selectCreatedClips: true,
            trackMode: { kind: "offset_tracks", offset: 1 },
        },
        "bulk duplicate payload",
    );

    const stateUpdates = buildBulkClipStateUpdates({
        clipIds: ["clip-a"],
        changesById: new Map([["clip-a", { muted: false }]]),
    });

    assertDeepEqual(stateUpdates, [{ clipId: "clip-a", muted: false }], "single state update");

    const duplicatePayloadForPaste = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["clip-1"],
        deltaSec: 0,
        copyLinkedParams: true,
        applyAutoCrossfade: false,
        trackMode: { kind: "same_track" },
    });

    assertDeepEqual(
        duplicatePayloadForPaste.trackMode,
        { kind: "same_track" },
        "paste can reuse duplicate bulk payload",
    );

    const duplicatePayloadWithoutRename = buildDuplicateClipsBulkPayload({
        sourceClipIds: ["clip-2"],
        deltaSec: 2,
        copyLinkedParams: true,
        applyAutoCrossfade: false,
        trackMode: { kind: "explicit_mapping", mapping: { t1: "t2" } },
        renameCopies: false,
    });

    assertDeepEqual(
        duplicatePayloadWithoutRename.renameCopies,
        false,
        "duplicate payload can preserve source names",
    );
});
