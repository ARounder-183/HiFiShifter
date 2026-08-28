import { test } from "vitest";

import {
    buildReferencePitchStrokeColor,
    cleanupVisibleReferenceRootTrackIds,
    listReferenceRootTracks,
} from "./referenceRootTracks.ts";

test("components/layout/pianoRoll/referenceRootTracks.test.ts scripted checks", async () => {
    function assertDeepEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function assertMatch(actual: string, pattern: RegExp, label: string): void {
        if (!pattern.test(actual)) {
            throw new Error(`${label}: value ${actual} did not match ${pattern}`);
        }
    }

    const tracks = [
        { id: "root-a", name: "Lead", parentId: null, color: "#ff0000" },
        { id: "child-a", name: "Lead Double", parentId: "root-a", color: "#ff0000" },
        { id: "root-b", name: "Harmony", parentId: null, color: "#00ff00" },
        { id: "root-c", name: "Adlib", parentId: null, color: "#0000ff" },
    ];

    assertDeepEqual(
        listReferenceRootTracks({ tracks, currentRootTrackId: "root-a" }),
        [
            { id: "root-b", name: "Harmony", color: "#00ff00" },
            { id: "root-c", name: "Adlib", color: "#0000ff" },
        ],
        "reference root track list excludes current root and child tracks",
    );

    assertDeepEqual(
        cleanupVisibleReferenceRootTrackIds({
            tracks,
            currentRootTrackId: "root-b",
            visibleReferenceRootTrackIds: ["root-a", "child-a", "root-b", "missing", "root-c"],
        }),
        ["root-a", "root-c"],
        "cleanup removes invalid ids, child ids, and current root",
    );

    assertMatch(
        buildReferencePitchStrokeColor("#336699", false),
        /^rgba\(\d+, \d+, \d+, 0\.85\)$/,
        "reference pitch stroke color uses the standard reference alpha",
    );

    assertMatch(
        buildReferencePitchStrokeColor("#336699", true),
        /^rgba\(\d+, \d+, \d+, 1\)$/,
        "highlighted reference pitch stroke color uses full alpha",
    );

});
