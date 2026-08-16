import { computeTrackMoveBounds } from "./clipDropMoveUtils.ts";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    const actualJson = JSON.stringify(actual);
    const expectedJson = JSON.stringify(expected);
    if (actualJson !== expectedJson) {
        throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
    }
}

const trackIds = ["track-1", "track-2", "track-3", "track-4", "track-5"];
const trackIndexById = Object.fromEntries(trackIds.map((id, idx) => [id, idx]));

function initialById(
    entries: Array<[string, string]>,
): Record<string, { startSec: number; trackId: string }> {
    return Object.fromEntries(
        entries.map(([clipId, trackId]) => [
            clipId,
            {
                startSec: 0,
                trackId,
            },
        ]),
    );
}

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1", "clip-2", "clip-3"],
        initialById: initialById([
            ["clip-1", "track-2"],
            ["clip-2", "track-3"],
            ["clip-3", "track-4"],
        ]),
        trackIndexById,
    }),
    {
        minTrackOffset: -1,
        maxTrackOffset: 1,
    },
    "middle group can move one track in either direction",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1", "clip-2", "clip-3"],
        initialById: initialById([
            ["clip-1", "track-1"],
            ["clip-2", "track-2"],
            ["clip-3", "track-3"],
        ]),
        trackIndexById,
    }),
    {
        minTrackOffset: 0,
        maxTrackOffset: 2,
    },
    "group flush to top cannot move up past the boundary",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1", "clip-2", "clip-3"],
        initialById: initialById([
            ["clip-1", "track-3"],
            ["clip-2", "track-4"],
            ["clip-3", "track-5"],
        ]),
        trackIndexById,
    }),
    {
        minTrackOffset: -2,
        maxTrackOffset: 0,
    },
    "group flush to bottom cannot move down past the boundary",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1", "clip-2"],
        initialById: initialById([
            ["clip-1", "track-1"],
            ["clip-2", "track-5"],
        ]),
        trackIndexById,
    }),
    {
        minTrackOffset: 0,
        maxTrackOffset: 0,
    },
    "group spanning the whole timeline cannot move vertically",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1", "clip-2"],
        initialById: initialById([
            ["clip-1", "track-1"],
            ["clip-2", "track-3"],
        ]),
        trackIndexById,
    }),
    {
        minTrackOffset: 0,
        maxTrackOffset: 2,
    },
    "non-contiguous group uses the outermost selected tracks for bounds",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1"],
        initialById: initialById([["clip-1", "track-3"]]),
        trackIndexById,
    }),
    {
        minTrackOffset: -2,
        maxTrackOffset: 2,
    },
    "single clip can move across the whole timeline",
);

assertEqual(
    computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: ["clip-1"],
        initialById: initialById([["clip-1", "missing-track"]]),
        trackIndexById,
    }),
    null,
    "invalid source track disables track movement",
);

console.log("clip drop move utils checks passed");
