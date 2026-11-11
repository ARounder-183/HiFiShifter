import { describe, expect, it } from "vitest";

import { NEW_TRACK_SENTINEL } from "./constants";
import { resolveTrackDropTarget } from "./trackDropTargetDrag";

const PARTICIPANTS = [
    { clipId: "a", trackId: "track-a" },
    { clipId: "b", trackId: "track-b" },
];

describe("resolveTrackDropTarget", () => {
    it("returns every target track for a snapped cross-track move", () => {
        expect(
            resolveTrackDropTarget({
                moves: [
                    { clipId: "a", trackId: "track-b" },
                    { clipId: "b", trackId: "track-c" },
                ],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                snapped: true,
            }),
        ).toEqual({ trackIds: ["track-b", "track-c"] });
    });

    it("★ does not outline anything while no snap is engaged", () => {
        // 跨轨了、但这一刻没有吸附命中 ⇒ 没有描边。这是「未吸附也高亮」的回归锁。
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-b" }],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                snapped: false,
            }),
        ).toBeNull();
    });

    it("does not outline a same-track drag even when snapped", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-a" }],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                snapped: true,
            }),
        ).toBeNull();
    });

    it("returns null for the new-track sentinel even when snapped", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: NEW_TRACK_SENTINEL }],
                participants: PARTICIPANTS,
                dropToNewTrack: true,
                snapped: true,
            }),
        ).toBeNull();
    });

    it("drops the sentinel id from a mixed selection", () => {
        expect(
            resolveTrackDropTarget({
                moves: [
                    { clipId: "a", trackId: "track-c" },
                    { clipId: "b", trackId: NEW_TRACK_SENTINEL },
                ],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                snapped: true,
            }),
        ).toEqual({ trackIds: ["track-c"] });
    });

    it("deduplicates target tracks", () => {
        expect(
            resolveTrackDropTarget({
                moves: [
                    { clipId: "a", trackId: "track-c" },
                    { clipId: "b", trackId: "track-c" },
                ],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                snapped: true,
            }),
        ).toEqual({ trackIds: ["track-c"] });
    });
});
