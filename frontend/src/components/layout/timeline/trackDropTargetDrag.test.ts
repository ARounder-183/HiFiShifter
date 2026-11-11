import { describe, expect, it } from "vitest";

import { NEW_TRACK_SENTINEL } from "./constants";
import { resolveTrackDropTarget } from "./trackDropTargetDrag";

const PARTICIPANTS = [
    { clipId: "a", trackId: "track-a" },
    { clipId: "b", trackId: "track-b" },
];

describe("resolveTrackDropTarget", () => {
    it("returns every target track when the clip is pinned back to its origin", () => {
        expect(
            resolveTrackDropTarget({
                moves: [
                    { clipId: "a", trackId: "track-b" },
                    { clipId: "b", trackId: "track-c" },
                ],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                originPinned: true,
            }),
        ).toEqual({ trackIds: ["track-b", "track-c"] });
    });

    it("★ does not outline anything when the clip is not pinned to its origin", () => {
        // 跨轨了、但这一刻没被吸回原起点 ⇒ 没有描边。这是「只要吸附就高亮」的回归锁。
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-b" }],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                originPinned: false,
            }),
        ).toBeNull();
    });

    it("does not outline a same-track drag even when pinned", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-a" }],
                participants: PARTICIPANTS,
                dropToNewTrack: false,
                originPinned: true,
            }),
        ).toBeNull();
    });

    it("returns null for the new-track sentinel even when pinned", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: NEW_TRACK_SENTINEL }],
                participants: PARTICIPANTS,
                dropToNewTrack: true,
                originPinned: true,
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
                originPinned: true,
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
                originPinned: true,
            }),
        ).toEqual({ trackIds: ["track-c"] });
    });
});
