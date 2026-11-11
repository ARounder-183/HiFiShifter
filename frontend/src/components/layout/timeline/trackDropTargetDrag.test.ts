import { describe, expect, it } from "vitest";

import { NEW_TRACK_SENTINEL } from "./constants";
import { resolveTrackDropTarget } from "./trackDropTargetDrag";

describe("resolveTrackDropTarget", () => {
    it("returns every target track for a cross-track move", () => {
        expect(
            resolveTrackDropTarget({
                moves: [
                    { clipId: "a", trackId: "track-b" },
                    { clipId: "b", trackId: "track-c" },
                ],
                participants: [
                    { clipId: "a", trackId: "track-a" },
                    { clipId: "b", trackId: "track-b" },
                ],
                dropToNewTrack: false,
                snapActive: false,
                copyMode: false,
            }),
        ).toEqual({ trackIds: ["track-b", "track-c"], tone: "move" });
    });

    it("does not outline a same-track drag", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-a" }],
                participants: [
                    { clipId: "a", trackId: "track-a" },
                    { clipId: "b", trackId: "track-b" },
                ],
                dropToNewTrack: false,
                snapActive: false,
                copyMode: false,
            }),
        ).toBeNull();
    });

    it("uses copy and snap tones", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-b" }],
                participants: [
                    { clipId: "a", trackId: "track-a" },
                    { clipId: "b", trackId: "track-b" },
                ],
                dropToNewTrack: false,
                snapActive: false,
                copyMode: true,
            })?.tone,
        ).toBe("copy");
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: "track-b" }],
                participants: [
                    { clipId: "a", trackId: "track-a" },
                    { clipId: "b", trackId: "track-b" },
                ],
                dropToNewTrack: false,
                snapActive: true,
                copyMode: true,
            })?.tone,
        ).toBe("snap");
    });

    it("returns null for the new-track sentinel", () => {
        expect(
            resolveTrackDropTarget({
                moves: [{ clipId: "a", trackId: NEW_TRACK_SENTINEL }],
                participants: [
                    { clipId: "a", trackId: "track-a" },
                    { clipId: "b", trackId: "track-b" },
                ],
                dropToNewTrack: true,
                snapActive: false,
                copyMode: false,
            }),
        ).toBeNull();
    });
});
