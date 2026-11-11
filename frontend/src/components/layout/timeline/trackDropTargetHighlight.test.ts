import { afterEach, describe, expect, it, vi } from "vitest";

import {
    clearTrackDropTarget,
    getTrackDropTargetSnapshot,
    publishTrackDropTarget,
    subscribeTrackDropTarget,
} from "./trackDropTargetHighlight";

afterEach(() => clearTrackDropTarget());

describe("trackDropTargetHighlight", () => {
    it("stores a deduplicated, sorted set of target tracks", () => {
        const listener = vi.fn();
        const unsubscribe = subscribeTrackDropTarget(listener);
        publishTrackDropTarget({ trackIds: ["track-b", "track-a", "track-b"], tone: "move" });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual(["track-a", "track-b"]);
        expect(getTrackDropTargetSnapshot().tone).toBe("move");
        expect(listener).toHaveBeenCalledTimes(1);
        unsubscribe();
    });

    it("does not notify when the same tracks and tone are republished", () => {
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "copy" });
        const rev = getTrackDropTargetSnapshot().rev;
        const listener = vi.fn();
        const unsubscribe = subscribeTrackDropTarget(listener);
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "copy" });
        expect(getTrackDropTargetSnapshot().rev).toBe(rev);
        expect(listener).not.toHaveBeenCalled();
        unsubscribe();
    });

    it("changes tone without changing the track set", () => {
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "move" });
        const rev = getTrackDropTargetSnapshot().rev;
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "snap" });
        expect(getTrackDropTargetSnapshot().tone).toBe("snap");
        expect(getTrackDropTargetSnapshot().rev).toBe(rev + 1);
    });

    it("publishing an empty set clears the ring", () => {
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "move" });
        publishTrackDropTarget({ trackIds: [], tone: "move" });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual([]);
    });

    it("clear is idempotent and does not notify twice", () => {
        publishTrackDropTarget({ trackIds: ["track-a"], tone: "move" });
        clearTrackDropTarget();
        const rev = getTrackDropTargetSnapshot().rev;
        const listener = vi.fn();
        const unsubscribe = subscribeTrackDropTarget(listener);
        clearTrackDropTarget();
        expect(getTrackDropTargetSnapshot().rev).toBe(rev);
        expect(listener).not.toHaveBeenCalled();
        unsubscribe();
    });

    it("ignores empty and non-string ids", () => {
        publishTrackDropTarget({
            trackIds: ["", "track-a", null as unknown as string],
            tone: "move",
        });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual(["track-a"]);
    });
});
