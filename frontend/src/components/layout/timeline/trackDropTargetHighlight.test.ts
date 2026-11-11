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
        publishTrackDropTarget({ trackIds: ["track-b", "track-a", "track-b"] });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual(["track-a", "track-b"]);
        expect(listener).toHaveBeenCalledTimes(1);
        unsubscribe();
    });

    it("does not notify when the same tracks are republished", () => {
        publishTrackDropTarget({ trackIds: ["track-a"] });
        const rev = getTrackDropTargetSnapshot().rev;
        const listener = vi.fn();
        const unsubscribe = subscribeTrackDropTarget(listener);
        publishTrackDropTarget({ trackIds: ["track-a"] });
        expect(getTrackDropTargetSnapshot().rev).toBe(rev);
        expect(listener).not.toHaveBeenCalled();
        unsubscribe();
    });

    it("notifies when the track set changes", () => {
        publishTrackDropTarget({ trackIds: ["track-a"] });
        const rev = getTrackDropTargetSnapshot().rev;
        publishTrackDropTarget({ trackIds: ["track-a", "track-b"] });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual(["track-a", "track-b"]);
        expect(getTrackDropTargetSnapshot().rev).toBe(rev + 1);
    });

    it("publishing an empty set clears the ring", () => {
        publishTrackDropTarget({ trackIds: ["track-a"] });
        publishTrackDropTarget({ trackIds: [] });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual([]);
    });

    it("clear is idempotent and does not notify twice", () => {
        publishTrackDropTarget({ trackIds: ["track-a"] });
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
        publishTrackDropTarget({ trackIds: ["", "track-a", null as unknown as string] });
        expect(getTrackDropTargetSnapshot().trackIds).toEqual(["track-a"]);
    });
});
