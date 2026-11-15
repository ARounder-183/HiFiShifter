import { afterEach, describe, expect, it } from "vitest";
import {
    drainParamBlockCommits,
    pendingParamBlockCommitCount,
    resetParamBlockCommitCoordinatorForTests,
    trackParamBlockCommit,
} from "./paramBlockCommitCoordinator";

afterEach(() => resetParamBlockCommitCoordinatorForTests());

describe("paramBlockCommitCoordinator", () => {
    it("drains all registered writes", async () => {
        let release!: () => void;
        const promise = new Promise<void>((resolve) => {
            release = resolve;
        });
        trackParamBlockCommit("drag", "root", promise);
        let drained = false;
        const drain = drainParamBlockCommits().then(() => {
            drained = true;
        });
        await Promise.resolve();
        expect(drained).toBe(false);
        expect(pendingParamBlockCommitCount()).toBe(1);
        release();
        await drain;
        expect(drained).toBe(true);
        expect(pendingParamBlockCommitCount()).toBe(0);
    });

    it("also waits for writes queued while draining", async () => {
        let releaseFirst!: () => void;
        const first = new Promise<void>((resolve) => {
            releaseFirst = resolve;
        });
        trackParamBlockCommit("drag", "root", first);
        let releaseSecond!: () => void;
        const second = new Promise<void>((resolve) => {
            releaseSecond = resolve;
        });
        let drained = false;
        const drain = drainParamBlockCommits().then(() => {
            drained = true;
        });
        setTimeout(() => trackParamBlockCommit("layout", "root", second), 0);
        releaseFirst();
        await Promise.resolve();
        expect(drained).toBe(false);
        releaseSecond();
        await drain;
        expect(drained).toBe(true);
    });

    it("does not let a failed write block history jumps", async () => {
        const failed = Promise.reject(new Error("boom"));
        trackParamBlockCommit("amplitude", "root", failed);
        await expect(drainParamBlockCommits()).resolves.toBeUndefined();
    });
});
