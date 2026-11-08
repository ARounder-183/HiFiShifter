import { expect, test } from "vitest";
import {
    captureParamBlockCreateCurve,
    enqueueParamBlockCreateCommit,
} from "./paramBlockCreationTransaction";
import type { ParamViewSegment } from "./types";
const view: ParamViewSegment = {
    key: "root|pitch|0|8|1",
    framePeriodMs: 5,
    startFrame: 0,
    stride: 1,
    referenceKind: "source_curve",
    orig: [60, 60, 60, 60, 60, 60, 60, 60],
    edit: [60, 60, 60, 60, 60, 60, 60, 60],
};
function gate() {
    let release!: () => void;
    const promise = new Promise<void>((resolve) => {
        release = resolve;
    });
    return { promise, release };
}
test("rapid creation snapshots the visible curve, so previous edits do not spring back", () => {
    const first = { ...view, edit: [62, 62, 60, 60, 60, 60, 60, 60] };
    const secondBasis = captureParamBlockCreateCurve(view, first)!;
    const second = {
        ...secondBasis,
        edit: secondBasis.edit.map((value, i) => (i >= 4 && i < 6 ? 65 : value)),
    };
    const thirdBasis = captureParamBlockCreateCurve(view, second)!;
    expect(thirdBasis.edit).toEqual([62, 62, 60, 60, 65, 65, 60, 60]);
    first.edit[0] = 99;
    expect(secondBasis.edit[0]).toBe(62);
    expect(view.edit).toEqual([60, 60, 60, 60, 60, 60, 60, 60]);
    expect(captureParamBlockCreateCurve({ ...view, key: "other" }, first)?.edit).toEqual(view.edit);
});
test("full-resolution reads wait for preceding writes, and a completion cannot release a new pointer", async () => {
    const queue = { current: Promise.resolve() };
    const activity = { current: { pointerGestureToken: null as number | null, pendingCommits: 0 } };
    const live = { current: false };
    const pending = gate();
    const revisions: number[] = [];
    let revision = 0;
    const settled: boolean[] = [];
    const common = {
        queue,
        activity,
        liveActive: live,
        onSettled: () => settled.push(live.current),
    };
    const first = enqueueParamBlockCreateCommit({
        ...common,
        work: async () => {
            revisions.push(revision);
            await pending.promise;
            revision++;
        },
    });
    const second = enqueueParamBlockCreateCommit({
        ...common,
        work: async () => {
            revisions.push(revision);
            revision++;
        },
    });
    await Promise.resolve();
    await Promise.resolve();
    expect(revisions).toEqual([0]);
    expect(activity.current.pendingCommits).toBe(2);
    expect(live.current).toBe(true);
    activity.current.pointerGestureToken = 3;
    pending.release();
    await Promise.all([first, second]);
    expect(revisions).toEqual([0, 1]);
    expect(settled).toEqual([true, true]);
    expect(live.current).toBe(true);
    activity.current.pointerGestureToken = null;
    await enqueueParamBlockCreateCommit({
        ...common,
        work: async () => {
            revision++;
        },
    });
    expect(activity.current.pendingCommits).toBe(0);
    expect(live.current).toBe(false);
});
test("a failed earlier write releases its reservation and does not strand later creations", async () => {
    const queue = { current: Promise.resolve() };
    const activity = { current: { pointerGestureToken: null, pendingCommits: 0 } };
    const live = { current: false };
    let saved = false;
    const args = { queue, activity, liveActive: live, onSettled: () => {} };
    const first = enqueueParamBlockCreateCommit({
        ...args,
        work: async () => {
            throw new Error("rejected");
        },
    });
    const second = enqueueParamBlockCreateCommit({
        ...args,
        work: async () => {
            saved = true;
        },
    });
    await expect(first).rejects.toThrow("rejected");
    await second;
    expect(saved).toBe(true);
    expect(live.current).toBe(false);
    expect(activity.current.pendingCommits).toBe(0);
});
