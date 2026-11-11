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

/*
 * 【串行化去哪了】新建块的读-写串行化已经收口到
 * `useParamBlockPersistence.commitPlan`（一个根一条队列，见那边的
 * 「两个 plan 严格串行」用例）。这里只剩**手势侧**的两件事：
 * `pendingCommits` 记账与 `onSettled`。
 */
test("a completion cannot release a new pointer, and pendingCommits is balanced", async () => {
    const activity = { current: { pointerGestureToken: null as number | null, pendingCommits: 0 } };
    const live = { current: false };
    const pending = gate();
    const settled: boolean[] = [];
    const common = {
        activity,
        liveActive: live,
        onSettled: () => settled.push(live.current),
    };
    const first = enqueueParamBlockCreateCommit({
        ...common,
        run: () => pending.promise,
    });
    const second = enqueueParamBlockCreateCommit({
        ...common,
        run: async () => undefined,
    });
    expect(activity.current.pendingCommits).toBe(2);
    expect(live.current).toBe(true);
    await second;
    // 第二个已经结束，但指针手势仍在：liveEdit 必须保持开启。
    activity.current.pointerGestureToken = 3;
    pending.release();
    await first;
    expect(settled).toEqual([true, true]);
    expect(live.current).toBe(true);
    activity.current.pointerGestureToken = null;
    await enqueueParamBlockCreateCommit({ ...common, run: async () => undefined });
    expect(activity.current.pendingCommits).toBe(0);
    expect(live.current).toBe(false);
});

test("a failed creation releases its reservation and does not strand later ones", async () => {
    const activity = { current: { pointerGestureToken: null, pendingCommits: 0 } };
    const live = { current: false };
    let saved = false;
    const args = { activity, liveActive: live, onSettled: () => {} };
    const first = enqueueParamBlockCreateCommit({
        ...args,
        run: async () => {
            throw new Error("rejected");
        },
    });
    const second = enqueueParamBlockCreateCommit({
        ...args,
        run: async () => {
            saved = true;
        },
    });
    await expect(first).rejects.toThrow("rejected");
    await second;
    expect(saved).toBe(true);
    expect(live.current).toBe(false);
    expect(activity.current.pendingCommits).toBe(0);
});
