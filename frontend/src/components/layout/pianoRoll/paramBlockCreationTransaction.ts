import type { RefObject } from "react";
import type { ParamBlockDragActivity } from "./paramBlockDrag";
import type { ParamViewSegment } from "./types";

/** A new gesture continues the visible edits, including commits still in flight. */
export function captureParamBlockCreateCurve(
    authoritative: ParamViewSegment | null,
    interaction: ParamViewSegment | null,
): ParamViewSegment | null {
    const basis = interaction?.key === authoritative?.key ? interaction : authoritative;
    return basis
        ? { ...basis, preview: undefined, edit: basis.edit.slice(), orig: basis.orig.slice() }
        : null;
}

/** Serialize the entire read/plan/write job, not only the eventual upload. */
export function enqueueParamBlockCreateCommit(args: {
    queue: RefObject<Promise<void>>;
    activity: RefObject<ParamBlockDragActivity>;
    liveActive: RefObject<boolean>;
    work: () => Promise<void>;
    onSettled: () => void;
}): Promise<void> {
    args.activity.current.pendingCommits++;
    args.liveActive.current = true;
    const job = args.queue.current
        .catch(() => undefined)
        .then(args.work)
        .finally(() => {
            args.activity.current.pendingCommits--;
            if (
                !args.activity.current.pendingCommits &&
                args.activity.current.pointerGestureToken === null
            )
                args.liveActive.current = false;
            args.onSettled();
        });
    args.queue.current = job.catch(() => undefined);
    return job;
}
