/**
 * Global ordering barrier for asynchronous parameter-block writes.
 *
 * Block commits are queued locally, but undo/redo is a separate backend
 * command. Without a shared barrier a pointer-up commit can land after an undo
 * and silently restore the dragged state. Every backend mutation of parameter
 * blocks or their curves is registered here; history jumps drain this set
 * before issuing the jump command.
 */
import { reportParamBlockDiagnostic } from "./paramBlockDiagnostics";

export type ParamBlockCommitKind =
    | "drag"
    | "resize"
    | "create"
    | "split"
    | "transition"
    | "delete"
    | "group"
    | "ungroup"
    | "restore"
    | "normalize"
    | "amplitude"
    | "layout";

interface PendingCommit {
    readonly id: number;
    readonly kind: ParamBlockCommitKind;
    readonly rootTrackId: string | null;
    readonly promise: Promise<unknown>;
    readonly startedAt: number;
}

const pending = new Set<PendingCommit>();
let nextId = 1;

export function trackParamBlockCommit<T>(
    kind: ParamBlockCommitKind,
    rootTrackId: string | null,
    promise: Promise<T>,
): Promise<T> {
    const entry: PendingCommit = {
        id: nextId++,
        kind,
        rootTrackId,
        promise,
        startedAt: typeof performance !== "undefined" ? performance.now() : Date.now(),
    };
    pending.add(entry);
    // Observe settlement for cleanup without swallowing the rejection that the
    // original caller still needs to handle.
    void promise.then(
        () => pending.delete(entry),
        () => pending.delete(entry),
    );
    return promise;
}

export function pendingParamBlockCommitCount(): number {
    return pending.size;
}

/**
 * Wait until no registered write remains, including writes queued while
 * draining. A timeout is diagnostic rather than fatal: a broken IPC must not
 * make undo permanently unavailable.
 */
export async function drainParamBlockCommits(timeoutMs = 5000): Promise<void> {
    const deadline =
        (typeof performance !== "undefined" ? performance.now() : Date.now()) +
        Math.max(0, timeoutMs);
    while (pending.size > 0) {
        const now = typeof performance !== "undefined" ? performance.now() : Date.now();
        const remaining = deadline - now;
        if (remaining <= 0) {
            /*
             * 超时是**诊断**，不是致命错误：IPC 坏掉不能把撤销永久锁死。
             * 但 `console.warn` 在发布构建里等于没有 —— 走结构化错误日志，
             * 用户上报问题时能查到是哪几笔写没排空。
             */
            reportParamBlockDiagnostic({
                code: "commit_drain_timeout",
                gesture: "layout",
                rootTrackId: null,
                blockId: null,
                frame: null,
                detail: [...pending].map((entry) => entry.kind).join(","),
            });
            return;
        }
        await Promise.race([
            Promise.allSettled([...pending].map((entry) => entry.promise)),
            new Promise<void>((resolve) => setTimeout(resolve, remaining)),
        ]);
    }
}

/** Test-only reset. */
export function resetParamBlockCommitCoordinatorForTests(): void {
    pending.clear();
    nextId = 1;
}
