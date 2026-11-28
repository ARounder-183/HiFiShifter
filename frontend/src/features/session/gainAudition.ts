/**
 * 拖动增益时的瞬时试听控制器。
 *
 * 权威状态永远只在时间线里：本控制器只把“当前手势的最新值”经 rAF 合并后发给
 * 引擎，不写 Redux、不写 undo、不返回全量 payload。手势结束时由调用方先完成
 * 权威提交，再 clear 同名试听帧。
 */

export type GainPreviewTarget =
    | {
          kind: "clip";
          clipId: string;
          itemGain?: number;
          takeGain?: number;
          fadeInSec?: number;
          fadeOutSec?: number;
          fadeInShape?: number;
          fadeInDir?: number;
          fadeOutShape?: number;
          fadeOutDir?: number;
          volumeCurveWindow?: { startFrame: number; stride: number; values: number[] };
      }
    | { kind: "track"; trackId: string; volume: number }
    | {
          kind: "trackCurve";
          trackId: string;
          paramId: string;
          window: { startFrame: number; stride: number; values: number[] };
      };

export interface GainAudition {
    readonly gestureId: string | null;
    preview(targets: readonly GainPreviewTarget[]): void;
    commit(commit: Promise<unknown>): Promise<void>;
    cancel(): void;
    dispose(): void;
}

type SendPreview = (payload: {
    gestureId: string;
    seq: number;
    targets: readonly GainPreviewTarget[];
}) => Promise<void>;
type ClearPreview = (gestureId: string) => Promise<void>;
type Schedule = (callback: () => void) => number;
type CancelSchedule = (handle: number) => void;

function newGestureId(): string {
    const cryptoApi = globalThis.crypto;
    if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID();
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function targetKey(target: GainPreviewTarget): string | null {
    if (target.kind === "track") {
        return `t:${target.trackId}:${target.volume}`;
    }
    if (target.kind === "trackCurve" || target.volumeCurveWindow) {
        // 曲线窗口的 values 可能上万项；逐帧拼字符串比发送本身还贵。让它
        // 永远走发送路径，由后端按 seq 合并。
        return null;
    }
    return [
        "c",
        target.clipId,
        target.itemGain ?? "",
        target.takeGain ?? "",
        target.fadeInSec ?? "",
        target.fadeOutSec ?? "",
        target.fadeInShape ?? "",
        target.fadeInDir ?? "",
        target.fadeOutShape ?? "",
        target.fadeOutDir ?? "",
    ].join(":");
}

function targetsKey(targets: readonly GainPreviewTarget[]): string | null {
    const parts: string[] = [];
    for (const target of targets) {
        const key = targetKey(target);
        if (key === null) return null;
        parts.push(key);
    }
    return parts.join("|");
}

/**
 * 纯控制器（不依赖 React / Tauri），便于单测 rAF 合并、seq 与在途合并。
 */
export class GainAuditionController implements GainAudition {
    private gesture: string | null = null;
    private seq = 0;
    private pending: { targets: GainPreviewTarget[]; key: string | null } | null = null;
    private rafId: number | null = null;
    private chain: Promise<void> = Promise.resolve();
    private pumping = false;
    private idleWaiters: Array<() => void> = [];
    private disposed = false;
    private readonly send: SendPreview;
    private readonly clear: ClearPreview;
    private readonly schedule: Schedule;
    private readonly cancelSchedule: CancelSchedule;

    constructor(
        send: SendPreview,
        clear: ClearPreview,
        schedule?: Schedule,
        cancelSchedule?: CancelSchedule,
    ) {
        this.send = send;
        this.clear = clear;
        this.schedule =
            schedule ??
            (typeof requestAnimationFrame === "function"
                ? (callback) => requestAnimationFrame(callback)
                : (callback) => setTimeout(callback, 16) as unknown as number);
        this.cancelSchedule =
            cancelSchedule ??
            (typeof cancelAnimationFrame === "function"
                ? (handle) => cancelAnimationFrame(handle)
                : (handle) => clearTimeout(handle as unknown as ReturnType<typeof setTimeout>));
    }

    get gestureId(): string | null {
        return this.gesture;
    }

    preview(targets: readonly GainPreviewTarget[]): void {
        if (this.disposed) return;
        if (targets.length === 0) return;
        if (this.gesture === null) {
            this.gesture = newGestureId();
            this.seq = 0;
        }
        const key = targetsKey(targets);
        if (key !== null && key === this.pending?.key) return;
        this.pending = { targets: [...targets], key };
        if (this.rafId === null) {
            this.rafId = this.schedule(() => {
                this.rafId = null;
                this.pump();
            });
        }
    }

    private pump(): void {
        if (this.pumping || this.pending === null || this.gesture === null) return;
        const gestureId = this.gesture;
        const payload = this.pending;
        this.pending = null;
        this.pumping = true;
        const seq = ++this.seq;
        const run = this.chain
            .then(() => this.send({ gestureId, seq, targets: payload.targets }))
            .catch(() => undefined);
        this.chain = run;
        void run.finally(() => {
            this.pumping = false;
            if (this.pending !== null) this.pump();
            this.resolveIdleIfIdle();
        });
    }

    private resolveIdleIfIdle(): void {
        if (this.pumping || this.pending !== null) return;
        const waiters = this.idleWaiters.splice(0);
        for (const resolve of waiters) resolve();
    }

    private async waitForIdle(): Promise<void> {
        if (!this.pumping && this.pending === null) return;
        await new Promise<void>((resolve) => this.idleWaiters.push(resolve));
    }

    private clearGesture(gestureId: string): Promise<void> {
        const run = this.chain.then(() => this.clear(gestureId)).catch(() => undefined);
        this.chain = run;
        return run;
    }

    async commit(commit: Promise<unknown>): Promise<void> {
        if (this.rafId !== null) {
            this.cancelSchedule(this.rafId);
            this.rafId = null;
        }
        if (this.gesture !== null) {
            if (this.pending !== null) this.pump();
            await this.waitForIdle();
        }
        const gestureId = this.gesture;
        this.gesture = null;
        this.seq = 0;
        try {
            await commit;
        } finally {
            if (gestureId !== null) await this.clearGesture(gestureId);
        }
    }

    cancel(): void {
        if (this.rafId !== null) {
            this.cancelSchedule(this.rafId);
            this.rafId = null;
        }
        this.pending = null;
        const gestureId = this.gesture;
        this.gesture = null;
        this.seq = 0;
        if (gestureId !== null) void this.clearGesture(gestureId);
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.cancel();
    }
}
