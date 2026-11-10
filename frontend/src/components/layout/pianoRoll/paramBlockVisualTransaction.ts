/**
 * 参数块视觉事务（纯状态机）。
 *
 * 视觉预览不能无限期覆盖 canonical 渲染：每个事务都带一个 epoch，
 * 外部 Clip 编辑 / 撤销 / 根切换 / 参数窗口变化会让 epoch 失配并立即失效。
 * 提交后的等待阶段另有硬超时，避免撤销后交接条件永远不成立而把旧块锁在屏幕上。
 */
import type { PianoRollParamBlockSpec } from "./kernel/host/pianoRollKernelData";

export interface ParamBlockEpoch {
    readonly rootTrackId: string | null;
    readonly layoutKey: string | null;
    readonly editRevision: number;
    readonly historyKey: string;
    readonly framePeriodMs: number;
}

export type ParamBlockVisualTransactionKind = "drag" | "resize" | "create" | "transition";
export type ParamBlockVisualTransactionPhase =
    "gesture" | "committing" | "awaiting-authority" | "released";

export interface ParamBlockVisualTransaction {
    readonly id: number;
    readonly epoch: ParamBlockEpoch;
    readonly kind: ParamBlockVisualTransactionKind;
    spec: PianoRollParamBlockSpec;
    phase: ParamBlockVisualTransactionPhase;
    readonly startedAt: number;
    expectedEditRevision?: number;
    expectedGeometrySignature?: string;
    committedSegments?: readonly {
        readonly startFrame: number;
        readonly values: readonly number[];
    }[];
}

export function sameParamBlockEpoch(a: ParamBlockEpoch, b: ParamBlockEpoch): boolean {
    return (
        a.rootTrackId === b.rootTrackId &&
        a.layoutKey === b.layoutKey &&
        a.editRevision === b.editRevision &&
        a.historyKey === b.historyKey &&
        Math.abs(a.framePeriodMs - b.framePeriodMs) < 1e-9
    );
}

/** 拖拽期间允许本地 logicalCenters 提交改变 editRevision，但不能跨 Clip/历史/根切换。 */
export function sameParamBlockEpochLayout(a: ParamBlockEpoch, b: ParamBlockEpoch): boolean {
    return (
        a.rootTrackId === b.rootTrackId &&
        a.layoutKey === b.layoutKey &&
        a.historyKey === b.historyKey &&
        Math.abs(a.framePeriodMs - b.framePeriodMs) < 1e-9
    );
}

export function shouldUseParamBlockVisualTransaction(
    tx: ParamBlockVisualTransaction | null,
    currentEpoch: ParamBlockEpoch,
): boolean {
    return tx !== null && tx.phase !== "released" && sameParamBlockEpoch(tx.epoch, currentEpoch);
}

function finiteCenter(value: number | null | undefined): number | null {
    return value !== null && value !== undefined && Number.isFinite(value) ? value : null;
}

/**
 * 只比较参与块的可观察几何，不比较颜色 / label / selected 等表现字段。
 * 多 run 块按每个 run 的 id 逐项比较，避免用一个包围盒冒充完整几何。
 */
export function paramBlockSpecsMatchGeometry(
    left: PianoRollParamBlockSpec | null,
    right: PianoRollParamBlockSpec | null,
): boolean {
    if (left === null || right === null) return false;
    if (left.blocks.length !== right.blocks.length) return false;
    const rightById = new Map(right.blocks.map((block) => [block.id, block]));
    for (const block of left.blocks) {
        const other = rightById.get(block.id);
        if (other === undefined) return false;
        if (
            Math.abs(block.startSec - other.startSec) > 1e-6 ||
            Math.abs(block.endSec - other.endSec) > 1e-6
        ) {
            return false;
        }
        if (finiteCenter(block.frameCenterValue) !== finiteCenter(other.frameCenterValue)) {
            return false;
        }
        if (finiteCenter(block.displayCenterValue) !== finiteCenter(other.displayCenterValue)) {
            return false;
        }
    }
    return true;
}

/** 提交后等待权威回读的阶段推进。gesture 阶段不设超时（用户可以长时间拖拽）。 */
export function advanceParamBlockVisualTransaction(args: {
    readonly tx: ParamBlockVisualTransaction;
    readonly now: number;
    readonly canonicalSpec: PianoRollParamBlockSpec | null;
    readonly currentEpoch: ParamBlockEpoch;
    readonly authorityRevision: number;
    readonly authorityRevisionAtStart: number;
    readonly committedSegmentsPresent: boolean;
    readonly timeoutMs?: number;
}): "keep" | "release" | "expire" {
    const { tx } = args;
    if (tx.phase === "released") return "release";
    if (!sameParamBlockEpoch(tx.epoch, args.currentEpoch)) return "release";
    if (tx.phase === "gesture") return "keep";
    const timeoutMs = args.timeoutMs ?? 1200;
    if (args.now - tx.startedAt > timeoutMs) return "expire";
    if (tx.phase !== "awaiting-authority") return "keep";
    if (args.canonicalSpec === null) return "keep";
    const matches = paramBlockSpecsMatchGeometry(tx.spec, args.canonicalSpec);
    const authorityAdvanced = args.authorityRevision > args.authorityRevisionAtStart;
    if (matches && authorityAdvanced && args.committedSegmentsPresent) return "release";
    return "keep";
}
