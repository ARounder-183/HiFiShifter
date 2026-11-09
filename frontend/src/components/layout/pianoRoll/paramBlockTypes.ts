/**
 * 参数块的源域类型（与 Rust `param_blocks::types` 逐字段对应）。
 *
 * 【唯一权威】用户只声明“哪些源域区间属于同一个块”；工程帧只是派生视图。
 * 任何持久化字段都不得出现绝对帧。
 */

export type BlockLabel = "syllable" | "phoneme" | "midiNote" | "gameBaseline" | "manual";

export interface TakeKey {
    readonly clipId: string;
    readonly takeId: string;
}

export interface TakeSpan {
    readonly clipId: string;
    readonly takeId: string;
    readonly cycle: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly syllableId?: string;
    readonly label: BlockLabel;
}

export interface RootSpan {
    readonly startSec: number;
    readonly endSec: number;
}

export type BlockGeometry =
    | { readonly kind: "take"; readonly members: readonly TakeSpan[] }
    | { readonly kind: "root"; readonly span: RootSpan };

export interface UserParamBlock {
    readonly id: string;
    readonly geometry: BlockGeometry;
    readonly deleted?: boolean;
    readonly manual?: boolean;
    readonly value?: number | null;
    readonly transitionOutSec?: number;
    readonly createdSeq: number;
}

export interface ParamBlockEditSet {
    readonly schemaVersion: 1;
    readonly revision: number;
    readonly blocks: readonly UserParamBlock[];
}

export interface ProjectedMember {
    readonly clipId: string;
    readonly takeId: string;
    readonly cycle: number;
    readonly sourceStartSec: number;
    readonly sourceEndSec: number;
    readonly syllableId?: string;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

export type ProjectedBlockKind = "base" | "user" | "merged" | "manual";

export interface ProjectedParamBlock {
    readonly id: string;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
    readonly transitionFrames?: number;
    readonly kind: ProjectedBlockKind;
    readonly members: readonly ProjectedMember[];
    readonly value?: number | null;
}

export interface ParamBlockView {
    readonly schemaVersion: 1;
    readonly rootTrackId: string;
    readonly editRevision: number;
    readonly layoutKey: string;
    readonly framePeriodMs: number;
    readonly blocks: readonly ProjectedParamBlock[];
    readonly orphanCount: number;
    readonly analysisPending: boolean;
}

export const EMPTY_PARAM_BLOCK_EDIT_SET: ParamBlockEditSet = {
    schemaVersion: 1,
    revision: 0,
    blocks: [],
};

export const EMPTY_PARAM_BLOCK_VIEW: ParamBlockView = {
    schemaVersion: 1,
    rootTrackId: "",
    editRevision: 0,
    layoutKey: "",
    framePeriodMs: 5,
    blocks: [],
    orphanCount: 0,
    analysisPending: false,
};

export function takeSpanKey(span: TakeSpan): string {
    return `${span.clipId}|${span.takeId}|${span.cycle}|${span.sourceStartSec.toFixed(6)}|${span.sourceEndSec.toFixed(6)}`;
}
