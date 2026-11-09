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
    /** 可逆编组快照：本块由哪些原始用户块编组而来。 */
    readonly groupedFrom?: readonly UserParamBlock[];
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

/**
 * 视图块的编辑能力标签（与 Rust `ProjectedBlockKind` 一一对应）。
 *
 * 【`"grouped"` 只有一条语义】可逆编组（`groupedFrom` 非空）：几何是"若干成员 +
 * 一份快照"的联合，横向切开 / 伸缩没有唯一解，必须先取消编组。
 *
 * 它与"这一 run 里有几个成员"**无关** —— 多成员 run（手工跨两条相邻 Clip 拖出来的
 * 块、或对 2 成员块切一刀之后的左右两半）只是几何事实，切分有唯一解，伸缩也有。
 * 把它标成 `"grouped"` 会让菜单置灰、手柄消失，静默禁掉一条本来正确的路径。
 */
export type ProjectedBlockKind = "base" | "user" | "grouped" | "manual";

export interface ProjectedParamBlock {
    readonly id: string;
    /** 派生视图块所属的用户块 id；base 块没有，多 run 用户块指向父块。 */
    readonly userBlockId?: string;
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

/** 唯一规范化入口：线协议空编辑集必须补出 blocks 数组。 */
export function normalizeParamBlockEditSet(
    value: Partial<ParamBlockEditSet> | null | undefined,
): ParamBlockEditSet {
    return {
        schemaVersion: 1,
        revision: Number.isFinite(value?.revision) ? Number(value!.revision) : 0,
        blocks: Array.isArray(value?.blocks) ? value!.blocks! : [],
    };
}

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
