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

/**
 * 用户块对基块的“所有权覆盖”。
 *
 * - `full: true`：这些 span 命中完整基块时，隐藏整个基块（即使可见几何被缩短，
 *   缩短掉的部分也保持空白，而不是重新露出基块）；
 * - `full: false`：只覆盖这些 span 自身的投影区间（基块片段）。
 *
 * 普通手工块没有 baseCover：它只按自己的投影区间占用，因此基块会在两侧留下
 * 剩余段（这正是“在基块中间画一块”应有的结果）。
 */
export interface BaseCover {
    readonly spans: readonly TakeSpan[];
    readonly full: boolean;
}

export interface UserParamBlock {
    readonly id: string;
    readonly geometry: BlockGeometry;
    readonly deleted?: boolean;
    readonly manual?: boolean;
    readonly value?: number | null;
    readonly transitionOutSec?: number;
    readonly createdSeq: number;
    /** 从基块派生时记录“我替代了哪一段基块”。 */
    readonly baseCover?: BaseCover;
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
    /** 成员的语义标签；编辑回写时保留（缺省按 manual 处理）。 */
    readonly label?: BlockLabel;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
    /**
     * 该可见帧区间起点 / 终点对应的源域秒。
     *
     * 有方向：倒放时 `frameSourceStartSec > frameSourceEndSec`。它是编辑的局部
     * 仿射坐标系，**不是** inverseProject 的结果 —— Loop 回绕、循环边界与倒放
     * 都必须由它表达，而不是再从全局帧反推。
     */
    readonly frameSourceStartSec: number;
    readonly frameSourceEndSec: number;
    /** 可见区间覆盖的源域范围（source 顺序，供 BaseCover / 片段物化使用）。 */
    readonly visibleSourceStartSec: number;
    readonly visibleSourceEndSec: number;
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
    /** 基块候选的所有权覆盖；用户块为 undefined。 */
    readonly baseCover?: BaseCover;
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
