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
    | { readonly kind: "root"; readonly span: RootSpan }
    /** 一等公民组容器：children 是唯一几何真值。 */
    | { readonly kind: "group"; readonly children: readonly UserParamBlock[] };

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
}

export interface ParamBlockEditSet {
    readonly schemaVersion: 2;
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
    readonly cutStart?: boolean;
    readonly cutEnd?: boolean;
}

/**
 * 视图块的编辑能力标签（与 Rust `ProjectedBlockKind` 一一对应）。
 *
 * 【`"grouped"` 只有一条语义】组容器投影：几何由 children 决定，可切分 /
 * 可伸缩，但不可嵌套。它与"这一 run 里有几个成员"无关 —— 多成员 run
 * （手工跨两条相邻 Clip 拖出来的块、或对 2 成员块切一刀之后的左右两半）
 * 只是几何事实，切分有唯一解，伸缩也有。
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
    /** Read-only source MIDI note for a single-member block. */
    readonly sourceNote?: number | null;
    /** `clipId|takeId|cycle` identity for cross-source adjacency checks. */
    readonly sourceKey?: string;
    /** Stable full-source note-event identity; never derived from visible geometry. */
    readonly sourceEventId?: string;
    readonly cutStart?: boolean;
    readonly cutEnd?: boolean;
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
    schemaVersion: 2,
    revision: 0,
    blocks: [],
};

type LegacyBlock = Partial<UserParamBlock> & {
    readonly id?: string;
    readonly geometry?:
        | BlockGeometry
        | { readonly kind: "take"; readonly members: readonly TakeSpan[] }
        | { readonly kind: "root"; readonly span: RootSpan };
    readonly groupedFrom?: readonly LegacyBlock[];
    readonly createdSeq?: number;
};

const MAX_GROUP_DEPTH = 8;

const BLOCK_LABELS: ReadonlySet<string> = new Set([
    "syllable",
    "phoneme",
    "midiNote",
    "gameBaseline",
    "manual",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null;
}

function normalizeTakeSpan(
    value: unknown,
    diagnostics: string[] | undefined,
    code: string,
): TakeSpan | null {
    if (!isRecord(value)) {
        diagnostics?.push(code);
        return null;
    }
    const clipId = value.clipId;
    const takeId = value.takeId;
    const cycle = value.cycle;
    const sourceStartSec = value.sourceStartSec;
    const sourceEndSec = value.sourceEndSec;
    const rawLabel = value.label;
    const label = rawLabel === undefined ? "manual" : rawLabel;
    if (
        typeof clipId !== "string" ||
        clipId.length === 0 ||
        typeof takeId !== "string" ||
        takeId.length === 0 ||
        typeof cycle !== "number" ||
        !Number.isInteger(cycle) ||
        cycle < 0 ||
        typeof sourceStartSec !== "number" ||
        typeof sourceEndSec !== "number" ||
        !Number.isFinite(sourceStartSec) ||
        !Number.isFinite(sourceEndSec) ||
        sourceEndSec <= sourceStartSec ||
        typeof label !== "string" ||
        !BLOCK_LABELS.has(label)
    ) {
        diagnostics?.push(code);
        return null;
    }
    const syllableId = value.syllableId;
    return {
        clipId,
        takeId,
        cycle,
        sourceStartSec,
        sourceEndSec,
        ...(typeof syllableId === "string" ? { syllableId } : {}),
        label: label as BlockLabel,
    };
}

function normalizeBaseCover(
    value: unknown,
    diagnostics: string[] | undefined,
): BaseCover | undefined {
    if (value === undefined || value === null) return undefined;
    if (!isRecord(value) || !Array.isArray(value.spans)) {
        diagnostics?.push("block_malformed_base_cover");
        return undefined;
    }
    const spans = value.spans
        .map((span) => normalizeTakeSpan(span, diagnostics, "block_malformed_base_cover"))
        .filter((span): span is TakeSpan => span !== null);
    if (spans.length === 0) {
        diagnostics?.push("block_malformed_base_cover");
        return undefined;
    }
    return { spans, full: value.full === true };
}

function normalizeUserBlock(
    value: LegacyBlock,
    depth = 0,
    diagnostics?: string[],
): UserParamBlock | null {
    if (depth > MAX_GROUP_DEPTH) {
        diagnostics?.push("group_depth_exceeded");
        return null;
    }
    if (typeof value.id !== "string" || value.id.length === 0 || value.geometry === undefined) {
        diagnostics?.push("block_missing_geometry");
        return null;
    }
    const invalid = (code: string): Record<string, never> => {
        diagnostics?.push(code);
        return {};
    };
    const rawValue = value.value;
    const valueField =
        rawValue === undefined
            ? {}
            : rawValue === null || (typeof rawValue === "number" && Number.isFinite(rawValue))
              ? { value: rawValue }
              : invalid("block_malformed_value");
    const rawTransition = value.transitionOutSec;
    const transitionField =
        rawTransition === undefined
            ? {}
            : typeof rawTransition === "number" &&
                Number.isFinite(rawTransition) &&
                rawTransition >= 0
              ? { transitionOutSec: rawTransition }
              : invalid("block_malformed_transition");
    const base = {
        id: value.id,
        deleted: value.deleted === true,
        manual: value.manual === true,
        ...valueField,
        ...transitionField,
        createdSeq: Number.isFinite(value.createdSeq) ? Number(value.createdSeq) : 0,
    };
    const baseCover = normalizeBaseCover(value.baseCover, diagnostics);
    const legacyChildren = Array.isArray(value.groupedFrom) ? value.groupedFrom : [];
    const convertChildren = (children: readonly LegacyBlock[]) =>
        children
            .map((child) => normalizeUserBlock(child, depth + 1, diagnostics))
            .filter((child): child is UserParamBlock => child !== null);
    const finishGroup = (children: readonly LegacyBlock[]) => {
        const converted = convertChildren(children);
        if (converted.length === 0) {
            diagnostics?.push("group_too_small");
            return null;
        }
        if (converted.length === 1) {
            diagnostics?.push("group_too_small");
            return converted[0]!;
        }
        const leaves = converted.flatMap(flattenGroupChildren);
        const hasRoot = leaves.some((child) => child.geometry.kind === "root");
        const hasTake = leaves.some((child) => child.geometry.kind === "take");
        if (hasRoot && hasTake) {
            diagnostics?.push("group_mixed_geometry");
            return null;
        }
        return { ...base, geometry: { kind: "group" as const, children: converted } };
    };
    if (value.geometry.kind === "group") {
        if (!Array.isArray(value.geometry.children)) {
            diagnostics?.push("group_children_missing");
            return null;
        }
        return finishGroup(value.geometry.children as readonly LegacyBlock[]);
    }
    if (legacyChildren.length >= 2) return finishGroup(legacyChildren);
    if (value.geometry.kind === "take") {
        if (!Array.isArray(value.geometry.members)) {
            diagnostics?.push("block_missing_members");
            return null;
        }
        const members = value.geometry.members
            .map((member) => normalizeTakeSpan(member, diagnostics, "block_missing_members"))
            .filter((member): member is TakeSpan => member !== null);
        if (members.length === 0) {
            diagnostics?.push("block_missing_members");
            return null;
        }
        return {
            ...base,
            geometry: { kind: "take", members },
            ...(baseCover === undefined ? {} : { baseCover }),
        };
    }
    if (value.geometry.kind === "root") {
        const span = value.geometry.span;
        if (
            !isRecord(span) ||
            typeof span.startSec !== "number" ||
            typeof span.endSec !== "number" ||
            !Number.isFinite(span.startSec) ||
            !Number.isFinite(span.endSec) ||
            span.endSec <= span.startSec
        ) {
            diagnostics?.push("block_malformed_root");
            return null;
        }
        return { ...base, geometry: { kind: "root", span: { ...span } } };
    }
    return null;
}
/** 唯一规范化入口：空编辑集补 blocks 数组，并一次性把旧 groupedFrom 转成 Group 容器。 */
export function normalizeParamBlockEditSet(
    value:
        | (Partial<ParamBlockEditSet> & { readonly blocks?: readonly LegacyBlock[] })
        | null
        | undefined,
): ParamBlockEditSet {
    return normalizeParamBlockEditSetWithDiagnostics(value).edits;
}

export function normalizeParamBlockEditSetWithDiagnostics(
    value:
        | (Partial<ParamBlockEditSet> & { readonly blocks?: readonly LegacyBlock[] })
        | null
        | undefined,
): { readonly edits: ParamBlockEditSet; readonly diagnostics: readonly string[] } {
    const diagnostics: string[] = [];
    const blocks = Array.isArray(value?.blocks)
        ? value!.blocks
              .map((block) => normalizeUserBlock(block, 0, diagnostics))
              .filter((block): block is UserParamBlock => block !== null)
        : [];
    return {
        edits: {
            schemaVersion: 2,
            revision: Number.isFinite(value?.revision) ? Number(value!.revision) : 0,
            blocks,
        },
        diagnostics,
    };
}

/** 逻辑块身份：同一用户块的全部投影 run 共享一个 id；base 块用自身 id。 */
export function logicalBlockId(block: {
    readonly id: string;
    readonly userBlockId?: string;
}): string {
    return block.userBlockId ?? block.id;
}

/** 选中 id 可以来自逻辑块或任一投影 run；统一按逻辑身份匹配。 */
export function blockMatchesLogicalId(
    block: { readonly id: string; readonly userBlockId?: string },
    id: string,
): boolean {
    return block.id === id || block.userBlockId === id || logicalBlockId(block) === id;
}

/** 递归拍平组，返回全部叶子块。 */
export function flattenGroupChildren(block: UserParamBlock): UserParamBlock[] {
    if (block.geometry.kind !== "group") return [block];
    return block.geometry.children.flatMap(flattenGroupChildren);
}

/** Group 的全部叶子成员。 */
export function groupLeafMembers(block: UserParamBlock): TakeSpan[] {
    return flattenGroupChildren(block).flatMap((child) =>
        child.geometry.kind === "take" ? child.geometry.members : [],
    );
}

/** 所有叶子 value 完全一致时才返回公共值（嵌套组递归）。 */
export function groupCommonValue(block: UserParamBlock): number | null {
    if (block.geometry.kind !== "group") return null;
    const leaves = flattenGroupChildren(block);
    const first = leaves[0]?.value;
    if (first === undefined || first === null) return null;
    return leaves.every((child) => child.value === first) ? first : null;
}

export function takeSpanKey(span: TakeSpan): string {
    return `${span.clipId}|${span.takeId}|${span.cycle}|${span.sourceStartSec.toFixed(6)}|${span.sourceEndSec.toFixed(6)}`;
}
