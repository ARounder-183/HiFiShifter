/**
 * 参数块编辑原语（Rust `param_blocks::editing` 的预览端口）。
 *
 * 【职责】所有编辑只改写 `ParamBlockEditSet`，不写绝对帧。
 * 【约束】新块必须经过 `placeUserBlock`，由它裁剪被覆盖的旧用户块。
 */

import {
    normalizeLoopTakeSpans,
    projectTakeSpan,
    projectionForClip,
    sourceAtProjectedFrame,
    toFrame,
    toSec,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";
import { composeParamBlockView } from "./paramBlockComposition";
import { narrowBaseCoverToMembers } from "./paramBlockBaseCover";
import {
    flattenGroupChildren,
    logicalBlockId,
    takeSpanKey,
    type BaseCover,
    type BlockGeometry,
    type ParamBlockEditSet,
    type ProjectedMember,
    type ProjectedParamBlock,
    type RootSpan,
    type TakeSpan,
    type UserParamBlock,
} from "./paramBlockTypes";

const EPS = 1e-9;
const MAX_LOOP_CYCLES = 20_000;

function hash(text: string): string {
    let left = 0x811c9dc5;
    let right = 0x9e3779b9;
    for (let index = 0; index < text.length; index += 1) {
        left = Math.imul(left ^ text.charCodeAt(index), 0x01000193);
        right = Math.imul(right ^ text.charCodeAt(index), 0x85ebca6b);
    }
    return `${(left >>> 0).toString(16)}${(right >>> 0).toString(16)}`;
}

export function newParamBlockId(prefix = "ublock"): string {
    return `${prefix}:${hash(`${Date.now()}:${Math.random()}:${prefix}`)}`;
}

export function nextSeq(edits: ParamBlockEditSet): number {
    return edits.blocks.reduce((max, block) => Math.max(max, block.createdSeq), 0) + 1;
}

function subtractInterval(
    start: number,
    end: number,
    cutStart: number,
    cutEnd: number,
): Array<[number, number]> {
    if (cutEnd <= start + EPS || cutStart >= end - EPS) return [[start, end]];
    const out: Array<[number, number]> = [];
    if (start < cutStart) out.push([start, cutStart]);
    if (end > cutEnd) out.push([cutEnd, end]);
    return out;
}

function sameMember(left: TakeSpan, right: TakeSpan): boolean {
    return (
        left.clipId === right.clipId &&
        left.takeId === right.takeId &&
        left.cycle === right.cycle &&
        Math.abs(left.sourceStartSec - right.sourceStartSec) <= 1e-7 &&
        Math.abs(left.sourceEndSec - right.sourceEndSec) <= 1e-7
    );
}

function sameMemberSet(left: readonly TakeSpan[], right: readonly TakeSpan[]): boolean {
    return (
        left.length === right.length &&
        left.every((member) => right.some((other) => sameMember(member, other)))
    );
}

function rootPieceId(block: UserParamBlock, index: number, start: number, end: number): string {
    if (index === 0) return block.id;
    return `${block.id}:piece:${start.toFixed(9)}:${end.toFixed(9)}`;
}

/** 用 target 覆盖旧用户块；被部分覆盖的块保留未覆盖成员。 */
export function placeUserBlock(
    edits: ParamBlockEditSet,
    target: UserParamBlock,
    context?: { readonly clips: readonly ParamBlockSourceClip[]; readonly framePeriodMs: number },
): ParamBlockEditSet {
    const blocks: UserParamBlock[] = [];
    for (const block of edits.blocks) {
        if (block.id === target.id) continue;
        if (block.deleted || target.deleted) {
            blocks.push(block);
            continue;
        }
        blocks.push(...trimUserBlockAgainst(block, target, context));
    }
    blocks.push(target);
    return {
        schemaVersion: 2,
        revision: edits.revision + 1,
        blocks,
    };
}

/** 把一个块按 target 裁剪；组会递归到叶子，空组退化，单 child 组退化。 */
function trimUserBlockAgainst(
    existing: UserParamBlock,
    target: UserParamBlock,
    context?: { readonly clips: readonly ParamBlockSourceClip[]; readonly framePeriodMs: number },
): UserParamBlock[] {
    if (target.geometry.kind === "group") {
        let current: UserParamBlock[] = [existing];
        for (const child of target.geometry.children) {
            current = current.flatMap((block) => trimUserBlockAgainst(block, child, context));
            if (current.length === 0) break;
        }
        return current;
    }
    if (existing.geometry.kind === "group") {
        const children = existing.geometry.children.flatMap((child) =>
            trimUserBlockAgainst(child, target, context),
        );
        if (children.length === 0) return [];
        if (children.length === 1) return children;
        return [
            {
                ...existing,
                geometry: { kind: "group", children },
            },
        ];
    }
    if (existing.geometry.kind === "take" && target.geometry.kind === "take") {
        const kept: TakeSpan[] = [];
        for (const member of existing.geometry.members) {
            let pieces: Array<[number, number]> = [[member.sourceStartSec, member.sourceEndSec]];
            for (const cut of target.geometry.members) {
                if (
                    cut.clipId !== member.clipId ||
                    cut.takeId !== member.takeId ||
                    cut.cycle !== member.cycle
                ) {
                    continue;
                }
                const next: Array<[number, number]> = [];
                for (const [start, end] of pieces) {
                    next.push(
                        ...subtractInterval(start, end, cut.sourceStartSec, cut.sourceEndSec),
                    );
                }
                pieces = next;
            }
            for (const [start, end] of pieces) {
                if (end > start + EPS)
                    kept.push({ ...member, sourceStartSec: start, sourceEndSec: end });
            }
        }
        if (kept.length === 0) return [];
        return [{ ...existing, geometry: { kind: "take", members: kept } }];
    }
    if (existing.geometry.kind === "root" && target.geometry.kind === "root") {
        return subtractInterval(
            existing.geometry.span.startSec,
            existing.geometry.span.endSec,
            target.geometry.span.startSec,
            target.geometry.span.endSec,
        )
            .filter(([start, end]) => end > start + EPS)
            .map(([start, end], index) => ({
                ...existing,
                id: rootPieceId(existing, index, start, end),
                geometry: { kind: "root", span: { startSec: start, endSec: end } },
            }));
    }
    /*
     * take ↔ root 的交叉覆盖：只有调用方给出同源 clips 时才能把投影帧
     * 反算回源域；否则保留原块，避免在没有足够上下文时破坏几何。
     */
    if (
        context !== undefined &&
        (existing.geometry.kind === "take" || existing.geometry.kind === "root") &&
        (target.geometry.kind === "take" || target.geometry.kind === "root")
    ) {
        return trimCrossGeometry(existing, target, context.clips, context.framePeriodMs);
    }
    return [existing];
}

function trimCrossGeometry(
    existing: UserParamBlock,
    target: UserParamBlock,
    clips: readonly ParamBlockSourceClip[],
    framePeriodMs: number,
): UserParamBlock[] {
    const existingView = composeParamBlockView(
        "",
        clips,
        { schemaVersion: 2, revision: 0, blocks: [existing] },
        framePeriodMs,
    );
    const targetView = composeParamBlockView(
        "",
        clips,
        { schemaVersion: 2, revision: 0, blocks: [target] },
        framePeriodMs,
    );
    const cuts = targetView.blocks
        .filter((block) => block.kind !== "base")
        .map((block) => [block.startFrame, block.endFrameExclusive] as const);
    if (cuts.length === 0) return [existing];
    const pieces = existingView.blocks
        .filter((block) => block.kind !== "base")
        .flatMap((block) => {
            let spans: Array<[number, number]> = [[block.startFrame, block.endFrameExclusive]];
            for (const [cutStart, cutEnd] of cuts) {
                spans = spans.flatMap(([start, end]) => {
                    if (cutEnd <= start || cutStart >= end)
                        return [[start, end] as [number, number]];
                    const out: Array<[number, number]> = [];
                    if (start < cutStart) out.push([start, cutStart]);
                    if (end > cutEnd) out.push([cutEnd, end]);
                    return out;
                });
            }
            return spans;
        });
    if (pieces.length === 0) return [];
    if (existing.geometry.kind === "root") {
        return pieces.map(([start, end], index) => ({
            ...existing,
            id: rootPieceId(
                existing,
                index,
                toSec(start, framePeriodMs),
                toSec(end, framePeriodMs),
            ),
            geometry: {
                kind: "root",
                span: { startSec: toSec(start, framePeriodMs), endSec: toSec(end, framePeriodMs) },
            },
        }));
    }
    const sourcePieces: TakeSpan[] = [];
    for (const [start, end] of pieces) {
        const member = existingView.blocks
            .flatMap((block) => block.members)
            .find((item) => item.startFrame < end && item.endFrameExclusive > start);
        if (member === undefined) continue;
        const projection = projectionOfSpan(clips, memberSpan(member));
        if (projection === null) continue;
        const a = sourceAtProjectedFrame(projection, start, member.cycle, framePeriodMs);
        const b = sourceAtProjectedFrame(projection, end, member.cycle, framePeriodMs);
        if (a === null || b === null) continue;
        sourcePieces.push({
            ...memberSpan(member, member.label),
            sourceStartSec: Math.min(a, b),
            sourceEndSec: Math.max(a, b),
        });
    }
    return sourcePieces.length === 0
        ? []
        : [{ ...existing, geometry: { kind: "take", members: sourcePieces } }];
}

function memberSpan(member: ProjectedMember, label?: TakeSpan["label"]): TakeSpan {
    return {
        clipId: member.clipId,
        takeId: member.takeId,
        cycle: member.cycle,
        sourceStartSec: member.sourceStartSec,
        sourceEndSec: member.sourceEndSec,
        ...(member.syllableId === undefined ? {} : { syllableId: member.syllableId }),
        label: label ?? member.label ?? "manual",
    };
}

function projectedMembersToSpans(
    _clips: readonly ParamBlockSourceClip[],
    block: ProjectedParamBlock,
    _framePeriodMs: number,
): TakeSpan[] {
    if (block.members.length === 0) throw new Error("block has no source");
    const out: TakeSpan[] = [];
    for (const member of block.members) {
        const start = member.visibleSourceStartSec;
        const end = member.visibleSourceEndSec;
        if (!(end > start + EPS)) continue;
        const full =
            Math.abs(start - member.sourceStartSec) <= 1e-7 &&
            Math.abs(end - member.sourceEndSec) <= 1e-7;
        out.push({
            clipId: member.clipId,
            takeId: member.takeId,
            cycle: member.cycle,
            sourceStartSec: start,
            sourceEndSec: end,
            // 片段（被视口/占用裁过）不再拥有完整基块身份；保留 syllableId 会让
            // BaseCover.full 误伤同一基块的其他片段。
            ...(full && member.syllableId !== undefined ? { syllableId: member.syllableId } : {}),
            label: member.label ?? "manual",
        });
    }
    if (out.length === 0) throw new Error("block has no source");
    return out;
}

/**
 * 多 run 用户块的片段物化：只保留被点中的 run，父块剩余成员成为兄弟块。
 * 这是 Loop / 非连续源域块可编辑的前提。
 */
export function materializeProjectedFragment(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    framePeriodMs: number,
): { edits: ParamBlockEditSet; user: UserParamBlock } {
    const parentId = block.userBlockId ?? block.id;
    const parent = edits.blocks.find((item) => item.id === parentId);
    if (parent === undefined) throw new Error("block not found");
    if (parent.geometry.kind === "group") return { edits, user: parent };
    const fragment = projectedMembersToSpans(clips, block, framePeriodMs);
    if (parent.geometry.kind === "take" && sameMemberSet(parent.geometry.members, fragment)) {
        return { edits, user: parent };
    }

    let next: ParamBlockEditSet = {
        ...edits,
        revision: edits.revision + 1,
        blocks: edits.blocks.filter((item) => item.id !== parent.id),
    };
    if (parent.geometry.kind === "take") {
        const remaining: TakeSpan[] = [];
        for (const member of parent.geometry.members) {
            let pieces: Array<[number, number]> = [[member.sourceStartSec, member.sourceEndSec]];
            for (const cut of fragment) {
                if (
                    cut.clipId !== member.clipId ||
                    cut.takeId !== member.takeId ||
                    cut.cycle !== member.cycle
                ) {
                    continue;
                }
                const split: Array<[number, number]> = [];
                for (const [start, end] of pieces) {
                    split.push(
                        ...subtractInterval(start, end, cut.sourceStartSec, cut.sourceEndSec),
                    );
                }
                pieces = split;
            }
            for (const [start, end] of pieces) {
                if (end > start + EPS) {
                    remaining.push({ ...member, sourceStartSec: start, sourceEndSec: end });
                }
            }
        }
        if (remaining.length > 0) {
            next = placeUserBlock(next, {
                ...parent,
                geometry: { kind: "take", members: remaining },
            });
        }
    }

    const childId =
        block.id === parent.id || next.blocks.some((item) => item.id === block.id)
            ? newParamBlockId("fragment")
            : block.id;
    /*
     * 父块的 baseCover 不能整份复制给单 run 片段：多 run / Loop 块只编辑一个
     * run 时，其余 run 仍需露出基块。按 fragment 成员裁剪源域，只有片段真正
     * 覆盖整个基块 span 时才保留 full:true。
     */
    const childBaseCover = narrowBaseCoverToMembers(parent.baseCover, fragment);
    const child: UserParamBlock = {
        id: childId,
        geometry: { kind: "take", members: fragment },
        deleted: false,
        manual: parent.manual === true,
        value: parent.value ?? null,
        ...(parent.transitionOutSec === undefined
            ? {}
            : { transitionOutSec: parent.transitionOutSec }),
        createdSeq: nextSeq(next),
        ...(childBaseCover === undefined ? {} : { baseCover: childBaseCover }),
    };
    return { edits: placeUserBlock(next, child), user: child };
}

/** 确保一个投影块可以按普通用户块编辑；base 与多 run 片段在这里物化。 */
export function ensureEditableUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    framePeriodMs: number,
): { edits: ParamBlockEditSet; user: UserParamBlock } {
    const direct = edits.blocks.find((item) => item.id === block.id);
    if (direct !== undefined) return { edits, user: direct };
    if (block.userBlockId !== undefined && block.userBlockId !== block.id) {
        return materializeProjectedFragment(clips, edits, block, framePeriodMs);
    }
    /*
     * 基块候选自带 baseCover：它的 span 就是“我替代了哪一段基块”。直接用它，
     * 而不是再从可见帧反推 —— 反推会在 Loop 回绕/片段裁剪时丢掉完整身份。
     */
    const members =
        block.kind === "base" && block.baseCover !== undefined
            ? block.baseCover.spans.map((span) => ({ ...span }))
            : projectedMembersToSpans(clips, block, framePeriodMs);
    const target: UserParamBlock = {
        id: block.userBlockId ?? block.id,
        geometry: { kind: "take", members },
        deleted: false,
        manual: block.kind === "manual",
        value: block.value ?? null,
        ...(block.transitionFrames === undefined
            ? {}
            : { transitionOutSec: toSec(block.transitionFrames, framePeriodMs) }),
        createdSeq: nextSeq(edits),
        ...(block.baseCover === undefined ? {} : { baseCover: block.baseCover }),
    };
    return {
        edits: placeUserBlock(edits, target, { clips, framePeriodMs }),
        user: target,
    };
}

function firstProjectedFrame(
    clips: readonly ParamBlockSourceClip[],
    block: ParamBlockSlim,
    framePeriodMs: number,
): number {
    if (block.geometry.kind === "root") return toFrame(block.geometry.span.startSec, framePeriodMs);
    if (block.geometry.kind === "group") {
        return Math.min(
            ...block.geometry.children.map((child) =>
                firstProjectedFrame(clips, child, framePeriodMs),
            ),
        );
    }
    let best = Number.MAX_SAFE_INTEGER;
    for (const span of block.geometry.members) {
        const projection = projectionOfSpan(clips, span);
        if (projection === null) continue;
        const interval = projectTakeSpan(projection, span, framePeriodMs).find(
            (item) => item.cycle === span.cycle,
        );
        if (interval !== undefined) best = Math.min(best, interval.startFrame);
    }
    return best;
}

type ParamBlockSlim = {
    readonly id: string;
    readonly geometry: BlockGeometry;
    readonly deleted?: boolean;
    readonly manual?: boolean;
    readonly value?: number | null;
    readonly transitionOutSec?: number;
    readonly createdSeq: number;
    readonly baseCover?: UserParamBlock["baseCover"];
};

function replaceTopLevelBlocks(
    edits: ParamBlockEditSet,
    removeIds: ReadonlySet<string>,
    replacements: readonly UserParamBlock[],
): ParamBlockEditSet {
    const blocks = edits.blocks.filter((block) => !removeIds.has(block.id));
    blocks.push(...replacements);
    return {
        schemaVersion: 2,
        revision: edits.revision + 1,
        blocks,
    };
}

/** 同一次组/删除操作里，多 run 片段必须收敛到同一个逻辑父块。 */
function ensureLogicalUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    framePeriodMs: number,
): { edits: ParamBlockEditSet; user: UserParamBlock } {
    const parentId = block.userBlockId ?? block.id;
    const direct = edits.blocks.find((item) => item.id === parentId);
    if (direct !== undefined) return { edits, user: direct };
    return ensureEditableUserBlock(clips, edits, block, framePeriodMs);
}

function groupValue(children: readonly UserParamBlock[]): number | null {
    const first = children[0]?.value ?? null;
    return children.every((child) => (child.value ?? null) === first) ? first : null;
}

/** 组是逻辑容器，不是连续区间；但两个可见 run 真重叠时无法定义唯一并集。 */
function projectedBlocksOverlap(blocks: readonly ProjectedParamBlock[]): boolean {
    const ordered = blocks
        .slice()
        .sort((a, b) => a.startFrame - b.startFrame || a.endFrameExclusive - b.endFrameExclusive);
    for (let index = 1; index < ordered.length; index += 1) {
        if (ordered[index]!.startFrame < ordered[index - 1]!.endFrameExclusive) return true;
    }
    return false;
}

/** 可逆编组：children 是唯一几何真值；组本身不保存合并成员。 */
export function groupProjectedBlocks(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    selected: readonly ProjectedParamBlock[],
    framePeriodMs: number,
): ParamBlockEditSet {
    if (selected.length < 2) throw new Error("group needs two blocks");
    if (projectedBlocksOverlap(selected)) throw new Error("group blocks overlap");
    const byLogical = new Map<string, ProjectedParamBlock>();
    for (const block of selected) {
        const id = logicalBlockId(block);
        const existing = byLogical.get(id);
        if (existing === undefined) {
            byLogical.set(id, { ...block, id, userBlockId: id });
        } else {
            byLogical.set(id, {
                ...existing,
                startFrame: Math.min(existing.startFrame, block.startFrame),
                endFrameExclusive: Math.max(existing.endFrameExclusive, block.endFrameExclusive),
                members: [...existing.members, ...block.members],
            });
        }
    }
    const ordered = [...byLogical.values()].sort(
        (a, b) => a.startFrame - b.startFrame || a.endFrameExclusive - b.endFrameExclusive,
    );
    if (ordered.length < 2) throw new Error("group needs two blocks");

    let working = edits;
    const removeIds = new Set<string>();
    const children: UserParamBlock[] = [];
    for (const block of ordered) {
        const ensured = ensureLogicalUserBlock(clips, working, block, framePeriodMs);
        working = ensured.edits;
        removeIds.add(ensured.user.id);
        const flattened =
            ensured.user.geometry.kind === "group"
                ? ensured.user.geometry.children.slice()
                : [ensured.user];
        for (const child of flattened) {
            if (!children.some((existing) => existing.id === child.id)) children.push(child);
        }
    }
    if (children.length < 2) throw new Error("blocks are already grouped");
    children.sort(
        (a, b) =>
            firstProjectedFrame(clips, a, framePeriodMs) -
            firstProjectedFrame(clips, b, framePeriodMs),
    );
    const hasRoot = children.some((block) => block.geometry.kind === "root");
    const hasTake = children.some((block) => block.geometry.kind === "take");
    if (hasRoot && hasTake) throw new Error("cannot group root and take blocks together");
    if (children.some((block) => block.geometry.kind === "group")) throw new Error("nested group");

    const target: UserParamBlock = {
        id: newParamBlockId("group"),
        geometry: { kind: "group", children },
        deleted: false,
        manual: children.some((block) => block.manual === true),
        value: groupValue(children),
        ...(children[children.length - 1]?.transitionOutSec === undefined
            ? {}
            : { transitionOutSec: children[children.length - 1].transitionOutSec }),
        createdSeq: nextSeq(working),
    };
    return replaceTopLevelBlocks(working, removeIds, [target]);
}
export function ungroupUserBlock(edits: ParamBlockEditSet, blockId: string): ParamBlockEditSet {
    const block = edits.blocks.find((item) => item.id === blockId);
    if (block === undefined) throw new Error("block not found");
    if (block.geometry.kind !== "group" || block.geometry.children.length < 2) {
        throw new Error("block is not grouped");
    }
    return replaceTopLevelBlocks(edits, new Set([block.id]), block.geometry.children);
}

export type ParamBlockPlanErrorCode =
    | "block_not_found"
    | "block_deleted"
    | "split_outside_block"
    | "split_too_close_to_edge"
    | "split_at_block_edge"
    | "resize_outside_clip"
    | "resize_too_short"
    | "resize_blocked_by_other_source"
    | "loop_span_crosses_cycle"
    | "internal";

export class ParamBlockPlanError extends Error {
    readonly code: ParamBlockPlanErrorCode;
    readonly detail?: string;

    constructor(code: ParamBlockPlanErrorCode, detail?: string) {
        super(detail === undefined ? code : `${code}: ${detail}`);
        this.name = "ParamBlockPlanError";
        this.code = code;
        this.detail = detail;
    }
}

export interface FrameSpan {
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

export interface SplitProjectedBlockPlan {
    readonly edits: ParamBlockEditSet;
    readonly viewAfter: ReturnType<typeof composeParamBlockView>;
    readonly leftBlockId: string;
    readonly rightBlockId: string;
    readonly leftSpan: FrameSpan;
    readonly rightSpan: FrameSpan;
}

export interface ResizeProjectedBlockPlan {
    readonly edits: ParamBlockEditSet;
    readonly viewAfter: ReturnType<typeof composeParamBlockView>;
    /** 目标块的新区间；null = 目标块已被删除。 */
    readonly targetSpan: FrameSpan | null;
    readonly removedIds: readonly string[];
    readonly warnings: readonly string[];
    /** 目标边缘是否因 Clip / 周期 / 不同源邻居而被钳制。 */
    readonly clamped: boolean;
}

function planView(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    framePeriodMs: number,
) {
    return composeParamBlockView("", clips, edits, framePeriodMs);
}

function findViewBlock(
    view: ReturnType<typeof composeParamBlockView>,
    blockId: string,
): ProjectedParamBlock {
    const block = view.blocks.find((item) => item.id === blockId);
    if (block === undefined) throw new ParamBlockPlanError("block_not_found");
    return block;
}

function spanOfBlock(block: ProjectedParamBlock): FrameSpan {
    return { startFrame: block.startFrame, endFrameExclusive: block.endFrameExclusive };
}

function projectedSpanOfMember(member: ProjectedMember): TakeSpan {
    return memberSpan(member, member.label);
}

function projectionOfSpan(
    clips: readonly ParamBlockSourceClip[],
    span: TakeSpan,
): ReturnType<typeof projectionForClip> | null {
    const clip = clips.find((item) => item.id === span.clipId);
    return clip === undefined ? null : projectionForClip(clip);
}

function orderSpansByProjection(
    clips: readonly ParamBlockSourceClip[],
    spans: readonly TakeSpan[],
    framePeriodMs: number,
): TakeSpan[] {
    const rank = (span: TakeSpan): number => {
        const projection = projectionOfSpan(clips, span);
        if (projection === null) return Number.MAX_SAFE_INTEGER;
        const interval = projectTakeSpan(projection, span, framePeriodMs).find(
            (item) => item.cycle === span.cycle,
        );
        return interval?.startFrame ?? Number.MAX_SAFE_INTEGER;
    };
    return spans.slice().sort((a, b) => rank(a) - rank(b));
}

function normalizeSpanForProjection(
    clips: readonly ParamBlockSourceClip[],
    span: TakeSpan,
): TakeSpan[] {
    const projection = projectionOfSpan(clips, span);
    return projection === null ? [span] : normalizeLoopTakeSpans(projection, span);
}

function sourceAtMemberFrame(
    clips: readonly ParamBlockSourceClip[],
    member: ProjectedMember,
    frame: number,
    framePeriodMs: number,
): number {
    const projection = projectionOfSpan(clips, memberSpan(member));
    if (projection === null) throw new ParamBlockPlanError("internal", "member clip missing");
    const source = sourceAtProjectedFrame(projection, frame, member.cycle, framePeriodMs);
    if (source === null) throw new ParamBlockPlanError("internal", "source at frame");
    return source;
}

function memberIsReversed(
    member: ProjectedMember,
    clips: readonly ParamBlockSourceClip[],
): boolean {
    if (member.frameSourceStartSec !== member.frameSourceEndSec) {
        return member.frameSourceStartSec > member.frameSourceEndSec;
    }
    const projection = projectionOfSpan(clips, memberSpan(member));
    return projection?.reversed === true;
}

function minSourceWidth(
    clips: readonly ParamBlockSourceClip[],
    member: ProjectedMember,
    framePeriodMs: number,
): number {
    const projection = projectionOfSpan(clips, memberSpan(member));
    return (framePeriodMs / 1000) * Math.max(1e-9, projection?.rate ?? 1);
}

function replaceBlock(
    edits: ParamBlockEditSet,
    parentId: string,
    replacements: readonly UserParamBlock[],
): ParamBlockEditSet {
    let next: ParamBlockEditSet = {
        ...edits,
        revision: edits.revision + 1,
        blocks: edits.blocks.filter((block) => block.id !== parentId),
    };
    for (const block of replacements) next = placeUserBlock(next, block);
    return next;
}

function splitRootPlan(
    working: ParamBlockEditSet,
    user: UserParamBlock,
    frame: number,
    framePeriodMs: number,
    clips: readonly ParamBlockSourceClip[],
): SplitProjectedBlockPlan {
    const span = user.geometry.kind === "root" ? user.geometry.span : null;
    if (span === null) throw new ParamBlockPlanError("internal");
    const splitSec = toSec(frame, framePeriodMs);
    if (splitSec <= span.startSec + EPS || splitSec >= span.endSec - EPS) {
        throw new ParamBlockPlanError("split_at_block_edge");
    }
    const left: UserParamBlock = {
        id: `split:${user.id}:${frame}:left`,
        geometry: { kind: "root", span: { startSec: span.startSec, endSec: splitSec } },
        deleted: false,
        manual: user.manual,
        value: user.value,
        transitionOutSec: 0,
        createdSeq: nextSeq(working),
    };
    const right: UserParamBlock = {
        id: `split:${user.id}:${frame}:right`,
        geometry: { kind: "root", span: { startSec: splitSec, endSec: span.endSec } },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(working) + 1,
    };
    const edits = replaceBlock(working, user.id, [left, right]);
    const viewAfter = planView(clips, edits, framePeriodMs);
    const leftBlock = viewAfter.blocks.find((block) => block.id === left.id);
    const rightBlock = viewAfter.blocks.find((block) => block.id === right.id);
    if (leftBlock === undefined || rightBlock === undefined) {
        throw new ParamBlockPlanError("internal", "split result missing");
    }
    return {
        edits,
        viewAfter,
        leftBlockId: left.id,
        rightBlockId: right.id,
        leftSpan: spanOfBlock(leftBlock),
        rightSpan: spanOfBlock(rightBlock),
    };
}

/**
 * 分割计划：以**工程投影顺序**的成员列表为准。
 *
 * - 切点在成员内部：只切该成员；两侧至少各 1 帧，否则报 too-close；
 * - 切点在成员边界/成员间缝隙：按投影顺序把成员分成左右两组，不产生 1 帧碎片；
 * - 切点在块边界：报 at-block-edge。
 */
function groupSideBlock(
    original: UserParamBlock,
    children: UserParamBlock[],
    framePeriodMs: number,
    clips: readonly ParamBlockSourceClip[],
    createdSeq: number,
    stableId?: string,
): UserParamBlock {
    children.sort(
        (a, b) =>
            firstProjectedFrame(clips, a, framePeriodMs) -
            firstProjectedFrame(clips, b, framePeriodMs),
    );
    if (children.length === 1) {
        const child = children[0];
        return {
            ...child,
            ...(original.value === null || original.value === undefined
                ? {}
                : { value: original.value }),
        };
    }
    return {
        id: stableId ?? newParamBlockId("group"),
        geometry: { kind: "group", children },
        deleted: false,
        manual: children.some((child) => child.manual === true),
        value: original.value ?? groupValue(children),
        ...(original.transitionOutSec === undefined
            ? {}
            : { transitionOutSec: original.transitionOutSec }),
        createdSeq,
    };
}

function runMembersForUser(
    clips: readonly ParamBlockSourceClip[],
    user: UserParamBlock,
    framePeriodMs: number,
): ProjectedParamBlock[] {
    const view = composeParamBlockView(
        "",
        clips,
        { schemaVersion: 2, revision: 0, blocks: [user] },
        framePeriodMs,
    );
    return view.blocks.filter((block) => logicalBlockId(block) === user.id);
}

function projectedMemberKey(member: TakeSpan): string {
    return takeSpanKey(member);
}

function memberProjectionOfRun(
    run: ProjectedParamBlock,
    span: TakeSpan,
): ProjectedMember | undefined {
    return run.members.find(
        (member) =>
            member.clipId === span.clipId &&
            member.takeId === span.takeId &&
            member.cycle === span.cycle &&
            Math.abs(member.sourceStartSec - span.sourceStartSec) <= 1e-7 &&
            Math.abs(member.sourceEndSec - span.sourceEndSec) <= 1e-7,
    );
}

/**
 * 把 baseCover 按分割帧拆成左右两份。
 *
 * baseCover 是源域所有权，不是可见几何；分割块时必须把“我替代了哪段基块”
 * 也一分为二。跨切点的 span 两侧都降为 full:false：full:true 的语义是
 * “整个基块都被替代”，一旦分成两半就不再成立。
 */
export function splitBaseCoverAtFrame(
    clips: readonly ParamBlockSourceClip[],
    cover: BaseCover | undefined,
    frame: number,
    framePeriodMs: number,
): { readonly left?: BaseCover; readonly right?: BaseCover } {
    if (cover === undefined) return {};
    const leftSpans: TakeSpan[] = [];
    const rightSpans: TakeSpan[] = [];
    let leftFull = cover.full;
    let rightFull = cover.full;
    for (const span of cover.spans) {
        const clip = clips.find((item) => item.id === span.clipId);
        if (clip === undefined) {
            leftSpans.push({ ...span });
            rightSpans.push({ ...span });
            leftFull = false;
            rightFull = false;
            continue;
        }
        const projection = projectionForClip(clip);
        /*
         * 一个 span 只属于它自己的循环周期。projectTakeSpan 会返回该成员在
         * 所有周期上的投影；不过滤会把 cycle 0 的切点当成 cycle 1 的切点，
         * 把左 / 右 cover 的源域裁错。
         */
        const intervals = projectTakeSpan(projection, span, framePeriodMs).filter(
            (interval) => interval.cycle === span.cycle,
        );
        if (intervals.length === 0) {
            // 该周期没有可见投影（旧数据 / 已越界）：不把 span 同时塞到两侧，
            // 也不宣称 full；否则会凭空抑制两侧基块。
            leftFull = false;
            rightFull = false;
            continue;
        }
        let crossing: { startFrame: number; endFrameExclusive: number; cycle: number } | null =
            null;
        let side: "left" | "right" | "both" | null = null;
        for (const interval of intervals) {
            if (interval.startFrame < frame && frame < interval.endFrameExclusive) {
                crossing = interval;
                break;
            }
            if (interval.endFrameExclusive <= frame) side = side === "right" ? "both" : "left";
            else if (interval.startFrame >= frame) side = side === "left" ? "both" : "right";
        }
        if (crossing !== null) {
            const a = sourceAtProjectedFrame(
                projection,
                crossing.startFrame,
                crossing.cycle,
                framePeriodMs,
            );
            const b = sourceAtProjectedFrame(
                projection,
                crossing.endFrameExclusive,
                crossing.cycle,
                framePeriodMs,
            );
            const cut = sourceAtProjectedFrame(projection, frame, crossing.cycle, framePeriodMs);
            if (a !== null && b !== null && cut !== null) {
                const lo = Math.min(a, b);
                const hi = Math.max(a, b);
                const cutClamped = Math.max(lo, Math.min(hi, cut));
                /*
                 * Forward playback: visible time increases with source, so the
                 * left half is [lo, cut]. Reversed playback is the opposite:
                 * the left half is [cut, hi]. This must match the member split
                 * in splitLogicalUserBlock.
                 */
                const forward = a <= b;
                leftSpans.push(
                    forward
                        ? { ...span, sourceStartSec: lo, sourceEndSec: cutClamped }
                        : { ...span, sourceStartSec: cutClamped, sourceEndSec: hi },
                );
                rightSpans.push(
                    forward
                        ? { ...span, sourceStartSec: cutClamped, sourceEndSec: hi }
                        : { ...span, sourceStartSec: lo, sourceEndSec: cutClamped },
                );
                leftFull = false;
                rightFull = false;
                continue;
            }
        }
        if (side === "left") leftSpans.push({ ...span });
        else if (side === "right") rightSpans.push({ ...span });
        else {
            leftSpans.push({ ...span });
            rightSpans.push({ ...span });
            leftFull = false;
            rightFull = false;
        }
    }
    return {
        ...(leftSpans.length === 0 ? {} : { left: { spans: leftSpans, full: leftFull } }),
        ...(rightSpans.length === 0 ? {} : { right: { spans: rightSpans, full: rightFull } }),
    };
}

export function splitLogicalUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): SplitProjectedBlockPlan {
    const parentId = block.userBlockId ?? block.id;
    const parent = edits.blocks.find((item) => item.id === parentId);
    if (parent === undefined || parent.geometry.kind === "group") {
        throw new ParamBlockPlanError("block_not_found");
    }
    if (frame <= block.startFrame || frame >= block.endFrameExclusive) {
        throw new ParamBlockPlanError("split_outside_block");
    }
    if (parent.geometry.kind === "root") {
        return splitRootPlan(edits, parent, frame, framePeriodMs, clips);
    }
    const logical = projectLogicalUserBlock(clips, parent, framePeriodMs);
    const members = logical?.members ?? block.members;
    if (members.length === 0) throw new ParamBlockPlanError("internal", "no members");
    const insideIndex = members.findIndex(
        (member) => member.startFrame < frame && frame < member.endFrameExclusive,
    );
    let leftMembers: TakeSpan[];
    let rightMembers: TakeSpan[];
    if (insideIndex >= 0) {
        const hit = members[insideIndex];
        const leftFrames = frame - hit.startFrame;
        const rightFrames = hit.endFrameExclusive - frame;
        if (leftFrames < 1 || rightFrames < 1 || hit.endFrameExclusive - hit.startFrame <= 1) {
            throw new ParamBlockPlanError(
                "split_too_close_to_edge",
                leftFrames + "/" + rightFrames,
            );
        }
        const source = sourceAtMemberFrame(clips, hit, frame, framePeriodMs);
        const reversed = memberIsReversed(hit, clips);
        const leftPart: TakeSpan = {
            ...projectedSpanOfMember(hit),
            sourceStartSec: reversed ? source : hit.sourceStartSec,
            sourceEndSec: reversed ? hit.sourceEndSec : source,
        };
        const rightPart: TakeSpan = {
            ...projectedSpanOfMember(hit),
            sourceStartSec: reversed ? hit.sourceStartSec : source,
            sourceEndSec: reversed ? source : hit.sourceEndSec,
        };
        leftMembers = [...members.slice(0, insideIndex).map(projectedSpanOfMember), leftPart];
        rightMembers = [rightPart, ...members.slice(insideIndex + 1).map(projectedSpanOfMember)];
    } else {
        leftMembers = members
            .filter((member) => member.endFrameExclusive <= frame)
            .map(projectedSpanOfMember);
        rightMembers = members
            .filter((member) => member.startFrame >= frame)
            .map(projectedSpanOfMember);
        if (leftMembers.length === 0 || rightMembers.length === 0) {
            throw new ParamBlockPlanError("split_at_block_edge");
        }
    }
    const splitCover = splitBaseCoverAtFrame(clips, parent.baseCover, frame, framePeriodMs);
    const left: UserParamBlock = {
        id: "split:" + parent.id + ":" + frame + ":left",
        geometry: { kind: "take", members: leftMembers },
        deleted: false,
        manual: parent.manual,
        value: parent.value,
        transitionOutSec: 0,
        createdSeq: nextSeq(edits),
        ...(splitCover.left === undefined ? {} : { baseCover: splitCover.left }),
    };
    const right: UserParamBlock = {
        id: "split:" + parent.id + ":" + frame + ":right",
        geometry: { kind: "take", members: rightMembers },
        deleted: false,
        manual: parent.manual,
        value: parent.value,
        ...(parent.transitionOutSec === undefined
            ? {}
            : { transitionOutSec: parent.transitionOutSec }),
        createdSeq: nextSeq(edits) + 1,
        ...(splitCover.right === undefined ? {} : { baseCover: splitCover.right }),
    };
    const next = replaceBlock(edits, parent.id, [left, right]);
    const viewAfter = planView(clips, next, framePeriodMs);
    const leftRuns = viewAfter.blocks.filter((item) => logicalBlockId(item) === left.id);
    const rightRuns = viewAfter.blocks.filter((item) => logicalBlockId(item) === right.id);
    if (leftRuns.length === 0 || rightRuns.length === 0) {
        throw new ParamBlockPlanError("internal", "split result missing");
    }
    return {
        edits: next,
        viewAfter,
        leftBlockId: left.id,
        rightBlockId: right.id,
        leftSpan: {
            startFrame: Math.min(...leftRuns.map((item) => item.startFrame)),
            endFrameExclusive: Math.max(...leftRuns.map((item) => item.endFrameExclusive)),
        },
        rightSpan: {
            startFrame: Math.min(...rightRuns.map((item) => item.startFrame)),
            endFrameExclusive: Math.max(...rightRuns.map((item) => item.endFrameExclusive)),
        },
    };
}

export function resizeLogicalUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ResizeProjectedBlockPlan {
    const parentId = block.userBlockId ?? block.id;
    const parent = edits.blocks.find((item) => item.id === parentId);
    if (parent === undefined || parent.geometry.kind !== "take") {
        throw new ParamBlockPlanError("block_not_found");
    }
    const runs = runMembersForUser(clips, parent, framePeriodMs);
    const run =
        runs.find((item) => item.id === block.id) ??
        runs.find(
            (item) => item.startFrame <= targetFrame && item.endFrameExclusive >= targetFrame,
        ) ??
        (edge === "right" ? runs[runs.length - 1] : runs[0]);
    if (run === undefined) throw new ParamBlockPlanError("internal", "run not found");
    const runKeys = new Set(
        run.members.map((member) => projectedMemberKey(memberSpan(member, member.label))),
    );
    let nextMembers = parent.geometry.members.map((member) => ({ ...member }));
    const warnings: string[] = [];
    let clamped = false;
    /*
     * 多 run 逻辑块的伸缩也必须遵守 Clip / Loop 周期边界与相邻块占用。
     * 旧实现把目标帧直接交给 sourceAtMemberFrame，越过 Clip 末端时得到
     * null 后抛 internal；用户看到的是“拖到边缘就失败”。先把目标钳到
     * 合法边界，再走原来的成员改写。
     */
    const viewBefore = planView(clips, edits, framePeriodMs);
    const neighbor = nearestOtherBlock(viewBefore, block, edge);
    const edgeMember = edge === "right" ? run.members[run.members.length - 1] : run.members[0];
    let effectiveTargetFrame = targetFrame;
    if (edgeMember !== undefined) {
        const clip = clips.find((item) => item.id === edgeMember.clipId);
        if (clip !== undefined) {
            const bounds = findEdgeClipBounds(
                clip,
                projectionForClip(clip),
                edgeMember.cycle,
                framePeriodMs,
            );
            if (edge === "right") {
                effectiveTargetFrame = Math.min(
                    effectiveTargetFrame,
                    bounds.maxFrame,
                    neighbor?.startFrame ?? Number.POSITIVE_INFINITY,
                );
            } else {
                effectiveTargetFrame = Math.max(
                    effectiveTargetFrame,
                    bounds.minFrame,
                    neighbor?.endFrameExclusive ?? Number.NEGATIVE_INFINITY,
                );
            }
        }
    }
    if (effectiveTargetFrame !== targetFrame) {
        clamped = true;
        warnings.push("resize_clamped_to_edge");
    }
    const targetFrameClamped = effectiveTargetFrame;
    if (edge === "right") {
        if (targetFrameClamped >= run.endFrameExclusive) {
            const last = run.members[run.members.length - 1];
            if (last === undefined) throw new ParamBlockPlanError("internal", "run has no members");
            const lastKey = projectedMemberKey(memberSpan(last, last.label));
            const index = nextMembers.findIndex((member) => projectedMemberKey(member) === lastKey);
            if (index < 0) throw new ParamBlockPlanError("internal", "member not found");
            const source = sourceAtMemberFrame(clips, last, targetFrameClamped, framePeriodMs);
            const reversed = memberIsReversed(last, clips);
            if (reversed) {
                if (source < nextMembers[index].sourceStartSec) {
                    nextMembers[index] = { ...nextMembers[index], sourceStartSec: source };
                }
            } else if (source > nextMembers[index].sourceEndSec) {
                nextMembers[index] = { ...nextMembers[index], sourceEndSec: source };
            }
        } else {
            nextMembers = nextMembers.flatMap((member) => {
                if (!runKeys.has(projectedMemberKey(member))) return [member];
                const projected = memberProjectionOfRun(run, member);
                if (projected === undefined) return [member];
                if (projected.endFrameExclusive <= targetFrameClamped) return [member];
                if (projected.startFrame >= targetFrameClamped) return [];
                const source = sourceAtMemberFrame(
                    clips,
                    projected,
                    targetFrameClamped,
                    framePeriodMs,
                );
                const reversed = memberIsReversed(projected, clips);
                return [
                    reversed
                        ? { ...member, sourceStartSec: source }
                        : { ...member, sourceEndSec: source },
                ];
            });
        }
    } else if (targetFrameClamped <= run.startFrame) {
        const first = run.members[0];
        if (first === undefined) throw new ParamBlockPlanError("internal", "run has no members");
        const index = nextMembers.findIndex((member) => runKeys.has(projectedMemberKey(member)));
        if (index < 0) throw new ParamBlockPlanError("internal", "member not found");
        const source = sourceAtMemberFrame(clips, first, targetFrameClamped, framePeriodMs);
        const reversed = memberIsReversed(first, clips);
        if (reversed) {
            if (source > nextMembers[index].sourceEndSec) {
                nextMembers[index] = { ...nextMembers[index], sourceEndSec: source };
            }
        } else if (source < nextMembers[index].sourceStartSec) {
            nextMembers[index] = { ...nextMembers[index], sourceStartSec: source };
        }
    } else {
        nextMembers = nextMembers.flatMap((member) => {
            if (!runKeys.has(projectedMemberKey(member))) return [member];
            const projected = memberProjectionOfRun(run, member);
            if (projected === undefined) return [member];
            if (projected.startFrame >= targetFrameClamped) return [member];
            if (projected.endFrameExclusive <= targetFrameClamped) return [];
            const source = sourceAtMemberFrame(clips, projected, targetFrameClamped, framePeriodMs);
            const reversed = memberIsReversed(projected, clips);
            return [
                reversed
                    ? { ...member, sourceEndSec: source }
                    : { ...member, sourceStartSec: source },
            ];
        });
    }
    const expanded = nextMembers.flatMap((span) => normalizeSpanForProjection(clips, span));
    const ordered = orderSpansByProjection(clips, expanded, framePeriodMs);
    if (ordered.length === 0) {
        const next = removeOrTombstone(edits, parent, clips, framePeriodMs);
        return {
            edits: next,
            viewAfter: planView(clips, next, framePeriodMs),
            targetSpan: null,
            removedIds: [parent.id],
            warnings,
            clamped,
        };
    }
    const next = placeUserBlock(edits, {
        ...parent,
        geometry: { kind: "take", members: ordered },
    });
    const viewAfter = planView(clips, next, framePeriodMs);
    const targetRuns = viewAfter.blocks.filter((item) => logicalBlockId(item) === parent.id);
    if (targetRuns.length === 0) throw new ParamBlockPlanError("internal", "resize result missing");
    return {
        edits: next,
        viewAfter,
        targetSpan: {
            startFrame: Math.min(...targetRuns.map((item) => item.startFrame)),
            endFrameExclusive: Math.max(...targetRuns.map((item) => item.endFrameExclusive)),
        },
        removedIds: [],
        warnings,
        clamped,
    };
}

/**
 * 把逻辑块的全部可见 run 合成一个 ProjectedParamBlock。组算法、切分、
 * 伸缩都只应该看这个整体边界，而不是第一个 run。
 */
export function projectLogicalUserBlock(
    clips: readonly ParamBlockSourceClip[],
    user: UserParamBlock,
    framePeriodMs: number,
): ProjectedParamBlock | null {
    const view = composeParamBlockView(
        "",
        clips,
        { schemaVersion: 2, revision: 0, blocks: [user] },
        framePeriodMs,
    );
    const runs = view.blocks.filter((block) => logicalBlockId(block) === user.id);
    if (runs.length === 0) return null;
    return {
        id: user.id,
        userBlockId: user.id,
        startFrame: Math.min(...runs.map((run) => run.startFrame)),
        endFrameExclusive: Math.max(...runs.map((run) => run.endFrameExclusive)),
        ...(runs[0]?.transitionFrames === undefined
            ? {}
            : { transitionFrames: runs[0].transitionFrames }),
        kind: runs[0]?.kind ?? "user",
        members: runs.flatMap((run) => run.members),
        ...(user.value === undefined ? {} : { value: user.value }),
    };
}

function splitGroupPlan(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): SplitProjectedBlockPlan {
    const groupId = block.userBlockId ?? block.id;
    const group = edits.blocks.find((candidate) => candidate.id === groupId);
    if (group === undefined || group.geometry.kind !== "group") {
        throw new ParamBlockPlanError("block_not_found");
    }
    if (frame <= block.startFrame || frame >= block.endFrameExclusive) {
        throw new ParamBlockPlanError("split_outside_block");
    }
    const ordered = group.geometry.children
        .slice()
        .sort(
            (a, b) =>
                firstProjectedFrame(clips, a, framePeriodMs) -
                firstProjectedFrame(clips, b, framePeriodMs),
        );
    const left: UserParamBlock[] = [];
    const right: UserParamBlock[] = [];
    for (const child of ordered) {
        const projected = projectLogicalUserBlock(clips, child, framePeriodMs);
        if (projected === null) continue;
        if (projected.endFrameExclusive <= frame) {
            left.push(child);
        } else if (projected.startFrame >= frame) {
            right.push(child);
        } else {
            const childPlan = planSplitProjectedBlock(
                clips,
                { schemaVersion: 2, revision: 0, blocks: [child] },
                projected,
                frame,
                framePeriodMs,
            );
            const leftChild = childPlan.edits.blocks.find(
                (candidate) => candidate.id === childPlan.leftBlockId,
            );
            const rightChild = childPlan.edits.blocks.find(
                (candidate) => candidate.id === childPlan.rightBlockId,
            );
            if (leftChild === undefined || rightChild === undefined) {
                throw new ParamBlockPlanError("internal", "group child split missing");
            }
            left.push(leftChild);
            right.push(rightChild);
        }
    }
    if (left.length === 0 || right.length === 0) {
        throw new ParamBlockPlanError("split_at_block_edge");
    }
    const leftBlock = groupSideBlock(group, left, framePeriodMs, clips, nextSeq(edits));
    const rightBlock = groupSideBlock(group, right, framePeriodMs, clips, nextSeq(edits) + 1);
    const next = replaceTopLevelBlocks(edits, new Set([group.id]), [leftBlock, rightBlock]);
    const viewAfter = planView(clips, next, framePeriodMs);
    const leftView = viewAfter.blocks.find(
        (candidate) => candidate.id === leftBlock.id || candidate.userBlockId === leftBlock.id,
    );
    const rightView = viewAfter.blocks.find(
        (candidate) => candidate.id === rightBlock.id || candidate.userBlockId === rightBlock.id,
    );
    if (leftView === undefined || rightView === undefined) {
        throw new ParamBlockPlanError("internal", "group split result missing");
    }
    return {
        edits: next,
        viewAfter,
        leftBlockId: leftBlock.id,
        rightBlockId: rightBlock.id,
        leftSpan: spanOfBlock(leftView),
        rightSpan: spanOfBlock(rightView),
    };
}

export function planSplitProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): SplitProjectedBlockPlan {
    if (block.kind === "grouped") {
        return splitGroupPlan(clips, edits, block, frame, framePeriodMs);
    }
    if (block.userBlockId !== undefined && block.userBlockId !== block.id) {
        return splitLogicalUserBlock(clips, edits, block, frame, framePeriodMs);
    }
    if (frame <= block.startFrame || frame >= block.endFrameExclusive) {
        throw new ParamBlockPlanError("split_outside_block");
    }
    const ensured = ensureEditableUserBlock(clips, edits, block, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.geometry.kind === "group") {
        return splitGroupPlan(clips, working, block, frame, framePeriodMs);
    }
    if (user.geometry.kind === "root") {
        return splitRootPlan(working, user, frame, framePeriodMs, clips);
    }
    const members = block.members;
    if (members.length === 0) throw new ParamBlockPlanError("internal", "no members");
    const insideIndex = members.findIndex(
        (member) => member.startFrame < frame && frame < member.endFrameExclusive,
    );
    let leftMembers: TakeSpan[];
    let rightMembers: TakeSpan[];
    if (insideIndex >= 0) {
        const hit = members[insideIndex];
        const leftFrames = frame - hit.startFrame;
        const rightFrames = hit.endFrameExclusive - frame;
        if (leftFrames < 1 || rightFrames < 1 || hit.endFrameExclusive - hit.startFrame <= 1) {
            throw new ParamBlockPlanError(
                "split_too_close_to_edge",
                `${leftFrames}/${rightFrames}`,
            );
        }
        const source = sourceAtMemberFrame(clips, hit, frame, framePeriodMs);
        const reversed = memberIsReversed(hit, clips);
        const leftPart: TakeSpan = {
            ...projectedSpanOfMember(hit),
            sourceStartSec: reversed ? source : hit.sourceStartSec,
            sourceEndSec: reversed ? hit.sourceEndSec : source,
        };
        const rightPart: TakeSpan = {
            ...projectedSpanOfMember(hit),
            sourceStartSec: reversed ? hit.sourceStartSec : source,
            sourceEndSec: reversed ? source : hit.sourceEndSec,
        };
        leftMembers = [...members.slice(0, insideIndex).map(projectedSpanOfMember), leftPart];
        rightMembers = [rightPart, ...members.slice(insideIndex + 1).map(projectedSpanOfMember)];
    } else {
        leftMembers = members
            .filter((member) => member.endFrameExclusive <= frame)
            .map(projectedSpanOfMember);
        rightMembers = members
            .filter((member) => member.startFrame >= frame)
            .map(projectedSpanOfMember);
        if (leftMembers.length === 0 || rightMembers.length === 0) {
            throw new ParamBlockPlanError("split_at_block_edge");
        }
    }
    const splitCover = splitBaseCoverAtFrame(clips, user.baseCover, frame, framePeriodMs);
    const left: UserParamBlock = {
        id: `split:${user.id}:${frame}:left`,
        geometry: { kind: "take", members: leftMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        transitionOutSec: 0,
        createdSeq: nextSeq(working),
        ...(splitCover.left === undefined ? {} : { baseCover: splitCover.left }),
    };
    const right: UserParamBlock = {
        id: `split:${user.id}:${frame}:right`,
        geometry: { kind: "take", members: rightMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(working) + 1,
        ...(splitCover.right === undefined ? {} : { baseCover: splitCover.right }),
    };
    const next = replaceBlock(working, user.id, [left, right]);
    const viewAfter = planView(clips, next, framePeriodMs);
    const leftBlock = viewAfter.blocks.find((item) => item.id === left.id);
    const rightBlock = viewAfter.blocks.find((item) => item.id === right.id);
    if (leftBlock === undefined || rightBlock === undefined) {
        throw new ParamBlockPlanError("internal", "split result missing");
    }
    return {
        edits: next,
        viewAfter,
        leftBlockId: left.id,
        rightBlockId: right.id,
        leftSpan: spanOfBlock(leftBlock),
        rightSpan: spanOfBlock(rightBlock),
    };
}

/**
 * 递归收集组内所有需要保留的 base 派生叶子墓碑。
 *
 * 删除 / 缩短组时不能只看直接 children：嵌套组里的 base 派生 leaf 一旦
 * 丢掉 baseCover，基块会在原位置复活。手工 leaf（manual 且无 baseCover）
 * 不生成墓碑。
 */
function collectBaseDerivedTombstones(group: UserParamBlock, startSeq: number): UserParamBlock[] {
    const tombstones: UserParamBlock[] = [];
    let seq = startSeq;
    for (const leaf of flattenGroupChildren(group)) {
        if (leaf.manual === true && leaf.baseCover === undefined) continue;
        if (leaf.baseCover?.full === false) continue;
        if (leaf.geometry.kind !== "take") continue;
        tombstones.push({
            id: "tomb:" + leaf.id,
            geometry: {
                kind: "take",
                members: leaf.geometry.members.map((member) => ({ ...member })),
            },
            deleted: true,
            manual: false,
            createdSeq: seq,
            ...(leaf.baseCover === undefined ? {} : { baseCover: leaf.baseCover }),
        });
        seq += 1;
    }
    return tombstones;
}

function removeOrTombstone(
    working: ParamBlockEditSet,
    user: UserParamBlock,
    _clips: readonly ParamBlockSourceClip[],
    _framePeriodMs: number,
): ParamBlockEditSet {
    const isManual = user.manual === true && user.baseCover === undefined;
    if (isManual || user.baseCover?.full === false) {
        return {
            ...working,
            revision: working.revision + 1,
            blocks: working.blocks.filter((block) => block.id !== user.id),
        };
    }
    const tombstone: UserParamBlock = {
        ...user,
        id: `tomb:${user.id}`,
        deleted: true,
        manual: false,
        value: null,
        transitionOutSec: undefined,
        createdSeq: nextSeq(working),
        ...(user.baseCover === undefined ? {} : { baseCover: user.baseCover }),
    };
    return replaceBlock(working, user.id, [tombstone]);
}

function rootResizePlan(
    working: ParamBlockEditSet,
    user: UserParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
    clips: readonly ParamBlockSourceClip[],
): ResizeProjectedBlockPlan {
    const span = user.geometry.kind === "root" ? user.geometry.span : null;
    if (span === null) throw new ParamBlockPlanError("internal");
    const minSec = framePeriodMs / 1000;
    const targetSec = toSec(targetFrame, framePeriodMs);
    const nextSpan: RootSpan =
        edge === "left"
            ? { startSec: Math.max(0, targetSec), endSec: span.endSec }
            : { startSec: span.startSec, endSec: Math.max(span.startSec + minSec, targetSec) };
    if (nextSpan.endSec <= nextSpan.startSec + EPS) {
        const edits = removeOrTombstone(working, user, clips, framePeriodMs);
        return {
            edits,
            viewAfter: planView(clips, edits, framePeriodMs),
            targetSpan: null,
            removedIds: [user.id],
            warnings: [],
            clamped: false,
        };
    }
    const edits = placeUserBlock(working, { ...user, geometry: { kind: "root", span: nextSpan } });
    const viewAfter = planView(clips, edits, framePeriodMs);
    const block = viewAfter.blocks.find((item) => item.id === user.id);
    if (block === undefined) throw new ParamBlockPlanError("internal", "resize result missing");
    return {
        edits,
        viewAfter,
        targetSpan: spanOfBlock(block),
        removedIds: [],
        warnings: [],
        clamped: false,
    };
}

function findEdgeClipBounds(
    clip: ParamBlockSourceClip,
    projection: ReturnType<typeof projectionForClip>,
    cycle: number,
    framePeriodMs: number,
): { minFrame: number; maxFrame: number } {
    const clipStart = toFrame(clip.startSec, framePeriodMs);
    const clipEnd = toFrame(clip.startSec + clip.lengthSec, framePeriodMs);
    if (projection.loopEnabled && projection.loopCycleSec !== null) {
        const cycleTimeline = projection.loopCycleSec / Math.max(1e-9, projection.rate);
        const cycleStart = toFrame(
            projection.projectStartSec + cycle * cycleTimeline,
            framePeriodMs,
        );
        const cycleEnd = toFrame(
            projection.projectStartSec + (cycle + 1) * cycleTimeline,
            framePeriodMs,
        );
        return {
            minFrame: Math.max(clipStart, cycleStart),
            maxFrame: Math.min(clipEnd, cycleEnd),
        };
    }
    return { minFrame: clipStart, maxFrame: clipEnd };
}

function nearestOtherBlock(
    view: ReturnType<typeof composeParamBlockView>,
    block: ProjectedParamBlock,
    edge: "left" | "right",
): ProjectedParamBlock | null {
    const candidates = view.blocks
        .filter((item) => item.id !== block.id)
        .filter((item) =>
            edge === "right"
                ? item.startFrame >= block.endFrameExclusive - 1
                : item.endFrameExclusive <= block.startFrame + 1,
        )
        .sort((a, b) =>
            edge === "right"
                ? a.startFrame - b.startFrame
                : b.endFrameExclusive - a.endFrameExclusive,
        );
    return candidates[0] ?? null;
}

function sameSource(left: ProjectedMember, right: ProjectedMember): boolean {
    return (
        left.clipId === right.clipId && left.takeId === right.takeId && left.cycle === right.cycle
    );
}

function resizeGroupPlan(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ResizeProjectedBlockPlan {
    const groupId = block.userBlockId ?? block.id;
    const group = edits.blocks.find((candidate) => candidate.id === groupId);
    if (group === undefined || group.geometry.kind !== "group") {
        throw new ParamBlockPlanError("block_not_found");
    }
    const children = group.geometry.children
        .slice()
        .sort(
            (a, b) =>
                firstProjectedFrame(clips, a, framePeriodMs) -
                firstProjectedFrame(clips, b, framePeriodMs),
        );
    if (children.length === 0) throw new ParamBlockPlanError("internal", "group has no children");
    const projectedChildren = children.map((child) => ({
        child,
        projected: projectLogicalUserBlock(clips, child, framePeriodMs),
    }));
    const projectedFirst = projectedChildren[0]?.projected;
    const projectedLast = projectedChildren[projectedChildren.length - 1]?.projected;
    if (
        projectedFirst === null ||
        projectedFirst === undefined ||
        projectedLast === null ||
        projectedLast === undefined
    ) {
        throw new ParamBlockPlanError("internal", "group child not projected");
    }
    const groupStart = Math.min(
        ...projectedChildren.map((item) => item.projected?.startFrame ?? Infinity),
    );
    const groupEnd = Math.max(
        ...projectedChildren.map((item) => item.projected?.endFrameExclusive ?? -Infinity),
    );
    if (edge === "right" && targetFrame <= groupStart + 1) {
        return deleteGroupAsResize(clips, edits, block, framePeriodMs);
    }
    if (edge === "left" && targetFrame >= groupEnd - 1) {
        return deleteGroupAsResize(clips, edits, block, framePeriodMs);
    }

    const kept: UserParamBlock[] = [];
    const warnings: string[] = [];
    let clamped = false;
    const resizeChild = (
        child: UserParamBlock,
        projected: ProjectedParamBlock,
        childEdge: "left" | "right",
    ): ResizeProjectedBlockPlan =>
        child.geometry.kind === "take"
            ? resizeLogicalUserBlock(
                  clips,
                  { schemaVersion: 2, revision: 0, blocks: [child] },
                  projected,
                  childEdge,
                  targetFrame,
                  framePeriodMs,
              )
            : planResizeProjectedBlock(
                  clips,
                  { schemaVersion: 2, revision: 0, blocks: [child] },
                  projected,
                  childEdge,
                  targetFrame,
                  framePeriodMs,
              );
    if (edge === "right") {
        for (const item of projectedChildren) {
            if (item.projected === null) continue;
            if (item.projected.endFrameExclusive <= targetFrame) {
                kept.push(item.child);
            } else if (item.projected.startFrame >= targetFrame) {
                break;
            } else {
                const plan = resizeChild(item.child, item.projected, "right");
                const result = plan.edits.blocks[0];
                if (result !== undefined) kept.push(result);
                warnings.push(...plan.warnings);
                clamped ||= plan.clamped;
                break;
            }
        }
        if (kept.length === 0) return deleteGroupAsResize(clips, edits, block, framePeriodMs);
        const last = kept[kept.length - 1];
        const lastProjected = projectLogicalUserBlock(clips, last, framePeriodMs);
        if (
            targetFrame >= groupEnd &&
            lastProjected !== null &&
            lastProjected.endFrameExclusive < targetFrame
        ) {
            kept.pop();
            const plan = resizeChild(last, lastProjected, "right");
            kept.push(...plan.edits.blocks);
            warnings.push(...plan.warnings);
            clamped ||= plan.clamped;
        }
    } else {
        for (const item of [...projectedChildren].reverse()) {
            if (item.projected === null) continue;
            if (item.projected.startFrame >= targetFrame) {
                kept.push(item.child);
            } else if (item.projected.endFrameExclusive <= targetFrame) {
                break;
            } else {
                const plan = resizeChild(item.child, item.projected, "left");
                const result = plan.edits.blocks[0];
                if (result !== undefined) kept.push(result);
                warnings.push(...plan.warnings);
                clamped ||= plan.clamped;
                break;
            }
        }
        kept.reverse();
        if (kept.length === 0) return deleteGroupAsResize(clips, edits, block, framePeriodMs);
        const first = kept[0];
        const firstProjected = projectLogicalUserBlock(clips, first, framePeriodMs);
        if (
            targetFrame <= groupStart &&
            firstProjected !== null &&
            firstProjected.startFrame > targetFrame
        ) {
            kept.shift();
            const plan = resizeChild(first, firstProjected, "left");
            kept.unshift(...plan.edits.blocks);
            warnings.push(...plan.warnings);
            clamped ||= plan.clamped;
        }
    }
    if (kept.length === 0) return deleteGroupAsResize(clips, edits, block, framePeriodMs);
    /*
     * 被缩掉的 child 不能凭空消失：base 派生 child 携带的 baseCover 是
     * "这块基块已经被替代"的唯一证据。丢掉它，compose 会把被缩掉的基块
     * 重新露出来。手工 child 没有 baseCover，直接丢弃即可。
     */
    const keptIds = new Set(kept.map((child) => child.id));
    const dropped = children.filter((child) => !keptIds.has(child.id));
    const tombstones: UserParamBlock[] = [];
    let tombSeq = nextSeq(edits);
    for (const child of dropped) {
        const childTombstones = collectBaseDerivedTombstones(child, tombSeq);
        tombstones.push(...childTombstones);
        tombSeq += childTombstones.length;
    }
    const side = groupSideBlock(
        group,
        kept,
        framePeriodMs,
        clips,
        nextSeq(edits) + tombstones.length,
        group.id,
    );
    const next = replaceTopLevelBlocks(edits, new Set([group.id]), [side, ...tombstones]);
    const viewAfter = planView(clips, next, framePeriodMs);
    const target =
        viewAfter.blocks.find(
            (candidate) => candidate.id === side.id || candidate.userBlockId === side.id,
        ) ??
        viewAfter.blocks.find(
            (candidate) =>
                candidate.startFrame < targetFrame && candidate.endFrameExclusive > targetFrame,
        );
    if (target === undefined)
        throw new ParamBlockPlanError("internal", "group resize result missing");
    const beforeIds = new Set(planView(clips, edits, framePeriodMs).blocks.map((item) => item.id));
    const afterIds = new Set(viewAfter.blocks.map((item) => item.id));
    return {
        edits: next,
        viewAfter,
        targetSpan: spanOfBlock(target),
        removedIds: [...beforeIds].filter((id) => !afterIds.has(id)),
        warnings,
        clamped,
    };
}

function deleteGroupAsResize(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    framePeriodMs: number,
): ResizeProjectedBlockPlan {
    const result = deleteProjectedBlocks(clips, edits, [block], framePeriodMs);
    return {
        edits: result.edits,
        viewAfter: planView(clips, result.edits, framePeriodMs),
        targetSpan: null,
        removedIds: [block.userBlockId ?? block.id],
        warnings: [],
        clamped: false,
    };
}

/**
 * 伸缩计划：重写**工程投影顺序**的成员区间。
 *
 * 缩短时必须删除被跨过的成员，并把落点所在成员裁到目标；扩展时只改边缘成员，
 * 由 `placeUserBlock` 按源域裁剪被覆盖的旧块。倒放时右边缘改 sourceStart、
 * 左边缘改 sourceEnd。
 */
export function planResizeProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ResizeProjectedBlockPlan {
    const viewBefore = planView(clips, edits, framePeriodMs);
    const current = findViewBlock(viewBefore, block.id);
    if (block.kind === "grouped") {
        return resizeGroupPlan(clips, edits, current, edge, targetFrame, framePeriodMs);
    }
    if (current.userBlockId !== undefined && current.userBlockId !== current.id) {
        return resizeLogicalUserBlock(clips, edits, current, edge, targetFrame, framePeriodMs);
    }
    const ensured = ensureEditableUserBlock(clips, edits, current, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.geometry.kind === "group") {
        return resizeGroupPlan(clips, working, block, edge, targetFrame, framePeriodMs);
    }
    if (user.geometry.kind === "root") {
        return rootResizePlan(working, user, edge, targetFrame, framePeriodMs, clips);
    }
    const members = current.members;
    if (members.length === 0) throw new ParamBlockPlanError("internal", "no members");
    const edgeMember = edge === "left" ? members[0] : members[members.length - 1];
    const clip = clips.find((item) => item.id === edgeMember.clipId);
    if (clip === undefined) throw new ParamBlockPlanError("internal", "member clip missing");
    const projection = projectionForClip(clip);
    const bounds = findEdgeClipBounds(clip, projection, edgeMember.cycle, framePeriodMs);
    const minWidth = 1;
    let target = targetFrame;
    let clamped = false;
    const warnings: string[] = [];

    /*
     * 不允许跨到“不同源”的邻居：那种扩展无法用源域区间裁剪表达。钳在邻居边界，
     * 并返回警告；同一 Clip/Take/cycle 的邻居可以由 placeUserBlock 正常裁剪。
     */
    const neighbor = nearestOtherBlock(viewBefore, current, edge);
    if (neighbor !== null && neighbor.members.length > 0) {
        const neighborEdge =
            edge === "right" ? neighbor.members[0] : neighbor.members[neighbor.members.length - 1];
        if (!sameSource(edgeMember, neighborEdge)) {
            if (edge === "right" && target > neighbor.startFrame) {
                target = neighbor.startFrame;
                clamped = true;
                warnings.push("resize-blocked-by-other-source");
            } else if (edge === "left" && target < neighbor.endFrameExclusive) {
                target = neighbor.endFrameExclusive;
                clamped = true;
                warnings.push("resize-blocked-by-other-source");
            }
        }
    }
    if (edge === "right") {
        const lower = current.startFrame + minWidth;
        if (target <= lower) {
            const next = removeOrTombstone(working, user, clips, framePeriodMs);
            return {
                edits: next,
                viewAfter: planView(clips, next, framePeriodMs),
                targetSpan: null,
                removedIds: [user.id],
                warnings,
                clamped,
            };
        }
        if (target > bounds.maxFrame) {
            target = bounds.maxFrame;
            clamped = true;
            warnings.push("resize-clamped-to-clip");
        }
        if (target < lower) target = lower;
    } else {
        const upper = current.endFrameExclusive - minWidth;
        if (target >= upper) {
            const next = removeOrTombstone(working, user, clips, framePeriodMs);
            return {
                edits: next,
                viewAfter: planView(clips, next, framePeriodMs),
                targetSpan: null,
                removedIds: [user.id],
                warnings,
                clamped,
            };
        }
        if (target < bounds.minFrame) {
            target = bounds.minFrame;
            clamped = true;
            warnings.push("resize-clamped-to-clip");
        }
        if (target > upper) target = upper;
    }

    let nextMembers: TakeSpan[] = members.map(projectedSpanOfMember);
    if (edge === "right") {
        if (target >= current.endFrameExclusive) {
            const last = members[members.length - 1];
            const source = sourceAtMemberFrame(clips, last, target, framePeriodMs);
            const reversed = memberIsReversed(last, clips);
            const width = minSourceWidth(clips, last, framePeriodMs);
            const span = nextMembers[nextMembers.length - 1];
            if (reversed) {
                if (source >= span.sourceEndSec - width) {
                    throw new ParamBlockPlanError("resize_too_short");
                }
                nextMembers[nextMembers.length - 1] = { ...span, sourceStartSec: source };
            } else {
                if (source <= span.sourceStartSec + width) {
                    throw new ParamBlockPlanError("resize_too_short");
                }
                nextMembers[nextMembers.length - 1] = { ...span, sourceEndSec: source };
            }
        } else {
            const hitIndex = members.findIndex(
                (member) => member.startFrame < target && target < member.endFrameExclusive,
            );
            if (hitIndex >= 0) {
                const hit = members[hitIndex];
                const source = sourceAtMemberFrame(clips, hit, target, framePeriodMs);
                const reversed = memberIsReversed(hit, clips);
                const width = minSourceWidth(clips, hit, framePeriodMs);
                const span = nextMembers[hitIndex];
                if (reversed) {
                    if (source <= span.sourceStartSec + width) {
                        nextMembers = nextMembers.slice(0, hitIndex);
                    } else {
                        nextMembers[hitIndex] = { ...span, sourceStartSec: source };
                        nextMembers = nextMembers.slice(0, hitIndex + 1);
                    }
                } else {
                    if (source >= span.sourceEndSec - width) {
                        nextMembers = nextMembers.slice(0, hitIndex);
                    } else {
                        nextMembers[hitIndex] = { ...span, sourceEndSec: source };
                        nextMembers = nextMembers.slice(0, hitIndex + 1);
                    }
                }
            } else {
                const kept = members
                    .map((member, index) => ({ member, index }))
                    .filter((item) => item.member.endFrameExclusive <= target);
                if (kept.length === 0) {
                    const next = removeOrTombstone(working, user, clips, framePeriodMs);
                    return {
                        edits: next,
                        viewAfter: planView(clips, next, framePeriodMs),
                        targetSpan: null,
                        removedIds: [user.id],
                        warnings,
                        clamped,
                    };
                }
                nextMembers = kept.map((item) => projectedSpanOfMember(item.member));
                const lastKept = kept[kept.length - 1].member;
                if (lastKept.endFrameExclusive < target) {
                    const source = sourceAtMemberFrame(clips, lastKept, target, framePeriodMs);
                    const reversed = memberIsReversed(lastKept, clips);
                    const span = nextMembers[nextMembers.length - 1];
                    const width = minSourceWidth(clips, lastKept, framePeriodMs);
                    const canExtend = reversed
                        ? source < span.sourceStartSec - width
                        : source > span.sourceEndSec + width;
                    if (canExtend) {
                        nextMembers[nextMembers.length - 1] = reversed
                            ? { ...span, sourceStartSec: source }
                            : { ...span, sourceEndSec: source };
                    } else {
                        warnings.push("resize-stopped-at-member");
                    }
                }
            }
        }
    } else if (target <= current.startFrame) {
        const first = members[0];
        const source = sourceAtMemberFrame(clips, first, target, framePeriodMs);
        const reversed = memberIsReversed(first, clips);
        const width = minSourceWidth(clips, first, framePeriodMs);
        const span = nextMembers[0];
        if (reversed) {
            if (source <= span.sourceStartSec + width) {
                throw new ParamBlockPlanError("resize_too_short");
            }
            nextMembers[0] = { ...span, sourceEndSec: source };
        } else {
            if (source >= span.sourceEndSec - width) {
                throw new ParamBlockPlanError("resize_too_short");
            }
            nextMembers[0] = { ...span, sourceStartSec: source };
        }
    } else {
        const hitIndex = members.findIndex(
            (member) => member.startFrame < target && target < member.endFrameExclusive,
        );
        if (hitIndex >= 0) {
            const hit = members[hitIndex];
            const source = sourceAtMemberFrame(clips, hit, target, framePeriodMs);
            const reversed = memberIsReversed(hit, clips);
            const width = minSourceWidth(clips, hit, framePeriodMs);
            const span = nextMembers[hitIndex];
            if (reversed) {
                if (source >= span.sourceEndSec - width) {
                    nextMembers = nextMembers.slice(hitIndex + 1);
                } else {
                    nextMembers[hitIndex] = { ...span, sourceEndSec: source };
                    nextMembers = nextMembers.slice(hitIndex);
                }
            } else {
                if (source <= span.sourceStartSec + width) {
                    nextMembers = nextMembers.slice(hitIndex + 1);
                } else {
                    nextMembers[hitIndex] = { ...span, sourceStartSec: source };
                    nextMembers = nextMembers.slice(hitIndex);
                }
            }
        } else {
            const kept = members
                .map((member, index) => ({ member, index }))
                .filter((item) => item.member.startFrame >= target);
            if (kept.length === 0) {
                const next = removeOrTombstone(working, user, clips, framePeriodMs);
                return {
                    edits: next,
                    viewAfter: planView(clips, next, framePeriodMs),
                    targetSpan: null,
                    removedIds: [user.id],
                    warnings,
                    clamped,
                };
            }
            nextMembers = kept.map((item) => projectedSpanOfMember(item.member));
            const firstKept = kept[0].member;
            if (firstKept.startFrame > target) {
                const source = sourceAtMemberFrame(clips, firstKept, target, framePeriodMs);
                const reversed = memberIsReversed(firstKept, clips);
                const span = nextMembers[0];
                const width = minSourceWidth(clips, firstKept, framePeriodMs);
                const canExtend = reversed
                    ? source > span.sourceEndSec + width
                    : source < span.sourceStartSec - width;
                if (canExtend) {
                    nextMembers[0] = reversed
                        ? { ...span, sourceEndSec: source }
                        : { ...span, sourceStartSec: source };
                } else {
                    warnings.push("resize-stopped-at-member");
                }
            }
        }
    }

    /*
     * Loop 成员在扩展后可能跨过 anchor/media 边界：按周期拆成多个成员。
     * 这一步保证 compose 的每个成员都只覆盖一个周期的剩余部分。
     */
    const expanded = nextMembers.flatMap((span) => normalizeSpanForProjection(clips, span));
    const ordered = orderSpansByProjection(clips, expanded, framePeriodMs);
    if (ordered.length === 0) {
        const next = removeOrTombstone(working, user, clips, framePeriodMs);
        return {
            edits: next,
            viewAfter: planView(clips, next, framePeriodMs),
            targetSpan: null,
            removedIds: [user.id],
            warnings,
            clamped,
        };
    }
    const next = placeUserBlock(working, {
        ...user,
        geometry: { kind: "take", members: ordered },
    });
    const viewAfter = planView(clips, next, framePeriodMs);
    const targetBlocks = viewAfter.blocks.filter(
        (item) => item.id === user.id || item.userBlockId === user.id,
    );
    const targetBlock =
        targetBlocks.find(
            (item) => item.startFrame <= targetFrame && targetFrame <= item.endFrameExclusive,
        ) ?? targetBlocks[0];
    if (targetBlock === undefined) {
        throw new ParamBlockPlanError("internal", "resize result missing");
    }
    const targetSpan = spanOfBlock(targetBlock);
    const edgeActual = edge === "right" ? targetSpan.endFrameExclusive : targetSpan.startFrame;
    if (!clamped && edgeActual !== target) {
        throw new ParamBlockPlanError("internal", `resize edge ${edgeActual} != ${target}`);
    }
    const beforeIds = new Set(viewBefore.blocks.map((item) => item.id));
    const afterIds = new Set(viewAfter.blocks.map((item) => item.id));
    return {
        edits: next,
        viewAfter,
        targetSpan,
        removedIds: [...beforeIds].filter((id) => !afterIds.has(id)),
        warnings,
        clamped,
    };
}

/** 兼容旧调用：只返回编辑集。 */
export function splitProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    return planSplitProjectedBlock(clips, edits, block, frame, framePeriodMs).edits;
}

function baseCoverKey(cover: UserParamBlock["baseCover"]): string | null {
    if (cover === undefined) return null;
    return `${cover.full}:${cover.spans
        .map(
            (span) =>
                `${span.clipId}|${span.takeId}|${span.cycle}|${span.sourceStartSec.toFixed(6)}|${span.sourceEndSec.toFixed(6)}`,
        )
        .sort()
        .join(",")}`;
}

function tombstoneMatchesBlock(tombstone: UserParamBlock, block: UserParamBlock): boolean {
    if (tombstone.deleted !== true || tombstone.geometry.kind !== "take") return false;
    if (block.geometry.kind !== "take") return false;
    const tombstoneCover = baseCoverKey(tombstone.baseCover);
    const blockCover = baseCoverKey(block.baseCover);
    if (tombstoneCover !== null && blockCover !== null && tombstoneCover === blockCover) {
        return true;
    }
    return sameMemberSet(tombstone.geometry.members, block.geometry.members);
}

export interface DeletedParamBlockSnapshot {
    readonly block: UserParamBlock;
    /** 删除发生时的根；根切换后“恢复删除”不得把旧根源区间插进新根。 */
    readonly rootTrackId?: string;
    /** 组删除时为 base 派生 child 生成的墓碑；恢复时按 id 精确移除。 */
    readonly childTombstones?: readonly UserParamBlock[];
}

/** 删除块并返回精确到源区间的可逆快照。base / 多 run 片段先物化。 */
export function deleteProjectedBlocks(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    selected: readonly ProjectedParamBlock[],
    framePeriodMs: number,
): { edits: ParamBlockEditSet; deleted: DeletedParamBlockSnapshot[] } {
    let working = edits;
    const deleted: DeletedParamBlockSnapshot[] = [];
    const removed = new Set<string>();
    for (const block of selected) {
        const logicalId = logicalBlockId(block);
        if (removed.has(logicalId)) continue;
        const ensured = ensureLogicalUserBlock(clips, working, block, framePeriodMs);
        working = ensured.edits;
        if (removed.has(ensured.user.id)) continue;
        removed.add(ensured.user.id);
        const childTombstones: UserParamBlock[] = [];
        if (ensured.user.geometry.kind === "group") {
            childTombstones.push(...collectBaseDerivedTombstones(ensured.user, nextSeq(working)));
        }
        /*
         * 删除基块派生块时，墓碑必须同时拥有“它当前实际覆盖的基块片段”。
         * 否则把块拉长到相邻基块上再删除，被覆盖的那段基块会重新露出来。
         * 只对已有 baseCover 的块做；纯手工块仍然不生成墓碑。
         */
        let owned = ensured.user.baseCover;
        if (owned !== undefined) {
            const viewBefore = composeParamBlockView("", clips, working, framePeriodMs);
            const targets = viewBefore.blocks.filter(
                (item) => logicalBlockId(item) === ensured.user.id,
            );
            if (targets.length > 0) {
                const target = {
                    startFrame: Math.min(...targets.map((item) => item.startFrame)),
                    endFrameExclusive: Math.max(...targets.map((item) => item.endFrameExclusive)),
                };
                // 目标块自己会遮住它覆盖的基块；移除目标后再合成，才能看到这些基块。
                const withoutTarget: ParamBlockEditSet = {
                    ...working,
                    blocks: working.blocks.filter((item) => item.id !== ensured.user.id),
                };
                const baseView = composeParamBlockView("", clips, withoutTarget, framePeriodMs);
                const extra = baseView.blocks
                    .filter(
                        (item) =>
                            item.kind === "base" &&
                            item.startFrame < target.endFrameExclusive &&
                            item.endFrameExclusive > target.startFrame,
                    )
                    .flatMap((item) => item.baseCover?.spans ?? []);
                if (extra.length > 0) {
                    const seen = new Set(
                        owned.spans.map(
                            (span) =>
                                `${span.clipId}|${span.takeId}|${span.cycle}|${span.sourceStartSec.toFixed(6)}|${span.sourceEndSec.toFixed(6)}`,
                        ),
                    );
                    const spans = [...owned.spans];
                    for (const span of extra) {
                        const key = `${span.clipId}|${span.takeId}|${span.cycle}|${span.sourceStartSec.toFixed(6)}|${span.sourceEndSec.toFixed(6)}`;
                        if (seen.has(key)) continue;
                        // 已被某个覆盖 span 包含的片段不再追加，避免 baseCover 内部重叠。
                        if (
                            spans.some(
                                (existing) =>
                                    existing.clipId === span.clipId &&
                                    existing.takeId === span.takeId &&
                                    existing.cycle === span.cycle &&
                                    existing.sourceStartSec <= span.sourceStartSec + 1e-7 &&
                                    existing.sourceEndSec >= span.sourceEndSec - 1e-7,
                            )
                        ) {
                            continue;
                        }
                        seen.add(key);
                        spans.push(span);
                    }
                    owned = { spans, full: owned.full };
                }
            }
        }
        working = {
            ...working,
            revision: working.revision + 1,
            blocks: working.blocks.filter((item) => item.id !== ensured.user.id),
        };
        deleted.push({
            block: { ...ensured.user, ...(owned === undefined ? {} : { baseCover: owned }) },
            childTombstones,
        });
    }
    if (deleted.length === 0) throw new Error("block not found");
    const tombstones: UserParamBlock[] = [];
    for (const snapshot of deleted) {
        /*
         * 纯手工块删除后必须让基块回来：它没有 baseCover，也不需要墓碑。
         * 基块派生块（manual=false 或带 baseCover）才需要墓碑，且墓碑携带
         * 同一个 baseCover —— 删除的是“对基块的替代”，不是基块本身。
         */
        if (snapshot.block.manual === true && snapshot.block.baseCover === undefined) continue;
        if (snapshot.block.baseCover?.full === false) continue;
        if (snapshot.block.geometry.kind !== "take") continue;
        if (working.blocks.some((candidate) => tombstoneMatchesBlock(candidate, snapshot.block))) {
            continue;
        }
        tombstones.push({
            id: `tomb:${snapshot.block.id}`,
            geometry: {
                kind: "take",
                members: snapshot.block.geometry.members.map((member) => ({ ...member })),
            },
            deleted: true,
            manual: false,
            createdSeq: nextSeq(working),
            ...(snapshot.block.baseCover === undefined
                ? {}
                : { baseCover: snapshot.block.baseCover }),
        });
    }
    for (const tombstone of tombstones) working = placeUserBlock(working, tombstone);
    for (const snapshot of deleted) {
        for (const tombstone of snapshot.childTombstones ?? []) {
            if (!working.blocks.some((candidate) => candidate.id === tombstone.id)) {
                working = placeUserBlock(working, tombstone);
            }
        }
    }
    return { edits: { ...working, revision: edits.revision + 1 }, deleted };
}

export function restoreDeletedBlocks(
    edits: ParamBlockEditSet,
    snapshots: readonly DeletedParamBlockSnapshot[],
): ParamBlockEditSet {
    if (snapshots.length === 0) throw new Error("no deleted blocks");
    let next = edits;
    for (const snapshot of snapshots) {
        const childTombstoneIds = new Set(
            (snapshot.childTombstones ?? []).map((tombstone) => tombstone.id),
        );
        next = {
            ...next,
            blocks: next.blocks.filter((candidate) => {
                if (childTombstoneIds.has(candidate.id)) return false;
                if (candidate.id === `tomb:${snapshot.block.id}`) return false;
                return !tombstoneMatchesBlock(candidate, snapshot.block);
            }),
        };
        next = placeUserBlock(next, snapshot.block);
    }
    return { ...next, revision: edits.revision + 1 };
}

function sourceSpansForTimelineRange(
    clip: ParamBlockSourceClip,
    startFrame: number,
    endFrame: number,
    framePeriodMs: number,
): TakeSpan[] {
    const projection = projectionForClip(clip);
    const rate = Math.max(1e-9, projection.rate);
    const maxCycles =
        projection.loopEnabled && projection.loopCycleSec !== null
            ? Math.min(
                  MAX_LOOP_CYCLES,
                  Math.ceil(clip.lengthSec / Math.max(1e-9, projection.loopCycleSec / rate)) + 2,
              )
            : 1;
    const out: TakeSpan[] = [];
    for (let cycle = 0; cycle < maxCycles; cycle += 1) {
        const cycleTimeline =
            projection.loopEnabled && projection.loopCycleSec !== null
                ? projection.loopCycleSec / rate
                : clip.lengthSec;
        const cycleStartFrame = toFrame(clip.startSec + cycle * cycleTimeline, framePeriodMs);
        const cycleEndFrame = toFrame(clip.startSec + (cycle + 1) * cycleTimeline, framePeriodMs);
        const a = Math.max(startFrame, cycleStartFrame, toFrame(clip.startSec, framePeriodMs));
        const b = Math.min(
            endFrame,
            cycleEndFrame,
            toFrame(clip.startSec + clip.lengthSec, framePeriodMs),
        );
        if (b <= a) continue;
        /*
         * 覆盖整个周期时，源域是 [anchor, media] + [0, anchor] 两段；只算
         * sourceAt(a/b) 会得到 [anchor, media]，漏掉回绕后的尾部。
         */
        if (
            projection.loopEnabled &&
            !projection.reversed &&
            a <= cycleStartFrame &&
            b >= cycleEndFrame
        ) {
            const anchor = projection.sourceWindowStartSec;
            const media = projection.mediaDurationSec ?? anchor + cycleTimeline * rate;
            if (media > anchor + EPS) {
                out.push({
                    clipId: clip.id,
                    takeId: projection.takeId,
                    cycle,
                    sourceStartSec: anchor,
                    sourceEndSec: media,
                    label: "manual",
                });
            }
            if (anchor > EPS) {
                out.push({
                    clipId: clip.id,
                    takeId: projection.takeId,
                    cycle,
                    sourceStartSec: 0,
                    sourceEndSec: anchor,
                    label: "manual",
                });
            }
            continue;
        }
        const sourceA = sourceAtProjectedFrame(projection, a, cycle, framePeriodMs);
        const sourceB = sourceAtProjectedFrame(projection, b, cycle, framePeriodMs);
        if (sourceA === null || sourceB === null) continue;
        /*
         * 正放时 sourceA > sourceB 表示区间跨过媒体末尾回绕；必须拆成
         * [sourceA, media] + [0, sourceB]，否则会被当成一条从 0 到 media 的巨段。
         */
        if (projection.loopEnabled && !projection.reversed && sourceA > sourceB + EPS) {
            const media =
                projection.mediaDurationSec ??
                projection.sourceWindowStartSec + cycleTimeline * rate;
            if (media > sourceA + EPS) {
                out.push({
                    clipId: clip.id,
                    takeId: projection.takeId,
                    cycle,
                    sourceStartSec: sourceA,
                    sourceEndSec: media,
                    label: "manual",
                });
            }
            if (sourceB > EPS) {
                out.push({
                    clipId: clip.id,
                    takeId: projection.takeId,
                    cycle,
                    sourceStartSec: 0,
                    sourceEndSec: sourceB,
                    label: "manual",
                });
            }
            continue;
        }
        let lo = Math.min(sourceA, sourceB);
        let hi = Math.max(sourceA, sourceB);
        /*
         * 周期末帧对应的源坐标是“未回绕”的 anchor±cycle；在媒体域归一化后可能
         * 落到另一侧。先钳到 [0, media]，再由 normalizeLoopTakeSpans 补回回绕段。
         */
        const media = projection.mediaDurationSec;
        if (media !== null) {
            if (lo < 0) lo = 0;
            if (hi > media) hi = media;
        }
        if (!(hi > lo + EPS)) continue;
        const span: TakeSpan = {
            clipId: clip.id,
            takeId: projection.takeId,
            cycle,
            sourceStartSec: lo,
            sourceEndSec: hi,
            label: "manual",
        };
        out.push(...normalizeLoopTakeSpans(projection, span));
    }
    return orderSpansByProjection(clipsOf(clip), out, framePeriodMs);
}

function clipsOf(clip: ParamBlockSourceClip): readonly ParamBlockSourceClip[] {
    return [clip];
}

/** 新建块的两条显式失败原因（不再用裸 `Error` + 文案匹配）。 */
export type ParamBlockCreateFailure = "too_short" | "no_source";

/**
 * 新建块失败。
 *
 * 【为什么要有类型】旧实现抛裸 `Error("created block too short")`，再由上层用
 * `message.includes(...)` 猜原因。文案一改，失败就被归到 `unknown` —— 用户看到
 * "参数块操作失败"却不知道是"太短"还是"没有源"。带 code 之后映射是编译期的事。
 */
export class ParamBlockEditError extends Error {
    readonly code: ParamBlockCreateFailure;
    constructor(code: ParamBlockCreateFailure) {
        super(`param block create failed: ${code}`);
        this.name = "ParamBlockEditError";
        this.code = code;
    }
}

export function createUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    startFrame: number,
    endFrame: number,
    value: number | null,
    framePeriodMs: number,
    blockId?: string,
): ParamBlockEditSet {
    if (endFrame <= startFrame) throw new ParamBlockEditError("too_short");
    const startSec = toSec(startFrame, framePeriodMs);
    const endSec = toSec(endFrame, framePeriodMs);
    const members: TakeSpan[] = [];
    for (const clip of clips) {
        if (clip.startSec + clip.lengthSec <= startSec || clip.startSec >= endSec) continue;
        members.push(...sourceSpansForTimelineRange(clip, startFrame, endFrame, framePeriodMs));
    }
    if (
        members.length === 0 &&
        clips.some((clip) => clip.startSec < endSec && clip.startSec + clip.lengthSec > startSec)
    ) {
        throw new ParamBlockEditError("no_source");
    }
    const geometry: BlockGeometry =
        members.length > 0
            ? { kind: "take", members }
            : { kind: "root", span: { startSec, endSec } };
    return placeUserBlock(
        edits,
        {
            id: blockId ?? newParamBlockId("create"),
            geometry,
            deleted: false,
            manual: true,
            value,
            createdSeq: nextSeq(edits),
        },
        { clips, framePeriodMs },
    );
}

export function setProjectedBlockTransition(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    widthSec: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const ensured = ensureLogicalUserBlock(clips, edits, block, framePeriodMs);
    return {
        ...ensured.edits,
        revision: edits.revision + 1,
        blocks: ensured.edits.blocks.map((item) =>
            item.id === ensured.user.id
                ? { ...item, transitionOutSec: Math.max(0, widthSec) }
                : item,
        ),
    };
}

export { takeSpanKey };
