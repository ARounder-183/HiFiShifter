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
import {
    takeSpanKey,
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
): ParamBlockEditSet {
    const blocks: UserParamBlock[] = [];
    for (const block of edits.blocks) {
        if (block.id === target.id) continue;
        if (block.deleted || target.deleted) {
            blocks.push(block);
            continue;
        }
        if (block.geometry.kind === "take" && target.geometry.kind === "take") {
            const kept: TakeSpan[] = [];
            for (const member of block.geometry.members) {
                let pieces: Array<[number, number]> = [
                    [member.sourceStartSec, member.sourceEndSec],
                ];
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
            if (kept.length > 0)
                blocks.push({ ...block, geometry: { kind: "take", members: kept } });
            continue;
        }
        if (block.geometry.kind === "root" && target.geometry.kind === "root") {
            const pieces = subtractInterval(
                block.geometry.span.startSec,
                block.geometry.span.endSec,
                target.geometry.span.startSec,
                target.geometry.span.endSec,
            );
            pieces.forEach(([start, end], index) => {
                if (end <= start + EPS) return;
                blocks.push({
                    ...block,
                    id: rootPieceId(block, index, start, end),
                    geometry: { kind: "root", span: { startSec: start, endSec: end } },
                });
            });
            continue;
        }
        blocks.push(block);
    }
    blocks.push(target);
    return {
        schemaVersion: 1,
        revision: edits.revision + 1,
        blocks,
    };
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
    if (parent.groupedFrom !== undefined) return { edits, user: parent };
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
        ...(parent.baseCover === undefined ? {} : { baseCover: parent.baseCover }),
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
    return { edits: placeUserBlock(edits, target), user: target };
}

/** 可逆编组：保存每个被选中块的源域快照，取消编组时原样恢复。 */
export function groupProjectedBlocks(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    selected: readonly ProjectedParamBlock[],
    framePeriodMs: number,
): ParamBlockEditSet {
    if (selected.length < 2) throw new Error("group needs two blocks");
    const ordered = selected.slice().sort((a, b) => a.startFrame - b.startFrame);
    for (let index = 1; index < ordered.length; index += 1) {
        if (ordered[index - 1].endFrameExclusive !== ordered[index].startFrame) {
            throw new Error("group blocks are not adjacent");
        }
    }

    let working = edits;
    const snapshots: UserParamBlock[] = [];
    for (const block of ordered) {
        const ensured = ensureEditableUserBlock(clips, working, block, framePeriodMs);
        working = ensured.edits;
        snapshots.push(ensured.user);
    }
    if (snapshots.every((block) => block.id === snapshots[0].id)) {
        throw new Error("blocks are already grouped");
    }
    const hasRoot = snapshots.some((block) => block.geometry.kind === "root");
    const hasTake = snapshots.some((block) => block.geometry.kind === "take");
    if (hasRoot && hasTake) throw new Error("cannot group root and take blocks together");

    const members = snapshots.flatMap((block) =>
        block.geometry.kind === "take" ? block.geometry.members : [],
    );
    const geometry: BlockGeometry = hasTake
        ? { kind: "take", members }
        : {
              kind: "root",
              span: {
                  startSec: toSec(ordered[0].startFrame, framePeriodMs),
                  endSec: toSec(ordered[ordered.length - 1].endFrameExclusive, framePeriodMs),
              },
          };
    const values = snapshots.map((block) => block.value ?? null);
    const value = values.every((item) => item === values[0]) ? values[0] : null;
    const manual = snapshots.every((block) => block.manual === true);
    const transition = snapshots[snapshots.length - 1].transitionOutSec;
    const firstCover = snapshots[0].baseCover;
    const baseCover =
        firstCover !== undefined &&
        snapshots.every(
            (block) =>
                block.baseCover !== undefined &&
                JSON.stringify(block.baseCover) === JSON.stringify(firstCover),
        )
            ? firstCover
            : undefined;
    const target: UserParamBlock = {
        id: newParamBlockId("group"),
        geometry,
        deleted: false,
        manual,
        value,
        ...(transition === undefined ? {} : { transitionOutSec: transition }),
        createdSeq: nextSeq(working),
        ...(baseCover === undefined ? {} : { baseCover }),
        groupedFrom: snapshots,
    };
    return placeUserBlock(working, target);
}

export function ungroupUserBlock(edits: ParamBlockEditSet, blockId: string): ParamBlockEditSet {
    const block = edits.blocks.find((item) => item.id === blockId);
    if (block === undefined) throw new Error("block not found");
    if (block.groupedFrom === undefined || block.groupedFrom.length < 2) {
        throw new Error("block is not grouped");
    }
    let next: ParamBlockEditSet = {
        ...edits,
        revision: edits.revision + 1,
        blocks: edits.blocks.filter((item) => item.id !== blockId),
    };
    for (const child of block.groupedFrom) next = placeUserBlock(next, child);
    return { ...next, revision: edits.revision + 1 };
}

export type ParamBlockPlanErrorCode =
    | "block_not_found"
    | "block_deleted"
    | "grouped_locked"
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
        ...(user.baseCover === undefined ? {} : { baseCover: user.baseCover }),
    };
    const right: UserParamBlock = {
        id: `split:${user.id}:${frame}:right`,
        geometry: { kind: "root", span: { startSec: splitSec, endSec: span.endSec } },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(working) + 1,
        ...(user.baseCover === undefined ? {} : { baseCover: user.baseCover }),
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
export function planSplitProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): SplitProjectedBlockPlan {
    if (block.kind === "grouped") throw new ParamBlockPlanError("grouped_locked");
    if (frame <= block.startFrame || frame >= block.endFrameExclusive) {
        throw new ParamBlockPlanError("split_outside_block");
    }
    const ensured = ensureEditableUserBlock(clips, edits, block, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.groupedFrom !== undefined) throw new ParamBlockPlanError("grouped_locked");
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
    const left: UserParamBlock = {
        id: `split:${user.id}:${frame}:left`,
        geometry: { kind: "take", members: leftMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        transitionOutSec: 0,
        createdSeq: nextSeq(working),
        ...(user.baseCover === undefined ? {} : { baseCover: user.baseCover }),
    };
    const right: UserParamBlock = {
        id: `split:${user.id}:${frame}:right`,
        geometry: { kind: "take", members: rightMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(working) + 1,
        ...(user.baseCover === undefined ? {} : { baseCover: user.baseCover }),
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

function removeOrTombstone(
    working: ParamBlockEditSet,
    user: UserParamBlock,
    _clips: readonly ParamBlockSourceClip[],
    _framePeriodMs: number,
): ParamBlockEditSet {
    const isManual = user.manual === true && user.baseCover === undefined;
    if (isManual) {
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
    if (block.kind === "grouped") throw new ParamBlockPlanError("grouped_locked");
    const viewBefore = planView(clips, edits, framePeriodMs);
    const current = findViewBlock(viewBefore, block.id);
    const ensured = ensureEditableUserBlock(clips, edits, current, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.groupedFrom !== undefined) throw new ParamBlockPlanError("grouped_locked");
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

export function resizeProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    return planResizeProjectedBlock(clips, edits, block, edge, targetFrame, framePeriodMs).edits;
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
    readonly previousEdits: ParamBlockEditSet;
    /** 删除发生时的根；根切换后“恢复删除”不得把旧根源区间插进新根。 */
    readonly rootTrackId?: string;
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
        const ensured = ensureEditableUserBlock(clips, working, block, framePeriodMs);
        working = ensured.edits;
        if (removed.has(ensured.user.id)) continue;
        removed.add(ensured.user.id);
        const before = working;
        /*
         * 删除基块派生块时，墓碑必须同时拥有“它当前实际覆盖的基块片段”。
         * 否则把块拉长到相邻基块上再删除，被覆盖的那段基块会重新露出来。
         * 只对已有 baseCover 的块做；纯手工块仍然不生成墓碑。
         */
        let owned = ensured.user.baseCover;
        if (owned !== undefined) {
            const viewBefore = composeParamBlockView("", clips, working, framePeriodMs);
            const target =
                viewBefore.blocks.find((item) => item.id === ensured.user.id) ??
                viewBefore.blocks.find((item) => item.userBlockId === ensured.user.id);
            if (target !== undefined) {
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
            previousEdits: before,
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
    return { edits: { ...working, revision: edits.revision + 1 }, deleted };
}

export function restoreDeletedBlocks(
    edits: ParamBlockEditSet,
    snapshots: readonly DeletedParamBlockSnapshot[],
): ParamBlockEditSet {
    if (snapshots.length === 0) throw new Error("no deleted blocks");
    let next = edits;
    for (const snapshot of snapshots) {
        next = {
            ...next,
            blocks: next.blocks.filter((candidate) => {
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

export function createUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    startFrame: number,
    endFrame: number,
    value: number | null,
    framePeriodMs: number,
    blockId?: string,
): ParamBlockEditSet {
    if (endFrame <= startFrame) throw new Error("created block too short");
    const startSec = toSec(startFrame, framePeriodMs);
    const endSec = toSec(endFrame, framePeriodMs);
    const members: TakeSpan[] = [];
    for (const clip of clips) {
        if (clip.startSec + clip.lengthSec <= startSec || clip.startSec >= endSec) continue;
        members.push(...sourceSpansForTimelineRange(clip, startFrame, endFrame, framePeriodMs));
    }
    const geometry: BlockGeometry =
        members.length > 0
            ? { kind: "take", members }
            : { kind: "root", span: { startSec, endSec } };
    return placeUserBlock(edits, {
        id: blockId ?? newParamBlockId("create"),
        geometry,
        deleted: false,
        manual: true,
        value,
        createdSeq: nextSeq(edits),
    });
}

export function setProjectedBlockTransition(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    widthSec: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const ensured = ensureEditableUserBlock(clips, edits, block, framePeriodMs);
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
