/**
 * 参数块编辑原语（Rust `param_blocks::editing` 的预览端口）。
 *
 * 【职责】所有编辑只改写 `ParamBlockEditSet`，不写绝对帧。
 * 【约束】新块必须经过 `placeUserBlock`，由它裁剪被覆盖的旧用户块。
 */

import {
    inverseProject,
    projectTakeSpan,
    projectionForClip,
    toFrame,
    toSec,
    type ClipProjection,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";
import {
    takeSpanKey,
    type BlockGeometry,
    type ParamBlockEditSet,
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

function activeTake(clip: ParamBlockSourceClip) {
    return clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
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

function projectedMembersToSpans(
    clips: readonly ParamBlockSourceClip[],
    block: ProjectedParamBlock,
    framePeriodMs: number,
): TakeSpan[] {
    if (block.members.length === 0) throw new Error("block has no source");
    const out: TakeSpan[] = [];
    for (const member of block.members) {
        const clip = clips.find((item) => item.id === member.clipId);
        if (clip === undefined) throw new Error("member clip missing");
        const projection = projectionForClip(clip);
        const sourceA = inverseProject(projection, member.startFrame, member.cycle, framePeriodMs);
        const sourceB = inverseProject(
            projection,
            member.endFrameExclusive,
            member.cycle,
            framePeriodMs,
        );
        let start = member.sourceStartSec;
        let end = member.sourceEndSec;
        if (sourceA !== null && sourceB !== null) {
            const lo = Math.max(Math.min(sourceA, sourceB), member.sourceStartSec);
            const hi = Math.min(Math.max(sourceA, sourceB), member.sourceEndSec);
            if (hi > lo + EPS) {
                start = lo;
                end = hi;
            }
        }
        if (end > start + EPS) {
            out.push({
                clipId: member.clipId,
                takeId: member.takeId,
                cycle: member.cycle,
                sourceStartSec: start,
                sourceEndSec: end,
                ...(member.syllableId === undefined ? {} : { syllableId: member.syllableId }),
                label: "manual",
            });
        }
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
    const members = projectedMembersToSpans(clips, block, framePeriodMs);
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
    const target: UserParamBlock = {
        id: newParamBlockId("group"),
        geometry,
        deleted: false,
        manual,
        value,
        ...(transition === undefined ? {} : { transitionOutSec: transition }),
        createdSeq: nextSeq(working),
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

/** 一个成员在本帧域里的可见区间（与合成的 run 同源）。 */
interface SplitMemberInterval {
    readonly index: number;
    readonly member: TakeSpan;
    readonly projection: ClipProjection;
    readonly startFrame: number;
    readonly endFrameExclusive: number;
}

/** 要把 `frame` 挪多少帧才能严格落进该成员内部（0 = 已经在内）。 */
function framesToInside(interval: SplitMemberInterval, frame: number): number {
    if (frame <= interval.startFrame) return interval.startFrame + 1 - frame;
    if (frame >= interval.endFrameExclusive) return frame - (interval.endFrameExclusive - 1);
    return 0;
}

/**
 * 找出"切在哪一个成员上"，并把切点钳进它内部。
 *
 * 【为什么需要兜底】合成合并 run 的判据是 `start <= last.end + 1`（相差 1 帧也并），
 * 因此块的 `[startFrame, endFrameExclusive)` 里可能存在**成员内无帧**的缝。
 * 旧实现只做"哪个成员严格包含该帧"，落进缝里就抛 `split outside block` ——
 * 于是"菜单可点、播放光标在块内，却切不了"。这里改为：先找严格包含的成员；找不到
 * 就取**帧距离最近**的成员并把切点钳进它的内部。
 *
 * 钳位后仍然由该成员的投影反算源域切点，所以切出来的两半首尾相接、无重叠 ——
 * 与"正好点在成员内"的结果只差不到 1 帧。
 */
function resolveSplitMember(args: {
    readonly clips: readonly ParamBlockSourceClip[];
    readonly members: readonly TakeSpan[];
    readonly frame: number;
    readonly framePeriodMs: number;
}): { readonly index: number; readonly sourceSec: number } | null {
    const intervals: SplitMemberInterval[] = [];
    for (let index = 0; index < args.members.length; index += 1) {
        const member = args.members[index];
        const clip = args.clips.find((item) => item.id === member.clipId);
        if (clip === undefined) continue;
        const projection = projectionForClip(clip);
        const projected = projectTakeSpan(projection, member, args.framePeriodMs).filter(
            (interval) => interval.cycle === member.cycle,
        );
        const startFrame = projected[0]?.startFrame;
        const endFrameExclusive = projected[projected.length - 1]?.endFrameExclusive;
        if (startFrame === undefined || endFrameExclusive === undefined) continue;
        intervals.push({ index, member, projection, startFrame, endFrameExclusive });
    }
    if (intervals.length === 0) return null;

    let hit = intervals.find((interval) => framesToInside(interval, args.frame) === 0);
    let splitFrame = args.frame;
    if (hit === undefined) {
        hit = intervals.reduce((best, candidate) =>
            framesToInside(candidate, args.frame) < framesToInside(best, args.frame)
                ? candidate
                : best,
        );
        splitFrame = Math.min(Math.max(args.frame, hit.startFrame + 1), hit.endFrameExclusive - 1);
        if (!(splitFrame > hit.startFrame && splitFrame < hit.endFrameExclusive)) return null;
    }
    const sourceSec = inverseProject(
        hit.projection,
        splitFrame,
        hit.member.cycle,
        args.framePeriodMs,
    );
    return sourceSec === null ? null : { index: hit.index, sourceSec };
}

export function splitProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const ensured = ensureEditableUserBlock(clips, edits, block, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.groupedFrom !== undefined) throw new Error("grouped block must be ungrouped first");
    if (user.geometry.kind === "root") {
        const splitSec = toSec(frame, framePeriodMs);
        const span = user.geometry.span;
        if (splitSec <= span.startSec + EPS || splitSec >= span.endSec - EPS)
            throw new Error("split outside block");
        const left: UserParamBlock = {
            id: newParamBlockId("split"),
            geometry: { kind: "root", span: { startSec: span.startSec, endSec: splitSec } },
            deleted: false,
            manual: user.manual,
            value: user.value,
            transitionOutSec: 0,
            createdSeq: nextSeq(working),
        };
        const right: UserParamBlock = {
            id: newParamBlockId("split"),
            geometry: { kind: "root", span: { startSec: splitSec, endSec: span.endSec } },
            deleted: false,
            manual: user.manual,
            value: user.value,
            ...(user.transitionOutSec === undefined
                ? {}
                : { transitionOutSec: user.transitionOutSec }),
            createdSeq: nextSeq(working) + 1,
        };
        return placeUserBlock(placeUserBlock(working, left), right);
    }
    const members = user.geometry.members
        .slice()
        .sort((a, b) => a.sourceStartSec - b.sourceStartSec);
    const target = resolveSplitMember({ clips, members, frame, framePeriodMs });
    if (target === null) throw new Error("split outside block");
    const splitIndex = target.index;
    const splitSource = target.sourceSec;
    const containing = members[splitIndex];
    if (
        splitSource <= containing.sourceStartSec + EPS ||
        splitSource >= containing.sourceEndSec - EPS
    ) {
        throw new Error("split outside block");
    }
    const leftMembers = [
        ...members.slice(0, splitIndex),
        { ...containing, sourceEndSec: splitSource },
    ];
    const rightMembers = [
        { ...containing, sourceStartSec: splitSource },
        ...members.slice(splitIndex + 1),
    ];
    const left: UserParamBlock = {
        id: newParamBlockId("split"),
        geometry: { kind: "take", members: leftMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        transitionOutSec: 0,
        createdSeq: nextSeq(working),
    };
    const right: UserParamBlock = {
        id: newParamBlockId("split"),
        geometry: { kind: "take", members: rightMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(working) + 1,
    };
    return placeUserBlock(placeUserBlock(working, left), right);
}

export function resizeProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const ensured = ensureEditableUserBlock(clips, edits, block, framePeriodMs);
    const working = ensured.edits;
    const user = ensured.user;
    if (user.groupedFrom !== undefined) throw new Error("grouped block must be ungrouped first");
    if (user.geometry.kind === "root") {
        const sec = toSec(targetFrame, framePeriodMs);
        const span = user.geometry.span;
        const nextSpan: RootSpan =
            edge === "left"
                ? { startSec: Math.max(0, sec), endSec: span.endSec }
                : {
                      startSec: span.startSec,
                      endSec: Math.max(span.startSec + framePeriodMs / 1000, sec),
                  };
        if (nextSpan.endSec <= nextSpan.startSec) throw new Error("resized block too short");
        return placeUserBlock(working, { ...user, geometry: { kind: "root", span: nextSpan } });
    }
    const members = user.geometry.members.slice();
    if (members.length === 0) throw new Error("block has no members");
    const indices = members
        .map((_, index) => index)
        .sort(
            (left, right) =>
                members[left].sourceStartSec - members[right].sourceStartSec ||
                members[left].sourceEndSec - members[right].sourceEndSec,
        );
    const index = edge === "left" ? indices[0] : indices[indices.length - 1];
    const member = members[index];
    const clip = clips.find((item) => item.id === member.clipId);
    if (clip === undefined) throw new Error("member clip missing");
    const projection = projectionForClip(clip);
    const source = inverseProject(projection, targetFrame, member.cycle, framePeriodMs);
    if (source === null) throw new Error("resize target outside clip");
    members[index] =
        edge === "left"
            ? { ...member, sourceStartSec: Math.min(source, member.sourceEndSec - 1e-6) }
            : { ...member, sourceEndSec: Math.max(source, member.sourceStartSec + 1e-6) };
    return placeUserBlock(working, { ...user, geometry: { kind: "take", members } });
}

function tombstoneMatches(tombstone: UserParamBlock, member: TakeSpan): boolean {
    if (tombstone.deleted !== true || tombstone.geometry.kind !== "take") return false;
    return (
        tombstone.geometry.members.length === 1 && sameMember(tombstone.geometry.members[0], member)
    );
}

export interface DeletedParamBlockSnapshot {
    readonly block: UserParamBlock;
    readonly previousEdits: ParamBlockEditSet;
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
        working = {
            ...working,
            revision: working.revision + 1,
            blocks: working.blocks.filter((item) => item.id !== ensured.user.id),
        };
        deleted.push({ block: ensured.user, previousEdits: before });
    }
    if (deleted.length === 0) throw new Error("block not found");
    const tombstones: UserParamBlock[] = [];
    for (const snapshot of deleted) {
        for (const member of snapshot.block.geometry.kind === "take"
            ? snapshot.block.geometry.members
            : []) {
            if (working.blocks.some((candidate) => tombstoneMatches(candidate, member))) continue;
            tombstones.push({
                id: newParamBlockId("tomb"),
                geometry: { kind: "take", members: [member] },
                deleted: true,
                createdSeq: nextSeq(working),
            });
        }
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
                if (snapshot.block.geometry.kind !== "take") return true;
                return !snapshot.block.geometry.members.some((member) =>
                    tombstoneMatches(candidate, member),
                );
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
    const startSec = toSec(startFrame, framePeriodMs);
    const endSec = toSec(endFrame, framePeriodMs);
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
        const cycleStart =
            projection.loopEnabled && projection.loopCycleSec !== null
                ? clip.startSec + (cycle * projection.loopCycleSec) / rate
                : clip.startSec;
        const cycleEnd =
            projection.loopEnabled && projection.loopCycleSec !== null
                ? clip.startSec + ((cycle + 1) * projection.loopCycleSec) / rate
                : clip.startSec + clip.lengthSec;
        const a = Math.max(startSec, cycleStart, clip.startSec);
        const b = Math.min(endSec, cycleEnd, clip.startSec + clip.lengthSec);
        if (b <= a + EPS) continue;
        const endsAtCycleBoundary = projection.loopEnabled && b >= cycleEnd - EPS;
        const sourceA = inverseProject(projection, toFrame(a, framePeriodMs), cycle, framePeriodMs);
        const sourceB = inverseProject(
            projection,
            Math.max(0, toFrame(b, framePeriodMs) - (endsAtCycleBoundary ? 1 : 0)),
            cycle,
            framePeriodMs,
        );
        if (sourceA === null || sourceB === null) continue;
        const sourceStart = Math.min(sourceA, sourceB);
        const sourceEnd =
            Math.max(sourceA, sourceB) + (endsAtCycleBoundary ? (framePeriodMs / 1000) * rate : 0);
        if (sourceEnd > sourceStart + EPS) {
            out.push({
                clipId: clip.id,
                takeId: projection.takeId,
                cycle,
                sourceStartSec: sourceStart,
                sourceEndSec: sourceEnd,
                label: "manual",
            });
        }
    }
    return out;
}

export function createUserBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    startFrame: number,
    endFrame: number,
    value: number | null,
    framePeriodMs: number,
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
        id: newParamBlockId("create"),
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

export function restoreGameBlocks(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    startFrame: number,
    endFrame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const startSec = toSec(startFrame, framePeriodMs);
    const endSec = toSec(endFrame, framePeriodMs);
    let next: ParamBlockEditSet = {
        ...edits,
        revision: edits.revision + 1,
        blocks: edits.blocks.filter((block) => {
            if (block.geometry.kind === "root") {
                return (
                    block.geometry.span.endSec <= startSec || block.geometry.span.startSec >= endSec
                );
            }
            return block.geometry.members.every((member) => {
                const clip = clips.find((item) => item.id === member.clipId);
                if (clip === undefined) return true;
                const projection = projectionForClip(clip);
                return projectTakeSpan(projection, member, framePeriodMs).every(
                    (interval) =>
                        interval.endFrameExclusive <= startFrame || interval.startFrame >= endFrame,
                );
            });
        }),
    };
    for (const clip of clips) {
        const take = activeTake(clip);
        const spans = take?.syllableSpansSec ?? clip.syllableSpansSec ?? [];
        const projection = projectionForClip(clip);
        const maxCycles =
            projection.loopEnabled && projection.loopCycleSec !== null
                ? Math.min(
                      MAX_LOOP_CYCLES,
                      Math.ceil(
                          clip.lengthSec /
                              Math.max(1e-9, projection.loopCycleSec / projection.rate),
                      ) + 2,
                  )
                : 1;
        spans.forEach((span, index) => {
            for (let cycle = 0; cycle < maxCycles; cycle += 1) {
                const member: TakeSpan = {
                    clipId: clip.id,
                    takeId: projection.takeId,
                    cycle,
                    sourceStartSec: span.startSec,
                    sourceEndSec: span.endSec,
                    syllableId: `game:${index}:${span.startSec}`,
                    label: "gameBaseline",
                };
                const intersects = projectTakeSpan(projection, member, framePeriodMs)
                    .filter((interval) => interval.cycle === cycle)
                    .some(
                        (interval) =>
                            interval.endFrameExclusive > startFrame &&
                            interval.startFrame < endFrame,
                    );
                if (!intersects) continue;
                next = placeUserBlock(next, {
                    id: newParamBlockId("restore"),
                    geometry: { kind: "take", members: [member] },
                    deleted: false,
                    manual: false,
                    value: null,
                    createdSeq: nextSeq(next),
                });
            }
        });
    }
    return next;
}

export { takeSpanKey };
