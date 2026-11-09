/**
 * 参数块编辑原语（Rust `param_blocks::editing` 的预览端口）。
 *
 * 【职责】所有编辑只改写 `ParamBlockEditSet`，不写绝对帧。
 * 【约束】新块必须经过 `placeUserBlock`，由它裁剪被覆盖的旧用户块。
 */

import { composeParamBlockView } from "./paramBlockComposition";
import {
    inverseProject,
    projectTakeSpan,
    projectionForClip,
    toSec,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";
import {
    takeSpanKey,
    type BlockGeometry,
    type ParamBlockEditSet,
    type ParamBlockView,
    type ProjectedParamBlock,
    type RootSpan,
    type TakeSpan,
    type UserParamBlock,
} from "./paramBlockTypes";

const EPS = 1e-9;

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
                    )
                        continue;
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
            if (pieces.length > 0) {
                const [start, end] = pieces[0];
                blocks.push({
                    ...block,
                    geometry: { kind: "root", span: { startSec: start, endSec: end } },
                });
            }
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

function membersOf(selected: readonly ProjectedParamBlock[]): TakeSpan[] {
    const out: TakeSpan[] = [];
    const seen = new Set<string>();
    for (const block of selected) {
        for (const member of block.members) {
            const span: TakeSpan = {
                clipId: member.clipId,
                takeId: member.takeId,
                cycle: member.cycle,
                sourceStartSec: member.sourceStartSec,
                sourceEndSec: member.sourceEndSec,
                ...(member.syllableId === undefined ? {} : { syllableId: member.syllableId }),
                label: "manual",
            };
            const key = takeSpanKey(span);
            if (!seen.has(key)) {
                seen.add(key);
                out.push(span);
            }
        }
    }
    return out;
}

export function mergeProjectedBlocks(
    edits: ParamBlockEditSet,
    selected: readonly ProjectedParamBlock[],
    framePeriodMs: number,
): ParamBlockEditSet {
    if (selected.length < 2) throw new Error("merge needs two blocks");
    const ordered = selected.slice().sort((a, b) => a.startFrame - b.startFrame);
    for (let index = 1; index < ordered.length; index += 1) {
        if (ordered[index - 1].endFrameExclusive !== ordered[index].startFrame) {
            throw new Error("merge blocks are not adjacent");
        }
    }
    const members = membersOf(ordered);
    const geometry: BlockGeometry =
        members.length === 0
            ? {
                  kind: "root",
                  span: {
                      startSec: toSec(ordered[0].startFrame, framePeriodMs),
                      endSec: toSec(ordered[ordered.length - 1].endFrameExclusive, framePeriodMs),
                  },
              }
            : { kind: "take", members };
    const values = ordered.map((block) => block.value ?? null);
    const value = values.every((item) => item === values[0]) ? values[0] : null;
    const manual = ordered.every((block) => block.kind === "manual");
    const transition = ordered[ordered.length - 1].transitionFrames;
    const target: UserParamBlock = {
        id: newParamBlockId("merge"),
        geometry,
        deleted: false,
        manual,
        value,
        ...(transition === undefined ? {} : { transitionOutSec: toSec(transition, framePeriodMs) }),
        createdSeq: nextSeq(edits),
    };
    return placeUserBlock(edits, target);
}

export function unmergeUserBlock(
    edits: ParamBlockEditSet,
    blockId: string,
    _framePeriodMs: number,
): ParamBlockEditSet {
    const block = edits.blocks.find((item) => item.id === blockId);
    if (block === undefined || block.geometry.kind !== "take") return edits;
    let next: ParamBlockEditSet = {
        ...edits,
        revision: edits.revision + 1,
        blocks: edits.blocks.filter((item) => item.id !== blockId),
    };
    for (const member of block.geometry.members) {
        next = placeUserBlock(next, {
            id: newParamBlockId("split"),
            geometry: { kind: "take", members: [member] },
            deleted: false,
            manual: block.manual,
            value: block.value,
            createdSeq: nextSeq(next),
        });
    }
    return next;
}

export function splitProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    frame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const user = edits.blocks.find((item) => item.id === block.id);
    if (user === undefined) throw new Error("block not found");
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
            createdSeq: nextSeq(edits),
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
            createdSeq: nextSeq(edits) + 1,
        };
        return placeUserBlock(placeUserBlock(edits, left), right);
    }
    const members = user.geometry.members
        .slice()
        .sort((a, b) => a.sourceStartSec - b.sourceStartSec);
    let splitIndex = -1;
    let splitSource: number | null = null;
    for (let index = 0; index < members.length; index += 1) {
        const member = members[index];
        const clip = clips.find((item) => item.id === member.clipId);
        if (clip === undefined) continue;
        const projection = projectionForClip(clip);
        const intervals = projectTakeSpan(projection, member, framePeriodMs).filter(
            (interval) => interval.cycle === member.cycle,
        );
        const start = intervals[0]?.startFrame;
        const end = intervals[intervals.length - 1]?.endFrameExclusive;
        if (start !== undefined && end !== undefined && frame > start && frame < end) {
            splitIndex = index;
            splitSource = inverseProject(projection, frame, member.cycle, framePeriodMs);
            break;
        }
    }
    if (splitIndex < 0 || splitSource === null) throw new Error("split outside block");
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
        createdSeq: nextSeq(edits),
    };
    const right: UserParamBlock = {
        id: newParamBlockId("split"),
        geometry: { kind: "take", members: rightMembers },
        deleted: false,
        manual: user.manual,
        value: user.value,
        ...(user.transitionOutSec === undefined ? {} : { transitionOutSec: user.transitionOutSec }),
        createdSeq: nextSeq(edits) + 1,
    };
    return placeUserBlock(placeUserBlock(edits, left), right);
}

export function resizeProjectedBlock(
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    edge: "left" | "right",
    targetFrame: number,
    framePeriodMs: number,
): ParamBlockEditSet {
    const user = edits.blocks.find((item) => item.id === block.id);
    if (user === undefined) throw new Error("block not found");
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
        return placeUserBlock(edits, { ...user, geometry: { kind: "root", span: nextSpan } });
    }
    const members = user.geometry.members.slice();
    if (members.length === 0) throw new Error("block has no members");
    const index = edge === "left" ? 0 : members.length - 1;
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
    return placeUserBlock(edits, { ...user, geometry: { kind: "take", members } });
}

export function deleteProjectedBlocks(
    edits: ParamBlockEditSet,
    selected: readonly ProjectedParamBlock[],
): ParamBlockEditSet {
    let next = edits;
    const tombstones: UserParamBlock[] = [];
    for (const block of selected) {
        const user = edits.blocks.find((item) => item.id === block.id);
        if (user !== undefined) {
            next = { ...next, blocks: next.blocks.filter((item) => item.id !== user.id) };
            for (const member of user.geometry.kind === "take" ? user.geometry.members : []) {
                tombstones.push({
                    id: newParamBlockId("tomb"),
                    geometry: { kind: "take", members: [member] },
                    deleted: true,
                    createdSeq: nextSeq(next),
                });
            }
        } else {
            for (const member of block.members) {
                tombstones.push({
                    id: newParamBlockId("tomb"),
                    geometry: {
                        kind: "take",
                        members: [
                            {
                                clipId: member.clipId,
                                takeId: member.takeId,
                                cycle: member.cycle,
                                sourceStartSec: member.sourceStartSec,
                                sourceEndSec: member.sourceEndSec,
                                ...(member.syllableId === undefined
                                    ? {}
                                    : { syllableId: member.syllableId }),
                                label: "manual",
                            },
                        ],
                    },
                    deleted: true,
                    createdSeq: nextSeq(next),
                });
            }
        }
    }
    for (const tombstone of tombstones) next = placeUserBlock(next, tombstone);
    return { ...next, revision: edits.revision + 1 };
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
        const clipStart = clip.startSec;
        const clipEnd = clip.startSec + clip.lengthSec;
        if (clipEnd <= startSec || clipStart >= endSec) continue;
        const projection = projectionForClip(clip);
        const sourceStart = inverseProject(
            projection,
            Math.round((Math.max(startSec, clipStart) * 1000) / framePeriodMs),
            0,
            framePeriodMs,
        );
        const sourceEnd = inverseProject(
            projection,
            Math.round((Math.min(endSec, clipEnd) * 1000) / framePeriodMs),
            0,
            framePeriodMs,
        );
        if (sourceStart === null || sourceEnd === null) continue;
        const start = Math.min(sourceStart, sourceEnd);
        const end = Math.max(sourceStart, sourceEnd);
        if (end > start + EPS) {
            members.push({
                clipId: clip.id,
                takeId: projection.takeId,
                cycle: 0,
                sourceStartSec: start,
                sourceEndSec: end,
                label: "manual",
            });
        }
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
    edits: ParamBlockEditSet,
    block: ProjectedParamBlock,
    widthSec: number,
): ParamBlockEditSet {
    const user = edits.blocks.find((item) => item.id === block.id);
    if (user !== undefined) {
        return {
            ...edits,
            revision: edits.revision + 1,
            blocks: edits.blocks.map((item) =>
                item.id === user.id ? { ...item, transitionOutSec: Math.max(0, widthSec) } : item,
            ),
        };
    }
    if (block.members.length === 0) throw new Error("transition target has no source");
    return placeUserBlock(edits, {
        id: newParamBlockId("transition"),
        geometry: {
            kind: "take",
            members: block.members.map((member) => ({
                clipId: member.clipId,
                takeId: member.takeId,
                cycle: member.cycle,
                sourceStartSec: member.sourceStartSec,
                sourceEndSec: member.sourceEndSec,
                ...(member.syllableId === undefined ? {} : { syllableId: member.syllableId }),
                label: "manual",
            })),
        },
        deleted: false,
        manual: false,
        value: block.value ?? null,
        transitionOutSec: Math.max(0, widthSec),
        createdSeq: nextSeq(edits),
    });
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
                const intervals = projectTakeSpan(projection, member, framePeriodMs);
                return intervals.every(
                    (interval) =>
                        interval.endFrameExclusive <= startFrame || interval.startFrame >= endFrame,
                );
            });
        }),
    };
    for (const clip of clips) {
        const take = clip.takes?.find((item) => item.id === clip.activeTakeId) ?? clip.takes?.[0];
        const spans = take?.syllableSpansSec ?? clip.syllableSpansSec ?? [];
        const projection = projectionForClip(clip);
        spans.forEach((span, index) => {
            const member: TakeSpan = {
                clipId: clip.id,
                takeId: projection.takeId,
                cycle: 0,
                sourceStartSec: span.startSec,
                sourceEndSec: span.endSec,
                syllableId: `game:${index}:${span.startSec}`,
                label: "gameBaseline",
            };
            const intersects = projectTakeSpan(projection, member, framePeriodMs).some(
                (interval) =>
                    interval.endFrameExclusive > startFrame && interval.startFrame < endFrame,
            );
            if (!intersects) return;
            next = placeUserBlock(next, {
                id: newParamBlockId("restore"),
                geometry: { kind: "take", members: [member] },
                deleted: false,
                manual: false,
                value: null,
                createdSeq: nextSeq(next),
            });
        });
    }
    return next;
}

export function withEditSetView(
    rootTrackId: string,
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    framePeriodMs: number,
): { edits: ParamBlockEditSet; view: ParamBlockView } {
    return {
        edits,
        view: composeParamBlockView(rootTrackId, clips, edits, framePeriodMs),
    };
}
