/**
 * 参数块合成（Rust `param_blocks::compose` 的预览端口）。
 *
 * 【职责】base 抑制、用户优先级、重叠消解、过渡、视图校验。
 * 【约束】不持久化任何结果；同一输入必须得到同一输出。
 */

import {
    projectTakeSpan,
    projectionForClip,
    toFrame,
    toSec,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";
import type {
    ParamBlockEditSet,
    ParamBlockView,
    ProjectedBlockKind,
    ProjectedMember,
    ProjectedParamBlock,
    TakeSpan,
    UserParamBlock,
} from "./paramBlockTypes";

const EPS = 1e-9;
const ANCHOR_TOLERANCE_SEC = 0.005;
const ANCHOR_OVERLAP_RATIO = 0.5;

interface Candidate {
    readonly id: string;
    readonly start: number;
    readonly end: number;
    readonly kind: ProjectedBlockKind;
    readonly members: ProjectedMember[];
    readonly value?: number | null;
    readonly transitionWidthSec?: number;
    readonly rate: number;
    readonly priority: number;
    readonly seq: number;
    readonly zOrder: number;
}

function hash(text: string): string {
    let left = 0x811c9dc5;
    let right = 0x9e3779b9;
    for (let index = 0; index < text.length; index += 1) {
        left = Math.imul(left ^ text.charCodeAt(index), 0x01000193);
        right = Math.imul(right ^ text.charCodeAt(index), 0x85ebca6b);
    }
    return `${(left >>> 0).toString(16)}${(right >>> 0).toString(16)}`;
}

function memberOf(
    span: TakeSpan,
    interval: { startFrame: number; endFrameExclusive: number },
): ProjectedMember {
    return {
        clipId: span.clipId,
        takeId: span.takeId,
        cycle: span.cycle,
        sourceStartSec: span.sourceStartSec,
        sourceEndSec: span.sourceEndSec,
        ...(span.syllableId === undefined ? {} : { syllableId: span.syllableId }),
        startFrame: interval.startFrame,
        endFrameExclusive: interval.endFrameExclusive,
    };
}

function baseCandidates(
    clips: readonly ParamBlockSourceClip[],
    framePeriodMs: number,
): Candidate[] {
    const out: Candidate[] = [];
    clips.forEach((clip, zOrder) => {
        const take = clip.takes?.find((item) => item.id === clip.activeTakeId) ?? clip.takes?.[0];
        const spans =
            take?.syllableSpansSec ??
            clip.syllableSpansSec ??
            take?.midiNoteData?.map((note) => ({ startSec: note.startSec, endSec: note.endSec })) ??
            clip.midiNoteData?.map((note) => ({ startSec: note.startSec, endSec: note.endSec })) ??
            [];
        if (spans.length === 0) return;
        const normalized = spans
            .filter(
                (span) =>
                    span.endSec > span.startSec &&
                    Number.isFinite(span.startSec) &&
                    Number.isFinite(span.endSec),
            )
            .slice()
            .sort((a, b) => a.startSec - b.startSec);
        const projection = projectionForClip(clip);
        normalized.forEach((span, index) => {
            const takeSpan: TakeSpan = {
                clipId: clip.id,
                takeId: projection.takeId,
                cycle: 0,
                sourceStartSec: span.startSec,
                sourceEndSec: span.endSec,
                syllableId: `span:${projection.takeId}:${index}:${Math.round(span.startSec * 1000)}`,
                label: take?.syllableSpansSec ? "syllable" : "midiNote",
            };
            for (const interval of projectTakeSpan(projection, takeSpan, framePeriodMs)) {
                const cycleSpan = { ...takeSpan, cycle: interval.cycle };
                out.push({
                    id: `base:${hash(`${clip.id}|${projection.takeId}|${interval.cycle}|${Math.round(span.startSec * 1000)}|${Math.round(span.endSec * 1000)}`)}`,
                    start: interval.startFrame,
                    end: interval.endFrameExclusive,
                    kind: "base",
                    members: [memberOf(cycleSpan, interval)],
                    value: null,
                    rate: projection.rate,
                    priority: 1,
                    seq: 0,
                    zOrder,
                });
            }
        });
    });
    return out;
}

function projectUserBlock(
    clips: readonly ParamBlockSourceClip[],
    block: UserParamBlock,
    framePeriodMs: number,
): { candidates: Candidate[]; orphan: number } {
    if (block.geometry.kind === "root") {
        const start = toFrame(block.geometry.span.startSec, framePeriodMs);
        const end = toFrame(block.geometry.span.endSec, framePeriodMs);
        if (end <= start) return { candidates: [], orphan: 0 };
        return {
            candidates: [
                {
                    id: block.id,
                    start,
                    end,
                    kind: block.manual ? "manual" : "user",
                    members: [],
                    value: block.value,
                    transitionWidthSec: block.transitionOutSec,
                    rate: 1,
                    priority: block.manual ? 4 : 3,
                    seq: block.createdSeq,
                    zOrder: 0,
                },
            ],
            orphan: 0,
        };
    }
    let orphan = 0;
    const projected: Array<{ start: number; end: number; member: ProjectedMember; rate: number }> =
        [];
    for (const member of block.geometry.members) {
        const clip = clips.find((item) => item.id === member.clipId);
        if (clip === undefined) {
            orphan += 1;
            continue;
        }
        const projection = projectionForClip(clip);
        let matched = false;
        for (const interval of projectTakeSpan(projection, member, framePeriodMs)) {
            if (interval.cycle !== member.cycle) continue;
            matched = true;
            projected.push({
                start: interval.startFrame,
                end: interval.endFrameExclusive,
                member: memberOf(member, interval),
                rate: projection.rate,
            });
        }
        if (!matched) orphan += 1;
    }
    if (projected.length === 0) return { candidates: [], orphan };
    projected.sort((a, b) => a.start - b.start || a.end - b.end);
    const runs: Array<{ start: number; end: number; members: ProjectedMember[]; rate: number }> =
        [];
    for (const item of projected) {
        const last = runs[runs.length - 1];
        if (last !== undefined && item.start <= last.end + 1) {
            last.end = Math.max(last.end, item.end);
            last.members.push(item.member);
        } else {
            runs.push({
                start: item.start,
                end: item.end,
                members: [item.member],
                rate: item.rate,
            });
        }
    }
    return {
        candidates: runs.map((run, index) => ({
            id: runs.length === 1 ? block.id : `${block.id}:${index}`,
            start: run.start,
            end: run.end,
            kind: block.manual ? "manual" : run.members.length > 1 ? "merged" : "user",
            members: run.members,
            value: block.value,
            transitionWidthSec: block.transitionOutSec,
            rate: run.rate,
            priority: block.manual ? 4 : 3,
            seq: block.createdSeq,
            zOrder: 0,
        })),
        orphan,
    };
}

function anchorMatches(base: TakeSpan, member: TakeSpan): boolean {
    if (
        base.syllableId !== undefined &&
        member.syllableId !== undefined &&
        base.syllableId === member.syllableId
    ) {
        return true;
    }
    if (
        Math.abs(base.sourceStartSec - member.sourceStartSec) <= ANCHOR_TOLERANCE_SEC &&
        Math.abs(base.sourceEndSec - member.sourceEndSec) <= ANCHOR_TOLERANCE_SEC
    ) {
        return true;
    }
    const overlap =
        Math.min(base.sourceEndSec, member.sourceEndSec) -
        Math.max(base.sourceStartSec, member.sourceStartSec);
    const width = Math.min(
        base.sourceEndSec - base.sourceStartSec,
        member.sourceEndSec - member.sourceStartSec,
    );
    return overlap > 0 && width > EPS && overlap / width >= ANCHOR_OVERLAP_RATIO;
}

function suppressed(base: TakeSpan, edits: ParamBlockEditSet): boolean {
    return edits.blocks.some(
        (block) =>
            block.geometry.kind === "take" &&
            block.geometry.members.some(
                (member) =>
                    member.clipId === base.clipId &&
                    member.takeId === base.takeId &&
                    member.cycle === base.cycle &&
                    member.sourceStartSec < base.sourceEndSec - EPS &&
                    base.sourceStartSec < member.sourceEndSec - EPS &&
                    anchorMatches(base, member),
            ),
    );
}

function subtract(
    start: number,
    end: number,
    occupied: readonly [number, number][],
): Array<[number, number]> {
    let pieces: Array<[number, number]> = [[start, end]];
    for (const [a, b] of occupied) {
        const next: Array<[number, number]> = [];
        for (const [s, e] of pieces) {
            if (b <= s || a >= e) {
                next.push([s, e]);
                continue;
            }
            if (s < a) next.push([s, a]);
            if (e > b) next.push([b, e]);
        }
        pieces = next;
        if (pieces.length === 0) break;
    }
    return pieces;
}

export function composeParamBlockView(
    rootTrackId: string,
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    framePeriodMs: number,
): ParamBlockView {
    const fp = Math.max(0.1, framePeriodMs);
    const candidates = baseCandidates(clips, fp).filter((candidate) => {
        const member = candidate.members[0];
        const span: TakeSpan = {
            clipId: member.clipId,
            takeId: member.takeId,
            cycle: member.cycle,
            sourceStartSec: member.sourceStartSec,
            sourceEndSec: member.sourceEndSec,
            ...(member.syllableId === undefined ? {} : { syllableId: member.syllableId }),
            label: "syllable",
        };
        return !suppressed(span, edits);
    });
    let orphanCount = 0;
    for (const block of edits.blocks) {
        if (block.deleted) continue;
        const projected = projectUserBlock(clips, block, fp);
        orphanCount += projected.orphan;
        candidates.push(...projected.candidates);
    }
    candidates.sort(
        (a, b) =>
            b.priority - a.priority ||
            b.seq - a.seq ||
            b.zOrder - a.zOrder ||
            a.id.localeCompare(b.id),
    );
    const occupied: Array<[number, number]> = [];
    const accepted: Candidate[] = [];
    for (const candidate of candidates) {
        const pieces = subtract(candidate.start, candidate.end, occupied);
        occupied.push([candidate.start, candidate.end]);
        pieces.forEach(([start, end], index) => {
            if (end <= start) return;
            const trimmed = start !== candidate.start || end !== candidate.end || pieces.length > 1;
            accepted.push({
                ...candidate,
                id: trimmed ? `${candidate.id}:${start}:${end}:${index}` : candidate.id,
                start,
                end,
                members: candidate.members
                    .map((member) => ({
                        ...member,
                        startFrame: Math.max(member.startFrame, start),
                        endFrameExclusive: Math.min(member.endFrameExclusive, end),
                    }))
                    .filter((member) => member.endFrameExclusive > member.startFrame),
            });
        });
    }
    accepted.sort((a, b) => a.start - b.start || a.end - b.end || a.id.localeCompare(b.id));
    const blocks: ProjectedParamBlock[] = accepted.map((candidate, index) => {
        let transitionFrames: number | undefined;
        const next = accepted[index + 1];
        if (
            next !== undefined &&
            candidate.end === next.start &&
            candidate.transitionWidthSec !== undefined
        ) {
            const width = Math.round(
                (candidate.transitionWidthSec * 1000) / fp / Math.max(1e-9, candidate.rate),
            );
            const maxWidth = Math.min(
                candidate.end - candidate.start,
                Math.floor((next.end - next.start) / 2),
            );
            if (width > 0 && maxWidth > 0) transitionFrames = Math.min(width, maxWidth);
        }
        return {
            id: candidate.id,
            startFrame: candidate.start,
            endFrameExclusive: candidate.end,
            ...(transitionFrames === undefined ? {} : { transitionFrames }),
            kind: candidate.kind,
            members: candidate.members,
            ...(candidate.value === undefined ? {} : { value: candidate.value }),
        };
    });
    return {
        schemaVersion: 1,
        rootTrackId,
        editRevision: edits.revision,
        layoutKey: `layout:${hash(`${rootTrackId}|${fp}|${clips.map((clip) => `${clip.id}:${clip.startSec}:${clip.lengthSec}:${clip.sourceStartSec}:${clip.sourceEndSec}:${clip.playbackRate}:${clip.reversed}:${clip.loopEnabled}`).join("|")}`)}`,
        framePeriodMs: fp,
        blocks,
        orphanCount,
        analysisPending: clips.some(
            (clip) =>
                (clip.sourceStartSec !== undefined || clip.sourceEndSec !== undefined) &&
                (clip.syllableSpansSec?.length ?? 0) === 0 &&
                (clip.midiNoteData?.length ?? 0) === 0,
        ),
    };
}

export function viewBlockToSeconds(block: ProjectedParamBlock, framePeriodMs: number) {
    return {
        startSec: toSec(block.startFrame, framePeriodMs),
        endSec: toSec(block.endFrameExclusive, framePeriodMs),
    };
}
