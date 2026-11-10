/**
 * 参数块合成（Rust `param_blocks::compose` 的预览端口）。
 *
 * 【职责】base 衬底、用户覆盖、优先级、重叠消解、过渡、视图校验。
 * 【约束】不持久化任何结果；同一输入必须得到同一输出。
 *
 * 【base 是衬底，不是“命中即整段删除”】基块永远参与合成，用户块只覆盖它实际
 * 投影到的帧区间。从基块派生的用户块额外携带 `baseCover`，用来表达“我替代了
 * 哪一段基块”：即使可见几何被缩短，被替代的部分也保持空白，而不是重新露出基块。
 */

import {
    baseSourceSpansForClip,
    normalizeLoopTakeSpans,
    projectTakeSpan,
    projectionForClip,
    sourceAtProjectedFrame,
    toFrame,
    toSec,
    type ClipProjection,
    type ParamBlockSourceClip,
} from "./paramBlockProjection";
import { flattenGroupChildren, groupCommonValue, groupLeafMembers } from "./paramBlockTypes";
import type {
    BaseCover,
    ParamBlockEditSet,
    ParamBlockView,
    ProjectedBlockKind,
    ProjectedMember,
    ProjectedParamBlock,
    RootSpan,
    TakeSpan,
    UserParamBlock,
} from "./paramBlockTypes";

const EPS = 1e-9;
const ANCHOR_TOLERANCE_SEC = 0.005;
const ANCHOR_OVERLAP_RATIO = 0.5;

interface Candidate {
    readonly id: string;
    readonly userBlockId?: string;
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
    readonly baseCover?: BaseCover;
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
    interval: { startFrame: number; endFrameExclusive: number; cycle: number },
    projection: ClipProjection,
    framePeriodMs: number,
): ProjectedMember {
    const frameSourceStartSec =
        sourceAtProjectedFrame(projection, interval.startFrame, interval.cycle, framePeriodMs) ??
        span.sourceStartSec;
    const frameSourceEndSec =
        sourceAtProjectedFrame(
            projection,
            interval.endFrameExclusive,
            interval.cycle,
            framePeriodMs,
        ) ?? span.sourceEndSec;
    return {
        clipId: span.clipId,
        takeId: span.takeId,
        cycle: span.cycle,
        sourceStartSec: span.sourceStartSec,
        sourceEndSec: span.sourceEndSec,
        ...(span.syllableId === undefined ? {} : { syllableId: span.syllableId }),
        label: span.label,
        startFrame: interval.startFrame,
        endFrameExclusive: interval.endFrameExclusive,
        frameSourceStartSec,
        frameSourceEndSec,
        visibleSourceStartSec: Math.min(frameSourceStartSec, frameSourceEndSec),
        visibleSourceEndSec: Math.max(frameSourceStartSec, frameSourceEndSec),
    };
}

function baseCandidates(
    clips: readonly ParamBlockSourceClip[],
    framePeriodMs: number,
): Candidate[] {
    const out: Candidate[] = [];
    clips.forEach((clip, zOrder) => {
        const analysis = baseSourceSpansForClip(clip);
        if (analysis === null) return;
        const projection = projectionForClip(clip);
        const normalized = analysis.spans
            .filter(
                (span) =>
                    span.endSec > span.startSec &&
                    Number.isFinite(span.startSec) &&
                    Number.isFinite(span.endSec),
            )
            .slice()
            .sort((a, b) => a.startSec - b.startSec);
        normalized.forEach((span, index) => {
            const identity = `span:${projection.takeId}:${index}:${Math.round(span.startSec * 1000)}`;
            const baseSpan: TakeSpan = {
                clipId: clip.id,
                takeId: projection.takeId,
                cycle: 0,
                sourceStartSec: span.startSec,
                sourceEndSec: span.endSec,
                syllableId: identity,
                label: analysis.label,
            };
            const parts = normalizeLoopTakeSpans(projection, baseSpan);
            parts.forEach((part, partIndex) => {
                const cycleSpan: TakeSpan = {
                    ...part,
                    syllableId: parts.length > 1 ? `${identity}:part${partIndex}` : identity,
                };
                for (const interval of projectTakeSpan(projection, cycleSpan, framePeriodMs)) {
                    if (interval.cycle < cycleSpan.cycle) continue;
                    const intervalSpan: TakeSpan = { ...cycleSpan, cycle: interval.cycle };
                    out.push({
                        id: `base:${hash(`${clip.id}|${projection.takeId}|${interval.cycle}|${Math.round(intervalSpan.sourceStartSec * 1000)}|${Math.round(intervalSpan.sourceEndSec * 1000)}|${intervalSpan.label}`)}`,
                        start: interval.startFrame,
                        end: interval.endFrameExclusive,
                        kind: "base",
                        members: [memberOf(intervalSpan, interval, projection, framePeriodMs)],
                        value: null,
                        rate: projection.rate,
                        priority: 1,
                        seq: 0,
                        zOrder,
                        baseCover: { spans: [intervalSpan], full: true },
                    });
                }
            });
        });
    });
    return out;
}

function projectTakeMembers(
    clips: readonly ParamBlockSourceClip[],
    id: string,
    members: readonly TakeSpan[],
    kind: ProjectedBlockKind,
    value: number | null | undefined,
    transitionWidthSec: number | undefined,
    manual: boolean,
    createdSeq: number,
    baseCover: BaseCover | undefined,
    framePeriodMs: number,
): { candidates: Candidate[]; orphan: number } {
    let orphan = 0;
    const projected: Array<{ start: number; end: number; member: ProjectedMember; rate: number }> =
        [];
    for (const member of members) {
        const clip = clips.find((item) => item.id === member.clipId);
        if (clip === undefined) {
            orphan += 1;
            continue;
        }
        const projection = projectionForClip(clip);
        const parts = normalizeLoopTakeSpans(projection, member);
        let matched = false;
        for (const part of parts) {
            for (const interval of projectTakeSpan(projection, part, framePeriodMs)) {
                /*
                 * 用户成员是**周期特定**的：cycle=N 的成员只画第 N 个周期。
                 * 基块不同：一个 syllable span 在每个循环周期都出现，所以
                 * baseCandidates 用 `interval.cycle < cycleSpan.cycle` 收集
                 * 后续周期；用户块必须精确匹配，否则一个 cycle=0 的成员会
                 * 覆盖整段 Loop，后面的基块被错误吞掉。
                 */
                if (interval.cycle !== part.cycle) continue;
                matched = true;
                projected.push({
                    start: interval.startFrame,
                    end: interval.endFrameExclusive,
                    member: memberOf(part, interval, projection, framePeriodMs),
                    rate: projection.rate,
                });
            }
        }
        if (!matched) orphan += 1;
    }
    if (projected.length === 0) return { candidates: [], orphan };
    projected.sort((a, b) => a.start - b.start || a.end - b.end);
    const runs: Array<{ start: number; end: number; members: ProjectedMember[]; rate: number }> =
        [];
    for (const item of projected) {
        const last = runs[runs.length - 1];
        if (last !== undefined && item.start <= last.end) {
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
            id: runs.length === 1 ? id : `${id}:${index}`,
            userBlockId: id,
            start: run.start,
            end: run.end,
            kind,
            members: run.members,
            value,
            transitionWidthSec,
            rate: run.rate,
            priority: manual ? 4 : 3,
            seq: createdSeq,
            zOrder: 0,
            ...(baseCover === undefined ? {} : { baseCover }),
        })),
        orphan,
    };
}

function projectGroupBlock(
    clips: readonly ParamBlockSourceClip[],
    block: UserParamBlock,
    children: readonly UserParamBlock[],
    framePeriodMs: number,
): { candidates: Candidate[]; orphan: number } {
    const leaves = flattenGroupChildren(block);
    const manual = block.manual === true || leaves.some((child) => child.manual === true);
    const value = block.value ?? groupCommonValue(block);
    const transitionWidthSec =
        block.transitionOutSec ?? children[children.length - 1]?.transitionOutSec;
    const hasRoot = leaves.some((child) => child.geometry.kind === "root");
    if (hasRoot) {
        const spans = leaves
            .map((child) => (child.geometry.kind === "root" ? child.geometry.span : null))
            .filter((span): span is RootSpan => span !== null);
        if (spans.length === 0) return { candidates: [], orphan: 0 };
        /*
         * 根块组必须按每个 child 生成独立 run，不能用一个 min..max 包围盒：
         * 子块之间的空隙属于基块 / 空白，不能被组偷偷填满。
         */
        const runs = spans
            .map((span) => ({
                start: toFrame(span.startSec, framePeriodMs),
                end: toFrame(span.endSec, framePeriodMs),
            }))
            .filter((run) => run.end > run.start)
            .sort((a, b) => a.start - b.start || a.end - b.end);
        if (runs.length === 0) return { candidates: [], orphan: 0 };
        return {
            candidates: runs.map((run, index) => ({
                id: runs.length === 1 ? block.id : block.id + ":" + index,
                userBlockId: block.id,
                start: run.start,
                end: run.end,
                kind: "grouped" as const,
                members: [],
                value,
                transitionWidthSec,
                rate: 1,
                priority: manual ? 4 : 3,
                seq: block.createdSeq,
                zOrder: 0,
            })),
            orphan: 0,
        };
    }
    return projectTakeMembers(
        clips,
        block.id,
        groupLeafMembers(block),
        "grouped",
        value,
        transitionWidthSec,
        manual,
        block.createdSeq,
        undefined,
        framePeriodMs,
    );
}

function projectUserBlock(
    clips: readonly ParamBlockSourceClip[],
    block: UserParamBlock,
    framePeriodMs: number,
): { candidates: Candidate[]; orphan: number } {
    if (block.geometry.kind === "group") {
        return projectGroupBlock(clips, block, block.geometry.children, framePeriodMs);
    }
    if (block.geometry.kind === "root") {
        const start = toFrame(block.geometry.span.startSec, framePeriodMs);
        const end = toFrame(block.geometry.span.endSec, framePeriodMs);
        if (end <= start) return { candidates: [], orphan: 0 };
        return {
            candidates: [
                {
                    id: block.id,
                    userBlockId: block.id,
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
                    ...(block.baseCover === undefined ? {} : { baseCover: block.baseCover }),
                },
            ],
            orphan: 0,
        };
    }
    return projectTakeMembers(
        clips,
        block.id,
        block.geometry.members,
        block.manual ? "manual" : "user",
        block.value,
        block.transitionOutSec,
        block.manual === true,
        block.createdSeq,
        block.baseCover,
        framePeriodMs,
    );
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

function baseSpanOf(candidate: Candidate): TakeSpan | null {
    const span = candidate.baseCover?.spans[0];
    return span ?? null;
}

/** 递归收集组内全部叶子的 baseCover；组本身不携带 baseCover。 */
function collectBaseCovers(block: UserParamBlock): BaseCover[] {
    const covers: BaseCover[] = [];
    if (block.baseCover !== undefined) covers.push(block.baseCover);
    if (block.geometry.kind === "group") {
        for (const child of block.geometry.children) {
            covers.push(...collectBaseCovers(child));
        }
    }
    return covers;
}

/**
 * 收集所有会“拥有”这个基块的覆盖区间。
 *
 * 普通手工块没有 baseCover，不在这里出现；它只通过后面的占用区间覆盖基块，
 * 因此会留下两侧剩余段。
 */
function coverIntervalsForBase(
    base: Candidate,
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    framePeriodMs: number,
): Array<[number, number]> {
    const baseSpan = baseSpanOf(base);
    if (baseSpan === null) return [];
    const intervals: Array<[number, number]> = [];
    for (const block of edits.blocks) {
        const covers = collectBaseCovers(block);
        if (covers.length === 0) {
            // 旧版墓碑：deleted + 无 baseCover，按自身成员投影区间抑制基块。
            if (block.deleted && block.geometry.kind === "take") {
                for (const span of block.geometry.members) {
                    if (
                        span.clipId !== baseSpan.clipId ||
                        span.takeId !== baseSpan.takeId ||
                        span.cycle !== baseSpan.cycle
                    ) {
                        continue;
                    }
                    const clip = clips.find((item) => item.id === span.clipId);
                    if (clip === undefined) continue;
                    const projection = projectionForClip(clip);
                    for (const interval of projectTakeSpan(projection, span, framePeriodMs)) {
                        if (interval.cycle !== span.cycle) continue;
                        const start = Math.max(base.start, interval.startFrame);
                        const end = Math.min(base.end, interval.endFrameExclusive);
                        if (end > start) intervals.push([start, end]);
                    }
                }
            }
            continue;
        }
        for (const cover of covers) {
            for (const span of cover.spans) {
                if (
                    span.clipId !== baseSpan.clipId ||
                    span.takeId !== baseSpan.takeId ||
                    span.cycle !== baseSpan.cycle
                ) {
                    continue;
                }
                if (cover.full && anchorMatches(baseSpan, span)) {
                    return [[base.start, base.end]];
                }
                const clip = clips.find((item) => item.id === span.clipId);
                if (clip === undefined) continue;
                const projection = projectionForClip(clip);
                for (const interval of projectTakeSpan(projection, span, framePeriodMs)) {
                    if (interval.cycle !== span.cycle) continue;
                    const start = Math.max(base.start, interval.startFrame);
                    const end = Math.min(base.end, interval.endFrameExclusive);
                    if (end > start) intervals.push([start, end]);
                }
            }
        }
    }
    return intervals;
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

function basePieceCandidate(
    base: Candidate,
    start: number,
    end: number,
    clips: readonly ParamBlockSourceClip[],
    framePeriodMs: number,
): Candidate | null {
    const member = base.members[0];
    const baseSpan = baseSpanOf(base);
    if (member === undefined || baseSpan === null) return null;
    const clip = clips.find((item) => item.id === member.clipId);
    if (clip === undefined) return null;
    const projection = projectionForClip(clip);
    const a = sourceAtProjectedFrame(projection, start, member.cycle, framePeriodMs);
    const b = sourceAtProjectedFrame(projection, end, member.cycle, framePeriodMs);
    if (a === null || b === null) return null;
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);
    if (!(hi > lo + EPS)) return null;
    const pieceSpan: TakeSpan = {
        clipId: member.clipId,
        takeId: member.takeId,
        cycle: member.cycle,
        sourceStartSec: lo,
        sourceEndSec: hi,
        label: member.label ?? baseSpan.label,
    };
    return {
        ...base,
        id: `${base.id}:${start}:${end}`,
        start,
        end,
        members: [
            memberOf(
                pieceSpan,
                { startFrame: start, endFrameExclusive: end, cycle: pieceSpan.cycle },
                projection,
                framePeriodMs,
            ),
        ],
        baseCover: { spans: [pieceSpan], full: false },
    };
}

function trimMember(
    member: ProjectedMember,
    start: number,
    end: number,
    clips: readonly ParamBlockSourceClip[],
    framePeriodMs: number,
): ProjectedMember {
    const clip = clips.find((item) => item.id === member.clipId);
    if (clip === undefined) {
        return { ...member, startFrame: start, endFrameExclusive: end };
    }
    const projection = projectionForClip(clip);
    const a = sourceAtProjectedFrame(projection, start, member.cycle, framePeriodMs);
    const b = sourceAtProjectedFrame(projection, end, member.cycle, framePeriodMs);
    const frameSourceStartSec = a ?? member.frameSourceStartSec;
    const frameSourceEndSec = b ?? member.frameSourceEndSec;
    return {
        ...member,
        startFrame: start,
        endFrameExclusive: end,
        frameSourceStartSec,
        frameSourceEndSec,
        visibleSourceStartSec: Math.min(frameSourceStartSec, frameSourceEndSec),
        visibleSourceEndSec: Math.max(frameSourceStartSec, frameSourceEndSec),
    };
}

function spanListKey(
    spans:
        | readonly { readonly startSec: number; readonly endSec: number }[]
        | undefined,
): string {
    return (spans ?? [])
        .map((span) => span.startSec.toFixed(6) + "-" + span.endSec.toFixed(6))
        .join(";");
}

function noteListKey(
    notes:
        | readonly {
              readonly startSec: number;
              readonly endSec: number;
              readonly note: number;
          }[]
        | undefined,
): string {
    return (notes ?? [])
        .map(
            (note) =>
                note.startSec.toFixed(6) +
                "-" +
                note.endSec.toFixed(6) +
                ":" +
                note.note,
        )
        .join(";");
}

function takeLayoutKey(take: NonNullable<ParamBlockSourceClip["takes"]>[number]): string {
    return [
        take.id,
        take.sourceStartSec,
        take.sourceEndSec,
        take.playbackRate,
        take.reversed,
        take.loopEnabled,
        take.durationSec,
        take.durationFrames,
        take.sourceSampleRate,
        spanListKey(take.syllableSpansSec),
        spanListKey(take.gameBlockBaseline?.sourceSpansSec),
        noteListKey(take.midiNoteData),
    ].join(":");
}

/**
 * The hash must cover every input that can change projection or block
 * boundaries. In particular, clip-level analysis fallbacks and clip duration
 * are used by projectionForClip even when the take array is present.
 */
function clipLayoutKey(clip: ParamBlockSourceClip): string {
    return [
        clip.id,
        clip.activeTakeId ?? "",
        clip.startSec,
        clip.lengthSec,
        clip.sourceStartSec,
        clip.sourceEndSec,
        clip.playbackRate,
        clip.clipPlaybackRate,
        clip.reversed,
        clip.loopEnabled,
        clip.loopPeriodSec,
        clip.contentDurationSec,
        spanListKey(clip.syllableSpansSec),
        spanListKey(clip.gameBlockBaseline?.sourceSpansSec),
        noteListKey(clip.midiNoteData),
        (clip.takes ?? []).map(takeLayoutKey).join(","),
    ].join(":");
}

export function composeParamBlockView(
    rootTrackId: string,
    clips: readonly ParamBlockSourceClip[],
    edits: ParamBlockEditSet,
    framePeriodMs: number,
): ParamBlockView {
    const fp = Math.max(0.1, framePeriodMs);
    const candidates: Candidate[] = [];
    for (const base of baseCandidates(clips, fp)) {
        const covers = coverIntervalsForBase(base, clips, edits, fp);
        for (const [start, end] of subtract(base.start, base.end, covers)) {
            if (end <= start) continue;
            if (start === base.start && end === base.end) {
                candidates.push(base);
                continue;
            }
            const piece = basePieceCandidate(base, start, end, clips, fp);
            if (piece !== null) candidates.push(piece);
        }
    }
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
            if (candidate.kind === "base" && trimmed) {
                const piece = basePieceCandidate(candidate, start, end, clips, fp);
                if (piece !== null) {
                    accepted.push({ ...piece, id: `${piece.id}:${index}` });
                }
                return;
            }
            accepted.push({
                ...candidate,
                id: trimmed ? `${candidate.id}:${start}:${end}:${index}` : candidate.id,
                start,
                end,
                members: candidate.members
                    .map((member) =>
                        trimMember(
                            member,
                            Math.max(member.startFrame, start),
                            Math.min(member.endFrameExclusive, end),
                            clips,
                            fp,
                        ),
                    )
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
            ...(candidate.userBlockId === undefined ? {} : { userBlockId: candidate.userBlockId }),
            startFrame: candidate.start,
            endFrameExclusive: candidate.end,
            ...(transitionFrames === undefined ? {} : { transitionFrames }),
            kind: candidate.kind,
            members: candidate.members,
            ...(candidate.value === undefined ? {} : { value: candidate.value }),
            ...(candidate.baseCover === undefined ? {} : { baseCover: candidate.baseCover }),
        };
    });
    return {
        schemaVersion: 1,
        rootTrackId,
        editRevision: edits.revision,
        layoutKey: `layout:${hash(
            `${rootTrackId}|${fp}|${clips.map(clipLayoutKey).join("|")}`,
        )}`,
        framePeriodMs: fp,
        blocks,
        orphanCount,
        analysisPending: clips.some((clip) => {
            const take =
                clip.takes?.find((item) => item.id === clip.activeTakeId) ?? clip.takes?.[0];
            return (
                (clip.sourceStartSec !== undefined || clip.sourceEndSec !== undefined) &&
                (take?.syllableSpansSec?.length ?? 0) === 0 &&
                (clip.syllableSpansSec?.length ?? 0) === 0 &&
                (take?.gameBlockBaseline?.sourceSpansSec?.length ?? 0) === 0 &&
                (clip.gameBlockBaseline?.sourceSpansSec?.length ?? 0) === 0 &&
                (take?.midiNoteData?.length ?? 0) === 0 &&
                (clip.midiNoteData?.length ?? 0) === 0
            );
        }),
    };
}

export function viewBlockToSeconds(block: ProjectedParamBlock, framePeriodMs: number) {
    return {
        startSec: toSec(block.startFrame, framePeriodMs),
        endSec: toSec(block.endFrameExclusive, framePeriodMs),
    };
}
