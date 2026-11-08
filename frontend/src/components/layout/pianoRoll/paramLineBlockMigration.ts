/** The only bridge from the legacy Clip/member/merge representation. */
import type { ClipInfo } from "../../../features/session/sessionTypes";
import {
    autoBlocksForClips,
    materializeParamBlocks,
    projectSourceSpansToTimeline,
    EMPTY_PARAM_BLOCK_STATE,
    type ParamBlock,
    type ParamBlockSourceClip,
    type ParamBlockState,
} from "./paramBlocks";
import {
    placeParamLineBlock,
    type ParamLineBlock,
    type ParamLineBlockSet,
} from "./paramLineBlocks";
import type {
    ParamBlockMergeRecordPayload as ParamBlockMergeRecord,
    ParamBlockIntervalEditPayload as ParamBlockIntervalEdit,
} from "../../../services/api/timeline";

export function opaqueBlockId(value: string): string {
    let left = 0x811c9dc5,
        right = 0x9e3779b9;
    const stable = value.replace(/hfs-ui-\d+-\d+-/g, "");
    for (let i = 0; i < stable.length; i++) {
        left = Math.imul(left ^ stable.charCodeAt(i), 0x01000193);
        right = Math.imul(right ^ stable.charCodeAt(i), 0x85ebca6b);
    }
    return `block:${(left >>> 0).toString(16)}${(right >>> 0).toString(16)}`;
}

export function legacyBlockSources(
    root: string,
    clips: readonly ClipInfo[],
): ParamBlockSourceClip[] {
    return clips.map((clip) => {
        const take = clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
        const notes = take?.midiNoteData ?? clip.midiNoteData;
        const source: ParamBlockSourceClip = {
            id: clip.id,
            trackId: root,
            startSec: clip.startSec,
            lengthSec: clip.lengthSec,
            takeId: take?.id ?? clip.activeTakeId ?? `${clip.id}_take_1`,
            sourceStartSec: clip.sourceStartSec,
            sourceEndSec: clip.sourceEndSec,
            playbackRate: clip.playbackRate,
            reversed: clip.reversed,
            loopEnabled: clip.loopEnabled,
            contentDurationSec: clip.durationSec ?? null,
            midiNoteData: notes,
            sourceSyllableSpansSec: notes?.length
                ? notes.map((note) => [note.startSec, note.endSec] as const)
                : (take?.syllableSpansSec ?? clip.syllableSpansSec)?.map(
                      (span) => [span.startSec, span.endSec] as const,
                  ),
        };
        return { ...source, syllableSpansSec: projectSourceSpansToTimeline(source) };
    });
}

function intervals(blocks: readonly ParamBlock[], fp: number): ParamLineBlock[] {
    return blocks.reduce<ParamLineBlock[]>((layout, block) => {
        const startFrame = Math.max(0, Math.round((block.startSec * 1000) / fp));
        const endFrameExclusive = Math.max(startFrame + 1, Math.round((block.endSec * 1000) / fp));
        return placeParamLineBlock(layout, {
            id: opaqueBlockId(block.id),
            startFrame,
            endFrameExclusive,
            transitionFrames: Math.max(0, Math.round(((block.transitionSec ?? 0) * 1000) / fp)),
        });
    }, []);
}

export function migrateParamLineBlocks(args: {
    root: string;
    clips: readonly ClipInfo[];
    framePeriodMs: number;
    records: readonly ParamBlockMergeRecord[];
    intervalEdits: readonly ParamBlockIntervalEdit[];
}): ParamLineBlockSet {
    const sources = legacyBlockSources(args.root, args.clips);
    const legacy: ParamBlockState = {
        ...EMPTY_PARAM_BLOCK_STATE,
        persistedMerges: args.records.map((record) => ({
            id: record.mergedGroupId,
            members: record.memberRefs,
        })),
        persistedEdits: args.intervalEdits.map((edit) => ({
            ...edit,
            id: edit.editId,
            members: edit.memberRefs,
        })),
    };
    const current = materializeParamBlocks({
        rootTrackId: args.root,
        clips: sources,
        state: legacy,
    });
    const baseline = autoBlocksForClips(args.root, sources);
    return {
        schemaVersion: 1,
        revision: 0,
        framePeriodMs: args.framePeriodMs,
        blocks: intervals(current, args.framePeriodMs),
        gameBaseline: { revision: 0, segments: intervals(baseline, args.framePeriodMs) },
    };
}

export function analyzedLineIntervals(root: string, clip: ClipInfo, fp: number): ParamLineBlock[] {
    return intervals(autoBlocksForClips(root, legacyBlockSources(root, [clip])), fp);
}
