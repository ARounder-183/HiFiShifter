import type { ClipInfo } from "../../../features/session/sessionTypes";
import {
    analysisWindowStatus,
    audibleSourceWindow,
    paramBlockAnalysisRequestKey,
    projectionForClip,
    toParamBlockSourceClip,
} from "./paramBlockProjection";

export function paramBlockStaleAnalysisCandidates(args: {
    readonly clips: readonly ClipInfo[];
    readonly rootTrackId: string | null;
    readonly framePeriodMs: number;
    readonly attemptedKeys: ReadonlySet<string>;
}): { clips: ClipInfo[]; keysByClipId: Map<string, string> } {
    const keysByClipId = new Map<string, string>();
    const clips = args.clips.filter((clip) => {
        if (clip.trackId.length === 0 || (clip.midiNoteData?.length ?? 0) > 0 || !clip.sourcePath)
            return false;
        const take = clip.takes?.find((item) => item.id === clip.activeTakeId) ?? clip.takes?.[0];
        const spans = take?.syllableSpansSec ?? clip.syllableSpansSec;
        const analyzed = take?.syllableAnalysisWindowSec ?? clip.syllableAnalysisWindowSec;
        const audible = audibleSourceWindow(projectionForClip(toParamBlockSourceClip(clip)));
        if (analysisWindowStatus(audible, analyzed, spans) === "current") return false;
        const key = paramBlockAnalysisRequestKey({
            rootTrackId: args.rootTrackId ?? "",
            clipId: clip.id,
            activeTakeId: clip.activeTakeId ?? take?.id,
            sourceDigest: clip.sourcePath,
            algorithm: clip.gameBlockBaseline?.algorithm,
            algorithmVersion: clip.gameBlockBaseline?.algorithmVersion,
            audible,
            framePeriodMs: args.framePeriodMs,
        });
        keysByClipId.set(clip.id, key);
        return !args.attemptedKeys.has(key);
    });
    return { clips, keysByClipId };
}
