import type { ClipInfo } from "../../../features/session/sessionTypes";

function clipEndSec(clip: ClipInfo): number {
    return clip.startSec + clip.lengthSec;
}

function hasFadeAtTime(clip: ClipInfo, timeSec: number): boolean {
    const fadeInEnd = clip.startSec + Math.max(0, clip.fadeInSec || 0);
    const fadeOutStart = clipEndSec(clip) - Math.max(0, clip.fadeOutSec || 0);
    const inFadeIn =
        clip.fadeInSec > 0 && timeSec >= clip.startSec && timeSec <= fadeInEnd;
    const inFadeOut =
        clip.fadeOutSec > 0 && timeSec >= fadeOutStart && timeSec <= clipEndSec(clip);
    return inFadeIn || inFadeOut;
}

function overlapsWithContext(contextClip: ClipInfo, other: ClipInfo): boolean {
    const overlapStart = Math.max(contextClip.startSec, other.startSec);
    const overlapEnd = Math.min(clipEndSec(contextClip), clipEndSec(other));
    return overlapEnd > overlapStart;
}

function sortClipsByTimelineOrder(clips: ClipInfo[]): ClipInfo[] {
    return [...clips].sort((a, b) => {
        if (a.startSec !== b.startSec) return a.startSec - b.startSec;
        return a.id.localeCompare(b.id);
    });
}

export function collectFadeContextClips(params: {
    allClips: ClipInfo[];
    contextClip: ClipInfo;
    contextTimeSec: number;
    explicitOverlappingClipIds?: string[];
}): ClipInfo[] {
    const {
        allClips,
        contextClip,
        contextTimeSec,
        explicitOverlappingClipIds = [],
    } = params;

    const candidates =
        explicitOverlappingClipIds.length > 0
            ? allClips.filter(
                  (c) =>
                      explicitOverlappingClipIds.includes(c.id) &&
                      c.trackId === contextClip.trackId,
              )
            : allClips.filter(
                  (c) =>
                      c.trackId === contextClip.trackId &&
                      hasFadeAtTime(c, contextTimeSec),
              );

    return sortClipsByTimelineOrder(candidates).filter(
        (c, index, arr) =>
            c.id !== contextClip.id &&
            overlapsWithContext(contextClip, c) &&
            (index === 0 || c.id !== arr[index - 1].id),
    );
}

export function sortAndFilterFadedClips(params: {
    clip: ClipInfo;
    overlappingClips: ClipInfo[];
}): ClipInfo[] {
    const { clip, overlappingClips } = params;
    const unique = new Map<string, ClipInfo>();
    for (const item of [clip, ...overlappingClips]) {
        unique.set(item.id, item);
    }
    return sortClipsByTimelineOrder(Array.from(unique.values())).filter(
        (c) => c.fadeInSec > 0 || c.fadeOutSec > 0,
    );
}
