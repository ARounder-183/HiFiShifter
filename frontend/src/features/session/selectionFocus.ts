import type { ClipInfo } from "./sessionTypes.js";

export function resolveTrackIdForClipSelection(args: {
    currentTrackId: string | null;
    clips: ClipInfo[];
    clipId: string | null;
    preserveTrackFocus?: boolean;
}): string | null {
    const { currentTrackId, clips, clipId, preserveTrackFocus = false } = args;
    if (!clipId || preserveTrackFocus) {
        return currentTrackId;
    }

    const selectedClip = clips.find((clip) => clip.id === clipId);
    return selectedClip?.trackId ?? currentTrackId;
}
