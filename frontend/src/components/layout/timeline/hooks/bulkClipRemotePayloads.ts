type BulkClipRemoteChange = {
    gain?: number;
    muted?: boolean;
    fadeInSec?: number;
    fadeOutSec?: number;
    /** 倒放开关（后端 ClipStatePatch 支持，乐观更新同步应用）。 */
    reversed?: boolean;
    /** Loop（循环源）开关。 */
    loopEnabled?: boolean;
};

export function buildBulkClipStateUpdates(args: {
    clipIds: string[];
    changesById: Map<string, BulkClipRemoteChange>;
}): Array<{ clipId: string } & BulkClipRemoteChange> {
    return args.clipIds.flatMap((clipId) => {
        const changes = args.changesById.get(clipId);
        if (!changes) return [];
        return [{ clipId, ...changes }];
    });
}

export function buildDuplicateClipsBulkPayload(args: {
    sourceClipIds: string[];
    deltaSec: number;
    copyLinkedParams: boolean;
    applyAutoCrossfade: boolean;
    trackMode: Record<string, unknown>;
    placeOnSelectedTrack?: boolean;
    renameCopies?: boolean;
}) {
    return {
        sourceClipIds: args.sourceClipIds,
        deltaSec: args.deltaSec,
        copyLinkedParams: args.copyLinkedParams,
        applyAutoCrossfade: args.applyAutoCrossfade,
        selectCreatedClips: true,
        trackMode: args.trackMode,
        ...(args.placeOnSelectedTrack ? { placeOnSelectedTrack: true } : {}),
        ...(args.renameCopies === false ? { renameCopies: false } : {}),
    };
}
