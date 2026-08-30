type BulkClipRemoteChange = {
    gain?: number;
    muted?: boolean;
    /** 尺寸编辑（多选裁短/延长/拉伸的原子批量提交；后端 ClipStatePatch 原生支持）。 */
    startSec?: number;
    lengthSec?: number;
    snapOffsetSec?: number;
    clipPlaybackRate?: number;
    fadeInSec?: number;
    fadeOutSec?: number;
    /** REAPER 形状 id / 曲率：拖拽期间的修改必须随最终提交一并落盘，
     *  否则批量 fulfilled 的整份时间线回灌会把拖拽期修改丢弃（回退/无效）。 */
    fadeInShape?: number;
    fadeInDir?: number;
    fadeOutShape?: number;
    fadeOutDir?: number;
    /** 自动交叉淡化（交叉点拖拽/粘贴后跟随）：与手动 fade 分离。 */
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
    /** 倒放开关（后端 ClipStatePatch 支持，乐观更新同步应用）。 */
    reversed?: boolean;
    /** Loop（循环源）开关。 */
    loopEnabled?: boolean;
    /** 源窗口（派生窗口模型下随 Loop 开关等操作一并归一）。 */
    sourceStartSec?: number;
    sourceEndSec?: number;
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
