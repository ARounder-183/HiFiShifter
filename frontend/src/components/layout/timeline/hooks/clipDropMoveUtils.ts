/**
 * clipDropMoveUtils.ts
 *
 * 提供拖拽到新轨道场景下的 move payload 计算工具，
 * 统一使用初始位置 + 最终拖拽偏移量来生成持久化目标，
 * 避免中途状态同步覆盖导致落点偏移。
 */

export type DropMoveInitial = {
    startSec: number;
    trackId: string;
};

export type ClipDropMove = {
    clipId: string;
    startSec: number;
    trackId: string;
};

export type SelectedTrackSpan = {
    minTrackIndex: number;
    maxTrackIndex: number;
    span: number;
};

export function computeSelectedTrackSpan(args: {
    clipIds: string[];
    initialById: Record<string, DropMoveInitial>;
    trackIndexById: Record<string, number>;
}): SelectedTrackSpan | null {
    let minTrackIndex = Number.POSITIVE_INFINITY;
    let maxTrackIndex = Number.NEGATIVE_INFINITY;

    for (const clipId of args.clipIds) {
        const initial = args.initialById[clipId];
        if (!initial) continue;

        const trackIndex = args.trackIndexById[initial.trackId];
        if (!Number.isFinite(trackIndex)) continue;

        minTrackIndex = Math.min(minTrackIndex, trackIndex);
        maxTrackIndex = Math.max(maxTrackIndex, trackIndex);
    }

    if (!Number.isFinite(minTrackIndex) || !Number.isFinite(maxTrackIndex)) {
        return null;
    }

    return {
        minTrackIndex,
        maxTrackIndex,
        span: maxTrackIndex - minTrackIndex + 1,
    };
}

export function buildDropToNewTrackMoves(args: {
    clipIds: string[];
    initialById: Record<string, DropMoveInitial>;
    deltaSec: number;
    resolveTargetTrackId: (clipId: string, initialTrackId: string) => string | null | undefined;
}): ClipDropMove[] {
    const deltaSec = Number(args.deltaSec) || 0;
    const moves: ClipDropMove[] = [];

    for (const clipId of args.clipIds) {
        const initial = args.initialById[clipId];
        if (!initial) continue;

        const targetTrackId = args.resolveTargetTrackId(clipId, initial.trackId);
        if (!targetTrackId) continue;

        moves.push({
            clipId,
            startSec: Math.max(0, Number(initial.startSec) + deltaSec),
            trackId: String(targetTrackId),
        });
    }

    return moves;
}
