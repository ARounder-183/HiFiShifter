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

export type TrackMoveBounds = {
    minTrackOffset: number;
    maxTrackOffset: number;
};

/**
 * 计算整组 Clip 可一起上下移动的轨道偏移范围。
 *
 * 范围必须由选中集合最上/最下两条轨道决定，而不是由每个 Clip 各自到边界的
 * 距离取极值：否则拖到边界后继续越界时，部分 Clip 会先被允许超出轨道范围，
 * 导致它们被 fallback 到锚点轨道、破坏相对轨道顺序。
 */
export function computeTrackMoveBounds(args: {
    trackCount: number;
    clipIds: string[];
    initialById: Record<string, DropMoveInitial>;
    trackIndexById: Record<string, number>;
}): TrackMoveBounds | null {
    let minTrackIndex = Number.POSITIVE_INFINITY;
    let maxTrackIndex = Number.NEGATIVE_INFINITY;

    for (const clipId of args.clipIds) {
        const initial = args.initialById[clipId];
        if (!initial) continue;

        const trackIndex = args.trackIndexById[initial.trackId];
        if (!Number.isFinite(trackIndex)) return null;

        minTrackIndex = Math.min(minTrackIndex, trackIndex);
        maxTrackIndex = Math.max(maxTrackIndex, trackIndex);
    }

    if (!Number.isFinite(minTrackIndex) || !Number.isFinite(maxTrackIndex)) {
        return null;
    }

    return {
        minTrackOffset: -minTrackIndex,
        maxTrackOffset: Math.max(0, args.trackCount - 1 - maxTrackIndex),
    };
}

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
