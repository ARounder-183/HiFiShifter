/**
 * 时间轴 · 拖动目标轨道解析（纯函数）
 *
 * 把“本次拖拽移动了哪些 clip、它们从哪来、是否落到新建轨道哨兵”解析成
 * 目标轨道环的载荷。与吸附数学完全分离：这里只决定“哪些现有轨道应该被描边”，
 * 不参与任何行命中或落点计算。
 */
import { NEW_TRACK_SENTINEL } from "./constants";
import type { TrackDropTargetTone } from "./trackDropTargetHighlight";

export interface TrackDropTargetMove {
    readonly clipId: string;
    readonly trackId: string;
}

export interface TrackDropTargetDragInput {
    readonly moves: readonly TrackDropTargetMove[];
    readonly participants: readonly TrackDropTargetMove[];
    readonly dropToNewTrack: boolean;
    readonly snapActive: boolean;
    readonly copyMode: boolean;
}

export interface TrackDropTargetDragResult {
    readonly trackIds: readonly string[];
    readonly tone: TrackDropTargetTone;
}

/**
 * 只在**真正跨轨**时返回描边载荷；同轨拖动、空移动或哨兵落点返回 null。
 * 多选跨轨时返回全部目标轨（去重后交给 store 排序）。
 */
export function resolveTrackDropTarget(
    input: TrackDropTargetDragInput,
): TrackDropTargetDragResult | null {
    if (input.dropToNewTrack) return null;
    const initialTrackById = new Map(
        input.participants.map((participant) => [participant.clipId, participant.trackId]),
    );
    const crossTrack = input.moves.some(
        (move) => move.trackId !== initialTrackById.get(move.clipId),
    );
    if (!crossTrack) return null;
    const trackIds = [
        ...new Set(
            input.moves
                .map((move) => move.trackId)
                .filter((trackId) => trackId !== NEW_TRACK_SENTINEL),
        ),
    ];
    if (trackIds.length === 0) return null;
    return {
        trackIds,
        tone: input.snapActive ? "snap" : input.copyMode ? "copy" : "move",
    };
}
