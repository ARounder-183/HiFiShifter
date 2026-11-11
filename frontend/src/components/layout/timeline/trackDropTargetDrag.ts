/**
 * 时间轴 · 拖动目标轨道解析（纯函数）
 *
 * 【主要内容】把「本次拖拽移动了哪些 clip、它们从哪来、是否落到新建轨道哨兵、
 * 这一刻是否**真的**吸附住了」解析成目标轨道描边的载荷。
 *
 * 【为什么判据是 snapped 而不是「吸附开关」】
 * 描边表达的是「横向位置被钉住了」这一**事实**。旧实现按 `snapActive`（开关）
 * 上色，于是只要开了吸附、跨轨拖动就一路画着描边 —— 开关打开但一个候选都没吸到
 * 时，描边给用户的是一个假的「已对齐」信号。现在没有吸附命中就没有描边：
 * 「未吸附却高亮」在类型上不可能发生（返回 null）。
 *
 * 【边界】只决定「哪些现有轨道应该被描边」；不参与吸附计算、不参与行命中、
 * 不决定落点。
 */
import { NEW_TRACK_SENTINEL } from "./constants";

export interface TrackDropTargetMove {
    readonly clipId: string;
    readonly trackId: string;
}

export interface TrackDropTargetDragInput {
    readonly moves: readonly TrackDropTargetMove[];
    readonly participants: readonly TrackDropTargetMove[];
    readonly dropToNewTrack: boolean;
    /**
     * 本次指针位置是否**真的**吸附到了候选。
     *
     * 【为什么不是 snapActive】snapActive 是「吸附开关」；描边表达的是「这一刻
     * 横向位置被钉住了」。开关打开但没吸到任何东西时，描边会给用户一个假的
     * 「已对齐」信号 —— 这正是「没吸附也高亮」的报告。
     */
    readonly snapped: boolean;
}

export interface TrackDropTargetDragResult {
    readonly trackIds: readonly string[];
}

/**
 * 只在**真正跨轨且吸附命中**时返回描边载荷；其余情况一律 null。
 *
 * 多选跨轨时返回全部目标轨（去重后交给 store 排序）。
 */
export function resolveTrackDropTarget(
    input: TrackDropTargetDragInput,
): TrackDropTargetDragResult | null {
    if (input.dropToNewTrack) return null;
    if (!input.snapped) return null;
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
    return { trackIds };
}
