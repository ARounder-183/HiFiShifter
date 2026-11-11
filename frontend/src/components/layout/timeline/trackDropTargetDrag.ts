/**
 * 时间轴 · 拖动目标轨道解析（纯函数）
 *
 * 【主要内容】把「本次拖拽移动了哪些 clip、它们从哪来、是否落到新建轨道哨兵、
 * 这一刻是否**被拖拽起点吸附钉住了**」解析成目标轨道描边的载荷。
 *
 * 【为什么判据是 originPinned 而不是「有没有吸附命中」】
 * 描边表达的是「这一趟只换了轨道，时间没动」。而 `snapTimelineDetailed` 的命中
 * 候选是整个时间轴吸附体系（网格 / 相邻 Clip 边缘 / 播放头 / 选区标记 / 采样率
 * / 循环边界）：只要吸附开着，跨轨拖动几乎**必然**命中其中之一，于是描边几乎恒亮
 * ——用户报告的就是「什么情况下只要是吸附都会高亮」。
 *
 * 用户要的判据是「被拖的 Clip 被吸回**它自己的原始位置**」。那正是
 * `resolveClipOriginSnap().active` 的定义域（跨轨 + 横向漂移落在 24~32px 阈值内），
 * 也正好是竖向对齐线 `publishClipOriginSnapHighlight` 的触发条件。判据、吸附、
 * 对齐线三者同源：**描边与竖线是同一件事的两种投影**，必须同时出现、同时消失。
 *
 * 【为什么不再有 snapped 这个入参】它除了喂描边之外没有任何消费者（落点由
 * `applyKernelEditDelta` 单独决定）。删掉它，「用错判据」在类型上不可能发生。
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
     * 被拖动的 Clip 是否被**水平吸附钉回了它自己的起点**。
     *
     * 【为什么不是「有没有吸附命中」】网格 / 相邻 Clip 边缘 / 播放头 / 选区标记
     * 的命中都只是「顺手吸到了别的东西」，给整行画一圈描边会把它们说成同一件事。
     * 【为什么这个判据不会漏】「只换轨道」正是 `resolveClipOriginSnap` 的定义域：
     * 它只在跨轨且横向漂移落在阈值内时 active。
     */
    readonly originPinned: boolean;
}

export interface TrackDropTargetDragResult {
    readonly trackIds: readonly string[];
}

/**
 * 只在**真正跨轨且被吸回原起点**时返回描边载荷；其余情况一律 null。
 *
 * 多选跨轨时返回全部目标轨（去重后交给 store 排序）。
 */
export function resolveTrackDropTarget(
    input: TrackDropTargetDragInput,
): TrackDropTargetDragResult | null {
    if (input.dropToNewTrack) return null;
    if (!input.originPinned) return null;
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
