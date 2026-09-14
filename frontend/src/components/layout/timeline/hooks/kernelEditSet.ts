/**
 * 内核编辑手势 · 参与集合解析（纯函数）
 *
 * 【主要内容】
 * 把「内核回调给出的锚点 clip + 面板的多选集合 + 编组状态」解析成本次编辑
 * **实际作用于哪些 clip**，并把「锚点位移」换算成每个参与者的绝对几何。
 *
 * 【作用】
 * 旧实现的拖拽 / 裁切 / 淡变都由 `useClipDrag`、`useEditDrag` 在 DOM 事件里
 * 就地展开参与集合（多选集合 + 编组展开），因此一次手势会同时移动整组。内核
 * 的手势只回调**锚点 clip** 的位移，若不在这里展开，就会出现
 * 「选中多个 clip 却只移动了一个」「同组 clip 不联动」——属于数据语义错误，
 * 不只是缺视觉。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的内核手势回调（拖拽 / 裁切 / 淡变 / snap offset）。
 * - 依赖：`./useGroupExpansion` 的编组展开（与旧实现同一份实现，避免规则分叉）。
 * - 独立性：无 React / DOM / Redux 依赖，可直接单测。
 *
 * 【与旧实现的规则对齐（逐条）】
 * 1. 参与集合的起点：命中 clip **在多选集合内**时用整个多选集合，否则只有它自己
 *    （旧实现 `useClipDrag` 同源）。
 * 2. 编组展开：`ignoreGrouping`（全局「忽略编组」）为真时不展开；被禁用联动的组
 *    （`disabledGroupIds`）也不展开。**淡变 / 增益手势刻意不展开编组**
 *    （旧实现 `useEditDrag` 的 `supportsGroupExpansion` 明确排除 fade_in /
 *    fade_out / gain），由调用方的 `expandGroups` 决定。
 * 3. 水平钳制：起点不早于 0（`Math.max(0, …)`）。
 * 4. 跨轨：先按**选区最上/最下两条轨道**算出整组可用的轨道偏移区间
 *    （`computeTrackMoveBounds`，与旧实现同一份实现），把**整组共用的那个偏移量**
 *    钳进区间，再逐个参与者套用——每个参与者按各自的初始轨道序号 + 同一偏移量
 *    移动。**不是**逐参与者各自钳到 `[0, 轨道数 − 1]`：那样在轨道列表两端的
 *    越界量对每个成员不同，成员会先被挤到同一轨，整组的相对轨道布局被压扁
 *    （旧实现靠"整组一个偏移量"保持布局）。
 */

import { NEW_TRACK_SENTINEL } from "../constants";
import { computeTrackMoveBounds, type DropMoveInitial } from "./clipDropMoveUtils";
import { expandClipIdsWithGroups } from "./useGroupExpansion";

/** 参与集合解析所需的 clip 字段（结构化子集，便于单测构造）。 */
export interface KernelEditClip {
    readonly id: string;
    readonly groupId?: string;
    readonly startSec: number;
    readonly lengthSec: number;
    readonly trackId: string;
    /** clip 自身的吸附偏移点（秒）；缺省按 0。 */
    readonly snapOffsetSec?: number;
}

/** 一个参与者及其**手势开始时**的几何快照。 */
export interface KernelEditParticipant {
    readonly clipId: string;
    /** 手势开始时的起点（秒）。 */
    readonly startSec: number;
    /** 手势开始时的长度（秒）。 */
    readonly lengthSec: number;
    /** 手势开始时的轨道。 */
    readonly trackId: string;
    /** 手势开始时的轨道序号；轨道不存在时为 -1。 */
    readonly trackIndex: number;
    /** clip 自身的吸附偏移点（秒）。 */
    readonly snapOffsetSec: number;
}

/**
 * 解析本次编辑的参与集合。
 *
 * 流程：确定初始集合（多选或单点）→ 按需做编组展开 → 过滤掉不存在的 clip →
 * 补齐轨道序号与吸附偏移点。
 *
 * @param args.anchorClipId 内核回调的锚点 clip。
 * @param args.multiSelectedClipIds 面板的多选集合。
 * @param args.clips 当前全部 clip。
 * @param args.trackIds 当前轨道顺序（用于算轨道序号）。
 * @param args.ignoreGrouping 全局「忽略编组」开关。
 * @param args.disabledGroupIds 被临时禁用联动编辑的组。
 * @param args.expandGroups 本手势是否允许编组展开（淡变 / 增益为 false）。
 * @returns 参与者列表（含锚点）；锚点不存在时为空数组。
 */
export function resolveKernelEditParticipants(args: {
    readonly anchorClipId: string;
    readonly multiSelectedClipIds: readonly string[];
    readonly clips: readonly KernelEditClip[];
    readonly trackIds: readonly string[];
    readonly ignoreGrouping: boolean;
    readonly disabledGroupIds: readonly string[];
    readonly expandGroups: boolean;
}): KernelEditParticipant[] {
    const anchorExists = args.clips.some((clip) => clip.id === args.anchorClipId);
    if (!anchorExists) return [];

    const initialIds =
        args.multiSelectedClipIds.length > 0 &&
        args.multiSelectedClipIds.includes(args.anchorClipId)
            ? [...args.multiSelectedClipIds]
            : [args.anchorClipId];

    const shouldExpand = args.expandGroups && !args.ignoreGrouping;
    const ids = shouldExpand
        ? expandClipIdsWithGroups(
              initialIds,
              args.clips.map((clip) => ({ id: clip.id, groupId: clip.groupId })),
              false,
              [...args.disabledGroupIds],
          )
        : initialIds;

    const byId = new Map(args.clips.map((clip) => [clip.id, clip]));
    const participants: KernelEditParticipant[] = [];
    for (const id of ids) {
        const clip = byId.get(id);
        if (clip === undefined) continue;
        participants.push({
            clipId: clip.id,
            startSec: Number.isFinite(clip.startSec) ? clip.startSec : 0,
            lengthSec: Number.isFinite(clip.lengthSec) ? clip.lengthSec : 0,
            trackId: clip.trackId,
            trackIndex: args.trackIds.indexOf(clip.trackId),
            snapOffsetSec: Math.max(0, Number(clip.snapOffsetSec) || 0),
        });
    }
    return participants;
}

/**
 * 把锚点位移换算为每个参与者的绝对目标几何。
 *
 * 流程：先把**共享位移**钳制到「最靠左的参与者也不会越过 0」（`delta >= -minStart`）
 * → 逐参与者用「初始值 + 共享位移」计算（**不用上一帧结果累加**，避免逐帧漂移）
 * → 轨道偏移先整组钳进可用区间，再按各自初始序号 + 该偏移量移动。
 *
 * 特殊说明（为什么钳制共享位移而不是逐 clip 钳制）：旧实现先把位移钳到
 * `-minstartSec` 再逐 clip 应用，整组在左边界处**间距保持不变**。若改为逐 clip
 * 独立钳到 0，只有最左那个停在 0、其余继续左移，整组会被压缩——与旧实现分叉。
 *
 * 特殊说明（轨道方向同理）：轨道偏移也只能有一个——先由
 * `computeTrackMoveBounds` 求出「选区整体还能往上/往下挪几条」，
 * 再把整组套用同一个偏移量。逐参与者各自钳制会在列表两端把成员挤到同一轨，
 * 破坏相对布局（与水平方向被压缩是同一类错误）。
 *
 * @param args.participants 参与集合（手势开始时的快照）。
 * @param args.deltaStartSec 锚点的水平位移（秒，已含吸附结果）。
 * @param args.deltaTrack 轨道偏移量（条数，可为负）。
 * @param args.trackIds 当前轨道顺序。
 * @returns `moves`：每个参与者的目标几何（顺序与入参一致）；`deltaStartSec`：
 *   实际生效的共享位移（已钳制）——波纹跟随必须用它，否则左边界处会与参与者错位。
 */
export function applyKernelEditDelta(args: {
    readonly participants: readonly KernelEditParticipant[];
    readonly deltaStartSec: number;
    readonly deltaTrack: number;
    readonly trackIds: readonly string[];
    /**
     * 是否拖到了**全部轨道之下**（新建轨道哨兵）。
     *
     * 【为什么是布尔量而不是"传哨兵 id 进来"】`NEW_TRACK_SENTINEL` 不是一个真实
     * 轨道，`trackIds.indexOf(sentinel)` 恒为 -1；若让调用方把它混进 `trackIds`
     * 做索引运算，末行钳制会把它当成"越界"而退回原轨。这里用显式开关表达意图：
     * 为 true 时**所有参与者**都落到哨兵轨道（面板随后建轨并替换成真实 id）。
     *
     * 特殊说明：新建轨道时**不做末行钳制**——哨兵轨道在列表之外，钳到末行会退回
     * 已有轨道，与用户的"往下拖出新轨"意图相反。
     */
    readonly dropToNewTrack?: boolean;
}): {
    moves: { clipId: string; startSec: number; trackId: string }[];
    deltaStartSec: number;
} {
    const requestedDelta = Number.isFinite(args.deltaStartSec) ? args.deltaStartSec : 0;
    const lastIndex = args.trackIds.length - 1;

    // 共享位移的下界：最靠左的参与者停在 0 时对应的位移量。
    const minStart =
        args.participants.length > 0
            ? Math.min(...args.participants.map((participant) => participant.startSec))
            : 0;
    const deltaStartSec = Math.max(requestedDelta, -minStart);

    // 整组共用的轨道偏移量：由选区最上/最下两条轨道决定可用区间后一次性钳制
    // （与旧实现 `useClipDrag` 的 `computeTrackMoveBounds` 同一份实现）。
    const deltaTrack = resolveKernelTrackOffset(args.participants, args.trackIds, args.deltaTrack);

    const moves = args.participants.map((participant) => {
        let trackId = participant.trackId;
        if (args.dropToNewTrack === true) {
            trackId = NEW_TRACK_SENTINEL;
        } else if (deltaTrack !== 0 && participant.trackIndex >= 0 && lastIndex >= 0) {
            // 整组同一个偏移量 → 成员之间的相对轨道布局保持不变。
            trackId = args.trackIds[participant.trackIndex + deltaTrack] ?? participant.trackId;
        }
        return {
            clipId: participant.clipId,
            startSec: Math.max(0, participant.startSec + deltaStartSec),
            trackId,
        };
    });

    return { moves, deltaStartSec };
}

/**
 * 把「相对锚点轨道的偏移量」钳进整组可用区间。
 *
 * 区间由选区中**最小与最大**轨道序号决定（`computeTrackMoveBounds`）：
 * 只要整组一起上下移动，最上/最下两条轨道就是边界约束，与每个成员各自到边界的
 * 距离无关。任一团员的轨道序号无法解析（轨道已被删除等）时返回 0——与旧实现
 * `if (!trackBounds) allowTrackMove = false` 同源（放弃跨轨而不是产生分叉）。
 *
 * @param participants 参与集合。
 * @param trackIds 当前轨道顺序。
 * @param requestedDelta 锚点解析出的原始轨道偏移量（条数）。
 * @returns 钳制后的整组共用偏移量。
 */
function resolveKernelTrackOffset(
    participants: readonly KernelEditParticipant[],
    trackIds: readonly string[],
    requestedDelta: number,
): number {
    const delta = Number.isFinite(requestedDelta) ? Math.trunc(requestedDelta) : 0;
    if (delta === 0 || participants.length === 0) return delta;
    const initialById: Record<string, DropMoveInitial> = {};
    for (const participant of participants) {
        initialById[participant.clipId] = {
            startSec: participant.startSec,
            trackId: participant.trackId,
        };
    }
    const trackIndexById: Record<string, number> = {};
    trackIds.forEach((trackId, index) => {
        trackIndexById[trackId] = index;
    });
    const bounds = computeTrackMoveBounds({
        trackCount: trackIds.length,
        clipIds: participants.map((participant) => participant.clipId),
        initialById,
        trackIndexById,
    });
    if (bounds === null) return 0;
    return Math.max(bounds.minTrackOffset, Math.min(bounds.maxTrackOffset, delta));
}
