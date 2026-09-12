/**
 * createNewTrackForDrop.ts — 拖到轨道区下方「新建轨道并落库」的共享编排。
 *
 * 【主要内容】
 * 把「在轨道列表末尾新建一条轨道 → 把待落库的 clip 移到该轨 → 选中新轨」这一段
 * 编排抽成单一事实来源，供渲染内核的 `clip-drag` 手势（拖到全部轨道之下）调用。
 *
 * 【作用】
 * 旧实现把这段逻辑内联在 `useClipDrag` 的 `createNewTrackForDrop` /
 * `createNewTracksForDrop` 里（依赖该 hook 的事件闭包），内核的手势拿不到那些
 * 闭包。若在内核侧重写一遍，就会出现**两份建轨语义**——本工程已因同类重复吃过
 * 亏（`copyClipsFromDrag` 的抽取原因与此完全相同）。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的内核拖拽提交分支（`targetTrackId === NEW_TRACK_SENTINEL`）。
 * - 复用：`addTrackRemote`（后端建轨）与 `moveClipsRemote` / `moveClipRemote`
 *   （批量 / 单条移动），落库语义与旧实现一致。
 * - 独立性：不依赖 React 状态，依赖以参数注入，便于调用方复用其 `sessionRef`。
 *
 * 【设计约束】
 * 1. **建轨失败必须回滚**：先把 clip 还原到原轨 / 原起点（乐观位置已在预览期被
 *    改写成哨兵轨），否则 Redux 会停在一个不存在的哨兵轨道上——表现为 clip
 *    从画面上消失（画在最后一行下方，滚不到也点不到）。回滚用同步的乐观 reducer，
 *    不用异步 thunk。
 * 2. 新轨 id 从后端回包里**按差集**解析，而不是取"最后一条轨道"——并发建轨或
 *    后端返回顺序变化时，取末条会拿到别人的轨道。
 */

import type React from "react";

import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    addTrackRemote,
    moveClipRemote,
    moveClipsRemote,
    moveClipStart,
    moveClipTrack,
    selectTrackRemote,
} from "../../../../features/session/sessionSlice";

/** 依赖集合（全部由调用方注入，本模块不反向依赖 hook）。 */
export interface CreateNewTrackForDropDeps {
    /** 待落库的 clip id 列表（顺序即参与集合顺序）。 */
    readonly clipIds: readonly string[];
    /** 各 clip 的**目标**起点（秒，已含吸附与钳制）。 */
    readonly startSecById: Readonly<Record<string, number>>;
    /** 各 clip 取消 / 失败时回滚到的原轨。 */
    readonly originTrackIdById: Readonly<Record<string, string>>;
    /** 各 clip 取消 / 失败时回滚到的原起点（秒）。 */
    readonly originStartSecById: Readonly<Record<string, number>>;
    readonly dispatch: AppDispatch;
    readonly sessionRef: React.RefObject<SessionState>;
    /** 锁定的参数线是否随 clip 一起移动。 */
    readonly moveLinkedParams: boolean;
}

/**
 * 在轨道列表末尾新建一条轨道，并把指定 clip 全部移过去。
 *
 * 流程：
 * 1. 记录建轨前的轨道 id 集合（用于按差集解析新轨 id）；
 * 2. `addTrackRemote` 建轨；
 * 3. 解析新轨 id（差集 → 回包的 `selected_track_id` → 末条）；
 * 4. 批量（> 1）或单条（= 1）`move*Remote` 落库；
 * 5. 选中新轨（与旧实现 `maybeSelectTargetTrack` 同源）；
 * 6. 任一步失败 → 把 clip 回滚到原轨 / 原起点。
 *
 * @param deps 见 `CreateNewTrackForDropDeps`。
 * @returns 新轨 id；失败时为 null（调用方无需再处理，回滚已在此完成）。
 */
export async function createNewTrackForKernelDrop(
    deps: CreateNewTrackForDropDeps,
): Promise<string | null> {
    const { dispatch, sessionRef } = deps;
    const before = new Set(sessionRef.current.tracks.map((track) => track.id));
    const rollback = (): void => {
        // 乐观位置在预览期已被改写：失败必须还原，否则 Redux 停在不存在的哨兵轨道上。
        //
        // 特殊说明：这里用**同步的乐观 reducer**（`moveClipStart` / `moveClipTrack`）
        // 而不是 `moveClipRemote` thunk——thunk 是异步的，调用方在 `void …` 之后
        // 就不再等它，回滚的 dispatch 可能落在下一次渲染之后，期间画面仍停在哨兵轨
        // （表现为"建轨失败后 clip 消失在最后一行下方"）。回滚是纯前端状态修正，
        // 本就不需要走后端。
        for (const clipId of deps.clipIds) {
            const trackId = deps.originTrackIdById[clipId];
            if (trackId === undefined) continue;
            dispatch(
                moveClipStart({
                    clipId,
                    startSec: Math.max(0, deps.originStartSecById[clipId] ?? 0),
                }),
            );
            dispatch(moveClipTrack({ clipId, trackId }));
        }
    };

    try {
        const response = (await dispatch(
            addTrackRemote({ name: undefined, parentTrackId: null }),
        ).unwrap()) as {
            tracks?: Array<{ id?: string }>;
            selected_track_id?: string | null;
        };
        const nextTracks = Array.isArray(response?.tracks) ? response.tracks : [];
        const created = nextTracks.find((track) => !before.has(String(track?.id)));
        const newTrackId =
            (created && created.id != null ? String(created.id) : null) ??
            (response?.selected_track_id != null ? String(response.selected_track_id) : null) ??
            (nextTracks.length > 0 && nextTracks[nextTracks.length - 1]?.id != null
                ? String(nextTracks[nextTracks.length - 1]?.id)
                : null);
        if (newTrackId === null) {
            rollback();
            return null;
        }

        const moves = deps.clipIds.map((clipId) => ({
            clipId,
            startSec: Math.max(0, Number(deps.startSecById[clipId]) || 0),
            trackId: newTrackId,
        }));
        if (moves.length > 1) {
            await dispatch(
                moveClipsRemote({ moves, moveLinkedParams: deps.moveLinkedParams }),
            ).unwrap();
        } else if (moves.length === 1) {
            await dispatch(
                moveClipRemote({
                    clipId: moves[0].clipId,
                    startSec: moves[0].startSec,
                    trackId: newTrackId,
                    moveLinkedParams: deps.moveLinkedParams,
                }),
            ).unwrap();
        }

        // 选中新轨（与旧实现 `maybeSelectTargetTrack` 同一语义）。
        if (sessionRef.current.selectedTrackId !== newTrackId) {
            void dispatch(selectTrackRemote(newTrackId));
        }
        return newTrackId;
    } catch {
        rollback();
        return null;
    }
}
