/**
 * createNewTrackForDrop.ts — 拖到轨道区下方「新建轨道并落库」的共享编排。
 *
 * 【主要内容】
 * 两条建轨路径的共享编排：
 * 1. `createNewTrackForKernelDrop`：新建一条轨道 → 把待落库的 clip 移到该轨 →
 *    选中新轨（服务 **move** 路径：拖到全部轨道之下）。
 * 2. `createTrackIdsForDrop`：只新建 `count` 条**空**轨道并返回其 id（服务 **copy**
 *    路径：副本位置随后由 `duplicate_clips_bulk` 决定，建轨本身不能搬动原 clip）。
 *
 * 【作用】
 * 旧实现把这段逻辑内联在 `useClipDrag` 的 `createNewTrackForDrop` /
 * `createNewTracksForDrop` 里（依赖该 hook 的事件闭包），内核的手势拿不到那些
 * 闭包。若在内核侧重写一遍，就会出现**两份建轨语义**——本工程已因同类重复吃过
 * 亏（`copyClipsFromDrag` 的抽取原因与此完全相同）。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel` 的内核拖拽提交分支（`targetTrackId === NEW_TRACK_SENTINEL`）
 *   与 `copyClipsFromDrag` 注入的 `createNewTrack(s)ForDrop` 依赖。
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
 * 3. copy 路径**不得**复用 `createNewTrackForKernelDrop`：后者会把原 clip 移到新轨，
 *    与随后的复制叠加，等于把原 clip 搬走（详见 `createTrackIdsForDrop` 注释）。
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
import { computeSelectedTrackSpan, type DropMoveInitial } from "./clipDropMoveUtils";

/** 纯建轨的依赖集合。 */
export interface CreateTrackIdsDeps {
    readonly dispatch: AppDispatch;
    readonly sessionRef: React.RefObject<SessionState>;
}

/**
 * 在轨道列表末尾新建 `count` 条**空**轨道，返回它们的新 id（按建轨顺序）。
 *
 * 流程（每条轨道）：记录建轨前的 id 集合 → `addTrackRemote` → 按**差集**解析新 id
 * （回退到回包的 `selected_track_id`，再回退到末条）。
 *
 * 【为什么与 `createNewTrackForKernelDrop` 分开】后者服务 move 路径，会顺手把
 * clip 移到新轨；copy 路径的副本位置由 `duplicate_clips_bulk` 决定，只需要空轨道。
 * 把两者混用会让 copy 先发生一次移动，再发生一次复制 —— 原 clip 被搬走。
 *
 * 【为什么按差集而不是取末条】并发建轨或后端返回顺序变化时，取末条会拿到别人的
 * 轨道（与 `createNewTrackForKernelDrop` 同一约束，见该文件头部设计约束）。
 *
 * 【为什么去重】后备阶梯（差集 → `selected_track_id` → 末条）在「后端未返回新轨且
 * `selected_track_id` 陈旧」时会解析出**同一个 id** 两次。返回重复 id 能通过调用方的
 * 长度校验，却会把两个源轨道静默映射到同一目标轨道；故命中重复时不再入列，让长度
 * 校验按"少了一条"失败。
 *
 * @param deps 注入的 dispatch 与 sessionRef。
 * @param count 要新建的轨道数（<= 0 时返回空数组）。
 * @returns 新轨 id 列表；某条解析失败或被去重时该条不出现（列表可能短于 `count`，
 *          调用方据此判定失败）。
 */
export async function createTrackIdsForDrop(
    deps: CreateTrackIdsDeps,
    count: number,
): Promise<string[]> {
    const createdIds: string[] = [];
    for (let index = 0; index < Math.max(0, Math.floor(count)); index += 1) {
        const before = new Set(deps.sessionRef.current.tracks.map((track) => track.id));
        const res = (await deps
            .dispatch(addTrackRemote({ name: undefined, parentTrackId: null }))
            .unwrap()) as {
            tracks?: Array<{ id?: string }>;
            selected_track_id?: string | null;
        };
        const nextTracks = Array.isArray(res?.tracks) ? res.tracks : [];
        const created = nextTracks.find((track) => !before.has(String(track?.id)));
        const id =
            (created && String(created.id)) ||
            (res?.selected_track_id ? String(res.selected_track_id) : null) ||
            (nextTracks.length > 0 ? String(nextTracks[nextTracks.length - 1]?.id) : null) ||
            null;
        // 去重：后备阶梯（差集 → selected_track_id → 末条）在「后端没返回新轨、
        // 且 selected_track_id 是陈旧值」时会**解析出同一个 id**。调用方
        // （`copyClipsFromDrag`）只校验 `created.length !== span`，重复 id 能通过
        // 长度校验，却会把两个源轨道静默映射到同一目标轨道。
        // 命中重复时不再入列，让长度校验按"少了一条"失败——宁可报错也不静默错映射。
        if (id !== null && !createdIds.includes(id)) createdIds.push(id);
    }
    return createdIds;
}

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
    /**
     * 各 clip 按下时的**轨道序号**（用于解析选区跨度）。
     *
     * 选区跨多条轨道时必须按跨度建同样多的新轨并让成员各自落位
     * （旧实现 `useClipDrag` 的 `hasMixedTrackSelection` +
     * `computeSelectedTrackSpan` + `buildDropToNewTrackMoves` 那一套）；
     * 只建一条轨会把整组塌到同一轨。
     */
    readonly trackIndexById: Readonly<Record<string, number>>;
    readonly dispatch: AppDispatch;
    readonly sessionRef: React.RefObject<SessionState>;
    /** 锁定的参数线是否随 clip 一起移动。 */
    readonly moveLinkedParams: boolean;
}

/**
 * 在轨道列表末尾新建**一条或多条**轨道，并把指定 clip 按各自落点移过去。
 *
 * 流程：
 * 1. 由参与集合的初始轨道序号解析**选区跨度**（`computeSelectedTrackSpan`）；
 * 2. 跨多条轨道时建 `span` 条新轨（否则建 1 条）——与旧实现
 *    `hasMixedTrackSelection` 分支同一语义；
 * 3. 每个 clip 落到「自身来源序号 − 本组最小序号」对应的那条新轨（单轨时全部落它）；
 * 4. 批量（> 1）或单条（= 1）`move*Remote` 落库；
 * 5. 选中第一条新轨（与旧实现 `maybeSelectTargetTrack(created[0])` 同源）；
 * 6. 任一步失败 → 把 clip 回滚到原轨 / 原起点。
 *
 * 【为什么不能只建一条轨】选区跨多条轨道时只建一条会把整组塌到同一轨，
 * 破坏成员之间的相对轨道布局——旧实现按来源跨度建轨正是为了避免这一点。
 *
 * @param deps 见 `CreateNewTrackForDropDeps`。
 * @returns 第一条新轨的 id；失败时为 null（调用方无需再处理，回滚已在此完成）。
 */
export async function createNewTrackForKernelDrop(
    deps: CreateNewTrackForDropDeps,
): Promise<string | null> {
    const { dispatch, sessionRef } = deps;
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
        const initialById: Record<string, DropMoveInitial> = {};
        for (const clipId of deps.clipIds) {
            const trackId = deps.originTrackIdById[clipId];
            if (trackId === undefined) continue;
            initialById[clipId] = {
                startSec: Math.max(0, deps.originStartSecById[clipId] ?? 0),
                trackId,
            };
        }
        const spanInfo = computeSelectedTrackSpan({
            clipIds: [...deps.clipIds],
            initialById,
            trackIndexById: deps.trackIndexById,
        });
        // `span > 1` 等价于旧实现的 `hasMixedTrackSelection`（成员不在同一条轨道）。
        const span = spanInfo !== null && spanInfo.span > 1 ? spanInfo.span : 1;
        const created = await createTrackIdsForDrop({ dispatch, sessionRef }, span);
        if (created.length !== span) {
            rollback();
            return null;
        }

        const resolveTargetTrackId = (clipId: string): string | null => {
            const single = created[0] ?? null;
            if (span === 1 || spanInfo === null) return single;
            const originTrackId = deps.originTrackIdById[clipId];
            const sourceIndex =
                originTrackId === undefined ? Number.NaN : deps.trackIndexById[originTrackId];
            if (!Number.isFinite(sourceIndex)) return single;
            return created[Number(sourceIndex) - spanInfo.minTrackIndex] ?? single;
        };

        const moves: { clipId: string; startSec: number; trackId: string }[] = [];
        for (const clipId of deps.clipIds) {
            const trackId = resolveTargetTrackId(clipId);
            if (trackId === null) continue;
            moves.push({
                clipId,
                // 目标起点取调用方给的**当前**（乐观、含吸附）值。
                startSec: Math.max(0, Number(deps.startSecById[clipId]) || 0),
                trackId,
            });
        }
        if (moves.length === 0) {
            rollback();
            return null;
        }

        if (moves.length > 1) {
            await dispatch(
                moveClipsRemote({ moves, moveLinkedParams: deps.moveLinkedParams }),
            ).unwrap();
        } else {
            await dispatch(
                moveClipRemote({
                    clipId: moves[0].clipId,
                    startSec: moves[0].startSec,
                    trackId: moves[0].trackId,
                    moveLinkedParams: deps.moveLinkedParams,
                }),
            ).unwrap();
        }

        // 选中第一条新轨（与旧实现 `maybeSelectTargetTrack(created[0])` 同一语义）。
        //
        // `applySelectedClip: false` —— 这里的意图只是"把刚建的新轨设为当前轨道"，
        // 不含"恢复某条 clip 的选中"。特别注意：后端的选中记忆是**全工程唯一**的
        // （`state.rs::select_track` 只改 `selected_track_id`，`to_payload()` 返回的
        // 仍是上次 `select_clip` 记下的那条），因此**不存在**"恢复本轨上次选中的
        // clip"这种语义——纯字符串形式只会把用户此前"点空白取消选中"的结果异步复活
        // （契约与 `TimelinePanel.handleKernelSeek` 同源，见提交 019e93ed）。
        const primaryTrackId = created[0] ?? null;
        if (primaryTrackId !== null && sessionRef.current.selectedTrackId !== primaryTrackId) {
            void dispatch(selectTrackRemote({ trackId: primaryTrackId, applySelectedClip: false }));
        }
        return primaryTrackId;
    } catch {
        rollback();
        return null;
    }
}
