/**
 * 时间轴 · copy 拖拽的落库编排（共享函数）
 *
 * 【为什么单独抽出来】
 * 这段编排原先只存在于 `useClipDrag` 内部（事件驱动、依赖 `drag` 局部状态），
 * 而时间轴渲染内核的 `clip-drag` 手势走 `onDragPreview` / `onDragCommit`，
 * 拿不到 `drag`。若在内核侧重写一遍，就会出现**两份复制语义**——本工程已经
 * 吃过两次同类苦头（`snapTimelineDetailed` 的 `highlight` 漏传、`moveSnapOffsetSec`
 * 硬编码 0），所以这里抽成单一事实来源，旧 hook 与内核面板都调用它。
 *
 * 【边界】
 * 覆盖「目标轨解析 → 复制 → 选中 / 播放光标 / 自动交叉淡化」这一段。
 * **不含**调用方的状态机收尾（`useClipDrag` 里"拖拽中途从移动切到复制"的回滚、
 * 波纹跟随复位等）——那些只对同一次手势里的中间态有意义，内核手势没有该中间态。
 *
 * 【依赖注入而非直接闭包】
 * `createNewTrackForDrop` / `maybeSelectTargetTrack` 等在 `useClipDrag` 里是局部
 * 闭包（依赖该 hook 的其他状态），因此以 deps 传入。这样本模块保持可单测、
 * 且不反向依赖 hook。
 */

import type React from "react";

import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    checkpointHistory,
    duplicateClipsBulkRemote,
    selectClipRemote,
    seekPlayhead,
    setClipAutoFades,
    setplayheadSec,
} from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";
import { computeAutoCrossfadeFromPayload } from "./autoCrossfade";
import { buildDuplicateClipsBulkPayload } from "./bulkClipRemotePayloads";
import { computeSelectedTrackSpan } from "./clipDropMoveUtils";

/** 复制落库所需的外部依赖。 */
export interface CopyClipsFromDragDeps {
    /** 候选 clip（本函数会再过滤掉已被删除的）。 */
    readonly sourceClipIds: readonly string[];
    /** 各 clip 的初始位置与轨道（拖拽开始时的快照）。 */
    readonly initialById: Readonly<Record<string, { startSec: number; trackId: string }>>;
    /** 初始轨道序号（拖到新轨道时按 span 建轨用）。 */
    readonly initialTrackIndexById: Readonly<Record<string, number>>;
    /** 水平位移（秒）。 */
    readonly deltaSec: number;
    /** 落点是否为新轨道（拖到轨道列表空白处）。 */
    readonly dropToNewTrack: boolean;
    /** 垂直偏移（轨数）。 */
    readonly trackOffset: number;
    readonly allowTrackMove: boolean;
    /** 选中集是否跨多条轨道（决定建 1 条还是按 span 建多条）。 */
    readonly hasMixedTrackSelection: boolean;
    readonly autoCrossfadeEnabled: boolean;
    readonly dispatch: AppDispatch;
    readonly sessionRef: React.RefObject<SessionState>;
    /** 把新副本设为多选（由调用方提供，见 `useClipDrag` 的 deps）。 */
    readonly setMultiSelectedClipIds: (ids: string[]) => void;
    /** 按垂直偏移解析目标轨（普通落点用）。 */
    readonly resolveTrackIdByOffset: (clipId: string) => string | null;
    readonly maybeSelectTargetTrack: (trackId: string | null) => void;
    readonly createNewTracksForDrop: (span: number) => Promise<string[]>;
    readonly createNewTrackForDrop: () => Promise<string | null>;
}

/**
 * 执行一次 copy 拖拽的落库。
 *
 * 流程（与抽取前逐行等价）：
 * 1. `beginUndoGroup` 包裹整段（后端一次撤销步）
 * 2. 解析每个 clip 的目标轨道：新轨道（混合选择按 span 建多轨 / 否则单轨）
 *    或普通落点（有垂直偏移时按偏移解析）
 * 3. `trackMapping` → `trackMode`（`same_track` / `explicit_mapping`）
 * 4. `duplicateClipsBulkRemote` → `createdClipIds`
 * 5. 新副本设为多选 + 选中第一个
 * 6. 播放光标定位到副本中最靠前的起点
 * 7. 开启自动交叉淡化时，按后端回包计算并写回
 *
 * @param deps 见 `CopyClipsFromDragDeps`。
 */
export async function copyClipsFromDrag(deps: CopyClipsFromDragDeps): Promise<void> {
    const {
        dispatch,
        sessionRef,
        initialById,
        initialTrackIndexById,
        deltaSec,
        dropToNewTrack,
        trackOffset,
        allowTrackMove,
        hasMixedTrackSelection,
        autoCrossfadeEnabled,
        setMultiSelectedClipIds,
        resolveTrackIdByOffset,
        maybeSelectTargetTrack,
        createNewTracksForDrop,
        createNewTrackForDrop,
    } = deps;

    // 过滤掉拖拽期间已被删除的 clip（否则会拿不到 initial 而错位）。
    const sourceClipIds = deps.sourceClipIds.filter((id) =>
        sessionRef.current.clips.some((clip) => clip.id === id),
    );
    if (sourceClipIds.length === 0) return;

    dispatch(checkpointHistory());

    await webApi.beginUndoGroup();
    try {
        const targetTrackIdByClipId = new Map<string, string>();
        if (dropToNewTrack) {
            if (hasMixedTrackSelection) {
                const spanInfo = computeSelectedTrackSpan({
                    clipIds: [...deps.sourceClipIds],
                    initialById,
                    trackIndexById: initialTrackIndexById,
                });
                if (!spanInfo) throw new Error("create_track_failed");
                const created = await createNewTracksForDrop(spanInfo.span);
                if (created.length !== spanInfo.span) {
                    throw new Error("create_track_failed");
                }
                for (const clipId of sourceClipIds) {
                    const initial = initialById[clipId];
                    if (!initial) continue;
                    const srcIdx = initialTrackIndexById[initial.trackId];
                    if (!Number.isFinite(srcIdx)) continue;
                    const offset = Number(srcIdx) - spanInfo.minTrackIndex;
                    const targetTrackId = created[offset];
                    if (targetTrackId) {
                        targetTrackIdByClipId.set(clipId, targetTrackId);
                    }
                }
            } else {
                const newTrackId = await createNewTrackForDrop();
                if (!newTrackId) throw new Error("create_track_failed");
                for (const clipId of sourceClipIds) {
                    targetTrackIdByClipId.set(clipId, newTrackId);
                }
            }
        } else {
            for (const clipId of sourceClipIds) {
                const initial = initialById[clipId];
                if (!initial) continue;
                const targetTrackId =
                    allowTrackMove && trackOffset !== 0
                        ? (resolveTrackIdByOffset(clipId) ?? initial.trackId)
                        : initial.trackId;
                targetTrackIdByClipId.set(clipId, targetTrackId);
            }
        }

        const firstTargetTrackId = targetTrackIdByClipId.get(sourceClipIds[0]);
        if (firstTargetTrackId) {
            maybeSelectTargetTrack(firstTargetTrackId);
        }

        const trackMapping = new Map<string, string>();
        for (const clipId of sourceClipIds) {
            const initial = initialById[clipId];
            const targetTrackId = targetTrackIdByClipId.get(clipId);
            if (!initial || !targetTrackId) continue;
            trackMapping.set(initial.trackId, targetTrackId);
        }
        if (trackMapping.size === 0) return;

        const trackMode = Array.from(trackMapping.entries()).every(
            ([sourceTrackId, targetTrackId]) => sourceTrackId === targetTrackId,
        )
            ? { kind: "same_track" as const }
            : {
                  kind: "explicit_mapping" as const,
                  mapping: Object.fromEntries(trackMapping),
              };

        const payload = await dispatch(
            duplicateClipsBulkRemote(
                buildDuplicateClipsBulkPayload({
                    sourceClipIds,
                    deltaSec,
                    copyLinkedParams: sessionRef.current.lockParamLinesEnabled,
                    applyAutoCrossfade: autoCrossfadeEnabled,
                    trackMode,
                    renameCopies: false,
                }),
            ),
        ).unwrap();

        const created: string[] = payload?.createdClipIds ?? [];
        if (!Array.isArray(created) || created.length === 0) return;
        setMultiSelectedClipIds(created);
        void dispatch(selectClipRemote(created[0]));

        // 播放光标定位到副本中最靠前的起点。
        const targetStartSec = sourceClipIds.reduce((min, clipId) => {
            const initial = initialById[clipId];
            if (!initial) return min;
            return Math.min(min, Math.max(0, initial.startSec + deltaSec));
        }, Infinity);
        if (Number.isFinite(targetStartSec)) {
            dispatch(setplayheadSec(targetStartSec));
            void dispatch(seekPlayhead(targetStartSec));
        }

        if (autoCrossfadeEnabled) {
            const allClips = (payload?.clips ?? []) as Array<{
                id?: string;
                track_id?: string;
                start_sec?: number;
                length_sec?: number;
                fade_in_sec?: number;
                fade_out_sec?: number;
            }>;
            const fadeUpdates = computeAutoCrossfadeFromPayload(allClips, created);
            if (fadeUpdates.length > 0) {
                // 复制后的自动交叉淡化写入"自动 fade"（与手动 fade 分离）。
                for (const u of fadeUpdates) {
                    dispatch(
                        setClipAutoFades({
                            clipId: u.clipId,
                            autoFadeInSec: u.autoFadeInSec,
                            autoFadeOutSec: u.autoFadeOutSec,
                        }),
                    );
                    await webApi.setClipState({
                        clipId: u.clipId,
                        autoFadeInSec: u.autoFadeInSec,
                        autoFadeOutSec: u.autoFadeOutSec,
                        checkpoint: false,
                    });
                }
            }
        }
    } finally {
        void webApi.endUndoGroup();
    }
}
