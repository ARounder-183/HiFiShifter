import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import {
    setClipFades,
} from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";

/**
 * 自动交叉淡入淡出：为同轨道重叠的 clip 对设置 fade。
 * 对每个被拖动的 clip，找同轨道其它 clip 的重叠区域，
 * 设置左侧 clip 的 fadeOut 和右侧 clip 的 fadeIn 为重叠长度。
 * 无论新值比原值大还是小，都直接设置为重叠长度。
 * 同时将 fade 值持久化到后端。
 *
 * 对于没有重叠的方向，保留 clip 原有的 fade 值。
 *
 * 为了性能原因，这里直接调用 webApi.setClipState 持久化到后端，而不是分发 setClipStateRemote。
 * 返回一个 Promise，在所有 webApi.setClipState 调用完成后 resolve。
 */
export function applyAutoCrossfade(
    session: SessionState,
    movedIds: string[],
    dispatch: AppDispatch,
): Promise<void> {
    // 收集每个 clip 的 fadeIn/fadeOut 由重叠产生的值
    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    for (const id of movedIds) {
        const clip = session.clips.find((c) => c.id === id);
        if (!clip) continue;
        const clipStart = Number(clip.startSec);
        const clipEnd = clipStart + Number(clip.lengthSec);

        const sameTrack = session.clips.filter(
            (c) => c.trackId === clip.trackId && c.id !== id,
        );

        for (const other of sameTrack) {
            const otherStart = Number(other.startSec);
            const otherEnd = otherStart + Number(other.lengthSec);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            if (clipStart <= otherStart) {
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(other.id, Math.max(fadeInOverlaps.get(other.id) ?? 0, overlap));
            } else {
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(other.id, Math.max(fadeOutOverlaps.get(other.id) ?? 0, overlap));
            }
        }
    }

    // 先收集所有需要变更的 fade，统一批量处理
    const updates: Array<{ clipId: string; fadeInSec: number; fadeOutSec: number }> = [];
    const allClipIds = new Set([...fadeInOverlaps.keys(), ...fadeOutOverlaps.keys(), ...movedIds]);
    for (const clipId of allClipIds) {
        const clip = session.clips.find((c) => c.id === clipId);
        if (!clip) continue;

        const hasOverlapIn = fadeInOverlaps.has(clipId);
        const hasOverlapOut = fadeOutOverlaps.has(clipId);

        // 有重叠方向 → 使用重叠长度；无重叠方向 → 保留原始 fade 值
        const fadeInSec = hasOverlapIn
            ? (fadeInOverlaps.get(clipId) ?? 0)
            : Number(clip.fadeInSec ?? 0);
        const fadeOutSec = hasOverlapOut
            ? (fadeOutOverlaps.get(clipId) ?? 0)
            : Number(clip.fadeOutSec ?? 0);

        if (Math.abs(fadeInSec - Number(clip.fadeInSec ?? 0)) > 0.001 ||
            Math.abs(fadeOutSec - Number(clip.fadeOutSec ?? 0)) > 0.001) {
            updates.push({ clipId, fadeInSec, fadeOutSec });
        }
    }

    if (updates.length === 0) return Promise.resolve();

    // 同步批量 dispatch 所有 setClipFades（React 18 自动合并为一次重绘），
    // 然后直接调用 webApi 持久化到后端，而不走 setClipStateRemote thunk。
    // setClipStateRemote.fulfilled 会调用 applyTimelineState 替换整个 clips
    // 数组，当多个请求并行时，先完成的响应会用后端旧状态覆盖尚未更新的
    // clip 的本地乐观值，导致 fade 包络闪烁。直接调用 webApi 只做持久化，
    // 不触发 reducer，从而避免闪烁。
    for (const u of updates) {
        dispatch(setClipFades(u));
    }
    const remotePromises = updates.map((u) =>
        webApi.setClipState({
            clipId: u.clipId,
            fadeInSec: u.fadeInSec,
            fadeOutSec: u.fadeOutSec,
        }),
    );
    return Promise.allSettled(remotePromises).then(() => undefined);
}

/**
 * 从后端响应的原始 clip 数据计算自动交叉淡入淡出值。
 * 用于 import thunk 中，在 fulfilled reducer 运行前（Redux state 尚未更新）
 * 直接从后端响应计算并同步 fade 到后端。
 */
export function computeAutoCrossfadeFromPayload(
    allClips: Array<{
        id?: string;
        track_id?: string;
        start_sec?: number;
        length_sec?: number;
        fade_in_sec?: number;
        fade_out_sec?: number;
    }>,
    movedIds: string[],
): Array<{ clipId: string; fadeInSec: number; fadeOutSec: number }> {
    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    for (const id of movedIds) {
        const clip = allClips.find((c) => c.id === id);
        if (!clip) continue;
        const clipStart = Number(clip.start_sec ?? 0);
        const clipEnd = clipStart + Number(clip.length_sec ?? 0);

        const sameTrack = allClips.filter(
            (c) => c.track_id === clip.track_id && c.id !== id,
        );

        for (const other of sameTrack) {
            const otherStart = Number(other.start_sec ?? 0);
            const otherEnd = otherStart + Number(other.length_sec ?? 0);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            if (clipStart <= otherStart) {
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(other.id!, Math.max(fadeInOverlaps.get(other.id!) ?? 0, overlap));
            } else {
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(other.id!, Math.max(fadeOutOverlaps.get(other.id!) ?? 0, overlap));
            }
        }
    }

    const results: Array<{ clipId: string; fadeInSec: number; fadeOutSec: number }> = [];
    const allClipIds = new Set([...fadeInOverlaps.keys(), ...fadeOutOverlaps.keys(), ...movedIds]);
    for (const clipId of allClipIds) {
        const clip = allClips.find((c) => c.id === clipId);
        if (!clip) continue;

        const hasOverlapIn = fadeInOverlaps.has(clipId);
        const hasOverlapOut = fadeOutOverlaps.has(clipId);

        const fadeInSec = hasOverlapIn
            ? (fadeInOverlaps.get(clipId) ?? 0)
            : Number(clip.fade_in_sec ?? 0);
        const fadeOutSec = hasOverlapOut
            ? (fadeOutOverlaps.get(clipId) ?? 0)
            : Number(clip.fade_out_sec ?? 0);

        if (Math.abs(fadeInSec - Number(clip.fade_in_sec ?? 0)) > 0.001 ||
            Math.abs(fadeOutSec - Number(clip.fade_out_sec ?? 0)) > 0.001) {
            results.push({ clipId, fadeInSec, fadeOutSec });
        }
    }

    return results;
}
