import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { setClipAutoFades } from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";

/**
 * 编辑前某 clip 每侧是否有同轨重叠。
 *
 * 因为自动交叉淡化长度与手动 fade 分离存储，我们需要知道“这一侧的 fade 是否由
 * 交叉淡化持有”：在重叠期间自动值覆盖手动值；一旦分开（当前无重叠、编辑前该侧
 * 有重叠）就把自动值清成 0，手动 fade（fade_in_sec / fade_out_sec）自然恢复显示。
 */
export interface CrossfadeAffectedClip {
    /** 编辑前该 clip 的 fadeIn 侧（左侧）是否有同轨重叠。 */
    fadeIn: boolean;
    /** 编辑前该 clip 的 fadeOut 侧（右侧）是否有同轨重叠。 */
    fadeOut: boolean;
}

export interface AutoFadeUpdate {
    clipId: string;
    autoFadeInSec: number;
    autoFadeOutSec: number;
}

/**
 * 计算自动交叉淡化（预览与提交共用同一套判定，保证二者一致）。
 *
 * 规则（对齐 REAPER：自动交叉淡化长度 ≠ 手动 fade）：
 * - “受影响集合” = 被编辑剪辑 ∪ `affectedSides` 的键（编辑前与之重叠/相邻的邻居，
 *   用于在分开时把旧的自动交叉淡化清成 0）∪ 当前与被编辑剪辑重叠的同轨剪辑。
 * - 对每个受影响剪辑的每一侧，计算**自动交叉淡化长度**：
 *   - 当前有同轨重叠 → auto = 当前重叠长度（渲染/显示用 auto 覆盖手动）；
 *   - 当前无重叠、但编辑前该侧有重叠（被“拖开”）→ auto = 0（手动 fade 恢复生效）；
 *   - 当前无重叠、编辑前也无重叠 → auto 保持当前值（通常为 0；不影响手动 fade）。
 * - 手动 fade（fade_in_sec / fade_out_sec）**永不被修改**。
 */
export function computeAutoCrossfadeUpdates(
    session: SessionState,
    movedIds: string[],
    affectedSides?: Record<string, CrossfadeAffectedClip>,
): AutoFadeUpdate[] {
    const clipById = new Map(session.clips.map((c) => [c.id, c] as const));
    const affected = new Set<string>(movedIds);
    if (affectedSides) {
        for (const id of Object.keys(affectedSides)) {
            if (clipById.has(id)) affected.add(id);
        }
    }

    // 扩展受影响集合：当前与被编辑剪辑重叠的同轨剪辑。
    for (const id of movedIds) {
        const clip = clipById.get(id);
        if (!clip) continue;
        for (const other of session.clips) {
            if (other.trackId !== clip.trackId || other.id === id) continue;
            if (overlapLengthSec(clip, other) > 0.001) {
                affected.add(other.id);
            }
        }
    }

    const updates: AutoFadeUpdate[] = [];
    for (const clipId of affected) {
        const clip = clipById.get(clipId);
        if (!clip) continue;

        let fadeInOverlap = 0;
        let fadeOutOverlap = 0;
        for (const other of session.clips) {
            if (other.trackId !== clip.trackId || other.id === clip.id) continue;
            const overlap = overlapLengthSec(clip, other);
            if (overlap <= 0.001) continue;
            // clip 是“左侧”一方 → 用 fadeOut；是“右侧”一方 → 用 fadeIn。
            if (clip.startSec <= other.startSec) {
                fadeOutOverlap = Math.max(fadeOutOverlap, overlap);
            } else {
                fadeInOverlap = Math.max(fadeInOverlap, overlap);
            }
        }

        const pre = affectedSides?.[clipId] ?? { fadeIn: false, fadeOut: false };
        const currentAutoIn = Number(clip.autoFadeInSec ?? 0);
        const currentAutoOut = Number(clip.autoFadeOutSec ?? 0);

        const nextAutoIn = fadeInOverlap > 0.001 ? fadeInOverlap : pre.fadeIn ? 0 : currentAutoIn;
        const nextAutoOut =
            fadeOutOverlap > 0.001 ? fadeOutOverlap : pre.fadeOut ? 0 : currentAutoOut;

        if (
            Math.abs(nextAutoIn - currentAutoIn) > 0.001 ||
            Math.abs(nextAutoOut - currentAutoOut) > 0.001
        ) {
            updates.push({ clipId, autoFadeInSec: nextAutoIn, autoFadeOutSec: nextAutoOut });
        }
    }

    return updates;
}

/** 两个同轨剪辑的重叠长度（秒）；不重叠返回 ≤ 0。 */
function overlapLengthSec(
    a: { startSec: number; lengthSec: number },
    b: { startSec: number; lengthSec: number },
): number {
    const aStart = Number(a.startSec ?? 0);
    const aEnd = aStart + Number(a.lengthSec ?? 0);
    const bStart = Number(b.startSec ?? 0);
    const bEnd = bStart + Number(b.lengthSec ?? 0);
    return Math.min(aEnd, bEnd) - Math.max(aStart, bStart);
}

/**
 * 自动交叉淡化（提交持久化）：本地广播“自动 fade”更新，并把变更持久化到后端。
 *
 * 为了性能原因，这里直接调用 webApi.setClipState 持久化到后端，而不是分发
 * setClipStateRemote thunk（其 fulfilled 会调用 applyTimelineState 替换整个
 * clips 数组，多个并行请求时旧快照会覆盖本地乐观值导致 fade 闪烁）。
 * 返回一个 Promise，在所有 webApi.setClipState 调用完成后 resolve。
 *
 * `checkpoint` 默认 false：这些改动必须并入调用方已开启的 undo group。
 */
export function applyAutoCrossfade(
    session: SessionState,
    movedIds: string[],
    dispatch: AppDispatch,
    opts?: {
        checkpoint?: boolean;
        /** 编辑前每侧重叠关系（分开时只清自动交叉淡化、保留手动 fade）。 */
        affectedSides?: Record<string, CrossfadeAffectedClip>;
    },
): Promise<void> {
    const checkpoint = Boolean(opts?.checkpoint);
    const updates = computeAutoCrossfadeUpdates(session, movedIds, opts?.affectedSides);

    if (updates.length === 0) return Promise.resolve();

    for (const u of updates) {
        dispatch(setClipAutoFades(u));
    }
    const remotePromises = updates.map((u) =>
        webApi.setClipState({
            clipId: u.clipId,
            autoFadeInSec: u.autoFadeInSec,
            autoFadeOutSec: u.autoFadeOutSec,
            checkpoint,
        }),
    );
    return Promise.allSettled(remotePromises).then(() => undefined);
}

/**
 * 自动交叉淡化（拖拽实时预览）：仅本地乐观更新“自动 fade”，不持久化。
 */
export function previewAutoCrossfade(
    session: SessionState,
    movedIds: string[],
    dispatch: AppDispatch,
    affectedSides?: Record<string, CrossfadeAffectedClip>,
): void {
    for (const u of computeAutoCrossfadeUpdates(session, movedIds, affectedSides)) {
        dispatch(setClipAutoFades(u));
    }
}

/**
 * 从后端响应的原始 clip 数据计算自动交叉淡化值（import/paste 路径）。
 */
export function computeAutoCrossfadeFromPayload(
    allClips: Array<{
        id?: string;
        track_id?: string;
        start_sec?: number;
        length_sec?: number;
        auto_fade_in_sec?: number;
        auto_fade_out_sec?: number;
    }>,
    movedIds: string[],
): AutoFadeUpdate[] {
    const fadeInOverlaps = new Map<string, number>();
    const fadeOutOverlaps = new Map<string, number>();

    for (const id of movedIds) {
        const clip = allClips.find((c) => c.id === id);
        if (!clip) continue;
        const clipStart = Number(clip.start_sec ?? 0);
        const clipEnd = clipStart + Number(clip.length_sec ?? 0);

        const sameTrack = allClips.filter((c) => c.track_id === clip.track_id && c.id !== id);

        for (const other of sameTrack) {
            const otherStart = Number(other.start_sec ?? 0);
            const otherEnd = otherStart + Number(other.length_sec ?? 0);
            const overlapStart = Math.max(clipStart, otherStart);
            const overlapEnd = Math.min(clipEnd, otherEnd);
            const overlap = overlapEnd - overlapStart;
            if (overlap <= 0.001) continue;

            if (clipStart <= otherStart) {
                fadeOutOverlaps.set(id, Math.max(fadeOutOverlaps.get(id) ?? 0, overlap));
                fadeInOverlaps.set(
                    other.id!,
                    Math.max(fadeInOverlaps.get(other.id!) ?? 0, overlap),
                );
            } else {
                fadeInOverlaps.set(id, Math.max(fadeInOverlaps.get(id) ?? 0, overlap));
                fadeOutOverlaps.set(
                    other.id!,
                    Math.max(fadeOutOverlaps.get(other.id!) ?? 0, overlap),
                );
            }
        }
    }

    const results: AutoFadeUpdate[] = [];
    const allClipIds = new Set([...fadeInOverlaps.keys(), ...fadeOutOverlaps.keys(), ...movedIds]);
    for (const clipId of allClipIds) {
        const clip = allClips.find((c) => c.id === clipId);
        if (!clip) continue;

        const hasOverlapIn = fadeInOverlaps.has(clipId);
        const hasOverlapOut = fadeOutOverlaps.has(clipId);

        const autoFadeInSec = hasOverlapIn ? (fadeInOverlaps.get(clipId) ?? 0) : 0;
        const autoFadeOutSec = hasOverlapOut ? (fadeOutOverlaps.get(clipId) ?? 0) : 0;

        const currentAutoIn = Number(clip.auto_fade_in_sec ?? 0) || 0;
        const currentAutoOut = Number(clip.auto_fade_out_sec ?? 0) || 0;
        if (
            Math.abs(autoFadeInSec - currentAutoIn) > 0.001 ||
            Math.abs(autoFadeOutSec - currentAutoOut) > 0.001
        ) {
            results.push({ clipId, autoFadeInSec, autoFadeOutSec });
        }
    }

    return results;
}
