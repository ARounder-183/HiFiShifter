import { clamp, dbToGain, gainToDb } from "../math";

type BulkEditableArgs = {
    activeClipId: string;
    multiSelectedClipIds: string[];
    multiSelectedSet: Set<string>;
};

type ClipGainLike = {
    gain?: number;
};

type ClipLengthLike = {
    lengthSec?: number;
};

export function getBulkEditableClipIds(args: BulkEditableArgs): string[] {
    const { activeClipId, multiSelectedClipIds, multiSelectedSet } = args;
    if (multiSelectedClipIds.length > 0 && multiSelectedSet.has(activeClipId)) {
        return [...multiSelectedClipIds];
    }
    return [activeClipId];
}

/**
 * 单次遍历收集选中 clip 的实时快照：批量编辑热路径（每 rAF 帧）的
 * clipsById 只需选中子集 —— 对全部 clips 建 Map 在千级 clip 工程下
 * 每帧都是无谓的分配与拷贝。
 */
export function collectSelectedClipsById(
    clips: ReadonlyArray<ClipLengthLike & { id: string }>,
    selectedClipIds: readonly string[],
): Map<string, ClipLengthLike> {
    const selected = new Set(selectedClipIds);
    const result = new Map<string, ClipLengthLike>();
    for (const clip of clips) {
        if (selected.has(clip.id)) result.set(clip.id, clip);
    }
    return result;
}

export function applyBulkFadeValue(args: {
    clipIds: string[];
    clipsById: Map<string, ClipLengthLike>;
    target: "fadeInSec" | "fadeOutSec";
    nextValue: number;
}): Array<{ clipId: string; fadeInSec?: number; fadeOutSec?: number }> {
    const { clipIds, clipsById, target, nextValue } = args;
    return clipIds.flatMap((clipId) => {
        const clip = clipsById.get(clipId);
        if (!clip) return [];
        const lengthSec = Math.max(0, Number(clip.lengthSec ?? 0) || 0);
        const value = clamp(nextValue, 0, lengthSec);
        return [
            target === "fadeInSec" ? { clipId, fadeInSec: value } : { clipId, fadeOutSec: value },
        ];
    });
}

export function applyBulkGainDeltaDb(args: {
    clipIds: string[];
    clipsById: Map<string, ClipGainLike>;
    deltaDb: number;
    minDb: number;
    maxDb: number;
}): Array<{ clipId: string; gain: number }> {
    const { clipIds, clipsById, deltaDb, minDb, maxDb } = args;
    return clipIds.flatMap((clipId) => {
        const clip = clipsById.get(clipId);
        if (!clip) return [];
        const baseGain = Number(clip.gain ?? 1) || 1;
        const nextDb = clamp(gainToDb(baseGain) + deltaDb, minDb, maxDb);
        const gain = clamp(dbToGain(nextDb), dbToGain(minDb), dbToGain(maxDb));
        return [{ clipId, gain }];
    });
}
