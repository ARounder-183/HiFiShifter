import type { ClipInfo } from "../../../features/session/sessionTypes";

function clipEndSec(clip: ClipInfo): number {
    return clip.startSec + clip.lengthSec;
}

function hasFadeAtTime(clip: ClipInfo, timeSec: number): boolean {
    const fadeInEnd = clip.startSec + Math.max(0, clip.fadeInSec || 0);
    const fadeOutStart = clipEndSec(clip) - Math.max(0, clip.fadeOutSec || 0);
    const inFadeIn = clip.fadeInSec > 0 && timeSec >= clip.startSec && timeSec <= fadeInEnd;
    const inFadeOut = clip.fadeOutSec > 0 && timeSec >= fadeOutStart && timeSec <= clipEndSec(clip);
    return inFadeIn || inFadeOut;
}

function overlapsWithContext(contextClip: ClipInfo, other: ClipInfo): boolean {
    const overlapStart = Math.max(contextClip.startSec, other.startSec);
    const overlapEnd = Math.min(clipEndSec(contextClip), clipEndSec(other));
    return overlapEnd > overlapStart;
}

function sortClipsByTimelineOrder(clips: ClipInfo[]): ClipInfo[] {
    return [...clips].sort((a, b) => {
        if (a.startSec !== b.startSec) return a.startSec - b.startSec;
        return a.id.localeCompare(b.id);
    });
}

export function collectFadeContextClips(params: {
    allClips: ClipInfo[];
    contextClip: ClipInfo;
    contextTimeSec: number;
    explicitOverlappingClipIds?: string[];
}): ClipInfo[] {
    const { allClips, contextClip, contextTimeSec, explicitOverlappingClipIds = [] } = params;

    const candidates =
        explicitOverlappingClipIds.length > 0
            ? allClips.filter(
                  (c) =>
                      explicitOverlappingClipIds.includes(c.id) &&
                      c.trackId === contextClip.trackId,
              )
            : allClips.filter(
                  (c) => c.trackId === contextClip.trackId && hasFadeAtTime(c, contextTimeSec),
              );

    return sortClipsByTimelineOrder(candidates).filter(
        (c, index, arr) =>
            c.id !== contextClip.id &&
            overlapsWithContext(contextClip, c) &&
            (index === 0 || c.id !== arr[index - 1].id),
    );
}

export function sortAndFilterFadedClips(params: {
    clip: ClipInfo;
    overlappingClips: ClipInfo[];
}): ClipInfo[] {
    const { clip, overlappingClips } = params;
    const unique = new Map<string, ClipInfo>();
    for (const item of [clip, ...overlappingClips]) {
        unique.set(item.id, item);
    }
    return sortClipsByTimelineOrder(Array.from(unique.values())).filter(
        (c) => c.fadeInSec > 0 || c.fadeOutSec > 0,
    );
}

/**
 * 多选淡变形状行的"当前值"。
 *
 * 多选时形状行只给一行、选择即批量应用，因此这一行没有"自己的"形状 ——
 * 它代表整组 Clip 的共同状态：
 * - 所有参与 Clip 的形状一致 → 返回该形状（高亮对应按钮）；
 * - 不一致 → 返回 `null`（**不预选任何一项**），避免把某一个 Clip 的形状
 *   误展示成整组的状态，让用户以为点击是"保持不变"。
 *
 * 只统计**该方向确有淡变**的 Clip：没有淡变的一侧形状字段是默认值 0，
 * 把它算进来会让"共同形状"被无意义地判成不一致。
 *
 * 小数变体（如 1.1）按基础族比较（与 `FadeShapeRow` 的高亮规则、REAPER 语义一致）。
 */
export function sharedFadeShape(clips: ClipInfo[], side: "in" | "out"): number | null {
    if (clips.length === 0) return null;
    const shapeOf = (clip: ClipInfo): number => {
        const raw = side === "in" ? clip.fadeInShape : clip.fadeOutShape;
        return Number.isFinite(raw) ? Math.trunc(raw as number) : 0;
    };
    const first = shapeOf(clips[0]);
    return clips.every((clip) => shapeOf(clip) === first) ? first : null;
}
