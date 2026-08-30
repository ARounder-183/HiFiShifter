import type { AppDispatch } from "../../../../app/store";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { setClipAutoFades } from "../../../../features/session/sessionSlice";
import { webApi } from "../../../../services/webviewApi";

/**
 * 编辑前某 clip 每侧是否有同轨重叠。
 *
 * 由于自动交叉淡化长度与手动 fade 分离存储，我们需要知道“这一侧的 fade 是否由
 * 交叉淡化持有”：编辑开始时该侧有重叠，则本次编辑后如果该侧失去重叠，就应把
 * auto 清成 0，手动 fade 自动恢复显示。
 *
 * 该信息**只包含真正受本次编辑影响的 clip**：被编辑 clip 本身 + 编辑前与它们
 * 直接重叠的同轨邻居。不会包含同一轨道上无关的其它 clip。
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

type AutoCrossfadeClipLike = {
    id: string;
    trackId: string;
    startSec: number;
    lengthSec: number;
    autoFadeInSec?: number;
    autoFadeOutSec?: number;
};

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

/** 判断 a 是否完全位于 b 内部（含边界重合）。 */
function isFullyContainedWithin(
    a: { startSec: number; lengthSec: number },
    b: { startSec: number; lengthSec: number },
): boolean {
    const aStart = Number(a.startSec ?? 0);
    const aEnd = aStart + Number(a.lengthSec ?? 0);
    const bStart = Number(b.startSec ?? 0);
    const bEnd = bStart + Number(b.lengthSec ?? 0);
    const eps = 1e-9;
    return aStart >= bStart - eps && aEnd <= bEnd + eps;
}

/**
 * 可用于自动交叉淡化的重叠长度。
 *
 * 若其中一个 clip 完全位于另一个 clip 内部（含起始/终止重合），该重叠不是
 * 有效的交叉淡化关系，返回 0。否则返回原始重叠秒数（不重叠返回 ≤ 0）。
 */
function crossfadeOverlapLengthSec(
    a: { startSec: number; lengthSec: number },
    b: { startSec: number; lengthSec: number },
): number {
    const raw = overlapLengthSec(a, b);
    if (raw <= 0.001) return raw;
    if (isFullyContainedWithin(a, b) || isFullyContainedWithin(b, a)) return 0;
    return raw;
}

/**
 * 构建“编辑开始前”的受影响侧信息。
 *
 * @param clips 编辑开始前的全部 clips（未移动的原始位置）
 * @param editedIds 真正被本次编辑改变的 clip（如被拖拽/裁剪/拉伸的 clip，
 *                  以及波纹编辑中随动的 follower）
 *
 * 只包含被编辑 clip 本身 + 编辑前与它们直接重叠的同轨邻居；其它同轨 clip
 * 不会进入结果，从而保证“只服务因编辑而受影响的淡入淡出包络”。
 */
export function computeInitialCrossfadeSides(
    clips: Array<{ id: string; trackId: string; startSec: number; lengthSec: number }>,
    editedIds: string[],
): Record<string, CrossfadeAffectedClip> {
    const result: Record<string, CrossfadeAffectedClip> = {};
    for (const id of editedIds) {
        result[id] = { fadeIn: false, fadeOut: false };
    }

    for (const id of editedIds) {
        const a = clips.find((c) => c.id === id);
        if (!a) continue;
        for (const b of clips) {
            if (b.trackId !== a.trackId || b.id === a.id) continue;
            if (overlapLengthSec(a, b) <= 0.001) continue;
            if (!result[a.id]) result[a.id] = { fadeIn: false, fadeOut: false };
            if (!result[b.id]) result[b.id] = { fadeIn: false, fadeOut: false };
            if (a.startSec <= b.startSec) {
                result[a.id].fadeOut = true;
                result[b.id].fadeIn = true;
            } else {
                result[a.id].fadeIn = true;
                result[b.id].fadeOut = true;
            }
        }
    }

    return result;
}

/**
 * 自动交叉淡化的核心计算（预览与提交共用，保证一致）。
 *
 * 语义（对齐 REAPER）：
 * - 自动交叉淡化只服务于“因本次编辑而受影响”的淡入淡出侧；
 * - 受影响集合 = 被编辑 clip（含波纹随动 follower）∪ 编辑前直接重叠的邻居
 *   （affectedSides 的键）∪ 当前直接重叠的邻居；
 * - 对每个受影响 clip，只有**面对被编辑 clip 的那一侧**（或编辑前被标记重叠的
 *   那一侧）会被重新计算：有可交叉淡化重叠 → auto = 当前重叠量；
 *   完全包含（一个 clip 整体在另一个内部）或无重叠 → auto = 0；
 * - **其它侧一律保持原值**——不会因为无关 clip 的编辑而改变既有交叉淡化，
 *   也不会把“用户介入后已关闭的自动淡化”错误地重新开启。
 * - `editSides` 限定被编辑 clip 的“哪些侧真正被本次编辑触碰”（如 trim_left 只碰
 *   左缘，不会影响右缘的淡出包络）；未提供时默认双侧都算触碰（整体移动/复制）。
 *
 * @param mode "full" 用于开关打开：受影响侧 = 重叠量或 0；
 *             "clear-only" 用于开关关闭：仅在受影响侧失去重叠时清 0，不创建/更新。
 */
function computeAutoCrossfadeCore(
    clips: AutoCrossfadeClipLike[],
    movedIds: string[],
    affectedSides: Record<string, CrossfadeAffectedClip> | undefined,
    editSides: Record<string, CrossfadeAffectedClip> | undefined,
    mode: "full" | "clear-only",
): AutoFadeUpdate[] {
    const clipById = new Map(clips.map((c) => [c.id, c] as const));
    const movedSet = new Set(movedIds);
    const affected = new Set<string>(movedIds);
    if (affectedSides) {
        for (const id of Object.keys(affectedSides)) {
            if (clipById.has(id)) affected.add(id);
        }
    }

    // 当前与被编辑 clip 直接重叠的同轨邻居。
    for (const id of movedIds) {
        const a = clipById.get(id);
        if (!a) continue;
        for (const b of clips) {
            if (b.trackId !== a.trackId || b.id === a.id) continue;
            if (overlapLengthSec(a, b) > 0.001) {
                affected.add(b.id);
            }
        }
    }

    const updates: AutoFadeUpdate[] = [];
    for (const clipId of affected) {
        const clip = clipById.get(clipId);
        if (!clip) continue;

        const pre = affectedSides?.[clipId] ?? { fadeIn: false, fadeOut: false };
        const isMoved = movedSet.has(clipId);

        // 当前该 clip 每侧与所有同轨 clip 的最大重叠量。
        // raw* 用于判定“这一侧是否被触碰”（任何重叠都算，包含完全包含关系）；
        // fade*Overlap 只保存可产生自动交叉淡化的重叠（完全包含时按 0 处理）。
        let fadeInOverlap = 0;
        let fadeOutOverlap = 0;
        let rawFadeInOverlap = 0;
        let rawFadeOutOverlap = 0;
        for (const other of clips) {
            if (other.trackId !== clip.trackId || other.id === clipId) continue;
            const overlap = overlapLengthSec(clip, other);
            if (overlap <= 0.001) continue;
            const eligibleOverlap = crossfadeOverlapLengthSec(clip, other);
            // clip 是“左侧”一方 → 用 fadeOut；是“右侧”一方 → 用 fadeIn。
            if (clip.startSec <= other.startSec) {
                rawFadeOutOverlap = Math.max(rawFadeOutOverlap, overlap);
                fadeOutOverlap = Math.max(fadeOutOverlap, eligibleOverlap);
            } else {
                rawFadeInOverlap = Math.max(rawFadeInOverlap, overlap);
                fadeInOverlap = Math.max(fadeInOverlap, eligibleOverlap);
            }
        }

        // 每个被编辑 clip 的“本次编辑真正触碰的侧”（未提供时默认双侧，如整体移动/复制）。
        const defaultEdit = { fadeIn: true, fadeOut: true } as const;
        const editFor = (id: string): CrossfadeAffectedClip => editSides?.[id] ?? defaultEdit;

        // 是否有“被编辑 clip”当前在本 clip 的某一侧与之重叠，且它的对应侧被本次编辑触碰。
        const movedCurrentlyTouchesSide = (side: "fadeIn" | "fadeOut"): boolean => {
            for (const other of clips) {
                if (!movedSet.has(other.id) || other.id === clipId) continue;
                if (other.trackId !== clip.trackId) continue;
                if (overlapLengthSec(clip, other) <= 0.001) continue;
                if (side === "fadeOut") {
                    // 被编辑 clip 在本 clip 右侧 → 本 clip fadeOut 面对它的 fadeIn。
                    if (other.startSec >= clip.startSec && editFor(other.id).fadeIn) return true;
                } else if (other.startSec < clip.startSec && editFor(other.id).fadeOut) {
                    // 被编辑 clip 在本 clip 左侧 → 本 clip fadeIn 面对它的 fadeOut。
                    return true;
                }
            }
            return false;
        };

        // 编辑前是否有“被编辑 clip”的对应侧已被标记重叠（用于拖开后清除失去的 overlap）。
        const movedPreTouchesSide = (side: "fadeIn" | "fadeOut"): boolean => {
            for (const id of movedIds) {
                if (!affectedSides?.[id]) continue;
                if (side === "fadeIn") {
                    // clip 的 fadeIn 面对左侧被编辑 clip 的 fadeOut。
                    if (affectedSides[id].fadeOut && editFor(id).fadeOut) return true;
                } else if (affectedSides[id].fadeIn && editFor(id).fadeIn) {
                    return true;
                }
            }
            return false;
        };

        // 受影响侧：
        // - 被编辑 clip 自身：只有“本次编辑真正触碰的侧”才可能受影响（例如 trim_left
        //   不会影响右缘的淡出包络；trim_right 不会影响左缘的淡入包络）；
        // - 未被编辑的邻居：只在“被编辑 clip 的对应侧被触碰”时受关联影响。
        const fadeInInvolved = isMoved
            ? editFor(clipId).fadeIn && (pre.fadeIn || rawFadeInOverlap > 0.001)
            : movedPreTouchesSide("fadeIn") || movedCurrentlyTouchesSide("fadeIn");
        const fadeOutInvolved = isMoved
            ? editFor(clipId).fadeOut && (pre.fadeOut || rawFadeOutOverlap > 0.001)
            : movedPreTouchesSide("fadeOut") || movedCurrentlyTouchesSide("fadeOut");

        // 记录“本次拖拽已影响的侧”：即使后续该侧不再重叠（例如拖拽中先重叠、
        // 又分开），它仍会持续被视为受影响侧，从而在无重叠时正确清 0。
        // 这样“开始时未重叠 → 拖成重叠 → 再拖开”也能在预览中实时回到手动 fade。
        if (affectedSides) {
            let entry = affectedSides[clipId];
            if (!entry) {
                entry = affectedSides[clipId] = { fadeIn: false, fadeOut: false };
            }
            if (fadeInInvolved) entry.fadeIn = true;
            if (fadeOutInvolved) entry.fadeOut = true;
        }

        const currentAutoIn = Number(clip.autoFadeInSec ?? 0);
        const currentAutoOut = Number(clip.autoFadeOutSec ?? 0);

        const computeNext = (overlap: number, current: number): number => {
            if (mode === "clear-only") {
                // 开关关闭：只在受影响侧失去重叠时清 0，有重叠则保持现状。
                return overlap > 0.001 ? current : 0;
            }
            // 开关打开：受影响侧 = 当前重叠量（无重叠为 0）。
            return overlap > 0.001 ? overlap : 0;
        };

        const nextAutoIn = fadeInInvolved
            ? computeNext(fadeInOverlap, currentAutoIn)
            : currentAutoIn;
        const nextAutoOut = fadeOutInvolved
            ? computeNext(fadeOutOverlap, currentAutoOut)
            : currentAutoOut;

        if (
            Math.abs(nextAutoIn - currentAutoIn) > 0.001 ||
            Math.abs(nextAutoOut - currentAutoOut) > 0.001
        ) {
            updates.push({ clipId, autoFadeInSec: nextAutoIn, autoFadeOutSec: nextAutoOut });
        }
    }

    return updates;
}

function sessionClipsToLike(session: SessionState): AutoCrossfadeClipLike[] {
    return session.clips.map((c) => ({
        id: c.id,
        trackId: String(c.trackId),
        startSec: Number(c.startSec ?? 0),
        lengthSec: Number(c.lengthSec ?? 0),
        autoFadeInSec: c.autoFadeInSec,
        autoFadeOutSec: c.autoFadeOutSec,
    }));
}

/**
 * 计算自动交叉淡化（拖拽/裁剪：预览与提交共用同一套判定，保证二者一致）。
 *
 * 只影响“因本次编辑而受影响”的淡入淡出侧；无关侧保持原值。
 */
export function computeAutoCrossfadeUpdates(
    session: SessionState,
    movedIds: string[],
    affectedSides?: Record<string, CrossfadeAffectedClip>,
    editSides?: Record<string, CrossfadeAffectedClip>,
): AutoFadeUpdate[] {
    return computeAutoCrossfadeCore(
        sessionClipsToLike(session),
        movedIds,
        affectedSides,
        editSides,
        "full",
    );
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
        /** 被编辑 clip 中本次编辑真正触碰的侧（缺省 = 双侧，如整体移动/复制）。 */
        editSides?: Record<string, CrossfadeAffectedClip>;
    },
): Promise<void> {
    const checkpoint = Boolean(opts?.checkpoint);
    const updates = computeAutoCrossfadeUpdates(
        session,
        movedIds,
        opts?.affectedSides,
        opts?.editSides,
    );

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
 * 计算“应当被清空”的自动交叉淡化（auto → 0）。
 *
 * 当编辑使某个原本有自动交叉淡化的侧不再有重叠时，即使“自动交叉淡化”开关
 * 处于关闭状态，也必须把该侧 auto 清成 0，否则导入/历史遗留的自动值会永久
 * 盖住手动淡化。同样只处理“因本次编辑而受影响”的侧。
 *
 * 本函数**只清除、绝不创建/更新**自动淡化。
 */
export function computeDetachedAutoCrossfadeClears(
    session: SessionState,
    movedIds: string[],
    affectedSides?: Record<string, CrossfadeAffectedClip>,
    editSides?: Record<string, CrossfadeAffectedClip>,
): AutoFadeUpdate[] {
    return computeAutoCrossfadeCore(
        sessionClipsToLike(session),
        movedIds,
        affectedSides,
        editSides,
        "clear-only",
    );
}

/**
 * 应用“仅清除已脱离重叠的自动交叉淡化”（开关关闭时也执行）。
 *
 * 用于：开关关闭但已有自动 fade（如 REAPER 导入）时，clip 分开后手动 fade
 * 能恢复显示；不会创建新的自动交叉淡化。
 */
export function applyDetachedAutoCrossfadeClears(
    session: SessionState,
    movedIds: string[],
    dispatch: AppDispatch,
    affectedSides?: Record<string, CrossfadeAffectedClip>,
    editSides?: Record<string, CrossfadeAffectedClip>,
): Promise<void> {
    const updates = computeDetachedAutoCrossfadeClears(session, movedIds, affectedSides, editSides);

    if (updates.length === 0) return Promise.resolve();

    for (const u of updates) {
        dispatch(setClipAutoFades(u));
    }
    const remotePromises = updates.map((u) =>
        webApi.setClipState({
            clipId: u.clipId,
            autoFadeInSec: u.autoFadeInSec,
            autoFadeOutSec: u.autoFadeOutSec,
            checkpoint: false,
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
    editSides?: Record<string, CrossfadeAffectedClip>,
): void {
    for (const u of computeAutoCrossfadeUpdates(session, movedIds, affectedSides, editSides)) {
        dispatch(setClipAutoFades(u));
    }
}

/**
 * 从后端响应的原始 clip 数据计算自动交叉淡化值（复制粘贴/导入路径）。
 *
 * `movedIds` 为本次新建/复制的 clip；结果只更新“面对这些新 clip 的重叠侧”，
 * 既有 clip 与其它邻居（如既有交叉淡化）的那一侧保持原值。
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
    const clips: AutoCrossfadeClipLike[] = allClips
        .filter((c): c is { id: string; track_id?: string } & typeof c => Boolean(c.id))
        .map((c) => ({
            id: c.id!,
            trackId: String(c.track_id ?? ""),
            startSec: Number(c.start_sec ?? 0),
            lengthSec: Number(c.length_sec ?? 0),
            autoFadeInSec: Number(c.auto_fade_in_sec ?? 0),
            autoFadeOutSec: Number(c.auto_fade_out_sec ?? 0),
        }));
    return computeAutoCrossfadeCore(clips, movedIds, undefined, undefined, "full");
}
