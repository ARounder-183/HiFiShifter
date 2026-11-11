/**
 * 时间轴 · 目标轨道落点高亮（独立 store）
 *
 * 【主要内容】
 * 跨轨拖动 Clip 时记录「当前指针落在哪些轨道」，供内容坐标 overlay 画出整行
 * 圆角描边。tone 区分 move / copy / snap，snap 复用时间轴的 origin 颜色。
 *
 * 【为什么独立于 snapHighlight】
 * 吸附高亮是**点 / 线**语义（时间位置），只在吸附命中时出现；目标轨道描边是
 * **行**语义，跨轨拖动全程存在。把行塞进 SnapGuideEntry 会迫使 snap 模型重新
 * 引入 row/track 语义，也会让“没有吸附时是否显示”变得含糊。独立 store 的生命
 * 周期与拖拽一致，且不依赖 React / DOM，便于单测。
 *
 * 【与旧“竖直锁”的区别】
 * 旧实现会在一组候选行之间吸附，指针离开行中心仍锁定旧行；本 store 只反映
 * pointer 当前解析出的目标轨，不参与任何落点计算，也不改变行选择。
 */

export type TrackDropTargetTone = "move" | "copy" | "snap";

export interface TrackDropTargetSnapshot {
    readonly rev: number;
    readonly trackIds: readonly string[];
    readonly tone: TrackDropTargetTone;
}

const EMPTY_SNAPSHOT: TrackDropTargetSnapshot = {
    rev: 0,
    trackIds: [],
    tone: "move",
};

let snapshot: TrackDropTargetSnapshot = EMPTY_SNAPSHOT;
const listeners = new Set<() => void>();

function normalizeTrackIds(trackIds: readonly string[]): string[] {
    const unique = new Set<string>();
    for (const trackId of trackIds) {
        if (typeof trackId === "string" && trackId.length > 0) unique.add(trackId);
    }
    return [...unique].sort();
}

function commit(next: TrackDropTargetSnapshot): void {
    snapshot = next;
    for (const listener of listeners) listener();
}

function sameIds(a: readonly string[], b: readonly string[]): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
    return true;
}

/** Publish the tracks currently under the dragged clip group. */
export function publishTrackDropTarget(input: {
    readonly trackIds: readonly string[];
    readonly tone: TrackDropTargetTone;
}): void {
    const trackIds = normalizeTrackIds(input.trackIds);
    if (trackIds.length === 0) {
        clearTrackDropTarget();
        return;
    }
    if (sameIds(snapshot.trackIds, trackIds) && snapshot.tone === input.tone) return;
    commit({ rev: snapshot.rev + 1, trackIds, tone: input.tone });
}

/** Clear the ring. Idempotent: a second clear does not notify subscribers. */
export function clearTrackDropTarget(): void {
    if (snapshot.trackIds.length === 0) return;
    commit({ rev: snapshot.rev + 1, trackIds: [], tone: "move" });
}

export function getTrackDropTargetSnapshot(): TrackDropTargetSnapshot {
    return snapshot;
}

export function subscribeTrackDropTarget(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}
