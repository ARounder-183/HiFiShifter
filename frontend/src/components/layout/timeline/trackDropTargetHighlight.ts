/**
 * 时间轴 · 目标轨道落点高亮（独立 store）
 *
 * 【主要内容】跨轨拖动 Clip 且**吸附命中**时记录「当前指针落在哪些轨道」，
 * 供内容坐标 overlay 画出整行圆角描边。
 *
 * 【为什么独立于 snapHighlight】
 * 吸附高亮是**点 / 线**语义（时间位置）；目标轨道描边是**行**语义（落点所在的
 * 轨道）。把行塞进 SnapGuideEntry 会迫使 snap 模型重新引入 row/track 语义。
 * 独立 store 的生命周期与拖拽一致，且不依赖 React / DOM，便于单测。
 *
 * 【为什么没有 tone】描边只表示一件事：**吸附锁定**。它用与吸附竖线相同的
 * origin 色 —— 同一个事实的两种投影。曾经按 move / copy / snap 分三色，于是
 * 「没吸附也画着 move 色」被当成正常状态；删掉 tone 之后，未吸附在类型上就
 * 产生不了描边。
 *
 * 【与旧「竖直锁」的区别】旧实现会在一组候选行之间吸附，指针离开行中心仍锁定
 * 旧行；本 store 只反映 pointer 当前解析出的目标轨，不参与任何落点计算。
 */

export interface TrackDropTargetSnapshot {
    readonly rev: number;
    readonly trackIds: readonly string[];
}

const EMPTY_SNAPSHOT: TrackDropTargetSnapshot = {
    rev: 0,
    trackIds: [],
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

/** Publish the tracks currently under the dragged clip group (snap engaged only). */
export function publishTrackDropTarget(input: { readonly trackIds: readonly string[] }): void {
    const trackIds = normalizeTrackIds(input.trackIds);
    if (trackIds.length === 0) {
        clearTrackDropTarget();
        return;
    }
    if (sameIds(snapshot.trackIds, trackIds)) return;
    commit({ rev: snapshot.rev + 1, trackIds });
}

/** Clear the ring. Idempotent: a second clear does not notify subscribers. */
export function clearTrackDropTarget(): void {
    if (snapshot.trackIds.length === 0) return;
    commit({ rev: snapshot.rev + 1, trackIds: [] });
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
