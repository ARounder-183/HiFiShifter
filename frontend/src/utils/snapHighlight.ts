import type { SnapCandidateKind } from "./timelineSnapping";

/**
 * 吸附竖线高亮总线（轨道视图）。
 *
 * 职责：把“某次拖拽手势中当前生效的吸附”以声明式快照发布出来，由
 * TimelinePanel 内的 SnapHighlightLayer 订阅并渲染竖线高亮。
 *
 * 设计要点：
 * - 轻量 pub/sub（与 timelineSnapping 的手势登记同一风格），不走 Redux：
 *   吸附高亮是纯视觉瞬态，每帧随指针更新，进 store 会放大重渲染范围。
 * - 快照不可变：仅在 publish/clear 时替换对象引用，订阅方用
 *   useSyncExternalStore 直接比较引用即可。
 * - 分组（group）：不同交互源（主时间轴 / 框选等）各自成组，
 *   发布同组条目时整组替换；clear 可按组或全清。
 *
 * 高亮模型（对应需求“同时高亮吸附对象与被吸附对象的吸附处”）：
 * - 每个 entry 表示一条吸附关系：kind 决定颜色（网格/Clip 边缘/内容起点/
 *   源素材首尾/选择/光标/采样率/循环节），markers 描述参与对齐的竖线位置。
 * - marker.role === "target"：吸附目标（网格线、对方 Clip 的边缘、光标…）；
 *   role === "source"：被吸附对象（正在拖拽的 Clip 的对齐边、drop 预览起点…）。
 * - trackId 为 null 的 marker 参与全高竖线（网格线/光标/采样率本身即通栏）；
 *   带 trackId/clipId 的 marker 会额外在自己所在行内绘制贴边亮条。
 */

/** 主时间轴共享组：同一时刻只呈现一组吸附状态。 */
export const SNAP_HIGHLIGHT_GROUP = "timeline";

export type SnapHighlightRole = "target" | "source";

export type SnapHighlightKind =
    | "grid"
    | "clipStart"
    | "clipEnd"
    | "snapOffset"
    | "sourceStart"
    | "sourceEnd"
    | "selection"
    | "cursor"
    | "sampleRate"
    /** 循环节/源媒体边界吸附（loopSnap 体系，非 SnapCandidateKind）。 */
    | "loopBoundary";

export interface SnapHighlightMarker {
    role: SnapHighlightRole;
    /** 竖线 x 位置（时间线秒）。 */
    sec: number;
    /** 所属轨道；null 表示通栏（不限定行）。 */
    trackId: string | null;
    /** 所属 clip（用于将来更精细的行内定位）；可为 null。 */
    clipId: string | null;
}

export interface SnapHighlightEntry {
    id: string;
    kind: SnapHighlightKind;
    markers: readonly SnapHighlightMarker[];
}

export interface SnapHighlightSnapshot {
    /** 单调递增版本号：保证 publish 相同内容也能触发订阅方刷新。 */
    rev: number;
    entries: readonly SnapHighlightEntry[];
}

const EMPTY_SNAPSHOT: SnapHighlightSnapshot = { rev: 0, entries: [] };

let snapshot: SnapHighlightSnapshot = EMPTY_SNAPSHOT;
let rev = 0;
const listeners = new Set<() => void>();

/** 提交新的条目集合并通知订阅方。 */
function commit(entries: SnapHighlightEntry[]): void {
    rev += 1;
    snapshot = { rev, entries };
    for (const listener of listeners) listener();
}

/** 订阅快照变化；返回取消订阅函数。 */
export function subscribeSnapHighlight(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

/** 当前快照（引用稳定，适合 useSyncExternalStore）。 */
export function getSnapHighlightSnapshot(): SnapHighlightSnapshot {
    return snapshot;
}

/** 发布一组条目：整组替换 group 内的旧条目（entry.id 以 `${group}\0` 为前缀编码）。 */
export function publishSnapHighlights(group: string, entries: readonly SnapHighlightEntry[]): void {
    const prefix = `${group}\u0000`;
    const nextEntries = snapshot.entries.filter((entry) => !entry.id.startsWith(prefix));
    for (const entry of entries) {
        nextEntries.push({ ...entry, id: `${prefix}${entry.id}` });
    }
    commit(nextEntries);
}

/** 清除指定组的条目；省略 group 时清空全部。 */
export function clearSnapHighlights(group?: string): void {
    if (!group) {
        if (snapshot.entries.length === 0) return;
        commit([]);
        return;
    }
    const prefix = `${group}\u0000`;
    const nextEntries = snapshot.entries.filter((entry) => !entry.id.startsWith(prefix));
    if (nextEntries.length === snapshot.entries.length) return;
    commit(nextEntries);
}

// ── 构建工具 ─────────────────────────────────────────────────

export interface SnapHighlightSourceSpec {
    /** 被吸附对象的对齐边位置（秒）；缺省 = 吸附结果位置（已对齐）。 */
    sec?: number;
    /** 所属轨道；null/缺省 = 通栏。 */
    trackId?: string | null;
    /** 所属 clip；可为 null。 */
    clipId?: string | null;
}

/**
 * 由吸附候选构建一条高亮 entry：
 * - target marker 来自候选（grid/cursor/sampleRate 无 trackId → 通栏线；
 *   clip 族带 clipId/trackId → 行级亮边 + 与 source 的跨行连线）。
 * - sources 为被吸附对象侧的 marker 列表（通常一个：正在拖拽的锚边）。
 */
export function buildCandidateHighlightEntry(args: {
    id?: string;
    kind: SnapHighlightKind;
    sec: number;
    targetTrackId?: string | null;
    targetClipId?: string | null;
    sources?: readonly SnapHighlightSourceSpec[];
}): SnapHighlightEntry {
    const markers: SnapHighlightMarker[] = [
        {
            role: "target",
            sec: args.sec,
            trackId: args.targetTrackId ?? null,
            clipId: args.targetClipId ?? null,
        },
    ];
    for (const source of args.sources ?? []) {
        markers.push({
            role: "source",
            sec: source.sec ?? args.sec,
            trackId: source.trackId ?? null,
            clipId: source.clipId ?? null,
        });
    }
    return { id: args.id ?? "primary", kind: args.kind, markers };
}

/** 循环节/源媒体边界吸附的快捷构建（目标与被吸附边通常重合于同一 x）。 */
export function buildLoopBoundaryHighlightEntry(args: {
    id?: string;
    secs: readonly number[];
    trackId: string | null;
    clipId: string | null;
}): SnapHighlightEntry {
    const markers: SnapHighlightMarker[] = args.secs.map((sec, index) => ({
        role: index === 0 ? "target" : "source",
        sec,
        trackId: args.trackId,
        clipId: args.clipId,
    }));
    if (markers.length === 1) {
        markers.push({ ...markers[0], role: "source" });
    }
    return { id: args.id ?? "primary", kind: "loopBoundary", markers };
}

/** SnapCandidateKind → 高亮 kind（恒等映射，显式写出便于未来分化）。
 * 参数直接复用 SnapCandidateKind，避免手抄联合类型随源类型漂移。 */
export function snapHighlightKindFromCandidate(
    kind: SnapCandidateKind,
): Exclude<SnapHighlightKind, "loopBoundary"> {
    return kind;
}
