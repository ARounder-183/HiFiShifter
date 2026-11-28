// 编辑链路的结构化埋点：低频、有界、可丢弃。
//
// 【为什么需要它】用户报「撤销/分割在手机上不可用」，而诊断包里 `split` /
// `undo` / `redo` 的命中数是 **0** —— 不是没发生，而是**没有任何一处代码
// 在记录它们**。于是「用户说按了十次撤销」与「后端说只收到一次」之间没有任何
// 可以对齐的证据，只能靠猜。
//
// 【为什么不在 reducer 里直接写日志】拖动期间 reducer 每帧都跑，直接调
// `frontendLog` 会把去重表、策略矩阵、IPC 序列化全部拖进热路径。这里只做一次
// 数组 push（O(1)、无分配除对象本身），由 UI 侧的低频 flush 统一出口。
//
// 【为什么要限制容量】埋点是为了诊断，不是为了留存。缓冲区满就丢最旧的 ——
// 一次手势刷出上千条 `gesture.frame` 只会把真正有用的那几条挤掉。

export type EditTelemetryLevel = "debug" | "warn";

export interface EditTelemetryEvent {
    /** 单调递增序号（跨 flush 连续，便于在日志里排序）。 */
    seq: number;
    at: number;
    level: EditTelemetryLevel;
    kind: string;
    fields: Record<string, unknown>;
}

const MAX_BUFFER = 96;

let nextSeq = 1;
let buffer: EditTelemetryEvent[] = [];
let droppedSinceFlush = 0;

/** 记录一条埋点。缓冲区满时丢最旧的（并记一次丢弃计数）。 */
export function emitEditTelemetry(
    level: EditTelemetryLevel,
    kind: string,
    fields: Record<string, unknown> = {},
): void {
    if (buffer.length >= MAX_BUFFER) {
        buffer.shift();
        droppedSinceFlush += 1;
    }
    buffer.push({ seq: nextSeq++, at: Date.now(), level, kind, fields });
}

/** 取出并清空缓冲区（flush 侧调用）。 */
export function drainEditTelemetry(): { events: EditTelemetryEvent[]; dropped: number } {
    const events = buffer;
    const dropped = droppedSinceFlush;
    buffer = [];
    droppedSinceFlush = 0;
    return { events, dropped };
}

/** 仅测试用。 */
export function resetEditTelemetryForTests(): void {
    buffer = [];
    droppedSinceFlush = 0;
    nextSeq = 1;
}

/** 把一条事件压成单行文本（与 `frontendLog` 的单行化口径一致）。 */
export function formatEditTelemetryEvent(event: EditTelemetryEvent): string {
    const parts: string[] = [`#${event.seq}`, event.kind];
    for (const [key, value] of Object.entries(event.fields)) {
        if (value === undefined) continue;
        if (typeof value === "number") {
            parts.push(`${key}=${Number.isInteger(value) ? value : value.toFixed(2)}`);
        } else if (typeof value === "boolean") {
            parts.push(`${key}=${value ? 1 : 0}`);
        } else if (Array.isArray(value)) {
            parts.push(`${key}=[${value.length}]`);
        } else {
            const text = String(value);
            parts.push(`${key}=${text.length > 48 ? `${text.slice(0, 45)}...` : text}`);
        }
    }
    return parts.join(" ");
}
