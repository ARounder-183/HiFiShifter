// 手势会话：把「用户正按着手指」这件事变成 Redux 里可被 reducer 看见的状态。
//
// 【为什么必须进 store】回包乱序仲裁发生在 reducer 里（`applyTimelineState`），
// 而「现在是不是正拖着一个 Clip」这件事此前只存在于 React 组件的 ref 里。
// 于是 reducer 只能靠 `_interactionLockCount` 猜，而那个计数器在参数编辑器、
// 在触摸平移、在「按下但还没位移」的窗口里都是 0 —— 迟到的旧快照照样能盖掉
// 正在进行的编辑，用户看到的就是「拖着拖着弹回去了」。
//
// 本模块是**纯逻辑**（不 import store / 不 import React），因此三条规则可以被
// 单测钉死：
//   1. 深度 0 → 1 时记录基线版本号；
//   2. 会话期间到达的快照**不落地**，只保留最新一份（按 revision 比较）；
//   3. 深度 1 → 0 时把那份交回去，由调用方决定是否落地。
//
// 【为什么是「推迟」而不是「过滤冲突实体」】时间线的波纹编辑没有局部性：
// 右缘一拖，后续所有 Clip 全部平移。按实体做 diff 既算不对也说不清，
// 而「整份推迟到手势结束」是唯一既简单又正确的做法 —— 而且手势结束时后端
// 必然还会再回一份提交后的快照，被推迟的那份通常已经被它取代。

import type { TimelineState } from "../../types/api";

/** 手势种类。只用于日志与诊断，不参与仲裁判定。 */
export type GestureKind =
    | "clip-move"
    | "clip-trim"
    | "fade"
    | "param-paint"
    | "param-block"
    | "param-selection"
    | "seek"
    | "pan"
    | "interaction";

/** 被推迟的快照：版本号 + 载荷本体。 */
export interface DeferredSnapshot {
    /** 载荷自带的版本号；旧后端不带时为 `null`。 */
    revision: number | null;
    payload: TimelineState;
}

/** 手势会话需要的状态切片（结构化子集，便于用普通对象单测）。 */
export interface GestureSessionState {
    _gestureDepth: number;
    _gestureKind: GestureKind | null;
    /** 会话开始时的 `timelineRevision`（诊断用，不参与仲裁）。 */
    _gestureBaseRevision: number;
    _deferredSnapshot: DeferredSnapshot | null;
}

/** 读出载荷版本号；缺失 / 非法一律视为「旧后端」。 */
export function readPayloadRevision(payload: unknown): number | null {
    const raw = (payload as { revision?: unknown } | null | undefined)?.revision;
    if (typeof raw === "number" && Number.isFinite(raw)) return raw;
    // 后端用 u64：超过 2^53 会以字符串形式到达（serde_json 的默认行为）。
    // 这里不需要精确值，只需要单调比较，所以按 Number 解析即可。
    if (typeof raw === "string" && raw.length > 0) {
        const parsed = Number(raw);
        if (Number.isFinite(parsed)) return parsed;
    }
    return null;
}

/** 是否处于手势会话中。 */
export function isGestureActive(state: Pick<GestureSessionState, "_gestureDepth">): boolean {
    return (Number(state._gestureDepth) || 0) > 0;
}

/**
 * 开始一次手势（可嵌套）。
 *
 * @param currentRevision 调用瞬间的 `timelineRevision`，仅用于诊断。
 */
export function beginGestureSession(
    state: GestureSessionState,
    input: { kind?: GestureKind } | undefined,
    currentRevision: number,
): void {
    const depth = Math.max(0, Number(state._gestureDepth) || 0);
    state._gestureDepth = depth + 1;
    if (depth === 0) {
        state._gestureKind = input?.kind ?? "interaction";
        state._gestureBaseRevision = Number(currentRevision) || 0;
    } else if (input?.kind) {
        // 嵌套手势（例如拖拽中途又按下一根手指）：保留**最早**的种类，
        // 它才是这次连续交互的语义归属。
        state._gestureKind = state._gestureKind ?? input.kind;
    }
}

/**
 * 结束一次手势（可嵌套）。
 *
 * @returns 深度归零时应当落地的快照（可能为 `null`）。
 */
export function endGestureSession(state: GestureSessionState): DeferredSnapshot | null {
    const depth = Math.max(0, Number(state._gestureDepth) || 0);
    if (depth === 0) {
        // 没有开会就收会（丢包重放 / 异常路径）：无害，但要清掉可能的残留，
        // 否则一个永远落不了地的快照会一直挂在 store 里。
        state._deferredSnapshot = null;
        return null;
    }
    state._gestureDepth = depth - 1;
    if (state._gestureDepth > 0) return null;
    state._gestureKind = null;
    const deferred = state._deferredSnapshot;
    state._deferredSnapshot = null;
    return deferred;
}

/**
 * 会话期间收到一份快照：只保留「最新」的一份。
 *
 * @returns 是否真的替换掉了原有的推迟项（供埋点统计）。
 */
export function deferSnapshot(state: GestureSessionState, next: DeferredSnapshot): boolean {
    const current = state._deferredSnapshot;
    if (current) {
        if (current.revision != null) {
            // 手上那份带版本号：只有**严格更新**的才替换。
            // 一份没有版本号的载荷（旧后端 / 合成载荷）绝不能把它挤掉 ——
            // 「缺版本」不等于「版本 0」，那会让旧快照赢过新快照。
            if (next.revision == null) return false;
            if (next.revision <= current.revision) return false;
        }
        // 手上那份没有版本号时无法比较：保留**后到**的那份。
        // 这与旧行为（后者覆盖前者）一致，不会更差。
    }
    state._deferredSnapshot = next;
    return true;
}

/**
 * 判断一份带版本号的快照是否比手上那份**更旧**（应当丢弃）。
 *
 * 无版本号时返回 `false` —— 交给调用方走兼容分支。
 */
export function isStaleSnapshot(revision: number | null, currentRevision: number): boolean {
    if (revision == null) return false;
    const current = Number(currentRevision);
    if (!Number.isFinite(current)) return false;
    return revision < current;
}
