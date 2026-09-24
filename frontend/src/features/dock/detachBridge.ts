/**
 * 跨窗口状态桥 —— 让"被拆到独立窗口的面板"与主窗口共享同一份状态。
 *
 * ## 为什么需要它
 *
 * 一个 webview 就是一个 JS 上下文：模块级单例（Redux store、面板渲染器注册表、
 * 面板宿主表）在第二个窗口里都是**另一份**。面板因此只能在两处之一渲染：要么在
 * 主窗口（停靠或进程内浮动），要么在独立窗口 —— 绝不会同时在两处。这条性质让设计
 * 大幅简化：**副作用（IPC / 后端调用）永远由面板所在的那个窗口执行**，不存在两边
 * 各跑一次的问题；需要跨窗口共享的只有**状态**。
 *
 * ## 设计
 *
 * - 主窗口是**权威**：它持有真实 store，负责快照；
 * - 卫星窗口用自己的 store（同一套 reducer）启动，先取一次快照，此后靠**动作复制**
 *   保持一致：两边都把自己派发的动作广播出去，收到的一律以 `meta.hsRemote` 标记
 *   应用，中间件据此不再回传（防止回环）。
 * - 广播按动画帧合并：一次帧内的多个动作打包成一个数组发出，避免高频路径（播放
 *   轮询）把 IPC 打满。
 * - 不可序列化的动作**跳过广播**（只在本地生效）：宁可让另一侧短暂落后，也不能
 *   让桥抛错把派发路径打断。
 *
 * 特殊说明：快照与广播都走 Tauri 事件（`@tauri-apps/api/event`）。非 Tauri 环境
 * （单元测试、纯浏览器预览）下所有入口都退化为空操作。
 */

import type { Middleware, UnknownAction } from "@reduxjs/toolkit";

/** 窗口角色。 */
export type BridgeRole = "main" | "satellite";

/** 动作是否来自桥（收到即应用，不再回传）。 */
const REMOTE_META_KEY = "hsRemote";

export const BRIDGE_ACTION_EVENT = "hs:store/action";
export const BRIDGE_SNAPSHOT_REQUEST_EVENT = "hs:store/snapshot-request";
export const BRIDGE_SNAPSHOT_EVENT = "hs:store/snapshot";

interface BridgeEnvelope {
    /** 发送方窗口标签（用于忽略自己发出的事件）。 */
    origin: string;
    /** 打包的动作（按派发顺序）。 */
    actions: unknown[];
}

interface SnapshotRequest {
    origin: string;
}

interface SnapshotEnvelope {
    /** 目标卫星窗口标签。 */
    target: string;
    state: unknown;
}

/** 本窗口的稳定标识：主窗口固定为 `main`，卫星窗口为 `detached:<formId>`。 */
export function resolveWindowLabel(): string {
    if (typeof window === "undefined") return "main";
    const params = new URLSearchParams(window.location.search);
    const formId = params.get("hsDetachedForm");
    return formId ? `detached:${formId}` : "main";
}

/** 是否运行在卫星（独立窗口）上下文。 */
export function isSatelliteWindow(): boolean {
    return resolveWindowLabel() !== "main";
}

/** 当前卫星窗口承载的窗体 id（非卫星窗口为 null）。 */
export function satelliteFormId(): string | null {
    if (typeof window === "undefined") return null;
    return new URLSearchParams(window.location.search).get("hsDetachedForm");
}

/** 动作是否应被复制到其它窗口。 */
function isBroadcastable(action: unknown): action is UnknownAction {
    if (action == null || typeof action !== "object") return false;
    const type = (action as { type?: unknown }).type;
    if (typeof type !== "string" || type.length === 0) return false;
    // 桥自己发出的"应用远端动作"不再回传（否则两窗口互相转发形成回环）。
    if ((action as { meta?: Record<string, unknown> }).meta?.[REMOTE_META_KEY] === true) {
        return false;
    }
    return true;
}

/** 尝试序列化为可跨窗口传输的纯数据；失败（循环引用 / 函数载荷）时返回 null。 */
function trySerialize(value: unknown): unknown | null {
    try {
        return JSON.parse(JSON.stringify(value));
    } catch {
        return null;
    }
}

/** 动态导入 Tauri 事件 API（非 Tauri 环境返回 null）。 */
async function loadEventApi(): Promise<{
    emit: (event: string, payload?: unknown) => Promise<void>;
    listen: (event: string, handler: (event: { payload: unknown }) => void) => Promise<() => void>;
} | null> {
    try {
        const mod = await import("@tauri-apps/api/event");
        return { emit: mod.emit, listen: mod.listen };
    } catch {
        return null;
    }
}

/**
 * 动作复制中间件。
 *
 * @param role 本窗口角色（目前两侧行为一致：都广播自己派发的动作；差异体现在
 *   快照的请求/应答由 `installBridge` 按角色接线）。
 */
export function createStoreBridgeMiddleware(role: BridgeRole): Middleware {
    const origin = role === "main" ? "main" : resolveWindowLabel();
    let pending: UnknownAction[] = [];
    let flushHandle: number | null = null;
    let send: ((event: string, payload?: unknown) => Promise<void>) | null = null;
    let ready = false;

    const flush = () => {
        flushHandle = null;
        if (!ready) return; // 事件 API 未就绪：动作留在队列里，就绪后统一补发
        const actions = pending;
        pending = [];
        if (actions.length === 0 || send === null) return;
        void send(BRIDGE_ACTION_EVENT, { origin, actions } satisfies BridgeEnvelope).catch(() => {
            // 广播失败只影响另一侧的实时性，不影响本窗口；不抛给派发路径。
        });
    };

    // 事件 API 异步就绪。就绪前派发的动作**不丢**：它们留在队列里，就绪时补发
    // （否则启动瞬间的状态变更永远到不了另一侧，而独立窗口恰好是启动后马上创建的）。
    void loadEventApi().then((api) => {
        send = api ? api.emit : null;
        ready = true;
        if (pending.length > 0) flush();
    });

    return () => (next) => (action) => {
        const result = next(action);
        if (!isBroadcastable(action)) return result;
        const serialized = trySerialize(action);
        if (serialized === null) return result;
        pending.push(serialized as UnknownAction);
        if (flushHandle === null && typeof requestAnimationFrame === "function") {
            flushHandle = requestAnimationFrame(flush);
        } else if (flushHandle === null) {
            flushHandle = setTimeout(flush, 16) as unknown as number;
        }
        return result;
    };
}

/**
 * 接线桥：注册监听、按角色处理快照请求/应答。
 *
 * @param args.getState 取本窗口状态（主窗口应答快照用）。
 * @param args.dispatch 派发（应用远端动作 / 应用快照用）。
 * @param args.onSnapshot 卫星窗口收到快照时的回调（先 `replaceState` 再解除遮罩）。
 * @returns 拆除函数（窗口卸载时调用）。
 */
export function installBridge(args: {
    role: BridgeRole;
    getState: () => unknown;
    dispatch: (action: unknown) => void;
    onSnapshot?: (state: unknown) => void;
}): () => void {
    const { role, getState, dispatch, onSnapshot } = args;
    const origin = role === "main" ? "main" : resolveWindowLabel();
    let disposed = false;
    const unsubscribers: Array<() => void> = [];

    void loadEventApi().then(async (api) => {
        if (api === null || disposed) return;
        const offAction = await api.listen(BRIDGE_ACTION_EVENT, (event) => {
            const envelope = event.payload as BridgeEnvelope | null;
            if (!envelope || envelope.origin === origin) return;
            for (const action of envelope.actions ?? []) {
                dispatch({ ...(action as UnknownAction), meta: { [REMOTE_META_KEY]: true } });
            }
        });
        unsubscribers.push(offAction);

        if (role === "main") {
            // 主窗口应答快照请求：把整份状态发给请求方。
            const offRequest = await api.listen(BRIDGE_SNAPSHOT_REQUEST_EVENT, (event) => {
                const request = event.payload as SnapshotRequest | null;
                if (!request || request.origin === origin) return;
                const state = trySerialize(getState());
                if (state === null) return; // 状态不可序列化时不应答（卫星保持遮罩）
                void api.emit(BRIDGE_SNAPSHOT_EVENT, {
                    target: request.origin,
                    state,
                } satisfies SnapshotEnvelope);
            });
            unsubscribers.push(offRequest);
        } else {
            // 卫星窗口：监听快照应答。
            const offSnapshot = await api.listen(BRIDGE_SNAPSHOT_EVENT, (event) => {
                const envelope = event.payload as SnapshotEnvelope | null;
                if (!envelope || envelope.target !== origin) return;
                onSnapshot?.(envelope.state);
            });
            unsubscribers.push(offSnapshot);
            // 请求快照。
            void api.emit(BRIDGE_SNAPSHOT_REQUEST_EVENT, { origin } satisfies SnapshotRequest);
        }
    });

    return () => {
        disposed = true;
        for (const off of unsubscribers) off();
        unsubscribers.length = 0;
    };
}

/** 应用远端快照的动作类型（各窗口的 reducer 不认识它，由 `installBridge` 的
 *  调用方在 store 层特判；这里只提供常量以免拼写漂移）。 */
export const BRIDGE_SNAPSHOT_ACTION = "hs/bridgeSnapshot";
