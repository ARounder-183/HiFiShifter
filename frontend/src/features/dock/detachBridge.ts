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

import { reportFrontendError } from "../../services/frontendErrorLog";
import { loadAppearance } from "../../theme/themeStorage";

/** 窗口角色。 */
export type BridgeRole = "main" | "satellite";

/** 动作是否来自桥（收到即应用，不再回传）。 */
const REMOTE_META_KEY = "hsRemote";

export const BRIDGE_ACTION_EVENT = "hs:store/action";
/** 卫星窗口重发快照请求的间隔与放弃时限。 */
const SNAPSHOT_REQUEST_INTERVAL_MS = 500;
const SNAPSHOT_REQUEST_TIMEOUT_MS = 10_000;
export const BRIDGE_SNAPSHOT_REQUEST_EVENT = "hs:store/snapshot-request";
export const BRIDGE_SNAPSHOT_EVENT = "hs:store/snapshot";
/** 主窗口 → 卫星窗口：外观（主题 / 字体）变更。 */
export const BRIDGE_APPEARANCE_EVENT = "hs:store/appearance";
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
    /** 主窗口当前的外观（主题 / 字体）。见 `subscribeRemoteAppearance` 的说明。 */
    appearance?: unknown;
}

/**
 * 卫星窗口最近一次收到的外观（未收到为 null）与订阅者。
 *
 * 【为什么要跨窗口传外观，而不是让卫星读 localStorage】外观（含**自定义字体**）
 * 只存在 localStorage 里，卫星窗口过去依赖"两个窗口共享同一份存储"这一环境假设：
 * 一旦存储分区不同、或窗口在写入之前就挂载，卫星就退回默认字体（用户报告"独立
 * 窗口没有继承主窗口的自定义字体"）。改为**主窗口权威 + 显式下发**之后，卫星不再
 * 依赖任何环境假设：启动时从快照里拿一次，此后由主窗口在变更时推送。
 */
let remoteAppearance: unknown = null;
const appearanceListeners = new Set<(appearance: unknown) => void>();

function publishRemoteAppearance(appearance: unknown): void {
    if (appearance == null) return;
    remoteAppearance = appearance;
    for (const listener of appearanceListeners) listener(appearance);
}

/**
 * 订阅"主窗口下发的外观"。
 *
 * 订阅时会**立即**以当前值回调一次（若已收到过）—— 这样晚挂载的组件也能拿到值，
 * 不必再区分"先到 / 后到"。
 *
 * @returns 取消订阅。
 */
export function subscribeRemoteAppearance(listener: (appearance: unknown) => void): () => void {
    if (remoteAppearance !== null) listener(remoteAppearance);
    appearanceListeners.add(listener);
    return () => appearanceListeners.delete(listener);
}

/**
 * 上报桥的失败。
 *
 * 【为什么必须可见】桥的失败表现为"卫星窗口永远空白、而面板已从主窗口移除" ——
 * 用户看到的是功能完全没实现。此前这些分支都是静默 return，排障时毫无线索。
 */
function reportBridgeFailure(reason: string): void {
    reportFrontendError(`[detachBridge] ${reason} (${resolveWindowLabel()})`);
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
    // 快照只应从主窗口"应答"给请求它的卫星（BRIDGE_SNAPSHOT_EVENT），绝不能作为
    // 普通动作进广播通道 —— 否则卫星应用快照的动作会被原样弹回主窗口并在那里
    // 整体覆盖状态（一次 detach = 主窗口状态回滚）。
    if (type === BRIDGE_SNAPSHOT_ACTION) return false;
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

/**
 * 主窗口当前是否真的存在卫星窗口（独立窗口承载的窗体）。
 *
 * 【为什么在派发路径上查】`trySerialize` 是一次完整的 JSON 往返，主窗口每一次
 * 可广播的派发都要付一遍（包括 ~30Hz 的播放轮询、逐帧拖拽乐观动作、整份
 * TimelineState 的编辑回包），而卫星不存在时（绝大多数会话的全部时间）广播
 * 通道空无一物，序列化纯属浪费。dock 布局里 `floatMode === "osWindow"` 的浮动
 * 窗体就是卫星 —— 这份状态就在本窗口的 store 里，同步可查，零新协议。
 *
 * 【查不到时按 true 处理】dock 状态缺失（测试桩 / 未知宿主）时宁可白序列化，
 * 也不能把广播吞掉 —— 桥静默断开的表现是"卫星窗口空白"，比慢更糟。
 */
function hasLiveSatellites(getState: () => unknown): boolean {
    let state: unknown;
    try {
        state = getState();
    } catch {
        return true;
    }
    if (!state || typeof state !== "object") return true;
    const dock = (state as Record<string, unknown>).dock;
    if (!dock || typeof dock !== "object") return true;
    const layout = (dock as Record<string, unknown>).layout;
    if (!layout || typeof layout !== "object") return true;
    const order = (layout as Record<string, unknown>).order;
    const forms = (layout as Record<string, unknown>).forms;
    if (!Array.isArray(order) || !forms || typeof forms !== "object") return true;
    const formMap = forms as Record<string, unknown>;
    for (const formId of order) {
        const form = formMap[formId];
        if (
            form !== null &&
            typeof form === "object" &&
            (form as Record<string, unknown>).floating === true &&
            (form as Record<string, unknown>).floatMode === "osWindow"
        ) {
            return true;
        }
    }
    return false;
}

/**
 * 快照投影：剔掉**重载荷**字段。
 *
 * 【为什么要瘦身】`session` 切片携带逐 clip 的波形数组（`waveform` /
 * `waveformPreview`）、逐 clip 的检测音高曲线（`clipPitchCurves[*].midiCurve`，
 * 每 clip 数百到上千个浮点）与静音预览段，一个真实工程的快照轻易到几 MB ——
 * 跨窗口事件传这么大的 JSON 又慢又容易被序列化问题绊倒，而独立窗口承载的只有
 * 文件浏览器 / 记事本 / 撤销历史，**不需要波形与曲线**。整份 `JSON.stringify`
 * 一旦抛错，主窗口此前会静默不应答，卫星就永久停在占位符上（用户报告的
 * "完全没实现"）。请求方每 500ms 重试一次，投影不彻底会把大快照的代价变成
 * 周期性的。
 *
 * 投影是**结构性**的（按已知字段名剔除），不做深拷贝：拷贝由随后的 stringify 完成。
 */
export function projectSnapshot(state: unknown): unknown {
    if (!state || typeof state !== "object") return state;
    const root = state as Record<string, unknown>;
    const session = root.session;
    if (!session || typeof session !== "object") return state;
    const sessionRecord = session as Record<string, unknown>;
    const clips = sessionRecord.clips;
    const projectedClips = Array.isArray(clips)
        ? clips.map((clip) => {
              if (!clip || typeof clip !== "object") return clip;
              const { waveform, waveformPreview, ...rest } = clip as Record<string, unknown>;
              void waveform;
              void waveformPreview;
              return rest;
          })
        : clips;
    const projectedSession: Record<string, unknown> = {
        ...sessionRecord,
        clips: projectedClips,
        // 检测音高曲线（每 clip 一条大数组）：卫星面板不消费，整表置空。
        clipPitchCurves: {},
        // 静音预览段：同样只有主窗口的轨道区消费。
        silencePreviewSegments: null,
    };
    if (
        sessionRecord.selectedTrackSummary &&
        typeof sessionRecord.selectedTrackSummary === "object"
    ) {
        projectedSession.selectedTrackSummary = {
            ...(sessionRecord.selectedTrackSummary as Record<string, unknown>),
            waveformPreview: null,
        };
    }
    return { ...root, session: projectedSession };
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
 * 把当前外观广播给所有卫星窗口。
 *
 * 【为什么由面板**直接调用**，而不是监听 Tauri 事件】外观设置曾经是独立 OS 窗口，
 * 它改完外观后发 `appearance-applied / -preview / -reverted` 三个事件，主窗口的桥
 * 监听它们再转发给卫星窗口。现在外观设置是主窗口里的一个停靠面板，那三个事件
 * 不复存在 —— 于是改成面板直接调用本函数，链路比"面板 → 事件 → 桥 → 卫星"更短，
 * 也少一个"事件没送到就不同步"的失败面。
 *
 * 卫星窗口（拆到独立窗口的面板）不共享主题，必须显式通知，见 `BRIDGE_APPEARANCE_EVENT`。
 */
export async function broadcastAppearanceToSatellites(): Promise<void> {
    const api = await loadEventApi();
    if (!api) return;
    const appearance = trySerialize(loadAppearance());
    if (appearance === null) return;
    await api.emit(BRIDGE_APPEARANCE_EVENT, { appearance }).catch(() => undefined);
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

    return (api) => (next) => (action) => {
        const result = next(action);
        if (!isBroadcastable(action)) return result;
        // 主窗口在没有任何卫星时跳过整条序列化 / 排队路径（见 hasLiveSatellites）。
        // 卫星侧不做此判定：它派发的动作必须送回主窗口，否则状态永久分叉。
        if (role === "main" && !hasLiveSatellites(api.getState)) return result;
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
        // 监听注册本身也可能失败（非 Tauri 环境、权限缺失）：桥是"尽力而为"的
        // 增强，绝不能因为注册失败在控制台留下未捕获的 rejection。
        try {
            await registerListeners(api);
        } catch {
            // 忽略：两侧状态各自独立演进，功能不受影响（面板仍可用）。
        }
    });

    async function registerListeners(
        api: NonNullable<Awaited<ReturnType<typeof loadEventApi>>>,
    ): Promise<void> {
        const offAction = await api.listen(BRIDGE_ACTION_EVENT, (event) => {
            const envelope = event.payload as BridgeEnvelope | null;
            if (!envelope || envelope.origin === origin) return;
            for (const action of envelope.actions ?? []) {
                dispatch({ ...(action as UnknownAction), meta: { [REMOTE_META_KEY]: true } });
            }
        });
        unsubscribers.push(offAction);

        if (role === "main") {
            /*
             * 外观广播不再在这里接线：外观设置已是主窗口内的面板，它改完外观后
             * **直接调用** `broadcastAppearanceToSatellites()`（见该函数）。
             * 这里原本监听外观窗口发出的三个 Tauri 事件，那些事件随独立窗口一起消失了。
             */

            // 主窗口应答快照请求（卫星会重试，因此这里必须**幂等且无副作用**）。
            const offRequest = await api.listen(BRIDGE_SNAPSHOT_REQUEST_EVENT, (event) => {
                const request = event.payload as SnapshotRequest | null;
                if (!request || request.origin === origin) return;
                const state = trySerialize(projectSnapshot(getState()));
                if (state === null) {
                    // 序列化失败必须**可见**：此前静默 return，卫星永久停在占位符上，
                    // 而面板已从主窗口移除 —— 用户看到的是"窗口空白、面板也没了"。
                    reportBridgeFailure("snapshot-serialize-failed");
                    return;
                }
                void api.emit(BRIDGE_SNAPSHOT_EVENT, {
                    target: request.origin,
                    state,
                    // 外观随快照一起下发：卫星启动即拿到正确字体（见
                    // `subscribeRemoteAppearance`）。
                    appearance: trySerialize(loadAppearance()) ?? undefined,
                } satisfies SnapshotEnvelope);
            });
            unsubscribers.push(offRequest);
        } else {
            // 卫星窗口：监听快照应答。
            const offSnapshot = await api.listen(BRIDGE_SNAPSHOT_EVENT, (event) => {
                const envelope = event.payload as SnapshotEnvelope | null;
                if (!envelope || envelope.target !== origin) return;
                received = true;
                stopRequesting();
                publishRemoteAppearance(envelope.appearance ?? null);
                onSnapshot?.(envelope.state);
            });
            unsubscribers.push(offSnapshot);
            const offAppearance = await api.listen(BRIDGE_APPEARANCE_EVENT, (event) => {
                const payload = event.payload as { appearance?: unknown } | null;
                publishRemoteAppearance(payload?.appearance ?? null);
            });
            unsubscribers.push(offAppearance);
            // 【必须重试】请求是"尽力而为"的事件：主窗口的监听可能还没注册好（卫星
            // 在 store 模块求值期就发起请求，比主窗口的 App 挂载还早），事件会丢进
            // 空房间且**永远不会重发**。此前只发一次 ⇒ 卫星永久空白。这里按固定间隔
            // 重发直到收到快照（上限 `SNAPSHOT_REQUEST_TIMEOUT_MS`）。
            startRequesting(api);
        }
    }

    /** 是否已收到快照（收到后停止重发）。 */
    let received = false;
    let requestTimer: ReturnType<typeof setInterval> | null = null;

    function stopRequesting(): void {
        if (requestTimer !== null) {
            clearInterval(requestTimer);
            requestTimer = null;
        }
    }

    function startRequesting(api: NonNullable<Awaited<ReturnType<typeof loadEventApi>>>): void {
        const request = () => {
            if (disposed || received) {
                stopRequesting();
                return;
            }
            void api
                .emit(BRIDGE_SNAPSHOT_REQUEST_EVENT, { origin } satisfies SnapshotRequest)
                .catch(() => {
                    // 单次发送失败不影响重试。
                });
        };
        request();
        requestTimer = setInterval(request, SNAPSHOT_REQUEST_INTERVAL_MS);
        setTimeout(() => {
            if (received || disposed) return;
            stopRequesting();
            reportBridgeFailure("snapshot-timeout");
        }, SNAPSHOT_REQUEST_TIMEOUT_MS);
    }

    return () => {
        disposed = true;
        stopRequesting();
        for (const off of unsubscribers) off();
        unsubscribers.length = 0;
    };
}

/** 应用远端快照的动作类型（各窗口的 reducer 不认识它，由 `installBridge` 的
 *  调用方在 store 层特判；这里只提供常量以免拼写漂移）。 */
export const BRIDGE_SNAPSHOT_ACTION = "hs/bridgeSnapshot";
