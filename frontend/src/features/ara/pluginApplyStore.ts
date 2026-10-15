/**
 * 插件宿主「自动应用」状态的模块级 store。
 *
 * 【为什么不是组件里的 useState】这段状态显示在**状态栏**里，而状态栏属于
 * `AppInner` —— 把 250 ms 轮询的 setState 放进去，等于每 250 ms 重渲染整棵
 * 应用树（时间轴、参数编辑器、全部面板）。抽成外部 store 后，重渲染被限制在
 * 订阅它的那一个状态片上，与 `ParamDataLoadingChip` / `AppStatusProgressChips`
 * 是同一取舍。
 *
 * 【为什么状态是"快照对象"而不是散装 getter】`useSyncExternalStore` 靠引用相等
 * 判断"变了没有"，因此 `publish` 会在内容相同时**保留旧引用** —— 250 ms 一次的
 * 轮询里绝大多数回调其实什么都没变，不比较就等于每 250 ms 白重渲染一次。
 */
import { listen } from "../../services/hostEvents";
import { invoke, invokeErrorCode, invokeErrorKind } from "../../services/invoke";

/** 后端 `plugin_get_apply_state` / `plugin_apply_state` 事件的负载。 */
export interface PluginApplyState {
    generation: number;
    applied_generation: number;
    pending: boolean;
    host_version?: number;
    connected: boolean;
    ready?: boolean;
    error: string | null;
    /**
     * 后端是否真的在跑一次渲染（派生状态，不是事件）。
     *
     * 【为什么需要它】"渲染中"此前完全由 `playback_rendering_state` 事件流驱动，
     * 而该事件用 `try_send`（队列满即丢）—— 丢一个 `active:false` 就会让状态栏片
     * **永久亮着**。轮询快照每 250 ms 重读一次，是这条 UI 状态的自愈来源。
     */
    rendering_active?: boolean;
    /**
     * 本实例的运行形态（`"transfer"` / `"ara"`）。
     *
     * 【为什么不能只读 bootstrap 的 `hostKind`】bootstrap 是窗口创建那一刻算的，而形态
     * 判定要等 `setActive` 之后 5 秒宽限期才落定 —— 那时读到的永远是 `ara_only`。
     * 状态快照每 250ms 轮询一次，是不依赖时序的那条通道。
     */
    mode?: "transfer" | "ara";
}

export interface PluginApplySnapshot {
    /** 宿主自报的状态；首次轮询回来之前是 `null`。 */
    state: PluginApplyState | null;
    /** **通信**失败的原因（与宿主自报的 `state.error` 分开：那是宿主侧的诊断）。 */
    failure: string;
    /** `plugin_refresh` 正在进行。 */
    reloading: boolean;
    /**
     * 编辑会话已**永久**结束（后端回执分类为 `terminal`）。
     *
     * 【为什么需要它】后端在文档/组件关闭后会把同一个终态错误答给每一次轮询 ——
     * 实测在 Cubase 上一个终态错误被答了 19,056 次、日志轮转文件合计约 32 MB。
     * 轮询必须据此**停止**，而不是无限重试。
     */
    sessionEnded: boolean;
    /**
     * 终态的**协议码**（`services/sessionGate.ts` 的 `ERROR_CODE`）；未终结时为 `null`。
     *
     * 【为什么需要它】`sessionEnded` 只说"永久结束了"，说不出是**哪一种**结束：
     * "编辑文档已关闭"关窗重开有用，"宿主从未把本插件接入 ARA"关窗重开无用。状态栏
     * 要按码给不同的指引，而按错误文本分支是被明令禁止的（见 `pluginHost.ts`）。
     */
    sessionCode: number | null;
}

const EMPTY: PluginApplySnapshot = {
    state: null,
    failure: "",
    reloading: false,
    sessionEnded: false,
    sessionCode: null,
};

/** 正常轮询间隔（与旧实现的 250 ms 一致：状态栏要跟得上宿主的应用进度）。 */
const BASE_POLL_MS = 250;
/** 暂时性失败后的退避上限。 */
const MAX_POLL_MS = 30_000;

let snapshot: PluginApplySnapshot = EMPTY;
const listeners = new Set<() => void>();

function sameState(a: PluginApplyState | null, b: PluginApplyState | null): boolean {
    if (a === b) return true;
    if (!a || !b) return false;
    return (
        a.generation === b.generation &&
        a.applied_generation === b.applied_generation &&
        a.pending === b.pending &&
        a.host_version === b.host_version &&
        a.connected === b.connected &&
        a.ready === b.ready &&
        a.rendering_active === b.rendering_active &&
        a.error === b.error
    );
}

function publish(next: PluginApplySnapshot): void {
    if (
        sameState(snapshot.state, next.state) &&
        snapshot.failure === next.failure &&
        snapshot.reloading === next.reloading &&
        snapshot.sessionEnded === next.sessionEnded &&
        snapshot.sessionCode === next.sessionCode
    ) {
        // 内容没变就保留旧引用：订阅者不该因为"又轮询了一次"而重渲染。
        return;
    }
    snapshot = next;
    for (const listener of listeners) listener();
}

export function subscribePluginApply(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getPluginApplySnapshot(): PluginApplySnapshot {
    return snapshot;
}

/** 仅供测试重置模块级状态。 */
export function resetPluginApplyStoreForTests(): void {
    snapshot = EMPTY;
    listeners.clear();
}

/**
 * 开始轮询宿主状态；返回停止函数。
 *
 * `refreshTimeline` 由调用方注入（store 不持有 dispatch）：宿主版本前进或
 * 尚未就绪时，本地时间轴需要重取一次。
 *
 * 【保留的两处既有语义，别顺手简化】
 * - `timelineInFlight` 重入保护：宿主版本连续前进时不能并发重取时间轴；
 * - 重取失败**不推进** `lastHostVersion`，于是下一次轮询会重试。
 *
 * 【为什么失败要退避、终态要停止】原实现每 250 ms 无条件重试，且不区分失败种类。
 * 当后端已把会话终结（文档/组件关闭）时，这就是**永远**失败的 4 次/秒 —— 实测把
 * 一个终态错误放大成 19,056 条日志。现在：终态**停止轮询**，暂时性失败指数退避
 * （250ms → 2s → 8s，上限 30s），成功一次即复位。
 */
export function startPluginApplyPolling(refreshTimeline: () => Promise<unknown>): () => void {
    let disposed = false;
    let inFlight = false;
    let lastHostVersion: number | undefined;
    let timelineInFlight = false;
    let timer: number | undefined;
    let backoffMs = BASE_POLL_MS;

    async function updateTimeline(): Promise<boolean> {
        if (disposed || timelineInFlight) return false;
        timelineInFlight = true;
        try {
            await refreshTimeline();
            return true;
        } catch {
            return false;
        } finally {
            timelineInFlight = false;
        }
    }

    const hostSubscription = listen("plugin_host_changed", () => {
        void updateTimeline();
    });
    const stateSubscription = listen<PluginApplyState>("plugin_apply_state", (event) => {
        if (!disposed) publish({ ...snapshot, state: event.payload });
    });

    /** 安排下一次轮询。`delayMs` 为 undefined 表示"停止"（终态或已 dispose）。 */
    function schedule(delayMs: number | undefined): void {
        if (disposed || delayMs === undefined) return;
        timer = window.setTimeout(() => void poll(), delayMs);
    }

    async function poll(): Promise<void> {
        if (inFlight || disposed) return;
        inFlight = true;
        try {
            const current = await invoke<PluginApplyState>("plugin_get_apply_state");
            if (disposed) return;
            backoffMs = BASE_POLL_MS;
            publish({
                ...snapshot,
                state: current,
                failure: "",
                sessionEnded: false,
                sessionCode: null,
            });
            if (!current.ready || current.host_version !== lastHostVersion) {
                if (await updateTimeline()) lastHostVersion = current.host_version;
            }
            schedule(BASE_POLL_MS);
        } catch (error) {
            if (disposed) return;
            const kind = invokeErrorKind(error);
            if (kind === "terminal") {
                // 会话已永久结束：停止轮询，如实显示，不再制造 IPC 与日志。
                publish({
                    ...snapshot,
                    failure: String(error),
                    sessionEnded: true,
                    sessionCode: invokeErrorCode(error),
                });
                schedule(undefined);
                return;
            }
            publish({ ...snapshot, failure: String(error) });
            schedule(backoffMs);
            backoffMs = Math.min(backoffMs * 4, MAX_POLL_MS);
        } finally {
            inFlight = false;
        }
    }

    void poll();
    return () => {
        disposed = true;
        if (timer !== undefined) window.clearTimeout(timer);
        void stateSubscription.then((off) => off()).catch(() => undefined);
        void hostSubscription.then((off) => off()).catch(() => undefined);
    };
}

/**
 * 重新载入宿主。
 *
 * `force = false` 时**不**在这里做确认：是否需要确认取决于宿主的 `pending`
 * 状态，那是 UI 的判断（见 `PluginApplyStatus`）。
 */
export async function reloadPluginHost(
    force: boolean,
    refreshTimeline: () => Promise<unknown>,
): Promise<void> {
    publish({ ...snapshot, reloading: true });
    try {
        await invoke("plugin_refresh", force);
        await refreshTimeline();
        const state = await invoke<PluginApplyState>("plugin_get_apply_state");
        publish({
            ...snapshot,
            state,
            failure: "",
            reloading: false,
            sessionEnded: false,
            sessionCode: null,
        });
    } catch (error) {
        // 终态失败同样要如实记下：用户主动点"重新载入"却撞上会话已结束，
        // 界面必须说明原因，而不是只把按钮转一圈又回到原样。
        publish({
            ...snapshot,
            failure: String(error),
            reloading: false,
            sessionEnded: invokeErrorKind(error) === "terminal",
            sessionCode: invokeErrorCode(error),
        });
    }
}
