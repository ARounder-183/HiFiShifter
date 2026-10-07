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
import { invoke } from "../../services/invoke";

/** 后端 `plugin_get_apply_state` / `plugin_apply_state` 事件的负载。 */
export interface PluginApplyState {
    generation: number;
    applied_generation: number;
    pending: boolean;
    host_version?: number;
    connected: boolean;
    ready?: boolean;
    error: string | null;
}

export interface PluginApplySnapshot {
    /** 宿主自报的状态；首次轮询回来之前是 `null`。 */
    state: PluginApplyState | null;
    /** **通信**失败的原因（与宿主自报的 `state.error` 分开：那是宿主侧的诊断）。 */
    failure: string;
    /** `plugin_refresh` 正在进行。 */
    reloading: boolean;
}

const EMPTY: PluginApplySnapshot = { state: null, failure: "", reloading: false };

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
        a.error === b.error
    );
}

function publish(next: PluginApplySnapshot): void {
    if (
        sameState(snapshot.state, next.state) &&
        snapshot.failure === next.failure &&
        snapshot.reloading === next.reloading
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
 */
export function startPluginApplyPolling(refreshTimeline: () => Promise<unknown>): () => void {
    let disposed = false;
    let inFlight = false;
    let lastHostVersion: number | undefined;
    let timelineInFlight = false;

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

    async function poll(): Promise<void> {
        if (inFlight || disposed) return;
        inFlight = true;
        try {
            const current = await invoke<PluginApplyState>("plugin_get_apply_state");
            if (!disposed) publish({ ...snapshot, state: current, failure: "" });
            if (!current.ready || current.host_version !== lastHostVersion) {
                if (await updateTimeline()) lastHostVersion = current.host_version;
            }
        } catch (error) {
            if (!disposed) publish({ ...snapshot, failure: String(error) });
        } finally {
            inFlight = false;
        }
    }

    void poll();
    const timer = window.setInterval(() => void poll(), 250);
    return () => {
        disposed = true;
        window.clearInterval(timer);
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
        publish({ ...snapshot, state, failure: "", reloading: false });
    } catch (error) {
        publish({ ...snapshot, failure: String(error), reloading: false });
    }
}
