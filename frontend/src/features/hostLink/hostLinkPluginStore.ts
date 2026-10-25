/**
 * 插件侧"导出到 HiFiShifter"的状态拉取点。
 *
 * 【为什么是轮询】导出在插件后端是**异步**的（要写音频，可能上百 MB），命令分发与
 * actor 循环是同一个线程；同步写完会让 UI 在整个导出期间无响应。因此 `host_link_export`
 * 立刻返回，真正的进度与结果由 `host_link_status` 轮询。
 *
 * 【为什么是模块级 store】控件挂在状态栏上，而状态栏属于 `AppInner`；把 250ms 级的
 * 轮询 setState 放进去，等于每轮拖着重渲染整棵应用树（与 `pluginApplyStore` 同一取舍）。
 *
 * 【为什么终态要停止轮询】会话终结后后端会把同一个错误答给每一次轮询；实测一个终态
 * 错误被答了上万次、把日志刷满（见 `pluginApplyStore`）。终态据此**停止**。
 */
import { useSyncExternalStore } from "react";

import { invokeErrorKind } from "../../services/invoke";
import { hostLinkPluginApi, type HostLinkExportStatus } from "./hostLinkApi";

/** 轮询间隔：导出是秒级的事，但回写回答可以随后到来，所以保持一个温和的常驻轮询。 */
const POLL_MS = 750;
/** 暂时性失败后的退避上限。 */
const MAX_POLL_MS = 15_000;

export interface HostLinkPluginSnapshot {
    /** 宿主自报的状态；首次轮询回来之前是 `null`。 */
    status: HostLinkExportStatus | null;
    /** **通信**失败的原因（与宿主自报的 `status.error` 分开：那是宿主侧的诊断）。 */
    failure: string;
    /** `host_link_export` 正在进行（按钮据此禁用）。 */
    busy: boolean;
    /** 会话已**永久**结束（后端回执分类为 `terminal`）—— 停止轮询。 */
    ended: boolean;
}

const EMPTY: HostLinkPluginSnapshot = {
    status: null,
    failure: "",
    busy: false,
    ended: false,
};

let snapshot: HostLinkPluginSnapshot = EMPTY;
const listeners = new Set<() => void>();

function publish(next: HostLinkPluginSnapshot): void {
    snapshot = next;
    for (const listener of listeners) listener();
}

export function subscribeHostLinkPlugin(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

export function getHostLinkPluginSnapshot(): HostLinkPluginSnapshot {
    return snapshot;
}

export function useHostLinkPluginSnapshot(): HostLinkPluginSnapshot {
    return useSyncExternalStore(
        subscribeHostLinkPlugin,
        getHostLinkPluginSnapshot,
        getHostLinkPluginSnapshot,
    );
}

/** 仅供测试重置模块级状态。 */
export function resetHostLinkPluginStoreForTests(): void {
    if (timer !== undefined) {
        window.clearTimeout(timer);
        timer = undefined;
    }
    running = 0;
    backoffMs = POLL_MS;
    inFlight = false;
    snapshot = EMPTY;
    listeners.clear();
}

/**
 * 开始轮询；返回停止函数（多次调用只保留一条轮询，引用计数）。
 *
 * 【为什么用引用计数】控件与详情对话框各自在挂载时开始、卸载时停止；两者同时在场时
 * 只有一条轮询在跑，其中一个卸载也不会让另一个断流（与 `captureStore` 同一取舍）。
 */
export function startHostLinkPluginPolling(): () => void {
    running += 1;
    schedule(POLL_MS);
    let stopped = false;
    return () => {
        if (stopped) return;
        stopped = true;
        running -= 1;
        if (running <= 0) {
            if (timer !== undefined) {
                window.clearTimeout(timer);
                timer = undefined;
            }
            running = 0;
        }
    };
}

let running = 0;
let timer: number | undefined;
let backoffMs = POLL_MS;
/** 一轮请求在途时不再发下一轮（终态早退的 catch 分支由 finally 复位）。 */
let inFlight = false;

/**
 * 安排下一次轮询。
 *
 * 【为什么是 setTimeout 链而不是 setInterval】退避要求**间隔随失败时长变化**，
 * 而 setInterval 的间隔一经创建就固定 —— 旧实现里 `backoffMs` 只写不读，
 * 暂时性失败后仍以 750ms 固定频率打后端（与 `pluginApplyStore` 对照，后者
 * 正是为此改成的链式调度）。链式还有两个附带收益：上一次请求未返回时不会
 * 叠加下一轮；`ended` 后不再排程（旧实现的 interval 在终态后仍每 750ms 空转）。
 */
function schedule(delayMs: number): void {
    if (snapshot.ended) return;
    if (timer !== undefined) window.clearTimeout(timer);
    timer = window.setTimeout(() => void refresh(), delayMs);
}

async function refresh(): Promise<void> {
    timer = undefined;
    if (running <= 0 || snapshot.ended || inFlight) return;
    inFlight = true;
    try {
        const status = await hostLinkPluginApi.status();
        backoffMs = POLL_MS;
        publish({ ...snapshot, status, failure: "" });
    } catch (error) {
        if (invokeErrorKind(error) === "terminal") {
            // 会话已永久结束：停止轮询，如实显示，不再制造 IPC 与日志。
            publish({ ...snapshot, failure: String(error), ended: true });
            return;
        }
        publish({ ...snapshot, failure: String(error) });
        backoffMs = Math.min(backoffMs * 4, MAX_POLL_MS);
    } finally {
        inFlight = false;
    }
    if (running > 0) {
        schedule(backoffMs);
    }
}

/** 开始一次导出（后台）。返回值只是启动回执，真正的结果由轮询给出。 */
export async function exportHostLinkWorkspace(): Promise<void> {
    publish({ ...snapshot, busy: true });
    try {
        const status = await hostLinkPluginApi.export();
        publish({ ...snapshot, status, failure: "", busy: false });
    } catch (error) {
        publish({
            ...snapshot,
            failure: String(error),
            busy: false,
            ended: invokeErrorKind(error) === "terminal",
        });
    }
}

/** 停止监视（用户点了「断开」）。 */
export async function releaseHostLinkWorkspace(): Promise<void> {
    try {
        const status = await hostLinkPluginApi.release();
        publish({ ...snapshot, status, failure: "" });
    } catch (error) {
        publish({ ...snapshot, failure: String(error) });
    }
}

/** 手动刷新一次（打开详情时用，不必等 750ms）。 */
export const refreshHostLinkStatus = refresh;
