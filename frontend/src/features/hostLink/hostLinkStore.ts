/**
 * 独立 App 的宿主链接会话状态。
 *
 * 【为什么是一个模块级 store，而不是面板里的 useState】面板是一个可以被反复
 * 打开/关闭的浮出表面；会话却是一次"打开、编辑、提交"的长活过程。状态留在组件里
 * 会随面板关闭一起丢掉，用户再打开时看到的是"未连接"，而工作区其实还开着。
 * 模块级 store 让会话寿命独立于面板的挂载周期。
 *
 * 【为什么订阅判据是引用相等】`useSyncExternalStore` 按返回值是否相等决定要不要
 * 重渲染；快照对象只在真正变化时替换引用，未变化的读取不会触发渲染。
 */
import { useSyncExternalStore } from "react";

import { hostLinkApi, type HostLinkItem, type HostLinkSession } from "./hostLinkApi";

/** 最近一次成功动作的状态词；都是词表里的既有键，不拼字符串。 */
export type HostLinkStatusKey =
    | "host_link_status_opened"
    | "host_link_status_refreshed"
    | "host_link_status_submitted"
    | "host_link_status_released";

export interface HostLinkSnapshot {
    /** 扫描根目录（用于展示"在哪儿找）。 */
    root: string;
    /** 可打开（或读不出来）的工作区。 */
    items: HostLinkItem[];
    /** 当前打开的会话；未打开时为 `null`。 */
    session: HostLinkSession | null;
    /** 有命令在途（按钮据此禁用，避免重复提交）。 */
    busy: boolean;
    /** 最近一次失败的原始错误；文案在渲染时按当前语言本地化。 */
    error: unknown;
    /** 最近一次成功动作的状态词。 */
    status: HostLinkStatusKey | null;
}

const EMPTY: HostLinkSnapshot = {
    root: "",
    items: [],
    session: null,
    busy: false,
    error: null,
    status: null,
};

let snapshot: HostLinkSnapshot = EMPTY;
const listeners = new Set<() => void>();

function publish(next: HostLinkSnapshot): void {
    snapshot = next;
    for (const listener of listeners) listener();
}

export function subscribeHostLink(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

export function getHostLinkSnapshot(): HostLinkSnapshot {
    return snapshot;
}

export function useHostLinkSnapshot(): HostLinkSnapshot {
    return useSyncExternalStore(subscribeHostLink, getHostLinkSnapshot, getHostLinkSnapshot);
}

/** 仅供测试重置模块级状态。 */
export function resetHostLinkStoreForTests(): void {
    snapshot = EMPTY;
    listeners.clear();
}

/**
 * 跑一条会改变会话的动作：置忙、执行、用返回的结果刷新。
 *
 * 【为什么把原始错误返回给调用方】脏工程被**后端**发现时（前端状态过期，例如
 * 工程在别处被改脏），调用方要回到确认态并重试 `force: true` —— 它需要看到错误
 * 本身，而不只是 store 里那份用于展示的文本。因此这里既记录错误，也把它交回去。
 */
async function act(run: () => Promise<void>): Promise<unknown | null> {
    publish({ ...snapshot, busy: true, error: null });
    try {
        await run();
        publish({ ...snapshot, busy: false });
        return null;
    } catch (error) {
        publish({ ...snapshot, busy: false, error });
        return error;
    }
}

export const hostLinkActions = {
    /** 重扫工作区列表。当前选中项消失时由面板负责回落。 */
    list: (directory?: string) =>
        act(async () => {
            const found = await hostLinkApi.list(directory);
            publish({ ...snapshot, root: found.root, items: found.items });
        }),

    /** 打开一个工作区；`force` 用于确认后覆盖有未保存改动的本地工程。 */
    open: (directory: string, force: boolean) =>
        act(async () => {
            const session = await hostLinkApi.open(directory, force);
            publish({ ...snapshot, session, status: "host_link_status_opened" });
        }),

    /** 从宿主重新导入几何（保留仍存在的轨道上的曲线）。 */
    refresh: (force: boolean) =>
        act(async () => {
            const session = await hostLinkApi.refresh(force);
            publish({ ...snapshot, session, status: "host_link_status_refreshed" });
        }),

    /** 把当前的曲线编辑写回工作区。 */
    submit: () =>
        act(async () => {
            const session = await hostLinkApi.submit();
            publish({ ...snapshot, session, status: "host_link_status_submitted" });
        }),

    /** 断开（释放工作区的打开标记）；本地工程保留。 */
    release: () =>
        act(async () => {
            await hostLinkApi.release();
            publish({ ...snapshot, session: null, status: "host_link_status_released" });
        }),

    /** 清掉展示用的错误（例如进入脏工程确认态之前）。 */
    clearError: () => {
        if (snapshot.error !== null) publish({ ...snapshot, error: null });
    },
};
