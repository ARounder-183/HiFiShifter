/**
 * Transfer 状态的**唯一**拉取点。
 *
 * 【为什么是一个共享 store 而不是各自 useState】Transfer 现在有两个消费方：传输栏上的
 * 录制控件（主入口）与「Transfer 设置与诊断」面板。各拉各的会让同一条 200ms 轮询
 * 跑两遍（插件 webview 的每次往返都要过宿主消息循环），而且两处的读数会**错开一帧**
 * —— 控件显示"录制中"、面板显示"已录好"这种事，用户会当成 bug。
 *
 * 【为什么是轮询】录制进度每 100ms 变一次，宿主不会推送；后端 `transfer_status` 不取
 * 渲染锁，代价只是一个整数读取。
 *
 * 【为什么订阅判据是引用相等】轮询每 200ms 产生一个新对象。`useSyncExternalStore`
 * 按返回值是否相等决定要不要重渲染，所以内容没变时必须保留旧引用 —— 否则整个
 * 传输栏每 200ms 重渲染一次。
 */
import { useSyncExternalStore } from "react";

import {
    transferApi,
    transferError,
    type TransferOptions,
    type TransferStatus,
} from "./transferApi";

/** 轮询间隔：与后端录制读数的更新节奏一致（见 `transfer/mod.rs` 的说明）。 */
const POLL_MS = 200;

export interface TransferSnapshot {
    status: TransferStatus | null;
    /** 最近一次失败的原文（面板与控件各自决定怎么呈现）。 */
    error: string;
    /** 有命令在途（按钮据此禁用，避免重复提交）。 */
    busy: boolean;
    /**
     * 每次"录制完成"自增。
     *
     * 【为什么用票据而不是"state 变成 ready"】录制两次之间 `state` 一直是 `ready`，
     * 拿状态做判据只能捕捉到第一次。视口的"跳到刚录好的区间"需要**每次**都触发。
     */
    recordedToken: number;
    /** 最近一次录制完成后的工程秒跨度（`null` = 没有可定位的区间）。 */
    recordedSpanSec: [number, number] | null;
}

const EMPTY: TransferSnapshot = {
    status: null,
    error: "",
    busy: false,
    recordedToken: 0,
    recordedSpanSec: null,
};

let snapshot: TransferSnapshot = EMPTY;
const listeners = new Set<() => void>();

function publish(next: TransferSnapshot): void {
    if (snapshot === next) return;
    snapshot = next;
    for (const listener of listeners) listener();
}

/** 内容相同就保留旧引用：订阅者不该因为"又轮询了一次"而重渲染。 */
function withStatus(status: TransferStatus): TransferSnapshot {
    if (snapshot.status === status) return snapshot;
    return { ...snapshot, status };
}

export function subscribeTransfer(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

export function getTransferSnapshot(): TransferSnapshot {
    return snapshot;
}

/** 订阅版：返回整个快照（控件与面板都读它）。 */
export function useTransferSnapshot(): TransferSnapshot {
    return useSyncExternalStore(subscribeTransfer, getTransferSnapshot, getTransferSnapshot);
}

/** 只关心状态的那部分调用方（多数）用这个，避免为 `busy` 的变化重渲染。 */
export function useTransferStatus(): TransferStatus | null {
    return useSyncExternalStore(
        subscribeTransfer,
        () => snapshot.status,
        () => snapshot.status,
    );
}

/** 仅供测试重置模块级状态。 */
export function resetTransferStoreForTests(): void {
    snapshot = EMPTY;
    listeners.clear();
}

async function refresh(): Promise<void> {
    try {
        publish(withStatus(await transferApi.status()));
        if (snapshot.error !== "") publish({ ...snapshot, error: "" });
    } catch (cause) {
        // 【为什么不停止轮询】状态查询失败是暂时的（会话未就绪等），而面板的职责是
        // 持续反映真实状态。终态判定由 `invoke` 的统一闸门做。
        const error = transferError(cause);
        if (snapshot.error !== error) publish({ ...snapshot, error });
    }
}

let running = 0;
let timer: number | undefined;

/**
 * 开始轮询；返回停止函数。多次调用只保留一条轮询（引用计数）。
 *
 * 【为什么用引用计数】控件与面板各自在挂载时开始、卸载时停止；两者同时在场时只有
 * 一条轮询在跑，其中一个卸载也不会让另一个断流。
 */
export function startTransferPolling(): () => void {
    running += 1;
    void refresh();
    if (timer === undefined) {
        timer = window.setInterval(() => void refresh(), POLL_MS);
    }
    let stopped = false;
    return () => {
        if (stopped) return;
        stopped = true;
        running -= 1;
        if (running <= 0 && timer !== undefined) {
            window.clearInterval(timer);
            timer = undefined;
            running = 0;
        }
    };
}

/** 一次录制完成的后果：推进票据 + 记下区间（视口据此定位）。 */
function noteRecorded(status: TransferStatus | undefined): void {
    // 【为什么容忍 undefined】`stop` 的回信是 `{take, status}`；后端若改了形状（或
    // 调用方拿错了字段），这里宁可什么都不做，也不能把一个"录制完成"的假信号推给视口。
    if (!status) return;
    const span = status.span;
    const sampleRate = span?.sampleRate ?? 0;
    const seconds =
        span && sampleRate > 0
            ? ([span.startFrame / sampleRate, span.endFrame / sampleRate] as [number, number])
            : null;
    publish({
        ...snapshot,
        status,
        error: "",
        recordedToken: snapshot.recordedToken + 1,
        recordedSpanSec: seconds,
    });
}

/** 跑一条会改变状态的动作：置忙、执行、用返回的状态刷新。 */
async function act(
    run: () => Promise<TransferStatus>,
    onDone?: (status: TransferStatus | undefined) => void,
) {
    publish({ ...snapshot, busy: true });
    try {
        const status = await run();
        if (onDone) onDone(status);
        else publish({ ...snapshot, status, error: "" });
    } catch (cause) {
        publish({ ...snapshot, error: transferError(cause) });
    } finally {
        publish({ ...snapshot, busy: false });
    }
}

export const transferActions = {
    start: () => act(() => transferApi.start()),
    /** 停止并 promote：这是"录制完成"，视口要跳到刚录好的区间。 */
    stop: () => act(() => transferApi.stop().then((reply) => reply.status), noteRecorded),
    cancel: () => act(() => transferApi.cancel()),
    clear: () => act(() => transferApi.clear()),
    deleteTake: (takeId: number) => act(() => transferApi.deleteTake(takeId)),
    setOptions: (patch: Partial<TransferOptions>) => act(() => transferApi.setOptions(patch)),
    /** 手动刷新（面板打开时用一次，不必等 200ms）。 */
    refresh,
};

/** 供测试注入一条状态（不发命令）。 */
export function publishTransferStatusForTests(status: TransferStatus | null): void {
    publish({ ...snapshot, status });
}
