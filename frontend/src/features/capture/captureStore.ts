/**
 * Capture 状态的**唯一**拉取点。
 *
 * 【为什么是一个共享 store 而不是各自 useState】Capture 现在有两个消费方：传输栏上的
 * 录制控件（主入口）与「Capture 设置与诊断」面板。各拉各的会让同一条 200ms 轮询
 * 跑两遍（插件 webview 的每次往返都要过宿主消息循环），而且两处的读数会**错开一帧**
 * —— 控件显示"录制中"、面板显示"已录好"这种事，用户会当成 bug。
 *
 * 【为什么是轮询】录制进度每 100ms 变一次，宿主不会推送；后端 `capture_status` 不取
 * 渲染锁，代价只是一个整数读取。
 *
 * 【为什么订阅判据是引用相等】轮询每 200ms 产生一个新对象。`useSyncExternalStore`
 * 按返回值是否相等决定要不要重渲染，所以内容没变时必须保留旧引用 —— 否则整个
 * 传输栏每 200ms 重渲染一次。
 */
import { useSyncExternalStore } from "react";

import { captureApi, captureError, type CaptureOptions, type CaptureStatus } from "./captureApi";
import { pluginBootstrap } from "../../services/hostDetect";

/** 轮询间隔：与后端录制读数的更新节奏一致（见 `capture/mod.rs` 的说明）。 */
const POLL_MS = 200;

export interface CaptureSnapshot {
    status: CaptureStatus | null;
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

const EMPTY: CaptureSnapshot = {
    status: null,
    error: "",
    busy: false,
    recordedToken: 0,
    recordedSpanSec: null,
};

let snapshot: CaptureSnapshot = EMPTY;
const listeners = new Set<() => void>();

function publish(next: CaptureSnapshot): void {
    if (snapshot === next) return;
    snapshot = next;
    for (const listener of listeners) listener();
}

/**
 * 内容相同就保留旧引用：订阅者不该因为"又轮询了一次"而重渲染。
 *
 * 【为什么是序列化比较而不是引用比较】`captureApi.status()` 每 200ms 经 IPC 反序列化
 * 出**全新对象**，引用永不相等 —— 旧判据等于从未生效，订阅者恒定 5Hz 空重渲染
 * （第 5 轮审查 P2-1）。
 * 【为什么是 JSON.stringify 而不是字段枚举】`CaptureStatus` 的形状由后端定义且仍在
 * 演进，新增字段不该悄悄变成"内容变了"；一次 stringify 的开销远低于一次全栏重渲染。
 */
let lastStatusJson: string | null = null;

/**
 * 所有直接写 `snapshot.status` 的路径共用：同步更新比较缓存。
 *
 * 【为什么不能只在轮询里更新】`act()` / `noteRecorded` 用命令回信直写 status
 * 时若不动缓存，动作后的第一轮轮询会拿同一份状态与**旧缓存**比较，误判
 * "内容变了"，多发一次全栏重渲染 —— 正是这套缓存要防的事。
 */
function rememberStatus(status: CaptureStatus | null): void {
    lastStatusJson = status === null ? null : JSON.stringify(status);
}

function withStatus(status: CaptureStatus): CaptureSnapshot {
    const json = JSON.stringify(status);
    if (snapshot.status !== null && json === lastStatusJson) return snapshot;
    lastStatusJson = json;
    return { ...snapshot, status };
}

export function subscribeCapture(listener: () => void): () => void {
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

export function getCaptureSnapshot(): CaptureSnapshot {
    return snapshot;
}

/** 订阅版：返回整个快照（控件与面板都读它）。 */
export function useCaptureSnapshot(): CaptureSnapshot {
    return useSyncExternalStore(subscribeCapture, getCaptureSnapshot, getCaptureSnapshot);
}

/** 仅供测试重置模块级状态。 */
export function resetCaptureStoreForTests(): void {
    snapshot = EMPTY;
    lastStatusJson = null;
    listeners.clear();
}

/**
 * 路由未建立时，`capture_status` 一定失败（后端拿不到 owner）。
 *
 * 【为什么直接跳过而不是让它失败】失败会以 5 Hz 的速度把同一条错误写进日志与状态栏
 * —— 实测一次会话 60 秒 300 条。这条读数的意义是「现在能不能录」，而「编辑器还没连上
 * 处理器」已经把答案定死了；等路由恢复（bootstrap 事件会刷新页面状态）再问，
 * 结论一样，噪声少两个数量级。
 */
function routeIsUnavailable(): boolean {
    return pluginBootstrap()?.editorRoute?.state === "unbound";
}

async function refresh(): Promise<void> {
    if (routeIsUnavailable()) return;
    try {
        publish(withStatus(await captureApi.status()));
        if (snapshot.error !== "") publish({ ...snapshot, error: "" });
    } catch (cause) {
        // 【为什么不停止轮询】状态查询失败是暂时的（会话未就绪等），而面板的职责是
        // 持续反映真实状态。终态判定由 `invoke` 的统一闸门做。
        const error = captureError(cause);
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
export function startCapturePolling(): () => void {
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
function noteRecorded(status: CaptureStatus | undefined): void {
    // 【为什么容忍 undefined】`stop` 的回信是 `{take, status}`；后端若改了形状（或
    // 调用方拿错了字段），这里宁可什么都不做，也不能把一个"录制完成"的假信号推给视口。
    if (!status) return;
    const span = status.span;
    const sampleRate = span?.sampleRate ?? 0;
    const seconds =
        span && sampleRate > 0
            ? ([span.startFrame / sampleRate, span.endFrame / sampleRate] as [number, number])
            : null;
    rememberStatus(status);
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
    run: () => Promise<CaptureStatus>,
    onDone?: (status: CaptureStatus | undefined) => void,
) {
    publish({ ...snapshot, busy: true });
    try {
        const status = await run();
        if (onDone) onDone(status);
        else {
            rememberStatus(status);
            publish({ ...snapshot, status, error: "" });
        }
    } catch (cause) {
        publish({ ...snapshot, error: captureError(cause) });
    } finally {
        publish({ ...snapshot, busy: false });
    }
}

export const captureActions = {
    start: () => act(() => captureApi.start()),
    /** 停止并 promote：这是"录制完成"，视口要跳到刚录好的区间。 */
    stop: () => act(() => captureApi.stop().then((reply) => reply.status), noteRecorded),
    cancel: () => act(() => captureApi.cancel()),
    clear: () => act(() => captureApi.clear()),
    deleteTake: (takeId: number) => act(() => captureApi.deleteTake(takeId)),
    setOptions: (patch: Partial<CaptureOptions>) => act(() => captureApi.setOptions(patch)),
    /** 手动刷新（面板打开时用一次，不必等 200ms）。 */
    refresh,
};

/** 供测试注入一条状态（不发命令）。 */
export function publishCaptureStatusForTests(status: CaptureStatus | null): void {
    rememberStatus(status);
    publish({ ...snapshot, status });
}
