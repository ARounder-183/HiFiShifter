/**
 * 捕获中的工程秒区间 —— 时间轴与参数编辑器**共用的唯一来源**。
 *
 * 【为什么它必须与"某一层组件"解耦】捕获范围在插件里是**主工作面**（参数编辑器）
 * 也要看到的信息，而两个面板可能不在同一时刻可见（并列标签页）。把这段换算藏在
 * 时间轴那一层里，参数编辑器就永远看不到它 —— 用户只开着参数编辑器时，捕获过程中
 * 界面上完全没有任何东西在动。
 *
 * 【为什么是派生函数而不是第二个 store】真值已经在 `captureStore` 里（`capture` 与
 * `span` 两组帧号）。再开一份状态就必须有第二个发布点，而两个发布点必然会有先后
 * 与遗漏 —— 表现为"时间轴上的带子在动、参数编辑器里的不动"。派生 + 订阅同一份快照
 * 在结构上排除了这种分叉。
 */
import { useSyncExternalStore } from "react";

import { frameToSec, type CaptureStatus } from "./captureApi";
import { getCaptureSnapshot, subscribeCapture } from "./captureStore";

export { subscribeCapture };

/** 一段已经录到的工程秒区间。 */
export interface CaptureBand {
    startSec: number;
    endSec: number;
}

/**
 * 捕获中的工程秒区间**列表**（`null` = 当前没有在捕获）。
 *
 * 【为什么是列表而不是一个区间】用户在捕获过程中可以把播放光标跳到任意位置再继续
 * 录。并集会把"录了 0..3 秒、跳到 30 秒、又录了 30..33 秒"画成一条 0..33 秒的连续
 * 带子 —— 中间 27 秒其实什么都没录到。带子必须逐段画。
 */
export interface CaptureBands {
    bands: CaptureBand[];
    /** 当前位置（工程秒）：捕获中唯一"在动"的东西。 */
    positionSec: number;
}

/**
 * 从一次状态快照里取出捕获区间列表。
 *
 * 【为什么 `positionSec` 只延长**最后一段**】后端报的 `endFrame` 是"已封口段的右端"，
 * 而当前段的样本还在往里写 —— 只看 `endFrame` 会让带子在每个段边界上"停一下再跳"，
 * 看起来像卡住了。但把这个 `max` 应用到**每一段**就会让跳播之后所有历史段的右端都被
 * 拉到当前位置，等于又并成一整条 —— 那正是本函数要避免的错误。
 */
export function deriveCaptureBands(status: CaptureStatus | null): CaptureBands | null {
    const capture = status?.capture ?? null;
    if (capture === null) return null;
    const rate = capture.sampleRate;
    if (!(rate > 0)) return null;

    // 旧载荷 / 空载荷兜底：没有列表时退回并集那一段。
    const source =
        Array.isArray(capture.runs) && capture.runs.length > 0
            ? capture.runs
            : [{ startFrame: capture.startFrame, endFrame: capture.endFrame }];

    const bands: CaptureBand[] = [];
    for (const run of source) {
        const startSec = frameToSec(run.startFrame, rate);
        const endSec = frameToSec(run.endFrame, rate);
        if (!(endSec > startSec)) continue;
        bands.push({ startSec, endSec });
    }
    if (bands.length === 0) return null;
    bands.sort((a, b) => a.startSec - b.startSec);

    const positionSec = frameToSec(capture.positionFrame, rate);
    const last = bands[bands.length - 1];
    if (positionSec > last.endSec) last.endSec = positionSec;
    return { bands, positionSec: Math.max(positionSec, last.startSec) };
}

/**
 * 按快照身份缓存上一次的派生结果。
 *
 * 【为什么必须有】`useSyncExternalStore` 的 `getSnapshot` 必须**返回稳定引用**：
 * 每次调用都新建一个对象会让 React 认为"值一直在变"，于是无限重渲染（控制台会报
 * "The result of getSnapshot should be cached"）。派生本身是纯函数，用快照的引用
 * 做缓存键即可 —— 快照没换，区间就一定没变。
 */
let cachedStatus: CaptureStatus | null | undefined;
let cachedBands: CaptureBands | null = null;

function snapshotBands(): CaptureBands | null {
    const status = getCaptureSnapshot().status;
    if (status !== cachedStatus) {
        cachedStatus = status;
        cachedBands = deriveCaptureBands(status);
    }
    return cachedBands;
}

/**
 * 读一次当前区间列表（**不订阅**）。
 *
 * 【为什么需要非订阅读法】参数编辑器的带子由内核在**帧提交**里画：它只需要"这一帧
 * 的最新值"。若走 `useCaptureBands`，录制推进时整个面板会以十几 Hz 重渲染 ——
 * 为了画几条 4px 的带子重渲染一棵含 WebGL 宿主的树，代价荒谬。
 */
export function getCaptureBands(): CaptureBands | null {
    return snapshotBands();
}

/** 订阅捕获区间。两个面板都调用它，因此看到的必然是同一份。 */
export function useCaptureBands(): CaptureBands | null {
    return useSyncExternalStore(subscribeCapture, snapshotBands, () => null);
}
