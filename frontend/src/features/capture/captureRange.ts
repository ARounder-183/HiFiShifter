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

/** 捕获中的工程秒区间（`null` = 当前没有在捕获）。 */
export interface CaptureRange {
    /** 已覆盖区间的左端（工程秒）。 */
    startSec: number;
    /** 已覆盖区间的右端（工程秒）。录制推进时向右长。 */
    endSec: number;
    /** 当前位置（工程秒）。与 `endSec` 取较大者用于绘制右缘。 */
    positionSec: number;
}

/**
 * 从一次状态快照里取出捕获区间。
 *
 * 【为什么 `endSec` 要取 `max(endFrame, positionFrame)`】后端报的 `endFrame` 是"已封口
 * 段的右端"，而当前段的样本还在往里写 —— 只看 `endFrame` 会让带子在每个段边界上
 * "停一下再跳"，看起来像卡住了。
 */
export function deriveCaptureRange(status: CaptureStatus | null): CaptureRange | null {
    const capture = status?.capture ?? null;
    if (capture === null) return null;
    const rate = capture.sampleRate;
    if (!(rate > 0)) return null;
    const startSec = frameToSec(capture.startFrame, rate);
    const endSec = frameToSec(Math.max(capture.endFrame, capture.positionFrame), rate);
    return {
        startSec,
        endSec: Math.max(endSec, startSec),
        positionSec: frameToSec(capture.positionFrame, rate),
    };
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
let cachedRange: CaptureRange | null = null;

function snapshotRange(): CaptureRange | null {
    const status = getCaptureSnapshot().status;
    if (status !== cachedStatus) {
        cachedStatus = status;
        cachedRange = deriveCaptureRange(status);
    }
    return cachedRange;
}

/** 订阅捕获区间。两个面板都调用它，因此看到的必然是同一份。 */
export function useCaptureRange(): CaptureRange | null {
    return useSyncExternalStore(subscribeCapture, snapshotRange, () => null);
}
