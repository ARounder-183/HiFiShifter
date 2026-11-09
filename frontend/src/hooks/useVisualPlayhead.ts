// 原GUI视觉播放头：插件插值有界，独立App保留原来的连续外推语义。
import { useEffect, useRef, type MutableRefObject } from "react";
import { isPluginMode } from "../services/hostCapabilities";

/** 宿主采样短暂缺失时只填最多100ms，不让GUI在合成/宿主暂停期间自行前漂数秒。 */
export function projectVisualPlayhead(
    positionSec: number,
    elapsedSec: number,
    hostControlled: boolean,
): number {
    const elapsed = Number.isFinite(elapsedSec) ? Math.max(0, elapsedSec) : 0;
    return positionSec + (hostControlled ? Math.min(0.1, elapsed) : elapsed);
}

/**
 * 把一次**已经过去**的采样补齐到当前时刻。
 *
 * 【与 `projectVisualPlayhead` 的分工】那个函数带 `hostControlled` 上限，约束的是
 * "对未来推测多远"；本函数处理的是"过去已经过去了多久"，那是被测量的量，不该封顶。
 * 两者分开命名，任何一处混用都会立刻表现为光标系统性落后于它自己报出的时间。
 */
export function catchUpSampledPlayhead(positionSec: number, ageSec: number): number {
    return projectVisualPlayhead(positionSec, ageSec, false);
}

/**
 * 播放轮询采样仍被视为"新鲜"的最大年龄（秒）。
 *
 * 30Hz 轮询下，reducer 处理（≈IPC 到达）到 React 提交本 effect 通常只有
 * 几毫秒；超过该阈值意味着这次 playheadSec 变更不是轮询写入（seek / stop /
 * 粘贴 / 撤销等），或者发生了极端卡顿 —— 此时不得用陈旧时间戳外推，
 * 必须硬复位到提交值。
 */
const PLAYHEAD_SAMPLE_FRESH_SEC = 0.25;

/**
 * 纠偏速率上限（秒/秒）：每帧最多把"视觉位置"往采样值移动 25% 的时间差。
 *
 * 【为什么是这个量级】300 ms 的采样迟到要在 "1.2 s 内吃掉 —— 再快就能看出跳变，
 * 再慢就会长时间偏离真实位置。
 */
const SLEW_SEC_PER_SEC = 0.25;

/**
 * 超过这个差值就当成**真的跳转**（seek / 循环折返），直接硬复位。
 *
 * 【为什么必须区分】平滑一个真实的 seek 是错的：用户按了跳转，光标就该跳。
 */
const SEEK_RESET_SEC = 0.5;

/**
 * 视觉播放头的**进程级唯一时钟**。
 *
 * 【为什么必须共享】时间轴、参数编辑器与轨道头各有一个 `useVisualPlayhead` 消费者。
 * 每个消费者各自跑一条 rAF 时，它们的锚点建立时刻、回调时间戳与纠偏步长都不完全相同；
 * 自动滚屏又拿"时间轴时钟"算滚动位置，而参数编辑器拿"自己的时钟"画光标。两个时钟
 * 相差一帧时，光标在屏幕上的位置会随滚动位置前后摆动（用户看到的"抽动"）。共享一条
 * 时钟后，所有消费者在同一帧拿到**同一个数**，滚动与光标不再互相追赶。
 *
 * 【为什么订阅者分两阶段通知】自动滚屏的回调会在时间轴消费者里同步写共享视口，
 * 参数编辑器收到广播后会立即重绘。若此时参数编辑器的 `visualPlayheadSecRef` 还没
 * 更新，这一帧就会用旧时间画光标。因此先对所有订阅者 `setValue`，再统一 `onFrame`。
 */
interface VisualPlayheadSubscriber {
    /** 只写消费者的 ref，不触发绘制。 */
    setValue(sec: number): void;
    /** 所有消费者的 ref 都写好后再触发绘制/滚动。 */
    onFrame(sec: number): void;
}

let clockVisualSec = 0;
let clockAnchor = { playheadSec: 0, timestampMs: 0 };
let clockTargetSec = 0;
let clockAdvancing = false;
let clockInitialized = false;
let clockRafId: number | null = null;
let lastSampleKey: string | null = null;
let lastSampleAtMs = Number.NEGATIVE_INFINITY;
const clockSubscribers = new Set<VisualPlayheadSubscriber>();

function stopClock(): void {
    if (clockRafId !== null) {
        cancelAnimationFrame(clockRafId);
        clockRafId = null;
    }
}

function notifyClock(sec: number): void {
    clockVisualSec = sec;
    const subscribers = Array.from(clockSubscribers);
    // 第一阶段：所有消费者先拿到同一个时间值。
    for (const subscriber of subscribers) subscriber.setValue(sec);
    // 第二阶段：再允许消费者做会触发绘制 / 跨面板广播的工作。
    for (const subscriber of subscribers) subscriber.onFrame(sec);
}

function ensureClockRunning(): void {
    if (!clockAdvancing || clockSubscribers.size === 0 || clockRafId !== null) return;
    if (typeof requestAnimationFrame !== "function") return;
    const nowMs = performance.now();
    const anchorAgeSec = (nowMs - clockAnchor.timestampMs) / 1000;
    if (
        !Number.isFinite(anchorAgeSec) ||
        anchorAgeSec < 0 ||
        anchorAgeSec > PLAYHEAD_SAMPLE_FRESH_SEC
    ) {
        // 与旧实现的 rAF 重启防护一致：传输暂停过久后重新播放时，不能把整段
        // "未前进"的空闲时间一次性加到光标上。
        clockAnchor = { playheadSec: clockVisualSec, timestampMs: nowMs };
    }
    clockRafId = requestAnimationFrame(clockTick);
}

function clockTick(timestampMs: number): void {
    clockRafId = null;
    if (!clockAdvancing || clockSubscribers.size === 0) return;
    const elapsedSec = (timestampMs - clockAnchor.timestampMs) / 1000;
    // 1) 按 1x 前进。**不设上限**：宿主在播放是已知事实，采样只是它的估计值。
    const advanced = projectVisualPlayhead(clockAnchor.playheadSec, elapsedSec, false);
    // 2) 朝采样值纠偏，每帧最多走 `SLEW_SEC_PER_SEC × dt`。
    const maxStep = SLEW_SEC_PER_SEC * Math.max(0, elapsedSec);
    const error = clockTargetSec - advanced;
    const nextPlayheadSec = advanced + Math.max(-maxStep, Math.min(maxStep, error));
    clockAnchor = { playheadSec: nextPlayheadSec, timestampMs };
    notifyClock(nextPlayheadSec);
    ensureClockRunning();
}

function setVisualPlayheadAdvancing(next: boolean): void {
    if (clockAdvancing === next) return;
    clockAdvancing = next;
    if (next) ensureClockRunning();
    else stopClock();
}

function updateVisualPlayheadSample(args: {
    syncedPlayheadSec: number;
    syncedAtMs?: number;
    hostControlled: boolean;
}): void {
    const { syncedPlayheadSec, syncedAtMs, hostControlled } = args;
    const nowMs = performance.now();
    const sampledAtMs = typeof syncedAtMs === "number" && syncedAtMs > 0 ? syncedAtMs : NaN;
    const ageSec = Number.isFinite(sampledAtMs) ? Math.max(0, (nowMs - sampledAtMs) / 1000) : NaN;
    const hasAge = Number.isFinite(ageSec);
    const fresh = hasAge && ageSec <= PLAYHEAD_SAMPLE_FRESH_SEC;
    const sampleKey = `${syncedPlayheadSec}|${Number.isFinite(sampledAtMs) ? sampledAtMs : "none"}`;

    // 多个消费者会带着同一个 store 采样依次调用本函数。采样时刻更旧的调用
    // 不能把共享时钟的锚点拉回去。
    if (Number.isFinite(sampledAtMs) && sampledAtMs < lastSampleAtMs) return;

    if (!clockInitialized) {
        clockInitialized = true;
        clockVisualSec = syncedPlayheadSec;
        clockAnchor = { playheadSec: syncedPlayheadSec, timestampMs: nowMs };
        clockTargetSec = syncedPlayheadSec;
    }

    if (clockAdvancing && (hostControlled || fresh)) {
        if (sampleKey !== lastSampleKey) {
            const extrapolatedSec = catchUpSampledPlayhead(syncedPlayheadSec, hasAge ? ageSec : 0);
            const jumpSec = Math.abs(extrapolatedSec - clockVisualSec);
            if (!Number.isFinite(jumpSec) || jumpSec > SEEK_RESET_SEC) {
                clockAnchor = { playheadSec: extrapolatedSec, timestampMs: nowMs };
                clockTargetSec = extrapolatedSec;
                notifyClock(extrapolatedSec);
            } else {
                // 只更新目标，下一帧由 tick 平滑追上；不要在这里改视觉值，
                // 否则采样通道的抖动会直接变成光标跳变。
                clockTargetSec = extrapolatedSec;
            }
            lastSampleKey = sampleKey;
            if (Number.isFinite(sampledAtMs)) lastSampleAtMs = sampledAtMs;
        }
    } else if (sampleKey !== lastSampleKey || clockVisualSec !== syncedPlayheadSec) {
        // 非轮询写入（seek / stop / 粘贴……）或采样过旧：硬复位到提交值。
        clockAnchor = { playheadSec: syncedPlayheadSec, timestampMs: nowMs };
        clockTargetSec = syncedPlayheadSec;
        notifyClock(syncedPlayheadSec);
        lastSampleKey = sampleKey;
        if (Number.isFinite(sampledAtMs)) lastSampleAtMs = sampledAtMs;
    }

    if (clockAdvancing) ensureClockRunning();
    else stopClock();
}

function subscribeVisualPlayhead(subscriber: VisualPlayheadSubscriber): () => void {
    clockSubscribers.add(subscriber);
    if (clockInitialized) {
        subscriber.setValue(clockVisualSec);
        subscriber.onFrame(clockVisualSec);
    }
    if (clockAdvancing) ensureClockRunning();
    return () => {
        clockSubscribers.delete(subscriber);
        if (clockSubscribers.size === 0) stopClock();
    };
}

/** 读当前共享视觉播放头（秒）。用于非 React 的逐帧消费者与测试。 */
export function readSharedVisualPlayheadSec(): number {
    return clockVisualSec;
}

/** 仅供测试：清空共享时钟，避免用例之间互相污染。 */
export function resetVisualPlayheadClockForTests(): void {
    stopClock();
    clockVisualSec = 0;
    clockAnchor = { playheadSec: 0, timestampMs: 0 };
    clockTargetSec = 0;
    clockAdvancing = false;
    clockInitialized = false;
    lastSampleKey = null;
    lastSampleAtMs = Number.NEGATIVE_INFINITY;
    clockSubscribers.clear();
}

export interface UseVisualPlayheadArgs {
    syncedPlayheadSec: number;
    isTransportAdvancing: boolean;
    /**
     * syncedPlayheadSec 的采样时刻（performance.now() 毫秒时钟，与 RAF 同源）。
     *
     * 只由播放轮询 reducer 在写入 playheadSec 时更新（state.session
     * .playheadSampledAtMs）。轮询载荷是"后端执行命令那一刻"的位置，reducer
     * 已把它按时延外推到该时刻 —— 锚定采样时刻（而非 effect 运行时刻），
     * React 提交延迟就不再引入滞后/抖动；effect 内部会先外推到当前时刻，
     * 首帧即精确。seek/stop 等非轮询写入不更新该时间戳（保持陈旧），
     * 走下方硬复位分支。
     */
    syncedAtMs?: number;
    onFrame?: (playheadSec: number) => void;
}

/**
 * 订阅共享视觉播放头。所有调用方共用一条 rAF，并在同一帧拿到同一个值。
 *
 * 语义与旧实现逐条保持一致：
 * - 新鲜采样只更新目标，视觉值按 `SLEW_SEC_PER_SEC` 平滑纠偏；
 * - 真实跳转（seek / 循环折返）立即硬复位；
 * - 停止时只停表，不把视觉值拉回旧采样（光标冻结在停止那一拍）。
 */
export function useVisualPlayhead({
    syncedPlayheadSec,
    isTransportAdvancing,
    syncedAtMs,
    onFrame,
}: UseVisualPlayheadArgs): MutableRefObject<number> {
    const visualPlayheadSecRef = useRef(readSharedVisualPlayheadSec());
    const onFrameRef = useRef(onFrame);
    const hostControlled = isPluginMode();

    useEffect(() => {
        onFrameRef.current = onFrame;
    }, [onFrame]);

    // 必须先于采样 effect：挂载 / 恢复播放时共享时钟要已经处于"在走"的状态，
    // 采样 effect 才能走外推而不是把光标硬复位到旧采样。
    useEffect(() => {
        setVisualPlayheadAdvancing(isTransportAdvancing);
    }, [isTransportAdvancing]);

    useEffect(() => {
        updateVisualPlayheadSample({ syncedPlayheadSec, syncedAtMs, hostControlled });
    }, [syncedPlayheadSec, syncedAtMs, hostControlled]);

    useEffect(() => {
        return subscribeVisualPlayhead({
            setValue: (sec) => {
                visualPlayheadSecRef.current = sec;
            },
            onFrame: (sec) => onFrameRef.current?.(sec),
        });
    }, []);

    return visualPlayheadSecRef;
}
