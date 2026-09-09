import { useEffect, useRef, type MutableRefObject } from "react";

/**
 * 播放轮询采样仍被视为"新鲜"的最大年龄（秒）。
 *
 * 30Hz 轮询下，reducer 处理（≈IPC 到达）到 React 提交本 effect 通常只有
 * 几毫秒；超过该阈值意味着这次 playheadSec 变更不是轮询写入（seek / stop /
 * 粘贴 / 撤销等），或者发生了极端卡顿 —— 此时不得用陈旧时间戳外推，
 * 必须硬复位到提交值。
 */
const PLAYHEAD_SAMPLE_FRESH_SEC = 0.25;

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

export function useVisualPlayhead({
    syncedPlayheadSec,
    isTransportAdvancing,
    syncedAtMs,
    onFrame,
}: UseVisualPlayheadArgs): MutableRefObject<number> {
    const visualPlayheadSecRef = useRef(syncedPlayheadSec);
    const syncAnchorRef = useRef({
        playheadSec: syncedPlayheadSec,
        timestampMs: 0,
    });
    const advancingRef = useRef(isTransportAdvancing);
    const onFrameRef = useRef(onFrame);

    useEffect(() => {
        advancingRef.current = isTransportAdvancing;
    }, [isTransportAdvancing]);

    useEffect(() => {
        onFrameRef.current = onFrame;
    }, [onFrame]);

    useEffect(() => {
        const now = performance.now();
        const sampledAtMs = typeof syncedAtMs === "number" && syncedAtMs > 0 ? syncedAtMs : NaN;
        const ageSec = Number.isFinite(sampledAtMs) ? Math.max(0, (now - sampledAtMs) / 1000) : NaN;
        if (
            advancingRef.current &&
            Number.isFinite(ageSec) &&
            ageSec <= PLAYHEAD_SAMPLE_FRESH_SEC
        ) {
            // 新鲜的轮询采样：锚定采样时刻并外推到当前时刻。anchor 的
            // (值, 时刻) 二元组与 {synced@sampledAt} 表示同一条外推直线，
            // 因此 React 提交延迟与后续 RAF 帧的插值都不会引入滞后或回跳
            // （采样值本身已含 reducer 侧的往返时延外推，见
            // sessionSlice syncPlaybackState.fulfilled）。
            const extrapolatedSec = syncedPlayheadSec + ageSec;
            syncAnchorRef.current = {
                playheadSec: extrapolatedSec,
                timestampMs: now,
            };
            visualPlayheadSecRef.current = extrapolatedSec;
            onFrameRef.current?.(extrapolatedSec);
        } else {
            // 非轮询写入（seek / stop / 粘贴……）或采样过旧：硬复位到提交值。
            syncAnchorRef.current = {
                playheadSec: syncedPlayheadSec,
                timestampMs: now,
            };
            visualPlayheadSecRef.current = syncedPlayheadSec;
            onFrameRef.current?.(syncedPlayheadSec);
        }
    }, [syncedPlayheadSec, syncedAtMs]);

    useEffect(() => {
        if (!isTransportAdvancing) return;

        // RAF（重）启动时的锚点陈旧防护：锚点只在 playheadSec 写入时刷新
        // （轮询采样 / seek / 停止对齐），而本 effect 在 isTransportAdvancing
        // 翻转时重启 —— 该翻转可能发生在 playheadSec 长时间未变的窗口之后
        // （自动暂停后再次播放、阻塞式预渲染窗口、暂停后继续）。此时锚点的
        // 时刻停在翻转之前，直接按 1x 外推会把整段"传输未前进"的空闲时长
        // 一次性加到光标上（向前飞跃数秒），直到下一个轮询采样到达才弹回
        // —— 反复播放/暂停/停止时可见的光标跳变。引擎未前进期间位置是
        // 真实冻结的：把锚点重置到当前视觉位置、时刻重置为现在，视觉连续
        // （不跳变）且不携带外推债务。锚点新鲜（正常播放中翻转）时不做
        // 任何事，保持逐样本精确的既有插值。
        const nowMs = performance.now();
        const anchorAgeSec = (nowMs - syncAnchorRef.current.timestampMs) / 1000;
        if (
            !Number.isFinite(anchorAgeSec) ||
            anchorAgeSec < 0 ||
            anchorAgeSec > PLAYHEAD_SAMPLE_FRESH_SEC
        ) {
            syncAnchorRef.current = {
                playheadSec: visualPlayheadSecRef.current,
                timestampMs: nowMs,
            };
        }

        let rafId = 0;

        const tick = (timestampMs: number) => {
            const elapsedSec = (timestampMs - syncAnchorRef.current.timestampMs) / 1000;
            const nextPlayheadSec = Math.max(
                syncAnchorRef.current.playheadSec,
                syncAnchorRef.current.playheadSec + elapsedSec,
            );
            visualPlayheadSecRef.current = nextPlayheadSec;
            onFrameRef.current?.(nextPlayheadSec);
            rafId = requestAnimationFrame(tick);
        };

        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [isTransportAdvancing]);

    return visualPlayheadSecRef;
}
