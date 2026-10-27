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
    /** 最新采样值（视觉位置朝它纠偏，见 `SLEW_SEC_PER_SEC`）。 */
    const targetSecRef = useRef(syncedPlayheadSec);
    const onFrameRef = useRef(onFrame);
    const hostControlled = isPluginMode();

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
        /*
         * 【为什么宿主受控时不再要求"采样新鲜"】
         *
         * `PLAYHEAD_SAMPLE_FRESH_SEC` 存在的理由是：独立 App 的外推是**无界**的，
         * 一个陈旧采样会让光标自行前漂数秒，所以必须靠"新鲜度"决定要不要外推。
         *
         * 插件模式的外推是**有界**的（`projectVisualPlayhead` 把它夹在 100 ms 内），
         * 那个理由不成立；而继续套用同一条闸门反而制造了缺陷：插件命令的一次往返
         * 超过 250 ms 时，这一支会走下面的"硬复位"，把光标拉回`syncedPlayheadSec`
         * —— 那是**命令派发时刻**的位置，不是现在的位置。于是每收到一个采样就先
         * 往回跳一个往返时延，再被下一帧外推追回去：用户看到的就是"偏移 + 随机闪"。
         *
         * 所以判据改成：正在前进、且（宿主受控 或 采样新鲜）就外推。
         * 宿主受控时交给那个 100 ms 上限兜底，不需要也不应该再看新鲜度。
         */
        const hasAge = Number.isFinite(ageSec);
        const fresh = hasAge && ageSec <= PLAYHEAD_SAMPLE_FRESH_SEC;
        if (advancingRef.current && (hostControlled || fresh)) {
            // 新鲜的轮询采样：锚定采样时刻并外推到当前时刻。anchor 的
            // (值, 时刻) 二元组与 {synced@sampledAt} 表示同一条外推直线，
            // 因此 React 提交延迟与后续 RAF 帧的插值都不会引入滞后或回跳
            // （采样值本身已含 reducer 侧的往返时延外推，见
            // sessionSlice syncPlaybackState.fulfilled）。
            /*
             * 【为什么这里传 false（不封顶）】
             *
             * 那个 100 ms 上限的语义是"不要对未来**推测**太远"，它约束的是
             * rAF 帧里 `elapsedSec` 那种**向前**的外推。
             *
             * 而这里的 `ageSec = now − sampledAtMs` 是**已经过去的、被测量到的**
             * 时间：采样确实发生在那么久以前，把它补上不是推测，是还原。
             *
             * 此前这里也传了 `hostControlled`，于是每次采样只补 100 ms，剩下的
             * 往返时延**永久丢失**——光标系统地落后于它自己报出的时间（"偏移"），
             * 而下一个采样又把差值补回来一部分（"随机闪"）。
             */
            const extrapolatedSec = catchUpSampledPlayhead(syncedPlayheadSec, hasAge ? ageSec : 0);
            /*
             * 【为什么不再把锚点直接跳到采样值】那会把"采样通道的抖动"变成"光标的跳变"：
             * 一次迟到 300 ms 的采样会让光标先冻在原地、再瞬间前进 300 ms。
             *
             * 现在锚点只**追**采样值：每帧最多走 `SLEW_SEC_PER_SEC × dt`（25% 的速度差），
             * 于是 300 ms 的误差在 "1.2 s 内平滑吃掉，用户看不到跳变。真正的位置跳变
             * （seek / 循环折返）仍走硬复位 —— 那是"位置真的变了"，平滑它反而是错的。
             */
            const jumpSec = Math.abs(extrapolatedSec - visualPlayheadSecRef.current);
            if (!Number.isFinite(jumpSec) || jumpSec > SEEK_RESET_SEC) {
                syncAnchorRef.current = {
                    playheadSec: extrapolatedSec,
                    timestampMs: now,
                };
                targetSecRef.current = extrapolatedSec;
                visualPlayheadSecRef.current = extrapolatedSec;
                onFrameRef.current?.(extrapolatedSec);
            } else {
                targetSecRef.current = extrapolatedSec;
            }
        } else {
            // 非轮询写入（seek / stop / 粘贴……）或采样过旧：硬复位到提交值。
            syncAnchorRef.current = {
                playheadSec: syncedPlayheadSec,
                timestampMs: now,
            };
            visualPlayheadSecRef.current = syncedPlayheadSec;
            onFrameRef.current?.(syncedPlayheadSec);
        }
    }, [syncedPlayheadSec, syncedAtMs, hostControlled]);

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
            // 1) 按 1x 前进。**不设上限**：宿主在播放是已知事实（`isTransportAdvancing`），
            //    采样只是它的估计值 —— 用估计值的迟到去冻结一个已知在前进的量是本末倒置。
            const advanced = projectVisualPlayhead(
                syncAnchorRef.current.playheadSec,
                elapsedSec,
                false,
            );
            // 2) 朝采样值纠偏，每帧最多走 `SLEW_SEC_PER_SEC × dt`。
            const target = targetSecRef.current;
            const maxStep = SLEW_SEC_PER_SEC * Math.max(0, elapsedSec);
            const error = target - advanced;
            const nextPlayheadSec = advanced + Math.max(-maxStep, Math.min(maxStep, error));
            syncAnchorRef.current = { playheadSec: nextPlayheadSec, timestampMs };
            visualPlayheadSecRef.current = nextPlayheadSec;
            onFrameRef.current?.(nextPlayheadSec);
            rafId = requestAnimationFrame(tick);
        };

        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [isTransportAdvancing, hostControlled]);

    return visualPlayheadSecRef;
}
