import { isPluginMode } from "../../../../services/hostCapabilities";
import { playOriginal, stopAudioPlayback, syncPlaybackState } from "../../thunks/transportThunks";
import { setPending, setRejected } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

export function registerTransport(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(playOriginal.pending, (state) => {
            setPending(state, "Playing original...");
            // 播放操作开始即交接播放头所有权：在途播放轮询（采样于本次
            // 播放序列之前，可能携带停止间隙的 0 位置或旧播放位置）的
            // 迟到响应按乱序丢弃，防止光标被拽回旧位置/0。
            state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
        })
        .addCase(playOriginal.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                clipId?: string | null;
                anchorSec?: number;
                noop?: boolean;
                host_request?: boolean;
            };
            // 已在播放时的重复"播放"触发：thunk 已完全 no-op（未 seek、
            // 未调用后端）——不得重置任何传输状态（等待标志 / 位置报告 /
            // 纪元），否则会解除等待期的光标冻结并造成跳变。
            if (payload.noop) {
                return;
            }
            const ok = Boolean(payload.ok);
            // ARA的void请求只能确认已送达；实际播放态由宿主process时钟轮询确认。
            if (!payload.host_request) {
                state.runtime.isPlaying = ok;
                state.runtime.playbackTarget = ok ? "original" : null;
            }
            // 新播放会话：清除上一场遗留的"起点等待"标志（引擎侧由
            // set_playing 同步重置，此处保持两侧一致）。
            state.runtime.playbackWaitingForRender = false;
            // 新一次播放会话开始：引擎位置报告（playbackPositionSec）
            // 已是上一场播放/自动暂停的陈旧残留，必须归零。它驱动
            // isTransportAdvancing 代理（isPlaying && positionSec > ε），
            // 若保留旧值， fulfilled 翻转 isPlaying 的瞬间 advancing 即为
            // 真 —— 视觉插值从陈旧锚点按 1x 外推整个暂停时长，直到首个
            // 轮询采样到达才弹回（后台预渲染自动暂停后再次播放时的
            // 典型光标跳变）。归零后 advancing 保持 false，光标原地等待
            // 首个轮询采样建立新锚点。
            state.runtime.playbackPositionSec = 0;
            state.playbackClipId = ok ? (payload.clipId ?? null) : null;
            // Store the playhead position at which playback started,
            // so Play/Stop can restore it.
            state.playbackAnchorSec = ok ? (payload.anchorSec ?? 0) : 0;
            if (ok) {
                // 传输层接管（引擎随即开始播放/进入预渲染等待）后推进纪元：
                // 播放命令序列期间（set_transport 与引擎真正 set_playing 之间）
                // 派发的轮询采样到"尚未播放"的引擎态，其迟到响应不得在
                // isPlaying 已翻转之后应用（否则会把光标写成 0/旧位置并
                // 短暂掐灭播放状态）。
                state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
            }
            state.status = payload.host_request
                ? "Playback requested"
                : ok
                  ? "Playing original"
                  : "Play original failed";
        })
        .addCase(playOriginal.rejected, setRejected)

        .addCase(stopAudioPlayback.pending, (state) => {
            setPending(state, "Stopping audio...");
            // 捕获"翻转前"的真实播放状态供 thunk 判定 wasPlaying：
            // pending 先于 thunk 体运行，thunk 里读到的 runtime.isPlaying
            // 已被本 reducer 置 false；停止是否真的打断了播放（决定
            // Stop 是否恢复锚点）只能在此处留下快照。旧实现用
            // playheadSec/anchor 差值推断，会在"暂停后按 Stop"、"只 seek
            // 过就按 Stop"时误判，把光标恢复到 0（跳变）。
            state._stopInterruptedPlayback = state.runtime.isPlaying;
            // Pause UI immediately on user stop/pause command; backend sync will
            // confirm the final transport state shortly after.
            state.runtime.isPlaying = false;
            state.runtime.playbackTarget = null;
            // 停止/暂停操作开始即交接播放头所有权：在途播放轮询（采样于
            // 本次停止之前、携带旧播放位置）的迟到响应按乱序丢弃 —— 否则
            // 它会把 isPlaying 复活并把光标推回旧播放位置，随后的下一次
            // 轮询发现引擎已停（base/position 归零）再经跃迁分支把光标
            // 写成 0。这是后台预渲染负载下 IPC 延迟放大后最明显的跳变。
            state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
        })
        .addCase(stopAudioPlayback.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                restoreAnchor?: boolean;
                wasPlaying?: boolean;
                anchorSec?: number | null;
                stopped_at_sec?: number | null;
            };
            // If restoreAnchor is set (Play/Stop action), restore playhead to anchor position
            if (
                payload.restoreAnchor &&
                payload.wasPlaying &&
                typeof payload.anchorSec === "number" &&
                Number.isFinite(payload.anchorSec)
            ) {
                state.playheadSec = Math.max(0, payload.anchorSec);
            } else if (
                typeof payload.stopped_at_sec === "number" &&
                Number.isFinite(payload.stopped_at_sec)
            ) {
                // 暂停：把视觉光标对齐到引擎的实际停止位置。轮询存在至多
                // 一个周期（~33ms）+ 往返的滞后，最后一次采样后音频仍在
                // 前进；后端记录的暂停点是这个精确位置。若不同步，视觉
                // 位置会落后于真实位置，后续任何编辑回灌快照都会让光标
                // 再次右跳。
                state.playheadSec = Math.max(0, payload.stopped_at_sec);
            }
            state.runtime.isPlaying = false;
            // 等待中的用户停止立即生效：清除"起点等待"标志（引擎侧
            // handle_stop 同步清除，此处保持两侧一致）。
            state.runtime.playbackWaitingForRender = false;
            state.runtime.playbackTarget = null;
            state.runtime.playbackPositionSec = 0;
            state.runtime.playbackDurationSec = 0;
            state.playbackClipId = null;
            // ★ 保留锚点：暂停只结束"进行中的播放"，不清除"本次播放的起始
            // 位置"。旧行为在暂停时把锚点清零，让"最近一次播放起点"信息
            // 丢失（配合旧 wasPlaying 启发式会把 播放→暂停→停止 的光标
            // "恢复"到工程开头）。保留后 Stop 始终能回到最近一次播放的
            // 起点（与播放中按 Stop 的行为一致，也是 REAPER 语义）；是否
            // 恢复由 thunk 的 wasPlaying（_stopInterruptedPlayback 快照）
            // 把关，已停止后的重复 Stop 不会再次恢复，天然幂等。
            state.status = payload.ok ? "Audio stopped" : "Stop audio failed";
            // 权威停止已应用，推进传输纪元：丢弃停止命令序列期间派发的
            // 轮询响应（其采样可能来自停止前后的任意一侧）。
            state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
        })
        .addCase(stopAudioPlayback.rejected, (state, action) => {
            // `quiet`：调用方只是"尽力停一下"（基准测试），宿主不提供 ARA 播放
            // 控制时不该把"停不了"变成用户可见的错误 —— 那与它要做的事无关。
            // 其余情况（用户按 Stop）照旧报错。
            if ((action.meta.arg as { quiet?: boolean } | undefined)?.quiet) {
                state.busy = false;
                state.status = "Ready";
                return;
            }
            setRejected(state, action);
        })

        .addCase(syncPlaybackState.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                is_playing?: boolean;
                waiting_for_render?: boolean;
                target?: string | null;
                base_sec?: number;
                position_sec?: number;
                duration_sec?: number;
                host_authoritative?: boolean;
            };
            if (!payload.ok) {
                return;
            }
            // 阻塞式前台预渲染（target="original"）active 期间，引擎尚未
            // 真正进入 playing，此刻的传输态是上一场播放/seek 的陈旧值，
            // 一概拒采。渲染完成事件（active=false）先于渲染期间派发的
            // 轮询响应被处理（同一 IPC 通道按序送达），所以该镜像可靠：
            // 没有它，短渲染（全缓存命中）完成后实时轮询已推进光标，
            // 迟到的"未播放"采样（纪元未变、乱序防护拦不住）会经跃迁
            // 分支把光标拽回播放起点、掐灭 isPlaying 并让轮询停摆 ——
            // 光标卡死而音频继续播。
            // 注意读取的是 playbackBlockingRenderActive（按 target 隔离
            // 的前台渲染镜像）：后台渲染的完成事件不得提前关闭本窗口。
            if (state.playbackBlockingRenderActive) {
                return;
            }
            // 乱序防护：轮询在派发时携带当时的传输纪元，响应到达时若纪元
            // 已推进（期间发生过 播放/停止/seek 或 isPlaying 翻转），说明
            // 该快照采样于旧的传输状态 —— 按新语义应用会把 isPlaying 复活/
            // 掐灭，并把光标改写成旧播放位置或 0（反复播放/暂停/停止时
            // 播放光标跳变的主要根源）。直接丢弃，等下一次轮询读取新态。
            // arg 缺省（如测试夹具）时保持旧行为不拦截。
            const arg = action.meta.arg as { epoch?: number; dispatchedAtMs?: number } | undefined;
            if (arg && typeof arg.epoch === "number" && arg.epoch !== state._transportEpoch) {
                return;
            }
            if (isPluginMode() && state._pluginSeekRequestId !== null) return;

            const nextIsPlaying = Boolean(payload.is_playing);
            const nextTarget = payload.target ?? null;
            const nextPositionSec = payload.position_sec ?? 0;
            const nextDurationSec = payload.duration_sec ?? 0;
            // "传输层原地等待渲染"（Case A + Case B 统一）：后台渲染进行中
            // 遇到未渲染窗口（起播位置落在其中，或播放途中推进到其开头）
            // 时引擎保持 is_playing=true 但位置冻结，等待渲染完成后自动
            // 开始/继续播放。旧后端不带此字段 → 视为 false。
            const nextWaitingForRender = nextIsPlaying && Boolean(payload.waiting_for_render);

            // 原地等待采样：保持播放态、光标冻结 —— 不做时延外推（位置
            // 并未前进，外推会让光标按 IPC 时延缓慢漂移）、不改写
            // playheadSec，并**归零 playbackPositionSec**：它是
            // isTransportAdvancing 代理（isPlaying && positionSec > ε）的
            // 输入，Case A 进入等待时它停留在冻结前最后一个前进采样值
            // （>0）—— 不归零会让视觉插值 RAF 以 1x 从冻结前锚点外推
            // 整个等待时长（无界前漂），恢复采样到达时光标大幅回跳
            // （往复跳动的根源）。归零后 advancing 翻 false、RAF 停止，
            // 光标冻结在当前视觉位置；渲染完成后的首个前进采样经正常
            // 分支以新鲜锚点恢复推进。
            if (nextWaitingForRender) {
                if (!state.runtime.playbackWaitingForRender) {
                    state.runtime.playbackWaitingForRender = true;
                }
                if (state.runtime.playbackPositionSec !== 0) {
                    state.runtime.playbackPositionSec = 0;
                }
                return;
            }

            // 采样外推：payload 的 base+position 是"后端执行命令那一刻"
            // 的引擎位置；从派发到本 reducer 处理（≈IPC 到达）的往返时延
            // 内引擎继续前进。把它外推到处理时刻才是此刻的真实可听位置
            // —— 否则视觉插值会系统性滞后一个往返时延（后台预渲染负载下
            // 可达数十~数百毫秒），且时延抖动会让光标在每次采样到达时
            // 反复回跳、暂停时前跳、播放时后跳。外推后"采样值=到达时刻
            // 位置"与锚定时刻严格配对，插值逐样本精确（见
            // useVisualPlayhead 的锚定注释）。
            // 引擎已冻结（停止/自动暂停）时不外推：冻结位置是真实停止点。
            const sampledNowMs = performance.now();
            const dispatchedAtMs = arg?.dispatchedAtMs;
            const latencySec =
                // 宿主值在回复时读取；排队等待合成不是采样年龄，不能把整段RTT加到DAW位置。
                nextIsPlaying && !payload.host_authoritative && typeof dispatchedAtMs === "number"
                    ? Math.max(0, (sampledNowMs - dispatchedAtMs) / 1000)
                    : 0;

            // 0.5ms 阈值：避免轮询带来的浮点抖动导致无意义的 Redux 更新
            const EPS_SEC = 0.0005;

            let nextplayheadSec = state.playheadSec;
            if (payload.host_authoritative && !nextIsPlaying) {
                // DAW停播时的seek也有真实位置；独立app仍保留本地编辑游标。
                nextplayheadSec = Math.max(0, (payload.base_sec ?? 0) + nextPositionSec);
            } else if (nextIsPlaying) {
                const absSec = (payload.base_sec ?? 0) + nextPositionSec + latencySec;
                nextplayheadSec = Math.max(0, absSec);
            } else if (state.runtime.isPlaying) {
                // 播放→停止跃迁（音频自然结束 / 引擎等待重渲染时自动暂停）：
                // 引擎的 position 冻结在真实停止点，而本地光标还停留在最后
                // 一次轮询采样（略落后）。对齐一次，保证视觉位置 = 真实
                // 位置；此后的轮询（is_playing=false 分支）不再改写光标。
                //
                // 防呆：引擎被用户停止（handle_stop）后 base/position 一并
                // 归零。竞态下（阻塞预渲染窗口溜进的轮询、乱序响应）这个
                // 0/0 不代表真实停止点 —— 写 0 会把光标拽到工程开头。真实
                // 冻结点（自动暂停 / 自然结束）总是 > 0；从 0.0 起播且首个
                // 块即自动暂停时冻结点本就是 0，光标已在该处，不写亦无偏差。
                const frozenSec = Math.max(0, (payload.base_sec ?? 0) + nextPositionSec);
                if (frozenSec > EPS_SEC) {
                    nextplayheadSec = frozenSec;
                }
            }

            const shouldUpdatePlaybackFields =
                nextIsPlaying !== state.runtime.isPlaying ||
                nextTarget !== state.runtime.playbackTarget ||
                state.runtime.playbackWaitingForRender ||
                Math.abs(nextPositionSec - state.runtime.playbackPositionSec) > EPS_SEC ||
                Math.abs(nextDurationSec - state.runtime.playbackDurationSec) > EPS_SEC ||
                Math.abs(nextplayheadSec - state.playheadSec) > EPS_SEC;

            if (!shouldUpdatePlaybackFields) {
                // No state change needed.
                return;
            }

            if (nextIsPlaying !== state.runtime.isPlaying) {
                // 应用 isPlaying 跃迁的同时推进传输纪元：同一窗口内其他
                // 在途轮询（采样于跃迁另一侧，如 200ms 渲染完成延迟同步与
                // 30Hz 轮询并发）的迟到响应按乱序丢弃，避免把刚应用的
                // 跃迁再次翻转/回退。
                state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
            }

            state.runtime.isPlaying = nextIsPlaying;
            state.runtime.playbackTarget = nextTarget;
            // 走到正常应用分支说明已不在"起点等待渲染"状态（等待采样在
            // 上游被拦截）：清除标志。首个前进采样同时携带新位置与时延
            // 外推，视觉插值以新鲜锚点从起播位置自然恢复推进。
            state.runtime.playbackWaitingForRender = false;
            state.runtime.playbackDurationSec = nextDurationSec;

            // ★ 光标写入规则：播放中采样照常驱动；停止态采样**仅当引擎给出
            // > 0 的冻结点**（自然结束 / 冻结在真实位置）时才对齐光标 ——
            // handle_stop 会把引擎位置归零，若把该 0 应用进来会把用户光标
            // 拽到工程开头（停止时的光标定位由 stop 路径的 stopped_at /
            // 锚点回写负责）。
            // 该规则同时使"常驻低频看门狗轮询"（见 App.tsx）安全：未播放时
            // 采样通常读到 0，只用于纠正 isPlaying 镜像，不触碰光标。
            // 【为什么停止态要避让本地写入】见 `playheadLocalAtMs` 的注释：
            // 拖拽/seek 之后宿主光标还没跟上，这一拍读到的旧值会把光标拽回去。
            // 300 ms 覆盖"拖拽过程中"与"seek 刚发出"两个窗口；播放态不受影响。
            const localWriteRecent = !nextIsPlaying && sampledNowMs - state.playheadLocalAtMs < 300;
            if (!localWriteRecent && (nextIsPlaying || nextplayheadSec > 0)) {
                state.runtime.playbackPositionSec = nextPositionSec;
                if (nextplayheadSec !== state.playheadSec) {
                    state.playheadSec = nextplayheadSec;
                    // 记录采样时刻：视觉插值据此把外推直线锚定在该时刻
                    // （而非 React 提交时刻），使采样值与锚定时刻严格配对。
                    state.playheadSampledAtMs = sampledNowMs;
                }
            }
            if (!nextIsPlaying) {
                state.playbackClipId = null;
            }
        });
}
