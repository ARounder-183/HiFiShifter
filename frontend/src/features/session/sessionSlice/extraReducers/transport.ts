import { createAction } from "@reduxjs/toolkit";
import { isPluginMode } from "../../../../services/hostCapabilities";
import { playOriginal, stopAudioPlayback, syncPlaybackState } from "../../thunks/transportThunks";
import { setPending, setRejected } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

/**
 * 一条宿主传输采样（轮询与推送共用）。
 *
 * 【为什么要两个来源】轮询是"请求-响应"：一次慢回执、一次在途去重、一次乱序丢弃，
 * 任何一环都会变成用户看得见的一次停顿。推送由宿主侧的视图定时器产生，到达即代表
 * 此刻，没有往返时延可丢。轮询保留为看门狗（400 ms），推送负责让光标跟手。
 */
export interface TransportSample {
    ok?: boolean;
    is_playing?: boolean;
    waiting_for_render?: boolean;
    target?: string | null;
    base_sec?: number;
    position_sec?: number;
    duration_sec?: number;
    host_authoritative?: boolean;
    /** 这个读数是谁给的：host / block / seek / capture / offline / stale（见后端 Provenance）。 */
    position_source?: string;
    /** 同 position_source；显式命名，供前端按“捕获时钟”分支处理。 */
    transport_source?: string;
    position_age_ms?: number;
    seek_pending?: boolean;
    /** 单调读数序号：推送与轮询共用的乱序判据（见后端 `TransportView::sample_seq`）。 */
    sample_seq?: number;
}

/** 轮询采样携带的派发时刻（推送没有）。 */
export interface TransportSampleMeta {
    epoch?: number;
    dispatchedAtMs?: number;
}

/**
 * 推送通道的 action。
 *
 * 【为什么 action 定义在这里】它在**本文件**的 extraReducers 里被消费，而
 * `sessionSlice.ts` 又 import 了本文件 —— 反过来 import 会形成环。定义在消费它的
 * 同一个模块里，环就不存在。
 */
export const transportPushed = createAction<TransportSample>("session/transportPushed");

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
        .addCase(playOriginal.rejected, (state, action) => {
            setRejected(state, action);
        })

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
            // 【为什么必须在这里清】用户按下暂停/停止的同一拍就要冻结视觉播放头。
            // 只清 isPlaying 不够：宿主分支的 isTransportAdvancing 还看 positionAdvancing，
            // 而它要等下一条轮询采样才会翻 false —— 中间那一个 IPC 周期就是
            // “光标还要往前跑一段才停”的来源。
            state.runtime.positionAdvancing = false;
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
            applyTransportSample(
                state,
                action.payload as TransportSample,
                action.meta.arg as TransportSampleMeta | undefined,
                performance.now(),
            );
        })
        .addCase(transportPushed, (state, action) => {
            // 推送没有"派发时刻"：它由宿主侧的视图定时器产生，到达即代表此刻。
            applyTransportSample(
                state,
                action.payload as TransportSample,
                undefined,
                performance.now(),
            );
        });
}

/**
 * 把**一条**采样应用到会话状态上。
 *
 * 【为什么抽成纯函数】轮询与推送是两条通道、一份语义。两份实现迟早会漂移，而漂移
 * 的形态是"同一个宿主状态，两条通道画出两条不同的光标"。
 */
function applyTransportSample(
    state: SessionState,
    payload: TransportSample,
    arg: TransportSampleMeta | undefined,
    sampledNowMs: number,
): void {
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
    // 【为什么只对独立 App 生效】那里的"阻塞式前台预渲染"会让引擎位置暂时冻结，
    // 采样确实是陈旧的；插件形态的渲染**不改变宿主位置**，把采样一起吞掉只会
    // 让光标停住（"跳到位之后卡一会儿"）。
    if (!isPluginMode() && state.playbackBlockingRenderActive) {
        return;
    }
    // 乱序防护：轮询在派发时携带当时的传输纪元，响应到达时若纪元
    // 已推进（期间发生过 播放/停止/seek 或 isPlaying 翻转），说明
    // 该快照采样于旧的传输状态 —— 按新语义应用会把 isPlaying 复活/
    // 掐灭，并把光标改写成旧播放位置或 0（反复播放/暂停/停止时
    // 播放光标跳变的主要根源）。直接丢弃，等下一次轮询读取新态。
    // arg 缺省（推送通道、测试夹具）时不做乱序拦截：它没有"派发时刻"，也就没有迟到。
    if (arg && typeof arg.epoch === "number" && arg.epoch !== state._transportEpoch) {
        return;
    }
    // 【乱序防护的第二种形态：读数序号】纪元判的是"传输态有没有变"，它拦不住
    // "同一场播放里两条通道的先后"。推送与轮询现在都带 `sample_seq`（后端在产生
    // 读数的那一刻分配），序号**更小**的一律丢弃 —— 于是晚到的旧读数不会再拽回光标。
    //
    // 【为什么判据是"严格更小"而不是"小于等于"】`sample_seq` 标识的是**后端发布了
    // 哪一次读数**，而一条读数里还带着会随时间变化的派生事实：`position_age_ms` 与
    // 由它决定的 `is_playing`。同一个序号可以被读多次，每次读到的派生事实不同 ——
    // 典型场景正是 REAPER 的 Item（take FX）：宿主停播后**不再送音频块**，序号就冻住
    // 了，而"播放位已过期"恰恰是在后续这些同序号读数里才表达出来的。用 `<=` 把它们
    // 当重复丢掉，界面就永远停在"宿主还在播"，光标按 1x 无限前漂（"播放时光标能动、
    // 但不会停止"）。同序号的两条读数不可能乱序（它们是同一份发布的重复读取），
    // 后到的那条只会更新，不会把光标拽回旧位置。
    const sampleSeq = typeof payload.sample_seq === "number" ? payload.sample_seq : null;
    if (sampleSeq !== null && sampleSeq < state.runtime.lastTransportSeq) {
        return;
    }
    if (sampleSeq !== null) {
        state.runtime.lastTransportSeq = sampleSeq;
    }
    // 【为什么不再看 `_pluginSeekRequestId`】"这次定位落地了没有"现在是**后端**的
    // 事实（`SeekIntent` 窗口，见 render/transport.rs）。前端替权威猜"再压多久"
    // 只会在命令回执早于宿主落地时把旧读数放进来 —— 那正是"跳过去又跳回来"。

    const nextIsPlaying = Boolean(payload.is_playing);
    const nextTarget = payload.target ?? null;
    const nextPositionSec = payload.position_sec ?? 0;
    const nextDurationSec = payload.duration_sec ?? 0;
    // "传输层原地等待渲染"（Case A + Case B 统一）：后台渲染进行中
    // 遇到未渲染窗口（起播位置落在其中，或播放途中推进到其开头）
    // 时引擎保持 is_playing=true 但位置冻结，等待渲染完成后自动
    // 开始/继续播放。旧后端不带此字段 → 视为 false。
    const nextWaitingForRender = nextIsPlaying && Boolean(payload.waiting_for_render);
    // 【不再有"起播窗口"】此前这里用一条 500ms / 0.05s 的规则丢弃"看起来
    // 不像起播点"的采样，用来遮住"引擎从别处起播"的症状。起播点已收敛到
    // 引擎的绝对位置，这类采样不再存在；保留那条规则只会把真实状态挡在外面
    // （表现是"光标钉住、音频已经走了"）。停止态的陈旧读数仍由下方
    // `frozenSec > EPS_SEC` 与 `stale` / `localWriteRecent` 三道闸门挡住。
    // 轮询节奏的唯一依据（见 `runtime.hostAuthoritative`）：它回答"这份位置
    // 是不是宿主给的"，与"谁认为在播"无关。
    state.runtime.hostAuthoritative = Boolean(payload.host_authoritative);
    // 【读数年龄必须落到状态里】后端一直在算 `position_age_ms`，而全仓**零引用**。
    // 它回答的是"这份读数有多旧"，正是"宿主停止后光标还在走"要用的那道闸门
    // （见 `selectIsTransportAdvancing`）。
    state.runtime.positionAgeMs =
        typeof payload.position_age_ms === "number" ? payload.position_age_ms : null;
    state.runtime.transportSource = payload.transport_source ?? payload.position_source ?? "host";

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

    // 【为什么在"要不要更新"之前算】它是一条**派生读数**，不是更新决策的一部分：
    // 宿主反复报同一个位置时（冻结），它必须翻成 false —— 否则视觉插值会一直按
    // 1x 外推，光标在冻结的画面上继续前漂，比不动更糟。
    const positionAdvancing =
        nextIsPlaying && nextPositionSec > state.runtime.playbackPositionSec + EPS_SEC;
    const advancingChanged = positionAdvancing !== state.runtime.positionAdvancing;

    const shouldUpdatePlaybackFields =
        nextIsPlaying !== state.runtime.isPlaying ||
        nextTarget !== state.runtime.playbackTarget ||
        state.runtime.playbackWaitingForRender ||
        advancingChanged ||
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
    state.runtime.positionAdvancing = positionAdvancing;

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
    // 残值没有资格改光标：`stale` 说的是"这个位置是上一次观测"，不是"现在在哪"。
    const stale = payload.position_source === "stale";
    const localWriteRecent = !nextIsPlaying && sampledNowMs - state.playheadLocalAtMs < 300;
    if (!stale && !localWriteRecent && (nextIsPlaying || nextplayheadSec > 0)) {
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
}
