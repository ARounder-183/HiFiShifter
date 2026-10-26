import { isPluginMode } from "../../../../services/hostCapabilities";
import type { TimelineState, TrackSummaryResult } from "../../../../types/api";
import {
    addTrackRemote,
    fetchSelectedTrackSummary,
    moveTrackRemote,
    removeTrackRemote,
    createHostTrackRemote,
    selectClipRemote,
    selectTrackRemote,
    setProjectLengthRemote,
} from "../../thunks/timelineThunks";
import { seekPlayhead, updateTransportBpm } from "../../thunks/transportThunks";
import { setTrackStateRemote } from "../../thunks/trackThunks";
import { markProjectDirty } from "../../sessionDirtyState";
import { resolveTrackIdForClipSelection } from "../../selectionFocus";
import { parseSelectClipRemoteArg } from "../helpersTakes";
import {
    applyOptimisticTrackState,
    applyTimelineTracksOnly,
    restoreClipPitchCurves,
    applyAddedTrack,
    clamp,
} from "../helpersCore";
import { applyTimelineState } from "../helpersTimeline";
import type { AddedTrackPayload } from "../helpersCore";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

export function registerTrack(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(selectClipRemote.pending, (state, action) => {
            const { clipId, preserveTrackFocus } = parseSelectClipRemoteArg(action.meta.arg);
            state.selectedClipId = clipId;
            // 与 setSelectedClip 一致：点选 Clip 即把复制/剪切路由的上下文
            // 仲裁切换到 Clip 侧，否则上一次参数选区遗留的 "param" 会让
            // Ctrl+C/Ctrl+X 错误路由到参数编辑器。
            state.selectionContext = "clips";
            if (clipId) {
                const nextTrackId = resolveTrackIdForClipSelection({
                    currentTrackId: state.selectedTrackId,
                    clips: state.clips,
                    clipId,
                    preserveTrackFocus,
                });
                if (nextTrackId !== state.selectedTrackId) {
                    state.selectedTrackId = nextTrackId;
                }
            }
        })

        .addCase(selectClipRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                __preserveTrackFocus?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            const currentSelectedTrackId = state.selectedTrackId;
            state.selectedClipId = payload.selected_clip_id;
            if (payload.__preserveTrackFocus) {
                state.selectedTrackId = currentSelectedTrackId;
            } else if (payload.selected_track_id !== undefined) {
                state.selectedTrackId = payload.selected_track_id;
            }
        })

        .addCase(setTrackStateRemote.pending, (state, action) => {
            applyOptimisticTrackState(state, action.meta.arg);
        })

        .addCase(setTrackStateRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            // 保留当前播放头位置，避免 mute/solo 操作时播放头跳变
            const currentPlayheadSec = state.playheadSec;
            // 保留 paramsEpoch 和 clipPitchCurves，避免触发钢琴窗音高曲线重新渲染
            const currentParamsEpoch = state.paramsEpoch;
            // 浅拷贝快照：restoreClipPitchCurves 需遍历原始条目重建 map。
            const currentClipPitchCurves = { ...state.clipPitchCurves };
            applyTimelineTracksOnly(state, payload);
            state.playheadSec = currentPlayheadSec;
            state.paramsEpoch = currentParamsEpoch;
            restoreClipPitchCurves(state, currentClipPitchCurves);
        })
        .addCase(setTrackStateRemote.rejected, (state, action) => {
            // 后端拒绝：保留乐观值（用户意图可见），给出非致命反馈 ——
            // 与 setClipStateRemote.rejected 的 clip 侧策略一致。
            applyOptimisticTrackState(state, action.meta.arg);
            state.status = "Track edit rejected";
        })

        .addCase(updateTransportBpm.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                bpm?: number;
            } & Partial<TimelineState>;
            if (!payload.ok) {
                return;
            }
            // 与 Tempo Map 变化点一致的 BPM 范围（10-960）。
            //
            // 【必须忽略"迟到"的回声】传输命令经 `enqueueTransportCommand` 串行
            // 执行，滚轮手势期间会积压几十个请求；每个请求的回声都在**乐观值已经
            // 前进之后**才到达。若照单采纳，state.bpm 就会在"最新乐观值"与"若干个
            // 旧回声值"之间反复被拽回 —— 刻度时间（`sec = beat × 60 / bpm`）与网格
            // 因此来回平移，用户看到的就是标尺与网格"抽搐"。
            //
            // 判据与 `seekPlayhead.fulfilled` 完全一致（那里已经写明这条规则）：
            // **仅当前端值仍等于本次请求值**（即没有被更新的请求覆盖）**且后端确实
            // 修正了它**（clamp 之类）时才采纳。
            const requestedBpm = action.meta.arg as number;
            const echoedBpm = clamp(Number(payload.bpm ?? requestedBpm), 10, 960);
            const EPS_BPM = 1e-9;
            if (
                Math.abs(state.bpm - requestedBpm) <= EPS_BPM &&
                Math.abs(echoedBpm - requestedBpm) > EPS_BPM
            ) {
                state.bpm = echoedBpm;
            }
            if (payload.tracks && payload.clips) {
                applyTimelineState(state, payload as TimelineState, { force: true });
            }
        })

        .addCase(seekPlayhead.pending, (state, action) => {
            // seek 交接播放头所有权：seek 之前派发的在途播放轮询采样的是
            // seek 前的引擎位置，其迟到响应会把光标拽回旧位置（先跳回再
            // 被 seek 后的轮询拉回 —— 播放中点击标尺/拖拽时的可见跳变）。
            state._transportEpoch = (Number(state._transportEpoch) || 0) + 1;
            if (isPluginMode()) state._pluginSeekRequestId = action.meta.requestId;
        })
        .addCase(seekPlayhead.rejected, (state, action) => {
            if (isPluginMode() && state._pluginSeekRequestId === action.meta.requestId) {
                state._pluginSeekRequestId = null;
                state._transportEpoch += 1;
            }
        })
        .addCase(seekPlayhead.fulfilled, (state, action) => {
            if (isPluginMode()) {
                if (state._pluginSeekRequestId !== action.meta.requestId) return;
                state._pluginSeekRequestId = null;
                state._transportEpoch += 1;
            }
            const payload = action.payload as {
                ok?: boolean;
                playhead_sec?: number;
            } & Partial<TimelineState>;
            if (!payload.ok) {
                return;
            }
            // 使用请求参数（action.meta.arg）作为可信值，而非后端返回的
            // playhead_sec，因为并发请求时旧响应可能晚于新请求到达，
            // 用旧值覆盖前端已更新的 playheadSec 会导致光标闪烁。
            const requestedSec = action.meta.arg as number;
            const backendSec = Number(payload.playhead_sec ?? requestedSec);
            // 仅当【前端 playheadSec 仍等于本次请求值】（未被更新的请求
            // 覆盖）且后端返回值与请求参数不同（发生了 clamp 之类修正）
            // 时才采纳后端值 —— 缺少前一半条件时，先发请求的迟到响应
            // 会把光标从新位置拖回旧位置。
            const EPS = 0.001;
            if (
                Math.abs(state.playheadSec - requestedSec) <= EPS &&
                Math.abs(backendSec - requestedSec) > EPS &&
                !isPluginMode()
            ) {
                // 后端对位置做了修正（如 clamp），采纳后端值
                state.playheadSec = Math.max(0, backendSec);
            }
            // 否则保持前端已同步设好的 state.playheadSec，不再覆盖
            if (state.runtime.isPlaying) {
                state.playbackAnchorSec = state.playheadSec;
            }
            if (payload.tracks && payload.clips) {
                applyTimelineState(state, payload as TimelineState, { force: true });
            }
        })

        .addCase(addTrackRemote.fulfilled, (state, action) => {
            applyAddedTrack(state, action.payload as AddedTrackPayload);
        })

        // 宿主建轨（插件专属）与插件自己建轨在 UI 上是同一件事：多了一条轨道。
        // 复用同一个 reducer 体，免得两条路径的"标脏 / 交互锁"处理各自漂移。
        .addCase(createHostTrackRemote.fulfilled, (state, action) => {
            applyAddedTrack(state, action.payload as AddedTrackPayload);
        })

        .addCase(removeTrackRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            markProjectDirty(state.project);
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(moveTrackRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(selectTrackRemote.pending, (state, action) => {
            state.selectedTrackId =
                typeof action.meta.arg === "string" ? action.meta.arg : action.meta.arg.trackId;
            // 换轨手势（轨道头 / 时间轴行 / 键盘）即参数编辑器重定向：
            // 剪贴板上下文记为参数侧。
            //
            // 特殊说明：换轨**不**恢复后端记住的 `selected_clip_id`——所有
            // `selectTrackRemote` 调用点都传 `{ applySelectedClip: false }`
            // （见 fulfilled 分支的闸门）。后端的选中记忆是全工程唯一的、
            // 不是每轨一份，因此"恢复该轨道记住的 Clip"并非真实语义。
            state.selectionContext = "param";
        })

        .addCase(selectTrackRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            const currentPlayheadSec = state.playheadSec;
            applyTimelineTracksOnly(state, payload);
            // applySelectedClip: false（点击轨道空白切轨）时不得恢复后端
            // 记住的 selected_clip_id —— 否则异步 fulfilled 会把刚完成的
            // "点击空白取消选中"覆盖回去。
            const applySelectedClip =
                typeof action.meta.arg !== "object" || action.meta.arg.applySelectedClip !== false;
            if (applySelectedClip && payload.selected_clip_id !== undefined) {
                state.selectedClipId = payload.selected_clip_id;
            }
            state.playheadSec = currentPlayheadSec;
        })

        .addCase(setProjectLengthRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(setProjectLengthRemote.fulfilled, (state, action) => {
            // 乱序守卫：见 setClipStateRemote.fulfilled。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(fetchSelectedTrackSummary.fulfilled, (state, action) => {
            const payload = action.payload as TrackSummaryResult | { ok?: false };
            if (!payload.ok) {
                return;
            }
            // 过期响应丢弃：快速连续切轨时，旧轨道的迟到摘要不得覆盖
            // 当前选中轨道。
            if (payload.track_id !== state.selectedTrackId) {
                return;
            }
            state.selectedTrackSummary = {
                trackId: payload.track_id,
                clipCount: payload.clip_count,
                waveformPreview: payload.waveform_preview,
                pitchRange: payload.pitch_range,
            };
        });
}
