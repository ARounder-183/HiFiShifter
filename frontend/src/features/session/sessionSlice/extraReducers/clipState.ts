import { classifyClipEditFailure } from "../../clipEditFailure";
import type { TimelineState } from "../../../../types/api";
import { normalizeChannelMode } from "../../../../utils/channelMode";
import {
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    setClipActiveTakeRemote,
    cycleClipTakesRemote,
    packClipsIntoTakesRemote,
    explodeClipTakesRemote,
    duplicateClipTakeRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipTakeReversedRemote,
    setClipTakeChannelModeRemote,
    addClipTakeFromMediaRemote,
    replaceClipSourceRemote,
    replaceMidiClipDataRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    scanAndConvertFakeStereoRemote,
} from "../../thunks/timelineThunks";
import { applyTimelineState } from "../helpersTimeline";
import { applyOptimisticClipState, applyOptimisticBulkClipState } from "../helpersCore";
import {
    captureTakeRollback,
    applyActiveTakeToFlat,
    takeRequestIsStale,
    clearTakeRollback,
    restoreTakeRollback,
    applyTimelineStatePreservingPlayhead,
    flipTakeSourceWindowForDirection,
} from "../helpersTakes";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import { setRejected } from "./shared";
import type { SessionState } from "../types";

export function registerClipState(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(convertClipsToPitchReferenceRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(convertClipsToPitchReferenceRemote.fulfilled, (state, action) => {
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

        .addCase(updatePitchReferenceRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(updatePitchReferenceRemote.fulfilled, (state, action) => {
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

        .addCase(setClipStateRemote.fulfilled, (state, action) => {
            // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
            // 多选批量松手时若旧请求的中间状态快照迟到，会被这里丢弃，
            // 避免"闪回原状/部分 Clip 还原"；最新响应（含此前全部已提交
            // 变更）总是最后一个生效。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            // force：重设尺寸（右缘位移）的权威结果含波纹平移，交互锁期间不能丢弃。
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(setClipStateRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            applyOptimisticClipState(state, action.meta.arg);
            const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
            const formantEnabled =
                action.meta.arg.formantMorph?.enabled ?? clip?.formantMorph?.enabled ?? false;
            const formantRelevantChange =
                action.meta.arg.formantMorph !== undefined ||
                action.meta.arg.sourceStartSec !== undefined ||
                action.meta.arg.sourceEndSec !== undefined ||
                action.meta.arg.reversed !== undefined;
            if (formantEnabled && formantRelevantChange) {
                state.clipFormantStatus[action.meta.arg.clipId] = "rebuilding";
            }
        })
        .addCase(setClipStateRemote.rejected, (state, action) => {
            // 后端拒绝：保留乐观值（用户意图可见）。
            // 此后任何权威快照会以真实状态纠正；不静默回滚。
            applyOptimisticClipState(state, action.meta.arg);
            // 【为什么不一律报错】"宿主还在追上几何/媒体"是**瞬时**失败，后端会自己
            // 重试并收敛，用户不需要做任何事。此前它和"真正被拒绝"共用一句
            // "音频块编辑被拒绝"，于是每次编辑都闪一次错 —— 即使编辑其实生效了。
            if (classifyClipEditFailure(action.payload) === "permanent") {
                state.status = "Clip edit rejected";
                // 后端原文（`detail`）落到 ToolTip：截图即含诊断信息。
                state.statusDetail =
                    (action.payload as { detail?: string } | undefined)?.detail ?? null;
            }
        })

        .addCase(setClipsStateBulkRemote.fulfilled, (state, action) => {
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
            // force：批量状态（含未来尺寸波纹路径）的权威结果，交互锁期间不能丢弃。
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(setClipsStateBulkRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            applyOptimisticBulkClipState(state, action.meta.arg.updates);
        })
        .addCase(setClipsStateBulkRemote.rejected, (state, action) => {
            // 后端拒绝：保留乐观值（用户意图可见）；分类理由见
            // `setClipStateRemote.rejected`。
            applyOptimisticBulkClipState(state, action.meta.arg.updates);
            if (classifyClipEditFailure(action.payload) === "permanent") {
                state.status = "Clip edit rejected";
                // 后端原文（`detail`）落到 ToolTip：截图即含诊断信息。
                state.statusDetail =
                    (action.payload as { detail?: string } | undefined)?.detail ?? null;
            }
        })

        .addCase(scanAndConvertFakeStereoRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
                scanned?: number;
                converted?: number;
                pending?: number;
                eligibility?: {
                    /** 后端 `snake_case` 序列化：字段名必须逐字一致。 */
                    project_clips?: number;
                    matched_clips?: number;
                    takes_seen?: number;
                    skipped_no_source?: number;
                };
            };
            if (!payload.ok) {
                state.status = "Fake-stereo scan rejected";
                return;
            }
            const scanned = payload.scanned ?? 0;
            const converted = payload.converted ?? 0;
            const pending = payload.pending ?? 0;

            // 一个候选都没有时，**按真实计数**说清为什么。过去这里会猜一个
            // 原因，而计数本身因为字段名不一致恒为 0 —— 于是界面随口断言
            // "选中的音频块在工程里找不到"，一句听起来像用户操作有误、
            // 实际完全虚假的话。
            if (scanned === 0) {
                const eligibility = payload.eligibility ?? {};
                const projectClips = eligibility.project_clips ?? 0;
                const matched = eligibility.matched_clips ?? 0;
                const seen = eligibility.takes_seen ?? 0;
                const noSource = eligibility.skipped_no_source ?? 0;
                if (projectClips === 0) {
                    state.status = "Fake-stereo scan: project has no clips";
                } else if (matched === 0) {
                    state.status = `Fake-stereo scan: range matches no clip (${projectClips} in project)`;
                } else if (seen === 0) {
                    state.status = `Fake-stereo scan: ${matched} clip(s) have no takes`;
                } else if (noSource > 0) {
                    state.status = `Fake-stereo scan: ${noSource} take(s) have no source`;
                } else {
                    state.status = "Fake-stereo scan: nothing to decide";
                }
                return;
            }

            // 后缀在 App.tsx 的 statusText 里解析（`statusKey` 映射 + 双数量
            // 正则）。"unreadable"必须与"已折叠"区分开：前者下次打开会自动
            // 重试，后者永远不会再判。计数一律写成 {converted}/{scanned} 分数，
            // 比逐个念数量短一半。
            const pendingSuffix = pending > 0 ? `, ${pending} unreadable` : "";
            state.status = action.meta.arg.dryRun
                ? `Fake-stereo scan: ${converted}/${scanned} foldable${pendingSuffix}`
                : `Fake-stereo scan: ${converted}/${scanned} folded${pendingSuffix}`;
        })
        .addCase(scanAndConvertFakeStereoRemote.rejected, (state, action) => {
            setRejected(state, action);
        })

        .addCase(setClipActiveTakeRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
            if (!clip) return;
            const takes = clip.takes ?? [];
            const take = takes.find((entry) => entry.id === action.meta.arg.takeId);
            if (!take) return;
            captureTakeRollback(state, [action.meta.arg.clipId]);
            clip.activeTakeId = take.id;
            applyActiveTakeToFlat(clip, take);
        })
        .addCase(setClipActiveTakeRemote.fulfilled, (state, action) => {
            // 过期守卫：过期回包（含 !ok 的回滚）一律整份丢弃 —— 回滚的是
            // 更新一次的乐观切换，等于把新切换撤销掉。丢弃前释放本次登记
            // 的回滚快照，否则它会永远滞留（见 pendingTakeRollbacks）。
            if (takeRequestIsStale(state, action.meta.requestId)) {
                clearTakeRollback([action.meta.arg.clipId]);
                return;
            }
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                // 后端拒绝：回滚乐观切换并给出可见反馈。
                restoreTakeRollback(state, [action.meta.arg.clipId]);
                state.error = "Take switch rejected";
                state.status = "Failed";
                return;
            }
            clearTakeRollback([action.meta.arg.clipId]);
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(setClipActiveTakeRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) {
                clearTakeRollback([action.meta.arg.clipId]);
                return;
            }
            restoreTakeRollback(state, [action.meta.arg.clipId]);
            setRejected(state, action);
        })

        .addCase(cycleClipTakesRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            for (const clipId of action.meta.arg.clipIds) {
                const clip = state.clips.find((entry) => entry.id === clipId);
                const takes = clip?.takes ?? [];
                if (!clip || takes.length <= 1) continue;
                const direction = action.meta.arg.direction ?? 1;
                const currentIndex = Math.max(
                    0,
                    takes.findIndex((take) => take.id === clip.activeTakeId),
                );
                const nextIndex =
                    direction >= 0
                        ? (currentIndex + 1) % takes.length
                        : (currentIndex + takes.length - 1) % takes.length;
                const next = takes[nextIndex];
                captureTakeRollback(state, [clipId]);
                clip.activeTakeId = next.id;
                applyActiveTakeToFlat(clip, next);
            }
        })
        .addCase(cycleClipTakesRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) {
                clearTakeRollback(action.meta.arg.clipIds);
                return;
            }
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                restoreTakeRollback(state, action.meta.arg.clipIds);
                state.error = "Take cycle rejected";
                state.status = "Failed";
                return;
            }
            clearTakeRollback(action.meta.arg.clipIds);
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(cycleClipTakesRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) {
                clearTakeRollback(action.meta.arg.clipIds);
                return;
            }
            restoreTakeRollback(state, action.meta.arg.clipIds);
            setRejected(state, action);
        })

        .addCase(setClipTakeReversedRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            // 乐观翻转单个 Take：与后端 flip_take_playback_direction 同口径
            // 换算该 Take 的源窗口/锚点（保持消费内容不变）。active take
            // 需物化到 flat 投影；inactive take 只动自身条目。
            const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
            if (!clip) return;
            const takes = clip.takes ?? [];
            const take = takes.find((entry) => entry.id === action.meta.arg.takeId);
            if (!take) return;
            if (take.reversed !== action.meta.arg.reversed) {
                flipTakeSourceWindowForDirection(take, clip.lengthSec, clip.clipPlaybackRate ?? 1);
            }
            take.reversed = action.meta.arg.reversed;
            if (take.id === clip.activeTakeId) {
                applyActiveTakeToFlat(clip, take);
            }
        })
        .addCase(setClipTakeReversedRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Take reverse rejected";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(setClipTakeReversedRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })

        .addCase(setClipTakeChannelModeRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
            // 乐观切换单个 Take 的声道模式；active take 物化到 flat 投影，
            // inactive take 只动自身条目（与 reversed 同模式）。
            const clip = state.clips.find((entry) => entry.id === action.meta.arg.clipId);
            if (!clip) return;
            const takes = clip.takes ?? [];
            const take = takes.find((entry) => entry.id === action.meta.arg.takeId);
            if (!take) return;
            take.channelMode = normalizeChannelMode(action.meta.arg.channelMode);
            if (take.id === clip.activeTakeId) {
                applyActiveTakeToFlat(clip, take);
            }
        })
        .addCase(setClipTakeChannelModeRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Take channel mode rejected";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(setClipTakeChannelModeRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })

        .addCase(packClipsIntoTakesRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })

        .addCase(explodeClipTakesRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })

        .addCase(duplicateClipTakeRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })
        .addCase(removeClipTakeRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })
        .addCase(renameClipTakeRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })
        .addCase(addClipTakeFromMediaRemote.rejected, (state, action) => {
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            setRejected(state, action);
        })

        .addCase(packClipsIntoTakesRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(packClipsIntoTakesRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Pack into takes failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })

        .addCase(explodeClipTakesRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(explodeClipTakesRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Explode takes failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })

        .addCase(duplicateClipTakeRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(duplicateClipTakeRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Duplicate take failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(removeClipTakeRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(removeClipTakeRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Remove take failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(renameClipTakeRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(renameClipTakeRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Rename take failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })
        .addCase(addClipTakeFromMediaRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(addClipTakeFromMediaRemote.fulfilled, (state, action) => {
            // 过期守卫：见 setClipActiveTakeRemote.fulfilled。
            if (takeRequestIsStale(state, action.meta.requestId)) return;
            const payload = action.payload as { ok?: boolean } & TimelineState;
            if (!payload.ok) {
                state.error = "Add take from media failed";
                state.status = "Failed";
                return;
            }
            applyTimelineStatePreservingPlayhead(state, payload);
        })

        .addCase(replaceClipSourceRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(replaceClipSourceRemote.fulfilled, (state, action) => {
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

        .addCase(replaceMidiClipDataRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(replaceMidiClipDataRemote.fulfilled, (state, action) => {
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
        });
}
