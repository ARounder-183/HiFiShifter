import { classifyClipEditFailure } from "../../clipEditFailure";
import type { TimelineState } from "../../../../types/api";
import {
    addClipOnTrack,
    createClipsRemote,
    pasteTimelineClipboardRemote,
    duplicateClipsBulkRemote,
    duplicateTrackRemote,
    analyzeSilenceRemote,
    removeSilenceRemote,
    glueClipsRemote,
    groupClipsRemote,
    ungroupClipsRemote,
    toggleGroupDisabledRemote,
    moveClipRemote,
    moveClipsRemote,
    closeTrackGapsRemote,
    removeClipRemote,
    removeClipsRemote,
    splitClipRemote,
    splitClipsAtRemote,
} from "../../thunks/timelineThunks";
import { removeSelectedClipRemote } from "../../thunks/trackThunks";
import { applyTimelineState } from "../helpersTimeline";
import { applySplitSelection } from "../helpersTakes";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import { setPending } from "./shared";
import type { SessionState } from "../types";

export function registerTimelineEdits(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(addClipOnTrack.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(addClipOnTrack.fulfilled, (state, action) => {
            // 乱序守卫：见 setClipStateRemote.fulfilled（撤销/打开工程会作废在途响应）。
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

        .addCase(duplicateTrackRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            // 克隆轨道属于离散操作，必须立即同步后端快照，避免 UI 延迟刷新。
            applyTimelineState(state, payload, { force: true });
        })

        .addCase(createClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(createClipsRemote.fulfilled, (state, action) => {
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
            state.status = "Clips created";
        })

        .addCase(pasteTimelineClipboardRemote.pending, (state, action) => {
            setPending(state, "Pasting timeline clipboard...");
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(pasteTimelineClipboardRemote.fulfilled, (state, action) => {
            state.busy = false;
            // 乱序守卫：粘贴含多次往返，窗口最宽；迟到的粘贴快照不得覆盖
            // 更新的撤销/重做/工程切换结果（见 setClipStateRemote.fulfilled）。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
                timeline?: TimelineState;
                newClipIds?: string[];
                pasteEndSec?: number | null;
            };
            if (!payload?.ok || !payload.timeline?.tracks) {
                state.status = "Paste timeline clipboard failed";
                return;
            }
            applyTimelineState(state, payload.timeline, { force: true });
            if (payload.newClipIds && payload.newClipIds.length > 0) {
                state.multiSelectedClipIds = payload.newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = payload.newClipIds[0] ?? null;
            }
            // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
            // （transport 已由 thunk 同步，这里对齐本地状态）。
            // 视图聚焦延迟到提交后由 TimelinePanel 的
            // useLayoutEffect 依据 pendingPlayheadRevealSec 执行，
            // 避免在工程全长扩充前被旧滚动上限钳制。
            if (typeof payload.pasteEndSec === "number" && Number.isFinite(payload.pasteEndSec)) {
                state.playheadSec = Math.max(0, payload.pasteEndSec);
                state.pendingPlayheadRevealSec = Math.max(0, payload.pasteEndSec);
            }
            state.status = "Timeline clipboard pasted";
        })
        .addCase(pasteTimelineClipboardRemote.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ??
                action.error?.message ??
                "Paste timeline clipboard failed";
            state.status = "Paste timeline clipboard failed";
        })

        .addCase(duplicateClipsBulkRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(duplicateClipsBulkRemote.fulfilled, (state, action) => {
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
            state.status = "Clips duplicated";
        })

        .addCase(removeClipRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(removeClipRemote.fulfilled, (state, action) => {
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

        .addCase(removeClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(removeClipsRemote.fulfilled, (state, action) => {
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

        .addCase(removeSelectedClipRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(removeSelectedClipRemote.fulfilled, (state, action) => {
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

        .addCase(moveClipRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(moveClipRemote.fulfilled, (state, action) => {
            // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            // force：移动是本次交互的权威结果（含波纹编辑对后续剪辑的平移）。
            // 交互锁期间若不强制应用，后端返回的波纹位移会被丢弃，导致波纹“无效”。
            applyTimelineState(state, payload, { force: true });
        })
        .addCase(moveClipRemote.rejected, (state, action) => {
            // 【为什么移动此前完全没有 rejected 分支】失败是**静默**的：同一类宿主
            // 冲突在 set_clip_state 上会报"音频块编辑被拒绝"，在移动上却什么都不说，
            // 用户只看到片段弹回原位。现在与状态编辑用同一套分类：瞬时失败仍静默
            // （后端自行收敛），只有需要用户处置的才提示。
            if (classifyClipEditFailure(action.payload) === "permanent") {
                state.status = "Clip edit rejected";
                // 后端原文（`detail`）落到 ToolTip：截图即含诊断信息。
                state.statusDetail =
                    (action.payload as { detail?: string } | undefined)?.detail ?? null;
            }
        })
        .addCase(moveClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(moveClipsRemote.fulfilled, (state, action) => {
            // 乱序守卫：只采纳最近一次编辑请求的响应（见 _latestEditRequestId）。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            // force：见 moveClipRemote.fulfilled 注释（交互锁期间也必须应用波纹结果）。
            applyTimelineState(state, payload, { force: true });
        })
        .addCase(moveClipsRemote.rejected, (state, action) => {
            // 与 moveClipRemote.rejected 同一分类口径（此前完全静默）。
            if (classifyClipEditFailure(action.payload) === "permanent") {
                state.status = "Clip edit rejected";
                // 后端原文（`detail`）落到 ToolTip：截图即含诊断信息。
                state.statusDetail =
                    (action.payload as { detail?: string } | undefined)?.detail ?? null;
            }
        })

        .addCase(closeTrackGapsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(closeTrackGapsRemote.fulfilled, (state, action) => {
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
            // 关闭间隙由后端一次性计算并应用（含“锁定参数线”联动结果），
            // 这里整包采纳快照即可。
            applyTimelineState(state, payload, { force: true });
        })
        .addCase(analyzeSilenceRemote.pending, (state, action) => {
            state._silencePreviewRequestId = action.meta.requestId;
        })
        .addCase(analyzeSilenceRemote.rejected, (state, action) => {
            // 失败：与最新请求匹配时清掉过期预览（旧参数的区域不能再
            // 停留在覆盖层上误导用户）；非最新请求直接忽略。
            if (state._silencePreviewRequestId !== action.meta.requestId) {
                return;
            }
            state._silencePreviewRequestId = null;
            state.silencePreviewSegments = null;
        })
        .addCase(analyzeSilenceRemote.fulfilled, (state, action) => {
            // 乱序守卫：只采纳最近一次预览请求（防抖后旧响应覆盖新参数结果）。
            if (state._silencePreviewRequestId !== action.meta.requestId) {
                return;
            }
            state._silencePreviewRequestId = null;
            const map: Record<string, Array<[number, number]>> = {};
            for (const report of action.payload.reports) {
                if (report.ok && report.regions.length > 0) {
                    map[report.clipId] = report.regions.map(
                        (g) => [g.startSec, g.endSec] as [number, number],
                    );
                }
            }
            state.silencePreviewSegments = map;
        })
        .addCase(removeSilenceRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(removeSilenceRemote.fulfilled, (state, action) => {
            // 乱序守卫：见 setClipStateRemote.fulfilled。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as unknown as {
                ok?: boolean;
                timeline?: TimelineState;
                keptClipIds?: string[];
            };
            if (!payload.ok || !payload.timeline) {
                return;
            }
            // 整包采纳后端快照（applyTimelineState 会一并清除预览覆盖层）。
            applyTimelineState(state, payload.timeline, { force: true });
            // 恢复选区为处理后的发声片段（与 split 的“选中结果”惯例一致）。
            const kept = (payload.keptClipIds ?? []).filter((id) =>
                state.clips.some((c) => c.id === id),
            );
            if (kept.length > 0) {
                state.multiSelectedClipIds = kept;
                state.selectedClipId = kept[0];
                // 动作驱动的选区（用户没表达"整组拖动"），必须与 split /
                // 导入的 fulfilled 分支一样清掉意图标记，否则拖 kept 中的
                // 一个 clip 会把整组一起拖走；同时把复制/剪切的路由仲裁
                // 切回 Clip 侧（见 selectionContext 的注释）。
                state.multiSelectionIntentional = false;
                state.selectionContext = "clips";
            }
        })

        .addCase(splitClipRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(splitClipRemote.fulfilled, (state, action) => {
            // 乱序守卫：见 setClipStateRemote.fulfilled。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
                created_clip_ids?: string[] | null;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            applySplitSelection(state, action, payload);
        })

        .addCase(splitClipsAtRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(splitClipsAtRemote.fulfilled, (state, action) => {
            // 乱序守卫：见 setClipStateRemote.fulfilled。
            if (state._latestEditRequestId !== action.meta.requestId) {
                return;
            }
            const payload = action.payload as {
                ok?: boolean;
                created_clip_ids?: string[] | null;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            applySplitSelection(state, action, payload);
        })

        .addCase(glueClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(glueClipsRemote.fulfilled, (state, action) => {
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
            state.status = "Glue done";
        })

        .addCase(groupClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(groupClipsRemote.fulfilled, (state, action) => {
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

        .addCase(ungroupClipsRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(ungroupClipsRemote.fulfilled, (state, action) => {
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

        .addCase(toggleGroupDisabledRemote.pending, (state, action) => {
            state._latestEditRequestId = action.meta.requestId;
        })
        .addCase(toggleGroupDisabledRemote.fulfilled, (state, action) => {
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
