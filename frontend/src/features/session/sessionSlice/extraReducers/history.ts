import { isPluginMode } from "../../../../services/hostCapabilities";
import { parseHostAudio } from "../../../ara/hostAudio";
import type { TimelineState } from "../../../../types/api";
import {
    newProjectRemote,
    openProjectFromDialog,
    openProjectFromPath,
    openProjectFromPathForced,
    pickProjectToImport,
    importProjectFromPath,
    openVocalShifterFromDialog,
    openVocalShifterFromPath,
    openReaperFromDialog,
    openReaperFromPath,
    redoRemote,
    setHistoryPositionRemote,
    undoRemote,
} from "../../thunks/projectThunks";
import { fetchTimeline } from "../../thunks/transportThunks";
import { applyTimelineState } from "../helpersTimeline";
import {
    applyHistoryDepths,
    historyPayloadPresent,
    applyParamSelectionRestoreFromPayload,
} from "../history";
import { PLAYHEAD_MOVE_EPS_SEC } from "../settings";
import { setPending } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

export function registerHistory(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(fetchTimeline.pending, (state, action) => {
            if (isPluginMode())
                state._pluginTimelineFetchEpochs[action.meta.requestId] =
                    state._pluginTimelineEpoch;
        })
        .addCase(fetchTimeline.rejected, (state, action) => {
            delete state._pluginTimelineFetchEpochs[action.meta.requestId];
        })
        .addCase(fetchTimeline.fulfilled, (state, action) => {
            const epoch = state._pluginTimelineFetchEpochs[action.meta.requestId];
            delete state._pluginTimelineFetchEpochs[action.meta.requestId];
            if (
                isPluginMode() &&
                (state._interactionLockCount > 0 ||
                    Object.keys(state._pluginClipEditRequests).length > 0 ||
                    (epoch !== undefined && epoch !== state._pluginTimelineEpoch))
            )
                return;
            const payload = action.payload as {
                ok?: boolean;
                host_audio?: unknown;
            } & TimelineState;
            if (!payload.ok) {
                return;
            }
            /*
             * 宿主音频读数随快照更新。只认白名单里的分类名：后端将来加了新分类时，
             * 旧前端若把不认识的字符串当成已知状态渲染，就会给出一条**错误的指引**
             * （用户照着改工程，问题更糟）。未知分类与"本次没带该字段"同义 ——
             * 沿用上一次已知值，而不是清空或猜测（见 `parseHostAudio`）。
             */
            const nextHostAudio = parseHostAudio(payload.host_audio);
            // 【为什么在这里补 restore】恢复读数在 TimelineState 的顶层
            // （`ara_restore`），而宿主提示条订阅的是 `session.hostAudio`。把它
            // 镜像进宿主读数，提示条一处就能同时给出"音频为什么没到"与"参数为什么
            // 没读回"两边的真因 —— 后者此前只有一行后端日志。
            if (nextHostAudio && payload.ara_restore) {
                nextHostAudio.restore = payload.ara_restore;
            }
            if (nextHostAudio) state.hostAudio = nextHostAudio;
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                // 启动同步采纳后端播放头；播放中的回滚式重取（如 Tempo Map
                // 提交失败）保留本地播放头，避免跳回旧的起始位置。
                adoptPlayhead: !isPluginMode() && !state.runtime.isPlaying,
            });
        })

        .addCase(undoRemote.pending, (state, action) => {
            // 撤销/重做以后端为唯一权威：pending 阶段不做任何本地快照
            // 回放（旧实现会先乐观渲染前端快照——它与其后端检查点在深度
            // 与内容上都可能错位，造成轨道视图闪现任意旧状态后再被
            // fulfilled 的后端快照纠正）。这里仅记录最新请求，乱序到达
            // 的旧响应在 fulfilled/rejected 中被丢弃。
            state._latestHistoryOpRequestId = action.meta.requestId;
            // 作废在途编辑响应：迟到的编辑快照不得覆盖撤销结果。
            state._latestEditRequestId = null;
        })

        .addCase(undoRemote.fulfilled, (state, action) => {
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
            // 【为什么必须判空】本地权威栈（`undo_stack=false`，例如 ARA-only 宿主）
            // 在 UI 线程返回 `Ok(Value::Null)` —— 撤销结果由随后推送的完整时间线
            // 快照带回，这里没有可套用的载荷。此前直接解引用 `payload.undo_depth`
            // 会抛 `TypeError: Cannot read properties of null`，把整条撤销链路打断。
            if (!historyPayloadPresent(action.payload)) return;
            const payload = action.payload as {
                ok?: boolean;
                undo_depth?: number;
                redo_depth?: number;
                param_selection_restore?: [number, number][] | null;
            } & TimelineState;
            const payloadUndoDepth = Number(payload.undo_depth);
            const payloadRedoDepth = Number(payload.redo_depth);
            // 空栈（后端 ok = false）：这一步撤销并不存在 —— 不套用任何快照，
            // 界面零刷新零变更（否则用户会误以为撤销已执行），仅把响应带回的
            // 权威深度同步进镜像。
            if (!payload.ok) {
                if (Number.isFinite(payloadUndoDepth) && Number.isFinite(payloadRedoDepth)) {
                    applyHistoryDepths(state, payloadUndoDepth, payloadRedoDepth);
                }
                return;
            }
            // 撤销前的位置：被撤销的正是「产生该位置」的那一步。
            const depthBefore = state.historyUndoDepth;
            applyHistoryDepths(
                state,
                Number.isFinite(payloadUndoDepth) ? payloadUndoDepth : Math.max(0, depthBefore - 1),
                Number.isFinite(payloadRedoDepth) ? payloadRedoDepth : state.historyRedoDepth + 1,
            );
            // 被撤销的那一步若登记过选区（「边缘拉伸」），载荷会带回它拉伸前的
            // 选区，这里恢复；未携带则不动用户当前选区。
            applyParamSelectionRestoreFromPayload(state, payload);
            // 撤销把时间线（含 playhead_sec）整体回退到上一个检查点：快照里
            // 的 playhead_sec 就是该状态形成时的播放光标位置，也是回退之后
            // 一切以光标为锚点的编辑操作（粘贴/分割/录音起点等）在后端的
            // 实际操作点。必须采纳它让视觉光标同步归位 —— 若沿用撤销前的
            // 本地光标（可能是上一步操作挪过去的位置，如粘贴的
            // pasteEndSec），视觉停在 B、后端实际停在 A，下一次操作就会落
            // 在视觉之外的位置（视觉编辑点 ≠ 实际编辑点）。
            // 播放中例外：光标归传输层（音频时钟）所有，检查点值停留在
            // 本次播放的起始位置已过期，且轮询会持续覆写，不做本地改写。
            const prevPlayheadSec = state.playheadSec;
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: !state.runtime.isPlaying,
            });
            // 光标随回退真实挪动时登记"聚焦播放光标"：离屏时由
            // TimelinePanel 的 useLayoutEffect 滚动到可见（画面内不扰动），
            // 保证用户看得到回退后的实际操作点。
            if (
                !state.runtime.isPlaying &&
                Math.abs(state.playheadSec - prevPlayheadSec) > PLAYHEAD_MOVE_EPS_SEC
            ) {
                state.pendingPlayheadRevealSec = state.playheadSec;
            }
        })

        .addCase(undoRemote.rejected, (state, action) => {
            // 无乐观本地变更需要回滚：后端 undo 失败时时间线未变，
            // 前端保持现状即可（仅丢弃过期响应）。
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
        })

        .addCase(redoRemote.pending, (state, action) => {
            // 同 undoRemote.pending：以后端为唯一权威，不做本地快照回放。
            state._latestHistoryOpRequestId = action.meta.requestId;
            // 作废在途编辑响应：迟到的编辑快照不得覆盖重做结果。
            state._latestEditRequestId = null;
        })

        .addCase(redoRemote.fulfilled, (state, action) => {
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
            // 同 `undoRemote.fulfilled`：本地权威栈返回 null 载荷，必须判空。
            if (!historyPayloadPresent(action.payload)) return;
            const payload = action.payload as {
                ok?: boolean;
                undo_depth?: number;
                redo_depth?: number;
                param_selection_restore?: [number, number][] | null;
            } & TimelineState;
            const payloadUndoDepth = Number(payload.undo_depth);
            const payloadRedoDepth = Number(payload.redo_depth);
            // 空栈（后端 ok = false）：同 undoRemote.fulfilled —— 静默跳过。
            if (!payload.ok) {
                if (Number.isFinite(payloadUndoDepth) && Number.isFinite(payloadRedoDepth)) {
                    applyHistoryDepths(state, payloadUndoDepth, payloadRedoDepth);
                }
                return;
            }
            // 重做前的位置：被重做的正是「产生下一位置」的那一步。
            const depthBefore = state.historyUndoDepth;
            applyHistoryDepths(
                state,
                Number.isFinite(payloadUndoDepth) ? payloadUndoDepth : depthBefore + 1,
                Number.isFinite(payloadRedoDepth)
                    ? payloadRedoDepth
                    : Math.max(0, state.historyRedoDepth - 1),
            );
            // 被重做的那一步若登记过选区（「边缘拉伸」），载荷会带回它拉伸后的
            // 选区，这里恢复；未携带则不动用户当前选区。
            applyParamSelectionRestoreFromPayload(state, payload);
            // 与 undoRemote.fulfilled 对称：重做恢复的时间线快照携带的
            // playhead_sec 是该状态（被撤销暂存时）的光标位置，即重做后
            // 后端的实际操作点 —— 采纳它，视觉光标与实际编辑点保持一致。
            // 播放中例外同 undoRemote.fulfilled：光标归传输层所有。
            const prevPlayheadSec = state.playheadSec;
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: !state.runtime.isPlaying,
            });
            if (
                !state.runtime.isPlaying &&
                Math.abs(state.playheadSec - prevPlayheadSec) > PLAYHEAD_MOVE_EPS_SEC
            ) {
                state.pendingPlayheadRevealSec = state.playheadSec;
            }
        })

        .addCase(redoRemote.rejected, (state, action) => {
            // 同 undoRemote.rejected：无乐观本地变更需要回滚。
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
        })

        .addCase(setHistoryPositionRemote.pending, (state, action) => {
            // 「操作记录」跳转与撤销/重做同源：以后端快照为唯一权威，
            // pending 阶段不做任何本地回放。
            state._latestHistoryOpRequestId = action.meta.requestId;
            state._latestEditRequestId = null;
        })

        .addCase(setHistoryPositionRemote.fulfilled, (state, action) => {
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
            // 同 `undoRemote.fulfilled`：本地权威栈返回 null 载荷，必须判空。
            if (!historyPayloadPresent(action.payload)) return;
            const payload = action.payload as {
                ok?: boolean;
                undo_depth?: number;
                redo_depth?: number;
                param_selection_restore?: [number, number][] | null;
            } & TimelineState;
            const payloadUndoDepth = Number(payload.undo_depth);
            const payloadRedoDepth = Number(payload.redo_depth);
            // 越界 / 原地不动（后端 ok = false）：静默跳过，界面零变化。
            if (!payload.ok) {
                if (Number.isFinite(payloadUndoDepth) && Number.isFinite(payloadRedoDepth)) {
                    applyHistoryDepths(state, payloadUndoDepth, payloadRedoDepth);
                }
                return;
            }
            applyHistoryDepths(
                state,
                Number.isFinite(payloadUndoDepth) ? payloadUndoDepth : state.historyUndoDepth,
                Number.isFinite(payloadRedoDepth) ? payloadRedoDepth : state.historyRedoDepth,
            );
            // 跳到的那一步若登记过选区，载荷会带回该状态下的选区。
            applyParamSelectionRestoreFromPayload(state, payload);
            // 与撤销/重做一致：快照携带的 playhead_sec 即该状态形成时的
            // 光标位置（该状态下的实际操作点），播放中由传输层所有。
            const prevPlayheadSec = state.playheadSec;
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: !state.runtime.isPlaying,
            });
            if (
                !state.runtime.isPlaying &&
                Math.abs(state.playheadSec - prevPlayheadSec) > PLAYHEAD_MOVE_EPS_SEC
            ) {
                state.pendingPlayheadRevealSec = state.playheadSec;
            }
        })

        .addCase(setHistoryPositionRemote.rejected, (state, action) => {
            // 无乐观本地变更需要回滚（同撤销/重做）。
            if (state._latestHistoryOpRequestId !== action.meta.requestId) return;
        })

        .addCase(newProjectRemote.pending, (state) => {
            // 新建工程为权威替换：作废一切在途编辑响应。
            state._latestEditRequestId = null;
        })
        .addCase(newProjectRemote.fulfilled, (state, action) => {
            const payload = action.payload as {
                ok?: boolean;
            } & TimelineState;
            if (!payload.ok) return;
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            state.status = "New project";
        })

        .addCase(openProjectFromDialog.pending, (state) => {
            setPending(state, "Opening project...");
            // 工程载入为权威替换：作废一切在途编辑响应。
            state._latestEditRequestId = null;
        })
        .addCase(openProjectFromDialog.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: boolean;
                      canceled: false;
                      timeline?: TimelineState;
                      error?: string;
                      projectVersionTooNew?: boolean;
                  };
            if (!payload || payload.canceled) {
                state.status = "Open canceled";
                return;
            }
            if ((payload as { projectVersionTooNew?: boolean }).projectVersionTooNew) {
                state.status = "Project version confirmation required";
                return;
            }
            if (payload.ok === false) {
                // 后端返回了明确的失败原因（文件不存在/解析失败等）。
                state.error = payload.error ?? "Open project failed";
                state.status = "Open failed";
                return;
            }
            applyTimelineState(state, payload.timeline as TimelineState, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            state.status = "Project opened";
        })
        .addCase(openProjectFromDialog.rejected, (state, action) => {
            state.busy = false;
            state.error = action.error?.message ?? "Open project failed";
            state.status = "Open failed";
        })

        .addCase(openProjectFromPath.pending, (state) => {
            setPending(state, "Opening project...");
            // 工程载入为权威替换：作废一切在途编辑响应。
            state._latestEditRequestId = null;
        })
        .addCase(openProjectFromPath.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                ok?: boolean;
                error?: string;
                projectVersionTooNew?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                state.error = payload.error ?? "Open project failed";
                state.status = "Open failed";
                return;
            }
            if (payload.projectVersionTooNew) {
                state.status = "Project version confirmation required";
                return;
            }
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            state.status = "Project opened";
        })
        .addCase(openProjectFromPath.rejected, (state, action) => {
            state.busy = false;
            state.error = action.error?.message ?? "Open project failed";
            state.status = "Open failed";
        })

        .addCase(openProjectFromPathForced.pending, (state) => {
            setPending(state, "Opening project...");
            // 工程载入为权威替换：作废一切在途编辑响应（与其它 openProject* 一致）。
            state._latestEditRequestId = null;
        })
        .addCase(openProjectFromPathForced.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                ok?: boolean;
                error?: string;
                projectVersionTooNew?: boolean;
            } & TimelineState;
            if (!payload.ok) {
                state.error = payload.error ?? "Open project failed";
                state.status = "Open failed";
                return;
            }
            if (payload.projectVersionTooNew) {
                state.status = "Project version confirmation required";
                return;
            }
            applyTimelineState(state, payload, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            state.status = "Project opened";
        })
        .addCase(openProjectFromPathForced.rejected, (state, action) => {
            state.busy = false;
            state.error = action.error?.message ?? "Open project failed";
            state.status = "Open failed";
        })

        .addCase(pickProjectToImport.pending, (state) =>
            setPending(state, "Picking project to import..."),
        )
        .addCase(pickProjectToImport.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                { ok: true; canceled: true } | { ok: true; canceled: false; path: string };
            if (!payload || (payload as { canceled?: boolean }).canceled) {
                state.status = "Import canceled";
            }
        })
        .addCase(pickProjectToImport.rejected, (state, action) => {
            state.busy = false;
            state.error = action.error?.message ?? "Import project failed";
            state.status = "Import failed";
        })

        .addCase(importProjectFromPath.pending, (state) =>
            setPending(state, "Importing project..."),
        )
        .addCase(importProjectFromPath.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: true;
                      canceled: false;
                      timeline: TimelineState;
                      newClipIds?: string[];
                      sourceProject?: string;
                  };
            if (!payload || payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            applyTimelineState(state, payload.timeline, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            const newClipIds = payload.newClipIds;
            if (newClipIds && newClipIds.length > 0) {
                state.multiSelectedClipIds = newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = newClipIds[0] ?? null;
            }
            state.status = "Project imported";
        })
        .addCase(importProjectFromPath.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ?? action.error?.message ?? "Import project failed";
            state.status = "Import failed";
        })

        .addCase(openVocalShifterFromDialog.pending, (state) =>
            setPending(state, "Importing VocalShifter project..."),
        )
        .addCase(openVocalShifterFromDialog.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: true;
                      canceled: false;
                      timeline: TimelineState;
                      newClipIds?: string[];
                      skippedFiles?: string[];
                  };
            if (!payload || payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            applyTimelineState(state, payload.timeline, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            const newClipIds = payload.newClipIds;
            if (newClipIds && newClipIds.length > 0) {
                state.multiSelectedClipIds = newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = newClipIds[0] ?? null;
            }
            const skippedFiles = payload.skippedFiles;
            state.vocalShifterSkippedFilesDialog =
                Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
            state.status = "VocalShifter project imported";
        })
        .addCase(openVocalShifterFromDialog.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ?? action.error?.message ?? "Import VocalShifter failed";
            state.status = "Import failed";
        })

        .addCase(openVocalShifterFromPath.pending, (state) =>
            setPending(state, "Importing VocalShifter project..."),
        )
        .addCase(openVocalShifterFromPath.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: true;
                      canceled: false;
                      timeline: TimelineState;
                      newClipIds?: string[];
                      skippedFiles?: string[];
                  };
            if (!payload || payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            applyTimelineState(state, payload.timeline, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            const newClipIds = payload.newClipIds;
            if (newClipIds && newClipIds.length > 0) {
                state.multiSelectedClipIds = newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = newClipIds[0] ?? null;
            }
            const skippedFiles = payload.skippedFiles;
            state.vocalShifterSkippedFilesDialog =
                Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
            state.status = "VocalShifter project imported";
        })
        .addCase(openVocalShifterFromPath.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ?? action.error?.message ?? "Import VocalShifter failed";
            state.status = "Import failed";
        })

        .addCase(openReaperFromDialog.pending, (state) =>
            setPending(state, "Importing Reaper project..."),
        )
        .addCase(openReaperFromDialog.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: true;
                      canceled: false;
                      timeline: TimelineState;
                      newClipIds?: string[];
                      skippedFiles?: string[];
                  };
            if (!payload || payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            applyTimelineState(state, payload.timeline, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            const newClipIds = payload.newClipIds;
            if (newClipIds && newClipIds.length > 0) {
                state.multiSelectedClipIds = newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = newClipIds[0] ?? null;
            }
            const skippedFiles = payload.skippedFiles;
            state.reaperSkippedFilesDialog =
                Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
            state.status = "Reaper project imported";
        })
        .addCase(openReaperFromDialog.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ?? action.error?.message ?? "Import Reaper failed";
            state.status = "Import failed";
        })

        .addCase(openReaperFromPath.pending, (state) =>
            setPending(state, "Importing Reaper project..."),
        )
        .addCase(openReaperFromPath.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as
                | { ok: true; canceled: true }
                | {
                      ok: true;
                      canceled: false;
                      timeline: TimelineState;
                      newClipIds?: string[];
                      skippedFiles?: string[];
                  };
            if (!payload || payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            applyTimelineState(state, payload.timeline, {
                force: true,
                preserveProjectNotes: false,
                adoptPlayhead: true,
            });
            const newClipIds = payload.newClipIds;
            if (newClipIds && newClipIds.length > 0) {
                state.multiSelectedClipIds = newClipIds;
                state.multiSelectionIntentional = false;
                state.selectedClipId = newClipIds[0] ?? null;
            }
            const skippedFiles = payload.skippedFiles;
            state.reaperSkippedFilesDialog =
                Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
            state.status = "Reaper project imported";
        })
        .addCase(openReaperFromPath.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ?? action.error?.message ?? "Import Reaper failed";
            state.status = "Import failed";
        });
}
