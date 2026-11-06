import type { TimelineState } from "../../../../types/api";
import { loadDefaultModel, loadModel } from "../../thunks/modelThunks";
import {
    applyPitchShift,
    exportAudio,
    exportAudioAdvanced,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
    pickOutputPath,
    processAudio,
    synthesizeAudio,
} from "../../thunks/audioThunks";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromDialog,
    importAudioFromPath,
    importFolderAtPosition,
    importMidiAsClip,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "../../thunks/importThunks";
import { applyTimelineState, upsertImportedClip } from "../helpersTimeline";
import { intentPlayheadSec } from "../playheadIntent";
import {
    applyTimelineStatePreservingPitchVisuals,
    applyAutoCrossfadeInReducer,
} from "../helpersCore";
import { setPending, setRejected } from "./shared";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

export function registerAudio(builder: ActionReducerMapBuilder<SessionState>) {
    builder
        .addCase(loadDefaultModel.pending, (state) => setPending(state, "Loading default model..."))
        .addCase(loadDefaultModel.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            state.status = (action.payload as { ok?: boolean }).ok
                ? "Default model loaded"
                : "Load default model failed";
        })
        .addCase(loadDefaultModel.rejected, setRejected)

        .addCase(loadModel.pending, (state) => setPending(state, "Loading model..."))
        .addCase(loadModel.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            state.status = (action.payload as { ok?: boolean }).ok
                ? "Model loaded"
                : "Load model failed";
        })
        .addCase(loadModel.rejected, setRejected)

        .addCase(processAudio.pending, (state) => setPending(state, "Processing audio..."))
        .addCase(processAudio.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                audio?: { path?: string; duration_sec?: number };
                feature?: {
                    pitch_range?: { min: number; max: number };
                };
                timeline?: TimelineState;
            };
            if (payload.ok && payload.audio?.path) {
                state.audioPath = payload.audio.path;
                if (payload.timeline) {
                    applyTimelineState(state, payload.timeline, { force: true });
                } else {
                    upsertImportedClip(state, payload.audio.path, {
                        durationSec: payload.audio.duration_sec,
                        pitchRange: payload.feature?.pitch_range,
                    });
                }
            }
            state.status = payload.ok ? "Audio processed" : "Process audio failed";
        })
        .addCase(processAudio.rejected, setRejected)

        .addCase(importAudioFromDialog.pending, (state) => setPending(state, "Importing audio..."))
        .addCase(importAudioFromDialog.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                canceled?: boolean;
                path?: string;
                imported?: { ok?: boolean } & TimelineState;
                newClipIds?: string[];
                requiresModeChoice?: boolean;
                requiresStreamChoice?: boolean;
            };
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            // 多文件/多音轨流程在这里只是"等待用户下一步选择"，并没有
            // imported 载荷 —— 不能落入下面的 else 分支误报导入失败。
            if (payload.requiresModeChoice || payload.requiresStreamChoice) {
                state.status = "Waiting for import options";
                return;
            }
            if (payload.path) {
                state.audioPath = payload.path;
                if (payload.imported?.ok) {
                    applyTimelineState(state, payload.imported, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.multiSelectionIntentional = false;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                }
            }
            if (payload.imported?.ok) {
                state.status = "Audio imported";
            } else {
                state.status = "Import audio failed";
                state.error = "import_audio_failed";
            }
        })
        .addCase(importAudioFromDialog.rejected, setRejected)

        .addCase(importAudioFromPath.pending, (state) =>
            setPending(state, "Importing dropped audio..."),
        )
        .addCase(importAudioFromPath.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                canceled?: boolean;
                path?: string;
                imported?: { ok?: boolean } & TimelineState;
                newClipIds?: string[];
            };
            // 用户在导入在途时撤销 / 切换工程 → thunk 回取消结果且不带快照。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            if (payload.path) {
                state.audioPath = payload.path;
                if (payload.imported?.ok) {
                    applyTimelineState(state, payload.imported, { force: true });
                    if (payload.newClipIds && payload.newClipIds.length > 0) {
                        state.multiSelectedClipIds = payload.newClipIds;
                        state.multiSelectionIntentional = false;
                        state.selectedClipId = payload.newClipIds[0] ?? null;
                    }
                }
            }
            if (payload.imported?.ok) {
                state.status = "Dropped audio imported";
            } else {
                state.status = "Import audio failed";
                state.error = "import_audio_failed";
            }
        })
        .addCase(importAudioFromPath.rejected, setRejected)

        .addCase(importAudioAtPosition.pending, (state) => setPending(state, "Importing audio..."))
        .addCase(importAudioAtPosition.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            // 用户在导入在途时撤销 / 切换工程 → thunk 回取消结果且不带快照。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            if (ok) {
                state.status = "Import done";
            } else {
                state.status = "Import failed";
                // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                state.error = "import_audio_failed";
            }
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                // 导入定位光标：thunk 以独立字段带回的导入起始位置
                // （对应旧乐观路径 upsertImportedClip 的 playheadSec=startSec）。
                if (typeof payload.playheadSec === "number") {
                    if (intentPlayheadSec(state, payload.playheadSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importAudioAtPosition.rejected, setRejected)

        .addCase(importAudioFileAtPosition.pending, (state) =>
            setPending(state, "Importing audio..."),
        )
        .addCase(importAudioFileAtPosition.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            // 用户在导入在途时撤销 / 切换工程 → thunk 回取消结果且不带快照。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            if (ok) {
                state.status = "Import done";
            } else {
                state.status = "Import failed";
                // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                state.error = "import_audio_failed";
            }
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                // 导入定位光标：thunk 以独立字段带回的导入起始位置。
                if (typeof payload.playheadSec === "number") {
                    if (intentPlayheadSec(state, payload.playheadSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importAudioFileAtPosition.rejected, setRejected)

        .addCase(importMultipleAudioAtPosition.pending, (state) =>
            setPending(state, "Importing multiple audio files..."),
        )
        .addCase(importMultipleAudioAtPosition.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            // 用户在导入进行中撤销 / 跳转历史 → 循环提前退出，且不带时间线快照。
            // 必须早返回：否则会显示"Import done"，而且 `imported` 为空虽会跳过
            // 套用，状态栏却在骗人。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            if (ok) {
                state.status = "Import done";
            } else {
                state.status = "Import failed";
                // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                state.error = "import_audio_failed";
            }
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                // 导入定位光标：thunk 以独立字段带回的导入起始位置。
                if (typeof payload.playheadSec === "number") {
                    if (intentPlayheadSec(state, payload.playheadSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importMultipleAudioAtPosition.rejected, setRejected)

        /*
         * 目录导入。与 `importMultipleAudioAtPosition` 同一套收尾 —— 不创建
         * 轨道组时它就是被委托过去的那条路径，payload 形状完全一致；创建轨道组
         * 时多带回 `failedFiles` / `attempted`，由调用方汇总提示（状态栏 + 临时提示）。
         */
        .addCase(importFolderAtPosition.pending, (state) =>
            setPending(state, "Importing folder..."),
        )
        .addCase(importFolderAtPosition.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            // 用户在导入进行中撤销 / 跳转历史 → 循环提前退出并回一个取消结果。
            // 必须在这里早返回：否则会落进下面的 `ok` 分支显示"Import done"，
            // 而且 `imported` 为空会跳过时间线套用 —— 但状态栏会骗人。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            if (ok) {
                state.status = "Import done";
            } else {
                state.status = "Import failed";
                state.error = "import_audio_failed";
            }
            // 目录导入的汇总：成功数 / 失败数。状态行按这个格式翻译
            // （见 `resolveStatusText` 的 "Folder import:" 分支）。
            const summary = action.payload as {
                attempted?: number;
                failedFiles?: string[];
            };
            if (
                ok &&
                typeof summary.attempted === "number" &&
                summary.failedFiles &&
                summary.attempted > 0
            ) {
                const failed = summary.failedFiles.length;
                state.status = `Folder import: ${summary.attempted - failed} imported, ${failed} failed`;
            }
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                if (typeof payload.playheadSec === "number") {
                    if (intentPlayheadSec(state, payload.playheadSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    applyAutoCrossfadeInReducer(state, payload.newClipIds);
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importFolderAtPosition.rejected, setRejected)

        .addCase(importMultipleAudioFilesAtPosition.pending, (state) =>
            setPending(state, "Importing multiple audio files..."),
        )
        .addCase(importMultipleAudioFilesAtPosition.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            if (ok) {
                state.status = "Import done";
            } else {
                state.status = "Import failed";
                // 合成错误码：让状态栏走红色 error 通道（原版失败只写灰色 status）
                state.error = "import_audio_failed";
            }
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                // 导入定位光标：thunk 以独立字段带回的导入起始位置。
                if (typeof payload.playheadSec === "number") {
                    if (intentPlayheadSec(state, payload.playheadSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    applyAutoCrossfadeInReducer(state, payload.newClipIds);
                }
                // select all imported clips
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importMultipleAudioFilesAtPosition.rejected, setRejected)

        .addCase(importMidiAsClip.pending, (state) => setPending(state, "Importing MIDI clip..."))
        .addCase(importMidiAsClip.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                canceled?: boolean;
                imported?: TimelineState;
                newClipIds?: string[];
                playheadSec?: number;
            };
            // 用户在导入在途时撤销 / 切换工程 → thunk 回取消结果且不带快照。
            if (payload.canceled) {
                state.status = "Import canceled";
                return;
            }
            const ok = Boolean(payload.ok);
            state.status = ok ? "MIDI clip created" : "MIDI import failed";
            if (ok && payload.imported && payload.imported.tracks) {
                applyTimelineStatePreservingPitchVisuals(state, payload.imported);
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
            }
        })
        .addCase(importMidiAsClip.rejected, setRejected)

        .addCase(pickOutputPath.pending, (state) => setPending(state, "Selecting output path..."))
        .addCase(pickOutputPath.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                canceled?: boolean;
                path?: string;
            };
            if (payload.canceled) {
                state.status = "Pick output canceled";
                return;
            }
            if (payload.path) {
                state.outputPath = payload.path;
                state.status = "Output path selected";
            }
        })
        .addCase(pickOutputPath.rejected, setRejected)

        .addCase(applyPitchShift.pending, (state) => setPending(state, "Applying pitch shift..."))
        .addCase(applyPitchShift.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            state.status = (action.payload as { ok?: boolean }).ok
                ? "Pitch shift applied"
                : "Pitch shift failed";
        })
        .addCase(applyPitchShift.rejected, setRejected)

        .addCase(synthesizeAudio.pending, (state) => setPending(state, "Synthesizing..."))
        .addCase(synthesizeAudio.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            state.status = (action.payload as { ok?: boolean }).ok
                ? "Synthesis done"
                : "Synthesis failed";
        })
        .addCase(synthesizeAudio.rejected, setRejected)

        .addCase(exportAudio.pending, (state) => setPending(state, "Exporting WAV..."))
        .addCase(exportAudio.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                path?: string;
            };
            if (payload.ok) {
                // 状态栏文本会先匹配 statusKey 前缀 "Export done"，
                // 再把路径附加在后面，用户能看到 "导出完成 — D:\xxx.wav"。
                const suffix = payload.path ? ` — ${payload.path}` : "";
                state.status = `Export done${suffix}`;
            } else {
                state.status = "Export failed";
            }
        })
        .addCase(exportAudio.rejected, setRejected)

        .addCase(exportSeparated.pending, (state) =>
            setPending(state, "Exporting separated tracks..."),
        )
        .addCase(exportSeparated.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                count?: number;
                output_dir?: string;
            };
            if (payload.ok) {
                const suffix = payload.output_dir
                    ? ` — ${payload.output_dir} (${payload.count ?? 0} tracks)`
                    : "";
                state.status = `Export separated done${suffix}`;
            } else {
                state.status = "Export separated failed";
            }
        })
        .addCase(exportSeparated.rejected, setRejected)

        .addCase(exportAudioAdvanced.pending, (state) => setPending(state, "Exporting audio..."))
        .addCase(exportAudioAdvanced.fulfilled, (state, action) => {
            state.busy = false;
            state.lastResult = action.payload;
            const payload = action.payload as {
                ok?: boolean;
                mode?: "project" | "separated";
                path?: string;
                output_dir?: string;
                count?: number;
            };
            if (!payload.ok) {
                state.status =
                    payload.mode === "separated" ? "Export separated failed" : "Export failed";
                return;
            }
            if (payload.mode === "separated") {
                const suffix = payload.output_dir
                    ? ` — ${payload.output_dir} (${payload.count ?? 0} tracks)`
                    : "";
                state.status = `Export separated done${suffix}`;
                return;
            }
            const suffix = payload.path ? ` — ${payload.path}` : "";
            state.status = `Export done${suffix}`;
        })
        .addCase(exportAudioAdvanced.rejected, setRejected)

        .addCase(pasteVocalShifterClipboard.pending, (state) =>
            setPending(state, "Pasting VocalShifter clipboard data..."),
        )
        .addCase(pasteVocalShifterClipboard.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                tracks?: unknown;
                newClipIds?: string[];
                pasteEndSec?: number | null;
            };
            if (payload?.tracks) {
                applyTimelineState(state, payload as TimelineState, { force: true });
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
                // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
                // （transport 已由 thunk 同步，这里对齐本地状态）。
                // 视图聚焦延迟到提交后由 TimelinePanel 的
                // useLayoutEffect 依据 pendingPlayheadRevealSec 执行。
                if (
                    typeof payload.pasteEndSec === "number" &&
                    Number.isFinite(payload.pasteEndSec)
                ) {
                    if (intentPlayheadSec(state, payload.pasteEndSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
            }
            state.lastResult = payload;
            state.paramsEpoch = (Number(state.paramsEpoch) || 0) + 1;
            state.status = "Pasted VocalShifter clipboard data";
        })
        .addCase(pasteVocalShifterClipboard.rejected, (state, action) => {
            state.busy = false;
            state.error = (action.payload as string) ?? action.error?.message ?? "Request failed";
            state.status = "Failed";
        })

        .addCase(pasteReaperClipboard.pending, (state) =>
            setPending(state, "Pasting Reaper clipboard data..."),
        )
        .addCase(pasteReaperClipboard.fulfilled, (state, action) => {
            state.busy = false;
            const payload = action.payload as {
                timeline?: TimelineState;
                newClipIds?: string[];
                pasteEndSec?: number | null;
                skippedFiles?: string[];
            };
            if (payload?.timeline) {
                applyTimelineState(state, payload.timeline, { force: true });
                if (payload.newClipIds && payload.newClipIds.length > 0) {
                    state.multiSelectedClipIds = payload.newClipIds;
                    state.multiSelectionIntentional = false;
                    state.selectedClipId = payload.newClipIds[0] ?? null;
                }
                // 粘贴后光标跳到所有新 Clip 中最靠右的结束位置
                // （transport 已由 thunk 同步，这里对齐本地状态）。
                // 视图聚焦延迟到提交后由 TimelinePanel 的
                // useLayoutEffect 依据 pendingPlayheadRevealSec 执行。
                if (
                    typeof payload.pasteEndSec === "number" &&
                    Number.isFinite(payload.pasteEndSec)
                ) {
                    if (intentPlayheadSec(state, payload.pasteEndSec)) {
                        state.pendingPlayheadRevealSec = state.playheadSec;
                    }
                }
            }
            const skippedFiles = payload?.skippedFiles;
            state.reaperSkippedFilesDialog =
                Array.isArray(skippedFiles) && skippedFiles.length > 0 ? skippedFiles : null;
            state.status = "Pasted Reaper clipboard data";
        })
        .addCase(pasteReaperClipboard.rejected, (state, action) => {
            state.busy = false;
            state.error =
                (action.payload as string) ??
                action.error?.message ??
                "Paste Reaper clipboard failed";
            state.status = "Failed";
        });
}
