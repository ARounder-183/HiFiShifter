import { isAnyOf } from "@reduxjs/toolkit";
import { isPluginMode } from "../../../../services/hostCapabilities";
import {
    pasteTimelineClipboardRemote,
    duplicateClipsBulkRemote,
    moveClipRemote,
    moveClipsRemote,
    moveTrackRemote,
    removeClipRemote,
    removeClipsRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    splitClipRemote,
    splitClipsAtRemote,
} from "../../thunks/timelineThunks";
import { redoRemote, setHistoryPositionRemote, undoRemote } from "../../thunks/projectThunks";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromDialog,
    importAudioFromPath,
    importFolderAtPosition,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "../../thunks/importThunks";
import type { ActionReducerMapBuilder } from "@reduxjs/toolkit";
import type { SessionState } from "../types";

export function registerMatchers(builder: ActionReducerMapBuilder<SessionState>) {
    // ── 在途导入计数 ──────────────────────────────────────────────────
    // 【必须在所有 addCase 之后】RTK 要求 `addMatcher` 只能排在 `addCase`
    // 之后，提前调用会直接抛错。
    //
    // 只统计**会去读音频文件**的导入 thunk：`busy` 是全局在途标志，被大量
    // 无关操作置起，拿它驱动"正在导入"的提示会误报。
    //
    // 计数而非布尔：`importAudioFromDialog` 会转派给 `importAudioAtPosition`，
    // 两者同时在途，用布尔会提前清零。
    //
    // 用 typePrefix 匹配而不是 `isAnyOf` + `isPending`：后者在本工程的 RTK
    // 类型下推不出收窄签名（重载会退化成返回 boolean，无法当 matcher 用）。
    // thunk 的 `typePrefix` 是稳定的公开契约（状态为 `<prefix>/pending` 等），
    // 直接比字符串更直白，也不依赖类型体操。
    const audioImportPrefixes = [
        importAudioAtPosition.typePrefix,
        importAudioFileAtPosition.typePrefix,
        importAudioFromDialog.typePrefix,
        importAudioFromPath.typePrefix,
        importMultipleAudioAtPosition.typePrefix,
        importMultipleAudioFilesAtPosition.typePrefix,
        // 目录导入是**最需要**这个提示的一条（逐文件循环可能持续数秒），
        // 此前恰恰漏了它。
        importFolderAtPosition.typePrefix,
    ];
    const isAudioImportStep = (action: { type?: string }, step: string) =>
        audioImportPrefixes.some((prefix) => action.type === `${prefix}/${step}`);
    const settleImport = (state: SessionState) => {
        state.importInFlight = Math.max(0, state.importInFlight - 1);
    };
    // 宿主写入开始/结束都废弃旧刷新，覆盖键盘编辑（没有拖动锁）的同类竞态。
    builder
        .addMatcher(
            isAnyOf(
                moveClipRemote.pending,
                moveClipsRemote.pending,
                moveTrackRemote.pending,
                pasteTimelineClipboardRemote.pending,
                removeClipRemote.pending,
                removeClipsRemote.pending,
                duplicateClipsBulkRemote.pending,
                splitClipRemote.pending,
                splitClipsAtRemote.pending,
                importAudioAtPosition.pending,
                importAudioFileAtPosition.pending,
                setClipStateRemote.pending,
                setClipsStateBulkRemote.pending,
                undoRemote.pending,
                redoRemote.pending,
                setHistoryPositionRemote.pending,
            ),
            (state, action) => {
                if (!isPluginMode()) return;
                state._pluginTimelineEpoch += 1;
                state._pluginClipEditRequests[action.meta.requestId] = true;
            },
        )
        .addMatcher(
            isAnyOf(
                moveClipRemote.fulfilled,
                moveClipRemote.rejected,
                moveTrackRemote.fulfilled,
                moveTrackRemote.rejected,
                pasteTimelineClipboardRemote.fulfilled,
                pasteTimelineClipboardRemote.rejected,
                duplicateClipsBulkRemote.fulfilled,
                duplicateClipsBulkRemote.rejected,
                removeClipRemote.fulfilled,
                removeClipRemote.rejected,
                removeClipsRemote.fulfilled,
                removeClipsRemote.rejected,
                splitClipRemote.fulfilled,
                splitClipRemote.rejected,
                splitClipsAtRemote.fulfilled,
                splitClipsAtRemote.rejected,
                importAudioAtPosition.fulfilled,
                importAudioAtPosition.rejected,
                importAudioFileAtPosition.fulfilled,
                importAudioFileAtPosition.rejected,
                moveClipsRemote.fulfilled,
                moveClipsRemote.rejected,
                setClipStateRemote.fulfilled,
                setClipStateRemote.rejected,
                setClipsStateBulkRemote.fulfilled,
                setClipsStateBulkRemote.rejected,
                undoRemote.fulfilled,
                undoRemote.rejected,
                redoRemote.fulfilled,
                redoRemote.rejected,
                setHistoryPositionRemote.fulfilled,
                setHistoryPositionRemote.rejected,
            ),
            (state, action) => {
                if (!isPluginMode()) return;
                delete state._pluginClipEditRequests[action.meta.requestId];
                state._pluginTimelineEpoch += 1;
            },
        );
    builder
        .addMatcher(
            (action: { type?: string }) => isAudioImportStep(action, "pending"),
            (state: SessionState) => {
                state.importInFlight += 1;
            },
        )
        .addMatcher(
            (action: { type?: string }) => isAudioImportStep(action, "fulfilled"),
            settleImport,
        )
        .addMatcher(
            (action: { type?: string }) => isAudioImportStep(action, "rejected"),
            settleImport,
        );
}
