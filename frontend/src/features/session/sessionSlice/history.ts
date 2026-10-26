// Session 撤销/重做深度与参数选区恢复 helper。拆自 sessionSlice。
import type { ToolMode } from "../sessionTypes";
import type { ParamSelectionSnapshot, SessionState } from "./types";

export function applyHistoryDepths(
    state: SessionState,
    undoDepth: number,
    redoDepth: number,
): void {
    state.historyUndoDepth = Math.max(0, Math.floor(Number(undoDepth) || 0));
    state.historyRedoDepth = Math.max(0, Math.floor(Number(redoDepth) || 0));
}

/**
 * 撤销/重做响应的载荷是否**可解引用**。
 *
 * 【为什么需要】本地权威栈（`undo_stack=false`，例如 ARA-only 宿主）在 UI 线程
 * 立刻返回 `Ok(Value::Null)` —— 真正的结果由随后推送的时间线快照带回。这类响应
 * 没有 `ok` / `undo_depth` 字段，直接解引用会抛
 * `TypeError: Cannot read properties of null (reading 'undo_depth')`，
 * 把整条撤销链路打断（真机日志里可见）。
 */
export function historyPayloadPresent(payload: unknown): boolean {
    return payload !== null && payload !== undefined && typeof payload === "object";
}

/** 登记一次选区恢复请求（requestId 单调，消费方按 id 幂等应用一次）。 */
export function requestParamSelectionRestore(
    state: SessionState,
    selection: ParamSelectionSnapshot | null,
): void {
    state._paramSelectionRestoreSeq = (Number(state._paramSelectionRestoreSeq) || 0) + 1;
    state.pendingParamSelectionRestore = {
        requestId: state._paramSelectionRestoreSeq,
        selection: selection ? selection.map((range) => ({ ...range })) : null,
    };
}

/**
 * 按撤销/重做/跳转载荷恢复参数编辑器选区。
 *
 * 【谁决定恢复哪一份】后端。选区快照记在**产生它的那一步**上
 * （`HistoryRecord::param_selection`），撤销/重做/跳转时后端算出"该恢复成什么样"
 * 并随载荷带回（`param_selection_restore`），前端无脑套用。
 *
 * 【为什么前端不再自己记账】旧实现让前端按撤销深度索引步骤（"我这一步是第几步"），
 * 而前端只有一个由事件异步推进的深度镜像：镜像滞后、写回被抑制、以及"新检查点
 * 丢弃重做分支"时的裁剪，都会让位置对不上 —— 表现是"撤销后曲线回来了、选区却没
 * 回到拉伸前"。现在位置推断整体消失，那条链路自然不存在。
 *
 * 特殊说明：`undefined`（载荷未携带）表示这一步与选区无关，**不得**改动用户当前
 * 选区；`[]` 表示当时确实没有选区（要清空）。两者语义不同，不能合并。
 *
 * 【单位是帧】每一对是 `[startFrame, frameCount]`（与选区的内部单位一致，见
 * `paramSelection.ts`）。帧与 BPM 无关，因此"拉伸 → 改 BPM → 撤销"不会再让恢复
 * 出来的选区落到别的时间上。
 *
 * @param payload 撤销/重做/跳转响应载荷。
 */
export function applyParamSelectionRestoreFromPayload(
    state: SessionState,
    payload: { param_selection_restore?: [number, number][] | null },
): void {
    const ranges = payload.param_selection_restore;
    if (!Array.isArray(ranges)) return;
    const snapshot = ranges
        .filter(
            (range) =>
                Array.isArray(range) && Number.isFinite(range[0]) && Number.isFinite(range[1]),
        )
        .map(([startFrame, frameCount]) => ({ startFrame, frameCount }));
    // 空数组与「无选区」是同一件事（快照类型用 null 表达），此处归一，
    // 消费方（参数编辑器）只需要认识"要恢复成这样"。
    requestParamSelectionRestore(state, snapshot.length > 0 ? snapshot : null);
}

export {
    undoRemote,
    redoRemote,
    setHistoryPositionRemote,
    setProjectSaveUndoHistoryRemote,
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
    saveProjectRemote,
    saveProjectAsRemote,
    saveProjectToPathRemote,
    setProjectBaseScaleRemote,
    setProjectCustomScaleRemote,
    setProjectStretchSettingsRemote,
    setProjectTimelineSettingsRemote,
} from "../thunks/projectThunks";

export {
    fetchTimeline,
    seekPlayhead,
    updateTransportBpm,
    syncPlaybackState,
    playOriginal,
    stopAudioPlayback,
} from "../thunks/transportThunks";

export {
    addTrackRemote,
    createHostTrackRemote,
    removeTrackRemote,
    moveTrackRemote,
    selectTrackRemote,
    setProjectLengthRemote,
    fetchSelectedTrackSummary,
    addClipOnTrack,
    createClipsRemote,
    pasteTimelineClipboardRemote,
    removeClipRemote,
    removeClipsRemote,
    moveClipRemote,
    moveClipsRemote,
    closeTrackGapsRemote,
    analyzeSilenceRemote,
    removeSilenceRemote,
    duplicateClipsBulkRemote,
    duplicateTrackRemote,
    setClipStateRemote,
    setClipsStateBulkRemote,
    setClipActiveTakeRemote,
    cycleClipTakesRemote,
    packClipsIntoTakesRemote,
    explodeClipTakesRemote,
    duplicateClipTakeRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipTakeReversedRemote,
    setClipTakeChannelModeRemote,
    scanAndConvertFakeStereoRemote,
    addClipTakeFromMediaRemote,
    replaceClipSourceRemote,
    replaceMidiClipDataRemote,
    splitClipRemote,
    splitClipsAtRemote,
    convertClipsToPitchReferenceRemote,
    updatePitchReferenceRemote,
    glueClipsRemote,
    selectClipRemote,
} from "../thunks/timelineThunks";

export { setTrackStateRemote, removeSelectedClipRemote } from "../thunks/trackThunks";

export {
    refreshRuntime,
    clearWaveformCacheRemote,
    clearRenderCacheRemote,
    loadUiSettings,
    persistUiSettings,
} from "../thunks/runtimeThunks";

export { updateMetronome } from "../thunks/transportThunks";

export { loadModel, loadDefaultModel } from "../thunks/modelThunks";

export {
    processAudio,
    pickOutputPath,
    applyPitchShift,
    synthesizeAudio,
    exportAudio,
    exportAudioAdvanced,
    exportSeparated,
    pasteVocalShifterClipboard,
    pasteReaperClipboard,
} from "../thunks/audioThunks";

export {
    importAudioFromDialog,
    importAudioFromPath,
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMidiAsClip,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "../thunks/importThunks";

/**
 * 把一次工具选择写进状态。
 *
 * 【为什么抽成函数】`setToolMode`（用户切换）与 `loadUiSettings.fulfilled`
 * （启动恢复）必须落到**同一组**字段上：`toolMode` 是当前工具，`toolModeGroup`
 * 与 `drawToolMode` 是它的派生视图（"这一组里选的是哪个绘制工具"供 `Tab` 回跳）。
 * 两处各写一遍，迟早会有一处漏掉 `drawToolMode`，表现为"按 Tab 回不到刚才那个工具"。
 */
export function applyToolMode(state: SessionState, mode: ToolMode): void {
    state.toolMode = mode;
    if (mode === "select") {
        state.toolModeGroup = "select";
    } else {
        state.toolModeGroup = "draw";
        state.drawToolMode = mode;
    }
}
