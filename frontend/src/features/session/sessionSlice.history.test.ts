import { test } from "vitest";

import reducer, {
    recordParamSelectionStretchStep,
    setHistoryState,
    type ParamSelectionSnapshot,
} from "./sessionSlice.js";
import { redoRemote, setHistoryPositionRemote, undoRemote } from "./thunks/projectThunks.js";

/**
 * 撤销/重做的前端侧契约：
 * - 深度镜像（后端 history_state / 撤销·重做响应）驱动菜单置灰与快捷键前置判断；
 * - 空栈响应（ok = false）不得套用任何快照 —— 界面零刷新零变更；
 * - 只有「参数编辑器边缘拉伸」登记的步骤会在撤销/重做时恢复选区。
 */
test("features/session/sessionSlice.history.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        const actualJson = JSON.stringify(actual);
        const expectedJson = JSON.stringify(expected);
        if (actualJson !== expectedJson) {
            throw new Error(`${label}: expected ${expectedJson}, received ${actualJson}`);
        }
    }

    function createState(): ReturnType<typeof reducer> {
        return reducer(undefined, { type: "@@INIT" });
    }

    /** 撤销/重做响应：后端权威快照（这里只关心 ok 与深度字段）。 */
    function timelinePayload(overrides: Record<string, unknown>) {
        return {
            ok: true,
            tracks: [],
            clips: [],
            selected_track_id: null,
            selected_clip_id: null,
            bpm: 120,
            playhead_sec: 0,
            ...overrides,
        } as never;
    }

    function applyUndo(
        state: ReturnType<typeof reducer>,
        payload: Record<string, unknown>,
        requestId: string,
    ): ReturnType<typeof reducer> {
        const pending = reducer(state, undoRemote.pending(requestId, undefined));
        return reducer(pending, undoRemote.fulfilled(timelinePayload(payload), requestId, undefined));
    }

    function applyRedo(
        state: ReturnType<typeof reducer>,
        payload: Record<string, unknown>,
        requestId: string,
    ): ReturnType<typeof reducer> {
        const pending = reducer(state, redoRemote.pending(requestId, undefined));
        return reducer(pending, redoRemote.fulfilled(timelinePayload(payload), requestId, undefined));
    }

    const beforeSelection: ParamSelectionSnapshot = [{ startBeat: 2, endBeat: 4 }];
    const afterSelection: ParamSelectionSnapshot = [{ startBeat: 2, endBeat: 6 }];

    // ── 深度镜像 ──
    {
        const seeded = reducer(
            createState(),
            setHistoryState({ undoDepth: 3, redoDepth: 1 }),
        );
        assertEqual(seeded.historyUndoDepth, 3, "undo depth mirrored");
        assertEqual(seeded.historyRedoDepth, 1, "redo depth mirrored");
    }

    // ── 空栈撤销：不套用快照、界面零变更（仅同步深度），无选区恢复请求 ──
    {
        const seeded = reducer(createState(), setHistoryState({ undoDepth: 0, redoDepth: 2 }));
        const next = applyUndo(
            seeded,
            { ok: false, tracks: [], clips: [], undo_depth: 0, redo_depth: 2 },
            "req-undo-empty",
        );
        assertEqual(next.historyUndoDepth, 0, "empty undo keeps undo depth");
        assertEqual(next.historyRedoDepth, 2, "empty undo keeps redo depth");
        assertEqual(next.pendingParamSelectionRestore, null, "empty undo requests no restore");
        assertEqual(next.clips.length, seeded.clips.length, "empty undo leaves clips untouched");
    }

    // ── 拉伸步骤：撤销恢复拉伸前、重做恢复拉伸后的选区 ──
    {
        // 拉伸发生时深度 3 → 回写打点后深度 4。
        let state = reducer(createState(), setHistoryState({ undoDepth: 3, redoDepth: 0 }));
        state = reducer(
            state,
            recordParamSelectionStretchStep({
                positionBefore: 3,
                before: beforeSelection,
                after: afterSelection,
            }),
        );
        // 新检查点：深度 4。
        state = reducer(state, setHistoryState({ undoDepth: 4, redoDepth: 0 }));

        // 撤销这一步（4 → 3）：恢复拉伸前的选区。
        const undone = applyUndo(state, { undo_depth: 3, redo_depth: 1 }, "req-undo-stretch");
        assertEqual(undone.historyUndoDepth, 3, "undo depth after stretch undo");
        assertEqual(undone.historyRedoDepth, 1, "redo depth after stretch undo");
        assertEqual(
            undone.pendingParamSelectionRestore?.selection,
            beforeSelection,
            "undo restores pre-stretch selection",
        );
        assertEqual(
            undone.pendingParamSelectionRestore?.requestId,
            1,
            "restore request id increments",
        );

        // 重做（3 → 4）：恢复拉伸后的选区。
        const redone = applyRedo(undone, { undo_depth: 4, redo_depth: 0 }, "req-redo-stretch");
        assertEqual(redone.historyUndoDepth, 4, "undo depth after stretch redo");
        assertEqual(redone.historyRedoDepth, 0, "redo depth after stretch redo");
        assertEqual(
            redone.pendingParamSelectionRestore?.selection,
            afterSelection,
            "redo restores post-stretch selection",
        );
        assertEqual(
            redone.pendingParamSelectionRestore?.requestId,
            2,
            "second restore request uses a new id",
        );
    }

    // ── 其它步骤的撤销/重做不触碰选区（无恢复请求） ──
    {
        let state = reducer(createState(), setHistoryState({ undoDepth: 2, redoDepth: 0 }));
        state = reducer(
            state,
            recordParamSelectionStretchStep({
                positionBefore: 1,
                before: beforeSelection,
                after: afterSelection,
            }),
        );
        state = reducer(state, setHistoryState({ undoDepth: 3, redoDepth: 0 }));
        // 撤销第 3 步（非拉伸步骤）：不产生恢复请求。
        const undone = applyUndo(state, { undo_depth: 2, redo_depth: 1 }, "req-undo-other");
        assertEqual(
            undone.pendingParamSelectionRestore,
            null,
            "unrelated undo does not touch selection",
        );
    }

    // ── 新检查点截断重做分支：被丢弃分支上的步骤作废 ──
    {
        let state = reducer(createState(), setHistoryState({ undoDepth: 3, redoDepth: 0 }));
        state = reducer(
            state,
            recordParamSelectionStretchStep({
                positionBefore: 3,
                before: beforeSelection,
                after: afterSelection,
            }),
        );
        state = reducer(state, setHistoryState({ undoDepth: 4, redoDepth: 0 }));
        const undone = applyUndo(state, { undo_depth: 3, redo_depth: 1 }, "req-undo-branch");
        assertEqual(undone.paramSelectionSteps.length, 1, "step kept after undo (redo available)");
        // 撤销后做了别的编辑：新检查点清空重做分支（4），位在其上的旧步骤作废。
        const truncated = reducer(
            undone,
            setHistoryState({ undoDepth: 4, redoDepth: 0 }),
        );
        assertEqual(truncated.paramSelectionSteps.length, 0, "discarded branch steps pruned");
        // 再次撤到 4 的位置不应产生新的恢复请求（已有请求的 requestId 不变，
        // 消费方按 id 幂等应用，不会重放旧选区）。
        const requestIdBefore = truncated.pendingParamSelectionRestore?.requestId ?? 0;
        const undoneAgain = applyUndo(
            truncated,
            { undo_depth: 3, redo_depth: 1 },
            "req-undo-after-truncate",
        );
        assertEqual(
            undoneAgain.pendingParamSelectionRestore?.requestId ?? 0,
            requestIdBefore,
            "no stale selection restore after branch truncation",
        );
    }

    // ── 历史清空（新建 / 打开工程）：深度归零且步骤全部作废 ──
    {
        let state = reducer(createState(), setHistoryState({ undoDepth: 2, redoDepth: 0 }));
        state = reducer(
            state,
            recordParamSelectionStretchStep({
                positionBefore: 2,
                before: beforeSelection,
                after: afterSelection,
            }),
        );
        state = reducer(state, setHistoryState({ undoDepth: 0, redoDepth: 0 }));
        assertEqual(state.historyUndoDepth, 0, "history reset clears undo depth");
        assertEqual(state.paramSelectionSteps.length, 0, "history reset clears recorded steps");
    }

    // ── 「操作记录」跳转：落到拉伸步骤的状态恢复选区；越界静默跳过 ──
    {
        let state = reducer(createState(), setHistoryState({ undoDepth: 1, redoDepth: 0 }));
        state = reducer(
            state,
            recordParamSelectionStretchStep({
                positionBefore: 1,
                before: beforeSelection,
                after: afterSelection,
            }),
        );
        state = reducer(state, setHistoryState({ undoDepth: 2, redoDepth: 0 }));

        // 跳到位置 2（拉伸后的状态）→ 恢复拉伸后的选区。
        const pending = reducer(state, setHistoryPositionRemote.pending("req-jump", 2));
        const jumped = reducer(
            pending,
            setHistoryPositionRemote.fulfilled(
                timelinePayload({ undo_depth: 2, redo_depth: 0 }),
                "req-jump",
                2,
            ),
        );
        assertEqual(
            jumped.pendingParamSelectionRestore?.selection,
            afterSelection,
            "jump to stretch state restores post-stretch selection",
        );

        // 越界（后端 ok = false）：不套用快照、不产生新的恢复请求。
        const requestIdBefore = jumped.pendingParamSelectionRestore?.requestId ?? 0;
        const pendingOob = reducer(jumped, setHistoryPositionRemote.pending("req-jump-oob", 9));
        const jumpedOob = reducer(
            pendingOob,
            setHistoryPositionRemote.fulfilled(
                timelinePayload({ ok: false, undo_depth: 2, redo_depth: 0 }),
                "req-jump-oob",
                9,
            ),
        );
        assertEqual(
            jumpedOob.pendingParamSelectionRestore?.requestId ?? 0,
            requestIdBefore,
            "out-of-range jump is a silent no-op",
        );
    }
});
