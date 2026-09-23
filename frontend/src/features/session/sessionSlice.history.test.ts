import { test } from "vitest";

import reducer, { setHistoryState, type ParamSelectionSnapshot } from "./sessionSlice.js";
import { redoRemote, setHistoryPositionRemote, undoRemote } from "./thunks/projectThunks.js";

/**
 * 撤销/重做的前端侧契约：
 * - 深度镜像（后端 history_state / 撤销·重做响应）驱动菜单置灰与快捷键前置判断；
 * - 空栈响应（ok = false）不得套用任何快照 —— 界面零刷新零变更；
 * - **选区恢复完全由载荷驱动**：后端把「边缘拉伸」那一步的选区快照随撤销/重做/
 *   跳转的响应带回（`param_selection_restore`），前端只负责转成一次恢复请求。
 *
 * 【为什么不再有"按撤销深度索引步骤"的用例】旧实现让前端自己按深度记账，需要
 * 回答"我这一步是第几步"，而它只有一个由事件异步推进的深度镜像：镜像滞后、写回
 * 被抑制、以及"新检查点丢弃重做分支"时的裁剪都会让位置错位，表现为"撤销后曲线
 * 回来了、选区却没回到拉伸前"。现在位置推断整体消失（见 state.rs 的
 * `HistoryRecord::param_selection`），因此这里的用例只验证载荷 → 恢复请求。
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

    /** 撤销/重做响应：后端权威快照（这里只关心 ok、深度与选区恢复字段）。 */
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

    /** 把 beat 选区写成后端载荷里的 `[[startBeat, endBeat], …]` 形态。 */
    const asPayloadRanges = (selection: ParamSelectionSnapshot): [number, number][] =>
        selection.map((range) => [range.startBeat, range.endBeat]);

    function applyUndo(
        state: ReturnType<typeof reducer>,
        payload: Record<string, unknown>,
        requestId: string,
    ): ReturnType<typeof reducer> {
        const pending = reducer(state, undoRemote.pending(requestId, undefined));
        return reducer(
            pending,
            undoRemote.fulfilled(timelinePayload(payload), requestId, undefined),
        );
    }

    function applyRedo(
        state: ReturnType<typeof reducer>,
        payload: Record<string, unknown>,
        requestId: string,
    ): ReturnType<typeof reducer> {
        const pending = reducer(state, redoRemote.pending(requestId, undefined));
        return reducer(
            pending,
            redoRemote.fulfilled(timelinePayload(payload), requestId, undefined),
        );
    }

    const beforeSelection: ParamSelectionSnapshot = [{ startBeat: 2, endBeat: 4 }];
    const afterSelection: ParamSelectionSnapshot = [{ startBeat: 2, endBeat: 6 }];

    // ── 深度镜像 ──
    {
        const seeded = reducer(createState(), setHistoryState({ undoDepth: 3, redoDepth: 1 }));
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
        const state = reducer(createState(), setHistoryState({ undoDepth: 4, redoDepth: 0 }));

        const undone = applyUndo(
            state,
            {
                undo_depth: 3,
                redo_depth: 1,
                param_selection_restore: asPayloadRanges(beforeSelection),
            },
            "req-undo-stretch",
        );
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

        const redone = applyRedo(
            undone,
            {
                undo_depth: 4,
                redo_depth: 0,
                param_selection_restore: asPayloadRanges(afterSelection),
            },
            "req-redo-stretch",
        );
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

    // ── 「拉伸 → 撤销 → 再拉伸 → 撤销」：每次撤销都恢复拉伸前的选区 ──
    //
    // 这是该机制真正的使用场景（撤销一次是为了"回到原范围再拉一次"）。旧实现
    // 在这里会静默失效：第二次拉伸的检查点丢弃重做分支，前端按深度索引的旧步骤
    // 被裁掉、新登记的步骤又可能先于广播落地而被裁，撤销便再也找不到步骤。
    // 现在选区快照由后端随步骤持有，撤销只需套用载荷 —— 反复迭代都成立。
    {
        let state = reducer(createState(), setHistoryState({ undoDepth: 1, redoDepth: 0 }));
        const firstBefore: ParamSelectionSnapshot = [{ startBeat: 0, endBeat: 10 }];
        const secondAfter: ParamSelectionSnapshot = [{ startBeat: 0, endBeat: 18 }];

        state = applyUndo(
            state,
            {
                undo_depth: 0,
                redo_depth: 1,
                param_selection_restore: asPayloadRanges(firstBefore),
            },
            "req-iterate-undo-1",
        );
        assertEqual(
            state.pendingParamSelectionRestore?.selection,
            firstBefore,
            "first stretch undo restores the pre-stretch selection",
        );
        const idAfterFirst = state.pendingParamSelectionRestore?.requestId ?? 0;

        state = applyRedo(
            state,
            {
                undo_depth: 1,
                redo_depth: 0,
                param_selection_restore: asPayloadRanges(secondAfter),
            },
            "req-iterate-redo-2",
        );
        assertEqual(
            state.pendingParamSelectionRestore?.selection,
            secondAfter,
            "re-stretch redo restores the post-stretch selection",
        );

        state = applyUndo(
            state,
            {
                undo_depth: 0,
                redo_depth: 1,
                param_selection_restore: asPayloadRanges(firstBefore),
            },
            "req-iterate-undo-2",
        );
        assertEqual(
            (state.pendingParamSelectionRestore?.requestId ?? 0) > idAfterFirst,
            true,
            "second stretch undo issues a new restore request",
        );
        assertEqual(
            state.pendingParamSelectionRestore?.selection,
            firstBefore,
            "second stretch undo restores the pre-stretch selection too",
        );
    }

    // ── 其它步骤的撤销/重做不触碰选区（载荷不带该字段） ──
    {
        const state = reducer(createState(), setHistoryState({ undoDepth: 2, redoDepth: 0 }));
        const undone = applyUndo(state, { undo_depth: 1, redo_depth: 1 }, "req-undo-other");
        assertEqual(
            undone.pendingParamSelectionRestore,
            null,
            "unrelated undo does not touch selection",
        );
    }

    // ── 拉伸前本来就没有选区：载荷带空数组 → 恢复请求要求"清空" ──
    {
        const state = reducer(createState(), setHistoryState({ undoDepth: 1, redoDepth: 0 }));
        const undone = applyUndo(
            state,
            { undo_depth: 0, redo_depth: 1, param_selection_restore: [] },
            "req-undo-empty-selection",
        );
        assertEqual(
            undone.pendingParamSelectionRestore?.selection,
            null,
            "empty array means clear the selection",
        );
        assertEqual(
            (undone.pendingParamSelectionRestore?.requestId ?? 0) > 0,
            true,
            "clearing still issues a restore request",
        );
    }

    // ── 「操作记录」跳转：落到拉伸步骤的状态恢复选区；越界静默跳过 ──
    {
        const state = reducer(createState(), setHistoryState({ undoDepth: 2, redoDepth: 0 }));
        const pending = reducer(state, setHistoryPositionRemote.pending("req-jump", 2));
        const jumped = reducer(
            pending,
            setHistoryPositionRemote.fulfilled(
                timelinePayload({
                    undo_depth: 2,
                    redo_depth: 0,
                    param_selection_restore: asPayloadRanges(afterSelection),
                }),
                "req-jump",
                2,
            ),
        );
        assertEqual(
            jumped.pendingParamSelectionRestore?.selection,
            afterSelection,
            "jump to stretch state restores post-stretch selection",
        );

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

    // ── 历史清空（新建 / 打开工程）：深度归零 ──
    {
        const state = reducer(createState(), setHistoryState({ undoDepth: 2, redoDepth: 0 }));
        const cleared = reducer(state, setHistoryState({ undoDepth: 0, redoDepth: 0 }));
        assertEqual(cleared.historyUndoDepth, 0, "history reset clears undo depth");
        assertEqual(cleared.historyRedoDepth, 0, "history reset clears redo depth");
    }
});
