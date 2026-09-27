/**
 * 导入在途计数的契约（驱动状态栏那个"正在导入"提示）。
 *
 * 【要钉死的性质】
 * 1. 只有**会去读音频文件**的导入 thunk 才计数 —— 别的操作置起的 `busy` 不该
 *    让界面显示"正在导入"。
 * 2. 是**计数**不是布尔：对话框导入会把位置导入转派出去，两者同时在途。
 * 3. 无论如何都要回到 0 —— 一个永久挂着的计数会让提示再也关不掉。
 */
import { test } from "vitest";

import reducer from "./sessionSlice.js";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importAudioFromDialog,
    importAudioFromPath,
    importMultipleAudioAtPosition,
    importMultipleAudioFilesAtPosition,
} from "./thunks/importThunks.js";
import { updateTransportBpm } from "./thunks/transportThunks.js";

test("features/session/sessionSlice.importInFlight.test.ts scripted checks", () => {
    type State = ReturnType<typeof reducer>;
    function createState(): State {
        return reducer(undefined, { type: "@@INIT" });
    }

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    function inFlight(state: State): number {
        return (state as unknown as { importInFlight: number }).importInFlight;
    }

    // 初始为 0：没有任何导入在途时提示不该亮。
    assertEqual(inFlight(createState()), 0, "初始在途计数");

    // 单个导入：pending 加一，fulfilled 回到 0。
    let state = reducer(createState(), importAudioAtPosition.pending("req-1", { audioPath: "a.wav" }));
    assertEqual(inFlight(state), 1, "pending 后计数");
    state = reducer(
        state,
        importAudioAtPosition.fulfilled(
            { ok: true, imported: {} as never, newClipIds: [], playheadSec: undefined },
            "req-1",
            { audioPath: "a.wav" },
        ),
    );
    assertEqual(inFlight(state), 0, "fulfilled 后计数");

    // rejected 同样要收尾，否则一次失败的导入会让提示永久停留。
    state = reducer(createState(), importAudioFromPath.pending("req-2", "a.wav"));
    assertEqual(inFlight(state), 1, "rejected 用例的 pending");
    state = reducer(
        state,
        importAudioFromPath.rejected(new Error("boom"), "req-2", "a.wav"),
    );
    assertEqual(inFlight(state), 0, "rejected 后计数");

    // 嵌套：对话框导入转派位置导入 ⇒ 两条同时在途，用布尔会提前清零。
    state = reducer(createState(), importAudioFromDialog.pending("req-3", undefined));
    state = reducer(state, importAudioAtPosition.pending("req-4", { audioPath: "b.wav" }));
    assertEqual(inFlight(state), 2, "嵌套导入的在途计数");
    state = reducer(
        state,
        importAudioAtPosition.fulfilled(
            { ok: true, imported: {} as never, newClipIds: [], playheadSec: undefined },
            "req-4",
            { audioPath: "b.wav" },
        ),
    );
    assertEqual(inFlight(state), 1, "内层结束后仍认为在导入");
    state = reducer(
        state,
        importAudioFromDialog.fulfilled({ ok: true, canceled: false }, "req-3", undefined),
    );
    assertEqual(inFlight(state), 0, "两层都结束后归零");

    // 每个导入 thunk 都要被覆盖到：漏一个就会出现"正在导入却不提示"。
    const covered = [
        importAudioFileAtPosition,
        importAudioFromPath,
        importMultipleAudioAtPosition,
        importMultipleAudioFilesAtPosition,
    ];
    for (const thunk of covered) {
        const opened = reducer(createState(), thunk.pending("req-x", undefined as never));
        assertEqual(inFlight(opened), 1, `${thunk.typePrefix} 必须计入在途`);
        const closed = reducer(opened, thunk.rejected(new Error("x"), "req-x", undefined as never));
        assertEqual(inFlight(closed), 0, `${thunk.typePrefix} 必须能收尾`);
    }

    // 无关操作不得影响计数，否则提示会误报。
    const untouched = reducer(createState(), updateTransportBpm.pending("req-9", 140));
    assertEqual(inFlight(untouched), 0, "无关 thunk 不得影响在途计数");
});
