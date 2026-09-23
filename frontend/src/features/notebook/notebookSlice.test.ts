import { test } from "vitest";

import reducer, {
    closeNotebook,
    openNotebook,
    setNotebookMode,
    toggleNotebookVisible,
} from "./notebookSlice.ts";

test("features/notebook/notebookSlice.test.ts scripted checks", async () => {
    function assertEqual<T>(actual: T, expected: T, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    let state = reducer(undefined, { type: "@@INIT" });
    assertEqual(state.visible, false, "initial notebook visibility");
    // 默认富文本：打开记事本先看到排好版的文档，而不是 Markdown 源码。
    assertEqual(state.mode, "rich", "initial notebook mode");

    state = reducer(state, toggleNotebookVisible());
    assertEqual(state.visible, true, "toggle opens notebook");

    state = reducer(state, setNotebookMode("source"));
    assertEqual(state.mode, "source", "can switch to markdown source");

    state = reducer(state, setNotebookMode("split"));
    assertEqual(state.mode, "split", "can switch to split view");

    state = reducer(state, closeNotebook());
    assertEqual(state.visible, false, "closeNotebook hides panel");

    state = reducer(state, openNotebook());
    assertEqual(state.visible, true, "openNotebook shows panel");
});
