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
    assertEqual(state.mode, "edit", "initial notebook mode");

    state = reducer(state, toggleNotebookVisible());
    assertEqual(state.visible, true, "toggle opens notebook");

    state = reducer(state, setNotebookMode("preview"));
    assertEqual(state.mode, "preview", "can switch notebook mode");

    state = reducer(state, closeNotebook());
    assertEqual(state.visible, false, "closeNotebook hides panel");

    state = reducer(state, openNotebook());
    assertEqual(state.visible, true, "openNotebook shows panel");

});
