import { test } from "vitest";

import reducer, { setNotebookMode, setNotebookSettings } from "./notebookSlice.ts";
import { DEFAULT_NOTEBOOK_SETTINGS } from "../../components/layout/notebook/notebookSettings.ts";

/*
 * 记事本切片只保存**面板内部**状态：视图模式、设置、附件索引。
 *
 * 显隐（打开/关闭/停靠位置）由停靠布局统一持有 —— 曾经这里的 `visible` 布尔
 * 与布局树并存，两份事实源会漂移成"按钮亮着但面板不在屏幕上"。现在可见性由
 * `isFormVisible` 从布局树派生，本切片不再有它。
 */
test("features/notebook/notebookSlice.test.ts scripted checks", async () => {
    function assertEqual<T>(actual: T, expected: T, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    let state = reducer(undefined, { type: "@@INIT" });
    // 默认富文本：打开记事本先看到排好版的文档，而不是 Markdown 源码。
    assertEqual(state.mode, "rich", "initial notebook mode");
    assertEqual(
        state.settings.panelWidth,
        DEFAULT_NOTEBOOK_SETTINGS.panelWidth,
        "settings start at their defaults",
    );

    state = reducer(state, setNotebookMode("source"));
    assertEqual(state.mode, "source", "can switch to markdown source");

    state = reducer(state, setNotebookMode("split"));
    assertEqual(state.mode, "split", "can switch to split view");

    // 设置写入必须过归一化：越界值被钳制，非法类型退回默认。
    state = reducer(state, setNotebookSettings({ sourceFontSize: -5 }));
    assertEqual(state.settings.sourceFontSize, 9, "out-of-range font size is clamped");

    state = reducer(state, setNotebookSettings({ sourceFontSize: 999 }));
    assertEqual(state.settings.sourceFontSize, 24, "font size is clamped high too");

    state = reducer(state, setNotebookSettings({ showToolbar: "yes" as unknown as boolean }));
    assertEqual(
        state.settings.showToolbar,
        DEFAULT_NOTEBOOK_SETTINGS.showToolbar,
        "non-boolean falls back to the default",
    );

    state = reducer(state, setNotebookSettings(null));
    assertEqual(
        state.settings.showToolbar,
        DEFAULT_NOTEBOOK_SETTINGS.showToolbar,
        "null settings reset to defaults",
    );
});
