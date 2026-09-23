import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

import {
    DEFAULT_NOTEBOOK_SETTINGS,
    normalizeNotebookSettings,
    type NotebookSettings,
    type ResolvedNotebookSettings,
} from "../../components/layout/notebook/notebookSettings";

/**
 * 记事本的视图模式。
 *
 * - `rich`（默认）：**富文本**。Markdown 的交互式所见即所得编辑 ——
 *   输入 `## ` 直接变标题、`- ` 变列表，图片与剪贴板暂存块以卡片呈现。
 * - `source`：**Markdown 源码**。等宽纯文本，所见即源码，用于精确控制。
 * - `split`：**分栏**。左富文本、右只读渲染，滚动联动。
 *
 * 【为什么默认是富文本而不是源码】纯文本编辑是"专家模式"，用户打开记事本
 * 时看到的应当是排好版的文档，而不是一堆 `#` 和 `*`。
 */
export type NotebookMode = "rich" | "source" | "split";

type NotebookState = {
    visible: boolean;
    mode: NotebookMode;
    /**
     * 已归一化的设置。
     *
     * 放在这里而不是 `session`：设置由面板自己按需拉取与回写（见
     * `NotebookPanel` 的挂载副作用），不参与工程会话的撤销/重做，
     * 也不该出现在工程文件里。
     */
    settings: ResolvedNotebookSettings;
    /**
     * 附件索引（id → 摘要），用于暂存块判断"载荷还在不在"、附件管理器列表。
     *
     * 只存摘要不存字节：正文里可能有几十个引用，逐个 `readAsset` 会把图片
     * 数据全部读进内存 —— 而这里只需要回答"存在吗、多大、什么类型"。
     */
    assetIndex: Record<string, NotebookAssetSummary>;
};

export interface NotebookAssetSummary {
    id: string;
    kind: "image" | "clip_payload";
    ext: string;
    mime: string;
    byteLen: number;
    createdAtMs: number;
    meta: unknown;
    exists: boolean;
}

const initialState: NotebookState = {
    visible: false,
    mode: "rich",
    settings: DEFAULT_NOTEBOOK_SETTINGS,
    assetIndex: {},
};

const notebookSlice = createSlice({
    name: "notebook",
    initialState,
    reducers: {
        toggleNotebookVisible(state) {
            state.visible = !state.visible;
        },
        openNotebook(state) {
            state.visible = true;
        },
        closeNotebook(state) {
            state.visible = false;
        },
        setNotebookMode(state, action: PayloadAction<NotebookMode>) {
            state.mode = action.payload;
        },
        /** 用后端返回的设置整体替换（挂载时调用一次）。 */
        setNotebookSettings(state, action: PayloadAction<NotebookSettings | null | undefined>) {
            state.settings = normalizeNotebookSettings(action.payload);
        },
        /** 局部更新（面板内的设置对话框调用，随后由面板回写后端）。 */
        patchNotebookSettings(state, action: PayloadAction<NotebookSettings>) {
            state.settings = normalizeNotebookSettings({ ...state.settings, ...action.payload });
        },
        /** 用后端返回的附件清单整体替换索引。 */
        setNotebookAssetIndex(state, action: PayloadAction<NotebookAssetSummary[]>) {
            const next: Record<string, NotebookAssetSummary> = {};
            // 载荷形状来自 IPC：这里做一次防御性判断。reducer 里抛异常会
            // 直接把 dispatch 的调用方带崩，而附件索引本来就只是缓存。
            if (Array.isArray(action.payload)) {
                for (const entry of action.payload) {
                    if (entry && typeof entry.id === "string") next[entry.id] = entry;
                }
            }
            state.assetIndex = next;
        },
    },
});

export const {
    toggleNotebookVisible,
    openNotebook,
    closeNotebook,
    setNotebookMode,
    setNotebookSettings,
    patchNotebookSettings,
    setNotebookAssetIndex,
} = notebookSlice.actions;

export default notebookSlice.reducer;
