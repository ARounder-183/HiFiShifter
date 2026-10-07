import { createSlice, createAsyncThunk, type PayloadAction } from "@reduxjs/toolkit";
import { fileBrowserApi, type FileEntry } from "../../services/api/fileBrowser";
import { readUiValue, writeUiValue } from "../../services/uiStorage";
import type { SearchOptionsPayload } from "../search/searchSettings";

/**
 * 「计算机」虚拟路径：Windows 盘符根（`C:\`）的上一级，列表内容是全部盘符。
 *
 * 后端 `list_directory` 识别这个哨兵值并返回盘符清单，其余命令不应收到它 ——
 * 递归搜索在「计算机」层没有意义，调用方需先做守卫。非 Windows 平台没有
 * 这一层（`/` 已是文件系统顶端），永远不会导航到它。
 */
export const FILE_BROWSER_COMPUTER_PATH = "computer://";

/**
 * 本分片的状态类型。
 *
 * 导出是因为 `RootState` 由它组合而成 —— SDK 的声明产出需要能命名它
 * （否则 `tsc --emitDeclarationOnly` 报 TS4023「cannot be named」）。
 *
 * 【为什么这里没有排序 / 仅媒体 / 隐藏文件】它们是**用户偏好**，随
 * `app_config.json` 一起备份与迁移，因此住在 `session.fileBrowserView`
 * （见 `features/fileBrowser/fileBrowserViewOptions.ts`）。本分片只保留
 * "当前这一次浏览"的状态：在哪个目录、列表内容、搜索词、试听对象。
 */
export interface FileBrowserState {
    currentPath: string;
    entries: FileEntry[];
    loading: boolean;
    error: string | null;
    previewVolume: number; // 0~1
    previewingFile: string | null;
    searchQuery: string; // 搜索过滤关键词
    searchResults: FileEntry[] | null; // null = 非搜索模式
    searchLoading: boolean;
    regexEnabled: boolean;
    // 最近一次目录/搜索请求的 requestId：快速连续导航/搜索时，迟到的旧
    // 响应若不丢弃，会把面板拉回用户已离开的目录或过期的搜索结果。
    latestLoadRequestId: string | null;
    latestSearchRequestId: string | null;
}

const STORAGE_KEY = "hifishifter.fileBrowser.lastPath";

/*
 * 存储访问全部包在 try/catch 里。
 *
 * 【为什么不能直接调 localStorage】分片在**模块加载期**就要读一次初始路径；在没有
 * localStorage 的环境（node 环境的单测、未来的非浏览器宿主）里，直接调用会在
 * `import` 那一刻抛错 —— 于是"只想引用一个常量"的模块也被连坐（本模块的
 * `FILE_BROWSER_COMPUTER_PATH` 正是被路径工具模块引用的）。存储不可用只应意味着
 * "没有记住上次的目录"，不是加载失败。
 */
function readStoredPath(): string {
    try {
        return readUiValue(STORAGE_KEY) || "";
    } catch {
        return "";
    }
}

function writeStoredPath(path: string): void {
    try {
        writeUiValue(STORAGE_KEY, path);
    } catch {
        /* 存储不可用：本次会话仍然正常工作，只是下次启动不记得 */
    }
}

const initialState: FileBrowserState = {
    currentPath: readStoredPath(),
    entries: [],
    loading: false,
    error: null,
    previewVolume: 0.8,
    previewingFile: null,
    searchQuery: "",
    searchResults: null,
    searchLoading: false,
    regexEnabled: false,
    latestLoadRequestId: null,
    latestSearchRequestId: null,
};

/**
 * 加载目录。
 *
 * 【为什么从 state 里读"是否显示隐藏文件"而不是由调用方传参】它是全局视图选项，
 * 每个调用点都传一遍等于把同一个决定抄六份；而这里只需要一个布尔量，走
 * `getState()` 读一次比给六个调用点各加一个参数更不容易漏。
 *
 * 用结构化类型而不是 `RootState`，避免分片与 store 之间产生类型环。
 */
export const loadDirectory = createAsyncThunk(
    "fileBrowser/loadDirectory",
    async (dirPath: string, { rejectWithValue, getState }) => {
        const state = getState() as {
            session?: { fileBrowserView?: { showHiddenFiles?: boolean } };
        };
        const includeHidden = state.session?.fileBrowserView?.showHiddenFiles === true;
        try {
            const entries = await fileBrowserApi.listDirectory(
                dirPath,
                includeHidden ? { includeHidden } : undefined,
            );
            return { dirPath, entries };
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "Failed to load directory");
        }
    },
);

export const searchFilesRecursive = createAsyncThunk(
    "fileBrowser/searchFilesRecursive",
    async (
        {
            dirPath,
            query,
            options,
        }: { dirPath: string; query: string; options?: SearchOptionsPayload },
        { rejectWithValue },
    ) => {
        try {
            const entries = await fileBrowserApi.searchFilesRecursive(dirPath, query, options);
            return entries;
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "Search failed");
        }
    },
);

const fileBrowserSlice = createSlice({
    name: "fileBrowser",
    initialState,
    reducers: {
        /** 原生选择器失败必须可见，不能让用户点击文件夹后没有任何反应。 */
        setFileBrowserError(state, action: PayloadAction<string | null>) {
            state.error = action.payload;
        },
        setPreviewVolume(state, action: PayloadAction<number>) {
            state.previewVolume = Math.max(0, Math.min(1, action.payload));
        },
        setPreviewingFile(state, action: PayloadAction<string | null>) {
            state.previewingFile = action.payload;
        },
        setSearchQuery(state, action: PayloadAction<string>) {
            state.searchQuery = action.payload;
            if (!action.payload.trim()) {
                state.searchResults = null;
                state.searchLoading = false;
            }
        },
        toggleRegex(state) {
            state.regexEnabled = !state.regexEnabled;
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(loadDirectory.pending, (state, action) => {
                state.loading = true;
                state.error = null;
                state.latestLoadRequestId = action.meta.requestId;
            })
            .addCase(loadDirectory.fulfilled, (state, action) => {
                // 只接受最新一次请求的结果：进入慢目录 A 后又快速进入 B 时，
                // A 的迟到响应不得覆盖 B 的列表。
                if (action.meta.requestId !== state.latestLoadRequestId) return;
                state.loading = false;
                state.currentPath = action.payload.dirPath;
                state.entries = action.payload.entries;
                writeStoredPath(action.payload.dirPath);
            })
            .addCase(loadDirectory.rejected, (state, action) => {
                if (action.meta.requestId !== state.latestLoadRequestId) return;
                state.loading = false;
                state.error = String(action.payload ?? "Unknown error");
            })
            .addCase(searchFilesRecursive.pending, (state, action) => {
                state.searchLoading = true;
                state.latestSearchRequestId = action.meta.requestId;
            })
            .addCase(searchFilesRecursive.fulfilled, (state, action) => {
                if (action.meta.requestId !== state.latestSearchRequestId) return;
                state.searchLoading = false;
                state.searchResults = action.payload;
            })
            .addCase(searchFilesRecursive.rejected, (state, action) => {
                if (action.meta.requestId !== state.latestSearchRequestId) return;
                state.searchLoading = false;
                state.searchResults = [];
                // 失败必须可见：静默置空会让用户误以为“确实无匹配”。
                state.error = String(action.payload ?? "Search failed");
            });
    },
});

export const {
    setPreviewVolume,
    setPreviewingFile,
    setSearchQuery,
    toggleRegex,
    setFileBrowserError,
} = fileBrowserSlice.actions;

export default fileBrowserSlice.reducer;
