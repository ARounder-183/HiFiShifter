import { createSlice, createAsyncThunk, type PayloadAction } from "@reduxjs/toolkit";
import { fileBrowserApi, type FileEntry } from "../../services/api/fileBrowser";

export type SortMode = "name" | "date" | "size";

interface FileBrowserState {
    visible: boolean;
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
    sortMode: SortMode;
    audioOnly: boolean; // 仅显示音频文件
}

const STORAGE_KEY = "hifishifter.fileBrowser.lastPath";
const AUDIO_ONLY_KEY = "hifishifter.fileBrowser.audioOnly";

function getInitialPath(): string {
    return localStorage.getItem(STORAGE_KEY) || "";
}

const initialState: FileBrowserState = {
    visible: false,
    currentPath: getInitialPath(),
    entries: [],
    loading: false,
    error: null,
    previewVolume: 0.8,
    previewingFile: null,
    searchQuery: "",
    searchResults: null,
    searchLoading: false,
    regexEnabled: false,
    sortMode: "name" as SortMode,
    audioOnly: localStorage.getItem(AUDIO_ONLY_KEY) === "true",
};

export const loadDirectory = createAsyncThunk(
    "fileBrowser/loadDirectory",
    async (dirPath: string, { rejectWithValue }) => {
        try {
            const entries = await fileBrowserApi.listDirectory(dirPath);
            return { dirPath, entries };
        } catch (err) {
            return rejectWithValue(err instanceof Error ? err.message : "Failed to load directory");
        }
    },
);

export const searchFilesRecursive = createAsyncThunk(
    "fileBrowser/searchFilesRecursive",
    async ({ dirPath, query }: { dirPath: string; query: string }, { rejectWithValue }) => {
        try {
            const entries = await fileBrowserApi.searchFilesRecursive(dirPath, query);
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
        toggleVisible(state) {
            state.visible = !state.visible;
        },
        setVisible(state, action: PayloadAction<boolean>) {
            state.visible = action.payload;
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
        setSortMode(state, action: PayloadAction<SortMode>) {
            state.sortMode = action.payload;
        },
        toggleAudioOnly(state) {
            state.audioOnly = !state.audioOnly;
            localStorage.setItem(AUDIO_ONLY_KEY, String(state.audioOnly));
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(loadDirectory.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(loadDirectory.fulfilled, (state, action) => {
                state.loading = false;
                state.currentPath = action.payload.dirPath;
                state.entries = action.payload.entries;
                localStorage.setItem(STORAGE_KEY, action.payload.dirPath);
            })
            .addCase(loadDirectory.rejected, (state, action) => {
                state.loading = false;
                state.error = String(action.payload ?? "Unknown error");
            })
            .addCase(searchFilesRecursive.pending, (state) => {
                state.searchLoading = true;
            })
            .addCase(searchFilesRecursive.fulfilled, (state, action) => {
                state.searchLoading = false;
                state.searchResults = action.payload;
            })
            .addCase(searchFilesRecursive.rejected, (state) => {
                state.searchLoading = false;
                state.searchResults = [];
            });
    },
});

export const {
    toggleVisible,
    setVisible,
    setPreviewVolume,
    setPreviewingFile,
    setSearchQuery,
    toggleRegex,
    setSortMode,
    toggleAudioOnly,
} = fileBrowserSlice.actions;

export default fileBrowserSlice.reducer;
