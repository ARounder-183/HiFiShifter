/**
 * 跨窗口瞬态交互状态。
 *
 * 【为什么需要这一片】独立窗口是另一个 JS 上下文：模块级 bus / Map / boolean
 * 在卫星窗口里各有一份。此前「时间轴发布手势快照 → 参数编辑器读取」走
 * clipGeometryPreviewBus，内置窗口正确、独立窗口永远读不到。把需要跨窗口的
 * 瞬态状态放进 Redux 后，它自然复用 detachBridge 的快照与动作复制：
 * - 动作顺序与乐观 clip 更新在同一队列里，begin preview 先于第一次几何写入；
 * - 卫星晚加入时，快照自带当前手势；
 * - 不需要第二条 IPC 协议，也不需要回声/重试逻辑。
 *
 * 【什么不该放这里】持久 / 可撤销 / 工程数据仍归 session；同窗口逐帧视觉信号
 * （视口、播放头插值）仍归模块级 bus。这里只放「跨窗口且短生命周期」的状态。
 */

import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

import type { ClipGeometrySnapshot, ClipSourceGeometry } from "./clipGeometryTypes";

/** 参数编辑器的选区摘要（MenuBar 判断工程音阶影响范围时使用）。 */
export interface PianoRollSelectionInfo {
    /** 选区起始帧（绝对帧）。 */
    startFrame: number;
    /** 选区帧数。 */
    frameCount: number;
    /** 帧周期（毫秒）。 */
    framePeriodMs: number;
}

/** 几何手势起点的完整投影（见 timeline/clipGeometryPreview.ts）。 */
export interface ClipGeometryPreviewState {
    /** 本手势的唯一标识；用于丢弃迟到的 end。 */
    token: string;
    /** 按下时的几何快照（loudnessGeometryWarp 用）。 */
    origin: ClipGeometrySnapshot[];
    /** 按下时的完整源几何投影（paramBlockGeometryFreeze 用），不含 waveform。 */
    clips: ClipSourceGeometry[];
}

/** 目录导入进度（跨窗口显示在主窗口状态栏）。 */
export interface FolderImportProgress {
    active: boolean;
    done: number;
    total: number;
}

export interface InteractionState {
    clipGeometryPreview: ClipGeometryPreviewState | null;
    /**
     * 拉伸提交期闸门。
     *
     * 模块级 loudnessFetchGate 仍保留同步读取能力（Promise 回调 / flushPending
     * 等非 React 时机需要它），但它是本字段的**派生缓存**：由 app/store 的
     * 订阅器同步，hold/release 只负责派发本字段。
     */
    loudnessFetchHeld: boolean;
    /** MenuBar 读取的参数编辑器选区。 */
    pianoRollSelection: PianoRollSelectionInfo | null;
    /** 状态栏读取的「任一参数编辑器在取数」。按窗体 id 记账。 */
    pianoRollLoadingById: Record<string, boolean>;
    /** 最近实际获得焦点的编辑窗口；跨窗口 UI 命令按它路由。 */
    activeEditorWindow: string | null;
    /** 独立窗口文件浏览器发起的目录导入进度。 */
    folderImportProgress: FolderImportProgress;
}

const initialState: InteractionState = {
    clipGeometryPreview: null,
    loudnessFetchHeld: false,
    pianoRollSelection: null,
    pianoRollLoadingById: {},
    activeEditorWindow: null,
    folderImportProgress: { active: false, done: 0, total: 0 },
};

const interactionSlice = createSlice({
    name: "interaction",
    initialState,
    reducers: {
        beginClipGeometryPreview(state, action: PayloadAction<ClipGeometryPreviewState>) {
            state.clipGeometryPreview = action.payload;
        },
        endClipGeometryPreview(state, action: PayloadAction<string | undefined>) {
            const preview = state.clipGeometryPreview;
            if (preview === null) return;
            if (action.payload !== undefined && preview.token !== action.payload) return;
            state.clipGeometryPreview = null;
        },
        setLoudnessFetchHeld(state, action: PayloadAction<boolean>) {
            state.loudnessFetchHeld = action.payload;
        },
        setPianoRollSelection(state, action: PayloadAction<PianoRollSelectionInfo | null>) {
            state.pianoRollSelection = action.payload;
        },
        setPianoRollLoading(
            state,
            action: PayloadAction<{ publisherId: string; loading: boolean }>,
        ) {
            const { publisherId, loading } = action.payload;
            if (loading) state.pianoRollLoadingById[publisherId] = true;
            else delete state.pianoRollLoadingById[publisherId];
        },
        clearPianoRollLoading(state, action: PayloadAction<string>) {
            delete state.pianoRollLoadingById[action.payload];
        },
        setActiveEditorWindow(state, action: PayloadAction<string | null>) {
            state.activeEditorWindow = action.payload;
        },
        setFolderImportProgress(state, action: PayloadAction<FolderImportProgress>) {
            state.folderImportProgress = action.payload;
        },
    },
});

export const {
    beginClipGeometryPreview,
    endClipGeometryPreview,
    setLoudnessFetchHeld,
    setPianoRollSelection,
    setPianoRollLoading,
    clearPianoRollLoading,
    setActiveEditorWindow,
    setFolderImportProgress,
} = interactionSlice.actions;

interface InteractionRoot {
    interaction: InteractionState;
}

export const selectClipGeometryPreview = (state: InteractionRoot) =>
    state.interaction.clipGeometryPreview;
export const selectClipGeometryPreviewOrigin = (state: InteractionRoot) =>
    state.interaction.clipGeometryPreview?.origin ?? null;
export const selectClipGeometryPreviewClips = (state: InteractionRoot) =>
    state.interaction.clipGeometryPreview?.clips ?? null;
export const selectLoudnessFetchHeld = (state: InteractionRoot) =>
    state.interaction.loudnessFetchHeld;
export const selectPianoRollSelection = (state: InteractionRoot) =>
    state.interaction.pianoRollSelection;
export const selectPianoRollLoadingById = (state: InteractionRoot) =>
    state.interaction.pianoRollLoadingById;
export const selectAnyPianoRollLoading = (state: InteractionRoot): boolean =>
    Object.values(state.interaction.pianoRollLoadingById).some(Boolean);
export const selectActiveEditorWindow = (state: InteractionRoot) =>
    state.interaction.activeEditorWindow;
export const selectFolderImportProgress = (state: InteractionRoot) =>
    state.interaction.folderImportProgress;

export default interactionSlice.reducer;
