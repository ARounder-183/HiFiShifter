// VST 插件宿主相关的 Redux async thunks
//
// 提供 VST 扫描、FX 链管理、编辑器打开等异步操作，
// 通过 webApi / vstApi 与 Tauri 后端 VST 命令通信。

import { createAsyncThunk } from "@reduxjs/toolkit";
import { webApi } from "../../../services/webviewApi";

/** 触发 VST 插件全量扫描 */
export const vstScanPluginsRemote = createAsyncThunk(
    "session/vstScanPluginsRemote",
    async () => {
        return webApi.vstScanPlugins();
    },
);

/** 获取已扫描的插件列表（不触发新扫描） */
export const vstListPluginsRemote = createAsyncThunk(
    "session/vstListPluginsRemote",
    async () => {
        return webApi.vstListPlugins();
    },
);

/** 获取指定轨道的 FX 链 */
export const vstGetTrackChainRemote = createAsyncThunk(
    "session/vstGetTrackChainRemote",
    async (trackId: string) => {
        return webApi.vstGetTrackChain(trackId);
    },
);

/** 向轨道 FX 链添加插件 */
export const vstAddToChainRemote = createAsyncThunk(
    "session/vstAddToChainRemote",
    async (payload: { trackId: string; pluginUid: string; index?: number }) => {
        const result = await webApi.vstAddToChain(
            payload.trackId,
            payload.pluginUid,
            payload.index,
        );
        return { ...result, trackId: payload.trackId };
    },
);

/** 从轨道 FX 链移除插件 */
export const vstRemoveFromChainRemote = createAsyncThunk(
    "session/vstRemoveFromChainRemote",
    async (payload: { trackId: string; index: number }) => {
        const result = await webApi.vstRemoveFromChain(
            payload.trackId,
            payload.index,
        );
        return { ...result, trackId: payload.trackId };
    },
);

/** 设置 FX 链中某个插件的 bypass 状态 */
export const vstSetBypassRemote = createAsyncThunk(
    "session/vstSetBypassRemote",
    async (payload: { trackId: string; index: number; bypassed: boolean }) => {
        const result = await webApi.vstSetBypass(
            payload.trackId,
            payload.index,
            payload.bypassed,
        );
        return { ...result, trackId: payload.trackId };
    },
);

/** FX 链内重排序 */
export const vstReorderChainRemote = createAsyncThunk(
    "session/vstReorderChainRemote",
    async (payload: {
        trackId: string;
        fromIndex: number;
        toIndex: number;
    }) => {
        const result = await webApi.vstReorderChain(
            payload.trackId,
            payload.fromIndex,
            payload.toIndex,
        );
        return { ...result, trackId: payload.trackId };
    },
);

/** 打开 VST 插件编辑器 GUI 窗口 */
export const vstOpenEditorRemote = createAsyncThunk(
    "session/vstOpenEditorRemote",
    async (payload: { trackId: string; index: number }) => {
        return webApi.vstOpenEditor(payload.trackId, payload.index);
    },
);

/** 关闭 VST 插件编辑器 GUI 窗口 */
export const vstCloseEditorRemote = createAsyncThunk(
    "session/vstCloseEditorRemote",
    async (payload: { trackId: string; index: number }) => {
        return webApi.vstCloseEditor(payload.trackId, payload.index);
    },
);

/** 添加自定义 VST 扫描路径 */
export const vstAddScanPathRemote = createAsyncThunk(
    "session/vstAddScanPathRemote",
    async (path: string) => {
        return webApi.vstAddScanPath(path);
    },
);

/** 获取当前自定义 VST 扫描路径列表 */
export const vstListScanPathsRemote = createAsyncThunk(
    "session/vstListScanPathsRemote",
    async () => {
        return webApi.vstListScanPaths();
    },
);

/** 移除自定义 VST 扫描路径 */
export const vstRemoveScanPathRemote = createAsyncThunk(
    "session/vstRemoveScanPathRemote",
    async (path: string) => {
        return webApi.vstRemoveScanPath(path);
    },
);

/** 获取 VST 功能状态 */
export const vstGetStatusRemote = createAsyncThunk(
    "session/vstGetStatusRemote",
    async () => {
        return webApi.vstGetStatus();
    },
);
