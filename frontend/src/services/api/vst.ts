// VST 插件宿主 API 模块
//
// 提供 VST 插件扫描、FX 链管理、编辑器 GUI 等操作的前端调用接口。
// 通过 invoke 与 Tauri 后端 VST 命令通信。

import { invoke } from "../invoke";

// ─── 类型定义 ─────────────────────────────────────────────────────────────

/** VST 插件描述信息（来自后端扫描结果） */
export interface VstPluginInfo {
    uid: string;
    name: string;
    vendor: string;
    format: "vst2" | "vst3";
    path: string;
    category: string;
    isInstrument: boolean;
    numInputs: number;
    numOutputs: number;
}

/** FX 链中单个插件槽位的视图 */
export interface VstChainSlot {
    index: number;
    pluginUid: string;
    pluginName: string;
    pluginPath: string;
    format: "vst2" | "vst3";
    bypassed: boolean;
}

/** VST 扫描结果 */
export interface VstScanResult {
    ok: boolean;
    plugins: VstPluginInfo[];
    error?: string;
}

/** VST FX 链查询结果 */
export interface VstChainResult {
    ok: boolean;
    trackId?: string;
    slots: VstChainSlot[];
    error?: string;
}

/** VST 功能状态 */
export interface VstStatusResult {
    ok: boolean;
    available: boolean;
    formats: string[];
}

// ─── API 函数 ──────────────────────────────────────────────────────────────

export const vstApi = {
    /** 触发 VST 插件扫描并返回扫描结果 */
    scanPlugins: () =>
        invoke<VstScanResult>("vst_scan_plugins"),

    /** 获取已扫描的插件列表（不触发新扫描） */
    listPlugins: () =>
        invoke<VstScanResult>("vst_list_plugins"),

    /** 获取指定轨道的 VST FX 链 */
    getTrackChain: (trackId: string) =>
        invoke<VstChainResult>("vst_get_track_chain", trackId),

    /** 向轨道 FX 链添加插件 */
    addToChain: (trackId: string, pluginUid: string, index?: number) =>
        invoke<{ ok: boolean; error?: string }>(
            "vst_add_to_chain",
            trackId,
            pluginUid,
            index,
        ),

    /** 从轨道 FX 链移除插件 */
    removeFromChain: (trackId: string, index: number) =>
        invoke<{ ok: boolean; error?: string }>(
            "vst_remove_from_chain",
            trackId,
            index,
        ),

    /** 设置 FX 链中某个插件的 bypass 状态 */
    setBypass: (trackId: string, index: number, bypassed: boolean) =>
        invoke<{ ok: boolean; error?: string }>(
            "vst_set_bypass",
            trackId,
            index,
            bypassed,
        ),

    /** 在 FX 链内重新排序插件 */
    reorderChain: (trackId: string, fromIndex: number, toIndex: number) =>
        invoke<{ ok: boolean; error?: string }>(
            "vst_reorder_chain",
            trackId,
            fromIndex,
            toIndex,
        ),

    /** 打开 VST 插件编辑器窗口 */
    openEditor: (trackId: string, index: number) =>
        invoke<{ ok: boolean; message?: string; error?: string }>(
            "vst_open_editor",
            trackId,
            index,
        ),

    /** 关闭 VST 插件编辑器窗口 */
    closeEditor: (trackId: string, index: number) =>
        invoke<{ ok: boolean; error?: string }>(
            "vst_close_editor",
            trackId,
            index,
        ),

    /** 添加自定义 VST 扫描路径 */
    addScanPath: (path: string) =>
        invoke<{ ok: boolean; error?: string }>("vst_add_scan_path", path),

    /** 获取当前自定义 VST 扫描路径列表 */
    listScanPaths: () =>
        invoke<{ ok: boolean; paths: string[]; error?: string }>("vst_list_scan_paths"),

    /** 移除自定义 VST 扫描路径 */
    removeScanPath: (path: string) =>
        invoke<{ ok: boolean; error?: string }>("vst_remove_scan_path", path),

    /** 获取 VST 功能是否可用 */
    getStatus: () => invoke<VstStatusResult>("vst_get_status"),
};
