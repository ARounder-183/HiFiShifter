/**
 * dockLayoutApi — 停靠布局的导入导出（视图 → 布局）。
 *
 * 【导出为什么走后端命令】Tauri 的 WebView（wry）默认拦截页面发起的下载，
 * `<a download>` + Blob 的浏览器方案在壳内静默失败 —— 用户点「导出布局」
 * 什么都不会发生。导出走原生保存对话框 + 后端写文件（与「导出诊断信息」
 * 同一模式，见 `backend/src-tauri/src/commands/layout_export.rs`）。导入仍走
 * 浏览器原生 `<input type=file>`：文件选择在 WebView 内可用，不必动用 IPC。
 */

import { invoke } from "../invoke";

export interface LayoutExportResult {
    ok: boolean;
    path?: string;
    canceled?: boolean;
    error?: string;
}

export function exportLayoutJson(json: string): Promise<LayoutExportResult> {
    return invoke("export_layout_json", json);
}
