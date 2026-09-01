/**
 * diagnosticsApi — 诊断支持（Help 菜单）：
 * - openLogFolder: 打开日志所在文件夹
 * - pickDiagnosticsOutputPath / exportDiagnostics: 导出诊断信息 zip
 *   （系统信息 + 全部日志 + 推理设备基准测试结果）
 */

import { invoke } from "../invoke";

export interface LogFolderResult {
    ok: boolean;
    path?: string;
    error?: string;
}

export interface DiagnosticsExportResult {
    ok: boolean;
    path?: string;
    canceled?: boolean;
    error?: string;
}

export function openLogFolder(): Promise<LogFolderResult> {
    return invoke("open_log_folder");
}

export function pickDiagnosticsOutputPath(): Promise<DiagnosticsExportResult> {
    return invoke("pick_diagnostics_output_path");
}

export function exportDiagnostics(outputPath: string): Promise<DiagnosticsExportResult> {
    return invoke("export_diagnostics", outputPath);
}
