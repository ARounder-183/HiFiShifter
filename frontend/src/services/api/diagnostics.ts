/**
 * diagnosticsApi — 诊断支持（Help 菜单）：
 * - openLogFolder: 打开日志所在文件夹
 * - pickDiagnosticsOutputPath / exportDiagnostics: 导出诊断信息 zip
 *   （系统信息 + 用户设置 + 全部日志 + 推理设备基准测试结果）
 */

import { invoke } from "../invoke";

/** 前端设置（含外观 / 自定义主题 / 快捷键 / 布局偏好）统一使用该 localStorage 前缀。 */
const FRONTEND_SETTINGS_PREFIX = "hifishifter.";

/**
 * 收集前端 `localStorage` 中的用户设置，随诊断包一并导出，便于复现问题。
 * 值尽量解析为 JSON（可读）；解析失败则保留原始字符串。localStorage 不可用
 * （隐私模式等）时返回空对象。
 */
export function collectFrontendSettings(): Record<string, unknown> {
    const settings: Record<string, unknown> = {};
    try {
        for (let i = 0; i < localStorage.length; i += 1) {
            const key = localStorage.key(i);
            if (!key || !key.startsWith(FRONTEND_SETTINGS_PREFIX)) continue;
            const raw = localStorage.getItem(key);
            if (raw == null) continue;
            try {
                settings[key] = JSON.parse(raw);
            } catch {
                settings[key] = raw;
            }
        }
    } catch {
        // localStorage 不可用：忽略，诊断包不含前端设置。
    }
    return settings;
}

export interface LogFolderResult {
    ok: boolean;
    path?: string;
    /**
     * 插件回报的日志文件完整路径。
     *
     * 【为什么与 `path` 并存】用户真正想复制给开发者的往往是 `plugin-<format>-<host>.log` 本身，
     * 而不是它所在的目录。
     */
    file?: string;
    /**
     * 资源管理器是否真的被打开了。
     *
     * 【为什么要区分】独立 App 走 Tauri opener，成功即静默；插件是**尽力而为**地
     * 调用系统 shell，失败时仍要把路径交给用户（可选中、可复制），不能假装打开成功。
     */
    opened?: boolean;
    /**
     * 文件日志当前是否在写。
     *
     * 【为什么要报】目录存在不等于里面有日志：用户可能把「日志详细程度」选成了
     * 「无」，此时后端一个字节都不写。只给路径会让他在空文件夹里找半天。
     * 插件侧的 `open_log_folder` 不返回这个字段（那里的日志始终开启）。
     */
    fileEnabled?: boolean;
    /** 日志子系统自检是否健康（目录解析 / 文件打开失败时为 false）。 */
    loggingHealthy?: boolean;
    /** 日志子系统最近一次失败原因（健康时为 `null`）。 */
    lastError?: string | null;
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

/**
 * 弹出诊断包的保存对话框。
 *
 * `dialogTitle` 由调用方按当前语言传入（后端不再硬编码英文标题）；
 * 空串 / 缺省时后端回退系统默认标题。
 */
export function pickDiagnosticsOutputPath(dialogTitle: string): Promise<DiagnosticsExportResult> {
    return invoke("pick_diagnostics_output_path", dialogTitle);
}

export function exportDiagnostics(outputPath: string): Promise<DiagnosticsExportResult> {
    // 附带前端用户设置：诊断包里除系统信息 / 日志 / 基准测试外，还包含用户设置，
    // 便于在不接触用户工程数据的前提下复现其配置相关的问题。
    return invoke("export_diagnostics", outputPath, collectFrontendSettings());
}
