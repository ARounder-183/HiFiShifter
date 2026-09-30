/**
 * jsonExportApi — 「把一段 JSON 存成文件」的唯一前端入口。
 *
 * 【为什么必须走后端命令】Tauri 的 WebView（wry）默认拦截页面发起的下载，
 * Blob + `<a download>` 的浏览器方案在壳内**静默失败** —— 用户点「导出」什么都
 * 不会发生。布局导出早已因此改走后端（原生保存对话框 + 写文件），而外观主题导出
 * 当时漏了这一步，于是它单独坏掉了：点「导出」没有任何反应，也不会有任何报错。
 * 现在两者共用后端 `commands/json_export.rs` 的同一份实现。
 *
 * 【为什么后端是两条命令而不是一条通用的】命令名是 invoke 的稳定契约，也让
 * `invoke.wiring.test.ts` 能穷举核对；真正会重复的部分（选路径 / 写字节 / 定位 /
 * 报错）只有一份实现，随业务变化的只有标题与默认文件名。
 */

import { invoke } from "../invoke";

export interface JsonExportResult {
    ok: boolean;
    /** 落盘路径（成功时存在）。 */
    path?: string;
    /** 用户在原生对话框里取消了 —— 不是错误，调用方不应报错。 */
    canceled?: boolean;
    error?: string;
}

/** 导出布局（视图 → 布局 → 导出布局...）。 */
export function exportLayoutJson(json: string): Promise<JsonExportResult> {
    return invoke("export_layout_json", json);
}

/**
 * 导出外观主题（视图 → 外观设置 → 导出）。
 *
 * @param defaultFileName 默认文件名（含扩展名）。主题名是用户起的、可能含非 ASCII，
 *   因此由前端给而不是后端写死。
 */
export function exportThemeJson(json: string, defaultFileName: string): Promise<JsonExportResult> {
    return invoke("export_theme_json", json, defaultFileName);
}

/**
 * 导出颤音预设（颤音预设管理器 → 「导出...」）。
 *
 * 与主题导出同一模式：默认文件名由前端给（预设名是用户起的，含非 ASCII，
 * 且需要先剔除文件系统的保留字符）。
 */
export function exportVibratoPresetsJson(
    json: string,
    defaultFileName: string,
): Promise<JsonExportResult> {
    return invoke("export_vibrato_presets_json", json, defaultFileName);
}
