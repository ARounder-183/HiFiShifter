/**
 * 剪贴板错误码 → 状态栏提示文案（i18n key）映射。
 *
 * 应用采用单剪贴板纪律：时间轴 Clip 与参数线载荷共享同一个槽位、互相
 * 覆盖，最后复制的获胜；时间轴粘贴在槽位内容无法按时间轴载荷解析时
 * （被参数线复制覆盖 / 损坏 / 未知版本）统一报 `timeline_clipboard_empty`。
 *
 * 后端错误可能带回退链后缀（如
 * "timeline_clipboard_empty; Reaper clipboard: clipboard_empty"），因此按
 * **首个错误码 token** 精确匹配，而非整串或前缀模糊匹配 —— 避免把
 * "clipboard_empty_failed" 之类写入期错误误映射成"剪贴板为空"。
 * 未收录的错误码返回空串，调用方回退为显示原始错误文本（保留诊断信息）。
 */
const CLIPBOARD_ERROR_CODE_KEYS: Record<string, string> = {
    timeline_clipboard_empty: "clipboard_error_empty",
    clipboard_empty: "clipboard_error_empty",
};

export function clipboardErrorKey(error: string): string {
    const code = error.split(/[;:\s]/, 1)[0] ?? "";
    return CLIPBOARD_ERROR_CODE_KEYS[code] ?? "";
}
