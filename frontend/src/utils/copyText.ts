/**
 * 把纯文本写进系统剪贴板。
 *
 * 【为什么优先原生通道】`navigator.clipboard.writeText` 需要 transient user
 * activation 与剪贴板权限。「复制路径 / 复制链接」排在动态 import + 若干 IPC
 * 之后，手势早已过期；Tauri 自定义协议与 WebView2 也不保证给权限。原生通道
 * 不需要手势，避免复制静默失败。
 *
 * `write_clipboard_text` 走 `hifishifter-clipboard` 的原生实现：无需手势，
 * Windows 下共用 `CLIPBOARD_LOCK` + 真实睡眠退避的 `OpenClipboard` 重试，
 * 外部程序短暂独占时最多约 800ms 后自动成功；macOS/Linux 用同一会话写纯文本。
 *
 * 【为什么保留两级浏览器兜底】纯浏览器预览（没有 App / 插件宿主）以及尚未注入
 * `__TAURI__` 的启动竞态下仍需可用；`execCommand("copy")` 是安全上下文里
 * 无需权限的最后一条路。结构化对象剪贴板仍走 `utils/systemClipboard.ts`，
 * 互不干扰：两者共用后端的同一把进程锁。
 *
 * @returns 是否写入成功。失败不抛 —— 调用方通常只想静默忽略。
 */
import { buildTauriArgs } from "../services/invoke";
import { getPluginHost } from "../services/pluginHost";
import { frontendLog } from "../services/frontendLog";

/** 与两个宿主的 `write_clipboard_text` 返回结构同形。 */
interface ClipboardTextResult {
    ok: boolean;
    error?: string;
}

/**
 * 原生通道：插件走 WebView2 消息口，独立 App 走 Tauri core.invoke。
 *
 * 【为什么不经通用 invoke()】通用层在两者都缺席时会等 pywebview 最多 1.5 秒，
 * 而「复制」是用户手势驱动的即时动作；纯浏览器预览必须立刻走 WebView API。
 * 这里只认两个真实宿主（与 frontendLog 的零日志 IPC 同一形态），参数映射仍
 * 复用 `buildTauriArgs`，不手抄第二份协议。
 */
async function nativeWrite(text: string): Promise<ClipboardTextResult | null> {
    const mapped = buildTauriArgs("write_clipboard_text", [text]);
    const args = mapped && "__unwired" in mapped ? undefined : (mapped ?? undefined);
    try {
        const plugin = getPluginHost();
        if (plugin) {
            return await plugin.invoke<ClipboardTextResult>("write_clipboard_text", args);
        }
        const tauri = window as unknown as {
            __TAURI__?: {
                core?: {
                    invoke?: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
                };
                invoke?: <T>(cmd: string, args?: Record<string, unknown>) => Promise<T>;
            };
        };
        const invokeFn = tauri.__TAURI__?.core?.invoke ?? tauri.__TAURI__?.invoke;
        if (typeof invokeFn === "function") {
            return await invokeFn<ClipboardTextResult>("write_clipboard_text", args);
        }
    } catch (error) {
        // 原生通道抛错时仍可尝试浏览器 API；诊断留给统一日志通道。
        frontendLog("warn", "Native clipboard text write failed", error);
    }
    return null;
}

/** 浏览器预览 / 原生通道缺席时的最后兜底。 */
async function browserWrite(text: string): Promise<boolean> {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch {
        try {
            const textarea = document.createElement("textarea");
            textarea.value = text;
            textarea.style.position = "fixed";
            textarea.style.opacity = "0";
            document.body.appendChild(textarea);
            textarea.select();
            const ok = document.execCommand("copy");
            textarea.remove();
            return ok;
        } catch {
            return false;
        }
    }
}

export async function copyTextToClipboard(text: string): Promise<boolean> {
    if (!text) return false;
    const native = await nativeWrite(text);
    if (native?.ok) return true;
    // 原生通道明确拒答时浏览器 API 多半同样拿不到权限，仍试一次，把
    // "确实拿不到"与"从未尝试"区分开；纯浏览器预览里 native 为 null。
    return browserWrite(text);
}
