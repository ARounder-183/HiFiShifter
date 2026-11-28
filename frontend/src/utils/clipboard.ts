/**
 * 写入系统剪贴板（带 `execCommand` 兜底）。
 *
 * 【为什么需要兜底】WebView2 里 `navigator.clipboard.writeText` 在两种常见情形下会
 * 直接 reject：文档尚未获得焦点（刚切窗口回来）、以及非安全上下文。此时用一次
 * 隐藏 textarea + `execCommand("copy")` 仍能成功 —— 用户要的只是"复制成功"，
 * 不该因为底层通道换了就静默失败。
 */
export async function copyTextToClipboard(text: string): Promise<boolean> {
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
