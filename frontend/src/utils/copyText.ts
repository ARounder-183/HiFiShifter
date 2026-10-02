/**
 * 把纯文本写进系统剪贴板。
 *
 * 【为什么要两级兜底】`navigator.clipboard` 只在安全上下文（https / localhost）与
 * 用户手势中可用；Tauri 的 WebView 在部分平台/自定义协议下拿不到它。这时退回到
 * `execCommand("copy")` 配一个屏幕外 textarea —— 这条路径在本仓已有先例
 * （`TimelinePanel` 的"复制播放头时间"、`KernelUnavailableNotice`），只是此前是
 * 内联复制的。
 *
 * 【为什么不用后端的 `write_system_clipboard_object`】那条通道写的是 HiFiShifter
 * 私有格式并附带"N 个片段已复制"的摘要，用它复制文件路径会顶掉用户的剪贴板内容
 * 并给出错误摘要。这里要的是最朴素的文本剪贴板。
 *
 * @returns 是否写入成功。失败不抛 —— 调用方通常只想静默忽略。
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
