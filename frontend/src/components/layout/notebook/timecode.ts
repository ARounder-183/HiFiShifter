/*
 * 记事本里的"内部链接"：把笔记和时间轴连起来。
 *
 * - `hifi://seek/<秒>`  —— 点击把播放头移到该时刻；
 * - `hifi://clip/<id>`  —— 点击选中并定位到该 Clip。
 *
 * 两者都是**标准 Markdown 链接**（`[显示文本](hifi://...)`），因此不需要任何
 * 自定义语法：源码视图可读可编辑，别的编辑器打开也只是普通链接。只有渲染层
 * 需要拦截点击。
 *
 * 纯函数，无 DOM 依赖。
 */

export const SEEK_SCHEME = "hifi://seek/";
export const CLIP_SCHEME = "hifi://clip/";

export type NotebookInternalLink =
    | { type: "seek"; seconds: number }
    | { type: "clip"; clipId: string };

/** 解析内部链接；不是内部链接返回 null。 */
export function parseInternalLink(href: string): NotebookInternalLink | null {
    if (href.startsWith(SEEK_SCHEME)) {
        const parsed = Number.parseFloat(href.slice(SEEK_SCHEME.length));
        if (!Number.isFinite(parsed) || parsed < 0) return null;
        return { type: "seek", seconds: parsed };
    }
    if (href.startsWith(CLIP_SCHEME)) {
        const clipId = href.slice(CLIP_SCHEME.length).trim();
        if (!clipId) return null;
        return { type: "clip", clipId };
    }
    return null;
}

/**
 * 时间码文本。
 *
 * `showHours` 为真时输出 `h:mm:ss.mmm`，否则 `m:ss.mmm` —— 歌曲工程几乎
 * 不会超过一小时，多出来的 `0:` 前缀只是噪声。
 */
export function formatTimecode(seconds: number, showHours = false): string {
    const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
    const hours = Math.floor(safe / 3600);
    const minutes = Math.floor((safe - hours * 3600) / 60);
    const rest = safe - hours * 3600 - minutes * 60;
    const secondsText = rest.toFixed(3).padStart(6, "0");
    if (showHours || hours > 0) {
        return `${hours}:${String(minutes).padStart(2, "0")}:${secondsText}`;
    }
    return `${minutes}:${secondsText}`;
}

/** 构造播放头跳转链接（Markdown 源码形态）。 */
export function buildSeekLink(seconds: number, label?: string): string {
    const safe = Math.max(0, seconds);
    const text = label && label.trim() ? label.trim() : formatTimecode(safe);
    return `[${text}](${SEEK_SCHEME}${safe.toFixed(3)})`;
}

/** 构造 Clip 引用链接。 */
export function buildClipLink(clipId: string, label: string): string {
    // 方括号会截断链接文本（`[a[b]c](url)` 解析不出），换成圆括号保形。
    const text = label.trim().replace(/\[/g, "(").replace(/\]/g, ")") || clipId;
    return `[${text}](${CLIP_SCHEME}${clipId})`;
}
