/*
 * HiFiShifter 剪贴板暂存块的 Markdown 表示。
 *
 * 正文里长这样：
 *
 * ```hifi-clip
 * id: 7c1e9a4b2d
 * kind: clips
 * title: 副歌 A
 * source: 我的歌
 * clips: 3
 * tracks: 1
 * duration: 4.820
 * captured: 2026-09-23T10:04:11Z
 * ```
 *
 * 【为什么用围栏而不是自定义语法】围栏是标准 Markdown，任何别的编辑器打开
 * 这篇笔记都会把它显示成一个代码块 —— 内容不会丢、不会把文档搞乱。载荷
 * 字节本身存在工程附件表里（正文只放引用），所以这里非常轻。
 *
 * 本文件是纯字符串编解码（无 DOM、无 TipTap），可被 vitest 直接测试；
 * 节点扩展、markdown-it 渲染规则都复用它。
 */

/** 围栏语言标记。 */
export const HIFI_CLIP_FENCE_LANG = "hifi-clip";

/** 载荷种类：时间轴片段 / 轨道 / 整个工程 / 参数线。 */
export type HifiClipKind = "clips" | "tracks" | "project" | "param";

export interface HifiClipBlockAttrs {
    /** 附件 id（载荷字节在工程附件表里的键）。 */
    id: string;
    kind: HifiClipKind;
    /** 用户可改的标题。 */
    title: string;
    /** 来源工程名。 */
    source: string;
    clipCount: number;
    trackCount: number;
    /** 秒。参数线载荷没有时长，为 0。 */
    durationSec: number;
    /** 暂存时刻（ISO 8601）。 */
    captured: string;
    /** 载荷编码：msgpack 片段 / JSON 参数线。 */
    encoding: "fragment" | "param";
    /** 参数线载荷的参数名（仅 `param`）。 */
    param?: string;
    /** 参数线载荷的总帧数（仅 `param`）。 */
    frameCount?: number;
}

const KIND_VALUES: readonly HifiClipKind[] = ["clips", "tracks", "project", "param"];

/** 解析围栏正文；`id` 缺失或非法时返回 null（视为普通代码块）。 */
export function parseHifiClipFenceBody(body: string): HifiClipBlockAttrs | null {
    const fields = new Map<string, string>();
    for (const rawLine of body.split("\n")) {
        const line = rawLine.trim();
        if (!line) continue;
        const colon = line.indexOf(":");
        if (colon <= 0) continue;
        const key = line.slice(0, colon).trim().toLowerCase();
        const value = line.slice(colon + 1).trim();
        if (!fields.has(key)) fields.set(key, value);
    }

    const id = fields.get("id") ?? "";
    if (!/^[A-Za-z0-9_-]+$/.test(id)) return null;

    const rawKind = (fields.get("kind") ?? "clips").toLowerCase();
    const kind = (KIND_VALUES as readonly string[]).includes(rawKind)
        ? (rawKind as HifiClipKind)
        : "clips";

    const encoding = fields.get("encoding")?.toLowerCase() === "param" || kind === "param"
        ? "param"
        : "fragment";

    return {
        id,
        kind,
        title: fields.get("title") ?? "",
        source: fields.get("source") ?? "",
        clipCount: toCount(fields.get("clips")),
        trackCount: toCount(fields.get("tracks")),
        durationSec: toNumber(fields.get("duration")),
        captured: fields.get("captured") ?? "",
        encoding,
        param: fields.get("param") || undefined,
        frameCount: fields.has("frames") ? toCount(fields.get("frames")) : undefined,
    };
}

/** 序列化围栏正文（键序固定，便于工程文件做文本比对）。 */
export function serializeHifiClipFenceBody(attrs: HifiClipBlockAttrs): string {
    const lines = [`id: ${attrs.id}`, `kind: ${attrs.kind}`];
    if (attrs.title) lines.push(`title: ${sanitizeValue(attrs.title)}`);
    if (attrs.source) lines.push(`source: ${sanitizeValue(attrs.source)}`);
    lines.push(`encoding: ${attrs.encoding}`);
    if (attrs.kind === "param") {
        if (attrs.param) lines.push(`param: ${sanitizeValue(attrs.param)}`);
        if (typeof attrs.frameCount === "number") lines.push(`frames: ${attrs.frameCount}`);
    } else {
        lines.push(`clips: ${attrs.clipCount}`);
        lines.push(`tracks: ${attrs.trackCount}`);
        lines.push(`duration: ${attrs.durationSec.toFixed(3)}`);
    }
    if (attrs.captured) lines.push(`captured: ${sanitizeValue(attrs.captured)}`);
    return lines.join("\n");
}

/** 完整围栏文本（含首尾 ```）。 */
export function serializeHifiClipFence(attrs: HifiClipBlockAttrs): string {
    return ["```" + HIFI_CLIP_FENCE_LANG, serializeHifiClipFenceBody(attrs), "```"].join("\n");
}

/**
 * 值里的换行会破坏"一行一个字段"的结构，换行符一律折叠成空格。
 * 冒号在值里是允许的（只在第一个冒号处切分）。
 */
function sanitizeValue(value: string): string {
    return value.replace(/[\r\n]+/g, " ").trim();
}

function toCount(value: string | undefined): number {
    const parsed = Number.parseInt(value ?? "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function toNumber(value: string | undefined): number {
    const parsed = Number.parseFloat(value ?? "");
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

/** 人类可读的时长（`4.82 s` / `1:23.456`）。 */
export function formatClipDuration(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds <= 0) return "0:00.000";
    const minutes = Math.floor(seconds / 60);
    const rest = seconds - minutes * 60;
    return `${minutes}:${rest.toFixed(3).padStart(6, "0")}`;
}

/** 暂存块的默认标题。 */
export function defaultClipTitle(attrs: HifiClipBlockAttrs): string {
    if (attrs.kind === "param") {
        const name = attrs.param || "参数线";
        return attrs.frameCount ? `${name} · ${attrs.frameCount} 帧` : name;
    }
    const parts = [`${attrs.clipCount} clips`];
    if (attrs.trackCount > 1) parts.push(`${attrs.trackCount} tracks`);
    parts.push(formatClipDuration(attrs.durationSec));
    return parts.join(" · ");
}
