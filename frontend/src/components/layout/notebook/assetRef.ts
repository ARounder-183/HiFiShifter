/*
 * 记事本附件引用的编解码。
 *
 * Markdown 正文里用 `hifi-asset://<id>[.<ext>][#w=<px>]` 引用附件（图片），
 * 用 ```hifi-clip 围栏引用剪贴板暂存块。两种引用都只带 id，字节在工程旁挂
 * 目录里 —— 见后端 `notebook_assets` 模块的说明。
 *
 * 本文件只做纯字符串处理（无 DOM、无 TipTap），因此可以在 vitest 的 node
 * 环境里直接测；后端 `notebook_assets::scan_asset_refs` 是同一套语法的
 * Rust 实现，两边必须保持一致（导出、清理都依赖它）。
 */

/** 附件引用的 scheme。刻意不用 `file:`：markdown-it 会拒绝 `file:` 链接。 */
export const ASSET_SCHEME = "hifi-asset://";

export interface AssetRef {
    /** 在字符串中的起止下标（用于拼接替换）。 */
    start: number;
    end: number;
    id: string;
    ext?: string;
    /** `#w=` 片段携带的显示宽度（像素）。 */
    width?: number;
}

const ID_CHAR = /[A-Za-z0-9_-]/;
const EXT_CHAR = /[A-Za-z0-9]/;

/** 扫描字符串中全部附件引用。 */
export function scanAssetRefs(content: string): AssetRef[] {
    const refs: AssetRef[] = [];
    let cursor = 0;

    while (true) {
        const rel = content.indexOf(ASSET_SCHEME, cursor);
        if (rel < 0) break;
        const start = rel;
        let pos = start + ASSET_SCHEME.length;

        const idStart = pos;
        while (pos < content.length && ID_CHAR.test(content[pos])) pos += 1;
        if (pos === idStart) {
            cursor = pos;
            continue;
        }
        const id = content.slice(idStart, pos);

        let ext: string | undefined;
        if (content[pos] === ".") {
            const extStart = pos + 1;
            let extEnd = extStart;
            while (extEnd < content.length && EXT_CHAR.test(content[extEnd])) extEnd += 1;
            if (extEnd > extStart) {
                ext = content.slice(extStart, extEnd).toLowerCase();
                pos = extEnd;
            }
        }

        let width: number | undefined;
        if (content.startsWith("#w=", pos)) {
            const digitsStart = pos + 3;
            let digitsEnd = digitsStart;
            while (digitsEnd < content.length && /[0-9]/.test(content[digitsEnd])) digitsEnd += 1;
            if (digitsEnd > digitsStart) {
                const parsed = Number.parseInt(content.slice(digitsStart, digitsEnd), 10);
                if (Number.isFinite(parsed) && parsed > 0) width = parsed;
                pos = digitsEnd;
            }
        }

        refs.push({ start, end: pos, id, ext, width });
        cursor = pos;
    }

    return refs;
}

/** 构造一条附件引用。`width` 为 0 / undefined 时不写 `#w=`。 */
export function formatAssetRef(id: string, ext?: string, width?: number): string {
    const suffix = ext ? `.${ext}` : "";
    const widthPart = width && width > 0 ? `#w=${Math.round(width)}` : "";
    return `${ASSET_SCHEME}${id}${suffix}${widthPart}`;
}

/** 引用是否指向本地附件（而不是 http(s) 外链、data URI 或相对路径）。 */
export function isAssetRef(src: string): boolean {
    return src.startsWith(ASSET_SCHEME);
}

/**
 * 从任意图片 src 里取出附件 id；非附件引用返回 null。
 *
 * 图片节点渲染、附件管理器、导出都走这里，避免各自写一遍前缀判断。
 */
export function assetIdFromSrc(src: string): string | null {
    if (!isAssetRef(src)) return null;
    const refs = scanAssetRefs(src);
    return refs.length > 0 && refs[0].start === 0 ? refs[0].id : null;
}

/** 从 src 里取出显示宽度（`#w=`），没有则 null。 */
export function assetWidthFromSrc(src: string): number | null {
    if (!isAssetRef(src)) return null;
    const refs = scanAssetRefs(src);
    return refs.length > 0 && refs[0].start === 0 ? (refs[0].width ?? null) : null;
}

/**
 * 去掉 src 里的 `#w=` 片段。
 *
 * 文档内的 `src` 属性**不含**宽度片段：宽度是节点自己的 `width` 属性，只在
 * 序列化成 Markdown 时才拼回 `src`。这样"宽度"在文档里只有一个真源，不会
 * 出现 `src` 与 `width` 属性各说各话。
 */
export function stripWidthFragment(src: string): string {
    if (!isAssetRef(src)) return src;
    const refs = scanAssetRefs(src);
    if (refs.length === 0 || refs[0].start !== 0) return src;
    const ref = refs[0];
    if (ref.width === undefined) return src;
    // 只截掉 `#w=` 那一段，保留其后可能存在的其它内容（当前没有，但不要假设）。
    const widthStart = ref.end - `#w=${ref.width}`.length;
    return src.slice(0, widthStart) + src.slice(ref.end);
}

/** 给 src 拼上 `#w=` 片段（`width` 为空或 0 时原样返回）。 */
export function withWidthFragment(src: string, width: number | null | undefined): string {
    if (!width || width <= 0) return stripWidthFragment(src);
    return `${stripWidthFragment(src)}#w=${Math.round(width)}`;
}

/** 取出所有被引用的附件 id（去重）。 */
export function referencedAssetIds(markdown: string): Set<string> {
    const ids = new Set<string>();
    for (const ref of scanAssetRefs(markdown)) ids.add(ref.id);
    for (const id of clipBlockIdsInMarkdown(markdown)) ids.add(id);
    return ids;
}

/** 扫描 ```hifi-clip 围栏里声明的 `id:`（与后端 referenced_asset_ids 对齐）。 */
export function clipBlockIdsInMarkdown(markdown: string): string[] {
    const ids: string[] = [];
    for (const rawLine of markdown.split("\n")) {
        const line = rawLine.trim();
        if (!line.startsWith("id:")) continue;
        const value = line.slice(3).trim();
        if (value && /^[A-Za-z0-9_-]+$/.test(value)) ids.push(value);
    }
    return ids;
}
