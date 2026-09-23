/*
 * 路径工具（纯函数，可单测）。
 *
 * 记事本的 `link` 图片模式与"导出到旁挂目录"都要处理路径：
 * - `link`：正文里存**相对工程目录**的路径，工程和素材一起搬走仍然可用；
 * - 导出：把 `hifi-asset://` 换成 `./<名>.assets/<id>.<ext>`。
 *
 * Windows 的盘符、反斜杠、大小写不敏感都要照顾到 —— 这里统一用正斜杠做
 * 规范形态，只在拼绝对路径时还原。
 */

/** 把路径规范化为正斜杠形态（不做大小写转换）。 */
export function toPosix(path: string): string {
    return path.replace(/\\/g, "/");
}

/** 取目录部分（`a/b/c.png` → `a/b`）。无目录时返回空串。 */
export function dirName(path: string): string {
    const normalized = toPosix(path);
    const index = normalized.lastIndexOf("/");
    return index < 0 ? "" : normalized.slice(0, index);
}

/** 取文件名（`a/b/c.png` → `c.png`）。 */
export function baseName(path: string): string {
    const normalized = toPosix(path);
    const index = normalized.lastIndexOf("/");
    return index < 0 ? normalized : normalized.slice(index + 1);
}

/** 取主名（去掉最后一个扩展名）。 */
export function stemName(path: string): string {
    const name = baseName(path);
    const dot = name.lastIndexOf(".");
    return dot <= 0 ? name : name.slice(0, dot);
}

function splitSegments(path: string): string[] {
    return toPosix(path)
        .split("/")
        .filter((segment) => segment.length > 0 && segment !== ".");
}

/** 是否是绝对路径（POSIX 根 / Windows 盘符 / UNC）。 */
export function isAbsolutePath(path: string): boolean {
    const normalized = toPosix(path);
    return (
        normalized.startsWith("/") ||
        /^[A-Za-z]:\//.test(normalized) ||
        normalized.startsWith("//")
    );
}

/**
 * 计算 `target` 相对 `fromDir` 的相对路径。
 *
 * 不同盘符（Windows 上 `C:` → `D:`）无法表达相对路径，此时返回 null，调用方
 * 应退回"复制到旁挂目录"。
 */
export function relativePath(fromDir: string, target: string): string | null {
    const from = splitSegments(fromDir);
    const to = splitSegments(target);

    // 盘符（Windows）：不同盘直接放弃。
    const fromDrive = /^[A-Za-z]:$/.exec(from[0] ?? "");
    const toDrive = /^[A-Za-z]:$/.exec(to[0] ?? "");
    if (fromDrive || toDrive) {
        if (!fromDrive || !toDrive) return null;
        if (fromDrive[0].toLowerCase() !== toDrive[0].toLowerCase()) return null;
    }

    let common = 0;
    while (common < from.length && common < to.length && from[common] === to[common]) {
        common += 1;
    }

    const up = from.length - common;
    const rest = to.slice(common);
    const parts = [...Array<string>(up).fill(".."), ...rest];
    return parts.length === 0 ? "." : parts.join("/");
}

/** 把相对路径解析为绝对路径（`baseDir` 必须是绝对路径）。 */
export function resolvePath(baseDir: string, relative: string): string {
    const normalized = toPosix(relative);
    if (isAbsolutePath(normalized)) return normalized;
    const base = splitSegments(baseDir);
    const parts = normalized.split("/");
    for (const part of parts) {
        if (!part || part === ".") continue;
        if (part === "..") {
            base.pop();
            continue;
        }
        base.push(part);
    }
    const joined = base.join("/");
    // 保留 Windows 盘符后的根斜杠：`C:/a` 已经正确；POSIX 根 `/a` 需要补回。
    if (toPosix(baseDir).startsWith("/") && !joined.startsWith("/")) return `/${joined}`;
    return joined;
}

/**
 * 把路径编码成可以放进 Markdown 链接的形式。
 *
 * 空格、`(`、`)`、`#`、`?` 都会破坏链接或引用语法，逐段做百分号编码；
 * 中文等非 ASCII 字符保留原样（Markdown 与 markdown-it 都支持，可读性更好）。
 */
export function encodePathForMarkdown(path: string): string {
    return toPosix(path)
        .split("/")
        .map((segment) =>
            segment.replace(/[ ()#?%<>[\]\\^`{|}]/g, (ch) => `%${ch.charCodeAt(0).toString(16).toUpperCase().padStart(2, "0")}`),
        )
        .join("/");
}

/** `encodePathForMarkdown` 的逆操作（解码每一段）。 */
export function decodePathFromMarkdown(path: string): string {
    return path
        .split("/")
        .map((segment) => {
            try {
                return decodeURIComponent(segment);
            } catch {
                return segment;
            }
        })
        .join("/");
}
