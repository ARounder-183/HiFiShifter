/*
 * 路径工具（纯函数，可单测）。
 *
 * 附件本身内嵌在工程文件里（`hifi-asset://`），这里只处理**用户手写**的
 * 相对/绝对路径引用：正文里写 `![](素材/截图.png)` 时，要相对工程目录解析
 * 成绝对路径才能读到文件。
 *
 * Windows 的盘符、反斜杠都要照顾到 —— 这里统一用正斜杠做规范形态，只在拼
 * 绝对路径时还原。
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
        normalized.startsWith("/") || /^[A-Za-z]:\//.test(normalized) || normalized.startsWith("//")
    );
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

/** 解码 Markdown 里的路径（逐段 decodeURIComponent，容错）。 */
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
