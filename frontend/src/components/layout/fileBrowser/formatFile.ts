/**
 * 文件元信息的格式化（大小 / 修改时间）。
 *
 * 【为什么单独成文件】它们被行组件与属性对话框共用；放在行组件里会让那个文件
 * 同时导出组件与函数，破坏 Fast Refresh 的"只导出组件"约定（eslint 会报
 * `react-refresh/only-export-components`）。
 */

/** 格式化文件大小。 */
export function formatSize(bytes: number | null): string {
    if (bytes == null) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    // ≥1 GiB 之前一律显示成 "4096.0 MB"，大到读不出量级 —— 补上 GB / TB 档。
    if (bytes < 1024 * 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
    return `${(bytes / (1024 * 1024 * 1024 * 1024)).toFixed(1)} TB`;
}

/** 格式化修改时间（`FileEntry.modifiedTime` 是 Unix 秒）。 */
export function formatModified(seconds: number | null): string {
    if (seconds == null) return "";
    return new Date(seconds * 1000).toLocaleString();
}
