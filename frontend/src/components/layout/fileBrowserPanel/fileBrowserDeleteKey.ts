/**
 * 文件浏览器的 Delete / Backspace 语义（平台差异集中在这里）。
 *
 * Finder 惯例：
 * - macOS 主 Delete 键（⌫）不带修饰 = 返回上级目录；
 * - ⌘+⌫ = 把选中项移到废纸篓；
 * - Fn+Delete = 前向删除（event.key === "Delete"），同样删除选中项。
 *
 * Windows / Linux：Backspace = 上级目录，Delete = 删除。
 */
import { IS_MAC } from "../../../utils/platform";

export type FileBrowserDeleteAction = "delete" | "parent" | null;

export function resolveFileBrowserDeleteAction(
    event: {
        key: string;
        metaKey: boolean;
        ctrlKey: boolean;
        altKey: boolean;
    },
    mac: boolean = IS_MAC,
): FileBrowserDeleteAction {
    if (event.altKey) return null;
    if (event.key === "Delete" && !event.metaKey && !event.ctrlKey) return "delete";
    if (mac && event.metaKey && !event.ctrlKey && event.key === "Backspace") return "delete";
    if (!event.metaKey && !event.ctrlKey && event.key === "Backspace") return "parent";
    return null;
}
