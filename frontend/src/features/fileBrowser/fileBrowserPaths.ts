/**
 * 文件浏览器的路径算术（纯函数）。
 *
 * 【为什么单独成模块】`parentDirOf` / `locationLabel` 都是"给一条路径算出另一条
 * 字符串"，与面板的 Redux / Tauri 依赖无关。放在面板组件文件里就既不能单测、
 * 又会破坏 Fast Refresh 的"只导出组件"约定（eslint 会报
 * `react-refresh/only-export-components`）。
 *
 * 两个函数都要同时处理 Windows 反斜杠与 POSIX 正斜杠 —— 路径来自后端，格式随平台。
 */

import { FILE_BROWSER_COMPUTER_PATH } from "./fileBrowserSlice";

/** 把路径统一成可比较、可切分的正斜杠形式（去掉尾部斜杠）。 */
function normalizeSeparators(path: string): string {
    return path.replace(/\\/g, "/").replace(/\/+$/, "");
}

/**
 * 取所在目录；没有上级（盘符根 / `/`）时返回 `null`。
 *
 * 【为什么盘符根要保留尾部斜杠】`D:\music` 的上级是 `D:\` 而不是 `D:` —— 后者在
 * Windows 上表示"当前目录所在的盘"，不是一个可列出的目录。
 */
export function parentDirOf(path: string): string | null {
    const normalized = normalizeSeparators(path);
    const cut = normalized.lastIndexOf("/");
    if (cut <= 0) return null;
    const parent = normalized.slice(0, cut);
    if (/^[A-Za-z]:$/.test(parent)) {
        return path.includes("\\") ? `${parent}\\` : `${parent}/`;
    }
    return path.includes("\\") ? parent.replace(/\//g, "\\") : parent;
}

/**
 * 常用位置的显示名：取路径最后一段；盘符根显示成 `C:`；「计算机」用传入的本地化名。
 *
 * 【为什么不用整条路径】菜单里放 `D:\Projects\HiFiShifter\assets\audio\takes` 会把
 * 菜单撑得比屏幕宽，而真正用来辨认的往往只是最后一段。完整路径走 tooltip。
 *
 * @param computerLabel 「计算机」层的本地化名称（由调用方从 i18n 取）。
 */
export function locationLabel(path: string, computerLabel: string): string {
    if (path === FILE_BROWSER_COMPUTER_PATH) return computerLabel;
    const normalized = normalizeSeparators(path);
    const cut = normalized.lastIndexOf("/");
    const last = cut >= 0 ? normalized.slice(cut + 1) : normalized;
    return last || normalized;
}
