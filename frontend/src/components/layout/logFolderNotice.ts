/**
 * 「打开日志文件夹」的结果该怎么呈现。
 *
 * 【为什么单独一个纯函数】这段判断此前内联在 `MenuBar` 的一个 `onSelect` 里，
 * 而 `MenuBar` 没有测试宿主（它太大）。可这里恰恰是用户抱怨的地方：插件里点菜单
 * 只弹一段文字、还不能复制。把它抽出来，这条行为就有了回归测试。
 */

import type { LogFolderResult } from "../../services/api/diagnostics";

/** 该给用户看什么。 */
export type LogFolderNotice =
    /** 什么都不用弹（资源管理器已经打开，或独立 App 已经处理好了）。 */
    | { kind: "none" }
    /** 出错：把后端的原文（或兜底文案）显示出来。 */
    | { kind: "error"; detail: string }
    /** 打不开：把路径摆出来，必须可选中、可复制。 */
    | { kind: "path"; path: string };

/**
 * 判定结果。
 *
 * 【为什么插件要区分 `opened`】独立 App 走 Tauri opener，成功即静默；插件是**尽力
 * 而为**地调用系统 shell（`browser_files::reveal_directory`），可能被宿主环境拒绝。
 * 失败时不能假装成功 —— 用户最需要路径的时刻正是"打不开"的时候。
 */
export function logFolderNotice(res: LogFolderResult, pluginMode: boolean): LogFolderNotice {
    if (!res.ok) {
        return { kind: "error", detail: res.error ?? "" };
    }
    if (pluginMode && res.opened) {
        return { kind: "none" };
    }
    // 独立 App 没有 `opened` 字段：`ok` 即表示已经打开了。
    if (!pluginMode && res.opened !== false) {
        return { kind: "none" };
    }
    // 优先给**文件**路径：用户真正想复制给开发者的往往是 `plugin.log` 本身。
    const path = res.file || res.path;
    return path ? { kind: "path", path } : { kind: "none" };
}
