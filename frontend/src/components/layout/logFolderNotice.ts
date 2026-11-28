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
    | { kind: "path"; path: string }
    /**
     * 日志功能本身关着：路径照给，但必须先说清"里面不会有东西"。
     *
     * 【为什么单独一种】用户点「打开日志文件夹」时默认假设"那里有我刚才那次
     * 崩溃的记录"。若档位是「无」，目录是空的 —— 此时只显示路径等于让他去
     * 找一个永远不存在的文件。原因（没开日志）比路径更该先看到。
     */
    | { kind: "disabled"; path: string };

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
    // 日志没在写：路径仍然要给（用户可能想确认目录位置），但理由必须一起给。
    if (res.fileEnabled === false) {
        const path = res.file || res.path;
        return path ? { kind: "disabled", path } : { kind: "none" };
    }
    if (pluginMode && res.opened) {
        return { kind: "none" };
    }
    // 独立 App 没有 `opened` 字段：`ok` 即表示已经打开了。
    if (!pluginMode && res.opened !== false) {
        return { kind: "none" };
    }
    // 优先给**文件**路径：用户真正想复制给开发者的往往是 `plugin-<format>-<host>.log` 本身。
    const path = res.file || res.path;
    return path ? { kind: "path", path } : { kind: "none" };
}
