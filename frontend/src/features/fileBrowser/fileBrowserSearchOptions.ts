/**
 * 文件浏览器搜索参数：视图状态 → 后端查询参数。
 *
 * 【为什么单独成模块】"仅显示媒体文件"与"搜索结果里有没有文件夹"是同一条规则的
 * 两个说法。用户**关掉**这个开关时说的是"我要在这个目录下找东西"——子目录是合法
 * 命中；**打开**时说的是"我只要可导入的媒体"——目录不是媒体文件。
 *
 * 这条绑定此前只活在面板内部：一半在后端调用参数里、一半在一行 `e.isDir ||` 的
 * 过滤里，两处各自解释同一个开关，很容易在某次改动中拆开（关掉开关后目录仍被
 * 滤掉，或打开后目录混进媒体清单）。集中到这里，两处消费同一份判断。
 */

import {
    searchOptionsPayload,
    type SearchOptionsPayload,
    type SearchSettings,
} from "../search/searchSettings";
import type { FileEntry } from "../../services/api/fileBrowser";
import { isMediaFile } from "./fileKinds";

export interface FileBrowserSearchInput {
    settings: SearchSettings;
    /** 正则模式：正则作用于**原文**，与转写互斥，因此匹配模式降为 `off`。 */
    regexEnabled: boolean;
    /** 「仅显示媒体文件」当前是否开启。 */
    mediaOnly: boolean;
    /** 「显示隐藏文件」当前是否开启。 */
    showHiddenFiles: boolean;
}

/**
 * 组装下发给 `search_files_recursive` 的参数。
 *
 * 【为什么 `includeDirs` 与 `mediaOnly` 是反相关】见文件头。这条关系是**契约**，
 * 不是巧合 —— 断言它的测试会挡住"顺手把目录也滤掉"这类回归。
 */
export function fileBrowserSearchOptions(input: FileBrowserSearchInput): SearchOptionsPayload {
    const payload = searchOptionsPayload(input.settings);
    const base: SearchOptionsPayload = input.regexEnabled ? { ...payload, mode: "off" } : payload;
    return {
        ...base,
        includeDirs: !input.mediaOnly,
        // 与目录列表同口径：列表里被隐藏的项，搜索也不该搜得到。
        includeHidden: input.showHiddenFiles,
    };
}

/**
 * 组装一次 `search_files_recursive` 的完整参数。
 *
 * 【为什么必须共用】正则模式下后端不参与过滤：`query` 要传空串，由前端按正则筛
 * `searchResults`（见面板的 `regexFilteredEntries`）。这条规则若在"输入时"与
 * "写操作后刷新列表时"各写一遍，两者迟早分叉 —— 刷新会把正则原文发给后端，于是
 * 刷新后的结果与输入时的不是同一批（用户看到列表"变了个样"）。
 *
 * @param input.query 用户输入的原文（本函数负责在正则模式下改写成空串）。
 */
export function fileBrowserSearchRequest(input: {
    dirPath: string;
    query: string;
    regexEnabled: boolean;
    options: SearchOptionsPayload;
}): { dirPath: string; query: string; options: SearchOptionsPayload } {
    return {
        dirPath: input.dirPath,
        query: input.regexEnabled ? "" : input.query,
        options: input.options,
    };
}

/**
 * 「仅显示媒体文件」在**目录列表**（非搜索）下的过滤。
 *
 * 【为什么搜索模式直接原样返回】搜索模式下目录是否出现已由后端的 `includeDirs`
 * 决定（见 `fileBrowserSearchOptions`）。这里再判一次就会互相打架：关闭开关时
 * `entry.isDir ||` 恒真（等于没过滤），打开开关时又把目录挡掉（目录不是媒体文件）
 * —— 同一个开关被解释两次，且两次结论相反。
 *
 * 【为什么目录列表里目录永远可见】你在浏览一个目录，它的子目录当然要列出来，
 * 否则无法导航。`mediaOnly` 只筛文件。
 */
export function visibleFileBrowserEntries(
    entries: readonly FileEntry[],
    input: { isSearchMode: boolean; mediaOnly: boolean },
): FileEntry[] {
    if (input.isSearchMode || !input.mediaOnly) return [...entries];
    return entries.filter((entry) => entry.isDir || isMediaFile(entry));
}
