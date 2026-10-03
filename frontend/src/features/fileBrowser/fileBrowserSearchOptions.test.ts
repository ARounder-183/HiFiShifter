/**
 * 文件浏览器搜索参数与媒体过滤的契约。
 *
 * 【这里钉住的是同一条规则的两个方向】
 * 「仅显示媒体文件」这个开关同时决定两件事：
 *   - 后端要不要把**目录**作为搜索结果产出（`includeDirs`）；
 *   - 目录列表里要不要滤掉非媒体文件。
 *
 * 两件事此前分别写在"下发参数"和"一行 filter"里，各自解释同一个开关。一旦某次
 * 改动把其中一处拆开，症状是：关掉开关后目录仍不出现（搜不到想跳转的子目录），
 * 或打开开关后目录混进"只要媒体"的清单。这两条断言把关系固定下来。
 */

import { describe, expect, test } from "vitest";

import { fileBrowserSearchOptions, fileBrowserSearchRequest, visibleFileBrowserEntries } from "./fileBrowserSearchOptions";
import { DEFAULT_SEARCH_SETTINGS } from "../search/searchSettings";
import type { FileEntry } from "../../services/api/fileBrowser";

function entry(name: string, isDir = false): FileEntry {
    const dot = name.lastIndexOf(".");
    return {
        name,
        path: `C:/music/${name}`,
        isDir,
        size: isDir ? null : 1024,
        extension: isDir || dot <= 0 ? null : name.slice(dot + 1).toLowerCase(),
        modifiedTime: null,
    };
}

const BASE_INPUT = {
    settings: DEFAULT_SEARCH_SETTINGS,
    regexEnabled: false,
    showHiddenFiles: false,
};

describe("includeDirs 与 mediaOnly 反相关", () => {
    test("关闭「仅显示媒体文件」→ 搜索请求包含目录", () => {
        const options = fileBrowserSearchOptions({ ...BASE_INPUT, mediaOnly: false });
        expect(options.includeDirs).toBe(true);
    });

    test("开启「仅显示媒体文件」→ 搜索请求不含目录", () => {
        const options = fileBrowserSearchOptions({ ...BASE_INPUT, mediaOnly: true });
        expect(options.includeDirs).toBe(false);
    });

    test("正则模式把匹配模式降为 off，但不影响 includeDirs", () => {
        const options = fileBrowserSearchOptions({
            ...BASE_INPUT,
            regexEnabled: true,
            mediaOnly: false,
        });
        expect(options.mode).toBe("off");
        expect(options.includeDirs).toBe(true);
    });

    test("隐藏项口径跟随「显示隐藏文件」", () => {
        expect(fileBrowserSearchOptions({ ...BASE_INPUT, mediaOnly: false }).includeHidden).toBe(
            false,
        );
        expect(
            fileBrowserSearchOptions({
                ...BASE_INPUT,
                mediaOnly: false,
                showHiddenFiles: true,
            }).includeHidden,
        ).toBe(true);
    });
});

describe("目录列表的媒体过滤", () => {
    const entries = [
        entry("Takes", true),
        entry("take_01.wav"),
        entry("notes.txt"),
        entry("melody.mid"),
    ];

    test("关闭媒体开关：全部可见", () => {
        expect(
            visibleFileBrowserEntries(entries, { isSearchMode: false, mediaOnly: false }),
        ).toHaveLength(4);
    });

    test("开启媒体开关：目录仍在，非媒体文件被滤掉", () => {
        const visible = visibleFileBrowserEntries(entries, {
            isSearchMode: false,
            mediaOnly: true,
        });
        // 目录必须保留：浏览一个目录时它的子目录是导航入口，滤掉就出不去了。
        expect(visible.map((e) => e.name)).toEqual(["Takes", "take_01.wav", "melody.mid"]);
    });

    test("搜索模式不二次过滤：目录由后端的 includeDirs 决定", () => {
        // 关键：搜索模式下即便 mediaOnly 开着，这里也必须原样返回 —— 否则
        // "后端给不给目录"和"前端滤不滤目录"会互相打架。
        const visible = visibleFileBrowserEntries(entries, {
            isSearchMode: true,
            mediaOnly: true,
        });
        expect(visible).toHaveLength(4);
    });
});

describe("fileBrowserSearchRequest", () => {
    const options = fileBrowserSearchOptions({ ...BASE_INPUT, mediaOnly: false });

    test("普通模式下原文下发（后端负责匹配）", () => {
        const request = fileBrowserSearchRequest({
            dirPath: "C:\\music",
            query: "vocal",
            regexEnabled: false,
            options,
        });
        expect(request).toEqual({ dirPath: "C:\\music", query: "vocal", options });
    });

    test("正则模式下 query 传空串（后端不过滤，前端自己筛）", () => {
        const request = fileBrowserSearchRequest({
            dirPath: "C:\\music",
            query: "^vocal\\d+$",
            regexEnabled: true,
            options,
        });
        // 把正则原文发给后端会得到"另一批结果"，而前端再按正则筛一次 ——
        // 列表于是与输入时看到的不一致。
        expect(request.query).toBe("");
        expect(request.dirPath).toBe("C:\\music");
    });

    test("options 原样透传（调用方按切换后的模式决定它）", () => {
        const regexOptions = { ...options, mode: "off" as const };
        const request = fileBrowserSearchRequest({
            dirPath: "C:\\music",
            query: "abc",
            regexEnabled: true,
            options: regexOptions,
        });
        expect(request.options).toBe(regexOptions);
    });
});
