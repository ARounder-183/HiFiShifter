/**
 * 「打开日志文件夹」的结果 → 该给用户看什么。
 *
 * 【要钉死什么】插件里点这个菜单曾经只弹一段**不可选中、不可复制**的路径文本，
 * 而它恰恰是用户最需要交给我们的时候。这里把"什么时候必须把路径摆出来"钉住。
 */
import { expect, test } from "vitest";

import { logFolderNotice } from "./logFolderNotice";

const PATH = String.raw`C:\Users\me\AppData\Local\com.arounder.hifishitter\logs`;
const FILE = `${PATH}\plugin.log`;

test("a failure reports the backend detail", () => {
    expect(logFolderNotice({ ok: false, error: "boom" }, true)).toEqual({
        kind: "error",
        detail: "boom",
    });
});

test("the standalone app stays silent once the folder opens", () => {
    // 独立 App 的 `open_log_folder` 不带 `opened`：`ok` 即已打开。
    expect(logFolderNotice({ ok: true, path: PATH }, false)).toEqual({ kind: "none" });
});

test("the plugin stays silent when the file manager really opened", () => {
    expect(logFolderNotice({ ok: true, path: PATH, file: FILE, opened: true }, true)).toEqual({
        kind: "none",
    });
});

test("the plugin hands over the path when it could not open the file manager", () => {
    expect(logFolderNotice({ ok: true, path: PATH, file: FILE, opened: false }, true)).toEqual({
        kind: "path",
        // 优先给文件路径：用户想复制给开发者的往往是 plugin.log 本身。
        path: FILE,
    });
});

test("the directory is used when no file path is reported", () => {
    expect(logFolderNotice({ ok: true, path: PATH, opened: false }, true)).toEqual({
        kind: "path",
        path: PATH,
    });
});

test("nothing is shown when there is no path to show", () => {
    expect(logFolderNotice({ ok: true, opened: false }, true)).toEqual({ kind: "none" });
});
