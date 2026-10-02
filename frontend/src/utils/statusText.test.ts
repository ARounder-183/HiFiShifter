/**
 * 状态行文案解析。
 *
 * 【为什么值得单测】这些正则一旦写错，表现是"非英语语系下状态行变成英文原文" ——
 * 不报错、不崩溃，只是静默降级。目录导入的汇总（成功数 / 失败数）正是这一类：
 * 两个数量、单数量模板表达不了，必须有一条专门的模式。
 */

import { describe, expect, it } from "vitest";

import { resolveStatusText } from "./statusText";

/** 把 i18n 键当模板直接返回，便于断言"走了哪条键、回填了什么"。 */
const t = (key: string, _count?: number) => key;

describe("目录导入的汇总状态行", () => {
    it("解析成两个数量的模板", () => {
        const text = resolveStatusText("Folder import: 47 imported, 3 failed", {}, (key) =>
            key === "status_folder_import_summary" ? "已导入 {m} 个，{n} 个无法导入" : key,
        );
        expect(text).toBe("已导入 47 个，3 个无法导入");
    });

    it("全部成功时同样成立（失败数为 0）", () => {
        const text = resolveStatusText("Folder import: 12 imported, 0 failed", {}, (key) =>
            key === "status_folder_import_summary" ? "{m}|{n}" : key,
        );
        expect(text).toBe("12|0");
    });

    it("不带数量的普通导入状态走既有的键映射", () => {
        expect(resolveStatusText("Import done", { "Import done": "status_import_done" }, t)).toBe(
            "status_import_done",
        );
    });

    it("认不出的状态原样返回，不猜", () => {
        expect(resolveStatusText("Something odd happened", {}, t)).toBe("Something odd happened");
    });
});
