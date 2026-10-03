import { describe, expect, it } from "vitest";

import {
    DEFAULT_FILE_BROWSER_VIEW_OPTIONS,
    DEFAULT_SORT_DESCENDING,
    defaultSortDescending,
    migrateLegacyMediaOnly,
    normalizeFileBrowserViewOptions,
    rowDensityOf,
} from "./fileBrowserViewOptions";

describe("normalizeFileBrowserViewOptions", () => {
    it("缺省 / 非法输入整体回落到默认（紧凑、按名称升序、目录优先、显示状态行）", () => {
        for (const input of [undefined, null, 42, "nope", true, []]) {
            expect(normalizeFileBrowserViewOptions(input)).toEqual(
                DEFAULT_FILE_BROWSER_VIEW_OPTIONS,
            );
        }
    });

    it("空对象也得到一份完整默认值（不是空对象）", () => {
        const options = normalizeFileBrowserViewOptions({});
        expect(options).toEqual(DEFAULT_FILE_BROWSER_VIEW_OPTIONS);
        // 必须是拷贝：改一份不能影响另一份。
        options.showHiddenFiles = true;
        expect(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.showHiddenFiles).toBe(false);
    });

    it("保留合法字段、只把非法字段回落", () => {
        const options = normalizeFileBrowserViewOptions({
            sortMode: "size",
            sortDescending: true,
            showHiddenFiles: true,
            // 非法：枚举之外 / 类型不对
            density: "enormous",
            detailsColumn: 7,
            foldersFirst: "yes",
        });
        expect(options.sortMode).toBe("size");
        expect(options.sortDescending).toBe(true);
        expect(options.showHiddenFiles).toBe(true);
        expect(options.density).toBe(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.density);
        expect(options.detailsColumn).toBe(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.detailsColumn);
        expect(options.foldersFirst).toBe(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.foldersFirst);
    });

    it("枚举取值只接受白名单内的字符串", () => {
        expect(normalizeFileBrowserViewOptions({ sortMode: "date" }).sortMode).toBe("date");
        expect(normalizeFileBrowserViewOptions({ sortMode: "Date" }).sortMode).toBe("name");
        expect(normalizeFileBrowserViewOptions({ density: "comfortable" }).density).toBe(
            "comfortable",
        );
        expect(normalizeFileBrowserViewOptions({ detailsColumn: "none" }).detailsColumn).toBe(
            "none",
        );
    });

    it("点击试听默认开启（老配置缺这一项时升级后应当出声）", () => {
        expect(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.previewOnClick).toBe(true);
        // 缺键的旧配置：回落到默认（开），而不是被当成 false。
        expect(normalizeFileBrowserViewOptions({ mediaOnly: true }).previewOnClick).toBe(true);
    });

    it("点击试听：显式关闭被保留，非法值回落默认", () => {
        expect(normalizeFileBrowserViewOptions({ previewOnClick: false }).previewOnClick).toBe(
            false,
        );
        expect(normalizeFileBrowserViewOptions({ previewOnClick: true }).previewOnClick).toBe(true);
        // 类型不对（字符串 / 数字）一律回落，而不是被真值化。
        expect(normalizeFileBrowserViewOptions({ previewOnClick: "false" }).previewOnClick).toBe(
            true,
        );
        expect(normalizeFileBrowserViewOptions({ previewOnClick: 0 }).previewOnClick).toBe(true);
    });

    it("点击试听与移动试听正交（各自的默认互不影响）", () => {
        expect(DEFAULT_FILE_BROWSER_VIEW_OPTIONS.previewOnNavigate).toBe(false);
        const options = normalizeFileBrowserViewOptions({
            previewOnClick: false,
            previewOnNavigate: true,
        });
        expect(options.previewOnClick).toBe(false);
        expect(options.previewOnNavigate).toBe(true);
    });
});

describe("排序方向的自然默认", () => {
    it("名称升序，日期与大小降序", () => {
        expect(DEFAULT_SORT_DESCENDING).toEqual({ name: false, date: true, size: true });
        expect(defaultSortDescending("name")).toBe(false);
        expect(defaultSortDescending("date")).toBe(true);
        expect(defaultSortDescending("size")).toBe(true);
    });
});

describe("migrateLegacyMediaOnly", () => {
    it("配置里没有 mediaOnly 时，用旧 localStorage 的值补上", () => {
        expect(migrateLegacyMediaOnly({}, "true")).toEqual({ mediaOnly: true });
        expect(migrateLegacyMediaOnly({}, "false")).toEqual({ mediaOnly: false });
    });

    it("配置里已有 mediaOnly 时以配置为准（旧键不得打回旧值）", () => {
        expect(migrateLegacyMediaOnly({ mediaOnly: false }, "true")).toEqual({
            mediaOnly: false,
        });
        expect(migrateLegacyMediaOnly({ mediaOnly: true }, "false")).toEqual({ mediaOnly: true });
    });

    it("没有旧键时原样返回", () => {
        expect(migrateLegacyMediaOnly({}, null)).toEqual({});
        expect(migrateLegacyMediaOnly(undefined, "true")).toBeUndefined();
    });
});

describe("rowDensityOf", () => {
    it("视图语义映射到行原语的取值", () => {
        expect(rowDensityOf("compact")).toBe("compact");
        expect(rowDensityOf("comfortable")).toBe("default");
    });
});
