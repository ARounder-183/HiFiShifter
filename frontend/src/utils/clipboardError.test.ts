import { describe, expect, it } from "vitest";
import { clipboardErrorKey } from "./clipboardError";

describe("clipboardErrorKey（剪贴板错误码 → i18n key）", () => {
    it("按首个错误码 token 匹配，忽略回退链后缀", () => {
        expect(
            clipboardErrorKey("timeline_clipboard_empty; Reaper clipboard: clipboard_empty"),
        ).toBe("clipboard_error_empty");
        expect(clipboardErrorKey("clipboard_empty")).toBe("clipboard_error_empty");
    });

    it("被参数线复制覆盖后的时间轴粘贴（原 fixint 解析错误场景）映射为剪贴板为空", () => {
        // 单剪贴板纪律下，槽位里不是时间轴载荷（含参数线 JSON 首字节
        // `{` 被当作 fixint 123 的历史场景）统一归为 timeline_clipboard_empty。
        expect(
            clipboardErrorKey("timeline_clipboard_empty; Reaper clipboard: clipboard_empty"),
        ).toBe("clipboard_error_empty");
    });

    it("写入期错误与未知错误码不误映射（回退显示原文）", () => {
        // "clipboard_empty_failed" 是写入失败，不能误映射成"剪贴板为空"。
        expect(clipboardErrorKey("clipboard_empty_failed: os error 5")).toBe("");
        expect(clipboardErrorKey("clipboard_open_failed: timeout")).toBe("");
        expect(clipboardErrorKey("clipboard_parse_failed: invalid type: integer `123`")).toBe("");
        expect(clipboardErrorKey("some_other_error")).toBe("");
    });
});
