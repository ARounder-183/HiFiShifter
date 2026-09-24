/**
 * 变化点文本的滚轮步进自检。
 *
 * 【要钉死的语义】滚轮只改**文本里的 BPM 数字**，其余部分逐字符保留 —— 包括用户
 * 手打的、解析器认不出来的片段。若改成"解析后再序列化"，这些内容会被抹掉。
 */
import { describe, expect, it } from "vitest";

import { applyWheelToTempoText } from "./tempoMap.js";

describe("applyWheelToTempoText（变化点文本的滚轮步进）", () => {
    it("只替换前导 BPM，其余文本原样保留", () => {
        expect(applyWheelToTempoText("120 4/4 - C / Am", 1, 1)).toBe("121 4/4 - C / Am");
        expect(applyWheelToTempoText("120 4/4 - C / Am", -1, 1)).toBe("119 4/4 - C / Am");
    });

    it("精细调整步长（0.1）保留小数位", () => {
        expect(applyWheelToTempoText("120.5 - X", 1, 0.1)).toBe("120.6 - X");
        expect(applyWheelToTempoText("120 - X", 1, 0.1)).toBe("120.1 - X");
    });

    it("★ 无法解析的尾串也逐字符保留（不做序列化重建）", () => {
        expect(applyWheelToTempoText("120 我的备注 !!", 1, 1)).toBe("121 我的备注 !!");
        expect(applyWheelToTempoText("120", 1, 1)).toBe("121");
    });

    it("不以数字开头时返回 null（绝不覆盖用户正在输入的内容）", () => {
        expect(applyWheelToTempoText("abc", 1, 1)).toBeNull();
        expect(applyWheelToTempoText("", 1, 1)).toBeNull();
        expect(applyWheelToTempoText("   ", 1, 1)).toBeNull();
        expect(applyWheelToTempoText("- 120", 1, 1)).toBeNull();
    });

    it("到 BPM 边界时返回 null（不产生无变化的写入）", () => {
        expect(applyWheelToTempoText("960", 1, 1)).toBeNull();
        expect(applyWheelToTempoText("10", -1, 1)).toBeNull();
    });

    it("保留前导空白（用户缩进不影响）", () => {
        expect(applyWheelToTempoText("  120", 1, 1)).toBe("  121");
    });

    it("越过边界时钳制到合法范围", () => {
        expect(applyWheelToTempoText("959.5", 1, 1)).toBe("960");
        expect(applyWheelToTempoText("10.5", -1, 1)).toBe("10");
    });
});
