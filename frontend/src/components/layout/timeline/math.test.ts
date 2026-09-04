/**
 * 时间轴数学工具的不变量回归（./math）。
 *
 * 覆盖 `formatEditNumber`：编辑态数值文本必须保留 6 位小数精度、只清理
 * 浮点噪声而不降到展示级取整——由时长/BPM 换算出的倍率（如 1.2456）被
 * 舍入为 1.25 后，提交的实际倍率就与用户输入的时长不再精确对应。
 */
import { expect, test } from "vitest";

import { formatEditNumber } from "./math";

test("formatEditNumber keeps 6-decimal precision without display rounding", () => {
    expect(formatEditNumber(1.2456)).toBe("1.2456");
    // 浮点噪声被清理，但精度不降到展示级
    expect(formatEditNumber(1.2556000000001)).toBe("1.2556"); // 超出 6 位小数的浮点噪声
    expect(formatEditNumber(0.1 + 0.02)).toBe("0.12");
    expect(formatEditNumber(2)).toBe("2");
    expect(formatEditNumber(NaN)).toBe("0");
});
