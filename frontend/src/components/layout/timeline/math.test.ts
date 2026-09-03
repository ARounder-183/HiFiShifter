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
