import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("录音倒计时使用固定槽位，-3 到 -10 不改变可占宽度", () => {
    // 【为什么读的是 ToolbarStrip 而不是 ActionBar】工具栏的渲染体住在
    // `toolbar/ToolbarStrip.tsx`（桌面与移动共用同一份）。`ActionBar` 只剩一行
    // 薄壳，读它只会读到一句 `<ToolbarStrip variant="desktop" />` —— 那样的断言
    // 会永远失败，而不是永远通过，因此这里必须跟着定义走。
    const source = readFileSync("src/components/layout/toolbar/ToolbarStrip.tsx", "utf8");
    expect(source).toContain('data-testid="recording-countdown"');
    expect(source).toContain('reserveText="-99"');
    expect(source).toContain("<SlotText");
});
