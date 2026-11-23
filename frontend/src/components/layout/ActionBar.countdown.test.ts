import { expect, test } from "vitest";
import { readFileSync } from "node:fs";

test("录音倒计时使用固定槽位，-3 到 -10 不改变可占宽度", () => {
    const source = readFileSync("src/components/layout/ActionBar.tsx", "utf8");
    expect(source).toContain('data-testid="recording-countdown"');
    expect(source).toContain('reserveText="-99"');
    expect(source).toContain("<SlotText");
});
