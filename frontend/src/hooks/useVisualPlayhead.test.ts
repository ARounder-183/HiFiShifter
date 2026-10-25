// 视觉插值回归：宿主采样缺失有上限；不抹除真实seek/loop，也不改变独立App行为。
import { expect, test } from "vitest";
import { catchUpSampledPlayhead, projectVisualPlayhead } from "./useVisualPlayhead";

test("host interpolation fills only a short sampling gap", () => {
    expect(projectVisualPlayhead(3, 0.03, true)).toBe(3.03);
    expect(projectVisualPlayhead(3, 8, true)).toBe(3.1);
    expect(projectVisualPlayhead(1, 0, true)).toBe(1);
    expect(projectVisualPlayhead(1, -1, true)).toBe(1);
    expect(projectVisualPlayhead(1, NaN, true)).toBe(1);
});
test("standalone visual playback retains unbounded continuous extrapolation", () => {
    expect(projectVisualPlayhead(3, 8, false)).toBe(11);
});

test("catching up a sample is not capped the way projecting forward is", () => {
    // 已经过去 300 ms：补满 300 ms，而不是被宿主模式的 100 ms 上限吃掉 200 ms。
    expect(catchUpSampledPlayhead(5, 0.3)).toBeCloseTo(5.3, 9);
    // 同一组入参走"向前推测"则必须仍被夹住 —— 两者语义不同，不能混用。
    expect(projectVisualPlayhead(5, 0.3, true)).toBeCloseTo(5.1, 9);
    // 非有限/负值不推进位置。
    expect(catchUpSampledPlayhead(2, Number.NaN)).toBe(2);
    expect(catchUpSampledPlayhead(2, -1)).toBe(2);
});
