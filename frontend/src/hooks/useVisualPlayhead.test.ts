// 视觉插值回归：宿主采样缺失有上限；不抹除真实seek/loop，也不改变独立App行为。
import { expect, test } from "vitest";
import { projectVisualPlayhead } from "./useVisualPlayhead";

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
