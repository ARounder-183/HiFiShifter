import { test } from "vitest";

import { clipItemGain } from "./clipGain";

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    if (actual !== expected) {
        throw new Error(label + ": expected " + String(expected) + ", received " + String(actual));
    }
}

test("clipItemGain reads the item layer and never mistakes silence for absence", () => {
    assertEqual(clipItemGain({ clip_gain: 0.5 }), 0.5, "explicit item gain");
    // 0 是合法的静音值 —— 用 `|| 1` 兜底会把它变成"不增不减"，那是一次静默的听感改变。
    assertEqual(clipItemGain({ clip_gain: 0 }), 0, "silence is a value, not a missing field");
    // 旧载荷没有这个字段：按单位增益处理（不改变听感）。
    assertEqual(clipItemGain({}), 1, "legacy payload falls back to unity");
    assertEqual(clipItemGain(null), 1, "missing clip falls back to unity");
    assertEqual(clipItemGain(undefined), 1, "undefined clip falls back to unity");
    // 非法值（NaN / Infinity）同样退回单位增益，而不是把 NaN 灌进渲染。
    assertEqual(clipItemGain({ clip_gain: Number.NaN }), 1, "NaN falls back to unity");
    assertEqual(clipItemGain({ clip_gain: Number.POSITIVE_INFINITY }), 1, "Infinity falls back");
    // 负值钳到静音（与后端 clamp_gain 一致）。
    assertEqual(clipItemGain({ clip_gain: -3 }), 0, "negative clamps to silence");
});
