import { test } from "vitest";

import {
    clipAudibleGain,
    clipItemGain,
    clipTakeGain,
    clipTakeWaveformGain,
    clipWaveformGain,
} from "./clipGain";

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

test("clip gain layers multiply and preserve take polarity", () => {
    assertEqual(clipTakeGain({ gain: -0.5 }), -0.5, "take sign survives");
    assertEqual(clipAudibleGain({ clip_gain: 2, gain: -0.5 }), -1, "item × signed take");
    assertEqual(clipWaveformGain({ clip_gain: 2, gain: -0.5 }), 1, "waveform is absolute");
    assertEqual(clipTakeWaveformGain({ clip_gain: 4 }, -0.25), 1, "per-take lane amplitude");
    assertEqual(clipAudibleGain({ clip_gain: 0, gain: 4 }), 0, "item silence wins");
});
