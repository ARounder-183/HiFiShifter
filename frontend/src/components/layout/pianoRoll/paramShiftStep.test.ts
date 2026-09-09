import { test } from "vitest";

import {
    getParamShiftStep,
    parseParamShiftMagnitude,
    type ParamShiftMagnitude,
} from "./paramShiftStep.js";
import type { ProcessorParamDescriptor } from "../../../types/api.js";

const AUTOMATION = (min: number, max: number): ProcessorParamDescriptor["kind"] => ({
    type: "automation_curve",
    unit: "x",
    default_value: (min + max) / 2,
    min_value: min,
    max_value: max,
});

const DESCRIPTOR = (id: string, min: number, max: number): ProcessorParamDescriptor => ({
    id,
    display_name: id,
    group: "NSF-HiFiGAN",
    kind: AUTOMATION(min, max),
});

test("components/layout/pianoRoll/paramShiftStep.test.ts scripted checks", async () => {
    function assertEqual(actual: number, expected: number): void {
        if (Math.abs(actual - expected) > 1e-6) {
            throw new Error(`Expected ${expected}, received ${actual}`);
        }
    }

    // ── 默认档位（与既有行为一致）──────────────────────────────
    assertEqual(getParamShiftStep("pitch"), 1);
    assertEqual(getParamShiftStep("breath_gain", DESCRIPTOR("breath_gain", 0, 2)), 0.05);
    assertEqual(getParamShiftStep("hifigan_tension", DESCRIPTOR("hifigan_tension", -100, 100)), 5);
    assertEqual(getParamShiftStep("unknown_param"), 0.05);

    // ── 微调档（Ctrl 变体）────────────────────────────────────
    // 音高：±1 音分（0.01 半音）。
    assertEqual(getParamShiftStep("pitch", undefined, "fine"), 0.01);
    // 连续量参数：量程的 0.25%。
    assertEqual(getParamShiftStep("breath_gain", DESCRIPTOR("breath_gain", 0, 2), "fine"), 0.005);
    assertEqual(
        getParamShiftStep("hifigan_tension", DESCRIPTOR("hifigan_tension", -100, 100), "fine"),
        0.5,
    );
    // 未知参数：默认步长的 1/10。
    assertEqual(getParamShiftStep("unknown_param", undefined, "fine"), 0.005);

    // ── 大幅档（Shift 变体）───────────────────────────────────
    // 音高：±1200 音分 = 12 半音 = 一个八度。
    assertEqual(getParamShiftStep("pitch", undefined, "coarse"), 12);
    // 连续量参数：量程的 12.5%（默认步长的 5 倍）。
    assertEqual(getParamShiftStep("breath_gain", DESCRIPTOR("breath_gain", 0, 2), "coarse"), 0.25);
    assertEqual(
        getParamShiftStep("hifigan_tension", DESCRIPTOR("hifigan_tension", -100, 100), "coarse"),
        25,
    );
    // 未知参数：默认步长的 5 倍。
    assertEqual(getParamShiftStep("unknown_param", undefined, "coarse"), 0.25);

    // 子轨道音高偏移（音分）：默认 ±100（一个半音），微调 ±1 音分，
    // 大幅 ±1200（一个八度）。
    assertEqual(getParamShiftStep("child_pitch_offset_cents@track-1"), 100);
    assertEqual(getParamShiftStep("child_pitch_offset_cents@track-1", undefined, "fine"), 1);
    assertEqual(getParamShiftStep("child_pitch_offset_cents@track-1", undefined, "coarse"), 1200);

    // 子轨道度数偏移（音级）：默认/微调 ±1 音级，大幅 ±7（七声音阶八度）。
    assertEqual(getParamShiftStep("child_pitch_offset_degrees@track-1"), 1);
    assertEqual(getParamShiftStep("child_pitch_offset_degrees@track-1", undefined, "fine"), 1);
    assertEqual(getParamShiftStep("child_pitch_offset_degrees@track-1", undefined, "coarse"), 7);

    // 子轨道共振峰偏移（音分）：默认 ±50，微调 ±1，大幅 ±1200。
    assertEqual(getParamShiftStep("child_formant_offset_cents@track-1"), 50);
    assertEqual(getParamShiftStep("child_formant_offset_cents@track-1", undefined, "fine"), 1);
    assertEqual(getParamShiftStep("child_formant_offset_cents@track-1", undefined, "coarse"), 1200);

    // ── 幅度档解析：未知值回退 normal ─────────────────────────
    const magnitudes: ParamShiftMagnitude[] = ["normal", "coarse", "fine"];
    assertEqual(magnitudes.length, 3);
    for (const m of magnitudes) {
        if (parseParamShiftMagnitude(m) !== m) {
            throw new Error(`parseParamShiftMagnitude(${m}) should round-trip`);
        }
    }
    if (parseParamShiftMagnitude("weird") !== "normal") {
        throw new Error("parseParamShiftMagnitude should fall back to normal");
    }
    if (parseParamShiftMagnitude(undefined) !== "normal") {
        throw new Error("parseParamShiftMagnitude(undefined) should be normal");
    }
});
