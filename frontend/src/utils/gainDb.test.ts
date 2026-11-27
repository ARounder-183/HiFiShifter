import { test } from "vitest";

import {
    GAIN_DISPLAY_FLOOR_DB,
    GAIN_FLOOR,
    GAIN_MAX_DB,
    GAIN_MAX_LINEAR,
    GAIN_TAPER_FLOOR_DB,
    clampGain,
    clampTakeGain,
    dbToGain,
    formatGainDbLabel,
    formatGainDbValue,
    gainAt,
    gainKnobAngleDeg,
    gainTaper,
    gainToDb,
    parseGainDbInput,
    positionOf,
    resolveGainDrag,
} from "./gainDb.ts";

function near(actual: number, expected: number, label: string, eps = 1e-6): void {
    if (!Number.isFinite(actual) || Math.abs(actual - expected) > eps) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

test("gain domain mirrors the kernel and keeps silence unique", () => {
    near(GAIN_MAX_DB, 24.0824, "upper bound is REAPER item limit", 1e-3);
    near(GAIN_FLOOR, 10 ** (-150 / 20), "display floor linear value");
    if (clampGain(0) !== 0) throw new Error("zero must stay silence");
    if (clampGain(-4) !== 0) throw new Error("negative item gain is silence");
    if (clampGain(GAIN_FLOOR / 2) !== 0) throw new Error("sub-floor canonicalises");
    if (clampGain(1e9) !== GAIN_MAX_LINEAR) throw new Error("upper clamp");
    if (clampGain(Number.NaN) !== 1) throw new Error("NaN is unity");
    if (clampTakeGain(-0.5) !== -0.5) throw new Error("take polarity survives");
    if (clampTakeGain(-99) !== -16) throw new Error("take magnitude clamps");
    if (clampTakeGain(Number.NaN) !== 1) throw new Error("take NaN is unity");
    if (dbToGain(GAIN_DISPLAY_FLOOR_DB) <= 0) throw new Error("-150 dB is representable");
    if (dbToGain(GAIN_DISPLAY_FLOOR_DB - 1) !== 0) throw new Error("below floor is silence");
    if (gainToDb(0) !== Number.NEGATIVE_INFINITY) throw new Error("silence is not -150");
});

test("taper endpoints and roundtrip are exact", () => {
    if (gainAt(0) !== 0) throw new Error("position 0 is silence");
    near(gainAt(1), GAIN_FLOOR, "position 1 is the display floor", 1e-12);
    near(gainAt(10), dbToGain(GAIN_TAPER_FLOOR_DB), "position 10 is the taper floor");
    near(gainAt(58), 1, "position 58 is unity");
    near(gainAt(106), GAIN_MAX_LINEAR, "position 106 is the top");
    near(positionOf(GAIN_FLOOR), 1, "floor maps back to 1px");
    near(positionOf(10 ** (-60 / 20)), 10, "taper floor maps back to 10px");
    near(positionOf(1), 58, "unity maps to 58px");
    near(positionOf(GAIN_MAX_LINEAR), 106, "top maps to 106px");
    if (positionOf(0) !== 0) throw new Error("silence maps to 0px");
    near(positionOf(Number.NaN), 58, "NaN position uses unity");
    if (gainAt(Number.NaN) !== 0) throw new Error("NaN position is silence");
    for (const gain of [GAIN_FLOOR, 1e-4, 0.1, 1, 4, 16]) {
        near(gainAt(positionOf(gain)), gain, `roundtrip ${gain}`, gain * 1e-5);
    }
});

test("taper is monotonic and keeps the 48/48 handfeel", () => {
    let previous = -1;
    for (let px = 0; px <= 106; px += 0.25) {
        const value = gainAt(px);
        if (value + 1e-12 < previous) throw new Error(`not monotonic at ${px}`);
        previous = value;
    }
    near(gainAt(58) / gainAt(58 - 24), 10 ** (30 / 20), "lower half is 1.25 dB/px", 1e-3);
    near(gainAt(58 + 24) / 1, 10 ** (12.0412 / 20), "upper half is 0.5 dB/px", 1e-3);
    if (gainTaper.zeroPx !== 58 || gainTaper.topPx !== 106 || gainTaper.taperPx !== 10) {
        throw new Error("taper geometry drifted");
    }
});

test("knob angle is continuous and negative values are not pinned", () => {
    near(gainKnobAngleDeg(1), 0, "0 dB is centred");
    near(gainKnobAngleDeg(16), 135, "+24.08 is +135");
    near(gainKnobAngleDeg(dbToGain(-6)), -11.7, "-6 dB is -11.7", 1e-3);
    near(gainKnobAngleDeg(dbToGain(-30)), -58.5, "-30 dB is -58.5", 1e-3);
    near(gainKnobAngleDeg(dbToGain(-60)), -117, "-60 dB is -117", 1e-3);
    near(gainKnobAngleDeg(0), -135, "silence is -135");
    near(gainKnobAngleDeg(GAIN_FLOOR), -135, "-150 dB is -135", 1e-3);
});

test("format and parse share one boundary", () => {
    if (formatGainDbLabel(0) !== "-inf dB") throw new Error("silence label");
    if (formatGainDbLabel(1) !== "0 dB") throw new Error("unity label");
    if (formatGainDbLabel(2) !== "+6.0 dB") throw new Error("positive label");
    if (formatGainDbLabel(GAIN_FLOOR) !== "-150 dB") throw new Error("floor label");
    if (formatGainDbValue(-150.1) !== "-inf") throw new Error("below floor value");
    if (parseGainDbInput("+3") !== 3) throw new Error("parse +3");
    if (parseGainDbInput("3 dB") !== 3) throw new Error("parse unit suffix");
    if (Math.abs((parseGainDbInput("-6,5") ?? 0) + 6.5) > 1e-9) throw new Error("parse comma");
    if (parseGainDbInput("-inf") !== Number.NEGATIVE_INFINITY) throw new Error("parse -inf");
    if (parseGainDbInput("inf") !== null) throw new Error("positive infinity rejected");
    if (parseGainDbInput("loud") !== null) throw new Error("garbage rejected");
});

test("resolveGainDrag pins the anchor to the pointer and shifts others by dB", () => {
    const single = resolveGainDrag({
        anchorBaseLinear: 1,
        deltaPx: 10,
        fine: false,
        participants: [{ clipId: "a", baseLinear: 1 }],
        anchorId: "a",
    });
    near(single.anchorLinear, gainAt(68), "anchor follows pointer position");
    if (single.others.length !== 0) throw new Error("anchor excluded from others");
    const multi = resolveGainDrag({
        anchorBaseLinear: 0.5,
        deltaPx: 0,
        fine: false,
        participants: [
            { clipId: "a", baseLinear: 0.5 },
            { clipId: "b", baseLinear: 0.25 },
            { clipId: "c", baseLinear: 0 },
        ],
        anchorId: "a",
    });
    near(multi.anchorLinear, 0.5, "zero delta keeps anchor");
    near(multi.others[0].linear, 0.25, "relative difference preserved");
    if (multi.others[1].linear !== 0) throw new Error("silent member stays silent");
    const fromSilence = resolveGainDrag({
        anchorBaseLinear: 0,
        deltaPx: 1,
        fine: false,
        participants: [{ clipId: "a", baseLinear: 0 }],
        anchorId: "a",
    });
    if (!(fromSilence.anchorLinear > 0)) throw new Error("silence can be pulled up");
    const fine = resolveGainDrag({
        anchorBaseLinear: 1,
        deltaPx: 10,
        fine: true,
        participants: [{ clipId: "a", baseLinear: 1 }],
        anchorId: "a",
    });
    near(fine.anchorLinear, gainAt(60.5), "fine drag is quarter speed");
});
