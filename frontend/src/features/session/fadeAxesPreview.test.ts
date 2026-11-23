import { expect, test } from "vitest";

import type { HostFadeMetadata } from "../../types/api";
import { applyHostFadeAxesPreview, fadeRestorePatch, mirrorHostFadeAxes } from "./fadeAxesPreview";

const continuous: HostFadeMetadata = {
    curve_mode: "continuous",
    in_curvature: 0,
    out_curvature: 0,
    in_s: 0,
    out_s: 0,
};

test("shape 0..6 映射为实测表里的 (curvature, S)", () => {
    const expected: Array<[number, number, number]> = [
        [0, 0, 0],
        [1, 0.5, 0],
        [2, -0.5, 0],
        [3, 1, 0],
        [4, -1, 0],
        [5, 0, 0.5],
        [6, 0, 1],
    ];
    for (const [shape, curvature, s] of expected) {
        const result = applyHostFadeAxesPreview(continuous, { fadeInShape: shape });
        expect(result.hostFades?.in_curvature, "shape " + shape + " curvature").toBe(curvature);
        expect(result.hostFades?.in_s, "shape " + shape + " S").toBe(s);
        expect(result.correctedPatch.fadeInDir).toBe(curvature);
        expect(result.correctedPatch.fadeInS).toBe(s);
    }
});

test("显式 S 覆盖预设 S，显式 dir 在 shape 存在时被忽略（与后端口径一致）", () => {
    const result = applyHostFadeAxesPreview(continuous, {
        fadeInShape: 5,
        fadeInDir: -0.7,
        fadeInS: 0.25,
    });
    expect(result.hostFades?.in_curvature).toBe(0);
    expect(result.hostFades?.in_s).toBe(0.25);
    expect(result.correctedPatch.fadeInDir).toBe(0);
    expect(result.correctedPatch.fadeInS).toBe(0.25);
});

test("没有 shape 时 dir/S 各自覆盖，未给出的轴保持原值", () => {
    const withCurve = applyHostFadeAxesPreview(continuous, { fadeInDir: -0.3 });
    expect(withCurve.hostFades?.in_curvature).toBe(-0.3);
    expect(withCurve.hostFades?.in_s).toBe(0);
    const withS = applyHostFadeAxesPreview(withCurve.hostFades, { fadeInS: 0.9 });
    expect(withS.hostFades?.in_curvature).toBe(-0.3);
    expect(withS.hostFades?.in_s).toBe(0.9);
});

test("legacy / 非 continuous / undefined 原样返回，不写 S", () => {
    const legacy: HostFadeMetadata = { ...continuous, curve_mode: "legacy" };
    expect(mirrorHostFadeAxes(legacy, { fadeInShape: 6 })).toBe(legacy);
    expect(mirrorHostFadeAxes(undefined, { fadeInDir: 0.5 })).toBeUndefined();
    const hifishifter: HostFadeMetadata = { ...continuous, curve_mode: "hifishifter" };
    const unknown: HostFadeMetadata = { ...continuous, curve_mode: "unknown" };
    expect(mirrorHostFadeAxes(hifishifter, { fadeInDir: 0.5 })).toBe(hifishifter);
    expect(mirrorHostFadeAxes(unknown, { fadeInDir: 0.5 })).toBe(unknown);
});

test("仅长度 patch 返回同一引用；非法形状不制造后端不会写的组合", () => {
    expect(mirrorHostFadeAxes(continuous, { fadeInSec: 1 })).toBe(continuous);
    const fractional = applyHostFadeAxesPreview(continuous, {
        fadeInShape: 5.1,
        fadeInDir: -0.4,
        fadeInS: 0.4,
    });
    expect(fractional.hostFades).toBe(continuous);
    expect(fractional.correctedPatch).toEqual({});
});

test("取消还原 patch：continuous 带 S，legacy 不带", () => {
    const base = {
        fadeInShape: 2,
        fadeOutShape: 6,
        fadeInDir: -0.2,
        fadeOutDir: 0.1,
        fadeInS: 0.3,
        fadeOutS: 0.4,
    };
    expect(fadeRestorePatch({ base, continuous: true })).toEqual(base);
    expect(fadeRestorePatch({ base, continuous: false })).toEqual({
        fadeInShape: 2,
        fadeOutShape: 6,
        fadeInDir: -0.2,
        fadeOutDir: 0.1,
    });
});
