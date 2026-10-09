// 宿主淡化示意显示：本应用包络有界/端点正确，保留原App路径，不伪称REAPER公式。
import { expect, test } from "vitest";
import {
    hostFadeDisplay,
    hostFadeDisplayShape,
    hostFadeGainForAxes,
    visualFadeGain,
} from "./hostFadeDisplay";
import { defaultFadeDirFor, fadeGainSigned } from "./reaperFade";
import { HOST_FADE_PRESET_AXES } from "./hostFadeAxes";
import type { HostFadeMetadata } from "../../../types/api";

const metadata: HostFadeMetadata = {
    curve_mode: "reaper_new",
    in_curvature: -0.2,
    out_curvature: 0,
    in_s: 0.65,
    out_s: 0,
};
test("HFS-owned envelope displays the same shape/curvature family used by audio, not host c/S", () => {
    const owned = { ...metadata, curve_mode: "hifishifter" as const };
    for (const mode of ["in", "out"] as const)
        for (const t of [0, 0.15, 0.5, 0.85, 1]) {
            expect(visualFadeGain(owned, 5, 0.3, mode, t)).toBe(fadeGainSigned(5, 0.3, mode, t));
        }
    expect(visualFadeGain(owned, 1, 0, "in", 0.5)).not.toBe(visualFadeGain(owned, 5, 0, "in", 0.5));
});
test("new axes preserve both values and use an explicit HFS visual style", () => {
    expect(hostFadeDisplay(metadata, false)).toBe("hifishifter");
    expect(hostFadeDisplay(metadata, true)).toBe("hifishifter");
    expect(hostFadeDisplay({ ...metadata, curve_mode: "unknown" }, true)).toBe("hifishifter");
});

/**
 * 显示形状必须**跟着画布画的那一族曲线**走：落在实测表上就是那个预设，表外按渲染器
 * 实际用的权重报占优的一族（`hostFadeGainForAxes` 的 |S| 混合），绝不假称某个具体预设。
 */
test("the displayed shape follows the family the canvas actually draws", () => {
    // 落在实测表上：逐行验证七个预设都能被认出来。
    for (let shape = 0; shape < HOST_FADE_PRESET_AXES.length; shape += 1) {
        const [curvature, s] = HOST_FADE_PRESET_AXES[shape];
        const axes = { ...metadata, in_curvature: curvature, in_s: s };
        expect(hostFadeDisplayShape(axes, false, -1)).toBe(shape);
    }
    // 表外：|S| 过半 → S 族（渲染器的 sigmoid 分量用的就是 5 号）；否则线性。
    expect(hostFadeDisplayShape({ ...metadata, in_curvature: 0.25, in_s: 0.6 }, false, -1)).toBe(5);
    expect(hostFadeDisplayShape({ ...metadata, in_curvature: 0.25, in_s: 0.1 }, false, -1)).toBe(0);
    // 独立 App / 旧轴宿主：原样用它自己的形状（越界才归零）。
    expect(hostFadeDisplayShape(undefined, false, 3)).toBe(3);
    expect(hostFadeDisplayShape({ ...metadata, curve_mode: "legacy" }, false, 6)).toBe(6);
    expect(hostFadeDisplayShape({ ...metadata, curve_mode: "unknown" }, false, -1)).toBe(0);
});

/** 拖拽投影与画布必须共用同一个求值器 —— 曲率由调用方给出，其余轴照旧。 */
test("the shared evaluator agrees with the canvas at the current axes", () => {
    for (const mode of ["in", "out"] as const)
        for (const t of [0, 0.2, 0.5, 0.9, 1]) {
            const curvature = mode === "out" ? metadata.out_curvature : metadata.in_curvature;
            const s = mode === "out" ? metadata.out_s : metadata.in_s;
            expect(hostFadeGainForAxes(mode, curvature, s, t)).toBe(
                visualFadeGain(metadata, 0, 0, mode, t),
            );
        }
});

test("HFS visuals are bounded monotone with exact in/out endpoints for both axes", () => {
    for (const c of [-1, -0.35, 0, 0.7, 1])
        for (const s of [-1, -0.25, 0, 0.5, 1])
            for (const mode of ["in", "out"] as const) {
                const axes = { ...metadata, in_curvature: c, out_curvature: c, in_s: s, out_s: s };
                const values = Array.from({ length: 201 }, (_, i) =>
                    visualFadeGain(axes, 6, 0.8, mode, i / 200),
                );
                expect(values[0]).toBe(mode === "in" ? 0 : 1);
                expect(values[200]).toBe(mode === "in" ? 1 : 0);
                values.forEach((v, i) => {
                    expect(v).toBeGreaterThanOrEqual(0);
                    expect(v).toBeLessThanOrEqual(1);
                    if (i)
                        expect(
                            mode === "in" ? v - values[i - 1] : values[i - 1] - v,
                        ).toBeGreaterThanOrEqual(-1e-12);
                });
            }
});
test("changing either axis changes visual curves and App/legacy remain exactly unchanged", () => {
    const a = visualFadeGain(metadata, 0, 0, "in", 0.3);
    expect(visualFadeGain({ ...metadata, in_curvature: 0.7 }, 0, 0, "in", 0.3)).not.toBe(a);
    expect(visualFadeGain({ ...metadata, in_s: -0.65 }, 0, 0, "in", 0.3)).not.toBe(a);
    for (const mode of ["in", "out"] as const)
        for (const t of [0, 0.025, 0.3, 0.85, 1]) {
            expect(visualFadeGain(undefined, 5, -0.3, mode, t)).toBe(
                fadeGainSigned(5, -0.3, mode, t),
            );
            expect(visualFadeGain({ ...metadata, curve_mode: "legacy" }, 5, -0.3, mode, t)).toBe(
                fadeGainSigned(5, -0.3, mode, t),
            );
        }
    // `(0, 0)` 是实测表里的**预设 0（线性）**，不是"宿主默认曲线"：REAPER 新建 item 的
    // 默认读数是 `SHAPE=1` / `c=0.5`（`probe/ara/captures/fade-axis-7.82.json` 的 baseline）。
    // 坐标落在表内时画的是那个预设自己的曲线，而线性预设就是恒等。
    expect(visualFadeGain({ ...metadata, in_curvature: 0, in_s: 0 }, 0, 0, "in", 0.3)).toBeCloseTo(
        0.3,
    );
    expect(
        Number.isFinite(
            visualFadeGain({ ...metadata, in_curvature: NaN, in_s: Infinity }, 0, 0, "in", NaN),
        ),
    ).toBe(true);
});
test("off-table host axes stay bounded curved visuals, and unknown hosts keep the schematic", () => {
    // 表外的坐标（用户拖过曲率滑杆）没有可命名的形状，沿用原来的有界混合示意。
    const offTable: HostFadeMetadata = {
        curve_mode: "reaper_new",
        in_curvature: 0.25,
        out_curvature: 0.25,
        in_s: 0.25,
        out_s: 0.25,
    };
    const before = JSON.stringify(offTable);
    for (const mode of ["in", "out"] as const) {
        const values = Array.from({ length: 101 }, (_, i) =>
            visualFadeGain(offTable, 0, 0, mode, i / 100),
        );
        expect(values[0]).toBe(mode === "in" ? 0 : 1);
        expect(values[100]).toBe(mode === "in" ? 1 : 0);
        values.forEach((v, i) => {
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThanOrEqual(1);
            if (i)
                expect(
                    mode === "in" ? v - values[i - 1] : values[i - 1] - v,
                ).toBeGreaterThanOrEqual(-1e-12);
        });
    }
    // `unknown` 连轴语义都不知道，只能给"曲线示意"，不能宣称任何形状。
    const unknown: HostFadeMetadata = { ...offTable, curve_mode: "unknown" };
    for (const mode of ["in", "out"] as const) {
        const values = Array.from({ length: 101 }, (_, i) =>
            visualFadeGain(unknown, 0, 0, mode, i / 100),
        );
        expect(values[0]).toBe(mode === "in" ? 0 : 1);
        expect(values[100]).toBe(mode === "in" ? 1 : 0);
        expect(values[50]).toBeCloseTo(Math.SQRT1_2);
    }
    expect(JSON.stringify(offTable)).toBe(before);
});

test("axes landing exactly on a measured preset draw that preset's own curve", () => {
    for (let shape = 0; shape < HOST_FADE_PRESET_AXES.length; shape += 1) {
        const [curvature, s] = HOST_FADE_PRESET_AXES[shape];
        const axes: HostFadeMetadata = {
            curve_mode: "reaper_new",
            in_curvature: curvature,
            out_curvature: curvature,
            in_s: s,
            out_s: s,
        };
        for (const mode of ["in", "out"] as const)
            for (const t of [0, 0.25, 0.5, 0.75, 1])
                expect(visualFadeGain(axes, 0, 0, mode, t)).toBe(
                    fadeGainSigned(shape, defaultFadeDirFor(shape, mode === "out"), mode, t),
                );
    }
});

test("standalone and known legacy hosts retain the existing fade renderer", () => {
    expect(hostFadeDisplay(undefined, false)).toBe("legacy");
    expect(hostFadeDisplay({ ...metadata, curve_mode: "legacy" }, false)).toBe("legacy");
});
