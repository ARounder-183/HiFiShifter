// 宿主淡化示意显示：本应用包络有界/端点正确，保留原App路径，不伪称REAPER公式。
import { expect, test } from "vitest";
import { hostFadeDisplay, hostFadeLabel, visualFadeGain } from "./hostFadeDisplay";
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
/** 只查得到 unknown 标记的极简词表；其余键原样返回，便于断言"没走硬编码"。 */
const label = (key: string) => (key === "fade_info_host_unknown" ? "REAPER curve unknown" : key);
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
    expect(hostFadeLabel(metadata, false, label)).toBe("REAPER c=-0.20 S=0.65");
    expect(hostFadeDisplay(metadata, true)).toBe("hifishifter");
    expect(hostFadeDisplay({ ...metadata, curve_mode: "unknown" }, true)).toBe("hifishifter");
});

test("the unknown-axes marker comes from the catalog instead of a literal", () => {
    const unknown = { ...metadata, curve_mode: "unknown" as const };
    expect(hostFadeLabel(unknown, false, label)).toBe("REAPER curve unknown");
    expect(hostFadeLabel(unknown, false, label)).not.toContain("fade_info_host_unknown");
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

test("a preset reading is named in the label; an off-table reading only reports the axes", () => {
    const lookup = (key: string) =>
        ({
            common_parenthetical: "{value} ({note})",
            fade_shape_fast_start: "Fast Start",
        })[key] ?? key;
    // 预设 1「快起」= `(c 0.5, S 0)`，实测表里的第二行。
    const preset: HostFadeMetadata = {
        curve_mode: "reaper_new",
        in_curvature: 0.5,
        out_curvature: 0.5,
        in_s: 0,
        out_s: 0,
    };
    expect(hostFadeLabel(preset, false, lookup)).toBe("REAPER c=0.50 S=0.00 (Fast Start)");
    // 差一点点就不是预设：只报原始读数，不替用户认领一个形状。
    expect(hostFadeLabel({ ...preset, in_curvature: 0.25 }, false, lookup)).toBe(
        "REAPER c=0.25 S=0.00",
    );
    expect(hostFadeLabel({ ...preset, in_s: 0.0001 }, false, lookup)).toBe("REAPER c=0.50 S=0.00");
});
test("standalone and known legacy hosts retain the existing fade renderer", () => {
    expect(hostFadeDisplay(undefined, false)).toBe("legacy");
    expect(hostFadeDisplay({ ...metadata, curve_mode: "legacy" }, false)).toBe("legacy");
});
