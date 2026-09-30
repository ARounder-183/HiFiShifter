import { describe, expect, it } from "vitest";

import { sanitizeVibratoPreset } from "../../../features/vibrato/vibratoPresets";
import { glyphPath } from "./vibratoDialogLogic";

describe("glyphPath（缩略图折线）", () => {
    it("深度 0 的预设是水平中线（直线预设的缩略图就是直线）", () => {
        const preset = sanitizeVibratoPreset({ id: "custom_a", depthCents: 0 });
        const path = glyphPath(preset, 40, 14);
        const ys = path.split(" ").map((pair) => Number(pair.split(",")[1]));
        for (const y of ys) expect(y).toBeCloseTo(7, 1);
    });

    it("正弦预设：首点在中线，上下都触到峰（缩略图画的是真实窗口的 ~2 个周期）", () => {
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            depthCents: 30,
            attackMs: 0,
            releaseMs: 0,
        });
        const path = glyphPath(preset, 40, 14);
        const ys = path.split(" ").map((pair) => Number(pair.split(",")[1]));
        expect(ys.length).toBe(64);
        // 64 点窗口 = 320ms，5.5Hz ≈ 1.8 个周期：起点在中线（sin 从 0 起振），
        // 峰与谷都应出现。
        expect(ys[0]).toBeCloseTo(7, 1);
        expect(Math.min(...ys)).toBeLessThan(2);
        expect(Math.max(...ys)).toBeGreaterThan(12);
    });

    it("按自身峰值定标：5 分与 100 分的缩略图都触到同样的高度（形状可辨优先）", () => {
        const shallow = glyphPath(sanitizeVibratoPreset({ id: "custom_a", depthCents: 5 }), 40, 14);
        const deep = glyphPath(sanitizeVibratoPreset({ id: "custom_b", depthCents: 100 }), 40, 14);
        const peakOf = (path: string) =>
            Math.max(...path.split(" ").map((pair) => Math.abs(Number(pair.split(",")[1]) - 7)));
        expect(peakOf(shallow)).toBeCloseTo(peakOf(deep), 1);
    });

    it("table 波形（手绘 / 提取来源）同样可用", () => {
        const preset = sanitizeVibratoPreset({
            id: "custom_a",
            cycle: {
                kind: "table",
                table: [0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5].concat([
                    0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5,
                ]),
            },
        });
        const path = glyphPath(preset, 40, 14);
        expect(path.split(" ").length).toBe(64);
        for (const pair of path.split(" ")) {
            expect(Number.isFinite(Number(pair.split(",")[1]))).toBe(true);
        }
    });
});
