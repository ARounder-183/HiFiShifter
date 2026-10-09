/**
 * 命中几何必须与画布画的是**同一条**曲线。
 *
 * 【为什么值得单测】宿主是 REAPER 7.81+ 时，画布画的是宿主自己的 `(curvature, S)`
 * 两轴曲线，而命中块一度仍按 HiFiShifter 的 `(shape, dir)` 铺 —— 于是用户得**离开
 * 看得见的曲线**去抓，且抓取点会随曲率变化漂移。文件头写着"看到的 = 可点的"，
 * 这条测试就是那句话的判据。
 *
 * 判据取法：命中块中心必须落在 `visualFadeGain`（画布用的求值器）上；拿旧曲线
 * `fadeGainSigned` 当期望值时必须**不**成立 —— 否则这条测试对缺陷不敏感。
 */
import { describe, expect, it } from "vitest";

import type { HostFadeMetadata } from "../../../../../types/api";
import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { buildFadeHitTargets } from "../../fadeHitTargets";
import { visualFadeGain } from "../../hostFadeDisplay";
import { fadeGainSigned } from "../../reaperFade";

const BODY_TOP = CLIP_HEADER_HEIGHT;
const BODY_HEIGHT = 80 - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT;
const FADE_PX = 200;
const CLIP_PX = 400;

/** 宿主"轻微凸"（预设 1 = curvature 0.5）；clip 自身仍是线性（shape 0 / dir 0）。 */
const HOST_CONVEX: HostFadeMetadata = {
    curve_mode: "reaper_new",
    in_curvature: 0.5,
    out_curvature: 0.5,
    in_s: 0,
    out_s: 0,
};

/** 命中块中心点：`t` 归一到该侧淡化区（淡入区在左端、淡出区在右端）。 */
function lineCenters(side: "in" | "out", hostFades?: HostFadeMetadata) {
    const targets = buildFadeHitTargets({
        clipLeftPx: 0,
        clipWidthPx: CLIP_PX,
        bodyTop: BODY_TOP,
        bodyHeight: BODY_HEIGHT,
        fadeInPx: side === "in" ? FADE_PX : 0,
        fadeOutPx: side === "out" ? FADE_PX : 0,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
        hostFades,
    });
    const regionLeftPx = side === "in" ? 0 : CLIP_PX - FADE_PX;
    return targets
        .filter((target) => target.kind === "line" && target.type === `fade_${side}`)
        .map((target) => ({
            t: (target.left + target.width / 2 - regionLeftPx) / FADE_PX,
            y: target.top + target.height / 2,
        }));
}

describe("淡变命中块跟随画布真正画出的曲线", () => {
    it("宿主两轴曲线下：命中块落在宿主曲线上", () => {
        const centers = lineCenters("in", HOST_CONVEX);
        expect(centers.length).toBeGreaterThan(2);
        for (const center of centers) {
            const expectedY =
                BODY_TOP + BODY_HEIGHT * (1 - visualFadeGain(HOST_CONVEX, 0, 0, "in", center.t));
            expect(Math.abs(center.y - expectedY)).toBeLessThan(0.001);
        }
    });

    it("宿主曲线与 HFS 自己的曲线不同：旧口径会明显错位", () => {
        const centers = lineCenters("in", HOST_CONVEX);
        // 取曲线中段（两端点两条曲线都经过，差异在那里最大）。
        const middle = centers[Math.floor(centers.length / 2)];
        const hfsY = BODY_TOP + BODY_HEIGHT * (1 - fadeGainSigned(0, 0, "in", middle.t));
        expect(Math.abs(middle.y - hfsY)).toBeGreaterThan(2);
    });

    it("没有宿主读数（独立 App）时行为不变：仍是 HFS 自己的曲线", () => {
        const centers = lineCenters("in", undefined);
        for (const center of centers) {
            const expectedY = BODY_TOP + BODY_HEIGHT * (1 - fadeGainSigned(0, 0, "in", center.t));
            expect(Math.abs(center.y - expectedY)).toBeLessThan(0.001);
        }
    });

    it("legacy / hifishifter 读数同样退回 HFS 曲线（visualFadeGain 的既有语义）", () => {
        const legacy: HostFadeMetadata = { ...HOST_CONVEX, curve_mode: "legacy" };
        const centers = lineCenters("out", legacy);
        for (const center of centers) {
            const expectedY = BODY_TOP + BODY_HEIGHT * (1 - fadeGainSigned(0, 0, "out", center.t));
            expect(Math.abs(center.y - expectedY)).toBeLessThan(0.001);
        }
    });
});
