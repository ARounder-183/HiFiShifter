/**
 * 交叉点抓手必须落在**画布画出的**两条曲线交点上。
 *
 * 【为什么值得单测】抓手是"拖动它同时移动双方边缘"的唯一入口。画布在 REAPER 7.81+
 * 上画的是宿主的两轴曲线，而抓手一度按 HiFiShifter 自己的 `(shape, dir)` 求交 ——
 * 于是抓手出现在一个两条**可见**曲线并不相交的位置，用户按它拖动时手感完全不对。
 */
import { describe, expect, it } from "vitest";

import type { HostFadeMetadata } from "../../../types/api";
import { computeCrossfadeGripPoint } from "./crossfadeGrip";
import { visualFadeGain } from "./hostFadeDisplay";
import { fadeGainSigned } from "./reaperFade";

const BODY_TOP = 20;
const BODY_HEIGHT = 60;
const FADE_PX = 120;
/** 前一个 clip 的右缘 = 后一个 clip 的左缘 + 重叠（40px 重叠才有抓手）。 */
const EARLIER_END_PX = 200;
const LATER_START_PX = 160;
const EARLIER_FADE_LEFT_PX = EARLIER_END_PX - FADE_PX;

/** 两侧都是"轻微凸"（预设 1 = curvature 0.5）；clip 自身是线性（shape 0 / dir 0）。 */
const HOST: HostFadeMetadata = {
    curve_mode: "reaper_new",
    in_curvature: 0.5,
    out_curvature: 0.5,
    in_s: 0,
    out_s: 0,
};

function gripPoint(withHost: boolean) {
    return computeCrossfadeGripPoint({
        earlierEndPx: EARLIER_END_PX,
        earlierFadePx: FADE_PX,
        earlierShape: 0,
        earlierDir: 0,
        ...(withHost ? { earlierHostFades: HOST } : {}),
        laterStartPx: LATER_START_PX,
        laterFadePx: FADE_PX,
        laterShape: 0,
        laterDir: 0,
        ...(withHost ? { laterHostFades: HOST } : {}),
        bodyTop: BODY_TOP,
        bodyHeight: BODY_HEIGHT,
    });
}

describe("交叉点抓手跟随画布真正画出的曲线", () => {
    it("宿主两轴曲线下：交点同时落在两条宿主曲线上", () => {
        const point = gripPoint(true);
        expect(point).not.toBeNull();
        const tA = (point!.x - EARLIER_FADE_LEFT_PX) / FADE_PX;
        const tB = (point!.x - LATER_START_PX) / FADE_PX;
        const yA = BODY_TOP + BODY_HEIGHT * (1 - visualFadeGain(HOST, 0, 0, "out", tA));
        const yB = BODY_TOP + BODY_HEIGHT * (1 - visualFadeGain(HOST, 0, 0, "in", tB));
        expect(Math.abs(point!.y - yA)).toBeLessThan(0.01);
        expect(Math.abs(point!.y - yB)).toBeLessThan(0.01);
    });

    it("宿主曲线与 HFS 曲线的交点不是同一个点（旧口径会错位）", () => {
        const withHost = gripPoint(true)!;
        const withoutHost = gripPoint(false)!;
        // 两条可见曲线的交点在 Y 上明显不同于 HFS 曲线的交点 —— 这正是用户
        // "抓手在曲线上方/下方一点"的手感来源。
        expect(Math.abs(withHost.y - withoutHost.y)).toBeGreaterThan(1);
    });

    it("没有宿主读数时行为不变（独立 App）", () => {
        const point = gripPoint(false)!;
        const tA = (point.x - EARLIER_FADE_LEFT_PX) / FADE_PX;
        const yA = BODY_TOP + BODY_HEIGHT * (1 - fadeGainSigned(0, 0, "out", tA));
        expect(Math.abs(point.y - yA)).toBeLessThan(0.01);
    });
});
