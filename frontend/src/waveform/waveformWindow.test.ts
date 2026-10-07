/**
 * 波形几何窗口 / 渲染原点自检。
 *
 * 【核心守护】"波形不随视口宽度（或行位置）改变而整体平移"。
 *
 * 渲染器把屏幕位置算作 `local − snap(origin, dpr)`（`surfaceRenderer.snappedOriginPx`
 * 与本文件使用的 `snapToDevicePx` 是同一个式子），理想位置是 `content − scroll`。
 * 两者之差即残差，必须恒为 0 —— 否则整幅波形相对网格 / Clip / 参数曲线平移。
 *
 * 历史缺陷（用户报告"拖动窗口宽度时波形水平抖动"）：水平原点 = 余量，而余量取
 * 整数 CSS 像素；dpr = 1.25 / 1.5 下 `余量 × dpr` 不是整数，残差在 ±0.5 物理像素
 * 间循环，余量又随视口宽度变化 —— 于是拖动宽度时波形反复跳变。
 *
 * 【与其他模块的关系】仅覆盖 `waveformWindow.ts`；不依赖 DOM / WebGL / React。
 */

import { describe, expect, it } from "vitest";

import { snapToDevicePx } from "../utils/devicePixelLine";
import {
    WAVEFORM_MARGIN_MAX_PX,
    WAVEFORM_MARGIN_MIN_PX,
    computeWaveformWindow,
    type WaveformWindowArgs,
} from "./waveformWindow";

const DPRS = [1, 1.25, 1.5, 1.75, 2, 2.5, 3];

/** 模拟 `rasterize` 的设备像素吸附（画布 CSS 宽 = round(css × dpr) / dpr）。 */
function fitToDevice(cssPx: number, dpr: number): number {
    return Math.max(1, Math.round(cssPx * dpr)) / dpr;
}

function windowFor(overrides: Partial<WaveformWindowArgs> = {}) {
    return computeWaveformWindow({
        scrollLeftPx: 0,
        scrollTopPx: 0,
        widthPx: 1000,
        heightPx: 400,
        dpr: 1,
        horizontalOverscan: true,
        firstRowTopPx: 0,
        geometryBottomPx: 400,
        ...overrides,
    });
}

describe("computeWaveformWindow", () => {
    it("水平：内容点的屏幕位置与视口宽度无关（回归守护）", () => {
        for (const dpr of DPRS) {
            const scrollLeftPx = 137.5;
            const contentX = 600;
            const positions = new Set<number>();
            for (let viewportCss = 600; viewportCss <= 1400; viewportCss += 0.25) {
                const w = windowFor({
                    dpr,
                    scrollLeftPx,
                    widthPx: fitToDevice(viewportCss, dpr),
                });
                // 屏幕位置 = 窗口局部坐标 − 渲染器对原点的吸附值
                const screen = contentX - w.windowStartPx - snapToDevicePx(w.originXPx, dpr);
                positions.add(Math.round(screen * 1e6) / 1e6);
                expect(screen).toBeCloseTo(contentX - scrollLeftPx, 9);
            }
            // 逐值相同：不存在"某些宽度下偏移"的取值
            expect(positions.size).toBe(1);
        }
    });

    it("水平：渲染原点恒为整数个物理像素", () => {
        for (const dpr of DPRS) {
            for (const viewportCss of [640, 800.5, 999, 1000.4, 1200, 1600, 2000]) {
                const w = windowFor({ dpr, widthPx: fitToDevice(viewportCss, dpr) });
                expect(Math.abs(w.originXPx * dpr - Math.round(w.originXPx * dpr))).toBeLessThan(
                    1e-9,
                );
            }
        }
    });

    it("竖直：内容点的屏幕位置与行位置 / 滚动位置无关", () => {
        for (const dpr of DPRS) {
            const contentY = 250;
            for (const scrollTopPx of [0, 37.5, 100.25]) {
                const positions = new Set<number>();
                for (const firstRowTopPx of [0, 1, 37.5, 100.4, 213]) {
                    const w = windowFor({
                        dpr,
                        scrollTopPx,
                        firstRowTopPx,
                        geometryBottomPx: 4000,
                    });
                    const screen = contentY - w.windowTopPx - snapToDevicePx(w.originYPx, dpr);
                    positions.add(Math.round(screen * 1e6) / 1e6);
                    expect(screen).toBeCloseTo(contentY - scrollTopPx, 9);
                }
                expect(positions.size).toBe(1);
            }
        }
    });

    it("竖直：渲染原点恒为整数个物理像素", () => {
        for (const dpr of DPRS) {
            for (const firstRowTopPx of [0, 1, 37.5, 100.4, 213]) {
                const w = windowFor({ dpr, firstRowTopPx, geometryBottomPx: 4000 });
                expect(Math.abs(w.originYPx * dpr - Math.round(w.originYPx * dpr))).toBeLessThan(
                    1e-9,
                );
            }
        }
    });

    it("dpr = 1 时余量与旧公式逐值相同（行为不变）", () => {
        for (const viewportCss of [100, 400, 512, 700.5, 1000, 2048, 3000]) {
            const legacy = Math.min(
                WAVEFORM_MARGIN_MAX_PX,
                Math.max(WAVEFORM_MARGIN_MIN_PX, Math.round(viewportCss * 0.25)),
            );
            const w = windowFor({ dpr: 1, widthPx: viewportCss });
            expect(w.originXPx).toBe(legacy);
        }
    });

    it("余量落在意图区间内（设备像素吸附后误差不超过 1 个物理像素）", () => {
        for (const dpr of DPRS) {
            for (const viewportCss of [1, 400, 700.5, 1000, 2048, 4000]) {
                const w = windowFor({ dpr, widthPx: viewportCss });
                expect(w.originXPx).toBeGreaterThanOrEqual(WAVEFORM_MARGIN_MIN_PX - 1 / dpr);
                expect(w.originXPx).toBeLessThanOrEqual(WAVEFORM_MARGIN_MAX_PX + 1 / dpr);
            }
        }
    });

    it("窗口完整覆盖视口", () => {
        for (const dpr of DPRS) {
            for (const viewportCss of [320, 1000.4, 2000]) {
                const widthPx = fitToDevice(viewportCss, dpr);
                const w = windowFor({ dpr, scrollLeftPx: 250, widthPx });
                expect(w.windowStartPx).toBeLessThanOrEqual(250);
                expect(w.windowEndPx).toBeGreaterThanOrEqual(250 + widthPx);
            }
        }
    });

    it("不留余量（Canvas2D 回退）：窗口与视口重合、原点为 0", () => {
        const w = windowFor({ horizontalOverscan: false, scrollLeftPx: 123, widthPx: 800 });
        expect(w.originXPx).toBe(0);
        expect(w.windowStartPx).toBe(123);
        expect(w.windowEndPx).toBe(923);
    });

    it("无行数据时退回视口高作窗口底端", () => {
        const w = windowFor({
            scrollTopPx: 40,
            heightPx: 300,
            firstRowTopPx: Number.POSITIVE_INFINITY,
            geometryBottomPx: Number.NEGATIVE_INFINITY,
        });
        expect(w.windowBottomPx).toBe(340);
        // 无行 → 竖直原点为 0（窗口起点 = 滚动位置）
        expect(w.originYPx).toBe(0);
        expect(w.windowTopPx).toBe(40);
    });

    it("非法 dpr 回退为 1", () => {
        for (const dpr of [0, -2, Number.NaN, Number.POSITIVE_INFINITY]) {
            const w = windowFor({ dpr, widthPx: 1000 });
            expect(Number.isFinite(w.originXPx)).toBe(true);
            expect(Number.isFinite(w.originYPx)).toBe(true);
            expect(w.originXPx).toBe(250);
        }
    });
});
