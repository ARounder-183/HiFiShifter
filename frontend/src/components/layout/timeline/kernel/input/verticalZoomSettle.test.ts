/**
 * 竖直缩放「在途结算」判定（./verticalZoomSettle）行为自检。
 *
 * 【主要内容】
 * 1. React 行高落地即结算（不等超时）；
 * 2. 行高未落地且未超时 → 保持等待（不结算）；
 * 3. 行高始终未落地时由兜底超时结算（防止镜像写入被永久跳过）；
 * 4. 行高近似相等（浮点）也算落地。
 *
 * 【作用】锁定"何时可以补写轨道头位置"这一时序判据 —— 早写会被旧行高的内容高钳制
 * 并触发回声反灌（竖直抽动），永不写则轨道头不再跟随滚动。二者都是用户可感知的缺陷。
 *
 * 【与其他模块的关系】覆盖 `verticalZoomSettle.ts`；不依赖 DOM / React。
 */

import { describe, expect, it } from "vitest";

import { createVerticalZoomFlight, shouldSettleVerticalZoom } from "./verticalZoomSettle";

describe("createVerticalZoomFlight", () => {
    it("记录请求行高与提交时刻", () => {
        const flight = createVerticalZoomFlight(96, 1000);
        expect(flight.rowHeight).toBe(96);
        expect(flight.startedAt).toBe(1000);
    });
});

describe("shouldSettleVerticalZoom", () => {
    const flight = createVerticalZoomFlight(96, 1000);

    it("行高落地即结算，无需等到超时", () => {
        expect(
            shouldSettleVerticalZoom({
                flight,
                landedRowHeight: 96,
                nowMs: 1010,
                timeoutMs: 250,
            }),
        ).toBe(true);
    });

    it("行高未落地且未超时 → 继续等待", () => {
        expect(
            shouldSettleVerticalZoom({
                flight,
                landedRowHeight: 80,
                nowMs: 1010,
                timeoutMs: 250,
            }),
        ).toBe(false);
    });

    it("行高始终未落地时由兜底超时结算", () => {
        expect(
            shouldSettleVerticalZoom({
                flight,
                landedRowHeight: 80,
                nowMs: 1250,
                timeoutMs: 250,
            }),
        ).toBe(true);
    });

    it("行高近似相等也算落地（浮点容差）", () => {
        expect(
            shouldSettleVerticalZoom({
                flight,
                landedRowHeight: 96 + 1e-9,
                nowMs: 1000,
                timeoutMs: 250,
            }),
        ).toBe(true);
    });
});
