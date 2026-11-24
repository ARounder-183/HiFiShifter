// 触摸意图判定表：两块画布共用的**唯一一份**判定，逐行钉死。
import { describe, expect, it } from "vitest";
import {
    isEditableHit,
    resolveTouchIntent,
    shouldCancelEditForPinch,
    type TouchHitKind,
} from "./touchIntent";

const HITS: TouchHitKind[] = [
    "clip-body",
    "clip-edge",
    "fade-handle",
    "curve",
    "param-block",
    "selection-edge",
    "ruler",
    "empty",
];

describe("resolveTouchIntent", () => {
    it("★ 双指永远是 pinch，无论命中什么（包括落在 Clip 上）", () => {
        for (const hit of HITS) {
            expect(
                resolveTouchIntent({ hit, pointerCount: 2, tool: "select", gestureEnabled: true }),
            ).toBe("pinch");
            expect(
                resolveTouchIntent({ hit, pointerCount: 3, tool: "draw", gestureEnabled: true }),
            ).toBe("pinch");
        }
    });

    it("★ 单指落在空白处 = 平移（这是手机上唯一的滚动方式）", () => {
        for (const tool of ["select", "draw", "line", "vibrato", "razor"] as const) {
            expect(
                resolveTouchIntent({ hit: "empty", pointerCount: 1, tool, gestureEnabled: true }),
            ).toBe("pan");
        }
    });

    it("标尺永远归播放头，不参与平移", () => {
        expect(
            resolveTouchIntent({
                hit: "ruler",
                pointerCount: 1,
                tool: "select",
                gestureEnabled: true,
            }),
        ).toBe("edit");
    });

    it("可编辑对象一律 edit（不管哪个工具）", () => {
        for (const hit of HITS.filter((h) => h !== "empty" && h !== "ruler")) {
            for (const tool of ["select", "draw", "line", "vibrato", "razor"] as const) {
                expect(
                    resolveTouchIntent({ hit, pointerCount: 1, tool, gestureEnabled: true }),
                ).toBe("edit");
            }
        }
    });

    it("★ 手势层关闭时恒 edit —— 桌面行为逐字节不变", () => {
        for (const hit of HITS) {
            for (const pointerCount of [1, 2, 3]) {
                expect(
                    resolveTouchIntent({
                        hit,
                        pointerCount,
                        tool: "select",
                        gestureEnabled: false,
                    }),
                ).toBe("edit");
            }
        }
    });
});

describe("isEditableHit", () => {
    it("只有 empty 不是可编辑目标", () => {
        expect(isEditableHit("empty")).toBe(false);
        for (const hit of HITS.filter((h) => h !== "empty")) expect(isEditableHit(hit)).toBe(true);
    });
});

describe("shouldCancelEditForPinch", () => {
    it("第二根手指落下必须取消正在进行的编辑（否则会一边拖一边缩、松手还提交）", () => {
        expect(
            shouldCancelEditForPinch({
                hit: "clip-body",
                pointerCount: 2,
                tool: "select",
                gestureEnabled: true,
            }),
        ).toBe(true);
        expect(
            shouldCancelEditForPinch({
                hit: "clip-body",
                pointerCount: 1,
                tool: "select",
                gestureEnabled: true,
            }),
        ).toBe(false);
        expect(
            shouldCancelEditForPinch({
                hit: "clip-body",
                pointerCount: 2,
                tool: "select",
                gestureEnabled: false,
            }),
        ).toBe(false);
    });
});
