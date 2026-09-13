import { describe, expect, it } from "vitest";

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { buildFadeHitTargets } from "../../fadeHitTargets";
import { effectiveFadeSec, hitClipFadeTarget, type FadeTargetClip } from "./fadeTargets";

const PX_PER_SEC = 100;
const ROW_HEIGHT = 80;
const BODY_TOP = CLIP_HEADER_HEIGHT;
const BODY_HEIGHT = ROW_HEIGHT - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT;

/** 取某侧包络线上第一个命中块的中心点（避免在测试里重复实现曲线数学）。 */
function firstLineCenter(clip: FadeTargetClip, side: "in" | "out", clipLeftPx = 0, widthPx = 400) {
    const targets = buildFadeHitTargets({
        clipLeftPx,
        clipWidthPx: widthPx,
        bodyTop: BODY_TOP,
        bodyHeight: BODY_HEIGHT,
        fadeInPx:
            side === "in" ? effectiveFadeSec(clip.fadeInSec, clip.autoFadeInSec) * PX_PER_SEC : 0,
        fadeOutPx:
            side === "out"
                ? effectiveFadeSec(clip.fadeOutSec, clip.autoFadeOutSec) * PX_PER_SEC
                : 0,
        fadeInShape: clip.fadeInShape ?? 0,
        fadeInDir: clip.fadeInDir ?? 0,
        fadeOutShape: clip.fadeOutShape ?? 0,
        fadeOutDir: clip.fadeOutDir ?? 0,
    });
    // 刻意取**离区域边缘竖线最远**的那一端采样点：边缘竖线在层叠上高于包络线，
    // 落在两者重叠处的点会（正确地）判为 `edge`，而这里要验的是"本体命中"。
    // 淡入的边缘竖线在右侧 → 取最左；淡出的在左侧 → 取最右。
    const lines = targets.filter(
        (target) => target.kind === "line" && target.type === `fade_${side}`,
    );
    if (lines.length === 0) throw new Error(`未生成 ${side} 侧包络线命中块`);
    const line = side === "in" ? lines[0] : lines[lines.length - 1];
    return { x: line.left + line.width / 2, y: line.top + line.height / 2 };
}

describe("effectiveFadeSec", () => {
    it("自动交叉淡化（> 0）覆盖手动值", () => {
        expect(effectiveFadeSec(0.2, 0.8)).toBe(0.8);
    });

    it("自动值为 0 / 缺省时回落到手动值", () => {
        expect(effectiveFadeSec(0.2, 0)).toBe(0.2);
        expect(effectiveFadeSec(0.2, undefined)).toBe(0.2);
    });

    it("两者都无效时为 0（无淡变）", () => {
        expect(effectiveFadeSec(undefined, undefined)).toBe(0);
        expect(effectiveFadeSec(0, 0)).toBe(0);
        expect(effectiveFadeSec(Number.NaN, Number.NaN)).toBe(0);
    });
});

describe("hitClipFadeTarget", () => {
    const base = {
        clipLeftPx: 0,
        clipWidthPx: 400,
        pxPerSec: PX_PER_SEC,
        rowHeight: ROW_HEIGHT,
    };

    it("无淡变时不命中", () => {
        expect(hitClipFadeTarget({ ...base, clip: {}, contentX: 50, localY: 40 })).toBe(null);
    });

    it("命中淡入包络线 → 返回 in，且标记为包络线本体", () => {
        const clip: FadeTargetClip = { fadeInSec: 1 };
        const point = firstLineCenter(clip, "in");
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })).toEqual({
            side: "in",
            kind: "line",
        });
    });

    it("命中淡出包络线 → 返回 out，且标记为包络线本体", () => {
        const clip: FadeTargetClip = { fadeOutSec: 1 };
        const point = firstLineCenter(clip, "out");
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })).toEqual({
            side: "out",
            kind: "line",
        });
    });

    it("命中区域边缘竖线 → kind 为 edge（双击重置曲率只对本体生效）", () => {
        const clip: FadeTargetClip = { fadeInSec: 1 };
        const targets = buildFadeHitTargets({
            clipLeftPx: 0,
            clipWidthPx: 400,
            bodyTop: BODY_TOP,
            bodyHeight: BODY_HEIGHT,
            fadeInPx: 100,
            fadeOutPx: 0,
            fadeInShape: 0,
            fadeInDir: 0,
            fadeOutShape: 0,
            fadeOutDir: 0,
        });
        const edge = targets.find((target) => target.kind === "edge");
        if (edge === undefined) throw new Error("未生成边缘竖线命中条");
        expect(
            hitClipFadeTarget({
                ...base,
                clip,
                contentX: edge.left + edge.width / 2,
                localY: edge.top + edge.height / 2,
            })?.kind,
        ).toBe("edge");
    });

    it("远离包络线时不命中（body 中间）", () => {
        // 淡入只占左侧 100px；x = 300 处没有淡变控件
        expect(
            hitClipFadeTarget({ ...base, clip: { fadeInSec: 1 }, contentX: 300, localY: 40 }),
        ).toBe(null);
    });

    it("自动交叉淡化覆盖手动值：手动为 0 时仍可命中", () => {
        const clip: FadeTargetClip = { fadeInSec: 0, autoFadeInSec: 1 };
        const point = firstLineCenter(clip, "in");
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })?.side).toBe(
            "in",
        );
    });

    it("clipXFrom / clipXTo 裁剪生效（重叠区解析依赖它）", () => {
        const clip: FadeTargetClip = { fadeInSec: 1 };
        const point = firstLineCenter(clip, "in");
        // 把范围裁到包络线右侧之外 → 不再命中
        expect(
            hitClipFadeTarget({
                ...base,
                clip,
                contentX: point.x,
                localY: point.y,
                clipXFrom: point.x + 20,
                clipXTo: point.x + 100,
            }),
        ).toBe(null);
    });
});
