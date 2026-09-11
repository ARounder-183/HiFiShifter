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
    const line = targets.find((target) => target.kind === "line" && target.type === `fade_${side}`);
    if (line === undefined) throw new Error(`未生成 ${side} 侧包络线命中块`);
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

    it("命中淡入包络线 → 返回 in", () => {
        const clip: FadeTargetClip = { fadeInSec: 1 };
        const point = firstLineCenter(clip, "in");
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })).toBe("in");
    });

    it("命中淡出包络线 → 返回 out", () => {
        const clip: FadeTargetClip = { fadeOutSec: 1 };
        const point = firstLineCenter(clip, "out");
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })).toBe(
            "out",
        );
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
        expect(hitClipFadeTarget({ ...base, clip, contentX: point.x, localY: point.y })).toBe("in");
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
