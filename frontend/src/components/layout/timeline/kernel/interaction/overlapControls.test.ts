import { describe, expect, it } from "vitest";

import { CLIP_BODY_PADDING_Y, CLIP_HEADER_HEIGHT } from "../../constants";
import { computeCrossfadeGripPoint } from "../../crossfadeGrip";
import { hitOverlapControl, type OverlapClip } from "./overlapControls";

/** 单个 clip 的默认值（无淡变）。 */
function clip(overrides: Partial<OverlapClip> & { id: string }): OverlapClip {
    return {
        startSec: 0,
        lengthSec: 1,
        fadeInSec: 0,
        autoFadeInSec: 0,
        fadeInShape: 0,
        fadeInDir: 0,
        fadeOutSec: 0,
        autoFadeOutSec: 0,
        fadeOutShape: 0,
        fadeOutDir: 0,
        ...overrides,
    };
}

/**
 * 标准重叠对：earlier = [1s, 3s)、later = [2s, 4s)。
 *
 * pxPerSec = 100 → earlier [100, 300)、later [200, 400)，重叠区 [200, 300)。
 */
function makePair(
    laterOverrides: Partial<OverlapClip> = {},
    earlierOverrides: Partial<OverlapClip> = {},
): OverlapClip[] {
    return [
        clip({ id: "earlier", startSec: 1, lengthSec: 2, ...earlierOverrides }),
        clip({ id: "later", startSec: 2, lengthSec: 2, ...laterOverrides }),
    ];
}

describe("hitOverlapControl", () => {
    const PX_PER_SEC = 100;
    const ROW_HEIGHT = 80;

    it("无重叠时不命中（首尾紧贴不算重叠）", () => {
        const clips = [
            clip({ id: "a", startSec: 1, lengthSec: 2 }),
            clip({ id: "b", startSec: 3, lengthSec: 2 }),
        ];
        expect(
            hitOverlapControl({
                clips,
                contentX: 300,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toBe(null);
    });

    it("命中重叠区左缘 → 后一个 clip 的左边缘", () => {
        // laterStartPx = 200 → 边缘带 [195, 205]
        expect(
            hitOverlapControl({
                clips: makePair(),
                contentX: 200,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toEqual({ kind: "clip-left-edge", clipId: "later" });
    });

    it("命中重叠区右缘 → 前一个 clip 的右边缘（内核原先完全不可达）", () => {
        // earlierEndPx = 300 → 边缘带 [295, 305]
        expect(
            hitOverlapControl({
                clips: makePair(),
                contentX: 300,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toEqual({ kind: "clip-right-edge", clipId: "earlier" });
    });

    it("重叠区内部、远离两侧边缘 → 不命中控件（交回通用命中）", () => {
        expect(
            hitOverlapControl({
                clips: makePair(),
                contentX: 250,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toBe(null);
    });

    it("边缘带覆盖整行高（header 与 body 都算）", () => {
        for (const localY of [2, 18, 40, 78]) {
            expect(
                hitOverlapControl({
                    clips: makePair(),
                    contentX: 300,
                    localY,
                    pxPerSec: PX_PER_SEC,
                    rowHeight: ROW_HEIGHT,
                }),
            ).toEqual({ kind: "clip-right-edge", clipId: "earlier" });
        }
    });

    it("重叠区内 later 的淡入包络线可命中（归属 later）", () => {
        // later 淡入 0.5s = 50px：包络线横跨 [200, 250]，取中点 225
        const hit = hitOverlapControl({
            clips: makePair({ fadeInSec: 0.5 }),
            contentX: 225,
            localY: 40,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(hit?.clipId).toBe("later");
        expect(hit?.fadeSide).toBe("in");
    });

    it("重叠区内 earlier 的淡出包络线可命中（归属 earlier）", () => {
        // earlier 淡出 0.5s = 50px：包络线横跨 [250, 300]，取中点 275
        const hit = hitOverlapControl({
            clips: makePair({}, { fadeOutSec: 0.5 }),
            contentX: 275,
            localY: 40,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(hit?.clipId).toBe("earlier");
        expect(hit?.fadeSide).toBe("out");
    });

    it("自动交叉淡化（autoFade）覆盖手动淡变长度", () => {
        // 手动 fadeInSec = 0，但 autoFadeInSec = 0.5 → 仍应产生淡入控件
        const hit = hitOverlapControl({
            clips: makePair({ autoFadeInSec: 0.5 }),
            contentX: 225,
            localY: 40,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(hit?.fadeSide).toBe("in");
    });

    it("clip 边缘优先于淡变包络线（与旧实现的层叠顺序一致）", () => {
        // later 淡入覆盖整个重叠区（100px），其包络线会穿过左缘位置
        const hit = hitOverlapControl({
            clips: makePair({ fadeInSec: 1 }),
            contentX: 200,
            localY: 40,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(hit?.kind).toBe("clip-left-edge");
    });

    it("命中交叉点抓手 → 归属后一个 clip，并给出前一个 clip 作为 partner", () => {
        // 两侧各 1s 淡变（100px），重叠区 [200, 300)
        const clips = makePair({ fadeInSec: 1 }, { fadeOutSec: 1 });
        const grip = computeCrossfadeGripPoint({
            earlierEndPx: 300,
            earlierFadePx: 100,
            earlierShape: 0,
            earlierDir: 0,
            laterStartPx: 200,
            laterFadePx: 100,
            laterShape: 0,
            laterDir: 0,
            bodyTop: CLIP_HEADER_HEIGHT,
            bodyHeight: ROW_HEIGHT - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
        });
        expect(grip).not.toBe(null);
        if (grip === null) return;
        expect(
            hitOverlapControl({
                clips,
                contentX: grip.x,
                localY: grip.y,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toEqual({ kind: "crossfade-grip", clipId: "later", partnerClipId: "earlier" });
    });

    it("抓手优先于淡变包络线（同一点上曲线命中块也在，必须判为抓手）", () => {
        const clips = makePair({ fadeInSec: 1 }, { fadeOutSec: 1 });
        const grip = computeCrossfadeGripPoint({
            earlierEndPx: 300,
            earlierFadePx: 100,
            earlierShape: 0,
            earlierDir: 0,
            laterStartPx: 200,
            laterFadePx: 100,
            laterShape: 0,
            laterDir: 0,
            bodyTop: CLIP_HEADER_HEIGHT,
            bodyHeight: ROW_HEIGHT - CLIP_BODY_PADDING_Y - CLIP_HEADER_HEIGHT,
        });
        if (grip === null) throw new Error("未求出交点");
        // 先确认这一点确实也在淡变命中区内（否则这条测试没有意义）
        const withoutGrip = hitOverlapControl({
            clips: makePair({ fadeInSec: 1 }), // 只有淡入 → 无抓手
            contentX: grip.x,
            localY: grip.y,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(withoutGrip?.kind).toBe("fade");
        // 两侧都有淡变时，同一点必须判为抓手
        expect(
            hitOverlapControl({
                clips,
                contentX: grip.x,
                localY: grip.y,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            })?.kind,
        ).toBe("crossfade-grip");
    });

    it("只有一侧有淡变时不产生抓手", () => {
        const clips = makePair({ fadeInSec: 1 });
        // (250, 48) 是两侧都有淡变时的交点位置（对称曲线），这里缺淡出侧，
        // 无法求交点 → 不得判为抓手（该点仍落在淡入包络线上，会退回淡变控件）。
        const hit = hitOverlapControl({
            clips,
            contentX: 250,
            localY: 48,
            pxPerSec: PX_PER_SEC,
            rowHeight: ROW_HEIGHT,
        });
        expect(hit?.kind).not.toBe("crossfade-grip");
    });

    it("三个 clip 相互重叠时，取覆盖该点的那一对（c 的左缘）", () => {
        const clips = [
            clip({ id: "a", startSec: 1, lengthSec: 4 }),
            clip({ id: "b", startSec: 2, lengthSec: 4 }),
            clip({ id: "c", startSec: 3, lengthSec: 4 }),
        ];
        // c 的左缘 = 300 → 命中 c 的左边缘（a/b 的左缘在 100 / 200）
        expect(
            hitOverlapControl({
                clips,
                contentX: 300,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toEqual({ kind: "clip-left-edge", clipId: "c" });
        // 250 处不在任何边缘带上，也没有淡变 → 无控件
        expect(
            hitOverlapControl({
                clips,
                contentX: 250,
                localY: 40,
                pxPerSec: PX_PER_SEC,
                rowHeight: ROW_HEIGHT,
            }),
        ).toBe(null);
    });
});
