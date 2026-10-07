/**
 * "看到的 = 可点的"：渐变角角标 ⊆ 渐变角命中带。
 *
 * 【为什么需要这条契约】渐变角命中区是本仓唯一**完全隐形**的可点区 —— 淡变为 0
 * 时屏幕上没有任何东西提示"这里能按"。Issue 141 的两点抱怨（命中区太大、顶部
 * 看起来能点却不能点）都是这种隐形造成的：用户只能靠试错建立肌肉记忆。
 *
 * 现在渲染端会在该侧**还没有淡变**时于 body 顶角画一个小三角（`drawClipDetails`），
 * 本文件钉住它必须整体落在 `hitTest` 判为 `fade-in/out-corner` 的横帽带内。
 * 两边共用 `constants` 里的同一份几何（`fadeCornerHandleBoxPx` /
 * `fadeCornerReservePx`），所以这条断言一旦变红，说明有人把绘制端与命中端
 * 拆成了两套偏移 —— 正是 `clipHeaderControls` 文件头警告过的漂移。
 *
 * 另一半同样重要：有两类 clip 的横帽带其实点不到（吸附偏移手柄盖住、淡出横帽被
 * 淡入横帽吞掉），那些情形下渲染端**不画**角标（`isFadeCornerBandReachable`）。
 * 本文件把这两条边界也钉住。
 */
import { describe, expect, it } from "vitest";

import {
    CLIP_BODY_PADDING_Y,
    CLIP_HEADER_HEIGHT,
    FADE_CORNER_CAP_WIDTH_PX,
    fadeCornerHandleBoxPx,
    fadeCornerReservePx,
    isFadeCornerBandReachable,
} from "../../constants";
import { hitTest, type HitTestArgs } from "./hitTest";

const PX_PER_SEC = 100;
const CLIP_START_SEC = 1;
const CLIP_LEFT_PX = CLIP_START_SEC * PX_PER_SEC;
const CLIP_WIDTH_PX = 200;

/** 受支持的行高范围（constants 的 MIN/MAX_ROW_HEIGHT）。 */
const SUPPORTED_ROW_HEIGHTS = [80, 96, 120, 192];
/** 退化矮 clip：横帽带被吸附偏移手柄带盖住。 */
const SHADOWED_ROW_HEIGHTS = [24, 20];
/** 窄 clip 的宽度：淡出横帽落在淡入横帽的判定范围内。 */
const NARROW_CLIP_WIDTH_PX = 12;

function geometryFor(rowHeight: number, clipWidthPx: number) {
    const clipHeightPx = rowHeight - CLIP_BODY_PADDING_Y;
    return { clipHeightPx, bodyHeightPx: clipHeightPx - CLIP_HEADER_HEIGHT, clipWidthPx };
}

function reachable(rowHeight: number, clipWidthPx: number, side: "in" | "out"): boolean {
    const { clipHeightPx, bodyHeightPx } = geometryFor(rowHeight, clipWidthPx);
    return isFadeCornerBandReachable({
        side,
        clipWidthPx,
        clipHeightPx,
        headerHeightPx: CLIP_HEADER_HEIGHT,
        bodyHeightPx,
    });
}

function boxFor(rowHeight: number, clipWidthPx: number, side: "in" | "out") {
    return fadeCornerHandleBoxPx({
        side,
        clipWidthPx,
        bodyTopPx: CLIP_HEADER_HEIGHT,
        bodyHeightPx: geometryFor(rowHeight, clipWidthPx).bodyHeightPx,
    });
}

function argsFor(rowHeight: number, clipWidthPx: number): HitTestArgs {
    return {
        contentX: 0,
        contentY: 0,
        pxPerSec: PX_PER_SEC,
        rowHeight,
        tracks: [{ id: "A" }],
        clipsByTrack: new Map([
            [
                "A",
                [
                    {
                        id: "a1",
                        trackId: "A",
                        startSec: CLIP_START_SEC,
                        lengthSec: clipWidthPx / PX_PER_SEC,
                    },
                ],
            ],
        ]),
        headerHeightPx: CLIP_HEADER_HEIGHT,
    };
}

/** 在 clip 内 `(dx, dy)` 处按下，返回命中的分区名（未命中 clip 时返回 kind）。 */
function regionAt(
    rowHeight: number,
    clipWidthPx: number,
    dx: number,
    dy: number,
): string | "empty" {
    const result = hitTest({
        ...argsFor(rowHeight, clipWidthPx),
        contentX: CLIP_LEFT_PX + dx,
        contentY: dy,
    });
    return result.kind === "clip" ? result.region : result.kind;
}

/** 角标矩形内的探针点（内缩 0.5px，避开半开区间边界）。 */
function probesInside(box: { left: number; top: number; width: number; height: number }) {
    const nearLeft = box.left + 0.5;
    const nearRight = box.left + box.width - 0.5;
    const nearTop = box.top + 0.5;
    const nearBottom = box.top + box.height - 0.5;
    return [
        [nearLeft, nearTop],
        [nearRight, nearTop],
        [nearLeft, nearBottom],
        [nearRight, nearBottom],
        [box.left + box.width / 2, box.top + box.height / 2],
    ] as const;
}

describe("fade corner handle affordance", () => {
    for (const rowHeight of SUPPORTED_ROW_HEIGHTS) {
        for (const clipWidthPx of [CLIP_WIDTH_PX, 40, NARROW_CLIP_WIDTH_PX]) {
            for (const side of ["in", "out"] as const) {
                if (!reachable(rowHeight, clipWidthPx, side)) continue;
                it(`rowHeight=${rowHeight} width=${clipWidthPx} side=${side}: 角标整体落在命中带内`, () => {
                    const expected = side === "in" ? "fade-in-corner" : "fade-out-corner";
                    for (const [dx, dy] of probesInside(boxFor(rowHeight, clipWidthPx, side))) {
                        expect(regionAt(rowHeight, clipWidthPx, dx, dy)).toBe(expected);
                    }
                });
            }
        }
    }

    it("角标既不越出横帽宽度，也不越出 body 顶边", () => {
        for (const rowHeight of [...SUPPORTED_ROW_HEIGHTS, ...SHADOWED_ROW_HEIGHTS]) {
            const { bodyHeightPx } = geometryFor(rowHeight, CLIP_WIDTH_PX);
            const reserve = fadeCornerReservePx(bodyHeightPx);
            for (const side of ["in", "out"] as const) {
                const box = boxFor(rowHeight, CLIP_WIDTH_PX, side);
                expect(box.top).toBeGreaterThanOrEqual(CLIP_HEADER_HEIGHT);
                expect(box.top + box.height).toBeLessThanOrEqual(CLIP_HEADER_HEIGHT + reserve);
                expect(box.left).toBeGreaterThanOrEqual(0);
                if (side === "in") {
                    expect(box.left + box.width).toBeLessThanOrEqual(FADE_CORNER_CAP_WIDTH_PX);
                } else {
                    expect(box.left + box.width).toBeLessThanOrEqual(CLIP_WIDTH_PX);
                    expect(CLIP_WIDTH_PX - box.left).toBeLessThanOrEqual(FADE_CORNER_CAP_WIDTH_PX);
                }
            }
        }
    });

    it("受支持的行高下横帽带始终可达（角标会画）", () => {
        for (const rowHeight of SUPPORTED_ROW_HEIGHTS) {
            for (const side of ["in", "out"] as const) {
                expect(reachable(rowHeight, CLIP_WIDTH_PX, side)).toBe(true);
            }
        }
    });

    it("退化矮 clip 不画角标：横帽带已被吸附偏移手柄盖住", () => {
        for (const rowHeight of SHADOWED_ROW_HEIGHTS) {
            for (const side of ["in", "out"] as const) {
                expect(reachable(rowHeight, CLIP_WIDTH_PX, side)).toBe(false);
            }
        }
        // 同一位置上按下的确是吸附偏移手柄 —— 这就是"画了也点不到"的证据。
        const rowHeight = SHADOWED_ROW_HEIGHTS[0];
        const box = boxFor(rowHeight, CLIP_WIDTH_PX, "in");
        expect(
            regionAt(rowHeight, CLIP_WIDTH_PX, box.left + box.width / 2, box.top + box.height / 2),
        ).toBe("snap-offset-handle");
    });

    it("窄 clip 不画淡出角标：淡出横帽被淡入横帽的判定范围吞掉", () => {
        for (const rowHeight of SUPPORTED_ROW_HEIGHTS) {
            // 淡出侧不可达：左右横帽带都固定 22px 宽，而判定顺序让左侧先赢。
            expect(reachable(rowHeight, NARROW_CLIP_WIDTH_PX, "out")).toBe(false);
            // 同一位置按下拿到的是**淡入**角，正是"画淡出角标会骗人"的原因。
            const box = boxFor(rowHeight, NARROW_CLIP_WIDTH_PX, "out");
            expect(
                regionAt(
                    rowHeight,
                    NARROW_CLIP_WIDTH_PX,
                    box.left + box.width / 2,
                    box.top + box.height / 2,
                ),
            ).toBe("fade-in-corner");
            // 淡入侧的角标仍然可达，不受影响。
            expect(reachable(rowHeight, NARROW_CLIP_WIDTH_PX, "in")).toBe(true);
        }
    });
});
