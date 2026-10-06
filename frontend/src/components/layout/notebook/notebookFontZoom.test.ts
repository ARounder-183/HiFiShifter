/*
 * 滚轮缩放字号的边界测试。
 *
 * 【要钉死什么】
 *   1. 方向换算：内核的 `-1 = 放大` 必须对应"字号变大"，别接反；
 *   2. 到界与"本次不缩放"（`direction === 0`）都返回 `null` —— 调用方靠它决定
 *      不写设置、不落盘；
 *   3. 界内每一档都能走到，且每步都停在整数上（不产生 13.000000000000002）。
 */

import { test } from "vitest";

import { nextFontSizeForZoomStep } from "./notebookFontZoom.ts";
import { NOTEBOOK_FONT_SIZE_MAX, NOTEBOOK_FONT_SIZE_MIN } from "./notebookSettings.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

test("components/layout/notebook/notebookFontZoom.test.ts scripted checks", () => {
    // 方向：内核 -1 = 放大（滚轮向上），1 = 缩小。
    assertEqual(nextFontSizeForZoomStep(13, -1), 14, "zoom in makes the font bigger");
    assertEqual(nextFontSizeForZoomStep(13, 1), 12, "zoom out makes the font smaller");
    assertEqual(nextFontSizeForZoomStep(13, 0), null, "a dead-zone event changes nothing");

    // 到界返回 null：调用方据此不写设置、不落盘。
    assertEqual(nextFontSizeForZoomStep(NOTEBOOK_FONT_SIZE_MAX, -1), null, "clamped at max");
    assertEqual(nextFontSizeForZoomStep(NOTEBOOK_FONT_SIZE_MIN, 1), null, "clamped at min");

    // 界内每一档都能走到，且每一步都是整数。
    let size = NOTEBOOK_FONT_SIZE_MIN;
    const walked: number[] = [size];
    for (;;) {
        const next = nextFontSizeForZoomStep(size, -1);
        if (next === null) break;
        size = next;
        walked.push(size);
    }
    assertEqual(size, NOTEBOOK_FONT_SIZE_MAX, "walking up stops at max");
    assertEqual(
        walked,
        Array.from(
            { length: NOTEBOOK_FONT_SIZE_MAX - NOTEBOOK_FONT_SIZE_MIN + 1 },
            (_, index) => NOTEBOOK_FONT_SIZE_MIN + index,
        ),
        "every integer step in range is reachable",
    );
});
