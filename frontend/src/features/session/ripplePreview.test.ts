/**
 * 波纹实时预览的批量派发。
 *
 * 【要钉住的性质】"all" 模式下每帧对 K 个跟随 clip 的平移必须是**一次**
 * `moveClipsStartBulk`（而非逐 clip `moveClipStart`）：后者每帧 K 次 immer
 * producer + O(N) 的 `clips.find`，大工程拖拽会卡。归零（delta=0）同样走批量，
 * 语义与逐条 `moveClipStart` 一致（钳零 + 越界扩展工程时长）。
 */

import { expect, test } from "vitest";

import type { AppDispatch } from "../../app/store";
import reducer, { moveClipsStartBulk } from "./sessionSlice.js";
import { applyRippleFollowerShift } from "./ripplePreview.js";

test("applyRippleFollowerShift：每帧只派发一次批量动作", () => {
    const dispatched: unknown[] = [];
    const dispatch = ((action: unknown) => {
        dispatched.push(action);
    }) as unknown as AppDispatch;

    applyRippleFollowerShift(dispatch, { "clip-a": 4, "clip-b": 10 }, 2.5);

    expect(dispatched).toHaveLength(1);
    expect(dispatched[0]).toEqual(
        moveClipsStartBulk([
            { clipId: "clip-a", startSec: 6.5 },
            { clipId: "clip-b", startSec: 12.5 },
        ]),
    );
});

test("applyRippleFollowerShift：空跟随集不派发", () => {
    const dispatched: unknown[] = [];
    const dispatch = ((action: unknown) => {
        dispatched.push(action);
    }) as unknown as AppDispatch;

    applyRippleFollowerShift(dispatch, {}, 1);

    expect(dispatched).toHaveLength(0);
});

test("moveClipsStartBulk：钳零并自动扩展工程时长（与 moveClipStart 同口径）", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const state = {
        ...base,
        projectSec: 4,
        clips: [
            { id: "clip-a", startSec: 1, lengthSec: 2, trackId: "t1" },
            { id: "clip-b", startSec: 2, lengthSec: 2, trackId: "t1" },
        ],
    } as unknown as ReturnType<typeof reducer>;

    const next = reducer(
        state,
        moveClipsStartBulk([
            { clipId: "clip-a", startSec: -3 },
            { clipId: "clip-b", startSec: 30 },
        ]),
    );

    expect(next.clips[0].startSec).toBe(0);
    expect(next.clips[1].startSec).toBe(30);
    // clip-b 末端 32 > projectSec 4 → 扩展为 32。
    expect(next.projectSec).toBe(32);
});
