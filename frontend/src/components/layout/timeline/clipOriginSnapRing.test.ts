/**
 * 时间轴 · 「回到原起点」与「目标轨描边」的**契约**。
 *
 * 这两件事住在两个模块里（`hooks/clipOriginSnap` 决定钉不钉、`trackDropTargetDrag`
 * 决定画不画），但它们表达的是同一个事实：这一趟只换了轨道，时间没动。
 * 只改一边就会分叉 —— 用户看到的正是「换了轨就多一圈描边」。
 *
 * 所以这里用**真实的** `resolveClipOriginSnap` 产出判据，再喂给**真实的**
 * `resolveTrackDropTarget`，把两个模块的契约钉在一起。
 *
 * 【它不做什么】不测吸附候选族（`timelineSnapping`）、不测落点（`applyKernelEditDelta`）、
 * 不测渲染层（`TrackDropTargetLayer`）。
 */
import { describe, expect, it } from "vitest";

import { resolveClipOriginSnap } from "./hooks/clipOriginSnap";
import { resolveTrackDropTarget } from "./trackDropTargetDrag";

const PX_PER_SEC = 150;
const PARTICIPANTS = [{ clipId: "a", trackId: "track-a" }];

/** 横向漂移 `driftPx` 时，被拖 Clip 会不会被描边。 */
function ringsAt(args: { driftPx: number; crossTrack: boolean; enabled?: boolean }): boolean {
    const originStartSec = 10;
    const snap = resolveClipOriginSnap({
        rawStartSec: originStartSec + args.driftPx / PX_PER_SEC,
        originStartSec,
        pxPerSec: PX_PER_SEC,
        crossTrack: args.crossTrack,
        enabled: args.enabled ?? true,
    });
    const target = resolveTrackDropTarget({
        moves: [{ clipId: "a", trackId: args.crossTrack ? "track-b" : "track-a" }],
        participants: PARTICIPANTS,
        dropToNewTrack: false,
        originPinned: snap.active,
    });
    return target !== null;
}

describe("回到原起点 ⇔ 目标轨描边", () => {
    it("★ 跨轨 + 横向只漂 5px（被吸回原起点）⇒ 有描边", () => {
        expect(ringsAt({ driftPx: 5, crossTrack: true })).toBe(true);
    });

    it("★ 跨轨 + 横向漂 200px（网格可能命中，但没回到原起点）⇒ 无描边", () => {
        expect(ringsAt({ driftPx: 200, crossTrack: true })).toBe(false);
    });

    it("同轨上下拖（横向不动）⇒ 无描边", () => {
        expect(ringsAt({ driftPx: 0, crossTrack: false })).toBe(false);
    });

    it("按住免吸附键（enabled = false）⇒ 既不钉也不描边", () => {
        expect(ringsAt({ driftPx: 5, crossTrack: true, enabled: false })).toBe(false);
    });

    it("阈值边界内（24px）仍有描边，越过阈值即消失", () => {
        expect(ringsAt({ driftPx: 24, crossTrack: true })).toBe(true);
        expect(ringsAt({ driftPx: 25, crossTrack: true })).toBe(false);
    });
});
