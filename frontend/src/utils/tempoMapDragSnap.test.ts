/**
 * 拖动 Tempo Map 变化点时的网格吸附自检（用户报告的场景）。
 *
 * 【用户场景】工程 120 BPM、4/4，网格 1/4（= 1 拍 = 0.5s）。0.75s 处有一个变化点，
 * 想把它拖到 1.0s：
 * - 没有这个点时，网格是 0 / 0.5 / 1.0 / …，1.0s 正是 `1.3.000`；
 * - 有了这个点时，它把网格重新锚定到 0.75s（段内原点），于是 1.0s 变成 `2.1.500`，
 *   而候选集是 `{0, 0.5}`（段 [0,0.75] 被段末截断）∪ `{0.75, 1.25, …}`，
 *   **1.0s 根本不在候选里**，怎么拖都吸不上。
 *
 * 【修复语义】被拖的点不参与自己要落位的网格：吸附用"移除本点"后的地图。
 */
import { test } from "vitest";

import { removeTempoPoint, type TempoMap } from "./tempoMap.js";
import { createDefaultTimelineSnapSettings } from "../features/session/sessionSlice";
import { snapTimelinePosition, type TimelineSnapContext } from "./timelineSnapping.js";

test("utils/tempoMapDragSnap.test.ts scripted checks", async () => {
    function assertNear(actual: number, expected: number, label: string, tol = 1e-6): void {
        if (!Number.isFinite(actual) || Math.abs(actual - expected) > tol) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    function assertTrue(condition: boolean, label: string): void {
        if (!condition) throw new Error(`${label}: expected true`);
    }

    /** 用户场景：0.75s 处的变化点（与工程同 BPM，因此它只重塑网格原点）。 */
    function userMap(): TempoMap {
        return {
            points: [
                {
                    id: "a",
                    positionSec: 0,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
                {
                    id: "b",
                    positionSec: 0.75,
                    bpm: 120,
                    timeSignature: { numerator: 4, denominator: 4 },
                    scale: null,
                },
            ],
        };
    }

    function context(tempoMap: TempoMap | null): TimelineSnapContext {
        return {
            settings: createDefaultTimelineSnapSettings(),
            grid: "1/4",
            bpm: 120,
            beatsPerBar: 4,
            tempoMap,
            pxPerSec: 400,
            clips: [],
            tracks: [],
            selectedClipIds: [],
            playheadSec: -1,
            object: "clip",
        };
    }

    const map = userMap();
    const snapTempoMap = removeTempoPoint(map, "b");
    assertTrue(snapTempoMap !== null, "移除被拖点后地图仍有效");
    assertNear(snapTempoMap!.points.length, 1, "只剩基准点");

    // 指针停在 1.0s 附近（2px 内，落在默认吸附阈值 4px 内）。
    const rawNearOne = 0.995;

    // ── 旧行为（含被拖点）：1.0s 不可达 ─────────────────────────
    const withSelf = snapTimelinePosition(context(map), rawNearOne);
    assertTrue(
        Math.abs(withSelf.sec - 1.0) > 1e-6,
        "含被拖点的网格吸不到 1.0s（这是缺陷本身的形状）",
    );

    // ── 新行为（移除被拖点）：1.0s 可达 ─────────────────────────
    const nearOne = snapTimelinePosition(context(snapTempoMap), rawNearOne);
    assertNear(nearOne.sec, 1.0, "移除被拖点后能吸到 1.0s");
    assertTrue(nearOne.snapped, "确实是吸附结果而非原值");

    // 0.5s 同样可达（它在两套地图里都是网格线）。
    const nearHalf = snapTimelinePosition(context(snapTempoMap), 0.503);
    assertNear(nearHalf.sec, 0.5, "0.5s 仍是吸附位");

    // 1.5s 也可达（1.0 之后的下一条网格线）。
    const nearOneHalf = snapTimelinePosition(context(snapTempoMap), 1.497);
    assertNear(nearOneHalf.sec, 1.5, "1.5s 也是吸附位");

    // ── 相邻变化点的位置仍必须是候选 ────────────────────────────
    // 拖到接近另一个变化点时，应当能精确落在它上面。
    const threePoints: TempoMap = {
        points: [
            { id: "a", positionSec: 0, bpm: 120, timeSignature: null, scale: null },
            { id: "b", positionSec: 0.75, bpm: 120, timeSignature: null, scale: null },
            { id: "c", positionSec: 4, bpm: 120, timeSignature: null, scale: null },
        ],
    };
    const reduced = removeTempoPoint(threePoints, "b");
    const nearNeighbor = snapTimelinePosition(context(reduced), 3.997);
    assertNear(nearNeighbor.sec, 4, "相邻变化点位置是吸附候选");
});
