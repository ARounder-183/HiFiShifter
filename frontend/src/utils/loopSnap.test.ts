import { test } from "vitest";

import {
    loopSnapThresholdSec,
    nearestBoundarySnapOffsetSec,
    slipBoundaryAlignedSides,
    type BoundarySnapClip,
} from "./loopSnap";

test("utils/loopSnap.test.ts scripted checks", async () => {
    /**
     * loopSnap（循环/内容边界吸附）测试：slip 对齐侧判定。
     */

    let checks = 0;
    function assertTrue(value: boolean, label: string) {
        checks += 1;
        if (!value) throw new Error(`assertion failed: ${label}`);
    }
    function assertJson(actual: unknown, expected: unknown, label: string) {
        checks += 1;
        const a = JSON.stringify(actual);
        const e = JSON.stringify(expected);
        if (a !== e) throw new Error(`${label}: expected ${e}, got ${a}`);
    }

    const BASE: BoundarySnapClip = {
        loopEnabled: false,
        reversed: false,
        sourceStartSec: 0,
        sourceEndSec: 0,
        playbackRate: 1,
        lengthSec: 4,
    };

    // ── 非 Loop 正放：ss=2、se=6、len=4、r=1、媒体时长 D=8 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            sourceStartSec: 2,
            sourceEndSec: 6,
            durationSec: 8,
        };
        // 起点对齐候选：(b−ss)/r → b=0: −2；b=8(D): 6。
        assertJson(slipBoundaryAlignedSides(clip, -2), { start: true, end: false }, "fwd start@b0");
        assertJson(slipBoundaryAlignedSides(clip, 6), { start: true, end: false }, "fwd start@bD");
        // 终点对齐候选：base−len → b=0: −6；b=8: 2。
        assertJson(slipBoundaryAlignedSides(clip, -6), { start: false, end: true }, "fwd end@b0");
        assertJson(slipBoundaryAlignedSides(clip, 2), { start: false, end: true }, "fwd end@bD");
        // 非候选位置：两侧都不对齐。
        assertJson(slipBoundaryAlignedSides(clip, 1.5), { start: false, end: false }, "fwd none");
        // 媒体时长未知：只有 b=0 候选族。
        const noDur: BoundarySnapClip = { ...BASE, sourceStartSec: 2, sourceEndSec: 6 };
        assertJson(
            slipBoundaryAlignedSides(noDur, 6),
            { start: false, end: false },
            "unknown duration drops b=D candidates",
        );
    }

    // ── 非 Loop 倒放：ss=6、se=10、len=4、r=1 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            reversed: true,
            sourceStartSec: 6,
            sourceEndSec: 10,
        };
        // 起点对齐：(b−se)/r → b=0: −10；终点对齐：base+len → −6。
        assertJson(slipBoundaryAlignedSides(clip, -10), { start: true, end: false }, "rev start");
        assertJson(slipBoundaryAlignedSides(clip, -6), { start: false, end: true }, "rev end");
    }

    // ── Loop：len·r 恰为整周期 → 两侧同时对齐 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            loopEnabled: true,
            durationSec: 4,
            sourceStartSec: 0,
            sourceEndSec: 4,
            lengthSec: 4,
        };
        assertJson(
            slipBoundaryAlignedSides(clip, 0),
            { start: true, end: true },
            "loop period both",
        );
    }

    // ── Loop 正放：d=4、len=1、ss=0 → 起点 φ=0、终点 φ=3 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            loopEnabled: true,
            durationSec: 4,
            lengthSec: 1,
        };
        assertJson(
            slipBoundaryAlignedSides(clip, 0),
            { start: true, end: false },
            "loop fwd start-only",
        );
        assertJson(
            slipBoundaryAlignedSides(clip, 3),
            { start: false, end: true },
            "loop fwd end-only",
        );
    }

    // ── Loop 倒放：d=4、se=9（seEff=min(se,D)=4）、len=1 → 起点 φ=0、终点 φ=1 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            loopEnabled: true,
            reversed: true,
            durationSec: 4,
            sourceEndSec: 9,
            lengthSec: 1,
        };
        assertJson(
            slipBoundaryAlignedSides(clip, 0),
            { start: true, end: false },
            "loop rev start-only",
        );
        assertJson(
            slipBoundaryAlignedSides(clip, 1),
            { start: false, end: true },
            "loop rev end-only",
        );
    }

    // ── 与 nearestBoundarySnapOffsetSec 一致性：吸附结果必落在对齐侧上 ──
    {
        const clip: BoundarySnapClip = {
            ...BASE,
            loopEnabled: true,
            durationSec: 4,
            lengthSec: 1,
        };
        const snapped = nearestBoundarySnapOffsetSec(clip, "slip", 2.9);
        assertTrue(snapped != null, "nearest returns candidate");
        if (snapped != null) {
            const sides = slipBoundaryAlignedSides(clip, snapped);
            assertTrue(sides.start || sides.end, "snapped offset aligns at least one side");
        }
    }

    // 阈值换算仍正常（回归）。
    assertTrue(loopSnapThresholdSec(40, 40) === 1, "threshold px/sec");

    void checks;
});
