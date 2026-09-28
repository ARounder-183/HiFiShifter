import { test } from "vitest";

import { createTimelineAxis } from "../renderKernel/timelineAxis.js";
import { frameFromViewportClientX } from "./seekPlayheadMapping.js";
import {
    frameRangeEndCut,
    frameRangeStartCut,
    selectionFromPointers,
    selectionToFrameRanges,
    type ParamSelection,
} from "./paramSelection.js";
import { framesToTime } from "./utils.js";

/**
 * 回归测试：**改 BPM 不得让参数编辑器的选区移动**。
 *
 * 【症状】用户调整 BPM 数值时，参数编辑器里的选区跟着移动。
 *
 * 【根因】选区当时以**拍**存储，而拍由 `secPerBeat = 60 / bpm` 从秒换算而来。
 * 改 BPM 后，同一份拍值被乘以**新的** `secPerBeat`：选区在画面上移位，实际作用的
 * 帧区间也跟着变 —— 选区与它真正编辑的数据一起"漂"到了别的时间上。
 *
 * 【现在的契约】选区的单位是工程级**帧**栅格（`frame_period_ms` 是常量，见
 * `paramSelection.ts`）。整条链路 —— 指针像素 → 帧坐标 → 选区边界 → 画出来的带 →
 * 交给后端的帧区间 —— 都**没有任何 BPM 输入**。
 *
 * 本文件把这个契约钉住：同一次手势、同一个视口、同一个帧周期，两组不同 BPM
 * 必须得到逐一相同的选区、带与帧区间；同时断言绝对值，避免"绕一圈再乘回去"
 * 这类看起来等价、实际会在取整处漂一帧的实现混进来。
 */

type Sessionish = {
    /**
     * 只用于**证明整条链路与它无关**：下面每个函数都不得读它。
     * 之所以还留着这个字段，是为了让"改 BPM"这件事在测试里显式发生，
     * 而不是靠读者自己想象。
     */
    bpm: number;
    pxPerSec: number;
    scrollLeftPx: number;
    framePeriodMs: number;
};

/** 一次框选手势：视口内 clientX → 连续帧坐标 → 选区（与 hook 同一路径）。 */
function boxSelect(
    session: Sessionish,
    startClientX: number,
    endClientX: number,
): ParamSelection | null {
    const axis = createTimelineAxis({
        pxPerSec: session.pxPerSec,
        scrollLeftPx: session.scrollLeftPx,
    });
    const toFrame = (clientX: number) =>
        frameFromViewportClientX({
            clientX,
            viewportLeft: 0,
            axis,
            framePeriodMs: session.framePeriodMs,
        });
    return selectionFromPointers(toFrame(startClientX), toFrame(endClientX));
}

/**
 * 画出来的选区带（与 `PianoRollPanel.buildSelectionBandSpec` 同一算法）：
 * 边界取**切点**（两帧中间），见 `paramSelection.frameRangeStartCut`。
 */
function bandSpansSec(
    selection: ParamSelection | null,
    framePeriodMs: number,
): Array<{ startSec: number; endSec: number }> {
    return (selection ?? []).map((range) => ({
        startSec: framesToTime(frameRangeStartCut(range), framePeriodMs),
        endSec: framesToTime(frameRangeEndCut(range), framePeriodMs),
    }));
}

test("components/layout/pianoRoll/paramSelection.bpmInvariance.test.ts scripted checks", async () => {
    function assertJson(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const b = JSON.stringify(expected);
        if (a !== b) {
            throw new Error(`${label}: expected ${b}, received ${a}`);
        }
    }

    // ── 绝对值：选区边界只由 (像素, 缩放, 滚动, 帧周期) 决定 ──────────────
    // pxPerSec = 200、fp = 5ms → **1 帧 = 1px**，便于核对。
    // clientX 250 → 帧 250；clientX 350 → 帧 350。
    // 左边界 = 指针最近的那一帧（250）、右边界 = 指针所在帧之后（351）
    // → 帧 [250, 351) = 第 250..350 帧，共 101 帧（指针扫过的每一帧都算）。
    {
        const session: Sessionish = {
            bpm: 120,
            pxPerSec: 200,
            scrollLeftPx: 0,
            framePeriodMs: 5,
        };
        const selection = boxSelect(session, 250, 350);
        assertJson(selection, [{ startFrame: 250, frameCount: 101 }], "absolute frame bounds");
        // 带画在两条切点上：249.5 ~ 350.5 帧 → 1.2475 ~ 1.7525 秒。
        assertJson(
            bandSpansSec(selection, session.framePeriodMs),
            [{ startSec: 1.2475, endSec: 1.7525 }],
            "band spans seconds",
        );
        assertJson(
            selectionToFrameRanges(selection),
            [{ startFrame: 250, frameCount: 101 }],
            "data ranges",
        );
    }

    // ── 核心：换 BPM，同一次手势的选区/带/帧区间逐一不变 ─────────────────
    {
        const at120: Sessionish = { bpm: 120, pxPerSec: 200, scrollLeftPx: 0, framePeriodMs: 5 };
        const at90: Sessionish = { bpm: 90, pxPerSec: 200, scrollLeftPx: 0, framePeriodMs: 5 };
        // 非整数像素（clientX 251.3 → 帧 251.3 → 切点 251.5）也一并覆盖：
        // 量化必须只发生一次。
        for (const [startX, endX] of [
            [250, 350],
            [251.3, 349.7],
            [0, 1000],
        ] as const) {
            const a = boxSelect(at120, startX, endX);
            const b = boxSelect(at90, startX, endX);
            assertJson(a, b, `selection identical across bpm (${startX}→${endX})`);
            assertJson(
                bandSpansSec(a, at120.framePeriodMs),
                bandSpansSec(b, at90.framePeriodMs),
                `band identical across bpm (${startX}→${endX})`,
            );
            assertJson(
                selectionToFrameRanges(a),
                selectionToFrameRanges(b),
                `data ranges identical across bpm (${startX}→${endX})`,
            );
        }
    }

    // ── 核心：**先存的选区**熬过一次 BPM 变更，渲染与作用范围都不许变 ──────
    // 这正是用户报告的场景：选区已经在那儿了，用户去调 BPM。
    {
        const at120: Sessionish = { bpm: 120, pxPerSec: 200, scrollLeftPx: 0, framePeriodMs: 5 };
        const stored = boxSelect(at120, 250, 350);
        const bandBefore = bandSpansSec(stored, at120.framePeriodMs);
        const rangesBefore = selectionToFrameRanges(stored);

        // 用户把 BPM 改成 90（帧栅格与工程级常量帧周期都不受影响）。
        const at90: Sessionish = { ...at120, bpm: 90 };
        assertJson(bandSpansSec(stored, at90.framePeriodMs), bandBefore, "band survives bpm edit");
        assertJson(selectionToFrameRanges(stored), rangesBefore, "ranges survive bpm edit");

        // 反向：从 90 改到 174（非整除的拍速，最容易暴露"绕一圈"的取整漂移）。
        const at174: Sessionish = { ...at120, bpm: 174 };
        assertJson(bandSpansSec(stored, at174.framePeriodMs), bandBefore, "band survives 174 bpm");
        assertJson(selectionToFrameRanges(stored), rangesBefore, "ranges survive 174 bpm");
    }

    // ── 滚动/缩放下同样成立（避免"只在某个缩放比下恰好相等"）────────────
    {
        const base: Sessionish = {
            bpm: 120,
            pxPerSec: 37.5,
            scrollLeftPx: 1234.5,
            framePeriodMs: 5,
        };
        const other: Sessionish = { ...base, bpm: 63 };
        const a = boxSelect(base, 10, 640);
        const b = boxSelect(other, 10, 640);
        assertJson(a, b, "selection identical at fractional zoom + scroll");
        assertJson(
            bandSpansSec(a, base.framePeriodMs),
            bandSpansSec(b, other.framePeriodMs),
            "band identical at fractional zoom + scroll",
        );
        assertJson(
            selectionToFrameRanges(a),
            selectionToFrameRanges(b),
            "ranges identical at fractional zoom + scroll",
        );
        // 起点 = (10 + 1234.5) / 37.5 s → 帧 6637.33… → 左边界取最近的帧 6637；
        // 终点 = (640 + 1234.5) / 37.5 s → 帧 9997.33… → 右边界取"所在帧之后" 9998。
        // 宽度 = 9998 − 6637 = 3361 帧（指针从第 6637 帧扫到第 9997 帧，共 3361 帧）。
        assertJson(
            a,
            [{ startFrame: 6637, frameCount: 3361 }],
            "scroll shifts frames without resizing",
        );
    }
});
