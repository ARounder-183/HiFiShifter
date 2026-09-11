import { test } from "vitest";

import {
    addBeatRange,
    beatRangesToFrameRanges,
    beatRangesToInclusiveSpans,
    makeBeatRange,
    normalizeSelection,
    rangeIndexAtBeat,
    removeRangeAtBeat,
    selectionBoundingRange,
    selectionContainsBeat,
    selectionCoversRange,
    selectionFromBeatRange,
    selectionTotalBeats,
    shiftSelectionRanges,
    subtractBeatRange,
    toggleBeatRange,
} from "./paramSelection.js";

/**
 * 这里锁住的是多选区的三条不变式（见 paramSelection.ts 模块注释）：
 *   1. 段升序、互不相交；
 *   2. 相邻（端点相接）即合并 —— 否则「每段独立计算统计量」会被
 *      邻段的边缘淡化侵入；
 *   3. 帧域换算与旧单选区算式逐帧一致（floor 起点 / ceil 时长，
 *      帧数下钳 1），单段选区不得因为多选区改造而漂移。
 */
test("components/layout/pianoRoll/paramSelection.test.ts scripted checks", async () => {
    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }
    function assertJson(actual: unknown, expected: unknown, label: string): void {
        const a = JSON.stringify(actual);
        const b = JSON.stringify(expected);
        if (a !== b) {
            throw new Error(`${label}: expected ${b}, received ${a}`);
        }
    }

    // ── normalizeSelection ───────────────────────────────────────────────
    assertEqual(normalizeSelection(null), null, "null stays null");
    assertEqual(normalizeSelection([]), null, "empty is null");
    assertEqual(normalizeSelection([{ startBeat: NaN, endBeat: 1 }]), null, "NaN dropped");
    assertEqual(normalizeSelection([{ startBeat: 0, endBeat: Infinity }]), null, "Inf dropped");

    // 乱序输入 → 升序输出
    assertJson(
        normalizeSelection([
            { startBeat: 5, endBeat: 6 },
            { startBeat: 1, endBeat: 2 },
        ]),
        [
            { startBeat: 1, endBeat: 2 },
            { startBeat: 5, endBeat: 6 },
        ],
        "sorted",
    );
    // 反向区间（a > b）自动排序两端点
    assertJson(
        normalizeSelection([{ startBeat: 3, endBeat: 1 }]),
        [{ startBeat: 1, endBeat: 3 }],
        "reversed endpoints",
    );
    // 负起点**不夹取**：拖拽到 0 左侧时选区随数据越界，宽度不能被吃掉
    assertJson(
        normalizeSelection([{ startBeat: -2, endBeat: 1 }]),
        [{ startBeat: -2, endBeat: 1 }],
        "negative preserved (frame clamping happens later)",
    );
    // 重叠合并
    assertJson(
        normalizeSelection([
            { startBeat: 0, endBeat: 3 },
            { startBeat: 2, endBeat: 5 },
        ]),
        [{ startBeat: 0, endBeat: 5 }],
        "overlap merged",
    );
    // 端点相接也合并（相邻段并存会互相侵入边缘淡化）
    assertJson(
        normalizeSelection([
            { startBeat: 0, endBeat: 3 },
            { startBeat: 3, endBeat: 5 },
        ]),
        [{ startBeat: 0, endBeat: 5 }],
        "touching merged",
    );
    // 真实缝隙保留断层
    assertJson(
        normalizeSelection([
            { startBeat: 0, endBeat: 1 },
            { startBeat: 2, endBeat: 3 },
        ]),
        [
            { startBeat: 0, endBeat: 1 },
            { startBeat: 2, endBeat: 3 },
        ],
        "gap preserved",
    );
    // 被完全包含的段不改变结果
    assertJson(
        normalizeSelection([
            { startBeat: 0, endBeat: 10 },
            { startBeat: 2, endBeat: 5 },
        ]),
        [{ startBeat: 0, endBeat: 10 }],
        "contained merged",
    );

    // ── makeBeatRange / selectionFromBeatRange ──────────────────────────
    assertJson(makeBeatRange(4, 2), { startBeat: 2, endBeat: 4 }, "makeBeatRange sorts");
    assertJson(selectionFromBeatRange(2, 4), [{ startBeat: 2, endBeat: 4 }], "single range");
    // 原地点击 = 退化段（沿用旧行为：仍是非空选区，占 1 帧）
    assertJson(selectionFromBeatRange(3, 3), [{ startBeat: 3, endBeat: 3 }], "degenerate click");

    // ── addBeatRange ────────────────────────────────────────────────────
    assertJson(
        addBeatRange(null, 0, 1),
        [{ startBeat: 0, endBeat: 1 }],
        "add to empty",
    );
    assertJson(
        addBeatRange([{ startBeat: 0, endBeat: 1 }], 2, 3),
        [
            { startBeat: 0, endBeat: 1 },
            { startBeat: 2, endBeat: 3 },
        ],
        "append disjoint",
    );
    // 与已有段重叠 → 合并（追加拖拽压到旧段上不应产生重叠段）
    assertJson(
        addBeatRange([{ startBeat: 0, endBeat: 2 }], 1, 4),
        [{ startBeat: 0, endBeat: 4 }],
        "add overlapping merges",
    );

    // ── removeRangeAtBeat ───────────────────────────────────────────────
    const twoRanges = [
        { startBeat: 0, endBeat: 1 },
        { startBeat: 2, endBeat: 3 },
    ];
    assertJson(
        removeRangeAtBeat(twoRanges, 2.5),
        [{ startBeat: 0, endBeat: 1 }],
        "remove second",
    );
    assertJson(
        removeRangeAtBeat(twoRanges, 0.5),
        [{ startBeat: 2, endBeat: 3 }],
        "remove first",
    );
    // 边界命中（端点属于该段）
    assertJson(removeRangeAtBeat(twoRanges, 1), [{ startBeat: 2, endBeat: 3 }], "remove at end");
    assertJson(removeRangeAtBeat(twoRanges, 2), [{ startBeat: 0, endBeat: 1 }], "remove at start");
    // 点在断层里 → 无变化
    assertJson(removeRangeAtBeat(twoRanges, 1.5), twoRanges, "remove in gap no-op");
    // 移除最后一段 → null（回到「无选区」）
    assertEqual(
        removeRangeAtBeat([{ startBeat: 0, endBeat: 1 }], 0.5),
        null,
        "remove last yields null",
    );
    assertEqual(removeRangeAtBeat(null, 0.5), null, "remove from null");

    // ── 命中查询 ────────────────────────────────────────────────────────
    assertEqual(rangeIndexAtBeat(twoRanges, 0.5), 0, "index first");    assertEqual(rangeIndexAtBeat(twoRanges, 1.5), -1, "index in gap");
    assertEqual(rangeIndexAtBeat(twoRanges, 2.5), 1, "index second");
    assertEqual(rangeIndexAtBeat(null, 1), -1, "index null");
    assertEqual(selectionContainsBeat(twoRanges, 3), true, "contains end");
    assertEqual(selectionContainsBeat(twoRanges, 1.2), false, "contains gap");

    // ── 覆盖判定 / 区间相减 / 切换（时间轴「修饰键 + 双击音频块」语义）──────
    // 覆盖：要求连续覆盖整个区间（断层不算覆盖）
    assertEqual(selectionCoversRange(twoRanges, 0, 3), false, "covers: gap is not coverage");
    assertEqual(selectionCoversRange(twoRanges, 0, 1), true, "covers: first range");
    assertEqual(selectionCoversRange(twoRanges, 0, 0.5), true, "covers: sub-range");
    assertEqual(selectionCoversRange(twoRanges, 0.5, 2.5), false, "covers: spanning gap");
    assertEqual(selectionCoversRange(twoRanges, 3, 4), false, "covers: beyond end");
    assertEqual(selectionCoversRange(null, 0, 1), false, "covers: null");
    // 相邻段合并后仍视为覆盖（归一化保证不出现相邻段，此处传入未归一化输入）
    assertEqual(
        selectionCoversRange(
            [
                { startBeat: 0, endBeat: 1 },
                { startBeat: 1, endBeat: 2 },
            ],
            0,
            2,
        ),
        true,
        "covers: touching ranges",
    );

    // 相减：中间挖洞 → 一段切成两段（断层即数据）
    assertJson(
        subtractBeatRange([{ startBeat: 0, endBeat: 3 }], 1, 2),
        [
            { startBeat: 0, endBeat: 1 },
            { startBeat: 2, endBeat: 3 },
        ],
        "subtract: splits into two",
    );
    // 相减：挖掉整段 → 空选区
    assertEqual(subtractBeatRange([{ startBeat: 0, endBeat: 1 }], 0, 1), null, "subtract: whole");
    // 相减：无重叠 → 原样（内容相同）
    assertJson(subtractBeatRange(twoRanges, 5, 6), twoRanges, "subtract: no overlap");
    // 相减：仅端点相接不构成重叠
    assertJson(subtractBeatRange(twoRanges, 1, 2), twoRanges, "subtract: touching endpoints");
    // 相减：挖掉跨段 + 空洞的一段 → 只吃掉真正重叠的部分
    assertJson(
        subtractBeatRange(twoRanges, 0.5, 2.5),
        [
            { startBeat: 0, endBeat: 0.5 },
            { startBeat: 2.5, endBeat: 3 },
        ],
        "subtract: spans gap",
    );
    assertEqual(subtractBeatRange(null, 0, 1), null, "subtract: null");

    // 切换：未覆盖 → 并入；已覆盖 → 挖掉（连按两次回到原状）
    assertJson(toggleBeatRange(null, 0, 1), [{ startBeat: 0, endBeat: 1 }], "toggle: add");
    assertJson(
        toggleBeatRange([{ startBeat: 0, endBeat: 1 }], 2, 3),
        [
            { startBeat: 0, endBeat: 1 },
            { startBeat: 2, endBeat: 3 },
        ],
        "toggle: append second range",
    );
    assertJson(
        toggleBeatRange([{ startBeat: 0, endBeat: 1 }], 0, 1),
        null,
        "toggle: removes a covered clip range",
    );
    // 部分覆盖 → 视为追加（不是挖掉）：避免把已有片段切碎
    assertJson(
        toggleBeatRange([{ startBeat: 0, endBeat: 1 }], 0.5, 2),
        [{ startBeat: 0, endBeat: 2 }],
        "toggle: partial overlap appends",
    );

    // ── 汇总 ────────────────────────────────────────────────────────────
    assertEqual(selectionTotalBeats(twoRanges), 2, "total beats skips gap");
    assertEqual(selectionTotalBeats(null), 0, "total null");
    assertJson(selectionBoundingRange(twoRanges), { startBeat: 0, endBeat: 3 }, "bounding");
    assertEqual(selectionBoundingRange(null), null, "bounding null");

    // ── 平移 ────────────────────────────────────────────────────────────
    assertJson(
        shiftSelectionRanges(twoRanges, 1),
        [
            { startBeat: 1, endBeat: 2 },
            { startBeat: 3, endBeat: 4 },
        ],
        "shift preserves gap",
    );
    // 负向平移：断层同样保留（纯平移不改变段间距，故不可能产生重叠段；
    // 越界部分交给帧域换算的 `startFrame >= 0` 夹取）
    assertJson(
        shiftSelectionRanges(twoRanges, -1),
        [
            { startBeat: -1, endBeat: 0 },
            { startBeat: 1, endBeat: 2 },
        ],
        "negative shift preserves gap",
    );
    assertEqual(shiftSelectionRanges(null, 1), null, "shift null");

    // ── beatRangesToFrameRanges：与旧单选区算式一致 ─────────────────────
    // secPerBeat = 0.5（120 BPM）、fp = 5ms：1 beat = 500ms = 100 帧
    {
        const single = beatRangesToFrameRanges([{ startBeat: 1, endBeat: 3 }], 0.5, 5);
        assertJson(single, [{ startFrame: 100, frameCount: 200 }], "single legacy math");
    }
    // 退化段（原地点击）→ 1 帧（旧实现的 clamp(..., 1, ...)）
    {
        const degenerate = beatRangesToFrameRanges([{ startBeat: 3, endBeat: 3 }], 0.5, 5);
        assertJson(degenerate, [{ startFrame: 300, frameCount: 1 }], "degenerate 1 frame");
    }
    // 多段：各自独立换算，断层保留
    {
        const multi = beatRangesToFrameRanges(
            [
                { startBeat: 0, endBeat: 1 },
                { startBeat: 2, endBeat: 3 },
            ],
            0.5,
            5,
        );
        assertJson(
            multi,
            [
                { startFrame: 0, frameCount: 100 },
                { startFrame: 200, frameCount: 100 },
            ],
            "multi ranges stay separate",
        );
    }
    // 不足一帧的缝隙（0.005 beat = 0.5 帧）在帧域消失 → 合并
    {
        const merged = beatRangesToFrameRanges(
            [
                { startBeat: 1, endBeat: 2 },
                { startBeat: 2.005, endBeat: 3 },
            ],
            0.5,
            5,
        );
        assertJson(merged, [{ startFrame: 100, frameCount: 200 }], "sub-frame gap merges");
    }

    // ── beatRangesToInclusiveSpans：闭区间约定（拖拽/拉伸路径） ──────────
    {
        const spans = beatRangesToInclusiveSpans(
            [
                { startBeat: 1, endBeat: 2 },
                { startBeat: 2.005, endBeat: 3 },
            ],
            0.5,
            5,
        );
        assertJson(spans, [{ startFrame: 100, endFrame: 300 }], "inclusive spans merge");
    }
    {
        const spans = beatRangesToInclusiveSpans(
            [
                { startBeat: 1, endBeat: 2 },
                { startBeat: 4, endBeat: 5 },
            ],
            0.5,
            5,
        );
        assertJson(
            spans,
            [
                { startFrame: 100, endFrame: 200 },
                { startFrame: 400, endFrame: 500 },
            ],
            "inclusive spans multi",
        );
    }
    assertEqual(beatRangesToInclusiveSpans(null, 0.5, 5).length, 0, "spans null");
    assertEqual(beatRangesToFrameRanges([], 0.5, 5).length, 0, "ranges empty");
});
