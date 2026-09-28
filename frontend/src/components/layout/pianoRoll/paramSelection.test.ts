import { test } from "vitest";

import {
    addPointerRange,
    clampSelectionShift,
    frameBoundLeftFromPointer,
    frameBoundRightFromPointer,
    frameRangeEnd,
    frameRangeEndCut,
    frameRangeFromFrames,
    frameRangeFromPointers,
    frameRangeStartCut,
    normalizeSelection,
    rangeIndexAtFrame,
    removeRangeAtFrame,
    resizeFrameRangeEdge,
    selectionBoundingSpan,
    selectionCoversFrameRange,
    selectionFromFrames,
    selectionFromPointers,
    selectionToFrameRanges,
    selectionToFrameSpans,
    shiftSelectionRanges,
    snapFrame,
    subtractFrameRange,
    toggleFrameRange,
} from "./paramSelection.js";

/**
 * 这里锁住的是多选区的四条不变式（见 paramSelection.ts 模块注释）：
 *   1. 段升序、互不相交、也不相接（相接即合并）—— 否则「每段独立计算统计量」
 *      会被邻段的边缘淡化侵入；
 *   2. 单位是**整数帧**，量化只发生在构造处；
 *   3. **指针 → 边界的两条规则**：左 = 指针最近的那一帧、右 = 指针所在帧之后
 *      （于是**第 0 帧可选**，且拖拽方向不影响结果）；
 *   4. 交给后端的帧区间只做夹取/钳制，单段选区不因为多选区改造而漂移。
 *
 * 【与改造前的关系】本文件的用例逐条对应改造前的拍制版本，只换了单位：凡是
 * 原来用「1 beat = 500ms = 100 帧」把拍换算成帧的断言，现在直接写帧号。
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

    // ── snapFrame（数据侧的取整）────────────────────────────────────────
    assertEqual(snapFrame(3.4), 3, "snapFrame rounds down below .5");
    assertEqual(snapFrame(3.5), 4, "snapFrame rounds up at .5");
    assertEqual(snapFrame(Number.NaN), 0, "snapFrame non-finite → 0");

    // ── 指针 → 边界的两条规则 ────────────────────────────────────────────
    // 左边界 = **指针最近的那一帧**。
    assertEqual(frameBoundLeftFromPointer(0), 0, "left bound: pointer at frame 0 → frame 0");
    assertEqual(frameBoundLeftFromPointer(0.4), 0, "left bound: first half of frame 0");
    assertEqual(frameBoundLeftFromPointer(0.5), 1, "left bound: second half of frame 0");
    assertEqual(frameBoundLeftFromPointer(1.2), 1, "left bound: frame 1");
    assertEqual(frameBoundLeftFromPointer(1.5), 2, "left bound: frame 2");
    assertEqual(frameBoundLeftFromPointer(2.7), 3, "left bound: frame 3");
    assertEqual(frameBoundLeftFromPointer(-0.4), 0, "left bound: left of frame 0 → frame 0");
    // 规则本身**照实**：指针在工程起点左侧就是负帧（第 0 帧的判定范围不会因为
    // 那片额外空间而变宽）。"起点之前不可选"由配对处下钳，见 frameRangeFromPointers。
    assertEqual(frameBoundLeftFromPointer(-0.6), -1, "left bound: just left of frame 0 → -1");
    assertEqual(frameBoundLeftFromPointer(-3.2), -3, "left bound: far left → the negative frame");
    assertEqual(
        Number.isNaN(frameBoundLeftFromPointer(Number.NaN)),
        true,
        "left bound: non-finite → NaN",
    );
    // 右边界 = **指针所在帧之后**（`floor(f) + 1`）。
    assertEqual(frameBoundRightFromPointer(0), 1, "right bound: frame 0 → 1");
    assertEqual(frameBoundRightFromPointer(0.9), 1, "right bound: still inside frame 0");
    assertEqual(frameBoundRightFromPointer(2), 3, "right bound: frame 2 → 3");
    assertEqual(frameBoundRightFromPointer(2.7), 3, "right bound: frame 2 → 3");
    assertEqual(frameBoundRightFromPointer(3), 4, "right bound: frame 3 → 4");
    assertEqual(
        Number.isNaN(frameBoundRightFromPointer(Number.NaN)),
        true,
        "right bound: non-finite → NaN",
    );

    assertEqual(frameRangeEnd({ startFrame: 10, frameCount: 5 }), 15, "range end is exclusive");
    // 帧边界 ↔ 切点：第 k 帧的领地是 [k-0.5, k+0.5]。
    assertEqual(frameRangeStartCut({ startFrame: 10, frameCount: 5 }), 9.5, "start cut");
    assertEqual(frameRangeEndCut({ startFrame: 10, frameCount: 5 }), 14.5, "end cut");

    // ── frameRangeFromPointers：**需求原文的两个例子** ───────────────────
    // 起点：指针在 [0.5, 1.5) → 左边界落在第 1 帧（视为已选中第 1 帧）。
    assertJson(
        frameRangeFromPointers(0.6, 1.4),
        { startFrame: 1, frameCount: 1 },
        "start rule: pointer in [0.5,1.5) starts at frame 1",
    );
    assertJson(
        frameRangeFromPointers(0.5, 1.4),
        { startFrame: 1, frameCount: 1 },
        "start rule: left endpoint of the bucket",
    );
    assertJson(
        frameRangeFromPointers(1.5, 3.4),
        { startFrame: 2, frameCount: 2 },
        "start rule: pointer in [1.5,2.5) starts at frame 2",
    );
    // 终点：指针停在 [2, 3) → 到第 2 帧为止；停在 [3, 4) → 到第 3 帧为止。
    assertJson(
        frameRangeFromPointers(0.6, 2.9),
        { startFrame: 1, frameCount: 2 },
        "end rule: pointer in [2,3) ends at frame 2",
    );
    assertJson(
        frameRangeFromPointers(0.6, 3.9),
        { startFrame: 1, frameCount: 3 },
        "end rule: pointer in [3,4) ends at frame 3",
    );
    // **第 0 帧可选**：指针落在工程最起始的半帧内即可。
    assertJson(
        frameRangeFromPointers(0, 0.3),
        { startFrame: 0, frameCount: 1 },
        "frame 0 is selectable from the very start",
    );
    assertJson(
        frameRangeFromPointers(0.2, 0.4),
        { startFrame: 0, frameCount: 1 },
        "frame 0 is selectable inside its first half",
    );
    // 方向无关：从右往左拖与从左往右拖得到同一结果。
    assertJson(
        frameRangeFromPointers(3.4, 1.2),
        frameRangeFromPointers(1.2, 3.4),
        "drag direction does not matter",
    );
    // 合法选择区域只有第 0 帧及以后：工程起点左侧那段额外空间（负时间）**选不中**。
    // 整段都在起点之前 → 不产生选区（调用方的归一化把它过滤成"无选区"）。
    assertEqual(
        Number.isNaN(frameRangeFromPointers(-3.2, -1.2).startFrame),
        true,
        "a drag entirely left of the project selects nothing",
    );
    assertEqual(selectionFromPointers(-3.2, -1.2), null, "…and normalizes to 'no selection'");
    // 从起点之前拖进工程 → 选区停在工程起点上（负帧永远不会被选进来）。
    assertJson(
        frameRangeFromPointers(-3.2, 2.4),
        { startFrame: 0, frameCount: 3 },
        "drag from the left margin starts at frame 0",
    );
    // 反向：从工程内往左拖出起点 → 同样停在工程起点。
    assertJson(
        frameRangeFromPointers(3.2, -3.2),
        { startFrame: 0, frameCount: 4 },
        "drag out to the left stops at frame 0",
    );
    // 起点之前的那段空间只可能贡献"工程起点"这一个位置，不会让第 0 帧的判定范围
    // 膨胀：`frameBoundLeftFromPointer` 的规则本身照实（-3.2 → -3），是**配对**时下钳。
    assertEqual(frameBoundLeftFromPointer(-3.2), -3, "the rule itself stays honest");
    // 至少一帧：两次指针落在同一帧的右半格时，仍选中按下的那一帧。
    assertJson(
        frameRangeFromPointers(1.7, 1.9),
        { startFrame: 2, frameCount: 1 },
        "at least one frame",
    );
    assertJson(
        frameRangeFromPointers(2, 2),
        { startFrame: 2, frameCount: 1 },
        "zero-length press still selects one frame",
    );
    // 非有限指针 → 非法帧区间（由 normalizeSelection 过滤成"无选区"）。
    assertEqual(
        Number.isNaN(frameRangeFromPointers(Number.NaN, 1).startFrame),
        true,
        "non-finite pointer → NaN range",
    );

    // ── frameRangeFromFrames：数据路径不经过指针规则 ─────────────────────
    assertJson(
        frameRangeFromFrames(4, 2),
        { startFrame: 4, frameCount: 2 },
        "frames passthrough (no half-frame shift)",
    );
    assertJson(
        frameRangeFromFrames(3, -2),
        { startFrame: 3, frameCount: 0 },
        "negative count clamped to zero",
    );

    // ── normalizeSelection ───────────────────────────────────────────────
    assertEqual(normalizeSelection(null), null, "null stays null");
    assertEqual(normalizeSelection([]), null, "empty is null");
    assertEqual(
        normalizeSelection([{ startFrame: NaN, frameCount: 1 }]),
        null,
        "NaN start dropped",
    );
    assertEqual(
        normalizeSelection([{ startFrame: 0, frameCount: Infinity }]),
        null,
        "Inf count dropped",
    );

    // 乱序输入 → 升序输出
    assertJson(
        normalizeSelection([
            { startFrame: 5, frameCount: 1 },
            { startFrame: 1, frameCount: 1 },
        ]),
        [
            { startFrame: 1, frameCount: 1 },
            { startFrame: 5, frameCount: 1 },
        ],
        "sorted",
    );
    // 负帧数（反向区间）→ 夹成零长段，不产生负宽度
    assertJson(
        normalizeSelection([{ startFrame: 3, frameCount: -2 }]),
        [{ startFrame: 3, frameCount: 0 }],
        "negative count clamped to zero",
    );
    // 负起点**不夹取**：拖拽到 0 左侧时选区随数据越界，宽度不能被吃掉
    assertJson(
        normalizeSelection([{ startFrame: -2, frameCount: 3 }]),
        [{ startFrame: -2, frameCount: 3 }],
        "negative preserved (clamping happens at selectionToFrameRanges)",
    );
    // 重叠合并
    assertJson(
        normalizeSelection([
            { startFrame: 0, frameCount: 3 },
            { startFrame: 2, frameCount: 3 },
        ]),
        [{ startFrame: 0, frameCount: 5 }],
        "overlap merged",
    );
    // 相接（前段的右切点 === 后段起点）也合并：相邻段并存会互相侵入边缘淡化
    assertJson(
        normalizeSelection([
            { startFrame: 0, frameCount: 3 },
            { startFrame: 3, frameCount: 2 },
        ]),
        [{ startFrame: 0, frameCount: 5 }],
        "touching merged",
    );
    // 真实缝隙（差 1 帧以上）保留断层
    assertJson(
        normalizeSelection([
            { startFrame: 0, frameCount: 1 },
            { startFrame: 2, frameCount: 1 },
        ]),
        [
            { startFrame: 0, frameCount: 1 },
            { startFrame: 2, frameCount: 1 },
        ],
        "gap preserved",
    );
    // 被完全包含的段不改变结果
    assertJson(
        normalizeSelection([
            { startFrame: 0, frameCount: 10 },
            { startFrame: 2, frameCount: 3 },
        ]),
        [{ startFrame: 0, frameCount: 10 }],
        "contained merged",
    );
    // 零长段落在邻段边界上 → 被邻段吸收
    assertJson(
        normalizeSelection([
            { startFrame: 10, frameCount: 5 },
            { startFrame: 15, frameCount: 0 },
        ]),
        [{ startFrame: 10, frameCount: 5 }],
        "degenerate at boundary absorbed",
    );
    // 孤立的零长段保留（单击产生的选区就是它）
    assertJson(
        normalizeSelection([{ startFrame: 42, frameCount: 0 }]),
        [{ startFrame: 42, frameCount: 0 }],
        "lone degenerate preserved",
    );

    // ── selectionFromPointers（指针路径）────────────────────────────────
    // 第 0 帧可选：指针落在工程最起始的半帧内。
    assertJson(selectionFromPointers(0, 0.3), [{ startFrame: 0, frameCount: 1 }], "frame 0");
    // 指针在 [1.5, 3.5) 的扫描 → 第 2、3 帧。
    assertJson(selectionFromPointers(1.5, 3.4), [{ startFrame: 2, frameCount: 2 }], "frames 2..3");
    // 原地按下（两端同一像素）= 至少一帧（单击的收尾仍会把整段清掉）。
    assertJson(
        selectionFromPointers(3, 3),
        [{ startFrame: 3, frameCount: 1 }],
        "press selects one",
    );
    // 小数像素：按两条规则各自定界，不再"两端取整"。
    assertJson(
        selectionFromPointers(10.6, 20.4),
        [{ startFrame: 11, frameCount: 10 }],
        "fractional pointers follow the two rules",
    );

    // ── selectionFromFrames（数据路径）──────────────────────────────────
    assertJson(selectionFromFrames(0, 1000), [{ startFrame: 0, frameCount: 1000 }], "whole curve");
    assertJson(
        selectionFromFrames(2, 0),
        [{ startFrame: 2, frameCount: 0 }],
        "degenerate from frames",
    );

    // ── resizeFrameRangeEdge（选区边界拖拽，含"拖过对侧 = 交换左右"）──────
    {
        // 覆盖帧 10..19。
        const range = { startFrame: 10, frameCount: 10 };
        // 抓左边界 → 用"指针最近的那一帧"；抓右边界 → 用"指针所在帧之后"。
        assertJson(
            resizeFrameRangeEdge(range, "left", 12),
            { startFrame: 12, frameCount: 8 },
            "drag left edge inward",
        );
        assertJson(
            resizeFrameRangeEdge(range, "left", 5),
            { startFrame: 5, frameCount: 15 },
            "drag left edge outward",
        );
        // **左边界拖到工程最左端 → 起点落到第 0 帧**（框选之外的另一条可达路径）。
        assertJson(
            resizeFrameRangeEdge(range, "left", 0),
            { startFrame: 0, frameCount: 20 },
            "drag left edge to the very start reaches frame 0",
        );
        assertJson(
            resizeFrameRangeEdge(range, "left", 0.4),
            { startFrame: 0, frameCount: 20 },
            "drag left edge inside the first half-frame reaches frame 0",
        );
        // 再往左（拖进工程起点之前的额外空间）也停在工程起点：合法区域只有第 0 帧
        // 及以后，负帧永远不会被选进来。
        assertJson(
            resizeFrameRangeEdge(range, "left", -3.2),
            { startFrame: 0, frameCount: 20 },
            "drag left edge into the margin stops at frame 0",
        );
        assertJson(
            resizeFrameRangeEdge(range, "right", 25),
            { startFrame: 10, frameCount: 16 },
            "drag right edge outward",
        );
        assertJson(
            resizeFrameRangeEdge(range, "right", 15),
            { startFrame: 10, frameCount: 6 },
            "drag right edge inward",
        );
        // 右边界停在帧内 → 到该帧为止（与框选的终点规则一致）。
        assertJson(
            resizeFrameRangeEdge(range, "right", 19.9),
            { startFrame: 10, frameCount: 10 },
            "right edge stops after the pointer's frame",
        );
        // 拖过对侧边界：**交换左右**，而不是卡在对侧边界上。
        assertJson(
            resizeFrameRangeEdge(range, "left", 22),
            { startFrame: 20, frameCount: 2 },
            "drag left edge past the right edge swaps sides",
        );
        assertJson(
            resizeFrameRangeEdge(range, "right", 8),
            { startFrame: 9, frameCount: 1 },
            "drag right edge past the left edge swaps sides",
        );
        // 交换可逆：越过之后再拖回来，与"从未越过"得到同一结果
        // （固定端始终是按下时的对侧边界，因此不需要记录"当前抓哪一侧"）。
        assertJson(
            resizeFrameRangeEdge(range, "left", 15),
            { startFrame: 15, frameCount: 5 },
            "swap is reversible",
        );
        // 恰好压在对侧边界上 → 退化为零长段（不会产生负宽度）。
        assertJson(
            resizeFrameRangeEdge(range, "left", 20),
            { startFrame: 20, frameCount: 0 },
            "exactly on the opposite bound degenerates",
        );
    }

    // ── clampSelectionShift（右键整段平移选区的位移夹取）────────────────
    {
        // 覆盖帧 10..19，右切点 20。
        const one = [{ startFrame: 10, frameCount: 10 }];
        // 区间内原样通过。
        assertEqual(clampSelectionShift(one, 5, 0, 100), 5, "shift within range");
        assertEqual(clampSelectionShift(one, -5, 0, 100), -5, "shift negative within range");
        // 夹到两端：包围区间不得越出 [0, 100]（两个边界都是**帧号**）。
        assertEqual(clampSelectionShift(one, -15, 0, 100), -10, "shift clamped at left edge");
        assertEqual(clampSelectionShift(one, 95, 0, 100), 80, "shift clamped at right edge");
        // 多选区按**包围区间**夹取：形状（段间距）保持不变，整段一起停住。
        const two = [
            { startFrame: 10, frameCount: 10 },
            { startFrame: 30, frameCount: 10 },
        ];
        assertEqual(clampSelectionShift(two, -30, 0, 100), -10, "multi range clamped by bounds");
        assertEqual(clampSelectionShift(two, 100, 0, 100), 60, "multi range right clamp");
        // 选区本身宽于 [minFrame, maxFrame]：约束交叉 → 自由平移（不裁成压扁的形状）。
        const wide = [{ startFrame: -5, frameCount: 205 }];
        assertEqual(clampSelectionShift(wide, 40, 0, 100), 40, "wider-than-domain passes through");
        // 退化输入。
        assertEqual(clampSelectionShift(null, 5, 0, 100), 0, "null selection");
        assertEqual(clampSelectionShift(one, Number.NaN, 0, 100), 0, "NaN delta");
        assertEqual(clampSelectionShift(one, 5, 0, Number.NaN), 5, "NaN bound");
    }

    // ── addPointerRange（指针路径：两次指针位置 → 并入）─────────────────
    assertJson(
        addPointerRange(null, 0, 0.3),
        [{ startFrame: 0, frameCount: 1 }],
        "add frame 0 to empty",
    );
    assertJson(
        addPointerRange([{ startFrame: 0, frameCount: 1 }], 2.4, 3.4),
        [
            { startFrame: 0, frameCount: 1 },
            { startFrame: 2, frameCount: 2 },
        ],
        "append disjoint",
    );
    // 与已有段相接/重叠 → 合并（追加拖拽压到旧段上不应产生重叠段）
    assertJson(
        addPointerRange([{ startFrame: 0, frameCount: 2 }], 1.5, 3.4),
        [{ startFrame: 0, frameCount: 4 }],
        "add adjacent merges",
    );

    // ── removeRangeAtFrame ──────────────────────────────────────────────
    const twoRanges = [
        { startFrame: 0, frameCount: 1 },
        { startFrame: 2, frameCount: 1 },
    ];
    assertJson(
        removeRangeAtFrame(twoRanges, 2),
        [{ startFrame: 0, frameCount: 1 }],
        "remove second",
    );
    assertJson(
        removeRangeAtFrame(twoRanges, 0),
        [{ startFrame: 2, frameCount: 1 }],
        "remove first",
    );
    // 边界命中（末帧属于该段）
    assertJson(
        removeRangeAtFrame(twoRanges, 0),
        [{ startFrame: 2, frameCount: 1 }],
        "remove at start",
    );
    // 点在断层里 → 无变化
    assertJson(removeRangeAtFrame(twoRanges, 1), twoRanges, "remove in gap no-op");
    // 移除最后一段 → null（回到「无选区」）
    assertEqual(
        removeRangeAtFrame([{ startFrame: 0, frameCount: 1 }], 0),
        null,
        "remove last yields null",
    );
    assertEqual(removeRangeAtFrame(null, 0), null, "remove from null");
    // 零长段：两条切点重合（band = [6.5, 6.5]），只有恰好压在那一点上才命中 ——
    // 与它"没有可见区域"的事实一致。
    const degenerate = [{ startFrame: 7, frameCount: 0 }];
    assertEqual(rangeIndexAtFrame(degenerate, 6.5), 0, "degenerate hit on its single cut");
    assertEqual(rangeIndexAtFrame(degenerate, 7), -1, "degenerate misses its own frame");

    // ── 命中查询（用**切点区间**判定，与画出来的带同口径）────────────────
    // twoRanges 的两条带分别是 [-0.5, 0.5] 与 [1.5, 2.5]。
    assertEqual(rangeIndexAtFrame(twoRanges, 0), 0, "index first");
    assertEqual(rangeIndexAtFrame(twoRanges, 1), -1, "index in gap");
    assertEqual(rangeIndexAtFrame(twoRanges, 2), 1, "index second");
    assertEqual(rangeIndexAtFrame(twoRanges, -0.5), 0, "band's left cut is inside");
    assertEqual(rangeIndexAtFrame(twoRanges, 0.5), 0, "band's right cut is inside");
    assertEqual(rangeIndexAtFrame(twoRanges, 0.6), -1, "just past the band's right cut");
    assertEqual(rangeIndexAtFrame(twoRanges, 1.5), 1, "second band's left cut is inside");
    assertEqual(rangeIndexAtFrame(twoRanges, 2.5), 1, "second band's right cut is inside");
    assertEqual(rangeIndexAtFrame(null, 1), -1, "index null");

    // ── 覆盖判定 / 区间相减 / 切换（时间轴「修饰键 + 双击音频块」语义）──────
    // 覆盖：要求连续覆盖整个区间（断层不算覆盖）。区间为半开 `[aCut, bCut)`。
    assertEqual(selectionCoversFrameRange(twoRanges, 0, 3), false, "covers: gap is not coverage");
    assertEqual(selectionCoversFrameRange(twoRanges, 0, 1), true, "covers: first range");
    assertEqual(selectionCoversFrameRange(twoRanges, 0, 1), true, "covers: sub-range");
    assertEqual(selectionCoversFrameRange(twoRanges, 0, 3), false, "covers: spanning gap");
    assertEqual(selectionCoversFrameRange(twoRanges, 3, 4), false, "covers: beyond end");
    assertEqual(selectionCoversFrameRange(null, 0, 1), false, "covers: null");
    // 相接的两段在**未归一化**输入下仍视为覆盖（归一化保证不会出现相邻段）
    assertEqual(
        selectionCoversFrameRange(
            [
                { startFrame: 0, frameCount: 1 },
                { startFrame: 1, frameCount: 1 },
            ],
            0,
            2,
        ),
        true,
        "covers: touching ranges",
    );

    // 相减：中间挖洞 → 一段切成两段（断层即数据）
    assertJson(
        subtractFrameRange([{ startFrame: 0, frameCount: 3 }], 1, 2),
        [
            { startFrame: 0, frameCount: 1 },
            { startFrame: 2, frameCount: 1 },
        ],
        "subtract: splits into two",
    );
    // 相减：挖掉整段 → 空选区
    assertEqual(
        subtractFrameRange([{ startFrame: 0, frameCount: 1 }], 0, 1),
        null,
        "subtract: whole",
    );
    // 相减：无重叠 → 原样（内容相同）
    assertJson(subtractFrameRange(twoRanges, 5, 6), twoRanges, "subtract: no overlap");
    // 相减：仅切点相接不构成重叠
    assertJson(subtractFrameRange(twoRanges, 1, 2), twoRanges, "subtract: touching cuts");
    // 相减：挖掉跨段 + 空洞的一段 → 只吃掉真正重叠的部分
    assertJson(
        subtractFrameRange(twoRanges, 0, 2),
        [{ startFrame: 2, frameCount: 1 }],
        "subtract: spans gap",
    );
    assertEqual(subtractFrameRange(null, 0, 1), null, "subtract: null");

    // 切换：未覆盖 → 并入；已覆盖 → 挖掉（连按两次回到原状）
    assertJson(toggleFrameRange(null, 0, 1), [{ startFrame: 0, frameCount: 1 }], "toggle: add");
    assertJson(
        toggleFrameRange([{ startFrame: 0, frameCount: 1 }], 2, 3),
        [
            { startFrame: 0, frameCount: 1 },
            { startFrame: 2, frameCount: 1 },
        ],
        "toggle: append second range",
    );
    assertJson(
        toggleFrameRange([{ startFrame: 0, frameCount: 1 }], 0, 1),
        null,
        "toggle: removes a covered clip range",
    );
    // 部分覆盖 → 视为追加（不是挖掉）：避免把已有片段切碎
    assertJson(
        toggleFrameRange([{ startFrame: 0, frameCount: 1 }], 0, 2),
        [{ startFrame: 0, frameCount: 2 }],
        "toggle: partial overlap appends",
    );

    // ── 汇总 ────────────────────────────────────────────────────────────
    assertJson(selectionBoundingSpan(twoRanges), { startFrame: 0, frameCount: 3 }, "bounding");
    assertEqual(selectionBoundingSpan(null), null, "bounding null");

    // ── 平移 ────────────────────────────────────────────────────────────
    assertJson(
        shiftSelectionRanges(twoRanges, 1),
        [
            { startFrame: 1, frameCount: 1 },
            { startFrame: 3, frameCount: 1 },
        ],
        "shift preserves gap",
    );
    // 负向平移：断层同样保留（纯平移不改变段间距，故不可能产生重叠段；
    // 越界部分交给 selectionToFrameRanges 的 `startFrame >= 0` 夹取）
    assertJson(
        shiftSelectionRanges(twoRanges, -1),
        [
            { startFrame: -1, frameCount: 1 },
            { startFrame: 1, frameCount: 1 },
        ],
        "negative shift preserves gap",
    );
    // 小数位移同样取整（拖动时指针给的是连续帧坐标）
    assertJson(shiftSelectionRanges(twoRanges, 0.4), twoRanges, "sub-frame shift snaps to zero");
    assertEqual(shiftSelectionRanges(null, 1), null, "shift null");

    // ── selectionToFrameRanges：交给后端的帧区间 ────────────────────────
    {
        const single = selectionToFrameRanges([{ startFrame: 100, frameCount: 200 }]);
        assertJson(single, [{ startFrame: 100, frameCount: 200 }], "single passthrough");
    }
    // 退化段（原地点击）→ 1 帧（与旧实现的 clamp(..., 1, ...) 一致）
    {
        const degenerate = selectionToFrameRanges([{ startFrame: 300, frameCount: 0 }]);
        assertJson(degenerate, [{ startFrame: 300, frameCount: 1 }], "degenerate 1 frame");
    }
    // 多段：各自独立，断层保留
    {
        const multi = selectionToFrameRanges([
            { startFrame: 0, frameCount: 100 },
            { startFrame: 200, frameCount: 100 },
        ]);
        assertJson(
            multi,
            [
                { startFrame: 0, frameCount: 100 },
                { startFrame: 200, frameCount: 100 },
            ],
            "multi ranges stay separate",
        );
    }
    // 越界到 0 左侧：**截掉**越界部分（而不是保留整段宽度整体右移）——
    // 后者会让实际编辑的帧与画出来的选区带错位。
    {
        const truncated = selectionToFrameRanges([{ startFrame: -20, frameCount: 50 }]);
        assertJson(truncated, [{ startFrame: 0, frameCount: 30 }], "negative start truncated");
    }
    // 完全落在 0 左侧 → 整段丢弃（工程起点之前没有任何数据可改）
    {
        const dropped = selectionToFrameRanges([{ startFrame: -50, frameCount: 10 }]);
        assertJson(dropped, [], "fully negative is dropped");
    }
    // 恰好贴到 0（右边界落在 0）→ 同样没有可改的帧
    {
        const touching = selectionToFrameRanges([{ startFrame: -3, frameCount: 3 }]);
        assertJson(touching, [], "range ending exactly at 0 writes nothing");
    }
    // 只压到第 0 帧一点 → 只写那一帧
    {
        const one = selectionToFrameRanges([{ startFrame: -3, frameCount: 4 }]);
        assertJson(one, [{ startFrame: 0, frameCount: 1 }], "only frame 0 overlaps");
    }
    // 截断后两段叠在 0 处 → 帧域再合并一次（避免同一批帧被写两遍）
    {
        const merged = selectionToFrameRanges([
            { startFrame: -10, frameCount: 12 },
            { startFrame: -3, frameCount: 8 },
        ]);
        assertJson(merged, [{ startFrame: 0, frameCount: 5 }], "truncated overlap merges");
    }
    assertEqual(selectionToFrameRanges(null).length, 0, "ranges null");
    assertEqual(selectionToFrameRanges([]).length, 0, "ranges empty");

    // ── selectionToFrameSpans：闭区间（取数 / 编辑窗口用）────────────────
    {
        const spans = selectionToFrameSpans([
            { startFrame: 100, frameCount: 100 },
            { startFrame: 400, frameCount: 100 },
        ]);
        assertJson(
            spans,
            [
                { startFrame: 100, endFrame: 199 },
                { startFrame: 400, endFrame: 499 },
            ],
            "inclusive spans multi",
        );
    }
    // 与 selectionToFrameRanges 同源：同一份夹取/截断结果，不会出现"取数窗口
    // 与写入窗口差一帧"。
    {
        const spans = selectionToFrameSpans([{ startFrame: -20, frameCount: 50 }]);
        assertJson(spans, [{ startFrame: 0, endFrame: 29 }], "inclusive spans share the clamping");
    }
    assertEqual(selectionToFrameSpans(null).length, 0, "spans null");
});
