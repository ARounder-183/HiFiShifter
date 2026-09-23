/**
 * paramSelection.ts — 参数编辑器（钢琴卷帘）多选区模型。
 *
 * 【单位是帧，不是拍】选区一律用**工程级参数帧栅格**表达：`frame_period_ms` 是
 * 工程级常量（后端 `state.rs` 的 `frame_period_ms()` 恒返回 5.0，忽略 per-track
 * 状态），所以帧号是**与 bpm / Tempo Map 无关**的绝对时间坐标，换算唯一：
 * `sec = frame × framePeriodMs / 1000`。
 *
 * 此前选区存的是**拍**（`beat × secPerBeat`，`secPerBeat = 60 / bpm`）。于是"改 bpm"
 * 会把同一个选区重新解释到另一个时刻 —— 选区在画面上跟着 bpm 移动，并且它实际作用
 * 的帧区间也跟着变。这不是预期交互：选区只应由指针像素、缩放与滚动决定。帧号没有
 * 这个隐含依赖。
 *
 * 【类型就是后端的 FrameRange】`{startFrame, frameCount}`（半开区间
 * `[startFrame, startFrame + frameCount)`）正是 `get/set_param_frames`、
 * `mapClipboardToTargetRanges`、`applySelectionEditOverRanges` 已经在用的类型，
 * 因此选区可以**直接透传**给这些消费者，不再需要一层换算。
 *
 * 【指针 → 边界：两条规则，落在"两帧中间"】指针给的是**连续帧坐标**，选区存的是
 * 整数帧。第 k 帧的采样点画在 `x = framesToTime(k)`，视觉上它占据左右各半帧的领地，
 * 所以两条边界画在**两帧中间**（`k ± 0.5`），并各由一条规则定出：
 *   - **左**边界 = 指针**最近**的那一帧（`frameBoundLeftFromPointer`）：指针在
 *     `[0, 0.5)` → 第 0 帧（左缘切点 `-0.5`）→ **第 0 帧可以选中**；指针在
 *     `[0.5, 1.5)` → 第 1 帧（左缘切点 `0.5`）。
 *   - **右**边界 = 指针**所在帧之后**（`frameBoundRightFromPointer`）：指针在
 *     `[k, k+1)` → 到第 k 帧为止（右缘切点 `k + 0.5`）。
 * 两条规则**刻意不对称**（起点纳入手按下的那一帧，终点停在手停下的地方），
 * {@link frameRangeFromPointers} 是它们唯一的配对处（并保证至少选中一帧），
 * {@link resizeFrameRangeEdge} 按抓住的是哪条边界选用对应的那一条。
 *
 * 【合法选择区域 = 第 0 帧及以后】**工程起点之前没有帧可选**，因此：
 *   - 左边界下钳到 0：指针落在起点左侧那段额外空间（同步到时间轴时与轨道头等宽的
 *     负时间区）时，选区停在工程起点上，不会把负帧选进来；
 *   - 两次指针都落在起点之前时**不产生选区** —— 在那片空白里拖拽既不建新选区，
 *     也不动原选区。
 * 于是那段额外空间最多只能贡献"工程起点"这一个位置，**不会**让第 0 帧的判定范围
 * 膨胀成整片空间（它与其他帧一样，就是自己那半帧 + 下一帧的前半帧）。
 *
 * 切点是**渲染/命中**口径：{@link frameRangeStartCut} / {@link frameRangeEndCut}
 * 是唯一的"帧边界 → 切点"回算处，{@link rangeIndexAtFrame} 也按切点区间判定，
 * 于是"指针在段内"与"画出来的带"逐像素一致。
 *
 * 【数据路径不经过指针规则】全选、剪贴板推导、拉伸窗口这些本来就是整数帧的量，走
 * {@link frameRangeFromFrames} 直接构造 —— 它们不该被"最近帧 / 所在帧之后"再挪半帧。
 *
 * 【不变式】本模块返回的选区一律满足：
 *   1. 段按 `startFrame` 升序；
 *   2. 任意两段互不相交、也不相接（相接即合并）—— 这是"每段独立计算统计量"
 *      语义成立的前提：若相邻两段并存，某段的边缘淡化会跨进另一段的起点；
 *   3. `frameCount >= 0`，非有限值不参与；
 *   4. `frameCount === 0` = 单击留下的**零长选区**（两条切点重合）：画出来没有宽度，
 *      边缘命中带（`SELECTION_EDGE_MIN_WIDTH_PX`）也据此判定"没有可抓的边缘"。
 *
 * 归一化**不夹负值**：拖拽到 0 左侧时选区随数据一起越界（渲染自然裁掉），与改造前
 * 一致 —— 若在此夹到 0，选区宽度会在越界期间被永久吃掉。夹取只发生在
 * {@link selectionToFrameRanges}（交给后端的帧区间：起点夹 `>= 0`，并截掉越界部分）。
 *
 * 【两套区间约定刻意并存】各有一处明确用途，交界处都有注释：
 *   - `FrameRange`（半开 `[start, start + count)`）—— 选区本身，以及所有交给后端的
 *     帧区间。`frameCount` 就是"要写几帧"，没有 +1/-1 的换算。
 *   - `FrameSpan`（闭区间 `[start, end]`）—— 编辑窗口：{@link mergeFrameWindows}
 *     按"相邻帧即合并"合并窗口，供 `selectionEditData` 的多段编辑计划复用。
 */

/** 帧区间（半开）：`[startFrame, startFrame + frameCount)`，与后端契约一致。 */
export type FrameRange = { startFrame: number; frameCount: number };

/** 帧域闭区间（含两端）：编辑窗口的合并单位，端点约定与旧拖拽/拉伸路径一致。 */
export type FrameSpan = { startFrame: number; endFrame: number };

/** 多选区：归一化的帧区间列表（升序、互不相交也不相接、非空）。 */
export type ParamSelection = FrameRange[];

/** 与旧 `handleEditOp` 的 `clamp(..., 1, 200_000)` 保持一致的一次操作帧数上限。 */
export const MAX_SELECTION_FRAMES = 200_000;

function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/** **唯一取整边界**：连续帧坐标 → 整数帧（数据侧）。非有限值退化为 0。 */
export function snapFrame(frame: number): number {
    const value = Number(frame);
    return Number.isFinite(value) ? Math.round(value) : 0;
}

/**
 * 指针位置 → 选区**左**边界（帧号）：**指针最近的那一帧**。
 *
 * 用户的直觉是"我按在哪个帧上，就从那个帧开始选"：第 k 帧的采样点画在
 * `x = framesToTime(k)`，视觉上它占据左右各半帧，因此"指针最近的那一帧"就是
 * `round(f)`；左边界取它的左缘（切点 `k - 0.5`）。
 *
 * 【为什么左边界必须用这个规则】它让**第 0 帧可以被选中**：指针落在 `[-0.5, 0.5)`
 * 时左边界就是 `-0.5`（工程最起始那一帧的左缘）。若左边界改用右规则（"指针所在帧
 * 之后"），指针在 `[0, 0.5)` 时左边界会落到 `0.5` —— 等于从第 1 帧开始，第 0 帧
 * 在任何缩放下都选不中。
 */
export function frameBoundLeftFromPointer(pointerFrame: number): number {
    const value = Number(pointerFrame);
    return Number.isFinite(value) ? Math.round(value) : Number.NaN;
}

/**
 * 指针位置 → 选区**右**边界（帧号）：**指针所在帧的下一个边界**（`floor(f) + 1`）。
 *
 * 即"指针已经扫过的那一帧之后"：指针停在第 k 帧的范围内（`[k, k+1)`）时，选区到
 * 第 k 帧为止。
 *
 * 【两端为什么刻意不对称】左边界取"**最近**的那一帧"（把按下时指着的那一帧纳入），
 * 右边界取"**扫过**的那一帧之后"（不把还没扫到的那一帧纳入）—— 这正是拖拽的语义：
 * 起点要包含手按下的地方，终点停在手停下的地方。
 */
export function frameBoundRightFromPointer(pointerFrame: number): number {
    const value = Number(pointerFrame);
    return Number.isFinite(value) ? Math.floor(value) + 1 : Number.NaN;
}

/** 半开区间的右切点（= 最后一帧的下标 + 1）。**数据**口径，不是视觉切点。 */
export function frameRangeEnd(range: FrameRange): number {
    return range.startFrame + range.frameCount;
}

/**
 * 选区段的**左切点**（视觉左边界，落在两帧中间）：`startFrame - 0.5`。
 *
 * 【唯一用途】绘制与命中测试：选区的左右边界在画面上画在切点上，因此像素位置
 * 必须由切点投影，而不是由帧号投影（后者会让整个选区带偏右半帧）。
 *
 * 【帧边界 ↔ 切点】数据边界 `b`（帧号）对应的切点是 `b - 0.5`：第 k 帧占
 * `[k-0.5, k+0.5]`，它的左右两条视觉边界就是 `k ∓ 0.5`。这两个函数是唯一的回算处。
 */
export function frameRangeStartCut(range: FrameRange): number {
    return range.startFrame - 0.5;
}

/** 选区段的**右切点**（视觉右边界）：`startFrame + frameCount - 0.5`。 */
export function frameRangeEndCut(range: FrameRange): number {
    return frameRangeEnd(range) - 0.5;
}

/**
 * 两次**指针位置** → 一段选区（框选与多选追加的唯一入口）。
 *
 * 【方向无关】左边界取两个位置中**较小**者、右边界取**较大**者，因此从右往左拖
 * 与从左往右拖得到同一结果（两条规则各自绑定"选区的哪一侧"，而不是"先按哪个"）。
 *
 * 【合法区域只有第 0 帧及以后】左边界下钳到 **0**：工程起点之前没有帧可选（见
 * 模块头的说明），所以拖到那一段时选区停在起点上，绝不会把负帧选进来。这样左侧
 * 那段额外空间只可能贡献"工程起点"这一个位置，**不会**让"第 0 帧的判定范围"膨胀
 * 成整片空间。
 *
 * 【整段都在起点之前 → 不产生选区】两端都落在 `f < 0`（额外空间里）时返回非法
 * 区间：调用方的归一化会把它过滤掉，于是"在左侧空白处拖拽"既不产生新选区，也不
 * 改动原选区。这正是"工程起点之前无法被选中"的可执行形式。
 *
 * 【至少一帧】右边界若算出来不超过左边界（两次指针落在同一帧的右半格），顺延到
 * 左边界之后一帧：拖拽**总是**至少选中按下的那一帧 —— 与
 * {@link selectionToFrameRanges} 的"帧数下钳 1"同一约定，也避免了"拖了却什么都没
 * 选中"的空结果。
 *
 * @param aPointerFrame 按下（或起始）处的连续帧坐标。
 * @param bPointerFrame 当前（或结束）处的连续帧坐标。
 * @returns 单段帧区间；非有限输入或整段落在工程起点之前时为非法区间
 *   （由调用方的归一化过滤成"无选区"）。
 */
export function frameRangeFromPointers(aPointerFrame: number, bPointerFrame: number): FrameRange {
    if (!Number.isFinite(aPointerFrame) || !Number.isFinite(bPointerFrame)) {
        // 非有限指针（投影异常）→ 非法区间，由 `normalizeSelection` 过滤成"无选区"。
        // **不能**直接落到 `frameRangeFromFrames`：那里的 `snapFrame` 把非有限值变成 0，
        // 于是"指针不可用"会静默变成"选中第 0 帧"。
        return { startFrame: Number.NaN, frameCount: Number.NaN };
    }
    const lo = Math.min(aPointerFrame, bPointerFrame);
    const hi = Math.max(aPointerFrame, bPointerFrame);
    // 两个指针都在工程起点之前：整段落在不可选区域 → 不产生选区（见上文）。
    if (hi < 0) return { startFrame: Number.NaN, frameCount: Number.NaN };
    const startBound = Math.max(0, frameBoundLeftFromPointer(lo));
    const endBound = Math.max(frameBoundRightFromPointer(hi), startBound + 1);
    return frameRangeFromFrames(startBound, endBound - startBound);
}

/**
 * 由**整数帧**构造单段（数据路径：全选、剪贴板推导、拉伸窗口）。
 *
 * 刻意**不**走指针侧的两条边界规则：这些量本来就落在帧上，再套一层"最近帧 /
 * 所在帧之后"会把整段挪半帧。
 */
export function frameRangeFromFrames(startFrame: number, frameCount: number): FrameRange {
    return {
        startFrame: snapFrame(startFrame),
        frameCount: Math.max(0, snapFrame(frameCount)),
    };
}

/**
 * 归一化：过滤非有限值、取整、排序、合并重叠或相接的段。
 * 无有效段时返回 null（与「无选区」统一）。
 */
export function normalizeSelection(
    ranges: readonly FrameRange[] | null | undefined,
): ParamSelection | null {
    if (!ranges || ranges.length === 0) return null;

    const cleaned: FrameRange[] = [];
    for (const range of ranges) {
        if (!Number.isFinite(range.startFrame) || !Number.isFinite(range.frameCount)) continue;
        cleaned.push({
            startFrame: snapFrame(range.startFrame),
            frameCount: Math.max(0, snapFrame(range.frameCount)),
        });
    }
    if (cleaned.length === 0) return null;

    cleaned.sort((a, b) => a.startFrame - b.startFrame);

    const merged: FrameRange[] = [];
    for (const range of cleaned) {
        const last = merged[merged.length - 1];
        // 相接（last 的右切点 === range 起点）也合并：相邻段并存会让「每段独立」的
        // 边界淡化互相侵入。零长段落在邻段边界上时同样被吸收。
        if (last && range.startFrame <= frameRangeEnd(last)) {
            last.frameCount = Math.max(frameRangeEnd(last), frameRangeEnd(range)) - last.startFrame;
            continue;
        }
        merged.push({ ...range });
    }
    return merged;
}

/** 单段选区（两次**指针位置** → 选区，框选的唯一入口）；无有效段时为 null。 */
export function selectionFromPointers(
    aPointerFrame: number,
    bPointerFrame: number,
): ParamSelection | null {
    return normalizeSelection([frameRangeFromPointers(aPointerFrame, bPointerFrame)]);
}

/** 单段选区（整数帧 → 选区，数据路径）；无有效段时为 null。 */
export function selectionFromFrames(startFrame: number, frameCount: number): ParamSelection | null {
    return normalizeSelection([frameRangeFromFrames(startFrame, frameCount)]);
}

/** 追加一段（两次**指针位置** → 选区，多选追加的唯一入口）。 */
export function addPointerRange(
    selection: ParamSelection | null,
    aPointerFrame: number,
    bPointerFrame: number,
): ParamSelection | null {
    return normalizeSelection([
        ...(selection ?? []),
        frameRangeFromPointers(aPointerFrame, bPointerFrame),
    ]);
}

/**
 * 追加一段（**帧边界** → 选区，数据路径：音频块范围等）。
 *
 * 两个入参是帧轴上的边界（半开 `[aFrame, bFrame)`），次序颠倒时自动归一。
 */
export function addFrameRange(
    selection: ParamSelection | null,
    aFrame: number,
    bFrame: number,
): ParamSelection | null {
    const lo = snapFrame(Math.min(aFrame, bFrame));
    const hi = snapFrame(Math.max(aFrame, bFrame));
    return normalizeSelection([...(selection ?? []), frameRangeFromFrames(lo, hi - lo)]);
}

/** 并入若干**帧区间**（数据路径的批量入口，重叠/相接自动合并）。 */
export function addFrameRanges(
    selection: ParamSelection | null,
    ranges: readonly FrameRange[],
): ParamSelection | null {
    return normalizeSelection([...(selection ?? []), ...ranges]);
}

/**
 * 移除包含 `frame`（通常是**指针位置**）的那一段（多选修饰键点击已有段 = 切换取消）。
 *
 * 判定复用 {@link rangeIndexAtFrame}（切点区间 = 画出来的带）：点在可见区域内即命中。
 */
export function removeRangeAtFrame(
    selection: ParamSelection | null,
    frame: number,
): ParamSelection | null {
    if (!selection) return null;
    const index = rangeIndexAtFrame(selection, frame);
    if (index === -1) return normalizeSelection(selection);
    return normalizeSelection(selection.filter((_, i) => i !== index));
}

/**
 * 选区是否**完整覆盖** `[aCut, bCut)`。
 *
 * 用于「切换式」手势的判定：覆盖了才挖掉，否则并入 —— 这样同一个手势连按两次
 * 能回到原状（心形自反），而不是第二次把一个片段切碎。段升序且互不相接，
 * 因此一次线性扫描即可：一旦出现空洞即未覆盖。
 */
export function selectionCoversFrameRange(
    selection: ParamSelection | null,
    aFrame: number,
    bFrame: number,
): boolean {
    if (!selection || selection.length === 0) return false;
    const a = Math.min(aFrame, bFrame);
    const b = Math.max(aFrame, bFrame);
    if (!Number.isFinite(a) || !Number.isFinite(b)) return false;

    let coveredUntil = a;
    for (const range of selection) {
        const end = frameRangeEnd(range);
        if (end < coveredUntil) continue;
        if (range.startFrame > coveredUntil) return false; // 空洞
        if (end > coveredUntil) coveredUntil = end;
        if (coveredUntil >= b) return true;
    }
    return coveredUntil >= b;
}

/**
 * 区间相减：从选区中挖掉 `[aFrame, bFrame)`（**帧边界**，数据路径）。
 *
 * 结果仍归一化 —— 挖掉中间一小段会把原来的段切成两段，这正是「取消某个片段」应有
 * 的形态（断层即数据，不合并）。无重叠时原样返回（新数组）。
 */
export function subtractFrameRange(
    selection: ParamSelection | null,
    aFrame: number,
    bFrame: number,
): ParamSelection | null {
    if (!selection || selection.length === 0) return null;
    const a = Math.min(aFrame, bFrame);
    const b = Math.max(aFrame, bFrame);
    if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return normalizeSelection(selection);

    const out: FrameRange[] = [];
    for (const range of selection) {
        const end = frameRangeEnd(range);
        // 无重叠（含仅端点相接：相接不构成重叠）
        if (end <= a || range.startFrame >= b) {
            out.push({ startFrame: range.startFrame, frameCount: range.frameCount });
            continue;
        }
        if (range.startFrame < a) {
            out.push({ startFrame: range.startFrame, frameCount: a - range.startFrame });
        }
        if (end > b) out.push({ startFrame: b, frameCount: end - b });
    }
    return normalizeSelection(out);
}

/**
 * 切换：已完整覆盖 `[aFrame, bFrame)`（**帧边界**，数据路径）则挖掉该区间，否则并入。
 *
 * 时间轴的「修饰键 + 双击音频块」手势即此语义：同一个块再点一次即撤销。
 */
export function toggleFrameRange(
    selection: ParamSelection | null,
    aFrame: number,
    bFrame: number,
): ParamSelection | null {
    return selectionCoversFrameRange(selection, aFrame, bFrame)
        ? subtractFrameRange(selection, aFrame, bFrame)
        : addFrameRange(selection, aFrame, bFrame);
}

/**
 * `frame`（连续帧坐标，通常是**指针位置**）落在第几段（-1 = 不在任何段内）。
 *
 * 判定用**切点区间**（与画出来的带同口径）：第 k 帧的领地是 `[k-0.5, k+0.5]`，
 * 因此"指针落在段内"就是 `frame ∈ [左切点, 右切点]` —— 与用户看到的区域逐像素一致。
 * 用数据帧区间（`[startFrame, 末帧]`）判定会让命中区比带整体左偏半帧。
 *
 * 闭区间（含两端）：指针事件的坐标本来就离散，开区间会让"正好落在边界"失效。
 * 零长段（单击）的两条切点重合，因此只有恰好压在那一点上才算命中 —— 与它"没有
 * 可见区域"的事实一致。
 */
export function rangeIndexAtFrame(selection: ParamSelection | null, frame: number): number {
    if (!selection || !Number.isFinite(frame)) return -1;
    for (let i = 0; i < selection.length; i += 1) {
        const range = selection[i];
        if (frame >= frameRangeStartCut(range) && frame <= frameRangeEndCut(range)) return i;
    }
    return -1;
}

/** 包围区间（首段起点 → 末段右切点）。用于「只有一个时间窗」的旧接口。 */
export function selectionBoundingSpan(selection: ParamSelection | null): FrameRange | null {
    if (!selection || selection.length === 0) return null;
    const first = selection[0];
    const last = selection[selection.length - 1];
    return {
        startFrame: first.startFrame,
        frameCount: frameRangeEnd(last) - first.startFrame,
    };
}

/**
 * 把一段选区的**某一条边界**移到指针位置，另一条边界不动。
 *
 * 【两条规则由 `edge` 决定】抓住左边界就用"指针最近的那一帧"（{@link
 * frameBoundLeftFromPointer}，于是把指针拖到工程最左端就能把起点拉到第 0 帧）；
 * 抓住右边界就用"指针所在帧之后"（{@link frameBoundRightFromPointer}）。调用方
 * 只传**指针位置**，不需要自己判断该用哪条规则 —— 那正是这里最容易搞错的地方。
 *
 * 【拖过对侧边界 = 交换左右】用户抓住左边界一路往右拖、越过右边界时，期望行为是
 * "抓的那条边继续跟着手走"（此时它在右、另一条边成了左），而不是在边界处卡住 ——
 * 于是可以拖过去、再拖回来，反复交换。
 *
 * 实现只需把「被拖动的那条边界」与「固定的那条边界」两个端点排序：越过对侧时两端
 * 自然对调，既不会出现负宽度（会被归一化丢弃、手感上像选区消失），也不需要任何
 * "当前抓的是哪一侧"的状态。
 *
 * @param range 被调整的那一段（取自拖拽起点快照）。
 * @param edge 被抓住的边界（决定哪一端是固定端，以及用哪条指针规则）。
 * @param pointerFrame 光标当前的**连续帧坐标**（未量化；量化规则由 `edge` 选）。
 * @returns 调整后的段（已排序、已量化，始终 `frameCount >= 0`）。
 */
export function resizeFrameRangeEdge(
    range: FrameRange,
    edge: "left" | "right",
    pointerFrame: number,
): FrameRange {
    const moving =
        edge === "left"
            ? // 合法区域只有第 0 帧及以后：把左边界一路拖到工程起点左侧时停在 0，
              // 与框选同一约定（见 `frameRangeFromPointers`）。
              Math.max(0, frameBoundLeftFromPointer(pointerFrame))
            : frameBoundRightFromPointer(pointerFrame);
    const fixed = edge === "left" ? frameRangeEnd(range) : range.startFrame;
    const lo = Math.min(moving, fixed);
    const hi = Math.max(moving, fixed);
    return frameRangeFromFrames(lo, hi - lo);
}

/**
 * 把"整段平移"的位移量夹到合法范围（返回**可用的位移**，不是位移后的选区）。
 *
 * 【为什么要夹位移而不是夹结果】逐段裁剪位移后的选区会把选区**压扁**（各段被
 * 顶到边界上，段间距离丢失），手感上像被"吸住"。夹位移则保持选区形状不变：整段
 * 一起停在工程两端。
 *
 * 【约束】位移后选区的**帧**仍落在 `[minFrame, maxFrame]` 内：
 * `delta ≥ minFrame − 包围起点帧` 且 `delta ≤ maxFrame − 包围末端帧`。
 * 两个边界都是**帧号**（`maxFrame` 通常是"工程末端的帧数"，即末帧 + 1）。
 * 位移本身是整数帧，因此平移前后两条切点始终落在同一组中点上（`k ± 0.5`）。
 *
 * 【选区宽于 `[minFrame, maxFrame]` 时】两条约束交叉（区间为空），此时原样返回
 * `delta`：任何裁剪都必然违反另一侧的约束，自由平移比"完全卡死"更合理。
 *
 * 特殊说明：契约按**包围区间**定义，因此对多段选区同样成立（夹取的是整组的包围
 * 区间）。当前调用方（"右键拖拽平移选段落"）每次只传入**被抓住的那一段** ——
 * 多段选区下其余段不参与，也就不该限制本次平移的可用范围。保留包围区间的一般化
 * 定义，是为了让"夹取"这条规则不依赖调用方传几段。
 *
 * @param selection 选区（其包围区间决定可用位移）。
 * @param delta 期望位移（帧，可为负）。
 * @param minFrame 允许的最小帧号（通常是 0）。
 * @param maxFrame 允许的最大帧号（通常是工程末端的帧数）。
 * @returns 夹取后的位移；选区为空或位移非有限值时返回 0。
 */
export function clampSelectionShift(
    selection: ParamSelection | null,
    delta: number,
    minFrame: number,
    maxFrame: number,
): number {
    if (!Number.isFinite(delta) || delta === 0) return 0;
    const bounds = selectionBoundingSpan(selection);
    if (!bounds) return 0;
    if (!Number.isFinite(minFrame) || !Number.isFinite(maxFrame)) return delta;
    const lo = minFrame - bounds.startFrame;
    const hi = maxFrame - frameRangeEnd(bounds);
    if (hi < lo) return delta;
    return clamp(delta, lo, hi);
}

/** 整体平移（选区拖动时随数据一起移动）；结果仍归一化（可能合并出新重叠）。 */
export function shiftSelectionRanges(
    selection: ParamSelection | null,
    frameDelta: number,
): ParamSelection | null {
    if (!selection) return null;
    const delta = snapFrame(frameDelta);
    if (delta === 0) return selection;
    return normalizeSelection(
        selection.map((range) => ({
            startFrame: range.startFrame + delta,
            frameCount: range.frameCount,
        })),
    );
}

/**
 * 帧域合并：重叠或相接的窗口合并成一个。
 *
 * 也供多段编辑计划复用：若干段的「原位 ∪ 落地位 ± 边缘淡化」窗口合并后，
 * 一次写入即可覆盖全部受影响帧，且缝隙帧写入的是读回的基准值（语义不变）。
 */
export function mergeFrameWindows(windows: readonly FrameSpan[]): FrameSpan[] {
    if (windows.length === 0) return [];
    const sorted = [...windows].sort((a, b) => a.startFrame - b.startFrame);
    const merged: FrameSpan[] = [];
    for (const window of sorted) {
        const last = merged[merged.length - 1];
        if (last && window.startFrame <= last.endFrame + 1) {
            if (window.endFrame > last.endFrame) last.endFrame = window.endFrame;
            continue;
        }
        merged.push({ startFrame: window.startFrame, endFrame: window.endFrame });
    }
    return merged;
}

/**
 * 选区 → 交给后端的帧区间（`{startFrame, frameCount}`，逐段独立）。
 *
 * 选区**本来就是**这个类型，所以这里只做三件事：起点夹 `>= 0`、帧数上钳
 * {@link MAX_SELECTION_FRAMES}、截断后可能在 0 处叠合的段再合并一次。
 *
 * 【越界为什么截断而不是整体右移】选区可以被拖到 0 左侧（见模块头的"不夹负值"；
 * 指针落在工程左侧那段额外空间时，左边界会落到负帧）。旧的两条换算路径对此处理
 * 不一致：`beatRangesToInclusiveSpans` 截断，`beatRangesToFrameRanges` 保留整段
 * 宽度、整体右移 —— 后者会让实际编辑的帧区间与画出来的选区带错位。这里统一为
 * **截断**（与画面对齐）。
 *
 * 【整段落在 0 左侧 → 不写任何帧】截断后一帧都不剩的段直接丢弃：那段选区覆盖的是
 * 工程起点之前，没有任何数据可改。**注意与"零长段"区分**：`frameCount === 0`
 * （单击留下的那一段）仍按旧算式下钳为 **1 帧** —— 单击必须恰好作用一帧，而不是
 * "什么都没做"。
 */
export function selectionToFrameRanges(selection: ParamSelection | null): FrameRange[] {
    if (!selection || selection.length === 0) return [];
    const clamped: FrameSpan[] = [];
    for (const range of selection) {
        const rawStart = snapFrame(range.startFrame);
        const startFrame = Math.max(0, rawStart);
        // 零长段先归一到 1 帧，再做越界截断；截断吃光（≤ 0）则整段丢弃。
        const rawCount = Math.max(1, snapFrame(range.frameCount));
        const frameCount = Math.min(MAX_SELECTION_FRAMES, rawCount - (startFrame - rawStart));
        if (frameCount <= 0) continue;
        clamped.push({ startFrame, endFrame: startFrame + frameCount - 1 });
    }
    return mergeFrameWindows(clamped).map(({ startFrame, endFrame }) => ({
        startFrame,
        frameCount: Math.min(MAX_SELECTION_FRAMES, endFrame - startFrame + 1),
    }));
}

/**
 * 选区 → 帧域**闭区间**（含两端），夹取规则与 {@link selectionToFrameRanges} 完全一致。
 *
 * 【为什么需要这一层】`FrameRange`（半开）是"要写几帧"的数据口径，而编辑窗口 /
 * 取数路径（`readPvRange`、`planSelectionEditWindows`、`buildMultiRangeEditPlan`）
 * 用的是 `FrameSpan`（闭区间，{@link mergeFrameWindows} 的合并单位）。两者只在
 * 这里交界，且**同源**于同一个夹取实现 —— 不会出现"取数窗口与写入窗口差一帧"。
 */
export function selectionToFrameSpans(selection: ParamSelection | null): FrameSpan[] {
    return selectionToFrameRanges(selection).map(({ startFrame, frameCount }) => ({
        startFrame,
        endFrame: startFrame + frameCount - 1,
    }));
}
