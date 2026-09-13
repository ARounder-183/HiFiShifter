/**
 * 时间轴渲染内核 · 拖拽起手时的选区收敛判定
 *
 * 【主要内容】
 * `shouldPrimeSelectionOnPress()`：判断「在 clip 上按下左键」时，是否应当**先把
 * 选区收敛到这一个 clip**（旧实现 `TrackLane.primeSelection` 的语义）。
 *
 * 【作用：为什么需要它——这是「拖一个 clip，右边的 clip 也跟着动」的根因】
 * 参与拖拽的集合来自 `multiSelectedClipIds`（见 `kernelEditSet.resolveKernelEditParticipants`）：
 * 只要**被拖的 clip 与邻居都在多选集合里**，整组就会一起移动——这是设计语义，
 * 不是缺陷。缺陷在于**用户没有主动多选**时集合里却已经有多个成员。
 *
 * 旧实现（`TrackLane.tsx`，重构已删除）在 `pointerdown` 就调用 `primeSelection`：
 * 只要没按多选 / 范围选择修饰键，就立刻 `ensureSelected(clipId)`，把陈旧的
 * 多选集合收敛为单个。内核重写时**只移植了"抬起且未位移才选中"**那条路径
 * （`onGesturePointerUp` 的 `pending-select` 分支），起手时的收敛被漏掉了。
 * 后果：`multiSelectedClipIds` 只要被任何**非选择动作**填成多个（多文件导入、
 * 分割、粘贴 / 复制拖拽、全选、Shift 范围选择），此后**直接拖拽**其中任一成员
 * 都会带上其余成员——而拖拽不会经过"抬起选中"，集合永远不会自愈。
 *
 * 由于多选描边只有 2px（单选 1px），用户完全看不出自己"选中了多个"，
 * 于是报告为「拖一个 clip，右边的 clip 跟着动」。
 *
 * 【与其他模块的关系】
 * - 上游：`host/timelineKernelHost` 在左键按下命中 clip 时调用。
 * - 下游：判定为 true 时宿主回调 `interactions.onSelectClip`（该回调最终走
 *   `ensureTrackLaneSelected` → 收敛集合）。
 * - 复用：修饰键语义由 `features/keybindings/clipSelectionModifiers` 提供
 *   （`shouldPrimeSelection` = 未按多选键且未按范围选择键）；本模块只负责把它与
 *   "命中是否已在选区内"合并成最终判定，避免宿主里散落分支。
 * - 独立性：纯函数，不依赖 DOM / React，可直接单测。
 *
 * 【维护说明】这条规则与「单击选中」共享同一个修饰键契约：若将来新增选择类
 * 修饰键，应改 `clipSelectionModifiers` 而不是在这里加分支。
 */

/** 起手收敛判定入参。 */
export interface PrimeSelectionOnPressArgs {
    /**
     * 修饰键解析结果里的"应当收敛选区"标志
     * （= 未按多选切换键、且未按范围选择键）。
     */
    readonly shouldPrimeSelection: boolean;
    /** 按下的 clip 是否已在多选集合内。 */
    readonly clipInMultiSelection: boolean;
    /** 当前多选集合的大小。 */
    readonly multiSelectionSize: number;
}

/**
 * 判断按下时是否需要把选区收敛到被按下的 clip。
 *
 * 流程：未按选择类修饰键（`shouldPrimeSelection`）→ 再看集合现状：
 * - 集合为空 → **无需收敛**（单选本就指向它，或首次点击由抬起路径处理）；
 * - 集合只有 1 个成员且就是它 → 无需收敛（已是目标状态）；
 * - 集合有多个成员，或该 clip 不在集合内 → **需要收敛**。
 *
 * 为什么"集合为空"返回 false：一致性与幂等性。空集合时 `selectedClipId` 才是
 * 权威（可能已指向该 clip），此时再发一次选中回调是多余的状态写入，且会打断
 * "抬起才选中"的既有路径（同一手势写两次选区）。
 *
 * 特殊说明：**不做任何钳制 / 校验**（如 clipId 是否存在）——那由调用方的命中
 * 结果保证；本函数只回答"集合是否需要收敛"。
 *
 * @param args 见 `PrimeSelectionOnPressArgs`。
 * @returns 需要收敛时为 true。
 */
export function shouldPrimeSelectionOnPress(args: PrimeSelectionOnPressArgs): boolean {
    if (!args.shouldPrimeSelection) return false;
    if (args.multiSelectionSize <= 0) return false;
    if (args.multiSelectionSize === 1 && args.clipInMultiSelection) return false;
    return true;
}
