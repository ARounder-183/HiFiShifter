/**
 * 时间轴渲染内核 · 拖拽起手时的选区收敛判定
 *
 * 【主要内容】
 * `shouldCollapseStaleSelectionOnDrag()`：判断「一次左键手势升级为 clip 拖拽」时，
 * 是否应当**忽略当前多选集合、只移动被抓住的那一个 clip**。
 *
 * 【作用：为什么需要它——「拖一个 clip，右边的 clip 也跟着动」的根因】
 * 参与拖拽的集合来自 `multiSelectedClipIds`（见
 * `kernelEditSet.resolveKernelEditParticipants`）：只要被拖的 clip 与邻居**都在**
 * 集合里，整组就会一起移动。问题在于集合有两种来源，而它们的用户意图完全不同：
 *
 * 1. **用户显式选择**（框选 / 主修饰键点击 / Shift 范围选择 / 全选）——
 *    "我要操作这几个"，此时整组移动是**正确**行为（旧实现与
 *    `kernelEditSet.test.ts` 都定义了该语义）。
 * 2. **动作驱动的批量填充**（多文件导入 / 打开工程 / 粘贴 / 分割 / 复制拖拽）——
 *    只是"把新产生的东西设为当前选中"，并**不表示**用户想整组拖动。多文件导入后
 *    同轨相邻的多个 clip 都在集合里，用户随手抓住其中一个拖，右邻就跟着走
 *    —— 这就是用户报告的缺陷。而多选描边只有 2px（单选 1px），他看不出自己
 *    "选中了多个"。
 *
 * 因此判定必须依赖**选区来源**（`selectionIntentional`），只对来源 2 收敛。
 * 仅凭"集合大小 > 1"收敛会砍掉来源 1 的既有功能（实测：主修饰键多选 A、B 后
 * 直接拖 A，两个本该一起移动）；完全不收敛则来源 2 的缺陷依旧（实测：Shift +
 * 拖拽会带上陈旧邻块）。
 *
 * 【为什么判定放在"升级为拖拽"而不是"按下"】
 * 因为 Shift 是**双重绑定**，且两个绑定分属不同类型（见
 * `features/keybindings/defaultKeybindings` 的 `modifierOperationType`）：
 * - `modifier.clipRangeSelect`（默认 Shift）= **click** 类型：范围选择，靠
 *   "上一次点击的锚点"，只在**点击**收尾时成立；
 * - `modifier.clipNoSnap`（默认 Shift）= **drag** 类型：拖拽时临时关闭吸附。
 *
 * 若在**按下**时按 click 语义处理 Shift，会改写范围选择锚点，破坏"先点 A、
 * 再 Shift 点 B"的既有行为；而若因 Shift 就跳过收敛，则 Shift + 拖拽会带着陈旧
 * 集合走（实测确认）。升级为拖拽时"这是拖拽而非点击"已确定，Shift 的 click 含义
 * 不再适用，只按 drag 语义判定即可让两条路径同时正确。
 *
 * 【与其他模块的关系】
 * - 上游：`TimelinePanel.handleKernelDragPreview` 在建立手势 origin 时调用
 *   （必须早于参与者集合的计算，否则第一次预览已按旧集合算过）。
 * - 下游：命中收敛时面板直接用 `[anchorClipId]` 作为参与者种子。
 * - 独立性：纯函数，不依赖 DOM / React / Redux，可直接单测。
 */

/** 收敛判定入参。 */
export interface CollapseStaleSelectionArgs {
    /**
     * 是否按住了**多选切换键**（`modifier.clipMultiSelectToggle`，默认 Ctrl/⌘）。
     *
     * 该键同时是复制拖拽键（`modifier.clipCopyDrag`）：按住时要把**整组**复制
     * 出去，收敛会让副本少成员，因此不收敛。
     */
    readonly multiSelectToggleActive: boolean;
    /**
     * 当前多选集合是否来自**用户显式选择**动作。
     *
     * `false` = 由导入 / 粘贴 / 分割 / 打开工程 / 复制拖拽等动作批量填充，
     * 不代表用户想整组拖动（见文件头）。
     */
    readonly selectionIntentional: boolean;
    /** 当前多选集合的大小。 */
    readonly multiSelectionSize: number;
    /** 被抓住（锚点）的 clip 是否在多选集合内。 */
    readonly anchorInMultiSelection: boolean;
}

/**
 * 判断拖拽起手时是否应当收敛为「只移动被抓住的那一个」。
 *
 * 流程（任一条件不满足即不收敛）：
 * 1. 按住多选切换键（Ctrl/⌘）→ 不收敛（复制拖拽要整组复制）；
 * 2. 选区来自用户显式选择 → 不收敛（整组移动是既有功能）；
 * 3. 集合 <= 1 个成员 → 不收敛（参与者本就只有它）；
 * 4. 锚点**不在**集合内 → 不收敛（`resolveKernelEditParticipants` 已经只取锚点，
 *    无需额外动作）；
 * 5. 其余（集合 > 1、锚点在集合内、选区来自动作填充）→ **收敛**。
 *
 * 特殊说明：判定**不看 Shift**。Shift 的 click 语义（范围选择）与 drag 语义
 * （免吸附）分属不同类型，拖拽路径只认后者，而后者不影响选区（见文件头）。
 *
 * @param args 见 `CollapseStaleSelectionArgs`。
 * @returns 需要收敛时为 true。
 */
export function shouldCollapseStaleSelectionOnDrag(args: CollapseStaleSelectionArgs): boolean {
    if (args.multiSelectToggleActive) return false;
    if (args.selectionIntentional) return false;
    if (args.multiSelectionSize <= 1) return false;
    if (!args.anchorInMultiSelection) return false;
    return true;
}
