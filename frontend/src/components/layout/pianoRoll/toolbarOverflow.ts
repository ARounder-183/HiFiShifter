/**
 * 参数编辑器工具栏「水平不足 ⇒ 分级隐藏」的判据（纯逻辑，便于单测）。
 *
 * # 为什么需要
 * 工具栏两行都是普通 flex：横向不足时先由**低频控件连续让步**
 * （平滑度滑块 `flex: 1 1 0` 缩到 16px、标签出省略号，见 `PianoRollPanel` 的既有注释），
 * 但连续让步是有下限的。到顶之后若仍放不下，就必须**按优先级离散地隐藏**冗余项，
 * 否则整行会溢出到卷帘画布上。
 *
 * 本模块只负责「当前该隐藏到第几级」，不关心每一级具体隐藏什么
 * （那是 `PianoRollPanel` 的渲染职责）。拆开是为了让判据可被单测穷举 ——
 * 分级逻辑一旦出错会表现为"阈值附近疯狂闪烁"，很难在界面上复现定位。
 *
 * # 层级约定
 * `0` = 全部显示。数值越大隐藏越多，见 `PianoRollPanel` 的 `TOOLBAR_TIER_*` 使用点。
 */

/**
 * 回落时要求留出的余量（px）。
 *
 * 【为什么必须存在】若"进入隐藏"与"退出隐藏"用同一个阈值，容器宽度在阈值附近
 * 抖动时（拖拽分栏、浮窗拉伸）会反复切换层级 —— 界面上表现为控件闪烁。
 * 进入时只要溢出就升级，退出时必须留出这么多余量才降级。
 *
 * 【为什么必须**大于单级隐藏掉的最大宽度**】设当前内容需求为 `need`、可用宽为
 * `avail`、本级的隐藏量为 `step`。升级条件 `need > avail` 与降级条件
 * `need - step <= avail - H` 同时成立时，就会在同一宽度下**永久来回切换**
 * （每次 setState → 重渲染 → DOM 变动 → 重新测量，自我维持，停不下来）。
 * 两者同时成立当且仅当 `step > H`。因此 `H` 必须不小于单级最大隐藏量。
 *
 * 实测各步宽度：标题 ~60、算法文本 ~30、单个药丸 ~106、平滑度滑块 ~120、
 * 数值 ~40、导入 MIDI ~70、参考轨道文本 ~60。最大是滑块 120，故取 130。
 * 被门禁的两个药丸因此**逐个**让位（各 ~106），而不是一次让两个（~212）。
 */
export const TOOLBAR_OVERFLOW_HYSTERESIS_PX = 130;

/**
 * 最大隐藏层级。
 *
 * 1 标题 · 2 算法文本 · 3 张力药丸 · 4 气声药丸（开关保留）· 5 平滑度文本 ·
 * 6 平滑度滑块 · 7 平滑度数值 · 8 导入 MIDI · 9 参考轨道组文本。
 * 再往下就没有"冗余"可砍了（剩下的都是唯一入口），到此为止。
 */
export const TOOLBAR_MAX_TIER = 9;

/** 一行工具栏的测量结果。 */
export type ToolbarRowMeasurement = {
    /** 可见宽度（`Element.clientWidth`）。 */
    available: number;
    /** 内容需求宽度（`Element.scrollWidth`）。 */
    needed: number;
};

/**
 * 计算下一隐藏层级。
 *
 * 每次调用**最多变化一级**：隐藏一级后内容需求宽度才会下降，下一轮测量再决定
 * 是否继续 —— 这样形成收敛的"隐藏 → 重测 → 还挤就再隐藏"，不会一次跳到底。
 *
 * @param currentTier 当前层级
 * @param maxTier 最大层级（通常 `TOOLBAR_MAX_TIER`）
 * @param rows 各行的测量结果；任一行溢出即视为"不够宽"
 * @param hysteresisPx 回落余量，缺省 `TOOLBAR_OVERFLOW_HYSTERESIS_PX`
 * @returns 下一层级（与 `currentTier` 相同表示保持）
 */
export function nextToolbarTier(params: {
    currentTier: number;
    maxTier: number;
    rows: readonly ToolbarRowMeasurement[];
    hysteresisPx?: number;
}): number {
    const { currentTier, maxTier, rows } = params;
    const hysteresisPx = params.hysteresisPx ?? TOOLBAR_OVERFLOW_HYSTERESIS_PX;

    // 还没有测量结果（首次渲染、或行尚未挂载）：保持不动，避免误升级。
    if (rows.length === 0) return currentTier;

    const overflowing = rows.some((row) => row.needed > row.available);
    // 回落比升级更保守：必须**每一行**都留出余量，避免"一行刚够、另一行还紧"时来回切。
    const slack = rows.every((row) => row.needed <= row.available - hysteresisPx);

    if (overflowing && currentTier < maxTier) return currentTier + 1;
    if (slack && currentTier > 0) return currentTier - 1;
    return currentTier;
}
