/**
 * 参数编辑器工具栏「水平不足 ⇒ 分级隐藏」的判据（纯逻辑，便于单测）。
 *
 * # 为什么需要
 * 工具栏横向不足时，若放任浏览器自行压缩，中文标签会被压到**逐字折行**
 * （`参数编辑器` 变成两行「参数编 / 辑器」）—— 这是最丑的一种降级。正确做法是
 * 按优先级**离散地隐藏**冗余项，直到放得下。
 *
 * # 判据为什么是对称的
 * 隐藏到"刚好放得下"为止，恢复也到"刚好放得下"为止：同一宽度下，
 * 压缩过程与恢复过程得到**同一套**排版。若改用"滞后余量"（进入用阈值 A、
 * 退出用 A - H），恢复会比压缩晚一截，用户能观察到两条路径排版不一致。
 *
 * 不用余量也不会来回抖动：恢复的判据用的是**实测过的"少隐藏一级"时的需求宽度**
 * （精确值），而不是估算。放不下就多藏一级，少藏一级也放得下就恢复一级，
 * 两者在"刚好放得下"处同时失效 —— 是个稳定不动点。
 *
 * # 层级约定
 * `0` = 全部显示；数值越大隐藏越多，具体对应见 `PianoRollPanel` 的使用点。
 */

/**
 * 最大隐藏层级。
 *
 * 1 标题 · 2 算法文本 · 3 张力药丸 · 4 气声药丸（开关保留）· 5 平滑度文本 ·
 * 6 平滑度数值 · 7 平滑度滑块 · 8 导入 MIDI · 9 参考轨道组文本。
 *
 * 顺序说明：**数值在滑块之前**让位 —— 滑块是控件、数值只是读数，读数先走才合理。
 * 两个被门禁的药丸**逐个**让位，且**先张力后气声**：气声药丸左侧挂着分离开关，
 * 先让气声会让开关独自悬空。
 *
 * 再往下就没有"冗余"可砍了（剩下的都是唯一入口），到此为止。
 */
export const TOOLBAR_MAX_TIER = 9;

/**
 * 计算下一隐藏层级。
 *
 * 每次调用**最多变化一级**：隐藏一级后内容才会变窄，下一轮测量再决定是否继续，
 * 形成收敛的"隐藏 → 重测 → 还挤就再隐藏"。
 *
 * @param currentTier 当前层级
 * @param maxTier 最大层级（通常 `TOOLBAR_MAX_TIER`）
 * @param available 工具栏可见宽度
 * @param needed 当前层级下内容的自然宽度
 * @param neededAtLowerTier 上一次处于"少隐藏一级"（`currentTier - 1`）时量到的
 *        内容自然宽度；尚未测过时传 `undefined` —— 此时**不恢复**，保守停在当前级
 * @returns 下一层级（与 `currentTier` 相同表示保持）
 */
export function nextToolbarTier(params: {
    currentTier: number;
    maxTier: number;
    available: number;
    needed: number;
    neededAtLowerTier?: number;
}): number {
    const { currentTier, maxTier, available, needed, neededAtLowerTier } = params;

    // 放不下：多隐藏一级（已到最大级则保持，此时确实放不下，由 overflow 裁掉）。
    if (needed > available && currentTier < maxTier) return currentTier + 1;

    // 少隐藏一级也放得下：恢复一级。判据用的是**实测**的上一级需求宽度，
    // 因此同一宽度下与"压缩"路径得到同一结果，且不会来回抖动。
    if (currentTier > 0 && neededAtLowerTier !== undefined && neededAtLowerTier <= available) {
        return currentTier - 1;
    }

    return currentTier;
}
