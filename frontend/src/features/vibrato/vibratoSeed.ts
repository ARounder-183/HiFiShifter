/**
 * 不规则度的确定性种子。
 *
 * 【为什么需要它】不规则度用值噪声抖动相位与深度（见 `vibratoCurve.ts`），
 * 而噪声必须是**确定性**的：预览与提交要逐帧一致，否则松手瞬间波形会跳。
 *
 * 【为什么按预设取种子】同一个种子会让两个不同预设画出**完全相同**的抖动
 * 图案 —— 用户切换"摇曳"与"自然"时，听到的只是幅度差异而抖动位置一样，预设
 * 之间的区别被削弱。把预设 id 混进种子，每个预设就有自己的"随机味"，而同一
 * 预设的每次调用仍然稳定。
 *
 * 【为什么不用随机数】`Math.random()` 会让同一次拖拽的每一帧都换一个图案
 * （预览闪烁），也会让"预览与提交一致"这条约束无法成立。
 */

/** 整数哈希：字符串 → 32 位无符号整数。与 `vibratoCurve.ts` 的哈希同族，避免引入依赖。 */
function hashString(value: string): number {
    let hash = 0x811c9dc5;
    for (let i = 0; i < value.length; i += 1) {
        hash ^= value.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
    }
    return hash >>> 0;
}

/**
 * 由预设派生一个稳定的种子。
 *
 * 只取 id：预设的参数在编辑过程中会变，若把参数也混进种子，用户每改一次深度
 * 抖动图案就重排一次，听起来像换了个预设。
 */
export function vibratoSeedForPreset(preset: { id: string }): number {
    return hashString(preset.id) % 100000;
}
