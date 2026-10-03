/**
 * 不规则度的确定性种子。
 *
 * 【为什么需要它】不规则度用值噪声抖动相位与深度（见 `vibratoCurve.ts`），
 * 而噪声必须是**确定性**的：预览与提交要逐帧一致，否则松手瞬间波形会跳。
 *
 * 【为什么是预设字段】种子曾经由预设 id 派生 —— 用户只能接受预设"摇成什么样"。
 * 成为 `VibratoPreset.seed` 字段后，管理器里的骰子按钮就能换图案；而它仍是
 * 确定的数字，同一预设的每次渲染 / 试听 / 提交逐帧一致。
 *
 * 【为什么不用随机数】`Math.random()` 会让同一次拖拽的每一帧都换一个图案
 * （预览闪烁），也会让"预览与提交一致"这条约束无法成立。
 */

/** 种子的合法区间（与 `VIBRATO_LIMITS.seed` 同源）。 */
export const VIBRATO_SEED_MAX = 99999;

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
 * 由预设取一个稳定的种子。
 *
 * 优先读字段；字段缺失或非有限时回落到按 id 派生 —— 旧配置里没有 `seed`，
 * 回落到 id 哈希既保证它们仍能加载，又保持"每个预设各有各的随机味"。
 */
export function vibratoSeedForPreset(preset: { id: string; seed?: number }): number {
    if (Number.isFinite(preset.seed)) {
        const value = Math.round(preset.seed as number);
        return Math.min(VIBRATO_SEED_MAX, Math.max(0, value));
    }
    return hashString(preset.id) % (VIBRATO_SEED_MAX + 1);
}

/** 掷一次骰子：一个 `0..VIBRATO_SEED_MAX` 的随机种子。 */
export function randomVibratoSeed(): number {
    return Math.floor(Math.random() * (VIBRATO_SEED_MAX + 1));
}
