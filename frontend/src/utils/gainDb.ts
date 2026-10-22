/**
 * Clip / 轨道增益的 dB 值域，以及旋钮拖拽的**手感映射**。
 *
 * 【值域与后端一一对应】`hifishifter_kernel::gain` 是权威（线性 `[0, 16]`），
 * 这里给出它的 dB 表达。两边必须一致 —— 与 `paramRanges.ts` / `DYN_SILENCE_FLOOR`
 * 的既有约定相同：前端需要**同步**读取（旋钮角度、徽标、行内输入都在渲染期算），
 * 不可能为一次换算去等 IPC。
 *
 * 【为什么上界是 +24.1 而不是 +24.08】REAPER 显示 `+24.08` 是 dB 的四舍五入；
 * 线性域取整为 16，换算回来是 `20*log10(16) = 24.082`。这里的常量按线性上界反算，
 * 保证"推到顶"恰好等于后端允许的最大值，而不是差 0.02 dB 被钳一下。
 */

/** 线性增益下界（0 = −inf dB，静音）。 */
export const GAIN_MIN_LINEAR = 0;

/** 线性增益上界（16 = +24.08 dB，与 REAPER 的 item 增益上限一致）。 */
export const GAIN_MAX_LINEAR = 16;

/**
 * dB 显示的**实用**下界。
 *
 * 【为什么不是 −Infinity】−inf 无法参与 `+delta` 运算、也无法映射成旋钮角度。
 * −60 dB（线性 0.001）是"听不见"的实用表达，再往下对听感没有区别；而
 * `formatGainDb` 在接近它时会显示成 `-inf dB`，与 REAPER 的写法一致。
 */
export const GAIN_MIN_DB = -60;

/** dB 上界（由线性上界反算，见文件头）。 */
export const GAIN_MAX_DB = 20 * Math.log10(GAIN_MAX_LINEAR);

/** 旋钮从 0 dB 推到任一端的**行程**（CSS px）。 */
const KNOB_HALF_TRAVEL_PX = 48;

/** 旋钮指示线在满量程处的偏转角（度）。 */
const KNOB_MAX_ANGLE_DEG = 135;

/**
 * 把线性增益钳进值域（与后端 `clamp_gain` 逐条同构）。
 *
 * 只有 NaN 特判为 1.0；±Infinity 由 min/max 自然钳到端点（与 Rust 的
 * `f32::clamp` 行为一致）。
 */
export function clampGain(linear: number): number {
    if (Number.isNaN(linear)) return 1;
    return Math.min(GAIN_MAX_LINEAR, Math.max(GAIN_MIN_LINEAR, linear));
}

/** 把 dB 值钳进值域。 */
export function clampGainDb(db: number): number {
    if (Number.isNaN(db)) return 0;
    return Math.min(GAIN_MAX_DB, Math.max(GAIN_MIN_DB, db));
}

/** 线性 → dB（`0` 映射到 [`GAIN_MIN_DB`]，不返回 −Infinity）。 */
export function gainToDbSafe(linear: number): number {
    const g = clampGain(linear);
    if (!(g > 0)) return GAIN_MIN_DB;
    return clampGainDb(20 * Math.log10(g));
}

/** dB → 线性。 */
export function dbToGainSafe(db: number): number {
    return clampGain(Math.pow(10, clampGainDb(db) / 20));
}

/**
 * dB 值 → 旋钮指示线的偏转角（度，0 = 正上方）。
 *
 * 【为什么分两段】旧实现是 `(db / 12) * 135` —— 靠"值域 ±12 对称"让 0 dB 落在正中。
 * 新值域 `(−60, +24.1)` **不对称**，若仍用一个除数，0 dB 会偏到一边。分段之后
 * 0 dB 恒在正中（与旧观感一致），两侧各自铺满自己的半程。
 */
export function gainKnobAngleDeg(db: number): number {
    const value = clampGainDb(db);
    if (value >= 0) return (value / GAIN_MAX_DB) * KNOB_MAX_ANGLE_DEG;
    return (value / -GAIN_MIN_DB) * KNOB_MAX_ANGLE_DEG;
}

/**
 * 旋钮拖拽：竖直位移 → 新的 dB 值。
 *
 * 【为什么不能用固定的 dB/px】值域从 ±12 扩到 (−60, +24.1) 之后，
 * 固定 `0.25 dB/px` 会让拖 145 px 就走完全程 —— 手感直接崩掉。用户明确要求：
 * **范围扩大不得让旋钮变得"一动就大幅调整"**。
 *
 * 【映射】先在**归一化行程** `u ∈ [−1, 1]` 上走（与值域大小无关）：
 *
 * ```text
 * u = clamp(startU + deltaPx / KNOB_HALF_TRAVEL_PX, −1, 1)
 * u < 0  ⇒  db = lerp(GAIN_MIN_DB, 0, u + 1)
 * u >= 0 ⇒  db = lerp(0, GAIN_MAX_DB, u)
 * ```
 *
 * 于是**全程行程恒为 96 px**（两段各 48），与值域大小无关 —— 与旧实现
 * （±12 dB / 0.25 dB·px⁻¹ ⇒ 96 px 走完全程）**逐像素一致**。
 *
 * 【为什么不是纯 dB/px】−inf 端没有下界可走；而且 dB 域在靠近 0 的地方非线性，
 * 纯 dB/px 会让"底部走得慢、顶部走得快"。归一化行程把两端都变成线性。
 *
 * @param startDb 按下时的 dB 值。
 * @param deltaPx 相对按下点的竖直位移（**向上为正**）。
 */
export function gainDbFromKnobTravel(startDb: number, deltaPx: number): number {
    if (!Number.isFinite(deltaPx)) return clampGainDb(startDb);
    const start = clampGainDb(startDb);
    // 起点在归一化行程上的位置（两段各自线性，见上方映射）。
    const startU = start >= 0 ? start / GAIN_MAX_DB : start / -GAIN_MIN_DB;
    const u = Math.min(1, Math.max(-1, startU + deltaPx / KNOB_HALF_TRAVEL_PX));
    // 上段：u ∈ [0, 1] ⇒ db ∈ [0, GAIN_MAX_DB]。
    if (u >= 0) return clampGainDb(u * GAIN_MAX_DB);
    // 下段：u ∈ [−1, 0) ⇒ db ∈ [GAIN_MIN_DB, 0)。u = −1 时必须恰好落在 GAIN_MIN_DB。
    return clampGainDb(GAIN_MIN_DB * -u);
}
