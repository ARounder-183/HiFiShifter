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
 * dB 的**显示**下界（REAPER 口径）。
 *
 * 【为什么是 −150】REAPER 的电平读数一路能下探到 −150 dB，低于它才显示 `-inf`。
 * 此前这里写死 −60：于是 −90 dB 的增益在界面上显示成 `-inf`，而它其实还听得见一点点，
 * 用户按 REAPER 的读数去核对时对不上。
 */
export const GAIN_DISPLAY_FLOOR_DB = -150;

/** dB 下界（= 显示下界，供"输入框 / 数值 / 提示"使用）。 */
export const GAIN_MIN_DB = GAIN_DISPLAY_FLOOR_DB;

/**
 * 旋钮**锥度**的下界。
 *
 * 【为什么与显示下界分开】旋钮角度与拖拽分辨率都按"到 0 dB 的距离"铺开。若按下界
 * −150 dB 铺，有用的 −12~+12 dB 会被压进整条弧的 4% —— 一动就飞，正是用户明确要求
 * 避免的手感崩坏。锥度仍按 −60 dB 铺满下半程，更低的取值（例如手输 −140）停在弧底，
 * 读数照常显示真实值。
 */
export const GAIN_TAPER_FLOOR_DB = -60;

/**
 * 底部的**静音棘齿**（CSS px）。
 *
 * 行程最后这 1px 直接落到 0（线性）= `-inf`，与 REAPER 推子最底下那段一致。
 *
 * 【为什么只有 1px】锥度是 `−60 dB / 48 px`（= 1.25 dB/px，与第一轮定下的手感逐像素
 * 一致）。棘齿若占 6px，下半程就只剩 42px，灵敏度变成 1.43 dB/px —— 用户明确要求过
 * "范围变大不能让旋钮变灵敏"，所以宁可让棘齿只是一格"到底"的吸附。
 */
export const GAIN_SILENCE_DETENT_PX = 1;

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

/**
 * 线性增益 → 展示文本（**唯一**的增益格式化入口）。
 *
 * 规则与 REAPER 一致：
 * - 0（静音）或低于 −150 dB ⇒ `-inf`；
 * - 其余四舍五入到一位小数，正数带 `+`；
 * - 恰好 0 dB 显示 `0`（不带符号、不带小数），免得界面上出现 `+0.0`。
 *
 * 【为什么要收口】此前仓里有**四套**格式化：轨道推子自己写 `-inf dB`、canvas 上直接
 * `toFixed(1)`（增益为 0 时打印 `-InfinitydB`）、时间轴用一个不带 −inf 分支的
 * `formatGainDbValue`、编辑框用 `formatEditNumber`。同一个值在四个地方四种写法。
 */
export function formatGainDbLabel(linear: number): string {
    const gain = clampGain(linear);
    // 线性 0 就是静音：不能借 gainToDbSafe 的"实用下界"回退（那会显示成 −150）。
    if (!(gain > 0)) return "-inf";
    return formatGainDbValue(20 * Math.log10(gain));
}

/** dB 值 → 展示文本（与 [`formatGainDbLabel`] 同一套规则，供已经是 dB 的调用点使用）。 */
export function formatGainDbValue(db: number): string {
    if (!Number.isFinite(db) || db < GAIN_DISPLAY_FLOOR_DB) return "-inf";
    const rounded = Math.round(db * 10) / 10;
    if (Math.abs(rounded) < 0.05) return "0";
    const sign = rounded > 0 ? "+" : "";
    return Number.isInteger(rounded) ? `${sign}${rounded}` : `${sign}${rounded.toFixed(1)}`;
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
    // 低于锥度下界（例如手输 −140）一律停在 −135°：读数仍是真实值，只是指针到底了。
    return (Math.max(value, -GAIN_TAPER_FLOOR_DB) / GAIN_TAPER_FLOOR_DB) * KNOB_MAX_ANGLE_DEG;
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
    // 起点低于锥度下界时按弧底算 —— 否则从 −140 往上一拉，前 80px 都在"追"回 −60。
    const startU =
        start >= 0
            ? start / GAIN_MAX_DB
            : Math.max(start, -GAIN_TAPER_FLOOR_DB) / -GAIN_TAPER_FLOOR_DB;
    const travel = startU + deltaPx / KNOB_HALF_TRAVEL_PX;
    // 棘齿：行程**最后那 1px** 直接落 0（线性）= −inf。判据用严格小于，
    // 让刚好推到边界时仍读到 −60 dB（同一个位置不能既算"到底"又算"−60"）。
    const detent = 1 - GAIN_SILENCE_DETENT_PX / KNOB_HALF_TRAVEL_PX;
    if (travel < -detent) return 0;
    const u = Math.min(1, Math.max(-1, travel));
    // 上段：u ∈ [0, 1] ⇒ db ∈ [0, GAIN_MAX_DB]。
    if (u >= 0) return clampGainDb(u * GAIN_MAX_DB);
    // 下段：u ∈ (−1, 0) ⇒ db ∈ (−GAIN_TAPER_FLOOR_DB, 0)。
    return clampGainDb(GAIN_TAPER_FLOOR_DB * -u);
}
