/**
 * 压感 → 位移倍率的映射曲线（纯函数，可单测）。
 *
 * 【它解决什么】数位笔的压力是 0..1 的连续量，但驱动上报的**可用区间千差万别**：
 * Wacom 常见 0..1，部分国产板到 0.15..0.9 就顶格，某些设备恒为 0。直接当 `[0,1]`
 * 做直线映射，结果是"轻划即拉满"或"怎么用力都不动"。
 *
 * 【产出的是什么】一个**正实数倍率**，由调用方**乘在这一帧的位移增量上**。
 * 绝不能拿它去重算已经累计的位移 —— 那会在抬笔的一瞬间让值与手分离并跳回，
 * 正是 `fineAxisDrag.ts:1-9` 记录的那个缺陷形状。本模块只负责"这一帧走多快"。
 *
 * 【为什么必须有自动标定】部分驱动会超过配置的物理上界（少数笔能报到 1.0 以上），
 * 不抬上界的话"再用力也不会更快"，量程的顶部被白白浪费。`observePressure` 把上界
 * 抬到"观察到的最大值 × 1.05"，于是满量程始终够得着。
 *
 * 【为什么只向上抬、不向下压】下压看着诱人（"这块板只出到 0.5，那就把 0.5 当满量程"），
 * 但它会把**轻描淡写的一笔**判成"你已经用尽全力"，于是同一段手势里值反而走得更快 ——
 * 与"轻 = 精细"的意图正好相反，而且用户完全无从察觉。上界只向上抬则永远只是
 * "让你还能更快"，不可能凭空变得比配置更灵敏。量程确实偏低的设备，出路是设置里
 * 调低 `ceiling` 或调高 `maxGain`（见 §设置），而不是让程序去猜。
 *
 * 【与 `inputProfile.ts` 的分工】那里回答"这台设备有没有压力通道"，这里回答
 * "有的话，把它换算成多大的倍率"。前者是能力，后者是手感。
 *
 * 【设计约束】纯函数 + 纯数据，不接触 DOM / React，可在 node 环境完整单测。
 * 无压力设备（鼠标 / 触摸）不会走到这里 —— 调用方按 `profile.hasPressure` 门控。
 */

/**
 * 压感映射参数。
 *
 * 默认值的目标是：**一次"正常用力"的按压落在 1.0 附近**（即与鼠标同速），
 * 轻按变细、重按变快。这样"随手一拖"不会因为设备差异而变慢，用户也不必为了
 * 恢复原手感去关掉这个功能。
 */
export interface PressureConfig {
    /**
     * 低于此值视为"未用力"（输出 `minGain`）。
     *
     * 【为什么需要死区】笔尖悬停（未接触）与极轻的接触都会上报一个小的非零压力；
     * 不设死区的话，"笔尖刚碰到板子"就已经开始改值。
     */
    deadZone: number;
    /** 物理下界（归一化起点）。 */
    floor: number;
    /**
     * 物理上界（归一化终点）。
     *
     * 这是**默认**上界；实际使用的上界由 `pressureCeiling()` 在自动标定后给出。
     */
    ceiling: number;
    /** 输出下界：即使轻触也保留这么点倍率，避免"拖不动"。 */
    minGain: number;
    /** 输出上界。 */
    maxGain: number;
    /**
     * 响应指数：`> 1` 让轻压段更细腻。
     *
     * 【为什么默认大于 1】自然的发力分布本来就偏轻（人手很难长时间稳定压在
     * 满量程上），指数大于 1 把分辨率多分给轻压段，正好对应"想微调时会不自觉地
     * 放轻"这一直觉。
     */
    gamma: number;
}

/** 出厂默认（与后端默认值对齐；见 `settings.ts` 的 `DEFAULT_PEN_INPUT_SETTINGS`）。 */
export const DEFAULT_PRESSURE_CONFIG: PressureConfig = {
    deadZone: 0.06,
    floor: 0.05,
    ceiling: 0.9,
    minGain: 0.25,
    maxGain: 1.6,
    gamma: 1.6,
};

/**
 * 一次手势内的压感标定状态。
 *
 * 【为什么按手势而不是全局】驱动的量程可能因笔 / 板子而异，也可能因握姿而变；
 * 每段手势重新观察，既不会把上一次的极端值长期固化，也不需要持久化。
 */
export interface PressureCalibration {
    /** 本段手势观察到的最大原始压力。 */
    observedMax: number;
}

/** 新建一份标定状态（每段手势一份）。 */
export function createPressureCalibration(): PressureCalibration {
    return { observedMax: 0 };
}

/**
 * 记录一次原始压力采样，必要时抬升观察到的最大值。就地更新。
 *
 * 非有限值忽略 —— 某些驱动在按下瞬间会上报 `NaN` / 负值。
 */
export function observePressure(calibration: PressureCalibration, raw: number): void {
    if (!Number.isFinite(raw) || raw <= 0) return;
    if (raw > calibration.observedMax) calibration.observedMax = raw;
}

/**
 * 本帧实际使用的物理上界。
 *
 * 取"配置上界"与"观察到上界 × 1.05"的较大者。乘 1.05 是留一点余量，让用户
 * 稍微再加力时仍能继续加速，而不是一碰就顶格。
 */
export function pressureCeiling(
    calibration: PressureCalibration,
    config: PressureConfig = DEFAULT_PRESSURE_CONFIG,
): number {
    const observed = calibration.observedMax * 1.05;
    const configured = Number.isFinite(config.ceiling)
        ? config.ceiling
        : DEFAULT_PRESSURE_CONFIG.ceiling;
    return Math.max(configured, observed);
}

function clamp01(value: number): number {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

/**
 * 把一次原始压力采样映射成**本帧的位移倍率**。
 *
 * 流程：扣死区 → 按 `deadZone..ceiling` 归一化 → 施加 `gamma` → 映射到
 * `minGain..maxGain`。
 *
 * 【为什么归一化用 `deadZone` 而不是 `floor`】`floor` 是"驱动的物理下界"这一
 * 概念上的说法，而真正决定"用户是否在用力"的是死区；两者都保留是为了让设置项
 * 的语义各自清晰，映射本身只用死区与上界，少一个自由度。
 *
 * @param raw 原始压力（`PointerEvent.pressure`）。
 * @param config 映射参数。
 * @param calibration 本段手势的标定状态；省略时只用配置上界。
 */
export function pressureToGain(
    raw: number,
    config: PressureConfig = DEFAULT_PRESSURE_CONFIG,
    calibration?: PressureCalibration,
): number {
    const deadZone = Number.isFinite(config.deadZone)
        ? config.deadZone
        : DEFAULT_PRESSURE_CONFIG.deadZone;
    const minGain = Number.isFinite(config.minGain)
        ? config.minGain
        : DEFAULT_PRESSURE_CONFIG.minGain;
    const maxGain = Number.isFinite(config.maxGain)
        ? config.maxGain
        : DEFAULT_PRESSURE_CONFIG.maxGain;
    const gamma =
        Number.isFinite(config.gamma) && config.gamma > 0
            ? config.gamma
            : DEFAULT_PRESSURE_CONFIG.gamma;
    const ceiling = calibration
        ? pressureCeiling(calibration, config)
        : Number.isFinite(config.ceiling)
          ? config.ceiling
          : DEFAULT_PRESSURE_CONFIG.ceiling;

    if (!Number.isFinite(raw)) return 1;
    // 上界必须明显高于死区，否则除零 / 负跨度。
    const span = ceiling - deadZone;
    if (!(span > 1e-6)) return maxGain;

    const normalized = clamp01((raw - deadZone) / span);
    const curved = Math.pow(normalized, gamma);
    return minGain + curved * (maxGain - minGain);
}

/**
 * 手绘笔画的最小写入权重。
 *
 * 【为什么不能是 0】权重 0 意味着"这一笔什么也没写" —— 用户看着笔尖划过画布却
 * 毫无变化，会以为功能坏了。给一个小的非零下界，轻描淡写也能留下痕迹（可以
 * 反复涂叠），只是推进得慢。
 */
export const PAINT_MIN_WEIGHT = 0.15;

/**
 * 把一次原始压力采样映射成**手绘笔画这一点的写入权重**（`PAINT_MIN_WEIGHT..1`）。
 *
 * 【它与 `pressureToGain` 的区别】那是"这一帧走多快"（拖拽），这是"这一点写多深"
 * （画笔）。两者的量纲完全不同：前者会累加，后者是**单点的混合系数** ——
 * `next = current + (target - current) * weight`。权重 1 表示完全覆盖（鼠标的既有
 * 行为），越小则越像"轻涂一层"。
 *
 * 归一化口径与 `pressureToGain` 共用（同一套死区 / 上界 / 标定），只是输出区间
 * 换成 `[PAINT_MIN_WEIGHT, 1]` —— 于是"多用力 = 多写一点"在两种手势里是同一个
 * 肌肉记忆。
 */
export function pressureToWeight(
    raw: number,
    config: PressureConfig = DEFAULT_PRESSURE_CONFIG,
    calibration?: PressureCalibration,
): number {
    const deadZone = Number.isFinite(config.deadZone)
        ? config.deadZone
        : DEFAULT_PRESSURE_CONFIG.deadZone;
    const gamma =
        Number.isFinite(config.gamma) && config.gamma > 0
            ? config.gamma
            : DEFAULT_PRESSURE_CONFIG.gamma;
    const ceiling = calibration
        ? pressureCeiling(calibration, config)
        : Number.isFinite(config.ceiling)
          ? config.ceiling
          : DEFAULT_PRESSURE_CONFIG.ceiling;

    if (!Number.isFinite(raw)) return 1;
    const span = ceiling - deadZone;
    if (!(span > 1e-6)) return 1;

    const normalized = clamp01((raw - deadZone) / span);
    const curved = Math.pow(normalized, gamma);
    return PAINT_MIN_WEIGHT + curved * (1 - PAINT_MIN_WEIGHT);
}

/**
 * 一段手势里压感的极值分布（增量维护，O(1) 每次采样）。
 *
 * 【为什么不用数组】手势可能持续十几秒、采样 250Hz，那就是几千个样本；每次判定
 * 都扫一遍整段序列是 O(n²)。这里只记极值与个数，判定等价而开销恒定。
 */
export interface PressureSpread {
    count: number;
    min: number;
    max: number;
}

/** 新建一份分布。 */
export function createPressureSpread(): PressureSpread {
    return { count: 0, min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY };
}

/** 记入一个采样（非有限值忽略）。 */
export function observePressureSpread(spread: PressureSpread, raw: number): void {
    if (!Number.isFinite(raw)) return;
    spread.count += 1;
    if (raw < spread.min) spread.min = raw;
    if (raw > spread.max) spread.max = raw;
}

/** 该分布是否"毫无变化"（样本不足 2 个时按"没有变化"处理）。 */
export function spreadLooksConstant(spread: PressureSpread, epsilon = 1e-3): boolean {
    if (spread.count < 2) return true;
    if (!Number.isFinite(spread.min) || !Number.isFinite(spread.max)) return true;
    return spread.max - spread.min <= epsilon;
}

/**
 * 一段手势的压感采样是否"毫无变化"。
 *
 * 【为什么需要】某些设备 / 驱动会上报恒定压力（恒 0、恒 1、或恒某个常数）。
 * 此时压感映射会把整段手势的倍率钉死在某个非 1 的值上 —— 用户会觉得"这次拖得
 * 莫名其妙地慢 / 快"，而且完全不知道为什么。判定为常数后调用方按 `1` 处理，
 * 等价于该设备没有压感通道。
 *
 * 需要逐次判定时请改用 `createPressureSpread` / `observePressureSpread` /
 * `spreadLooksConstant`：本函数会遍历整段序列，只适合一次性查询。
 *
 * @param samples 本段手势观察到的原始压力序列。
 * @param epsilon 视为"没有变化"的阈值。
 */
export function pressureLooksConstant(samples: readonly number[], epsilon = 1e-3): boolean {
    const spread = createPressureSpread();
    for (const sample of samples) observePressureSpread(spread, sample);
    return spreadLooksConstant(spread, epsilon);
}
