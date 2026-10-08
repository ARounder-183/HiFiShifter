/**
 * 宿主淡化轴：七个形状预设 ↔ REAPER `(curvature, S)` 的**实测**映射。
 *
 * 与 Rust 侧 `backend/hifishifter-kernel/src/fade_axes.rs` 是同一张表，两边各有一条
 * 钉住同一组数值的测试（`hostFadeAxes.test.ts` ↔ `fade_axes.rs::preset_table_matches_the_capture`）。
 * 改任一侧必须同步另一侧。
 *
 * ## 为什么是量出来的
 *
 * 官方头文件（`sdk/reaper_plugin_functions.h:2006-2015`）把两套轴标成互补区间
 * （`C_FADE*SHAPE`/`D_FADE*DIR` 是 v7.80 and earlier，`D_FADE*DIR_NEW`/
 * `D_FADE*DIR2_NEW` 是 v7.81 and later），并注明 7.81+ 由后两个轴决定形状 ——
 * 但**没有**公开 fade 求值函数，也没说七个预设各自对应哪一组坐标。
 * 证据：`probe/ara/FADE-AXIS-FINDINGS.md`，原始数据
 * `probe/ara/captures/fade-axis-7.82.json`（REAPER 7.82/x64，50 个采样）。
 *
 * 表里的七个点落在两条**正交**的轴上：0/1/2/3/4 只用 curvature，5/6 只用 S。
 */

/**
 * 七个预设在新轴上的坐标 `[curvature, S]`，下标即 `C_FADE*SHAPE` 的预设号。
 *
 * 数值是宿主原样吐回的读数（例如 `0.5` 就是 `0.5`），不是推算出来的近似值，
 * 所以按原值保留、不做任何"看起来更整齐"的改写。
 */
export const HOST_FADE_PRESET_AXES: ReadonlyArray<readonly [number, number]> = [
    [0, 0], // 0 线性
    [0.5, 0], // 1 轻微凸（快起）
    [-0.5, 0], // 2 轻微凹（快收）
    [1, 0], // 3 陡峭凸（快起陡）
    [-1, 0], // 4 陡峭凹（快收陡）
    [0, 0.5], // 5 轻微 S（慢起慢收）
    [0, 1], // 6 锐利 S
];

/** 匹配容差：宿主的读数是原样回吐的十进制值，`1e-9` 足够吸收浮点噪声。 */
const PRESET_AXIS_EPSILON = 1e-9;

/** 形状预设 → 宿主新轴坐标；非整数或越界返回 `null`。 */
export function hostFadePresetAxes(shape: number): readonly [number, number] | null {
    if (!Number.isInteger(shape) || shape < 0 || shape >= HOST_FADE_PRESET_AXES.length) return null;
    return HOST_FADE_PRESET_AXES[shape];
}

/**
 * 宿主新轴坐标 → 形状预设号（**精确匹配**）。
 *
 * 只用于显示：宿主报回的 `(c, S)` 恰好是表里某一行时，界面就能说出这是哪个预设，
 * 并画出该预设的曲线，而不是拿一段临时混合的示意曲线冒充。不在表内（用户拖过
 * 曲率滑杆、或宿主自己的连续值）返回 `null` —— 那是常态，不是错误。
 *
 * 刻意**不做**宿主那种多对一粗分类：实测 `(0.5, -1)` 与 `(-1, -1)` 都被宿主读成
 * `SHAPE=5`，若照着分类，一个用户自己拖出来的曲线会被误报成预设。
 */
export function hostFadePresetForAxes(curvature: number, s: number): number | null {
    if (!Number.isFinite(curvature) || !Number.isFinite(s)) return null;
    for (let index = 0; index < HOST_FADE_PRESET_AXES.length; index += 1) {
        const [presetCurvature, presetS] = HOST_FADE_PRESET_AXES[index];
        if (
            Math.abs(presetCurvature - curvature) <= PRESET_AXIS_EPSILON &&
            Math.abs(presetS - s) <= PRESET_AXIS_EPSILON
        )
            return index;
    }
    return null;
}
