//! 宿主淡化轴：七个形状预设 ↔ REAPER `(curvature, S)` 的**实测**映射。
//!
//! ## 为什么需要这张表
//!
//! 官方头文件（`sdk/reaper_plugin_functions.h:2006-2015`）把两套轴标成互补区间：
//!
//! ```text
//! C_FADE*SHAPE     int, 0..6, 0=linear     v7.80 and earlier
//! D_FADE*DIR       curvature, -1..1        v7.80 and earlier
//! D_FADE*DIR_NEW   curvature, -1..1        v7.81 and later
//! D_FADE*DIR2_NEW  S parameter, -1..1      v7.81 and later
//! ```
//!
//! 并注明 7.81+ 由 `DIR_NEW`/`DIR2_NEW` 决定形状 —— 但**没有**公开 fade 求值函数，
//! 也没有说明七个预设各自对应哪一组 `(curvature, S)`。所以这张表是量出来的，不是
//! 从文档抄的：`probe/ara/FADE-AXIS-FINDINGS.md`，原始数据
//! `probe/ara/captures/fade-axis-7.82.json`（REAPER 7.82/x64，50 个采样）。
//!
//! ## 表的形状
//!
//! 七个预设落在两条**正交**的轴上：0/1/2/3/4 是纯 curvature 上的五个点，
//! 5/6 是纯 S 上的两个点。没有"两轴同时非零"的预设，因此新轴能完整表达全部
//! 七个形状 —— 新轴宿主上摆预设按钮是可行的。
//!
//! 一个可交叉验证的旁证：全新 item 的默认读数是 `SHAPE=1`、`c=0.5`、`S=0`，
//! 正好是本表的预设 1。
//!
//! ## 两个必须记住的陷阱（都来自同一份实测）
//!
//! 1. **旧 `D_FADEINDIR` 与新 `D_FADEINDIR_NEW` 不是同一套参数化。** 写旧轴读新轴
//!    得到的是被重映射过的值（旧 `±0.75` ↔ 新 `±0.6878`，旧 `±0.25` ↔ 新 `±0.1831`），
//!    反之亦然（新 `0.5` → 旧 `0`）。**绝不能把一个轴的值直接抄进另一个轴** ——
//!    曲线会变，而且是静默的。
//! 2. **`C_FADEINSHAPE` 在 7.81+ 是派生读数，不是可写状态。** 当 `(c, S)` 恰好落在
//!    本表某一行时它读回该预设号，否则读回 `-1`；而且它是**多对一的粗分类**
//!    （实测 `(0.5, -1)` 与 `(-1, -1)` 都读回 5）。所以它不能用来做往返，
//!    权威状态是 `(c, S)`。

/// 七个形状预设在新轴上的坐标 `(curvature, S)`，下标即 `C_FADE*SHAPE` 的预设号。
///
/// 数值来自 `probe/ara/captures/fade-axis-7.82.json` 的 `legacy_shape_only` 用例：
/// 只写 `C_FADEINSHAPE = k`，读回 `D_FADEINDIR_NEW` / `D_FADEINDIR2_NEW`。
/// 表里的 `0.5` 不是"精确的一半"而是宿主原样吐回的读数，故按原值保留。
pub const HOST_FADE_PRESET_AXES: [(f64, f64); 7] = [
    (0.0, 0.0),  // 0 线性
    (0.5, 0.0),  // 1 轻微凸（快起）
    (-0.5, 0.0), // 2 轻微凹（快收）
    (1.0, 0.0),  // 3 陡峭凸（快起陡）
    (-1.0, 0.0), // 4 陡峭凹（快收陡）
    (0.0, 0.5),  // 5 轻微 S（慢起慢收）
    (0.0, 1.0),  // 6 锐利 S
];

/// 预设号是否落在表内（`0..=6` 的整数）。
pub fn is_host_fade_preset(shape: f64) -> bool {
    shape.fract() == 0.0 && (0.0..HOST_FADE_PRESET_AXES.len() as f64).contains(&shape)
}

/// 形状预设 → 宿主新轴坐标。非整数或越界的形状返回 `None`。
///
/// 【为什么越界要报错而不是夹紧】`C_FADEINSHAPE` 在头文件里是 `0..6`，写进去的
/// 越界值宿主会静默变成别的形状（实测写 `5.1` 读回 `SHAPE=7`）。宁可让调用方
/// 拿到 `None` 去报错，也不要静默产生一个用户没选的形状。
pub fn host_fade_preset_axes(shape: f64) -> Option<(f64, f64)> {
    if !is_host_fade_preset(shape) {
        return None;
    }
    HOST_FADE_PRESET_AXES.get(shape as usize).copied()
}

/// 宿主新轴坐标 → 形状预设号（**精确匹配**，容差 `1e-9`）。
///
/// 只用于显示：宿主报回的 `(c, S)` 恰好是本表某一行时，界面可以说出这是哪个预设，
/// 而不是拿 HFS 自己的曲线冒充。不在表内（用户拖过曲率滑杆、或宿主自己的连续值）
/// 返回 `None` —— 那才是常态，不是错误。
pub fn host_fade_preset_for_axes(curvature: f64, s: f64) -> Option<f64> {
    if !curvature.is_finite() || !s.is_finite() {
        return None;
    }
    const EPSILON: f64 = 1e-9;
    HOST_FADE_PRESET_AXES
        .iter()
        .position(|(c, s0)| (c - curvature).abs() <= EPSILON && (s0 - s).abs() <= EPSILON)
        .map(|index| index as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 钉住实测值。改这里必须同时改 `probe/ara/captures/fade-axis-7.82.json`
    /// 与 `probe/ara/FADE-AXIS-FINDINGS.md` 的表。
    #[test]
    fn preset_table_matches_the_capture() {
        assert_eq!(
            HOST_FADE_PRESET_AXES,
            [
                (0.0, 0.0),
                (0.5, 0.0),
                (-0.5, 0.0),
                (1.0, 0.0),
                (-1.0, 0.0),
                (0.0, 0.5),
                (0.0, 1.0),
            ]
        );
    }

    /// 预设 0/1/2/3/4 只在 curvature 轴上，5/6 只在 S 轴上 —— 两轴正交，
    /// 这是"新轴能完整表达七个预设"的全部理由，值得单独钉一条。
    #[test]
    fn presets_split_cleanly_across_the_two_axes() {
        for (index, (curvature, s)) in HOST_FADE_PRESET_AXES.iter().enumerate() {
            match index {
                0..=4 => assert_eq!(*s, 0.0, "preset {index} must not use the S axis"),
                5 | 6 => assert_eq!(*curvature, 0.0, "preset {index} must not use curvature"),
                _ => unreachable!(),
            }
        }
    }

    /// 实测：全新 item 的默认读数就是预设 1（`SHAPE=1`、`c=0.5`、`S=0`）。
    #[test]
    fn the_host_default_is_preset_one() {
        assert_eq!(host_fade_preset_axes(1.0), Some((0.5, 0.0)));
        assert_eq!(host_fade_preset_for_axes(0.5, 0.0), Some(1.0));
    }

    #[test]
    fn round_trips_every_preset() {
        for shape in 0..7 {
            let (curvature, s) = host_fade_preset_axes(shape as f64).expect("preset in table");
            assert_eq!(host_fade_preset_for_axes(curvature, s), Some(shape as f64));
        }
    }

    /// 越界与小数形状一律拒绝：宿主会把它们静默变成别的形状（实测 `5.1` → `SHAPE=7`）。
    #[test]
    fn rejects_shapes_outside_the_table() {
        for shape in [-1.0, 7.0, 1.1, 5.1, 6.5, f64::NAN, f64::INFINITY] {
            assert_eq!(host_fade_preset_axes(shape), None, "shape {shape}");
        }
    }

    /// 不在表内的 `(c, S)` 是常态（用户拖过滑杆），不是错误。
    #[test]
    fn axes_outside_the_table_are_not_an_error() {
        for (curvature, s) in [(0.25, 0.0), (0.0, -1.0), (-0.5, 0.5), (0.75, 0.0)] {
            assert_eq!(host_fade_preset_for_axes(curvature, s), None);
        }
        assert_eq!(host_fade_preset_for_axes(f64::NAN, 0.0), None);
        assert_eq!(host_fade_preset_for_axes(0.0, f64::INFINITY), None);
    }

    /// 实测：`(0.5, -1)` 与 `(-1, -1)` 都被宿主读成 `SHAPE=5`，即宿主的 `SHAPE`
    /// 是**多对一粗分类**。本函数刻意不做那种分类 —— 只认表内的精确匹配。
    #[test]
    fn matching_stays_exact_rather_than_copying_the_hosts_coarse_classification() {
        assert_eq!(host_fade_preset_for_axes(0.5, -1.0), None);
        assert_eq!(host_fade_preset_for_axes(-1.0, -1.0), None);
    }
}
