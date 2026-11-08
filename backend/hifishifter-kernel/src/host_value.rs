//! 宿主报来的两个浮点数是不是**同一个量** —— 全仓库唯一一份判据。
//!
//! 【为什么需要它】宿主读数几乎都是 f64，而模型里同一个量常常是 f32
//! （`playback_rate` / 增益），再叠加"按采样量化的秒"与"由秒反算的采样数"之间的
//! 往返，同一个量读回来会带 ~1e-7 的相对误差。用纯绝对阈值判"变没变"（哪怕
//! 1e-9），这种正常舍入就会被判成一次真实编辑 —— 而判据偏严的后果不是"少刷新
//! 一次"，是**把一次编辑当成另一次编辑**：分割被认成拉伸，于是整条参数线被压进
//! 半段之内（见 `editor::param_geometry::ClipGeometry::rate`）。
//!
//! 【为什么是相对 + 绝对两项】相对项吸收大数值上的浮点往返（长工程里的
//! `start_sec` 到 1e4 秒时，1e-7 的相对误差就是 1 ms 量级）；绝对项兜住接近 0 的
//! 量（那里没有相对项可用）。1e-6 秒在 48 kHz 上不到 1/20 个采样点 —— 低于这个
//! 尺度的"变化"不可能是用户做出来的编辑。
//!
//! 【唯一性】插件侧的绑定、分割规划、写回前置检查、剪贴板、完成门与几何分类器
//! 都调用这一个函数（`host::geometry::host_value_compatible` 只是它的别名）。
//! 两处各写一份阈值就会造出"规划接受、绑定拒绝"的窗口 —— 用户看到的是
//! "这个编辑偶尔被拒"。
pub fn compatible(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + 1e-9 * a.abs().max(b.abs())
}

#[cfg(test)]
mod tests {
    use super::compatible;

    #[test]
    fn identical_values_agree() {
        assert!(compatible(0.0, 0.0));
        assert!(compatible(-1.25, -1.25));
        assert!(compatible(4.0, 4.0));
    }

    /// f32 往返是**常态**：`playback_rate` 在模型里是 f32，与宿主 f64 比必然有
    /// ~1e-7 的相对误差。这一档必须判为"同一个量"，否则每次改速率都会撞上
    /// "宿主在提交前改了这个 clip"。
    #[test]
    fn an_f32_round_trip_still_agrees() {
        for rate in [0.25_f32, 0.5, 1.0, 1.5, 2.0, 4.0] {
            let projected = rate as f64;
            let host = (rate as f64) * (1.0 + 1e-7);
            assert!(compatible(projected, host), "f32 round-trip of {rate}");
        }
    }

    /// 真实变化必须被看见：1e-4 秒在 44.1 kHz 上约 4 个采样点。
    #[test]
    fn a_real_change_is_rejected() {
        assert!(!compatible(0.0, 1e-4));
        assert!(!compatible(1.0, 1.001));
        assert!(!compatible(48000.0, 48000.5));
    }

    /// 接近 0 的量没有相对项可用，由绝对项兜底。
    #[test]
    fn the_absolute_floor_covers_values_near_zero() {
        assert!(compatible(0.0, 5e-7));
        assert!(!compatible(0.0, 2e-6));
    }

    #[test]
    fn non_finite_values_never_agree() {
        assert!(!compatible(f64::NAN, f64::NAN));
        assert!(!compatible(f64::INFINITY, f64::INFINITY));
        assert!(!compatible(0.0, f64::NAN));
        assert!(!compatible(f64::INFINITY, 1.0));
    }
}
