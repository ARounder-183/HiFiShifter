//! REAPER 与 HiFiShifter 的声像符号转换。
//!
//! REAPER 的 PANENV/PANENV2/VOLPAN[1] 与 HiFiShifter 的 pan 曲线符号相反：
//! REAPER -1 = 右、+1 = 左；HiFiShifter -1 = 左、+1 = 右。转换只能发生在
//! 导入/导出边界；内部曲线、混音与工程文件始终保持 HiFiShifter 语义。
//!
//! 事实来源：REAPER 7.85 无头渲染实测，PANENV2 的 -1 只出右声道，
//! +1 只出左声道。

/// REAPER 声像值 → HiFiShifter 声像值。
pub(crate) fn reaper_pan_to_hifi(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (-value).clamp(-1.0, 1.0)
}

/// HiFiShifter 声像值 → REAPER 声像值。
pub(crate) fn hifi_pan_to_reaper(value: f64) -> f64 {
    if !value.is_finite() {
        return 0.0;
    }
    (-value).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pan_sign_flips_at_the_reaper_boundary() {
        assert_eq!(reaper_pan_to_hifi(-1.0), 1.0);
        assert_eq!(reaper_pan_to_hifi(1.0), -1.0);
        assert_eq!(reaper_pan_to_hifi(0.25), -0.25);
        assert_eq!(hifi_pan_to_reaper(-0.25), 0.25);
    }

    #[test]
    fn non_finite_values_fall_back_to_center() {
        assert_eq!(reaper_pan_to_hifi(f64::NAN), 0.0);
        assert_eq!(hifi_pan_to_reaper(f64::INFINITY), 0.0);
    }

    #[test]
    fn conversion_is_an_involution() {
        for v in [-1.0, -0.75, -0.1, 0.0, 0.1, 0.75, 1.0] {
            assert!((reaper_pan_to_hifi(reaper_pan_to_hifi(v)) - v).abs() < 1e-12);
        }
    }
}
