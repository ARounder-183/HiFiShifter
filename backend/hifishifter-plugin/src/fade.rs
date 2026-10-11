//! HiFiShifter自有渐变形状；长度仍由REAPER item管理，音频仅在ARA明确委托的边界处理。
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct FadeStyle {
    pub in_shape: f64,
    pub out_shape: f64,
    pub in_dir: f64,
    pub out_dir: f64,
}
impl Default for FadeStyle {
    fn default() -> Self {
        Self {
            in_shape: 1.,
            out_shape: 1.,
            in_dir: 0.,
            out_dir: 0.,
        }
    }
}
impl FadeStyle {
    /// 曲率和预设沿原内核定义校验；不把任意NaN/未知大数保存到宿主组件state。
    ///
    /// 【为什么形状用 `is_supported_fade_shape` 而不是 `0.0..7.0`】后者把 `6.5` 当合法，
    /// 而 `resolve_shape` 对 `0.5`/`3.7` 这类值会**静默退化为线性** —— 存得进去、读回来
    /// 却变了样。这里与内核曲线族的接受域逐字对齐（整数 `0..=6` 或官方小数变体）。
    pub fn validate(&self) -> Result<(), String> {
        if [self.in_shape, self.out_shape]
            .iter()
            .any(|value| !hifishifter_kernel::fade_curves::is_supported_fade_shape(*value))
            || [self.in_dir, self.out_dir]
                .iter()
                .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err("invalid HiFiShifter fade shape/curvature".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::FadeStyle;

    /// 默认值必须自校验通过 —— 它是新片段/缺字段时的落点。
    #[test]
    fn the_default_style_validates() {
        assert!(FadeStyle::default().validate().is_ok());
    }

    /// 形状只在**内核曲线族**的接受域内合法：整数 `0..=6`，或底数为 `1` / `5` / `6`
    /// 的官方小数变体（如 `1.5` / `6.9`）。`0.5` / `2.5` / `3.7` 这类既不是整数预设、
    /// 底数又不在 `{1,5,6}` 的值会被 `resolve_shape` **静默退化为线性** —— 存得进去、
    /// 读回来却变了样，因此必须在这里拒掉。
    #[test]
    fn unsupported_shapes_are_rejected_instead_of_silently_degrading() {
        for shape in [0.5, 2.5, 3.7, 7.0, -1.0, f64::NAN, f64::INFINITY] {
            let style = FadeStyle {
                in_shape: shape,
                ..FadeStyle::default()
            };
            assert!(style.validate().is_err(), "shape {shape} must be rejected");
            let style = FadeStyle {
                out_shape: shape,
                ..FadeStyle::default()
            };
            assert!(style.validate().is_err(), "shape {shape} must be rejected");
        }
    }

    /// 整数预设与官方小数变体（底数 1 / 5 / 6）都合法。
    ///
    /// 【注意与写回路径的差别】`host_edit::validate_fade_axes` 用的是
    /// `host_fade_preset_axes` —— 它只认**能映射到预设**的形状，因此 `6.5` 在写回时
    /// 会被拒。这里存的是曲线族成员，两者口径不同是刻意的：能存下来 ≠ 每个宿主都能写。
    #[test]
    fn integer_presets_and_official_variants_are_accepted() {
        for shape in [0.0, 1.0, 3.0, 6.0, 1.5, 5.2, 6.5, 6.9] {
            let style = FadeStyle {
                in_shape: shape,
                out_shape: shape,
                ..FadeStyle::default()
            };
            assert!(style.validate().is_ok(), "shape {shape} must be accepted");
        }
    }

    /// 曲率必须在 [-1, 1] 且有限。
    #[test]
    fn curvature_outside_the_unit_range_is_rejected() {
        for dir in [-1.5, 1.5, f64::NAN, f64::INFINITY] {
            let style = FadeStyle {
                in_dir: dir,
                ..FadeStyle::default()
            };
            assert!(style.validate().is_err(), "dir {dir} must be rejected");
            let style = FadeStyle {
                out_dir: dir,
                ..FadeStyle::default()
            };
            assert!(style.validate().is_err(), "dir {dir} must be rejected");
        }
        // 端点本身合法。
        assert!(FadeStyle {
            in_dir: -1.0,
            out_dir: 1.0,
            ..FadeStyle::default()
        }
        .validate()
        .is_ok());
    }
}
