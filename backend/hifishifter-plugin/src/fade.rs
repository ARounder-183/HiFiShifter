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
