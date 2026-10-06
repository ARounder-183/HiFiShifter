//! HiFiShifter自有渐变形状；长度仍由REAPER item管理，音频仅在ARA明确委托的边界处理。
use serde::{Serialize,Deserialize};

#[derive(Debug,Clone,PartialEq,Serialize,Deserialize)]
pub(crate) struct FadeStyle {pub in_shape:f64,pub out_shape:f64,pub in_dir:f64,pub out_dir:f64}
impl Default for FadeStyle {fn default()->Self {Self {in_shape:1.,out_shape:1.,in_dir:0.,out_dir:0.}}}
impl FadeStyle {
    /// 曲率和预设沿原内核定义校验；不把任意NaN/未知大数保存到宿主组件state。
    pub fn validate(&self)->Result<(),String> {
        if [self.in_shape,self.out_shape].iter().any(|value|!value.is_finite()||!(0.0..7.0).contains(value))
            ||[self.in_dir,self.out_dir].iter().any(|value|!value.is_finite()||!(-1.0..=1.0).contains(value)) {return Err("invalid HiFiShifter fade shape/curvature".into());}Ok(())
    }
}
