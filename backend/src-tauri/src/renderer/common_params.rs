//! 所有合成算法共通的混音级参数描述符。
//!
//! 音量（volume）与声像（pan）不依赖具体声码器内核：
//! - World / NSF-HiFiGAN 路径在音频引擎 mix 阶段实时应用这些曲线，
//!   因此未开启 Compose、或音高曲线没有变化时也立即生效；
//! - vslib 路径由于音量/声像是通过 vslib 控制点写入、在合成阶段烘焙进输出，
//!   仍由 `VslibProcessor` 消费同一组曲线（未开启 Compose 时按约定不生效，
//!   且 mix 阶段跳过，避免二次应用）。

use super::traits::{ParamDescriptor, ParamKind};

pub(crate) const VOLUME_PARAM_ID: &str = "volume";
pub(crate) const PAN_PARAM_ID: &str = "pan";
/// 旧 NSF-HiFiGAN 专有参数名，工程加载时迁移到 `volume`。
pub(crate) const LEGACY_HIFIGAN_VOLUME_PARAM_ID: &str = "hifigan_volume";

/// 共通音量参数（所有算法返回同一个描述符，保证曲线在算法间切换时完全互通）。
pub(crate) const VOLUME_PARAM: ParamDescriptor = ParamDescriptor {
    id: VOLUME_PARAM_ID,
    display_name: "Volume",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "×",
        default_value: 1.0,
        min_value: 0.0,
        max_value: 4.0,
    },
};

/// 共通声像参数（所有算法返回同一个描述符，-1 = 全左，1 = 全右）。
pub(crate) const PAN_PARAM: ParamDescriptor = ParamDescriptor {
    id: PAN_PARAM_ID,
    display_name: "Pan",
    group: "Mix",
    kind: ParamKind::AutomationCurve {
        unit: "",
        default_value: 0.0,
        min_value: -1.0,
        max_value: 1.0,
    },
};

/// 所有算法都会暴露的混音级参数。
pub(crate) static COMMON_MIX_PARAMS: [ParamDescriptor; 2] = [VOLUME_PARAM, PAN_PARAM];

/// 参数是否为混音阶段应用的共通音量/声像（含旧 nsf 专有名）。
pub(crate) fn is_common_mix_param(param_id: &str) -> bool {
    matches!(
        param_id,
        VOLUME_PARAM_ID | PAN_PARAM_ID | LEGACY_HIFIGAN_VOLUME_PARAM_ID
    )
}
