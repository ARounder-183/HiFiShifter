//! 分块流式音高分析的全局配置。
//!
//! 仅保留**实际接线**的字段：分块长度 / 上下文长度 / 是否启用分块。
//! 历史上这里还带过一整套 VAD（RMS 窗口 → 有声区间 → 合并 → 分块 → 交叉淡化）
//! 辅助函数与三个对应的环境变量，但那条链路从未接进任何调用方，已整体删除。

use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct PitchAnalysisConfig {
    pub chunk_sec: f64,
    pub chunk_ctx_sec: f64,
    /// 是否启用分块流式分析。
    ///
    /// 显式设 `HIFISHIFTER_PITCH_CHUNK_SEC=0` 可关闭分块，回落到一次性整份分析。
    /// 这是超长素材内存问题的逃生阀：整份路径在素材不大时结果更"原样"（无块边界
    /// 近似），也方便对照排查分块引入的差异。
    pub chunking_enabled: bool,
}

impl PitchAnalysisConfig {
    pub fn global() -> &'static Self {
        static CFG: OnceLock<PitchAnalysisConfig> = OnceLock::new();
        CFG.get_or_init(|| {
            // 注意 `env_f64` 会过滤掉 0，因此"关闭"要单独判一次原始字符串。
            let chunk_env = std::env::var("HIFISHIFTER_PITCH_CHUNK_SEC").ok();
            let chunking_enabled = chunk_env
                .as_deref()
                .map(|raw| raw.trim().parse::<f64>().map(|v| v > 0.0).unwrap_or(true))
                .unwrap_or(true);
            PitchAnalysisConfig {
                chunk_sec: env_f64("HIFISHIFTER_PITCH_CHUNK_SEC").unwrap_or(30.0),
                chunk_ctx_sec: env_f64("HIFISHIFTER_PITCH_CHUNK_CTX_SEC").unwrap_or(0.3),
                chunking_enabled,
            }
        })
    }
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v > 0.0)
}
