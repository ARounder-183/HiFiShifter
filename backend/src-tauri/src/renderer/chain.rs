//! ProcessorChain：可组合的 Stage 链。
//!
//! 每个 [`ProcessingStage`] 接收上一步输出的 PCM，返回新 PCM；
//! [`ProcessorChain`] 串联多个 Stage 并实现 [`ClipProcessor`] trait。
//!
//! 内置 Stage：
//! - [`WorldVocoderStage`]：WORLD 声码器合成
//! - [`HiFiGanStage`]：NSF-HiFiGAN 合成
//!
//! 预设链构造：[`world_chain()`]、[`hifigan_chain()`]

use super::traits::{
    ClipProcessContext, ClipProcessor, ParamDescriptor, ProcessorCapabilities, RenderContext,
    Renderer,
};

static HIFIGAN_BREATH_OPTIONS: [(&str, i32); 2] = [("Off", 0), ("On", 1)];

/// 仅 NSF-HiFiGAN 专有的参数；共通混音级参数（volume / pan / dyn）**不在此处**
/// —— 它们由 `renderer::common_params` 统一提供，见 `renderer::all_param_descriptors`。
static HIFIGAN_PARAM_DESCRIPTORS: [ParamDescriptor; 4] = [
    ParamDescriptor {
        id: "breath_enabled",
        display_name: "Breath",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::StaticEnum {
            options: &HIFIGAN_BREATH_OPTIONS,
            default_value: 0,
        },
    },
    ParamDescriptor {
        id: "breath_gain",
        display_name: "Breath Gain",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "x",
            default_value: 1.0,
            min_value: 0.0,
            max_value: 2.0,
        },
    },
    ParamDescriptor {
        id: "hifigan_tension",
        display_name: "Tension",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            unit: "%",
            default_value: 0.0,
            min_value: -100.0,
            max_value: 100.0,
        },
    },
    ParamDescriptor {
        id: "formant_shift_cents",
        display_name: "Formant Shift",
        group: "NSF-HiFiGAN",
        kind: super::traits::ParamKind::AutomationCurve {
            // ±1200 cents（±1 八度），对齐 OpenUtau hifisampler 的 gender
            // （gender ±100 = ±1200 cents）。
            //
            // 【符号】正值 = 共振峰**上移**（声音变细），与 OpenUtau 的 gender 相反：
            // `gender = -formant_shift_cents / 12`。曲线语义与扩域前完全一致，
            // 只是允许的范围更大，因此既有工程不受影响。
            unit: "cents",
            default_value: 0.0,
            min_value: -1200.0,
            max_value: 1200.0,
        },
    },
];

// ─── StageContext ──────────────────────────────────────────────────────────────

/// 传递给每个 Stage 的完整上下文（持有对 [`ClipProcessContext`] 的引用）。
pub struct StageContext<'a> {
    pub clip_ctx: &'a ClipProcessContext<'a>,
}

// ─── ProcessingStage trait ────────────────────────────────────────────────────

/// 单一处理阶段，接收上一步 PCM，输出处理后 PCM。
pub trait ProcessingStage: Send + Sync {
    fn id(&self) -> &str;
    #[allow(dead_code)]
    fn display_name(&self) -> &str;
    /// Stage 自身贡献的参数描述符（可选）。
    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &[]
    }
    /// 接收上一步 PCM，输出处理后 PCM。
    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String>;
}

// ─── ProcessorChain ───────────────────────────────────────────────────────────

/// 实现 `ClipProcessor` 的 Stage 链，将多个 Stage 串联。
pub struct ProcessorChain {
    pub id: String,
    #[allow(dead_code)]
    pub display_name: String,
    pub stages: Vec<Box<dyn ProcessingStage>>,
    /// 处理器是否自行处理时间拉伸。
    /// 为 `true` 时调用方会跳过外部预拉伸，并将 `playback_rate`
    /// 通过 [`ClipProcessContext`] 传入处理器链内部。
    pub handles_time_stretch: bool,
}

impl ClipProcessor for ProcessorChain {
    fn id(&self) -> &str {
        &self.id
    }

    fn display_name(&self) -> &str {
        &self.display_name
    }

    fn is_available(&self) -> bool {
        // 链路整体可用性由各 Stage 自行控制；此处返回 true 让调用方统一判断
        true
    }

    fn capabilities(&self) -> ProcessorCapabilities {
        ProcessorCapabilities {
            handles_time_stretch: self.handles_time_stretch,
            supports_formant: false,
            supports_breathiness: self.stages.iter().any(|stage| stage.id() == "nsf_hifigan"),
        }
    }

    fn param_descriptors(&self) -> Vec<ParamDescriptor> {
        self.stages
            .iter()
            .flat_map(|s| s.param_descriptors().iter().cloned())
            .collect()
    }

    fn process(&self, ctx: &ClipProcessContext<'_>) -> Result<Vec<f32>, String> {
        let stage_ctx = StageContext { clip_ctx: ctx };
        let mut pcm = ctx.mono_pcm.to_vec();
        for stage in &self.stages {
            pcm = stage.process(pcm, &stage_ctx)?;
        }
        Ok(pcm)
    }
}

// ─── 内置 Stage 实现 ──────────────────────────────────────────────────────────

/// Stage 1a：WORLD 声码器合成。
pub struct WorldVocoderStage;

impl ProcessingStage for WorldVocoderStage {
    fn id(&self) -> &str {
        "world_vocoder"
    }

    fn display_name(&self) -> &str {
        "WORLD 声码器"
    }

    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        // WORLD 没有任何专有曲线；共通混音级参数由 `renderer::all_param_descriptors`
        // 统一追加，此处返回空切片。
        &[]
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        if !crate::world_vocoder::is_available() {
            return Ok(input_pcm);
        }
        let render_ctx = RenderContext {
            mono_pcm: &input_pcm,
            channel_index: cc.channel_index,
            sample_rate: cc.sample_rate,
            seg_start_sec: cc.seg_start_sec,
            seg_end_sec: cc.seg_end_sec,
            clip_start_sec: cc.clip_start_sec,
            frame_period_ms: cc.frame_period_ms,
            pitch_edit: cc.pitch_edit,
            clip_midi: cc.clip_midi,
            clip_id: cc.clip_id,
        };
        crate::renderer::world::WorldRenderer.render(&render_ctx)
    }
}

/// Stage 1b：NSF-HiFiGAN ONNX 合成。
pub struct HiFiGanStage;

/// 在绝对时间处采样一条自动化曲线（首点对应时间轴 0，帧周期 `frame_period_ms`）。
///
/// **越界语义 = hold-last**：超出曲线末点后恒取末值，与
/// `audio_engine/mix.rs::sample_automation_curve` 一致。
///
/// 【为什么必须是 hold-last】预览与导出共用同一条曲线，但走不同的采样函数：
/// 预览经 `mix.rs`（hold-last），导出经本函数。历史上本函数在越界处回退
/// `default_value`，于是**同一条 `breath_gain` 曲线在预览与导出下听感不同**——
/// 用户画到的段落之后，预览保持末值、导出却突然弹回默认增益。统一为 hold-last
/// 后两者一致，也符合"画到哪就保持到哪"的直觉。
///
/// 【为什么不插值到 default】早期实现让 `i1` 钳到末元素后继续插值，得到
/// `default … 末值` 之间的衰减/振荡值；hold-last 是常量，天然规避该问题。
fn sample_curve_at_abs_sec(
    curve: Option<&[f32]>,
    abs_sec: f64,
    frame_period_ms: f64,
    default_value: f32,
) -> f32 {
    let Some(curve) = curve else {
        return default_value;
    };
    if curve.is_empty() {
        return default_value;
    }
    let last = curve.len() - 1;

    let fp = frame_period_ms.max(0.1);
    let idx_f = (abs_sec.max(0.0) * 1000.0) / fp;
    if !idx_f.is_finite() {
        return default_value;
    }
    let i0 = idx_f.floor().max(0.0) as usize;
    // 越界：保持末值（hold-last），不回退 default。
    if i0 >= last {
        return curve[last];
    }
    let i1 = i0 + 1;
    let frac = (idx_f - i0 as f64).clamp(0.0, 1.0) as f32;
    let a = curve[i0];
    let b = curve[i1];
    a + (b - a) * frac
}

/// 把噪声（气声）stem 对齐到谐波输出的（时间轴）长度。
///
/// 拉伸场景（playback_rate != 1）不能用线性重采样：气声噪声不是白噪声，
/// 其谱包络（共振峰形状）会随线性重采样按 1/rate 整体缩放 —— 慢放时气声
/// 变闷（频谱压半）、快放时变亮且混叠，听感上就是"气声没有被正确拉伸"。
/// 因此这里与谐波分支一致使用外部拉伸算法（用户选择 Linear 时仍为线性）；
/// Mel Stretch 只作用于谐波分支 —— 对噪声做 HiFiGAN 重合成会注入谐波伪影。
///
/// 若直接按 `min(harmonic, noise)` 混合还会更糟：
/// - `playback_rate < 1`（拉长）时谐波比噪声长，输出被截断到噪声长度，
///   clip 拉伸出来的尾巴整段丢失（听感上就是"下一段音频被截断"）；
/// - `playback_rate > 1`（缩短）时噪声比谐波长，噪声尾部被丢掉。
pub(crate) fn align_noise_stem_to_len(
    noise: &[f32],
    sample_rate: u32,
    target_len: usize,
    algorithm: crate::time_stretch::StretchAlgorithm,
) -> Vec<f32> {
    if target_len == 0 || noise.is_empty() {
        return Vec::new();
    }
    if noise.len() == target_len {
        return noise.to_vec();
    }
    if noise.len() == 1 {
        return vec![noise[0]; target_len];
    }
    crate::time_stretch::time_stretch_interleaved(noise, 1, sample_rate, target_len, algorithm)
}

impl ProcessingStage for HiFiGanStage {
    fn id(&self) -> &str {
        "nsf_hifigan"
    }

    fn display_name(&self) -> &str {
        "NSF-HiFiGAN"
    }

    fn param_descriptors(&self) -> &'static [ParamDescriptor] {
        &HIFIGAN_PARAM_DESCRIPTORS
    }

    fn process(&self, input_pcm: Vec<f32>, ctx: &StageContext<'_>) -> Result<Vec<f32>, String> {
        let cc = ctx.clip_ctx;
        if !crate::nsf_hifigan_onnx::is_available() {
            return Ok(input_pcm);
        }

        let breath_enabled =
            crate::pitch_editing::extra_param_enabled(cc.extra_params, "breath_enabled");
        let formant_curve = cc
            .extra_curves
            .get("formant_shift_cents")
            .map(|v| v.as_slice());
        // HNSEP 不可用时降级为非 Breath 路径：调用方已因气声跳过外部拉伸，
        // 硬错误会让整个 clip 以源速率输出（被截断/补零），比"没有气声"
        // 严重得多。
        if breath_enabled && crate::hnsep_onnx::is_available() {
            return self.process_breath(input_pcm, cc, formant_curve);
        }
        if breath_enabled {
            log::warn!(
                "HiFiGanStage: breath enabled but HNSEP unavailable, \
                 falling back to non-breath rendering (clip_id={})",
                cc.clip_id
            );
        }

        // ── 非 Breath 路径 ──────────────────────────────────────────────
        let render_ctx = RenderContext {
            mono_pcm: &input_pcm,
            channel_index: cc.channel_index,
            sample_rate: cc.sample_rate,
            seg_start_sec: cc.seg_start_sec,
            seg_end_sec: cc.seg_end_sec,
            clip_start_sec: cc.clip_start_sec,
            frame_period_ms: cc.frame_period_ms,
            pitch_edit: cc.pitch_edit,
            clip_midi: cc.clip_midi,
            clip_id: cc.clip_id,
        };
        let renderer = crate::renderer::hifigan::HiFiGanRenderer;
        if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
            if cc.clip_midi.is_empty() {
                // 无 F0 无法走 mel 拉伸（HiFiGAN 需要音高激励）：回退外部算法
                // 原位拉伸，保证输出帧数与"处理器内部拉伸"的承诺一致 ——
                // 调用方已因此跳过外部预拉伸，这里若再返回原 PCM 就会出现
                // "外部不拉伸、内部也不拉伸"，输出被截断/补零。
                return Ok(crate::time_stretch::time_stretch_interleaved(
                    &input_pcm,
                    1,
                    cc.sample_rate,
                    cc.out_frames,
                    crate::time_stretch::resolved_external_stretch_algorithm(),
                ));
            }
            return renderer.render_mel_stretch_with_formant(
                &render_ctx,
                cc.playback_rate,
                formant_curve,
            );
        }
        renderer.render_with_formant(&render_ctx, formant_curve)
    }
}

impl HiFiGanStage {
    /// Breath 路径：HNSEP 分离谐波/噪声 → 谐波走 HiFiGAN（mel 拉伸或外部
    /// 算法回退）→ 噪声对齐到时间轴长度后按 breath_gain 混合。
    fn process_breath(
        &self,
        input_pcm: Vec<f32>,
        cc: &crate::renderer::traits::ClipProcessContext<'_>,
        formant_curve: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let (harmonic, noise) =
            crate::hnsep_onnx::infer_harmonic_noise_mono(
                cc.clip_id,
                &input_pcm,
                cc.sample_rate,
                cc.channel_index,
                cc.source_fingerprint,
            )?;

        // 谐波分支：有 F0（clip_midi）时走 HiFiGAN mel 拉伸/渲染；无 F0 时
        // 回退外部算法拉伸 —— 两种情况输出都是时间轴长度 out_frames。
        let processed_harmonic = if cc.clip_midi.is_empty() {
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                crate::time_stretch::time_stretch_interleaved(
                    &harmonic,
                    1,
                    cc.sample_rate,
                    cc.out_frames,
                    crate::time_stretch::resolved_external_stretch_algorithm(),
                )
            } else {
                (*harmonic).clone()
            }
        } else {
            let render_ctx = RenderContext {
                mono_pcm: &harmonic,
                channel_index: cc.channel_index,
                sample_rate: cc.sample_rate,
                seg_start_sec: cc.seg_start_sec,
                seg_end_sec: cc.seg_end_sec,
                clip_start_sec: cc.clip_start_sec,
                frame_period_ms: cc.frame_period_ms,
                pitch_edit: cc.pitch_edit,
                clip_midi: cc.clip_midi,
                clip_id: cc.clip_id,
            };
            let renderer = crate::renderer::hifigan::HiFiGanRenderer;
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                renderer.render_mel_stretch_with_formant(
                    &render_ctx,
                    cc.playback_rate,
                    formant_curve,
                )?
            } else {
                renderer.render_with_formant(&render_ctx, formant_curve)?
            }
        };

        let breath_curve = cc.extra_curves.get("breath_gain").map(|v| v.as_slice());

        // Fast path: only when a curve is present **and** uniformly zero (e.g. when
        // computing harmonic_only for BreathNoiseCache) can we skip noise mixing.
        //
        // 曲线**缺失**必须走混合路径：`breath_gain` 的描述符默认值是 1.0，未绘制
        // 参数线即"整体使用默认增益"，而预览路径的 breath stem 也是按默认 1.0 混入
        // 的（见 `audio_engine/mix.rs`）。此处若把缺失当成 0 直接返回谐波，导出就
        // 会丢掉整条气声 —— 预览正常、导出无气声正是这个分支造成的。
        let gain_is_zero = breath_curve.map_or(false, |c| {
            c.is_empty() || c.iter().all(|&v| v.abs() < f32::EPSILON)
        });
        if gain_is_zero {
            return Ok(processed_harmonic);
        }

        // 噪声 stem 与谐波对齐到同一（时间轴）长度后再混合。
        // 不能用 `min(harmonic, noise)`：那会在拉伸后把谐波尾巴裁掉；
        // 也不能用线性重采样：气声的谱包络会被按 1/rate 缩放（变闷/混叠），
        // 必须与拉伸算法一致地做高质量时间拉伸。
        let noise_aligned = align_noise_stem_to_len(
            &noise,
            cc.sample_rate,
            processed_harmonic.len(),
            crate::time_stretch::resolved_external_stretch_algorithm(),
        );
        let out_len = processed_harmonic.len();

        let has_varying_curve = breath_curve.map_or(false, |c| {
            if c.len() <= 1 {
                return false;
            }
            let first = c[0];
            c.iter().any(|&v| (v - first).abs() > f32::EPSILON)
        });

        let mixed: Vec<f32> = if has_varying_curve {
            let inv_sample_rate = 1.0 / cc.sample_rate.max(1) as f64;
            processed_harmonic
                .iter()
                .zip(noise_aligned.iter())
                .take(out_len)
                .enumerate()
                .map(|(index, (&h, &n))| {
                    let abs_sec = cc.seg_start_sec + index as f64 * inv_sample_rate;
                    let gain =
                        sample_curve_at_abs_sec(breath_curve, abs_sec, cc.frame_period_ms, 1.0);
                    h + n * gain
                })
                .collect()
        } else {
            // Constant gain (typically 1.0): use uniform multiplier, auto-vectorizable
            let gain = breath_curve.and_then(|c| c.first().copied()).unwrap_or(1.0);
            if (gain - 1.0).abs() < f32::EPSILON {
                // gain == 1.0: simple addition, most common case for unity_breath
                processed_harmonic
                    .iter()
                    .zip(noise_aligned.iter())
                    .take(out_len)
                    .map(|(&h, &n)| h + n)
                    .collect()
            } else {
                processed_harmonic
                    .iter()
                    .zip(noise_aligned.iter())
                    .take(out_len)
                    .map(|(&h, &n)| h + n * gain)
                    .collect()
            }
        };

        Ok(mixed)
    }
}

// ─── 预设链构造 ───────────────────────────────────────────────────────────────

/// 构造 WORLD Vocoder 处理链。
pub fn world_chain() -> ProcessorChain {
    ProcessorChain {
        id: "world".into(),
        display_name: "WORLD Vocoder".into(),
        stages: vec![Box::new(WorldVocoderStage)],
        handles_time_stretch: false,
    }
}

/// 构造 NSF-HiFiGAN 处理链。
pub fn hifigan_chain() -> ProcessorChain {
    ProcessorChain {
        id: "nsf_hifigan".into(),
        display_name: "NSF-HiFiGAN".into(),
        stages: vec![Box::new(HiFiGanStage)],
        handles_time_stretch: false,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hifigan_chain_no_longer_handles_time_stretch() {
        let chain = super::hifigan_chain();
        assert!(!chain.handles_time_stretch);
    }

    #[test]
    fn align_noise_stem_keeps_identity_length() {
        let noise = vec![0.1f32, 0.2, 0.3, 0.4];
        assert_eq!(
            super::align_noise_stem_to_len(
                &noise,
                44_100,
                4,
                crate::time_stretch::StretchAlgorithm::SignalsmithStretch
            ),
            noise
        );
    }

    #[test]
    fn align_noise_stem_follows_runtime_stretch_algorithm() {
        // 线性算法下必须与 time_stretch_interleaved(LinearResample) 完全一致 ——
        // 保证"噪声跟随用户选择的拉伸算法"这一契约（此前是无条件线性重采样，
        // 气声谱包络被按 1/rate 缩放，听感即"气声没有被正确拉伸"）。
        let noise: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.25).sin()).collect();
        let out = super::align_noise_stem_to_len(
            &noise,
            44_100,
            128,
            crate::time_stretch::StretchAlgorithm::LinearResample,
        );
        let expected = crate::time_stretch::time_stretch_interleaved(
            &noise,
            1,
            44_100,
            128,
            crate::time_stretch::StretchAlgorithm::LinearResample,
        );
        assert_eq!(out, expected);
    }

    #[test]
    fn align_noise_stem_stretches_without_truncating() {
        // 拉伸场景（playback_rate < 1）：谐波比噪声长，噪声必须被拉长，
        // 否则 min() 会把谐波（进而整个 clip）的尾巴裁掉。
        let noise = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = super::align_noise_stem_to_len(
            &noise,
            44_100,
            8,
            crate::time_stretch::StretchAlgorithm::LinearResample,
        );
        assert_eq!(out.len(), 8);
        // 端点应贴合原始端点
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[7] - 4.0).abs() < 1e-6);
        // 中间值必须来自原始信号，而不是补零
        assert!(out.iter().all(|v| *v >= 1.0 - 1e-6 && *v <= 4.0 + 1e-6));
    }

    #[test]
    fn align_noise_stem_shrinks_for_speedup() {
        let noise = vec![0.0f32, 1.0, 2.0, 3.0];
        let out = super::align_noise_stem_to_len(
            &noise,
            44_100,
            2,
            crate::time_stretch::StretchAlgorithm::LinearResample,
        );
        assert_eq!(out.len(), 2);
        assert!((out[0] - 0.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn align_noise_stem_handles_degenerate_inputs() {
        use crate::time_stretch::StretchAlgorithm::SignalsmithStretch;
        assert!(super::align_noise_stem_to_len(&[], 44_100, 8, SignalsmithStretch).is_empty());
        assert!(super::align_noise_stem_to_len(&[0.5], 44_100, 0, SignalsmithStretch).is_empty());
        // 单样本输入：按常数填充，不得 panic
        assert_eq!(
            super::align_noise_stem_to_len(&[0.5], 44_100, 3, SignalsmithStretch),
            vec![0.5, 0.5, 0.5]
        );
    }

    /// 越界语义 = **hold-last**（保持末值），不得回退 default。
    ///
    /// 【为什么钉住】预览（`audio_engine/mix.rs::sample_automation_curve`）用的就是
    /// hold-last，而导出走本函数。历史上本函数在越界处回退 `default_value`，
    /// 导致同一条 `breath_gain` 曲线**预览与导出听感不同**：画到的段落之后，
    /// 预览保持末值、导出弹回默认增益。
    ///
    /// 同时保留早期修复的意图：越界区段必须是**常量**，不得是
    /// `default … 末值` 之间的衰减/振荡值。
    #[test]
    fn sample_curve_beyond_end_holds_last_value() {
        let curve = vec![0.0f32, 0.0, 0.0, 359.15]; // 末点 359.15 @ idx 3
        let fp = 5.0;
        let default = 0.0;

        // 末点本身 → 末值
        let at_last = super::sample_curve_at_abs_sec(Some(&curve), 3.0 * fp / 1000.0, fp, default);
        assert!((at_last - 359.15).abs() < 1e-4);

        // 末点之后任意位置 → 仍为末值（hold-last）
        let just_after =
            super::sample_curve_at_abs_sec(Some(&curve), 3.5 * fp / 1000.0, fp, default);
        assert!(
            (just_after - 359.15).abs() < 1e-4,
            "expected hold-last 359.15, got {just_after}"
        );
        let beyond = super::sample_curve_at_abs_sec(Some(&curve), 4.5 * fp / 1000.0, fp, default);
        assert!(
            (beyond - 359.15).abs() < 1e-4,
            "expected hold-last 359.15, got {beyond}"
        );
        // 大越界同样是末值，且不得出现振荡
        let far = super::sample_curve_at_abs_sec(Some(&curve), 100.0, fp, default);
        assert!(
            (far - 359.15).abs() < 1e-4,
            "expected hold-last 359.15, got {far}"
        );

        // 空曲线 / None → default（"没有曲线"仍表示用默认值，与 hold-last 不冲突）
        assert_eq!(
            super::sample_curve_at_abs_sec(Some(&[]), 1.0, fp, default),
            default
        );
        assert_eq!(
            super::sample_curve_at_abs_sec(None, 1.0, fp, default),
            default
        );
    }

    /// hold-last 必须对 breath_gain 的实际默认值（1.0）同样成立 ——
    /// 否则"画了一半的曲线"之后会从末值弹回 1.0（原缺陷的听感表现）。
    #[test]
    fn sample_curve_hold_last_uses_curve_not_default() {
        let curve = vec![0.0f32, 0.25, 0.5];
        let fp = 5.0;
        // 越界处 default 是 1.0，但结果必须是末值 0.5
        let v = super::sample_curve_at_abs_sec(Some(&curve), 10.0, fp, 1.0);
        assert!(
            (v - 0.5).abs() < 1e-6,
            "must hold last curve value 0.5, not default 1.0; got {v}"
        );
    }

    #[test]
    fn sample_curve_interpolates_within_range() {
        let curve = vec![0.0f32, 100.0, 200.0];
        let fp = 5.0;
        // idx 0.5 → 50
        let mid = super::sample_curve_at_abs_sec(Some(&curve), 0.5 * fp / 1000.0, fp, 0.0);
        assert!((mid - 50.0).abs() < 1e-4);
        // 负时间 → idx 0 → 第一个元素
        let neg = super::sample_curve_at_abs_sec(Some(&curve), -2.0, fp, 0.0);
        assert_eq!(neg, 0.0);
        // 区间内插值仍保留（idx 1.5 → 150）
        let mid2 = super::sample_curve_at_abs_sec(Some(&curve), 1.5 * fp / 1000.0, fp, 0.0);
        assert!((mid2 - 150.0).abs() < 1e-4);
    }

    /// **预览与导出必须一致**：同一曲线、同一时刻，渲染侧采样器
    /// （本模块，导出走它）与混音侧采样器（`audio_engine::mix`，预览走它）
    /// 必须给出相同的值 —— 包括越界区段。
    ///
    /// 这是本模块越界语义修复的**跨模块判据**：单看任一侧的测试都无法发现
    /// "两侧各自自洽、彼此不同"这一缺陷形态。
    ///
    /// 注意两个采样器的入参口径不同：
    /// - 本模块收**绝对秒**；
    /// - 混音侧收 `abs_frame`，它是**绝对采样点序号**（不是曲线帧号），
    ///   内部按 `abs_frame / sample_rate` 换成秒（见 `mix.rs` 的
    ///   `volume_curve_samples_at_timeline_absolute_frame`）。
    /// 因此这里用同一个「绝对秒」推出两者的入参，而不是直接传同一个整数。
    #[test]
    fn sample_curve_agrees_with_preview_mixer() {
        let fp = 5.0;
        let sr = 44_100u32;
        let curves: Vec<Option<Vec<f32>>> = vec![
            // 末值非默认值 —— 最能暴露 hold-last vs default 的分歧
            Some(vec![0.0, 0.25, 0.5]),
            // 单点曲线
            Some(vec![0.75]),
            // 全默认值
            Some(vec![1.0, 1.0, 1.0]),
            // 零值末点
            Some(vec![1.0, 0.0]),
            None,
            Some(vec![]),
        ];

        // 覆盖曲线内、末点、末点之后、以及很远的位置（单位：曲线帧）
        for &curve_frame in &[0.0f64, 1.0, 2.0, 3.0, 4.0, 10.0, 100.0, 10_000.0] {
            let abs_sec = curve_frame * fp / 1000.0;
            // 同一时刻换算成绝对采样点（四舍五入到最近的采样点）
            let abs_frame = (abs_sec * sr as f64).round() as u64;

            for curve in &curves {
                let export = super::sample_curve_at_abs_sec(curve.as_deref(), abs_sec, fp, 1.0);
                let preview = crate::audio_engine::mix::sample_automation_curve(
                    curve.as_deref(),
                    abs_frame,
                    sr,
                    fp,
                    1.0,
                );
                assert!(
                    (export - preview).abs() < 1e-3,
                    "preview/export disagree at {abs_sec}s (frame {abs_frame}) \
                     for curve {curve:?}: preview={preview}, export={export}"
                );
            }
        }
    }
}
