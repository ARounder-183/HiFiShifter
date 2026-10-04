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

/// 谐波/噪声分离开关的 id。语义见 `HIFIGAN_PARAM_DESCRIPTORS` 中该参数的说明。
pub const HIFIGAN_SEPARATION_PARAM_ID: &str = "breath_enabled";

/// 判定张力曲线"是否活跃"的阈值。
///
/// 与 `pitch_editing::hifigan_tension_active_for_clip`（容差 1e-3）同量级：
/// 曲线整体接近 0 时视为未编辑，避免为此白跑一遍 HNSEP 分离与逐帧 Rd 拟合。
///
/// `pub(crate)`：`state::TimelineState::migrate_legacy_breath_separation`
/// 也用它判断"曲线是否偏离默认值"。两处必须同源 —— 各写一个阈值必然漂移，
/// 而这里的漂移后果是"迁移无端打开开关、白跑一次 HNSEP"。
pub const TENSION_ACTIVE_EPSILON: f32 = 1.0e-3;

/// 仅 NSF-HiFiGAN 专有的参数；共通混音级参数（volume / pan / dyn）**不在此处**
/// —— 它们由 `renderer::common_params` 统一提供，见 `renderer::all_param_descriptors`。
static HIFIGAN_PARAM_DESCRIPTORS: [ParamDescriptor; 4] = [
    ParamDescriptor {
        id: "breath_enabled",
        // 语义是「是否做谐波/噪声分离」，而**不是**「气声开关」：
        // 气声曲线与张力曲线都依赖这次分离（Rd 张力只作用于谐波支）。
        // 改名是为了让 UI 不再把这一全局前提呈现成 breath_gain 的附属属性。
        display_name: "Harmonic Separation",
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
            extra_params: &cc.extra_params,
        };
        crate::renderer::world::WorldRenderer.render(&render_ctx)
    }
}

/// Stage 1b：NSF-HiFiGAN ONNX 合成。
pub struct HiFiGanStage;

/// 解析噪声支的（常量）混合增益。
///
/// **曲线缺失时返回 1.0**，即把噪声原样相加。这不是"默认关闭"而是"默认不变"：
/// 张力活跃但未开气声时也会走到混音路径（见 `HiFiGanStage::process` 的
/// `needs_separation`），此时若把缺失当成 0，整条噪声支（齿音、气声、嘶声）
/// 会被静默删除 —— 听感是"咬字发闷、辅音消失"。
/// 与 OpenUtau 的 `NoiseGain(breathiness=0) == 1.0` 语义一致。
fn resolve_noise_gain(breath_curve: Option<&[f32]>) -> f32 {
    breath_curve.and_then(|c| c.first().copied()).unwrap_or(1.0)
}

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
// `pub`：app 侧的跨边界一致性测试（`renderer_cross_checks.rs`）要拿它与预览侧采样器比对。
pub fn sample_curve_at_abs_sec(
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
pub fn align_noise_stem_to_len(
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

        // ── 曲线解析 ────────────────────────────────────────────────────
        let formant_curve = cc
            .extra_curves
            .get("formant_shift_cents")
            .map(|v| v.as_slice());
        let tension_curve = cc.extra_curves.get("hifigan_tension").map(|v| v.as_slice());

        // 张力是否需要真正参与运算（曲线偏离默认 0 即算活跃）。
        // 口径与渲染键一致，避免"曲线存在但全 0"时白跑一遍 HNSEP。
        //
        // 【注意】这里用的是 `cc.extra_curves`，而它已经过上游门禁：开关关闭时
        // `hifigan_tension` 已在 `ClipProcessContext` 的唯一构造点被剥离
        // （`pitch_editing::maybe_apply_pitch_edit_to_clip_segment`），
        // 因此 `tension_curve` 自然为 `None`。**不需要**在此重复判断开关 ——
        // 也不应该：两条判据并存正是本项目出现过的漂移来源。
        let tension_active = tension_curve.is_some_and(|c| {
            c.iter()
                .any(|v| v.is_finite() && v.abs() > TENSION_ACTIVE_EPSILON)
        });

        // ── 分离路径的门禁 ──────────────────────────────────────────────
        //
        // 【为什么张力也要求分离】Rd 张力只重塑**谐波**结构（`audio/rd_tension.rs`），
        // 而谐波/噪声的分支只存在于 `process_breath` 里 —— 它调用 HNSEP 拿到
        // `(harmonic, noise)`。没有分离就没有"只动谐波"这回事。这与 OpenUtau 的
        // `NeedsSeparation = HasTension || ...` 判据一致。
        //
        // 【开关关闭时不走 HNSEP】`breath_enabled`（UI 名「气声分离」）是气声与张力
        // **共同的前提**。关闭时上游已把 `breath_gain` 与 `hifigan_tension` 两条曲线
        // 从下发数据中剥离（见 `ClipProcessContext` 的唯一构造点），因此这里的
        // `tension_active` 必为 `false`，`needs_separation` 随之等价于
        // `separation_switch_on` —— 即**不产生任何 HNSEP 推理开销**。
        //
        // 仍保留 `tension_active ||` 这一项：它表达的是"张力本身就需要分离"这一
        // 不变的事实，而不是一个可能的触发源。若将来有人把曲线剥离移除，
        // 这里仍会正确地要求分离（而不是静默丢掉张力）。
        let separation_switch_on =
            crate::pitch_editing::extra_param_enabled(cc.extra_params, HIFIGAN_SEPARATION_PARAM_ID);

        let needs_separation = tension_active || separation_switch_on;

        if needs_separation && !crate::hnsep_onnx::is_available() {
            // 【绝不静默降级】历史实现只打一条日志然后走非分离路径，而那条路径
            // **根本不读张力曲线** —— 用户画了张力却听不到任何效果，界面上也无
            // 任何提示，属于最难排查的一类缺陷。这里返回错误：本次 clip 渲染失败、
            // 不写缓存，`render_background_pass` 会记录并保持等待（与
            // "pitch processor failed" 同一约定），HNSEP 恢复后重试即成功。
            let what = if tension_active && separation_switch_on {
                "气声分离与张力"
            } else if tension_active {
                "张力"
            } else {
                "气声分离"
            };
            return Err(format!(
                "{what}需要谐波/噪声分离，但 HNSEP 模型不可用；\
                 请检查 models/hnsep 是否存在，或改用其他音高算法 (clip_id={})",
                cc.clip_id
            ));
        }

        if needs_separation {
            return self.process_breath(input_pcm, cc, formant_curve, tension_curve);
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
            extra_params: &cc.extra_params,
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
        // 非分离路径：张力已在构造点被剥离（开关关闭）或本就不存在，
        // 因此不传张力曲线（None）。若开关开启，张力走分离路径（见下方）。
        renderer.render_with_formant(&render_ctx, formant_curve, None)
    }
}

impl HiFiGanStage {
    /// 分离路径：HNSEP 分离谐波/噪声 → 谐波支施加 Rd 张力 → 谐波走 HiFiGAN
    /// （mel 拉伸或外部算法回退）→ 噪声对齐到时间轴长度后按 breath_gain 混合。
    ///
    /// # 为什么张力放在这里
    /// Rd 张力**只重塑谐波结构**：噪声支（齿音、气声、嘶声）必须原样保留，
    /// 否则 1 kHz 以上被无差别提升，听感就是"刺耳/金属感"。而 harmonic/noise
    /// 分支只在本函数内存在，因此张力必须在此处施加（与 OpenUtau 的
    /// `HifiFeatures.Generate` 一致：先分离，再对 voiced 施加 Rd，最后混合）。
    ///
    /// # 顺序
    /// 张力在 **mel 分析之前**、作用于**波形域**（STFT→改谱→ISTFT 得到新波形），
    /// 随后才由 `render_with_formant` 提取 mel 并交给声码器重合成。
    /// 不能把张力挪到 mel 域：Rd 的增益是施加在**目标音高**的谐波位置上的，
    /// 而 mel 分析本身还要按 gender 伸缩频率轴，两者混在一起会互相污染。
    ///
    /// # 参数
    /// - `tension_curve`：张力曲线（-100..100），可按 clip 级 / 轨道级解析后传入
    ///
    /// # 气声开关为什么不作为参数
    /// 噪声支**总是**混回（曲线缺失时取默认增益 1.0，即原样相加）。气声开关只决定
    /// 调用方是否解析并下发 `breath_gain` 曲线，不影响本函数的混合逻辑 —— 无论
    /// 进入本路径的是"开了气声"还是"只开了张力"，噪声都必须以单位增益归还，
    /// 否则张力一开就会把齿音/气声整条删掉（详见下方混音处的说明）。
    fn process_breath(
        &self,
        input_pcm: Vec<f32>,
        cc: &crate::renderer::traits::ClipProcessContext<'_>,
        formant_curve: Option<&[f32]>,
        tension_curve: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let (harmonic, noise) = crate::hnsep_onnx::infer_harmonic_noise_mono(
            cc.clip_id,
            &input_pcm,
            cc.sample_rate,
            cc.channel_index,
            cc.source_fingerprint,
        )?;

        // ── 谐波支施加 Rd 张力（在 mel 分析之前，只动谐波）──────────────
        let tensioned_harmonic = Self::apply_rd_tension(&harmonic, cc, tension_curve);

        // 谐波分支：有 F0（clip_midi）时走 HiFiGAN mel 拉伸/渲染；无 F0 时
        // 回退外部算法拉伸 —— 两种情况输出都是时间轴长度 out_frames。
        let processed_harmonic = if cc.clip_midi.is_empty() {
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                crate::time_stretch::time_stretch_interleaved(
                    &tensioned_harmonic,
                    1,
                    cc.sample_rate,
                    cc.out_frames,
                    crate::time_stretch::resolved_external_stretch_algorithm(),
                )
            } else {
                tensioned_harmonic
            }
        } else {
            let render_ctx = RenderContext {
                mono_pcm: &tensioned_harmonic,
                channel_index: cc.channel_index,
                sample_rate: cc.sample_rate,
                seg_start_sec: cc.seg_start_sec,
                seg_end_sec: cc.seg_end_sec,
                clip_start_sec: cc.clip_start_sec,
                frame_period_ms: cc.frame_period_ms,
                pitch_edit: cc.pitch_edit,
                clip_midi: cc.clip_midi,
                clip_id: cc.clip_id,
                extra_params: &cc.extra_params,
            };
            let renderer = crate::renderer::hifigan::HiFiGanRenderer;
            if (cc.playback_rate - 1.0).abs() > 1.0e-6 {
                renderer.render_mel_stretch_with_formant(
                    &render_ctx,
                    cc.playback_rate,
                    formant_curve,
                )?
            } else {
                renderer.render_with_formant(&render_ctx, formant_curve, tension_curve)?
            }
        };

        // 气声未开时，噪声支仍必须按**单位增益**混回。
        //
        // 【为什么不能直接返回谐波】张力活跃时也会进入本路径（见
        // `HiFiGanStage::process` 的 `needs_separation`）。此时若只返回谐波，
        // 就等于把整条噪声支（齿音、气声、嘶声）**整个删掉** —— 听感是"咬字发闷、
        // 辅音消失"。OpenUtau 的处理是 `x = NoiseGain(breath)*(wave-h) + voiced`，
        // 而 breath 默认 0 对应 `NoiseGain(0) = 1.0`，即**原样混回**。
        // 因此这里取默认增益 1.0 继续走下面的混音路径，而不是提前返回。
        let breath_curve = cc.extra_curves.get("breath_gain").map(|v| v.as_slice());

        // Fast path: only when a curve is present **and** uniformly zero (e.g. when
        // computing harmonic_only for BreathNoiseCache) can we skip noise mixing.
        //
        // 曲线**缺失**必须走混合路径：`breath_gain` 的描述符默认值是 1.0，未绘制
        // 参数线即"整体使用默认增益"，而预览路径的 breath stem 也是按默认 1.0 混入
        // 的（见 `audio_engine/mix.rs`）。此处若把缺失当成 0 直接返回谐波，导出就
        // 会丢掉整条气声 —— 预览正常、导出无气声正是这个分支造成的。
        //
        // 【气声未开时不得走这条 fast path】`breath_gain` 曲线缺失 → `map_or(false)`
        // 为假 → 不会提前返回，正好落到下方"gain = 1.0"的常量分支，实现原样混回。
        // 若把"气声未开"也当成"跳过噪声"，就会重现上面那个缺陷。
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
            let gain = resolve_noise_gain(breath_curve);
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

    /// 在谐波支上施加 Rd 张力（波形域，mel 分析之前）。
    ///
    /// # 流程
    /// 1. 按 [`crate::rd_tension::HOP`] 的帧步长，为整段谐波采样出**源 f0** 与
    ///    **目标 f0**；
    /// 2. 交给 [`crate::rd_tension::RdTension::apply`] 做 STFT→逐帧 Rd 拟合→
    ///    按张力重塑→ISTFT；
    /// 3. 张力不活跃、或无源 f0 时**原样返回**输入（不做任何变换）。
    ///
    /// # 源 f0 与目标 f0 的时间基
    /// - 源 f0 取自 `cc.clip_midi`（FCPE 对**源文件**的分析结果，已按
    ///   `playback_rate` 重采样到时间线帧网格）；
    /// - 目标 f0 取自 `cc.pitch_edit`（用户编辑的目标音高），无编辑处回退源音高；
    /// - 处理器收到的 PCM 是**源速率**的，因此第 `i` 个样本对应的时间线绝对时间为
    ///   `seg_start_sec + i / sample_rate / playback_rate`
    ///   （与 `nsf_hifigan_onnx` 的分块时间映射同构）。
    ///
    /// Rd 的增益要落在**目标音高**的谐波上（变调后音色才正确），而谐波峰定位
    /// 用的是**源 f0** —— 两者缺一不可，这是本函数同时取两条曲线的原因。
    ///
    /// # 参数
    /// - `harmonic`：HNSEP 分离出的谐波支（与输入同采样率）
    /// - `cc`：处理器上下文（提供 `clip_midi` / `pitch_edit` / 时间基）
    /// - `tension_curve`：张力曲线（-100..100），`None` 或全 0 时不做处理
    ///
    /// # 特殊说明
    /// 未浊音帧（源 f0 <= 0）不参与；张力为 0 的帧跳过；两者都不会引入伪影。
    fn apply_rd_tension(
        harmonic: &[f32],
        cc: &crate::renderer::traits::ClipProcessContext<'_>,
        tension_curve: Option<&[f32]>,
    ) -> Vec<f32> {
        let Some(curve) = tension_curve else {
            return harmonic.to_vec();
        };
        if curve.is_empty() || harmonic.is_empty() {
            return harmonic.to_vec();
        }
        // 曲线整体接近 0 → 未编辑，直接跳过（省掉 STFT 与 Rd 拟合）。
        if !curve
            .iter()
            .any(|v| v.is_finite() && v.abs() > TENSION_ACTIVE_EPSILON)
        {
            return harmonic.to_vec();
        }
        // 无源音高（分析未完成）时无法定位谐波峰，跳过而非退化处理。
        if cc.clip_midi.is_empty() {
            return harmonic.to_vec();
        }

        let sr = cc.sample_rate;
        if sr == 0 {
            return harmonic.to_vec();
        }
        let rate = if cc.playback_rate.is_finite() && cc.playback_rate > 1e-6 {
            cc.playback_rate
        } else {
            1.0
        };
        let frames = harmonic.len().div_ceil(crate::rd_tension::HOP) + 1;

        // 逐帧采样源 f0（Hz）与目标 f0（Hz）。
        // 帧 m 的中心样本 ≈ m * HOP，对应时间线绝对秒见上方说明。
        let mut source_f0 = Vec::with_capacity(frames);
        let mut target_f0 = Vec::with_capacity(frames);
        for m in 0..frames {
            let abs_sec = cc.seg_start_sec
                + (m * crate::rd_tension::HOP) as f64 / (sr as f64) / rate;
            let src = crate::renderer::utils::clip_midi_at_time(
                cc.frame_period_ms,
                cc.clip_start_sec,
                cc.clip_midi,
                abs_sec,
            );
            source_f0.push(if src > 0.0 {
                440.0 * 2.0f64.powf((src - 69.0) / 12.0)
            } else {
                0.0
            });
            // 目标音高：用户编辑优先，缺编辑处回退源音高（与 hifigan 的
            // `midi_fn` 同一语义）。
            let target = crate::renderer::utils::edit_midi_at_time_or_none(
                cc.frame_period_ms,
                cc.pitch_edit,
                abs_sec,
            )
            .unwrap_or(src);
            target_f0.push(if target > 0.0 {
                440.0 * 2.0f64.powf((target - 69.0) / 12.0)
            } else {
                0.0
            });
        }

        // 张力曲线：按样本位置查询（帧 m 的样本位置 = m * HOP）。
        let fp = cc.frame_period_ms.max(0.1);
        let tension_at = |sample_idx: usize| -> f64 {
            let abs_sec =
                cc.seg_start_sec + (sample_idx as f64) / (sr as f64) / rate;
            sample_curve_at_abs_sec(Some(curve), abs_sec, fp, 0.0) as f64
        };
        let target_at = |sample_idx: usize| -> f64 {
            let m = sample_idx / crate::rd_tension::HOP;
            target_f0.get(m).copied().unwrap_or(0.0)
        };

        crate::rd_tension::RdTension::apply(
            harmonic,
            &source_f0,
            sr,
            tension_at,
            target_at,
        )
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
    use super::TENSION_ACTIVE_EPSILON;

    /// `breath_enabled`（UI 名「气声分离」）是气声与张力**共同的前提**。
    ///
    /// 【语义已变更，勿按旧版理解】早期实现里张力活跃**独立于**开关
    /// （`needs_separation = tension_active || switch`），使得"开关关闭但张力生效"。
    /// 现在关闭开关会在 `ClipProcessContext` 构造点剥离两条曲线
    /// （见 `pitch_editing::gate_separation_curves` 及其测试），
    /// 因此 `tension_active` 必为 `false`，`needs_separation` 完全由开关决定
    /// —— 即"关闭开关 ⇒ 不走 HNSEP"。
    #[test]
    fn separation_requires_the_switch_because_curves_are_stripped() {
        // 阈值口径：曲线需真正偏离默认才算活跃。
        assert!(TENSION_ACTIVE_EPSILON > 0.0);
        assert!(!(0.0f32).gt(&TENSION_ACTIVE_EPSILON));
        // 边界：恰好等于阈值不算活跃（口径与渲染键一致）
        assert!(!TENSION_ACTIVE_EPSILON.gt(&TENSION_ACTIVE_EPSILON));
        assert!((TENSION_ACTIVE_EPSILON * 2.0).gt(&TENSION_ACTIVE_EPSILON));
    }

    /// 非有限值不得被当成"活跃"（会把 NaN 传播进 STFT）。
    #[test]
    fn non_finite_tension_is_not_active() {
        let curve = [f32::NAN, f32::INFINITY, 0.0];
        let active = curve
            .iter()
            .any(|&v| v.is_finite() && v.abs() > TENSION_ACTIVE_EPSILON);
        assert!(!active, "NaN/Inf must not be treated as an active tension");
    }

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

    /// 噪声支的常量增益：**曲线缺失 → 1.0（原样混回）**。
    ///
    /// 【为什么钉住】张力活跃但未开气声时也会进入 HNSEP 分离路径。若"曲线缺失"
    /// 被当成增益 0，张力一开就会把整条噪声支删掉（齿音/气声/嘶声消失）。
    /// 这是本模块实际踩过的缺陷：`process_breath` 曾在 `!breath_enabled` 时提前
    /// 返回谐波，等价于把噪声静音。
    #[test]
    fn missing_breath_curve_means_unity_noise_gain() {
        assert_eq!(super::resolve_noise_gain(None), 1.0);
        assert_eq!(super::resolve_noise_gain(Some(&[])), 1.0);
        // 曲线存在时取首值
        assert_eq!(super::resolve_noise_gain(Some(&[0.25, 0.5])), 0.25);
        // 显式 0 才是"静音噪声"（用于 harmonic_only 计算）
        assert_eq!(super::resolve_noise_gain(Some(&[0.0, 0.0])), 0.0);
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
}
