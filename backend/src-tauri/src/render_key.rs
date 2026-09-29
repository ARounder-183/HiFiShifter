//! 整 Clip 渲染键（render key）的**唯一**构建处。
//!
//! 【为什么单独成模块】渲染缓存（内存 `RenderedClipCache` + 磁盘 `render_cache`）的
//! 键必须来自同一份输入口径：任何一处参数漂移都会让"写入的键"与"查询的键"不一致 ——
//! 轻则缓存永久 miss，重则跨参数误命中（把为别的参数渲染的音频当成本次结果）。
//!
//! 这份口径原先只存在于 `commands/playback.rs` 的私有函数里。把它上提为独立模块的
//! 目的是：**让导出等新的消费方也能算出与预览完全一致的键**，而不是各写一份。
//! 导出若要复用渲染缓存，这是前置条件（见 `docs/plans/2026-09-28-export-render-cache-reuse-and-progress-cancel.md`）。
//!
//! # 契约
//! - 键由三部分组成：clip 自身（源身份 / 时间与源窗口 / 速率 / 倒放 / Loop / 声道 /
//!   formant / Take）、根轨道的参数状态（pitch 曲线切片、extra 曲线与参数、帧周期）、
//!   以及调用方传入的渲染器 id、采样率、Compose 开关与音阶签名；随后由
//!   [`crate::synth_clip_cache::compute_rendered_clip_hash`] 混入**管线指纹**与
//!   拉伸设置。
//! - "这个 clip 是否参与渲染"由 [`resolve_render_material`] 统一回答
//!   （muted / 无源 / 无需处理器渲染 → 不参与）。它同时是"能否复用缓存"的天然门禁。
//!
//! # 特殊说明（给将来的导出复用）
//! 键**不包含任何"处理窗口"信息**：导出范围、Loop 平铺的窗口量化、前导静音的对齐
//! 方式都不在键里。所以"键相同"只说明**输入相同**；消费方还必须自行保证
//! "自己处理的那段音频 == 缓存里存的那段"（缓存存的是整条 clip、clip 局部帧 0 起、
//! 含前导静音），否则仍会误用。这条约束是导出复用设计里最容易漏掉的一环。

use crate::state::{Clip, TimelineState, Track, TrackParamsState};
use crate::synth_clip_cache::RenderedClipHashInput;

/// Clip 播放速率（非法值按 1.0 处理，与实时引擎口径一致）。
pub(crate) fn clip_playback_rate(clip: &Clip) -> f64 {
    let rate = clip.playback_rate as f64;
    if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        1.0
    }
}

/// clip 的渲染材料：根轨道的参数与轨道本身。
pub(crate) struct ClipRenderMaterial<'a> {
    pub(crate) entry: &'a TrackParamsState,
    pub(crate) track: &'a Track,
}

/// 解析 clip 的渲染材料，并回答"这个 clip 当前是否需要渲染"。
///
/// ★ 这是该判定的**唯一实现**：收集待渲染（热路径）与 miss 归因诊断都必须经由
/// 它。两处各写一份必然漂移，而漂移的后果是"写入的键"与"查询的键"不一致 ——
/// 轻则永久 miss，重则跨参数误命中（见本模块与 `synth_clip_cache` 的契约）。
///
/// `find_track` 由调用方注入：热路径传预构建的 O(1) 查找表，诊断路径传线性查找
/// （每次 miss 至多一次，轨道数是常数级）。
pub(crate) fn resolve_render_material<'a>(
    timeline: &'a TimelineState,
    clip: &Clip,
    find_track: impl Fn(&str) -> Option<&'a Track>,
) -> Option<ClipRenderMaterial<'a>> {
    if clip.muted || clip.source_path.is_none() {
        return None;
    }
    // 使用新的检测逻辑：检查 clip 是否需要 pitch edit
    let clip_start_sec = clip.start_sec.max(0.0);
    if !crate::pitch_editing::does_clip_need_processor_render(timeline, clip, clip_start_sec) {
        return None;
    }
    // 获取 pitch edit 参数（按根轨道）
    let clip_root = timeline.resolve_root_track_id(&clip.track_id)?;
    let entry = timeline.params_by_root_track.get(&clip_root)?;
    let track = find_track(&clip_root)?;
    Some(ClipRenderMaterial { entry, track })
}

/// 构造整 Clip 渲染哈希输入。
///
/// ★ 所有需要计算渲染缓存键的位置（收集待渲染、气声噪声键、快照回退，以及将来的
/// 导出复用）都必须经由本函数：任何一处参数口径漂移都会让"写入的键"与"查询的键"
/// 不一致 —— 轻则缓存永久 miss，重则跨参数误命中。
pub(crate) fn build_rendered_hash_input<'a>(
    clip: &'a Clip,
    entry: &'a TrackParamsState,
    renderer_id: &'a str,
    sr: u32,
    input_pitch_curve: Option<&'a [f32]>,
    compose_enabled: bool,
    scale_signature: &'a str,
) -> RenderedClipHashInput<'a> {
    let start_frame = (clip.start_sec.max(0.0) * sr as f64).round() as u64;
    let end_frame = start_frame + (clip.length_sec.max(0.0) * sr as f64).round().max(1.0) as u64;

    RenderedClipHashInput {
        clip_id: &clip.id,
        source_path: clip.source_path.as_deref().unwrap_or(""),
        source_file_mtime: clip.source_file_mtime,
        source_file_fingerprint: clip.source_file_fingerprint,
        active_take_id: clip.active_take_id.as_deref(),
        renderer_id,
        start_frame,
        end_frame,
        sample_rate: sr,
        playback_rate: clip_playback_rate(clip),
        reversed: clip.reversed,
        loop_enabled: clip.loop_enabled,
        channel_mode: clip.channel_mode,
        source_range_q: (
            (clip.source_start_sec * 1000.0).round() as i64,
            (clip.source_end_sec * 1000.0).round() as i64,
        ),
        pitch_edit: entry.pitch_edit.as_slice(),
        pitch_orig: Some(entry.pitch_orig.as_slice()),
        frame_period_ms: entry.frame_period_ms.max(0.1),
        extra_curves: &entry.extra_curves,
        extra_params: &entry.extra_params,
        formant_morph: clip.formant_morph.as_ref().filter(|params| params.enabled),
        input_pitch_curve,
        compose_enabled,
        scale_signature,
        source_file_size: clip.source_file_size,
    }
}

/// 组装单个 clip 的渲染键输入（与收集、快照回退共用同一口径）。
pub(crate) fn rendered_hash_input_for_clip<'a>(
    timeline: &'a TimelineState,
    clip: &'a Clip,
    sr: u32,
    scale_signature: &'a str,
) -> Option<RenderedClipHashInput<'a>> {
    let material = resolve_render_material(timeline, clip, |id| {
        timeline.tracks.iter().find(|t| t.id == id)
    })?;
    let kind =
        crate::state::SynthPipelineKind::from_track_algo(&material.track.pitch_analysis_algo);
    let renderer_id = crate::renderer::get_renderer(kind).id();
    Some(build_rendered_hash_input(
        clip,
        material.entry,
        renderer_id,
        sr,
        None,
        material.track.compose_enabled,
        scale_signature,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ClipFormantMorph, TrackParamsState};
    use crate::synth_clip_cache::compute_rendered_clip_hash;
    use std::collections::HashMap;

    const SR: u32 = 44_100;
    const RENDERER: &str = "nsf_hifigan_onnx";
    const SCALE: &str = "scale-signature";

    /// 基准 clip：字段取"常见值"，与 `pitch_editing` 的测试夹具同形。
    fn base_clip() -> Clip {
        Clip {
            id: "clip-a".to_string(),
            takes: vec![],
            active_take_id: Some("take-1".to_string()),
            clip_playback_rate: 1.0,
            track_id: "track-a".to_string(),
            name: "Clip".to_string(),
            start_sec: 0.0,
            length_sec: 2.0,
            color: "blue".to_string(),
            source_path: Some("a.wav".to_string()),
            source_path_relative: None,
            duration_sec: Some(2.0),
            duration_frames: None,
            source_sample_rate: Some(SR),
            source_file_mtime: Some(1_700_000_000),
            source_file_size: Some(123_456),
            source_file_fingerprint: Some(0xDEAD_BEEF),
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 2.0,
            playback_rate: 1.0,
            reversed: false,
            channel_mode: 0,
            source_channels: None,
            loop_enabled: false,
            snap_offset_sec: 0.0,
            fade_in_sec: 0.0,
            fade_out_sec: 0.0,
            fade_in_curve: "sine".to_string(),
            fade_out_curve: "sine".to_string(),
            fade_in_shape: 0.0,
            fade_out_shape: 0.0,
            fade_in_dir: 0.0,
            fade_out_dir: 0.0,
            auto_fade_in_sec: 0.0,
            auto_fade_out_sec: 0.0,
            extra_curves: None,
            extra_params: None,
            formant_morph: None,
            group_id: None,
            midi_fill_gaps: false,
            midi_note_data: None,
        }
    }

    /// 基准参数：带两条非空曲线，避免"空切片"让某些字段变化不可见。
    fn base_entry() -> TrackParamsState {
        let mut entry = TrackParamsState::default();
        entry.frame_period_ms = 5.0;
        entry.pitch_orig = vec![0.0, 1.0, 2.0, 3.0];
        entry.pitch_edit = vec![0.0, 1.0, 2.0, 3.0];
        entry
    }

    fn hash_of(clip: &Clip, entry: &TrackParamsState) -> u64 {
        let input = build_rendered_hash_input(clip, entry, RENDERER, SR, None, true, SCALE);
        compute_rendered_clip_hash(&input)
    }

    #[test]
    fn identical_inputs_produce_identical_hash() {
        assert_eq!(
            hash_of(&base_clip(), &base_entry()),
            hash_of(&base_clip(), &base_entry())
        );
    }

    /// 键的**敏感性**：任一影响渲染结果的输入变化都必须改变哈希。
    ///
    /// 【为什么逐项钉住】键一旦漏掉某个渲染输入，两种不同参数的渲染就会共用同一个
    /// 缓存条目 —— 缓存会把为 A 参数渲染的音频当作 B 参数的结果返回，属于本模块
    /// 契约里最严重的"跨参数误命中"。新增渲染输入时，必须同时在这里补一行。
    #[test]
    fn every_render_input_changes_the_hash() {
        let base = hash_of(&base_clip(), &base_entry());

        macro_rules! assert_changes {
            ($label:expr, $hash:expr) => {
                assert_ne!(
                    base, $hash,
                    "渲染输入 `{}` 未进入键：缓存会跨参数误命中",
                    $label
                );
            };
        }

        // ── clip 侧 ──
        let mut c = base_clip();
        c.id = "clip-b".to_string();
        assert_changes!("clip_id", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_path = Some("b.wav".to_string());
        assert_changes!("source_path", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_file_mtime = Some(1_700_000_001);
        assert_changes!("source_file_mtime", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_file_size = Some(999);
        assert_changes!("source_file_size", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_file_fingerprint = Some(0xFEED_FACE);
        assert_changes!("source_file_fingerprint", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.active_take_id = Some("take-2".to_string());
        assert_changes!("active_take_id", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.start_sec = 1.0;
        assert_changes!("start_sec(→start_frame)", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.length_sec = 3.0;
        assert_changes!("length_sec(→end_frame)", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.playback_rate = 1.5;
        assert_changes!("playback_rate", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.reversed = true;
        assert_changes!("reversed", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.loop_enabled = true;
        assert_changes!("loop_enabled", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.channel_mode = 2;
        assert_changes!("channel_mode", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_start_sec = 0.5;
        assert_changes!("source_start_sec", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.source_end_sec = 1.5;
        assert_changes!("source_end_sec", hash_of(&c, &base_entry()));

        let mut c = base_clip();
        c.formant_morph = Some(ClipFormantMorph {
            enabled: true,
            target_f1_hz: 700.0,
            target_f2_hz: 1_200.0,
            strength: 0.5,
        });
        assert_changes!("formant_morph", hash_of(&c, &base_entry()));

        // ── 参数侧 ──
        let mut e = base_entry();
        e.pitch_edit[2] = 9.0;
        assert_changes!("pitch_edit", hash_of(&base_clip(), &e));

        let mut e = base_entry();
        e.pitch_orig[2] = 9.0;
        assert_changes!("pitch_orig", hash_of(&base_clip(), &e));

        let mut e = base_entry();
        e.frame_period_ms = 10.0;
        assert_changes!("frame_period_ms", hash_of(&base_clip(), &e));

        let mut e = base_entry();
        e.extra_curves = HashMap::from([("formant_shift_cents".to_string(), vec![0.0, 0.5])]);
        assert_changes!(
            "extra_curves(formant_shift_cents)",
            hash_of(&base_clip(), &e)
        );

        let mut e = base_entry();
        e.extra_params = HashMap::from([("breath_enabled".to_string(), 1.0)]);
        assert_changes!("extra_params", hash_of(&base_clip(), &e));

        // ── 调用方上下文 ──
        // 注意：`RenderedClipHashInput` 借用 clip / entry，必须先落成局部变量，
        // 否则返回值的生命周期长于临时量。
        let clip = base_clip();
        let entry = base_entry();

        let renderer_input =
            build_rendered_hash_input(&clip, &entry, "world_vocoder", SR, None, true, SCALE);
        assert_changes!("renderer_id", compute_rendered_clip_hash(&renderer_input));

        let other_sr =
            build_rendered_hash_input(&clip, &entry, RENDERER, 48_000, None, true, SCALE);
        assert_changes!("sample_rate", compute_rendered_clip_hash(&other_sr));

        let compose_off =
            build_rendered_hash_input(&clip, &entry, RENDERER, SR, None, false, SCALE);
        assert_changes!("compose_enabled", compute_rendered_clip_hash(&compose_off));

        let other_scale =
            build_rendered_hash_input(&clip, &entry, RENDERER, SR, None, true, "other-scale");
        assert_changes!("scale_signature", compute_rendered_clip_hash(&other_scale));

        let curve = [0.0f32, 0.25, 0.5];
        let with_curve =
            build_rendered_hash_input(&clip, &entry, RENDERER, SR, Some(&curve), true, SCALE);
        assert_changes!("input_pitch_curve", compute_rendered_clip_hash(&with_curve));
    }

    /// 反向契约：**刻意不参与**本键的输入必须留在键外。
    ///
    /// 【为什么也要钉住】`breath_gain` / `hifigan_tension` 有各自的缓存键（噪声 stem /
    /// 张力变体，见 `synth_clip_cache::include_rendered_extra_curve`），把它们并进渲染键
    /// 会让"只调了气声"也触发整段重渲染 —— 那是本缓存设计里被明确避免的开销。
    /// 反过来，`volume` / `pan` / `dyn` 在混音阶段实时应用，同样不该进键。
    #[test]
    fn deliberately_excluded_inputs_stay_out_of_the_key() {
        let base = hash_of(&base_clip(), &base_entry());

        for excluded in [
            "breath_gain",
            "hifigan_tension",
            // 共通混音参数：实时应用，不烘焙进渲染。
            "volume",
            "pan",
            "dyn",
        ] {
            let mut e = base_entry();
            e.extra_curves = HashMap::from([(excluded.to_string(), vec![0.0, 0.5])]);
            assert_eq!(
                base,
                hash_of(&base_clip(), &e),
                "`{excluded}` 必须留在渲染键外（它有独立缓存键或在混音阶段实时应用）"
            );
        }
    }

    #[test]
    fn invalid_playback_rate_falls_back_to_one() {
        let mut c = base_clip();
        c.playback_rate = 0.0;
        assert_eq!(clip_playback_rate(&c), 1.0);
        c.playback_rate = f32::NAN;
        assert_eq!(clip_playback_rate(&c), 1.0);
        c.playback_rate = -2.0;
        assert_eq!(clip_playback_rate(&c), 1.0);
        c.playback_rate = 0.5;
        assert_eq!(clip_playback_rate(&c), 0.5);
    }
}
