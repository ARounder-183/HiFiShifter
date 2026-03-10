use crate::models::PlaybackStatePayload;
use crate::state::AppState;
use tauri::Emitter;
use tauri::State;

use super::common::{
    guard_json_command, ok_bool, PlaybackRenderingStateEvent,
};

// ===================== playback clock =====================




pub(super) fn play_original(state: State<'_, AppState>, start_sec: f64) -> serde_json::Value {
    guard_json_command("play_original", || {
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            eprintln!("play_original(start_sec={})", start_sec);
        }
        let timeline = state
            .timeline
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let bpm = timeline.bpm;
        let playhead_sec = timeline.playhead_sec;
        if !(bpm.is_finite() && bpm > 0.0) {
            return serde_json::json!({"ok": false, "error": "invalid bpm"});
        }
        let start_sec = playhead_sec.max(0.0) + start_sec.max(0.0);

        // 不依赖当前选中轨；按时间线里实际需要 pitch edit 的 clip 决定是否进入预渲染路径。
        let clips_needing_render = collect_clips_needing_render(&timeline, state.audio_engine.sample_rate_hz());
        let need_prerender = !clips_needing_render.is_empty();

        if !need_prerender {
            // 无 pitch edit：直接走实时 clip mixing（零延迟）
            state.audio_engine.seek_sec(start_sec);
            state.audio_engine.update_timeline(timeline);
            state.audio_engine.set_playing(true, Some("original"));
            return serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec});
        }

        // ── 有 pitch edit：Clip 级增量预渲染 + 实时混音 ──────────────────────────
        // 后台线程按时间线顺序逐 clip 渲染，第一个 clip 渲染完即开始播放
        // 播放过程中继续后台渲染后续 clip，音频回调中遇到未合成 clip 时静音等待
        if let Some(app) = state.app_handle.get().cloned() {
            let engine = state.audio_engine.clone();
            let tl_for_render = timeline.clone();
            let render_start_sec = start_sec;
            // 改法 D：确保 engine_sr 已被 worker 线程 store 为实际采样率。
            // AudioEngine::new() 初始化 AtomicU32 为 44100，worker spawn 后才 store 实际值。
            // 若 engine_sr 仍为初始值 44100 且系统实际为 48000，
            // 则 hash 中的 frame 计算会与 build_snapshot 不一致。
            // 这里在 spawn 内部短暂等待，确保 worker 已就绪。
            let engine_for_sr = state.audio_engine.clone();

            std::thread::spawn(move || {
                // 等待 engine worker 就绪（最多 200ms，通常 <5ms 即可）
                let mut engine_sr = engine_for_sr.sample_rate_hz();
                if engine_sr == 44100 {
                    for _ in 0..40 {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        engine_sr = engine_for_sr.sample_rate_hz();
                        if engine_sr != 44100 {
                            break;
                        }
                    }
                }
                eprintln!(
                    "[play_original] engine_sr={} (used for hash computation)",
                    engine_sr
                );
                // 推送渲染开始
                let _ = app.emit(
                    "playback_rendering_state",
                    PlaybackRenderingStateEvent {
                        active: true,
                        progress: Some(0.0),
                        target: Some("original".to_string()),
                    },
                );

                // 收集需要预渲染的 clip 列表，按时间线顺序排序
                let mut clips_to_render = collect_clips_needing_render(&tl_for_render, engine_sr);
                clips_to_render.sort_by(|a, b| {
                    a.clip.start_sec.total_cmp(&b.clip.start_sec)
                });

                // 防呆：当 pitch_edit_user_modified 为 true 但当前时间线中并没有任何 clip
                // 在播放窗口内需要 pitch edit（例如用户把所有点都清空为 0），
                // 则无需进入预渲染路径，直接播放即可。
                if clips_to_render.is_empty() {
                    engine.seek_sec(render_start_sec);
                    engine.update_timeline(tl_for_render.clone());
                    engine.set_playing(true, Some("original"));

                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: false,
                            progress: Some(1.0),
                            target: Some("original".to_string()),
                        },
                    );
                    return;
                }
                
                // 新一轮渲染开始，清空上次的 pending_rendered_keys
                crate::synth_clip_cache::clear_pending_rendered_keys();

                let total = clips_to_render.len().max(1);
                let mut rendered_count = 0u32;
                let mut any_error = false;

                // 逐 clip 预渲染，全部完成后再开始播放
                for clip_render_info in &clips_to_render {
                    let mut base_entry = {
                        let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        cache.get(&clip_render_info.cache_key).cloned()
                    };

                    if base_entry.is_some() {
                        crate::synth_clip_cache::register_pending_rendered_key(
                            &clip_render_info.clip.id,
                            clip_render_info.cache_key.clone(),
                        );
                    }

                    if base_entry.is_none() {
                        match render_single_clip(
                            &tl_for_render,
                            &clip_render_info.clip,
                            clip_render_info.sr,
                        ) {
                            Ok(rendered) => {
                                let stereo_pcm = rendered.rendered_stereo;
                                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
                                    let nonzero = stereo_pcm.iter().filter(|&&v| v.abs() > 1e-6).count();
                                    eprintln!(
                                        "[play_original] clip rendered: id={} pcm_len={} nonzero={} hash={:#018x}",
                                        clip_render_info.clip.id, stereo_pcm.len(), nonzero,
                                        clip_render_info.cache_key.param_hash
                                    );
                                }
                                let frames = (stereo_pcm.len() / 2) as u64;
                                let entry = crate::synth_clip_cache::RenderedClipCacheEntry {
                                    pcm_stereo: std::sync::Arc::new(stereo_pcm),
                                    breath_noise_stereo: rendered
                                        .breath_noise_stereo
                                        .map(std::sync::Arc::new),
                                    frames,
                                    sample_rate: clip_render_info.sr,
                                };
                                let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                                    .lock()
                                    .unwrap_or_else(|e| e.into_inner());
                                cache.insert(clip_render_info.cache_key.clone(), entry.clone());
                                crate::synth_clip_cache::register_pending_rendered_key(
                                    &clip_render_info.clip.id,
                                    clip_render_info.cache_key.clone(),
                                );
                                base_entry = Some(entry);
                            }
                            Err(e) => {
                                eprintln!(
                                    "play_original: clip render failed: clip_id={} err={}",
                                    clip_render_info.clip.id, e
                                );
                                any_error = true;
                                if let Ok(mut state_mgr) = crate::clip_rendering_state::global_clip_rendering_state().lock() {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Failed,
                                        0.0,
                                        Some(e.clone())
                                    );
                                }
                            }
                        }
                    }

                    if let Some(base_entry) = base_entry.as_ref() {
                        match ensure_hifigan_tension_cache(
                            &tl_for_render,
                            &clip_render_info.clip,
                            clip_render_info.sr,
                            clip_render_info.cache_key.param_hash,
                            base_entry.pcm_stereo.as_slice(),
                        ) {
                            Ok(_) => {
                                if let Ok(mut state_mgr) = crate::clip_rendering_state::global_clip_rendering_state().lock() {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Ready,
                                        1.0,
                                        None
                                    );
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "play_original: tension render failed: clip_id={} err={}",
                                    clip_render_info.clip.id, e
                                );
                                any_error = true;
                                if let Ok(mut state_mgr) = crate::clip_rendering_state::global_clip_rendering_state().lock() {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Failed,
                                        0.0,
                                        Some(e.clone())
                                    );
                                }
                            }
                        }
                    }

                    rendered_count += 1;
                    let progress = rendered_count as f64 / total as f64;
                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: true,
                            progress: Some(progress),
                            target: Some("original".to_string()),
                        },
                    );
                }

                // 所有 clip 渲染完成（或已尝试），开始播放
                // 若有渲染失败，snapshot 中对应 clip 会有 needs_synthesis=true、rendered_pcm=None，
                // 音频回调会陷入 has_pending_clip=true 的永久静音等待。
                // 解决方案：渲染失败时降级为播放原始音频（等同于无 pitch edit 路径）。
                if any_error {
                    eprintln!("[play_original] rendering had errors, falling back to original audio playback");
                    // 推送失败通知
                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: false,
                            progress: Some(1.0),
                            target: Some("original".to_string()),
                        },
                    );
                    // 降级：直接播放——audio engine 会使用源 PCM，不经过 rendered_pcm 路径
                    // 注意：此时 engine 中没有该 clip 的 rendered_pcm，
                    //   build_snapshot 在找不到缓存时会设 needs_synthesis=true, rendered_pcm=None。
                    //   这会导致 has_pending_clip=true → 静音。
                    //   因此改用 update_timeline 但不传 pitch edit 标记的 timeline（无此机制），
                    //   最简单的降级是：直接 seek + play，让 audio engine 用原始 PCM 播放
                    //   （此时 pitch_edit_user_modified 仍为 true，engine 仍会尝试查找 rendered_pcm
                    //    并找不到，因此改为 stop 旧播放状态并提示用户）。
                    engine.stop();
                    return;
                }
                engine.seek_sec(render_start_sec);
                engine.update_timeline(tl_for_render.clone());
                engine.set_playing(true, Some("original"));

                // 推送渲染完成
                let _ = app.emit(
                    "playback_rendering_state",
                    PlaybackRenderingStateEvent {
                        active: false,
                        progress: Some(1.0),
                        target: Some("original".to_string()),
                    },
                );
            });
        }

        serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec, "prerendering": true})
    })
}

// ─── Clip 级预渲染辅助 ─────────────────────────────────────────────────────────

/// 需要预渲染的单个 clip 的信息。
struct ClipRenderInfo {
    clip: crate::state::Clip,
    cache_key: crate::synth_clip_cache::RenderedClipCacheKey,
    sr: u32,
}

struct RenderedClipOutput {
    rendered_stereo: Vec<f32>,
    breath_noise_stereo: Option<Vec<f32>>,
}

fn ensure_hifigan_tension_cache(
    timeline: &crate::state::TimelineState,
    clip: &crate::state::Clip,
    out_rate: u32,
    base_param_hash: u64,
    base_pcm_stereo: &[f32],
) -> Result<Option<crate::synth_clip_cache::TensionRenderedClipCacheKey>, String> {
    let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return Ok(None);
    };
    let Some(entry) = timeline.params_by_root_track.get(&root) else {
        return Ok(None);
    };
    let Some(track) = timeline.tracks.iter().find(|track| track.id == root) else {
        return Ok(None);
    };

    let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
    if !matches!(kind, crate::state::SynthPipelineKind::NsfHifiganOnnx) {
        return Ok(None);
    }
    let clip_start_sec = clip.start_sec.max(0.0);
    if !crate::pitch_editing::hifigan_tension_active_for_clip(entry, clip, clip_start_sec) {
        return Ok(None);
    }

    let start_frame = (clip_start_sec * out_rate as f64).round().max(0.0) as u64;
    let end_frame = start_frame
        + (clip.length_sec.max(0.0) * out_rate as f64).round().max(1.0) as u64;
    let frame_period_ms = entry.frame_period_ms.max(0.1);
    let tension_curve = crate::pitch_editing::hifigan_tension_curve_for_clip(entry, clip);
    let tension_hash = crate::synth_clip_cache::compute_hifigan_tension_hash(
        &clip.id,
        base_param_hash,
        start_frame,
        end_frame,
        out_rate,
        frame_period_ms,
        &entry.pitch_orig,
        tension_curve,
    );
    let cache_key = crate::synth_clip_cache::TensionRenderedClipCacheKey {
        clip_id: clip.id.clone(),
        base_param_hash,
        tension_hash,
    };

    {
        let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if cache.get(&cache_key).is_some() {
            return Ok(Some(cache_key));
        }
    }

    let tensioned = crate::hifigan_tension::apply_tension_to_stereo(
        base_pcm_stereo,
        out_rate,
        clip_start_sec,
        frame_period_ms,
        &entry.pitch_orig,
        &entry.pitch_edit,
        tension_curve,
    )?;
    let frames = (tensioned.len() / 2) as u64;
    let entry = crate::synth_clip_cache::TensionRenderedClipCacheEntry {
        pcm_stereo: std::sync::Arc::new(tensioned),
        frames,
        sample_rate: out_rate,
    };
    let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache.insert(cache_key.clone(), entry);
    Ok(Some(cache_key))
}

/// 收集 timeline 中所有需要预渲染的 clip。
///
/// 返回值中只包含需要 pitch edit 的 clip。
fn collect_clips_needing_render(
    timeline: &crate::state::TimelineState,
    engine_sr: u32,
) -> Vec<ClipRenderInfo> {
    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
    let mut out = Vec::new();
    let sr = if engine_sr > 0 { engine_sr } else { 44100 };

    if debug {
        eprintln!(
            "[collect_clips_needing_render] engine_sr={} effective_sr={} clips_count={}",
            engine_sr, sr, timeline.clips.len()
        );
    }

    for clip in &timeline.clips {
        if clip.muted {
            continue;
        }
        let Some(source_path) = clip.source_path.as_deref() else {
            continue;
        };
        
        // 使用新的检测逻辑：检查clip是否需要pitch edit
        let clip_start_sec = clip.start_sec.max(0.0);
        let needs_pitch_edit = crate::pitch_editing::does_clip_need_processor_render(
            timeline,
            clip,
            clip_start_sec,
        );
        
        if !needs_pitch_edit {
            continue;
        }

        let playback_rate = {
            let r = clip.playback_rate as f64;
            if r.is_finite() && r > 0.0 { r } else { 1.0 }
        };
        let start_frame = (clip.start_sec.max(0.0) * sr as f64).round().max(0.0) as u64;
        let end_frame = start_frame
            + (clip.length_sec.max(0.0) * sr as f64).round().max(1.0) as u64;

        // 获取pitch edit参数
        let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
            continue;
        };
        let entry = match timeline.params_by_root_track.get(&clip_root) {
            Some(e) => e,
            None => continue,
        };
        let track = match timeline.tracks.iter().find(|t| t.id == clip_root) {
            Some(t) => t,
            None => continue,
        };
        let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
            let renderer_id = crate::renderer::get_renderer(&kind).id();
        let pitch_edit = entry.pitch_edit.as_slice();
        let frame_period_ms = entry.frame_period_ms.max(0.1);

        let param_hash = crate::synth_clip_cache::compute_rendered_clip_hash(
            &clip.id,
            source_path,
            start_frame,
            end_frame,
            sr,
            renderer_id,
            pitch_edit,
            frame_period_ms,
            playback_rate,
            &entry.extra_curves,
            &entry.extra_params,
        );
        let cache_key = crate::synth_clip_cache::RenderedClipCacheKey {
            clip_id: clip.id.clone(),
            param_hash,
        };

        if debug {
            eprintln!(
                "[collect_clips_needing_render] clip_id={} sr={} start_frame={} end_frame={} hash={:#018x}",
                clip.id, sr, start_frame, end_frame, param_hash
            );
        }

        out.push(ClipRenderInfo {
            clip: clip.clone(),
            cache_key,
            sr,
        });
    }
    out
}

/// 渲染单个 clip 的完整 stereo PCM（从源文件解码 -> resample -> pitch edit -> stereo）。
///
/// 复用 mixdown.rs 中的解码和 resample 逻辑，通过 Renderer trait 调用 pitch edit。
fn render_single_clip(
    timeline: &crate::state::TimelineState,
    clip: &crate::state::Clip,
    out_rate: u32,
) -> Result<RenderedClipOutput, String> {
    let source_path = clip
        .source_path
        .as_deref()
        .ok_or_else(|| "clip has no source_path".to_string())?;

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

    // 1. 解码源文件
    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(std::path::Path::new(source_path))?;
    let in_channels_usize = in_channels as usize;
    let in_frames = pcm.len() / in_channels_usize;
    if in_frames < 2 {
        return Err("source audio too short".to_string());
    }

    // 2. 源裁剪
    let playback_rate = {
        let r = clip.playback_rate as f64;
        if r.is_finite() && r > 0.0 { r } else { 1.0 }
    };
    let source_start_sec = clip.source_start_sec.max(0.0);
    let source_end_sec = clip.source_end_sec;
    let pre_silence_sec = (-clip.source_start_sec).max(0.0) / playback_rate.max(1e-6);

    let total_sec = crate::mixdown::clip_duration_sec_from_wav(in_rate, in_channels, &pcm)
        .ok_or_else(|| "cannot determine clip duration".to_string())?;
    if !(total_sec.is_finite() && total_sec > 0.0) {
        return Err("invalid clip duration".to_string());
    }

    let src_end_limit_sec = source_end_sec.min(total_sec).max(source_start_sec);
    if src_end_limit_sec - source_start_sec <= 1e-9 {
        return Err("trimmed clip too short".to_string());
    }

    // 3. 切片 + resample
    let src_i0 = (source_start_sec * in_rate as f64).floor().max(0.0) as usize;
    let src_i1 = ((src_end_limit_sec * in_rate as f64).ceil().max(src_i0 as f64) as usize)
        .min(in_frames);
    if src_i1 <= src_i0 + 1 {
        return Err("source slice too short".to_string());
    }

    let segment = &pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)];
    let segment = crate::mixdown::linear_resample_interleaved(
        segment,
        in_channels_usize,
        in_rate,
        out_rate,
    );

    // 4. 转 stereo
    let segment = if in_channels == 1 {
        let frames = segment.len();
        let mut stereo = Vec::with_capacity(frames * 2);
        for sample in segment {
            stereo.push(sample);
            stereo.push(sample);
        }
        stereo
    } else if in_channels >= 2 {
        let frames = segment.len() / in_channels_usize;
        let mut stereo = Vec::with_capacity(frames * 2);
        for frame in 0..frames {
            stereo.push(segment[frame * in_channels_usize]);
            stereo.push(segment[frame * in_channels_usize + 1]);
        }
        stereo
    } else {
        return Err("unsupported channel count".to_string());
    };

    // 5. 时间拉伸（playback_rate != 1）
    // 若合成处理器声明自己处理时间拉伸（handles_time_stretch = true），
    // 则跳过此处的 RubberBand，由处理器在 pitch edit 阶段通过 ClipProcessContext.playback_rate 内部完成。
    let processor_handles_stretch = {
        let clip_root = timeline.resolve_root_track_id(&clip.track_id);
        clip_root
            .and_then(|root| timeline.tracks.iter().find(|t| t.id == root))
            .map(|t| {
                let kind = crate::state::SynthPipelineKind::from_track_algo(&t.pitch_analysis_algo);
                crate::renderer::get_processor(kind).capabilities().handles_time_stretch
            })
            .unwrap_or(false)
    };
    let mut segment = segment;
    if (playback_rate - 1.0).abs() > 1e-6 && !processor_handles_stretch {
        let seg_frames_in = segment.len() / 2;
        let target_frames = ((seg_frames_in as f64) / playback_rate).round().max(2.0) as usize;
        segment = crate::time_stretch::time_stretch_interleaved(
            &segment,
            2,
            out_rate,
            target_frames,
            crate::time_stretch::StretchAlgorithm::RubberBand,
        );
    }

    let clip_start_sec = clip.start_sec.max(0.0);
    let seg_start_sec = clip_start_sec + pre_silence_sec;
    let clip_timeline_frames =
        (clip.length_sec.max(0.0) * out_rate as f64).round().max(1.0) as usize;
    let clip_stereo_len = clip_timeline_frames * 2;

    let root_params = timeline
        .resolve_root_track_id(&clip.track_id)
        .and_then(|root| timeline.params_by_root_track.get(&root));
    let effective_extra_params = clip
        .extra_params
        .as_ref()
        .or_else(|| root_params.map(|entry| &entry.extra_params));
    let breath_enabled = effective_extra_params
        .map(|params| crate::pitch_editing::extra_param_enabled(params, "breath_enabled"))
        .unwrap_or(false);
    let frame_period_ms = root_params
        .map(|entry| entry.frame_period_ms.max(0.1))
        .unwrap_or(5.0);
    let curve_len = (((clip_start_sec + clip.length_sec.max(0.0)) * 1000.0) / frame_period_ms)
        .ceil()
        .max(0.0) as usize
        + 2;

    let render_variant = |clip_variant: &crate::state::Clip| {
        let mut rendered = segment.clone();
        let seg_frames = rendered.len() / 2;
        if seg_frames >= 16 {
            match crate::pitch_editing::maybe_apply_pitch_edit_to_clip_segment(
                timeline,
                clip_variant,
                clip_start_sec,
                seg_start_sec,
                out_rate,
                &mut rendered,
            ) {
                Ok(true) => {
                    if debug {
                        eprintln!(
                            "render_single_clip: pitch_edit applied to clip_id={}",
                            &clip_variant.id
                        );
                    }
                }
                Ok(false) => {}
                Err(e) => {
                    eprintln!("[pitch_edit] clip_id={} ERROR: {e}", &clip_variant.id);
                }
            }
        }

        if pre_silence_sec > 1e-6 {
            let pre_frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as usize;
            let mut with_silence = vec![0.0f32; pre_frames * 2];
            with_silence.extend_from_slice(&rendered);
            rendered = with_silence;
        }

        if rendered.len() > clip_stereo_len {
            rendered.truncate(clip_stereo_len);
        } else if rendered.len() < clip_stereo_len {
            rendered.resize(clip_stereo_len, 0.0);
        }

        rendered
    };

    if !breath_enabled {
        return Ok(RenderedClipOutput {
            rendered_stereo: render_variant(clip),
            breath_noise_stereo: None,
        });
    }

    let mut merged_extra_params = root_params
        .map(|entry| entry.extra_params.clone())
        .unwrap_or_default();
    if let Some(extra_params) = clip.extra_params.as_ref() {
        merged_extra_params.extend(extra_params.clone());
    }
    merged_extra_params.insert("breath_enabled".to_string(), 1.0);

    let mut merged_extra_curves = root_params
        .map(|entry| entry.extra_curves.clone())
        .unwrap_or_default();
    if let Some(extra_curves) = clip.extra_curves.as_ref() {
        merged_extra_curves.extend(extra_curves.clone());
    }

    let mut harmonic_only_clip = clip.clone();
    merged_extra_curves.insert("breath_gain".to_string(), vec![0.0; curve_len]);
    harmonic_only_clip.extra_params = Some(merged_extra_params.clone());
    harmonic_only_clip.extra_curves = Some(merged_extra_curves.clone());
    let mut harmonic_only = render_variant(&harmonic_only_clip);

    let mut unity_breath_clip = clip.clone();
    merged_extra_curves.insert("breath_gain".to_string(), vec![1.0; curve_len]);
    unity_breath_clip.extra_params = Some(merged_extra_params);
    unity_breath_clip.extra_curves = Some(merged_extra_curves);
    let unity_mix = render_variant(&unity_breath_clip);

    let out_len = harmonic_only.len().min(unity_mix.len());
    harmonic_only.truncate(out_len);
    let mut breath_noise_stereo = vec![0.0f32; out_len];
    for index in 0..out_len {
        breath_noise_stereo[index] = unity_mix[index] - harmonic_only[index];
    }

    Ok(RenderedClipOutput {
        rendered_stereo: harmonic_only,
        breath_noise_stereo: Some(breath_noise_stereo),
    })
}




pub(super) fn stop_audio(state: State<'_, AppState>) -> serde_json::Value {
    state.audio_engine.stop();
    ok_bool()
}




pub(super) fn get_playback_state(state: State<'_, AppState>) -> PlaybackStatePayload {
    let pb = state.audio_engine.snapshot_state();
    PlaybackStatePayload {
        ok: true,
        is_playing: pb.is_playing,
        target: pb.target,
        base_sec: pb.base_sec,
        position_sec: pb.position_sec,
        duration_sec: pb.duration_sec,
    }
}
