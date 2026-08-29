use crate::models::PlaybackStatePayload;
use crate::state::AppState;
use tauri::Emitter;
use tauri::Manager;
use tauri::State;

use super::common::{guard_json_command, ok_bool, PlaybackRenderingStateEvent};

/// 全局后台渲染激活标志。
/// 当用户在"选项→推理设备"中启用"后台预渲染"后，编辑操作会触发
/// `start_background_render`，此标志置为 true；渲染完成（或被取消）后复原。
/// 引擎 worker 线程与音频回调均通过此标志判断是否应跳过对未渲染 clip 的暂停。
pub(crate) static BG_RENDER_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 用户是否在设置中启用了"后台预渲染"。
/// 由 `ui_settings.rs` 在加载/保存设置时同步。
/// 引擎 worker 在使缓存失效后检查此标志，若为 true 则自动启动后台渲染。
pub(crate) static AUTO_BG_RENDER_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 后台渲染取消标志。当用户在渲染中重新编辑参数时，
/// `handle_update_timeline` 设置此标志以中断旧渲染线程。
pub(crate) static BG_RENDER_CANCEL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 后台渲染重启标志。旧渲染被取消后，若此标志为 true，
/// 退出线程将自动启动新一轮渲染。
pub(crate) static BG_RENDER_RESTART_NEEDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 后台渲染开始时因音高分析未完成而跳过了 clip。
/// 音高分析完成后由 `handle_clip_pitch_ready` 消费此标记并自动补启动渲染。
pub(crate) static BG_RENDER_PITCH_PENDING: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 后台渲染代数。每次取消或启动都会递增；旧渲染线程结束时只有在代数仍匹配时
/// 才允许清理全局标志，避免取消旧渲染后开新一轮时被旧线程把新状态清掉。
pub(crate) static BG_RENDER_GENERATION: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// `render_single_clip` 内部检测到后台渲染取消时返回的错误标记。
const BG_RENDER_CANCELLED_ERR: &str = "bg_render_cancelled";

fn timeline_version_from_app(app: &tauri::AppHandle) -> u64 {
    let state = app.state::<AppState>();
    state
        .timeline_version
        .load(std::sync::atomic::Ordering::Acquire)
}

/// 一轮后台渲染结束后是否应该“补一轮”。
///
/// 补轮的目的是：第一轮可能因为音高分析尚未完成而跳过了部分 clip，
/// 而这些 clip 在分析完成后并没有新的“缓存失效”事件，因此主动再跑一轮，
/// 把它们也渲染进缓存。
///
/// ★ 必须要求本轮有实际进展（render_success_count > 0）：
/// `collect_clips_needing_render` 不会排除已命中渲染缓存的 clip，
/// 若无条件补轮，那些“永远无法就绪”的 clip（音高分析不可用、源文件缺失、
/// 非合成轨道上的手动音高编辑等）会让 skipped_not_ready 永远大于 0，
/// 每轮都“渲染成功 → 补轮 → 全部命中缓存 → 补轮 → …”，形成 100% CPU 的
/// 无限后台渲染循环，把整个应用拖到未响应。当本轮没有任何新渲染成功时，
/// 再补一轮也不可能有进展，必须停止；之后由音高分析完成事件
/// （`handle_clip_pitch_ready` 消费 `BG_RENDER_PITCH_PENDING`）主动补触发。
fn should_follow_up_render(skipped_not_ready: usize, render_success_count: u32) -> bool {
    skipped_not_ready > 0 && render_success_count > 0
}

/// 检查 clip 的音高分析是否完成（clip_midi 非空）。
///
/// 当音高分析未完成时，不应将渲染结果存入 RenderedClipCache，
/// 否则后续 snapshot rebuild 会命中这个"未编辑"的缓存，导致音高编辑不生效。
fn is_clip_pitch_analysis_ready(
    timeline: &crate::state::TimelineState,
    clip: &crate::state::Clip,
) -> bool {
    let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return false;
    };
    let Some(entry) = timeline.params_by_root_track.get(&clip_root) else {
        return false;
    };
    // 检查 clip_pitch（原始 MIDI 曲线）是否已分析
    let clip_pitch = crate::pitch_clip::get_or_compute_clip_pitch_midi_global(
        timeline,
        clip,
        &clip_root,
        entry.frame_period_ms.max(0.1),
    );
    clip_pitch.is_some()
}

pub(super) fn play_original(state: State<'_, AppState>, start_sec: f64) -> serde_json::Value {
    guard_json_command("play_original", || {
        eprintln!("[play_original] called start_sec={start_sec}");
        let timeline = match state.timeline.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        let render_timeline_version = state
            .timeline_version
            .load(std::sync::atomic::Ordering::Acquire);
        let bpm = timeline.bpm;
        let playhead_sec = timeline.playhead_sec;
        if !(bpm.is_finite() && bpm > 0.0) {
            return serde_json::json!({"ok": false, "error": "invalid bpm"});
        }
        let start_sec = playhead_sec.max(0.0) + start_sec.max(0.0);

        // ── 后台预渲染激活时：立即开始播放，不等待渲染 ──────────────────────
        let bg_render = BG_RENDER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed);
        if bg_render {
            eprintln!(
                "[play_original] background render active — playing immediately (non-blocking)"
            );
            // 更新引擎 snapshot，让已渲染完成的 clip 立即可用
            state.audio_engine.seek_sec(start_sec);
            state.audio_engine.update_timeline(timeline);
            state.audio_engine.set_playing(true, Some("original"));
            return serde_json::json!({
                "ok": true,
                "playing": "original",
                "start_sec": start_sec,
                "background_render_active": true
            });
        }

        let clips_needing_render =
            collect_clips_needing_render(&timeline, state.audio_engine.sample_rate_hz());
        let need_prerender = !clips_needing_render.is_empty();
        eprintln!(
            "[play_original] clips_needing_render={} need_prerender={} timeline_version={}",
            clips_needing_render.len(),
            need_prerender,
            render_timeline_version
        );

        // Check if the engine's current snapshot has clips awaiting synthesis
        // (rendered_pcm=None, needs_synthesis=true).  This can happen when
        // pitch_edit data was just invalidated but collect_clips_needing_render
        // returned empty because user_modified hadn't propagated yet.
        let snapshot_has_pending = state.audio_engine.snapshot_has_pending_clips();

        if !need_prerender && !snapshot_has_pending {
            // 无 pitch edit：直接走实时 clip mixing（零延迟）
            state.audio_engine.seek_sec(start_sec);
            state.audio_engine.update_timeline(timeline);
            state.audio_engine.set_playing(true, Some("original"));
            return serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec});
        }

        if !need_prerender && snapshot_has_pending {
            eprintln!("[play_original] clips_needing_render=0 but snapshot has pending synthesis — waiting for render");
            state.audio_engine.seek_sec(start_sec);
            state.audio_engine.update_timeline(timeline);
            if let Some(app) = state.app_handle.get().cloned() {
                let _ = app.emit(
                    "playback_rendering_state",
                    PlaybackRenderingStateEvent {
                        active: true,
                        progress: Some(0.0),
                        target: Some("original".to_string()),
                    },
                );
            }
            return serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec, "waiting_for_render": true});
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
                let cache_log = std::env::var("HIFISHIFTER_RENDER_CACHE_LOG")
                    .ok()
                    .as_deref()
                    == Some("1");
                let play_started_at = std::time::Instant::now();

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
                let rendering_state_active = true;

                // Set up chunk-level progress callback for granular UI updates
                let app_for_progress = app.clone();
                crate::nsf_hifigan_onnx::set_chunk_progress_callback(Some(Box::new(
                    move |progress: f64| {
                        let _ = app_for_progress.emit(
                            "playback_rendering_state",
                            PlaybackRenderingStateEvent {
                                active: true,
                                progress: Some(progress),
                                target: Some("original".to_string()),
                            },
                        );
                    },
                )));

                let _ = app.emit(
                    "playback_rendering_state",
                    PlaybackRenderingStateEvent {
                        active: true,
                        progress: Some(0.0),
                        target: Some("original".to_string()),
                    },
                );

                // 收集需要预渲染的 clip 列表，按时间线顺序排序
                let collect_started_at = std::time::Instant::now();
                let mut clips_to_render = collect_clips_needing_render(&tl_for_render, engine_sr);
                clips_to_render.sort_by(|a, b| a.clip.start_sec.total_cmp(&b.clip.start_sec));
                let collect_elapsed = collect_started_at.elapsed();

                let ready_filter_started_at = std::time::Instant::now();
                clips_to_render
                    .retain(|info| is_clip_pitch_analysis_ready(&tl_for_render, &info.clip));
                let ready_filter_elapsed = ready_filter_started_at.elapsed();

                clips_to_render.sort_by(|a, b| a.clip.start_sec.total_cmp(&b.clip.start_sec));

                if cache_log {
                    eprintln!(
                        "[play_original][cache] prerender_targets={} engine_sr={} collect_ms={:.2} ready_filter_ms={:.2}",
                        clips_to_render.len(),
                        engine_sr,
                        collect_elapsed.as_secs_f64() * 1000.0,
                        ready_filter_elapsed.as_secs_f64() * 1000.0
                    );
                }

                // 防呆：当 pitch_edit_user_modified 为 true 但当前时间线中并没有任何 clip
                // 在播放窗口内需要 pitch edit（例如用户把所有点都清空为 0），
                // 则无需进入预渲染路径，直接播放即可。
                if clips_to_render.is_empty() {
                    if timeline_version_from_app(&app) != render_timeline_version {
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
                    engine.seek_sec(render_start_sec);
                    engine.update_timeline(tl_for_render);
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

                // 预渲染批次保护：按本轮 clip 数动态扩容缓存，
                // 避免同一轮中早先渲染好的条目被后续插入提前淘汰。
                {
                    let mut rendered_cache = crate::synth_clip_cache::global_rendered_clip_cache()
                        .lock()
                        .unwrap_or_else(|e| e.into_inner());
                    let required = rendered_cache.len().saturating_add(clips_to_render.len());
                    rendered_cache.ensure_capacity(required);
                }
                {
                    let mut tension_cache =
                        crate::synth_clip_cache::global_tension_rendered_clip_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                    let required = clips_to_render.len().max(1);
                    tension_cache.ensure_capacity(required);
                }
                // 动态扩容 HNSEP 分离缓存：确保容量 >= 本轮 clip 数 + 余量，
                // 避免大量切片场景下 LRU 驱逐导致重复执行 HNSEP 推理。
                {
                    let breath_clips = clips_to_render.len();
                    // 预留 25% 余量，至少 128
                    let required = (breath_clips + breath_clips / 4).max(128);
                    crate::hnsep_onnx::ensure_cache_capacity(required);
                }
                // 动态扩容 Breath Noise 独立缓存：确保容量 >= 本轮 clip 数，
                // 使 formant 编辑时可复用已缓存的 noise stem，避免重复 HNSEP 推理。
                {
                    let mut breath_noise_cache =
                        crate::synth_clip_cache::global_breath_noise_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                    let required = clips_to_render.len().max(1);
                    breath_noise_cache.ensure_capacity(required);
                }

                let total = clips_to_render.len().max(1);

                crate::nsf_hifigan_onnx::reset_chunk_progress(total);

                let mut rendered_count = 0u32;
                let mut cache_hit_count = 0u32;
                let mut cache_miss_count = 0u32;
                let mut render_success_count = 0u32;
                let mut render_failed_count = 0u32;
                let mut cache_probe_elapsed = std::time::Duration::ZERO;
                let mut render_elapsed = std::time::Duration::ZERO;
                let mut tension_elapsed = std::time::Duration::ZERO;
                let mut timeline_sig_check_elapsed = std::time::Duration::ZERO;
                let mut any_error = false;
                let mut cancelled = false;
                let mut pending_clip_ids_written: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                // 逐 clip 预渲染，全部完成后再开始播放
                for clip_render_info in &clips_to_render {
                    if rendered_count % 32 == 0 {
                        let sig_check_started_at = std::time::Instant::now();
                        let changed = timeline_version_from_app(&app) != render_timeline_version;
                        timeline_sig_check_elapsed += sig_check_started_at.elapsed();
                        if changed {
                            cancelled = true;
                            break;
                        }
                    }

                    let cache_probe_started_at = std::time::Instant::now();
                    let mut base_entry = {
                        let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        cache.get(&clip_render_info.cache_key).cloned()
                    };
                    cache_probe_elapsed += cache_probe_started_at.elapsed();

                    // 由于上面已经通过 retain 过滤过了，这里直接放行
                    if base_entry.is_some() {
                        cache_hit_count += 1;
                        if cache_log {
                            eprintln!(
                                "[play_original][cache] HIT clip_id={} hash={:#018x}",
                                clip_render_info.clip.id, clip_render_info.cache_key.param_hash
                            );
                        }
                        crate::synth_clip_cache::register_pending_rendered_key(
                            &clip_render_info.clip.id,
                            clip_render_info.cache_key.clone(),
                        );
                        pending_clip_ids_written.insert(clip_render_info.clip.id.clone());
                    }

                    if base_entry.is_none() {
                        cache_miss_count += 1;
                        if cache_log {
                            eprintln!(
                                "[play_original][cache] MISS clip_id={} hash={:#018x}",
                                clip_render_info.clip.id, clip_render_info.cache_key.param_hash
                            );
                        }
                        if let Ok(mut state_mgr) =
                            crate::clip_rendering_state::global_clip_rendering_state().lock()
                        {
                            state_mgr.set_state(
                                &clip_render_info.clip.id,
                                crate::clip_rendering_state::ClipRenderingState::Rendering,
                                0.0,
                                None,
                            );
                        }

                        let render_started_at = std::time::Instant::now();
                        match render_single_clip(
                            &tl_for_render,
                            &clip_render_info.clip,
                            clip_render_info.sr,
                        ) {
                            Ok(rendered) => {
                                // render_single_clip 涵盖解码、resample、可选 stretch、pitch processor。
                                render_elapsed += render_started_at.elapsed();
                                let stereo_pcm = rendered.rendered_stereo;
                                if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref()
                                    == Some("1")
                                {
                                    let nonzero =
                                        stereo_pcm.iter().filter(|&&v| v.abs() > 1e-6).count();
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
                                    rendered_take_id: clip_render_info.clip.active_take_id.clone(),
                                };

                                // 现在存入缓存
                                let mut cache =
                                    crate::synth_clip_cache::global_rendered_clip_cache()
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner());
                                cache.insert(clip_render_info.cache_key.clone(), entry.clone());
                                crate::synth_clip_cache::register_pending_rendered_key(
                                    &clip_render_info.clip.id,
                                    clip_render_info.cache_key.clone(),
                                );
                                pending_clip_ids_written.insert(clip_render_info.clip.id.clone());

                                base_entry = Some(entry);
                                render_success_count += 1;
                            }
                            Err(e) => {
                                render_elapsed += render_started_at.elapsed();
                                eprintln!(
                                    "play_original: clip render failed: clip_id={} err={}",
                                    clip_render_info.clip.id, e
                                );
                                any_error = true;
                                render_failed_count += 1;
                                if let Ok(mut state_mgr) =
                                    crate::clip_rendering_state::global_clip_rendering_state()
                                        .lock()
                                {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Failed,
                                        0.0,
                                        Some(e.clone()),
                                    );
                                }
                            }
                        }
                    }

                    if let Some(base_entry) = base_entry.as_ref() {
                        let tension_started_at = std::time::Instant::now();
                        match ensure_hifigan_tension_cache(
                            &tl_for_render,
                            &clip_render_info.clip,
                            clip_render_info.sr,
                            clip_render_info.cache_key.param_hash,
                            base_entry.pcm_stereo.as_slice(),
                        ) {
                            Ok((_, _tension_generated)) => {
                                tension_elapsed += tension_started_at.elapsed();
                                if let Ok(mut state_mgr) =
                                    crate::clip_rendering_state::global_clip_rendering_state()
                                        .lock()
                                {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Ready,
                                        1.0,
                                        None,
                                    );
                                }
                            }
                            Err(e) => {
                                tension_elapsed += tension_started_at.elapsed();
                                eprintln!(
                                    "play_original: tension render failed: clip_id={} err={}",
                                    clip_render_info.clip.id, e
                                );
                                any_error = true;
                                if let Ok(mut state_mgr) =
                                    crate::clip_rendering_state::global_clip_rendering_state()
                                        .lock()
                                {
                                    state_mgr.set_state(
                                        &clip_render_info.clip.id,
                                        crate::clip_rendering_state::ClipRenderingState::Failed,
                                        0.0,
                                        Some(e.clone()),
                                    );
                                }
                            }
                        }
                    }

                    rendered_count += 1;
                }

                if cancelled {
                    crate::nsf_hifigan_onnx::set_chunk_progress_callback(None);
                    if cache_log {
                        eprintln!(
                            "[play_original][cache] CANCELLED total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} total_ms={:.2}",
                            clips_to_render.len(),
                            cache_hit_count,
                            cache_miss_count,
                            render_success_count,
                            render_failed_count,
                            cache_probe_elapsed.as_secs_f64() * 1000.0,
                            render_elapsed.as_secs_f64() * 1000.0,
                            tension_elapsed.as_secs_f64() * 1000.0,
                            play_started_at.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    for clip_id in pending_clip_ids_written {
                        crate::synth_clip_cache::remove_pending_rendered_key(&clip_id);
                    }
                    if rendering_state_active {
                        let _ = app.emit(
                            "playback_rendering_state",
                            PlaybackRenderingStateEvent {
                                active: false,
                                progress: Some(1.0),
                                target: Some("original".to_string()),
                            },
                        );
                    }
                    return;
                }

                // 所有 clip 渲染完成（或已尝试），开始播放
                // 若有渲染失败，snapshot 中对应 clip 会有 needs_synthesis=true、rendered_pcm=None，
                // 音频回调会陷入 has_pending_clip=true 的永久静音等待。
                // 解决方案：渲染失败时降级为播放原始音频（等同于无 pitch edit 路径）。
                if any_error {
                    if cache_log {
                        eprintln!(
                            "[play_original][cache] ERROR total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} total_ms={:.2}",
                            clips_to_render.len(),
                            cache_hit_count,
                            cache_miss_count,
                            render_success_count,
                            render_failed_count,
                            cache_probe_elapsed.as_secs_f64() * 1000.0,
                            render_elapsed.as_secs_f64() * 1000.0,
                            tension_elapsed.as_secs_f64() * 1000.0,
                            play_started_at.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    eprintln!("[play_original] rendering had errors, falling back to original audio playback");
                    // 推送失败通知
                    if rendering_state_active {
                        let _ = app.emit(
                            "playback_rendering_state",
                            PlaybackRenderingStateEvent {
                                active: false,
                                progress: Some(1.0),
                                target: Some("original".to_string()),
                            },
                        );
                    }
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

                if timeline_version_from_app(&app) != render_timeline_version {
                    if cache_log {
                        eprintln!(
                            "[play_original][cache] ABORTED_BY_TIMELINE_CHANGE total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} total_ms={:.2}",
                            clips_to_render.len(),
                            cache_hit_count,
                            cache_miss_count,
                            render_success_count,
                            render_failed_count,
                            cache_probe_elapsed.as_secs_f64() * 1000.0,
                            render_elapsed.as_secs_f64() * 1000.0,
                            tension_elapsed.as_secs_f64() * 1000.0,
                            play_started_at.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    for clip_id in pending_clip_ids_written {
                        crate::synth_clip_cache::remove_pending_rendered_key(&clip_id);
                    }
                    if rendering_state_active {
                        let _ = app.emit(
                            "playback_rendering_state",
                            PlaybackRenderingStateEvent {
                                active: false,
                                progress: Some(1.0),
                                target: Some("original".to_string()),
                            },
                        );
                    }
                    return;
                }

                let update_started_at = std::time::Instant::now();
                crate::nsf_hifigan_onnx::set_chunk_progress_callback(None);
                engine.seek_sec(render_start_sec);
                engine.update_timeline(tl_for_render);
                engine.set_playing(true, Some("original"));
                let update_elapsed = update_started_at.elapsed();

                eprintln!(
                    "[play_original][timing] total={} hit={} miss={} collect_ms={:.2} ready_filter_ms={:.2} sig_check_ms={:.2} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} update_timeline_ms={:.2} total_ms={:.2}",
                    clips_to_render.len(),
                    cache_hit_count,
                    cache_miss_count,
                    collect_elapsed.as_secs_f64() * 1000.0,
                    ready_filter_elapsed.as_secs_f64() * 1000.0,
                    timeline_sig_check_elapsed.as_secs_f64() * 1000.0,
                    cache_probe_elapsed.as_secs_f64() * 1000.0,
                    render_elapsed.as_secs_f64() * 1000.0,
                    tension_elapsed.as_secs_f64() * 1000.0,
                    update_elapsed.as_secs_f64() * 1000.0,
                    play_started_at.elapsed().as_secs_f64() * 1000.0,
                );

                if cache_log {
                    eprintln!(
                        "[play_original][cache] SUMMARY total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} update_timeline_ms={:.2} total_ms={:.2}",
                        clips_to_render.len(),
                        cache_hit_count,
                        cache_miss_count,
                        render_success_count,
                        render_failed_count,
                        cache_probe_elapsed.as_secs_f64() * 1000.0,
                        render_elapsed.as_secs_f64() * 1000.0,
                        tension_elapsed.as_secs_f64() * 1000.0,
                        update_elapsed.as_secs_f64() * 1000.0,
                        play_started_at.elapsed().as_secs_f64() * 1000.0
                    );
                }

                // 推送渲染完成
                if rendering_state_active {
                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: false,
                            progress: Some(1.0),
                            target: Some("original".to_string()),
                        },
                    );
                }
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
) -> Result<
    (
        Option<crate::synth_clip_cache::TensionRenderedClipCacheKey>,
        bool,
    ),
    String,
> {
    let Some(root) = timeline.resolve_root_track_id(&clip.track_id) else {
        return Ok((None, false));
    };
    let Some(entry) = timeline.params_by_root_track.get(&root) else {
        return Ok((None, false));
    };
    let Some(track) = timeline.tracks.iter().find(|track| track.id == root) else {
        return Ok((None, false));
    };

    let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
    if !matches!(kind, crate::state::SynthPipelineKind::NsfHifiganOnnx) {
        return Ok((None, false));
    }
    let clip_start_sec = clip.start_sec.max(0.0);
    if !crate::pitch_editing::hifigan_tension_active_for_clip(entry, clip, clip_start_sec) {
        return Ok((None, false));
    }

    let start_frame = (clip_start_sec * out_rate as f64).round() as u64;
    let end_frame = start_frame
        + (clip.length_sec.max(0.0) * out_rate as f64)
            .round()
            .max(1.0) as u64;
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
            return Ok((Some(cache_key), false));
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
        rendered_take_id: clip.active_take_id.clone(),
    };
    let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache.insert(cache_key.clone(), entry);
    Ok((Some(cache_key), true))
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
            engine_sr,
            sr,
            timeline.clips.len()
        );
    }
    // 预构建轨道的 O(1) 查找表，消除内部的 O(N) 线性扫描
    let tracks_by_id: std::collections::HashMap<&str, &crate::state::Track> =
        timeline.tracks.iter().map(|t| (t.id.as_str(), t)).collect();

    for clip in &timeline.clips {
        if clip.muted {
            continue;
        }
        let Some(source_path) = clip.source_path.as_deref() else {
            continue;
        };

        // 使用新的检测逻辑：检查clip是否需要pitch edit
        let clip_start_sec = clip.start_sec.max(0.0);
        let needs_pitch_edit =
            crate::pitch_editing::does_clip_need_processor_render(timeline, clip, clip_start_sec);

        if !needs_pitch_edit {
            continue;
        }

        let playback_rate = {
            let r = clip.playback_rate as f64;
            if r.is_finite() && r > 0.0 {
                r
            } else {
                1.0
            }
        };
        let start_frame = (clip.start_sec.max(0.0) * sr as f64).round() as u64;
        let end_frame =
            start_frame + (clip.length_sec.max(0.0) * sr as f64).round().max(1.0) as u64;

        // 获取pitch edit参数
        let Some(clip_root) = timeline.resolve_root_track_id(&clip.track_id) else {
            continue;
        };
        let entry = match timeline.params_by_root_track.get(&clip_root) {
            Some(e) => e,
            None => continue,
        };
        let track = match tracks_by_id.get(clip_root.as_str()) {
            Some(&t) => t,
            None => continue,
        };
        let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
        let renderer_id = crate::renderer::get_renderer(kind).id();
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
            clip.formant_morph.as_ref().filter(|params| params.enabled),
            None,
            clip.source_file_mtime,
            clip.loop_enabled,
            (
                (clip.source_start_sec * 1000.0).round() as i64,
                (clip.source_end_sec * 1000.0).round() as i64,
            ),
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
fn bg_render_cancel_requested() -> bool {
    BG_RENDER_CANCEL.load(std::sync::atomic::Ordering::Relaxed)
}

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
    if bg_render_cancel_requested() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }

    // 2. 源裁剪
    let playback_rate = {
        let r = clip.playback_rate as f64;
        if r.is_finite() && r > 0.0 {
            r
        } else {
            1.0
        }
    };
    let source_end_sec = clip.source_end_sec;

    let total_sec = crate::mixdown::clip_duration_sec_from_wav(in_rate, in_channels, &pcm)
        .ok_or_else(|| "cannot determine clip duration".to_string())?;
    if !(total_sec.is_finite() && total_sec > 0.0) {
        return Err("invalid clip duration".to_string());
    }

    // ── 片段构建 ─────────────────────────────────────────────────────────────
    // 非 Loop 统一使用**消费窗口模型**（clip_playback_window_sec，与 mixdown /
    // 实时 snapshot 一致）：
    //   正放 win = [ss, ss+len·r)；倒放 win = [se−len·r, se)。
    // win ∉ [0, D) 的部分渲染静音；前导静音按消费方向取值（正放看窗口起点、
    // 倒放看窗口终点越过媒体末端），绝不能把倒放的负窗口下沿误当前导静音。
    // Loop（循环源）：从完整媒体按整文件模运算回绕生成片段
    //   正放 idx(f) = floor_mod(source_start + f, D_frames)
    //   倒放 idx(f) = floor_mod(source_end − 1 − f, D_frames)
    // 负的 source_start 是环绕锚点而非 leading silence。
    let loop_mode = clip.loop_enabled;
    let (win_start_sec, win_end_sec) = crate::state::clip_playback_window_sec(clip);
    let pre_silence_sec =
        crate::state::clip_leading_silence_sec(clip, Some(total_sec)) / playback_rate.max(1e-6);
    let slice_start_sec = win_start_sec.max(0.0);
    let src_end_limit_sec = win_end_sec.min(total_sec).max(slice_start_sec);
    if !loop_mode && src_end_limit_sec - slice_start_sec <= 1e-9 {
        return Err("trimmed clip too short".to_string());
    }

    let anchor_frame: i64 = if clip.reversed {
        // 倒放末端只 clamp 到媒体时长（不能用含 `.max(source_start)` 的
        // src_end_limit_sec —— Loop 下 split 的"环绕窗口"会推错锚点）。
        (source_end_sec.min(total_sec) * in_rate as f64).round() as i64
    } else {
        // 负锚点合法：floor_mod 会环绕到文件末尾一侧。
        // 必须用**原始** source_start_sec（可为负），与实时引擎的
        // rem_euclid 回绕保持一致（clamp 到 0 会导致离线/实时内容错位）。
        (clip.source_start_sec * in_rate as f64).round() as i64
    };
    let segment: Vec<f32> = if loop_mode {
        let out_source_frames = ((clip.length_sec.max(0.0) * playback_rate * in_rate as f64)
            .ceil()
            .max(2.0)) as usize;
        crate::mixdown::build_loop_tiled_segment(
            &pcm,
            in_channels_usize,
            anchor_frame,
            clip.reversed,
            out_source_frames,
        )
    } else {
        // 3. 切片（非 Loop：消费窗口 clamp 到媒体内）
        let src_i0 = (slice_start_sec * in_rate as f64).floor().max(0.0) as usize;
        let src_i1 = ((src_end_limit_sec * in_rate as f64)
            .ceil()
            .max(src_i0 as f64) as usize)
            .min(in_frames);
        if src_i1 <= src_i0 + 1 {
            return Err("source slice too short".to_string());
        }
        pcm[(src_i0 * in_channels_usize)..(src_i1 * in_channels_usize)].to_vec()
    };

    let mut segment =
        crate::mixdown::linear_resample_interleaved(&segment, in_channels_usize, in_rate, out_rate);

    // Loop 模式的倒放方向已由回绕索引体现，不再整体反转。
    if !loop_mode && clip.reversed {
        crate::mixdown::reverse_interleaved_frames(&mut segment, in_channels_usize);
    }

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
        segment
            .chunks_exact(in_channels_usize)
            .flat_map(|chunk| [chunk[0], chunk[1]])
            .collect()
    } else {
        return Err("unsupported channel count".to_string());
    };
    let mut segment = segment;

    if bg_render_cancel_requested() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }

    if let Some(params) = clip.formant_morph.as_ref().filter(|params| params.enabled) {
        // Loop（循环源）键必须编码**实际消费的平铺区间**（与 mixdown 的键公式
        // 完全一致 —— 本函数无导出窗口，skip=0、consumed=整条 clip 消费量）。
        // 若固定取 [0, total_sec]，本函数与 mixdown 各导出窗口的内容会共享
        // 同一条目互相投毒（get_or_compute 不校验长度/内容）。
        let (key_start_sec, key_end_sec) = if loop_mode {
            let total_frames = ((total_sec * in_rate as f64).round() as i64).max(1);
            let consumed_frames = ((clip.length_sec.max(0.0) * playback_rate * in_rate as f64)
                .ceil()
                .max(2.0)) as i64;
            let start_frame = anchor_frame.rem_euclid(total_frames);
            (
                start_frame as f64 / in_rate as f64,
                (start_frame + consumed_frames) as f64 / in_rate as f64,
            )
        } else {
            // 非 Loop：键编码实际消费窗口（与 mixdown / snapshot 成对）。
            (slice_start_sec, win_end_sec)
        };
        let key = crate::formant_cache::make_formant_cache_key(
            &clip.id,
            std::path::Path::new(source_path),
            out_rate,
            key_start_sec,
            key_end_sec,
            clip.reversed && !loop_mode,
            // 离线 Loop 的处理对象是"回绕平铺 segment"，与实时域（完整文件
            // 自然顺序）不同 —— 用 tiled_wrap 域判别隔离，避免互相毒化缓存。
            loop_mode,
            params,
        );
        match crate::formant_cache::get_or_compute_formant_audio(key, &segment, out_rate, params) {
            Ok(entry) => {
                crate::formant_cache::formant_debug_log(format!(
                    "render_single_clip using formant clip_id={} frames={} diff={:.8}",
                    clip.id,
                    entry.frames,
                    crate::formant_cache::average_abs_diff(&segment, entry.pcm_stereo.as_ref())
                ));
                segment = entry.pcm_stereo.as_ref().clone();
            }
            Err(error) => {
                crate::formant_cache::formant_debug_log(format!(
                    "render_single_clip formant error clip_id={} error={}",
                    clip.id, error
                ));
            }
        }
    }

    // 5. 时间拉伸（playback_rate != 1）
    // 若合成处理器声明自己处理时间拉伸（handles_time_stretch = true），
    // 则跳过此处的时间拉伸，由处理器在 pitch edit 阶段通过 ClipProcessContext.playback_rate 内部完成。
    let processor_handles_stretch = {
        let clip_root = timeline.resolve_root_track_id(&clip.track_id);
        clip_root
            .and_then(|root| {
                let t = timeline.tracks.iter().find(|t| t.id == root)?;
                let kind = crate::state::SynthPipelineKind::from_track_algo(&t.pitch_analysis_algo);
                let has_adjustment = timeline
                    .params_by_root_track
                    .get(&root)
                    .map(|e| e.has_pitch_adjustment_active)
                    .unwrap_or(false);
                Some(crate::renderer::processor_handles_time_stretch(
                    kind,
                    t.compose_enabled || has_adjustment,
                ))
            })
            .unwrap_or(false)
    };
    if (playback_rate - 1.0).abs() > 1e-6 && !processor_handles_stretch {
        let seg_frames_in = segment.len() / 2;
        let target_frames = ((seg_frames_in as f64) / playback_rate).round().max(2.0) as usize;
        segment = crate::time_stretch::time_stretch_interleaved(
            &segment,
            2,
            out_rate,
            target_frames,
            crate::time_stretch::resolved_external_stretch_algorithm(),
        );
    }

    // Loop（循环源）：整文件回绕已在片段构建阶段完成（见上方
    // build_loop_tiled_segment）—— segment 天然覆盖整条 clip 的消费量，
    // 参数线阶段按绝对帧读取当前曲线即可，无需额外平铺。

    if bg_render_cancel_requested() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }

    let clip_start_sec = clip.start_sec.max(0.0);
    let seg_start_sec = clip_start_sec + pre_silence_sec;
    let clip_timeline_frames = (clip.length_sec.max(0.0) * out_rate as f64)
        .round()
        .max(1.0) as usize;
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

        if pre_silence_sec > 1e-6 {
            let pre_frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as usize;
            let mut with_silence = vec![0.0f32; pre_frames * 2];
            with_silence.extend_from_slice(&rendered);
            rendered = with_silence;
        }

        // Loop（循环源）：平铺已提前到参数线阶段之前完成（见上方），
        // 此处的输入已经覆盖整条 clip，只需截断/补零对齐长度。

        if rendered.len() > clip_stereo_len {
            rendered.truncate(clip_stereo_len);
        } else if rendered.len() < clip_stereo_len {
            rendered.resize(clip_stereo_len, 0.0);
        }

        rendered
    };

    if !breath_enabled {
        if bg_render_cancel_requested() {
            return Err(BG_RENDER_CANCELLED_ERR.to_string());
        }
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

    // ── 构造 BreathNoiseCache key（显式排除 formant_shift_cents）──
    let breath_noise_cache_key = {
        let clip_root = timeline.resolve_root_track_id(&clip.track_id);
        let entry = clip_root
            .as_ref()
            .and_then(|root| timeline.params_by_root_track.get(root));
        let track = clip_root
            .as_ref()
            .and_then(|root| timeline.tracks.iter().find(|t| &t.id == root));
        match (entry, track) {
            (Some(entry), Some(track)) => {
                let kind =
                    crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
                let renderer_id = crate::renderer::get_renderer(kind).id();
                let start_frame = (clip.start_sec.max(0.0) * out_rate as f64).round() as u64;
                let end_frame = start_frame
                    + (clip.length_sec.max(0.0) * out_rate as f64)
                        .round()
                        .max(1.0) as u64;
                let source_path = clip.source_path.as_deref().unwrap_or("");
                let param_hash = crate::synth_clip_cache::compute_breath_noise_hash(
                    &clip.id,
                    source_path,
                    start_frame,
                    end_frame,
                    out_rate,
                    renderer_id,
                    &entry.pitch_edit,
                    entry.frame_period_ms.max(0.1),
                    playback_rate,
                    &entry.extra_curves,
                    &entry.extra_params,
                    clip.formant_morph.as_ref().filter(|params| params.enabled),
                    clip.source_file_mtime,
                    clip.loop_enabled,
                    (
                        (clip.source_start_sec * 1000.0).round() as i64,
                        (clip.source_end_sec * 1000.0).round() as i64,
                    ),
                );
                Some(crate::synth_clip_cache::BreathNoiseCacheKey {
                    clip_id: clip.id.clone(),
                    param_hash,
                })
            }
            _ => None,
        }
    };

    // ── 尝试从 BreathNoiseCache 中命中已有的 noise stem ──────────────────
    //
    // 长度安全：缓存的 noise 是按"上次渲染时的 timeline 长度（含 playback_rate）"
    // 生成的。理论上 BreathNoiseCacheKey 已通过 `start_frame/end_frame/playback_rate`
    // 区分不同长度，但在拉伸时序竞态、参数 round-trip 等极端场景下，仍可能拿到
    // 与当前 harmonic_only 长度不一致的旧 noise。若直接复用并按 `min(...)`
    // 截短 harmonic_only, 会把当前帧的尾部 PCM 截掉, 导致开启气声后拉伸时
    // clip 后半段静音 (Bug 修复, 2026-06-30)。
    //
    // 因此命中时必须验证长度严格一致；不一致则视为未命中, 走完整的双 render
    // miss 路径重新生成 noise。
    let cached_noise = breath_noise_cache_key.as_ref().and_then(|key| {
        let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.get(key).map(|entry| entry.noise_stereo.clone())
    });

    if let Some(cached_noise_arc) = cached_noise {
        // BreathNoiseCache 命中：仅需渲染 harmonic_only（1 次 HNSEP + 1 次 HiFiGAN），
        // noise stem 直接复用缓存。
        if debug {
            eprintln!(
                "render_single_clip: breath_noise_cache HIT for clip_id={}, skipping second render_variant",
                clip.id
            );
        }

        let mut harmonic_only_clip = clip.clone();
        let mut harmonic_curves = merged_extra_curves.clone();
        harmonic_curves.insert("breath_gain".to_string(), vec![0.0; curve_len]);
        harmonic_only_clip.extra_params = Some(merged_extra_params.clone());
        harmonic_only_clip.extra_curves = Some(harmonic_curves);
        if bg_render_cancel_requested() {
            return Err(BG_RENDER_CANCELLED_ERR.to_string());
        }
        let harmonic_only = render_variant(&harmonic_only_clip);

        if harmonic_only.len() == cached_noise_arc.len() {
            // 长度严格一致：放心复用缓存
            let breath_noise_stereo = cached_noise_arc.as_slice().to_vec();
            return Ok(RenderedClipOutput {
                rendered_stereo: harmonic_only,
                breath_noise_stereo: Some(breath_noise_stereo),
            });
        }

        // 长度不一致：丢弃缓存, 走完整的双 render miss 路径重新生成 noise。
        if debug {
            eprintln!(
                "render_single_clip: breath_noise_cache STALE for clip_id={} (harmonic_len={} cached_noise_len={}), \
                 invalidating and falling back to full 2-pass render",
                clip.id,
                harmonic_only.len(),
                cached_noise_arc.len()
            );
        }
        if breath_noise_cache_key.is_some() {
            let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.invalidate(&clip.id);
        }
        // fall through to miss path 下方
    }

    // ── BreathNoiseCache 未命中（或长度不匹配已失效）：单次 HNSEP + 单次 HiFiGAN ──
    //
    // 优化（2026-07-18）：消除双重 render_variant。
    // 旧代码：两次完整 render_variant（harmonic_only + unity_breath）→ 2x HNSEP + 2x HiFiGAN。
    // 新代码：一次 HNSEP 分离 → 一次 HiFiGAN → noise = HNSEP_noise（无需第二次 HiFiGAN）。
    //
    // 原理：unity_mix = hifigan(harmonic) + noise×1.0, harmonic_only = hifigan(harmonic) + noise×0.0
    //       → unity_mix = harmonic_only + noise_stereo
    //       → breath_noise_stereo = noise_stereo
    // 直接使用 HNSEP 的 noise 输出作为 breath_noise，省去第二次完整的 ProcessorChain + HiFiGAN。
    if debug {
        eprintln!(
            "render_single_clip: breath_noise_cache MISS for clip_id={}, optimized 1-pass render",
            clip.id
        );
    }

    // Step 1: Extract mono from the (already time-stretched) stereo segment
    let mono: Vec<f32> = segment
        .chunks_exact(2)
        .map(|ch| (ch[0] + ch[1]) * 0.5f32)
        .collect();

    // Step 2: Pre-populate HNSEP cache by doing separation once.
    // This ensures the subsequent render_variant(harmonic_only) hits the cache
    // and only runs HiFiGAN, skipping HNSEP.
    if bg_render_cancel_requested() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }
    let noise_mono = crate::hnsep_onnx::infer_noise_mono(&clip.id, &mono, out_rate)?;

    // Step 3: Render harmonic_only through ProcessorChain (HNSEP cache hits → HiFiGAN only)
    let mut harmonic_only_clip = clip.clone();
    let mut harmonic_curves = merged_extra_curves.clone();
    harmonic_curves.insert("breath_gain".to_string(), vec![0.0; curve_len]);
    harmonic_only_clip.extra_params = Some(merged_extra_params.clone());
    harmonic_only_clip.extra_curves = Some(harmonic_curves);
    if bg_render_cancel_requested() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }
    let harmonic_only = render_variant(&harmonic_only_clip);

    // Step 4: Convert noise mono to stereo, matching harmonic_only length
    let out_len = harmonic_only.len();
    let out_frames = out_len / 2;
    let noise_stereo: Vec<f32> = {
        let noise_mono_raw = noise_mono.as_ref();
        // 时间拉伸若由处理器内部完成（mel 域），谐波输出是**时间轴**长度，
        // 而 HNSEP 的噪声 stem 仍是**源速率**长度。必须重采样对齐后再转立体声，
        // 否则拉伸出来的尾巴会整段缺失（此前是直接补静音）。
        let aligned = crate::renderer::chain::resample_noise_to_len(noise_mono_raw, out_frames);
        let mut stereo = Vec::with_capacity(out_len);
        // Duplicate each mono sample to L/R channels
        for &s in &aligned {
            stereo.push(s);
            stereo.push(s);
        }
        // 长度兜底（重采样在极小输入下可能少一帧）
        if stereo.len() < out_len {
            stereo.resize(out_len, 0.0f32);
        } else if stereo.len() > out_len {
            stereo.truncate(out_len);
        }
        stereo
    };
    let breath_noise_stereo = noise_stereo;

    // 将 noise stem 存入 BreathNoiseCache，后续 formant 编辑时可直接复用
    if let Some(key) = breath_noise_cache_key {
        let entry = crate::synth_clip_cache::BreathNoiseCacheEntry {
            noise_stereo: std::sync::Arc::new(breath_noise_stereo.clone()),
            frames: (breath_noise_stereo.len() / 2) as u64,
            sample_rate: out_rate,
        };
        let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        cache.insert(key, entry);
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

// ─── 后台预渲染（Background Pre-render）─────────────────────────────────────────

/// 后台预渲染：编辑操作使缓存失效后立即在后台启动渲染，
/// 而无需等待用户按下播放键。
///
/// 前端通过 `start_background_render` Tauri 命令调用此函数。
/// 渲染线程完成后自动重置 `BG_RENDER_ACTIVE`。
pub(crate) fn start_background_render(app: tauri::AppHandle) -> serde_json::Value {
    use std::sync::atomic::Ordering;

    // 防止重复启动
    if BG_RENDER_ACTIVE.swap(true, Ordering::AcqRel) {
        eprintln!("[bg_render] already active, skipping");
        return serde_json::json!({"ok": true, "skipped": true, "reason": "already_active"});
    }
    // 清除可能残留的上一轮取消标志
    BG_RENDER_CANCEL.store(false, Ordering::Release);
    // 每次启动递增代数：旧渲染线程的清理逻辑不能再影响这一次新渲染。
    BG_RENDER_GENERATION.fetch_add(1, Ordering::AcqRel);
    let render_generation = BG_RENDER_GENERATION.load(Ordering::Acquire);

    // ★ panic 隔离：收集/启动阶段（native 推理库、缓存锁、AppState 访问等）
    // 可能 panic。本函数现在也会被 `request_background_render` 放到独立线程
    // 上调用，panic 不再有命令线程兜底 —— 若不在此处复位 BG_RENDER_ACTIVE，
    // 后续所有渲染请求都会因 “already active” 被跳过，后台预渲染永久失效，
    // 且前端进度事件永远不结束。
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        start_background_render_inner(app, render_generation)
    }));
    match result {
        Ok(value) => value,
        Err(payload) => {
            eprintln!(
                "[bg_render] panic during render setup (payload={:?}); resetting active flag",
                payload
            );
            BG_RENDER_ACTIVE.store(false, Ordering::Release);
            BG_RENDER_CANCEL.store(false, Ordering::Release);
            BG_RENDER_PITCH_PENDING.store(false, Ordering::Release);
            serde_json::json!({"ok": false, "error": "bg_render_setup_panicked"})
        }
    }
}

fn start_background_render_inner(app: tauri::AppHandle, render_generation: u64) -> serde_json::Value {
    use std::sync::atomic::Ordering;

    // Clone app before getting state (state borrows from the clone),
    // so the original app can be moved into the thread.
    let app_clone = app.clone();
    let state = app_clone.state::<AppState>();
    let timeline = match state.timeline.lock() {
        Ok(g) => g.clone(),
        Err(p) => p.into_inner().clone(),
    };

    let engine_sr = state.audio_engine.sample_rate_hz();
    let sr = if engine_sr > 0 { engine_sr } else { 44100 };

    let mut clips_to_render = collect_clips_needing_render(&timeline, sr);
    let unfiltered_total = clips_to_render.len();
    clips_to_render.retain(|info| is_clip_pitch_analysis_ready(&timeline, &info.clip));
    let skipped_not_ready = unfiltered_total.saturating_sub(clips_to_render.len());
    BG_RENDER_PITCH_PENDING.store(skipped_not_ready > 0, Ordering::Release);
    clips_to_render.sort_by(|a, b| a.clip.start_sec.total_cmp(&b.clip.start_sec));

    // Save len before clips_to_render is moved into the thread closure
    let total = clips_to_render.len();

    if total == 0 {
        eprintln!(
            "[bg_render] no clips need rendering (ready={} skipped_not_ready={})",
            clips_to_render.len(),
            skipped_not_ready
        );
        BG_RENDER_ACTIVE.store(false, Ordering::Release);
        BG_RENDER_CANCEL.store(false, Ordering::Release);
        // 不发送渲染事件，避免前端状态栏闪烁。
        // 当前端有实质性编辑时，自然会触发下一次渲染。
        return serde_json::json!({"ok": true, "rendered": 0});
    }

    let render_timeline_version = state.timeline_version.load(Ordering::Acquire);

    eprintln!(
        "[bg_render] starting background render: {} clips, engine_sr={}, timeline_version={}",
        total, sr, render_timeline_version
    );

    // 清空上次的 pending_rendered_keys
    crate::synth_clip_cache::clear_pending_rendered_keys();

    // 动态扩容缓存
    {
        let mut rendered_cache = crate::synth_clip_cache::global_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let required = rendered_cache.len().saturating_add(total);
        rendered_cache.ensure_capacity(required);
    }
    {
        let mut tension_cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        tension_cache.ensure_capacity(total.max(1));
    }
    {
        let breath_clips = total;
        let required = (breath_clips + breath_clips / 4).max(128);
        crate::hnsep_onnx::ensure_cache_capacity(required);
    }
    {
        let mut breath_noise_cache = crate::synth_clip_cache::global_breath_noise_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        breath_noise_cache.ensure_capacity(total.max(1));
    }

    crate::nsf_hifigan_onnx::reset_chunk_progress(total);

    let app_for_progress = app.clone();
    let progress_generation = render_generation;
    crate::nsf_hifigan_onnx::set_chunk_progress_callback(Some(Box::new(move |progress: f64| {
        // 旧代渲染线程的进度事件不得再刷新前端状态，否则新工程渲染结束后
        // 会被迟到的旧事件重新点亮“渲染中 XX%”。
        if BG_RENDER_GENERATION.load(Ordering::Acquire) != progress_generation {
            return;
        }
        let _ = app_for_progress.emit(
            "playback_rendering_state",
            PlaybackRenderingStateEvent {
                active: true,
                progress: Some(progress),
                target: Some("background".to_string()),
            },
        );
    })));

    // Explicitly drop app_clone's state borrow before moving app into the thread
    drop(state);
    drop(app_clone);

    // 后台渲染线程
    std::thread::spawn(move || {
        // ★ panic 隔离：渲染循环中的 panic（native 推理、缓存锁等）不能
        // 让 BG_RENDER_ACTIVE 卡死 —— 否则后续渲染请求全部被跳过、
        // 前端进度事件永远不结束（直到应用重启）。
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            render_background_pass(
                app.clone(),
                &clips_to_render,
                total,
                render_timeline_version,
                &timeline,
                skipped_not_ready,
                render_generation,
            );
        }));
        if outcome.is_err() {
            eprintln!("[bg_render] render thread panicked; resetting flags");
            // 只有当前代数仍属于本线程时才清理，避免把新工程的新一轮状态清掉。
            if BG_RENDER_GENERATION.load(Ordering::Acquire) == render_generation {
                BG_RENDER_ACTIVE.store(false, Ordering::Release);
                BG_RENDER_CANCEL.store(false, Ordering::Release);
                BG_RENDER_PITCH_PENDING.store(false, Ordering::Release);
            }
            let _ = app.emit(
                "playback_rendering_state",
                PlaybackRenderingStateEvent {
                    active: false,
                    progress: Some(1.0),
                    target: Some("background".to_string()),
                },
            );
        }
    });

    serde_json::json!({"ok": true, "rendering": total})
}

/// 后台渲染单轮主循环（在渲染线程上执行；由调用方负责 panic 隔离）。
#[allow(clippy::too_many_arguments)]
fn render_background_pass(
    app: tauri::AppHandle,
    clips_to_render: &[ClipRenderInfo],
    total: usize,
    render_timeline_version: u64,
    timeline: &crate::state::TimelineState,
    skipped_not_ready: usize,
    render_generation: u64,
) {
    use std::sync::atomic::Ordering;

    {
        let cache_log = std::env::var("HIFISHIFTER_RENDER_CACHE_LOG")
            .ok()
            .as_deref()
            == Some("1");
        let started_at = std::time::Instant::now();

        let mut rendered_count = 0u32;
        let mut cache_hit_count = 0u32;
        let mut cache_miss_count = 0u32;
        let mut render_success_count = 0u32;
        let mut render_failed_count = 0u32;
        let mut cache_probe_elapsed = std::time::Duration::ZERO;
        let mut render_elapsed = std::time::Duration::ZERO;
        let mut tension_elapsed = std::time::Duration::ZERO;
        let mut cancelled = false;
        let mut pending_clip_ids_written: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        // 只有真正开始合成时才向前端发出进度事件。若本轮全部命中缓存，
        // 则完全不需要展示渲染进度，避免打开工程后的首次播放闪一下进度条。
        let mut rendering_started = false;

        for clip_render_info in clips_to_render {
            // 检查取消标志（用户在渲染中重新编辑参数时会设置）
            if BG_RENDER_CANCEL.load(Ordering::Relaxed) {
                eprintln!(
                    "[bg_render] cancel flag detected at clip {}/{}",
                    rendered_count, total
                );
                cancelled = true;
                break;
            }
            // 每隔 32 个 clip 检查时间线版本是否已变更
            if rendered_count % 32 == 0 {
                let state = app.state::<AppState>();
                let changed =
                    state.timeline_version.load(Ordering::Acquire) != render_timeline_version;
                if changed {
                    cancelled = true;
                    break;
                }
            }

            let cache_probe_started_at = std::time::Instant::now();
            let mut base_entry = {
                let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                cache.get(&clip_render_info.cache_key).cloned()
            };
            cache_probe_elapsed += cache_probe_started_at.elapsed();

            if base_entry.is_some() {
                cache_hit_count += 1;
                if cache_log {
                    eprintln!(
                        "[bg_render][cache] HIT clip_id={} hash={:#018x}",
                        clip_render_info.clip.id, clip_render_info.cache_key.param_hash
                    );
                }
                crate::synth_clip_cache::register_pending_rendered_key(
                    &clip_render_info.clip.id,
                    clip_render_info.cache_key.clone(),
                );
                pending_clip_ids_written.insert(clip_render_info.clip.id.clone());
            }

            if base_entry.is_none() {
                cache_miss_count += 1;
                if cache_log {
                    eprintln!(
                        "[bg_render][cache] MISS clip_id={} hash={:#018x}",
                        clip_render_info.clip.id, clip_render_info.cache_key.param_hash
                    );
                }
                if !rendering_started {
                    rendering_started = true;
                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: true,
                            progress: Some(0.0),
                            target: Some("background".to_string()),
                        },
                    );
                }
                if let Ok(mut state_mgr) =
                    crate::clip_rendering_state::global_clip_rendering_state().lock()
                {
                    state_mgr.set_state(
                        &clip_render_info.clip.id,
                        crate::clip_rendering_state::ClipRenderingState::Rendering,
                        0.0,
                        None,
                    );
                }

                let render_started_at = std::time::Instant::now();
                match render_single_clip(&timeline, &clip_render_info.clip, clip_render_info.sr) {
                    Ok(rendered) => {
                        render_elapsed += render_started_at.elapsed();
                        let stereo_pcm = rendered.rendered_stereo;
                        let frames = (stereo_pcm.len() / 2) as u64;
                        let entry = crate::synth_clip_cache::RenderedClipCacheEntry {
                            pcm_stereo: std::sync::Arc::new(stereo_pcm),
                            breath_noise_stereo: rendered
                                .breath_noise_stereo
                                .map(std::sync::Arc::new),
                            frames,
                            sample_rate: clip_render_info.sr,
                            rendered_take_id: clip_render_info.clip.active_take_id.clone(),
                        };

                        let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        cache.insert(clip_render_info.cache_key.clone(), entry.clone());
                        crate::synth_clip_cache::register_pending_rendered_key(
                            &clip_render_info.clip.id,
                            clip_render_info.cache_key.clone(),
                        );
                        pending_clip_ids_written.insert(clip_render_info.clip.id.clone());

                        base_entry = Some(entry);
                        render_success_count += 1;
                    }
                    Err(e) => {
                        if e == BG_RENDER_CANCELLED_ERR {
                            cancelled = true;
                            break;
                        }
                        render_elapsed += render_started_at.elapsed();
                        eprintln!(
                            "[bg_render] clip render failed: clip_id={} err={}",
                            clip_render_info.clip.id, e
                        );
                        render_failed_count += 1;
                        if let Ok(mut state_mgr) =
                            crate::clip_rendering_state::global_clip_rendering_state().lock()
                        {
                            state_mgr.set_state(
                                &clip_render_info.clip.id,
                                crate::clip_rendering_state::ClipRenderingState::Failed,
                                0.0,
                                Some(e.clone()),
                            );
                        }
                    }
                }
            }

            if let Some(base_entry) = base_entry.as_ref() {
                let tension_started_at = std::time::Instant::now();
                match ensure_hifigan_tension_cache(
                    &timeline,
                    &clip_render_info.clip,
                    clip_render_info.sr,
                    clip_render_info.cache_key.param_hash,
                    base_entry.pcm_stereo.as_slice(),
                ) {
                    Ok((_, tension_generated)) => {
                        tension_elapsed += tension_started_at.elapsed();
                        if tension_generated && !rendering_started {
                            rendering_started = true;
                            let _ = app.emit(
                                "playback_rendering_state",
                                PlaybackRenderingStateEvent {
                                    active: true,
                                    progress: Some(0.0),
                                    target: Some("background".to_string()),
                                },
                            );
                        }
                        if let Ok(mut state_mgr) =
                            crate::clip_rendering_state::global_clip_rendering_state().lock()
                        {
                            state_mgr.set_state(
                                &clip_render_info.clip.id,
                                crate::clip_rendering_state::ClipRenderingState::Ready,
                                1.0,
                                None,
                            );
                        }
                    }
                    Err(e) => {
                        tension_elapsed += tension_started_at.elapsed();
                        eprintln!(
                            "[bg_render] tension render failed: clip_id={} err={}",
                            clip_render_info.clip.id, e
                        );
                        if let Ok(mut state_mgr) =
                            crate::clip_rendering_state::global_clip_rendering_state().lock()
                        {
                            state_mgr.set_state(
                                &clip_render_info.clip.id,
                                crate::clip_rendering_state::ClipRenderingState::Failed,
                                0.0,
                                Some(e.clone()),
                            );
                        }
                    }
                }
            }

            rendered_count += 1;
        }

        if cancelled {
            // 如果取消后已经启动了新一轮渲染（代数已变），旧线程不得再清理
            // 全局状态或触发旧工程的重启，直接退出即可。
            if BG_RENDER_GENERATION.load(Ordering::Acquire) != render_generation {
                return;
            }
            crate::nsf_hifigan_onnx::set_chunk_progress_callback(None);
            for clip_id in pending_clip_ids_written {
                crate::synth_clip_cache::remove_pending_rendered_key(&clip_id);
            }
            BG_RENDER_ACTIVE.store(false, Ordering::Release);
            BG_RENDER_CANCEL.store(false, Ordering::Release);
            // 本轮被取消（新编辑/时间线版本变更）：清除“音高分析未完成”挂起标记，
            // 避免音高分析稍后完成时触发一次多余的渲染（下一轮新渲染会重新设置它）。
            BG_RENDER_PITCH_PENDING.store(false, Ordering::Release);

            // 检查是否需要立即重启（用户在渲染中重新编辑参数时会设置）
            if BG_RENDER_RESTART_NEEDED.swap(false, Ordering::AcqRel) {
                eprintln!(
                    "[bg_render] cancelled by new edit (rendered {}/{}), restarting with fresh params...",
                    rendered_count, total
                );
                // 直接启动新一轮渲染，不发送中间完成事件，对用户无感
                start_background_render(app.clone());
                return;
            }

            // 真正取消（时间线版本变更等）：发出完成事件
            if cache_log {
                eprintln!(
                    "[bg_render][cache] CANCELLED total={} hit={} miss={} rendered_ok={} rendered_fail={}",
                    total, cache_hit_count, cache_miss_count,
                    render_success_count, render_failed_count
                );
            }
            let _ = app.emit(
                "playback_rendering_state",
                PlaybackRenderingStateEvent {
                    active: false,
                    progress: Some(1.0),
                    target: Some("background".to_string()),
                },
            );
            return;
        }

        // 渲染完成：缓存已填入，下次 play_original 调用 update_timeline
        // 时会自动通过 build_snapshot 读取缓存中的 clip。
        // 不在此处调用 engine.update_timeline，以避免触发 handle_update_timeline
        // 中的 auto-trigger 形成反馈循环。

        if cache_log {
            eprintln!(
                "[bg_render][cache] DONE total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} total_ms={:.2}",
                total,
                cache_hit_count,
                cache_miss_count,
                render_success_count,
                render_failed_count,
                cache_probe_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
                tension_elapsed.as_secs_f64() * 1000.0,
                started_at.elapsed().as_secs_f64() * 1000.0
            );
        }
        eprintln!(
            "[bg_render] complete: {} clips, {} hit, {} miss, {} ok, {} fail in {:.2}s",
            total,
            cache_hit_count,
            cache_miss_count,
            render_success_count,
            render_failed_count,
            started_at.elapsed().as_secs_f64()
        );

        // 旧代数线程完成时不得清理新一轮渲染的全局状态。
        if BG_RENDER_GENERATION.load(Ordering::Acquire) != render_generation {
            return;
        }

        crate::nsf_hifigan_onnx::set_chunk_progress_callback(None);
        BG_RENDER_ACTIVE.store(false, Ordering::Release);
        BG_RENDER_CANCEL.store(false, Ordering::Release);

        // 第一轮可能因为音高分析尚未完成而跳过了部分 clip。
        // 音高分析完成后没有新的“缓存失效”事件，因此这里主动补一轮渲染，
        // 保证用户等待后台渲染进度结束后，所有需要渲染的 clip 都真正进入缓存。
        // ★ 补轮必须受“本轮是否有实际进展”约束（见 should_follow_up_render），
        // 否则会形成 100% CPU 的无限后台渲染循环，把整个应用拖到未响应。
        if should_follow_up_render(skipped_not_ready, render_success_count)
            && AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed)
        {
            eprintln!(
                "[bg_render] follow-up pass needed: {} clip(s) were not pitch-ready, {} clip(s) newly rendered in this pass",
                skipped_not_ready, render_success_count
            );
            start_background_render(app.clone());
            return;
        }
        if skipped_not_ready > 0 && render_success_count == 0 {
            eprintln!(
                "[bg_render] {} clip(s) still not pitch-ready but this pass made no new progress; waiting for pitch analysis completion instead of re-rendering",
                skipped_not_ready
            );
        }

        // 若完成时恰好有新编辑触发的重启请求，立即启动新一轮渲染
        if BG_RENDER_RESTART_NEEDED.swap(false, Ordering::AcqRel) {
            eprintln!("[bg_render] completed but restart was requested during finalization, starting new render");
            start_background_render(app.clone());
            return;
        }

        let _ = app.emit(
            "playback_rendering_state",
            PlaybackRenderingStateEvent {
                active: false,
                progress: Some(1.0),
                target: Some("background".to_string()),
            },
        );
    }
}

/// Request a background pre-render after render caches have been invalidated.
///
/// Unlike `start_background_render`, this is safe to call even while a render is
/// already running: it cancels the in-flight render and requests a restart with
/// the fresh cache state. It is a no-op when background pre-render is disabled.
///
/// ★ 线程安全约定（防死锁）：
/// `start_background_render` 内部会锁定 `state.timeline`，而本函数常被
/// “已经持有时间线锁”的调用方使用（如 `set_timeline_tempo_map` 的音阶
/// 变化分支）。std Mutex 不可重入，若在调用方线程上同步启动渲染，
/// 调用方会自我死锁 —— 命令线程永久阻塞，整个应用进入“未响应”。
/// 因此这里把真正的启动工作转移到新线程：本函数只做原子的状态检查与
/// 标记（无锁），渲染启动线程会等待时间线锁自然释放后再开始收集。
pub(crate) fn request_background_render(app: &tauri::AppHandle) -> serde_json::Value {
    use std::sync::atomic::Ordering;

    if !AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed) {
        return serde_json::json!({"ok": true, "skipped": true, "reason": "disabled"});
    }

    if BG_RENDER_ACTIVE.load(Ordering::Relaxed) {
        BG_RENDER_CANCEL.store(true, Ordering::Release);
        BG_RENDER_RESTART_NEEDED.store(true, Ordering::Release);
        eprintln!("[bg_render] caches invalidated while render active; requesting restart");
        return serde_json::json!({"ok": true, "restart_requested": true});
    }

    // A fresh render request supersedes any stale restart marker.
    BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);

    // 在新线程上启动渲染：既避免调用方持有时间线锁时自我死锁，
    // 也让命令线程尽快返回、界面保持响应。
    let app = app.clone();
    std::thread::spawn(move || {
        let _ = start_background_render(app);
    });

    serde_json::json!({"ok": true, "starting": true})
}

/// 取消当前正在运行的后台预渲染（如果有）。
pub(super) fn cancel_background_render(app: Option<&tauri::AppHandle>) -> serde_json::Value {
    use std::sync::atomic::Ordering;
    let was_active = BG_RENDER_ACTIVE.swap(false, Ordering::AcqRel);
    // 让正在运行的渲染循环在下一个 clip 边界立刻退出，而不是继续把旧工程
    // 渲染完。同时清除重启标记，避免取消后被错误地按旧渲染状态自动重启。
    BG_RENDER_CANCEL.store(true, Ordering::Release);
    BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);
    BG_RENDER_PITCH_PENDING.store(false, Ordering::Release);
    // 递增代数，使旧渲染线程的收尾清理不再影响新的一轮渲染。
    BG_RENDER_GENERATION.fetch_add(1, Ordering::AcqRel);
    eprintln!("[bg_render] cancel requested, was_active={was_active}");
    // 立即通知前端“后台渲染已结束”，避免旧线程迟到的进度事件让状态卡在
    // “渲染中 100%”。如果随后有新工程的新一轮渲染，会再发 active=true。
    if let Some(app) = app {
        let _ = app.emit(
            "playback_rendering_state",
            PlaybackRenderingStateEvent {
                active: false,
                progress: Some(1.0),
                target: Some("background".to_string()),
            },
        );
    }
    serde_json::json!({"ok": true, "was_active": was_active})
}

#[cfg(test)]
mod tests {
    use super::should_follow_up_render;

    #[test]
    fn bg_render_follow_up_requires_progress() {
        // 有跳过 + 有进展 → 补一轮。
        assert!(should_follow_up_render(1, 1));
        // 有跳过但本轮没有任何新渲染（全部命中缓存或全部失败）→ 不补轮。
        // 这是防止“永远无法就绪的 clip”造成 100% CPU 无限后台渲染循环的关键。
        assert!(!should_follow_up_render(1, 0));
        // 没有跳过 → 不补轮。
        assert!(!should_follow_up_render(0, 3));
    }
}
