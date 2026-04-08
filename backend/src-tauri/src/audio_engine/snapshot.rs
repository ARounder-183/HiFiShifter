use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use crate::state::{Clip, TimelineState, Track};

use super::io::{get_resampled_stereo_cached, is_audio_path};
use super::types::{EngineClip, EngineSnapshot, ResampledStereo, StretchJob, StretchKey};
use super::util::{quantize_i64, quantize_u32};

pub(crate) fn compute_track_gains<'a>(tracks: &'a [Track]) -> HashMap<&'a str, (f32, bool, bool)> {
    let by_id: HashMap<&str, &Track> = tracks.iter().map(|t| (t.id.as_str(), t)).collect();
    let any_solo = tracks.iter().any(|t| t.solo);

    let mut out = HashMap::with_capacity(tracks.len());

    for t in tracks {
        let mut gain = 1.0f32;
        let mut muted = false;
        let mut soloed = false;

        let mut cur = Some(t.id.as_str());
        let mut safety = 0;

        while let Some(id) = cur {
            if let Some(node) = by_id.get(id) {
                gain *= node.volume.clamp(0.0, 4.0);
                muted |= node.muted;
                soloed |= node.solo;
                cur = node.parent_id.as_deref();
            } else {
                break;
            }

            safety += 1;
            if safety > 2048 {
                break;
            }
        }

        // Solo overrides mute: when a track (or its ancestor) is soloed,
        // its own mute flag is ignored so that solo always wins.
        let effective_muted = if any_solo && soloed { false } else { muted };

        out.insert(
            t.id.as_str(),
            (gain, effective_muted, if any_solo { soloed } else { true }),
        );
    }

    out
}

pub(crate) fn source_bounds_frames(
    source_start_sec: f64,
    source_end_sec: f64,
    src_total_frames: usize,
    sr: u32,
) -> (u64, u64) {
    let source_start_sec = source_start_sec.max(0.0);

    let total_sec = (src_total_frames as f64) / sr.max(1) as f64;
    let start = (source_start_sec * sr as f64).round().max(0.0);
    let end_limit_sec = source_end_sec.min(total_sec).max(source_start_sec);
    let end = (end_limit_sec * sr as f64).round().max(start);

    // Keep within source length.
    let max_start = src_total_frames.saturating_sub(1) as u64;
    let mut start_u = (start as u64).min(max_start);
    let mut end_u = (end as u64).min(src_total_frames as u64);
    if end_u <= start_u {
        end_u = (start_u + 1).min(src_total_frames as u64);
    }
    // Ensure exclusive end.
    if end_u > src_total_frames as u64 {
        end_u = src_total_frames as u64;
    }
    if start_u >= end_u {
        start_u = end_u.saturating_sub(1);
    }
    (start_u, end_u)
}

fn clip_source_bounds_frames(clip: &Clip, src_total_frames: usize, sr: u32) -> (u64, u64) {
    source_bounds_frames(
        clip.source_start_sec.max(0.0),
        clip.source_end_sec,
        src_total_frames,
        sr,
    )
}

pub(crate) fn make_stretch_key(
    path: &Path,
    out_rate: u32,
    source_start: f64,
    source_end: f64,
    playback_rate: f64,
) -> StretchKey {
    StretchKey {
        path: path.to_path_buf(),
        out_rate,
        bpm_q: 0, // 不再依赖 BPM
        trim_start_q: quantize_i64(source_start, 1000.0),
        trim_end_q: quantize_i64(source_end, 1000.0),
        playback_rate_q: quantize_u32(playback_rate, 10000.0),
    }
}

pub(crate) fn schedule_stretch_jobs(
    timeline: &TimelineState,
    out_rate: u32,
    stretch_tx: &mpsc::Sender<StretchJob>,
    inflight: &Mutex<HashSet<StretchKey>>,
    stretch_cache: &Arc<Mutex<HashMap<StretchKey, ResampledStereo>>>,
    app_handle: Option<&tauri::AppHandle>,
) {
    // 计算 track_gain，删除了无用的 bpm 和冗余的 audible_tracks
    let track_gain = compute_track_gains(&timeline.tracks);

    for clip in &timeline.clips {
        if clip.muted {
            continue;
        }

        // 直接查字典，取代之前额外的 HashSet 分配
        let (_, track_muted, track_solo_ok) = track_gain
            .get(clip.track_id.as_str())
            .cloned()
            .unwrap_or((1.0, false, true));

        // 轨道静音或没被 solo 时直接跳过
        if track_muted || !track_solo_ok {
            continue;
        }

        let Some(source_path) = clip.source_path.as_ref() else {
            continue;
        };
        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };
        if (playback_rate - 1.0).abs() <= 1e-6 {
            continue;
        }
        let path = Path::new(source_path);
        if !is_audio_path(path) {
            continue;
        }

        let key = make_stretch_key(
            path,
            out_rate,
            clip.source_start_sec.max(0.0),
            clip.source_end_sec,
            playback_rate,
        );
        if let Ok(m) = stretch_cache.lock() {
            if m.contains_key(&key) {
                continue;
            }
        }

        // 利用 HashSet 本身的机制，取代之前的 9 行锁判断
        let _should_enqueue = inflight
            .lock()
            .map(|mut s| {
                if s.contains(&key) {
                    false
                } else {
                    s.insert(key.clone());
                    true
                }
            })
            .unwrap_or(false);

        // 只有确实需要 enqueue 的时候，才去消耗 CPU 分配字符串
        let clip_name = clip
            .source_path
            .as_deref()
            .and_then(|p| Path::new(p).file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();

        let _ = stretch_tx.send(StretchJob {
            key,
            source_start_sec: clip.source_start_sec.max(0.0),
            source_end_sec: clip.source_end_sec,
            playback_rate,
            clip_name,
            app_handle: app_handle.map(|h| std::sync::Arc::new(h.clone())),
        });
    }
}

pub(crate) fn build_snapshot(
    timeline: &TimelineState,
    out_rate: u32,
    cache: &Arc<Mutex<HashMap<(PathBuf, u32), ResampledStereo>>>,
    stretch_cache: &Arc<Mutex<HashMap<StretchKey, ResampledStereo>>>,
) -> EngineSnapshot {
    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
    let bpm = if timeline.bpm.is_finite() && timeline.bpm > 0.0 {
        timeline.bpm
    } else {
        120.0
    };

    let duration_frames = (timeline.project_sec.max(0.0) * out_rate as f64)
        .round()
        .max(0.0) as u64;

    let track_gain = compute_track_gains(&timeline.tracks);
    let tracks_by_id: HashMap<&str, &Track> =
        timeline.tracks.iter().map(|t| (t.id.as_str(), t)).collect();

    // 预分配内存
    let mut clips_out: Vec<EngineClip> = Vec::with_capacity(timeline.clips.len());

    for clip in &timeline.clips {
        if clip.muted {
            continue;
        }

        let (track_gain_value, track_muted, track_solo_ok) = track_gain
            .get(clip.track_id.as_str())
            .cloned()
            .unwrap_or((1.0, false, true));

        if track_muted || !track_solo_ok {
            continue;
        }

        let Some(source_path) = clip.source_path.as_ref() else {
            continue;
        };
        let path = Path::new(source_path);
        if !is_audio_path(path) {
            continue;
        }

        // 直接用刚才提出来的 track_gain_value
        let gain = (clip.gain.max(0.0) * track_gain_value).clamp(0.0, 4.0);
        if gain <= 0.0 {
            continue;
        }

        let timeline_len_sec = clip.length_sec.max(0.0);
        if !(timeline_len_sec.is_finite() && timeline_len_sec > 1e-6) {
            continue;
        }
        let length_frames = (timeline_len_sec * out_rate as f64).round().max(1.0) as u64;

        let start_sec = clip.start_sec.max(0.0);
        let start_frame = (start_sec * out_rate as f64).round().max(0.0) as u64;

        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };

        let src = match get_resampled_stereo_cached(path, out_rate, cache) {
            Some(v) => v,
            None => {
                if debug {
                    eprintln!(
                        "[snapshot] SKIP clip_id={} reason=source_not_in_resource_cache path={}",
                        clip.id,
                        path.display()
                    );
                }
                continue;
            }
        };

        let (mut src_start, mut src_end) = clip_source_bounds_frames(clip, src.frames, out_rate);
        // Keep 1-frame slices audible; only drop truly empty source ranges.
        if src_end.saturating_sub(src_start) == 0 {
            continue;
        }

        // Timeline clips never loop/repeat; out-of-range source time is treated as silence.
        let mut repeat = false;

        // Negative trimStart means the clip starts before the source: render leading silence.
        // trim_* are expressed in SOURCE seconds (i.e. they already incorporate playbackRate in UI).
        // Therefore leading silence in timeline time scales by 1 / playback_rate.
        let local_src_offset_frames: i64 =
            if clip.source_start_sec.is_finite() && clip.source_start_sec < 0.0 {
                let pr = playback_rate.max(1e-6);
                let pre_silence_sec = (-clip.source_start_sec) / pr;
                let frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as i64;
                -frames
            } else {
                0
            };

        // If playback_rate != 1, prefer an asynchronously precomputed, pitch-preserving buffer.
        // Never block snapshot building here.
        let mut src_render = src;
        let mut playback_rate_render = playback_rate;
        if (playback_rate - 1.0).abs() > 1e-6 {
            let key = make_stretch_key(
                path,
                out_rate,
                clip.source_start_sec.max(0.0),
                clip.source_end_sec,
                playback_rate,
            );
            if let Ok(m) = stretch_cache.lock() {
                if let Some(stretched) = m.get(&key) {
                    src_render = stretched.clone();
                    src_start = 0;
                    src_end = src_render.frames as u64;
                    playback_rate_render = 1.0;
                    repeat = false;
                }
            }
        }

        let fade_in_frames = (clip.fade_in_sec.max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as u64;
        let fade_out_frames = (clip.fade_out_sec.max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as u64;

        // 提前计算 root_track_id，避免后续冗余溯源
        let root_track_id = timeline.resolve_root_track_id(&clip.track_id);
        let processor_params = root_track_id.as_ref().and_then(|root| {
            let entry = timeline.params_by_root_track.get(root)?;
            let track = tracks_by_id.get(root.as_str())?;
            let kind = crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo);
            let renderer_id = crate::renderer::get_renderer(kind).id();
            Some((
                entry.pitch_orig.as_slice(),
                entry.pitch_edit.as_slice(),
                entry.frame_period_ms.max(0.1),
                renderer_id,
                entry,
                &entry.extra_curves,
                &entry.extra_params,
            ))
        });
        let (breath_curve, breath_curve_frame_period_ms) = processor_params
            .and_then(
                |(_, _, frame_period_ms, renderer_id, _, extra_curves, extra_params)| {
                    if renderer_id == "nsf_hifigan_onnx"
                        && crate::pitch_editing::extra_param_enabled(extra_params, "breath_enabled")
                    {
                        Some((
                            extra_curves
                                .get("breath_gain")
                                .cloned()
                                .map(std::sync::Arc::new),
                            frame_period_ms,
                        ))
                    } else {
                        None
                    }
                },
            )
            .unwrap_or((None, 5.0));

        let (volume_curve, volume_curve_frame_period_ms) = processor_params
            .and_then(|(_, _, frame_period_ms, renderer_id, _, extra_curves, _)| {
                if renderer_id == "nsf_hifigan_onnx" {
                    Some((
                        extra_curves
                            .get("hifigan_volume")
                            .cloned()
                            .map(std::sync::Arc::new),
                        frame_period_ms,
                    ))
                } else {
                    None
                }
            })
            .unwrap_or((None, 5.0));

        // ── 查询整 Clip 渲染缓存 ───────────────────────────────────────────
        // 改法 C+D：优先从 pending_rendered_keys 查找渲染线程传递的 cache_key，
        // 消除双重 hash 计算导致的不一致问题（采样率竞态、浮点精度差异等）。
        // 若 pending_rendered_keys 中无记录，回退到自行计算 hash（兼容非预渲染路径）。
        let (rendered_pcm, breath_noise_pcm, needs_synthesis) = {
            let needs_pitch_edit =
                crate::pitch_editing::does_clip_need_processor_render(timeline, clip, start_sec);

            if needs_pitch_edit {
                // 优先从 pending_rendered_keys 查找渲染线程传递的 cache_key
                let pending_key = crate::synth_clip_cache::lookup_pending_rendered_key(&clip.id);

                let cache_key = if let Some(pk) = pending_key {
                    if debug {
                        eprintln!(
                            "[snapshot] clip_id={} using pending_rendered_key hash={:#018x}",
                            clip.id, pk.param_hash
                        );
                    }
                    Some(pk)
                } else {
                    // 回退：自行计算 hash（兼容非预渲染路径，如 AudioReady rebuild）
                    if let Some((
                        _,
                        pitch_edit,
                        frame_period_ms,
                        renderer_id,
                        _,
                        extra_curves,
                        extra_params,
                    )) = processor_params
                    {
                        let end_frame = start_frame.saturating_add(length_frames);
                        let param_hash = crate::synth_clip_cache::compute_rendered_clip_hash(
                            &clip.id,
                            source_path,
                            start_frame,
                            end_frame,
                            out_rate,
                            renderer_id,
                            pitch_edit,
                            frame_period_ms,
                            playback_rate,
                            extra_curves,
                            extra_params,
                            None,
                        );
                        if debug {
                            eprintln!(
                                "[snapshot] clip_id={} fallback self-computed hash={:#018x} (no pending key)",
                                clip.id, param_hash
                            );
                        }
                        Some(crate::synth_clip_cache::RenderedClipCacheKey {
                            clip_id: clip.id.clone(),
                            param_hash,
                        })
                    } else {
                        None
                    }
                };
                if let Some(key) = cache_key {
                    // 【缩小锁范围，防止死锁】
                    let (mut pcm, breath_noise) = {
                        let mut rendered_cache =
                            crate::synth_clip_cache::global_rendered_clip_cache()
                                .lock()
                                .unwrap_or_else(|e| e.into_inner());
                        let cache_entry = rendered_cache.get(&key).cloned();
                        (
                            cache_entry.as_ref().map(|e| e.pcm_stereo.clone()),
                            cache_entry.and_then(|e| e.breath_noise_stereo.clone()),
                        )
                    };

                    if let Some((
                        pitch_orig,
                        _pitch_edit,
                        frame_period_ms,
                        renderer_id,
                        entry,
                        _,
                        _,
                    )) = processor_params
                    {
                        if renderer_id == "nsf_hifigan_onnx"
                            && crate::pitch_editing::hifigan_tension_active_for_clip(
                                entry, clip, start_sec,
                            )
                        {
                            let tension_curve =
                                crate::pitch_editing::hifigan_tension_curve_for_clip(entry, clip);
                            let tension_hash =
                                crate::synth_clip_cache::compute_hifigan_tension_hash(
                                    &clip.id,
                                    key.param_hash,
                                    start_frame,
                                    start_frame.saturating_add(length_frames),
                                    out_rate,
                                    frame_period_ms,
                                    pitch_orig,
                                    tension_curve,
                                );
                            let tension_key =
                                crate::synth_clip_cache::TensionRenderedClipCacheKey {
                                    clip_id: clip.id.clone(),
                                    base_param_hash: key.param_hash,
                                    tension_hash,
                                };

                            // 同样缩小 tension 缓存的锁范围
                            pcm = {
                                let mut tension_cache =
                                    crate::synth_clip_cache::global_tension_rendered_clip_cache()
                                        .lock()
                                        .unwrap_or_else(|e| e.into_inner());
                                tension_cache
                                    .get(&tension_key)
                                    .map(|entry| entry.pcm_stereo.clone())
                            };

                            if debug {
                                eprintln!(
                                    "[snapshot] clip_id={} tension_hash={:#018x} tension_cache_hit={}",
                                    clip.id, tension_hash, pcm.is_some()
                                );
                            }
                        }
                    }

                    if debug {
                        eprintln!(
                            "[snapshot] clip_id={} hash={:#018x} rendered_cache_hit={} needs_synthesis=true",
                            clip.id, key.param_hash, pcm.is_some()
                        );
                    }

                    if pcm.is_none() {
                        // 【优雅降级】：尝试获取该 Clip 最近一次成功的渲染结果作为过渡垫音
                        let mut fallback_pcm = None;
                        let mut fallback_breath = None;

                        let needs_tension = processor_params.map_or(
                            false,
                            |(_, _, _, renderer_id, entry, _, _)| {
                                renderer_id == "nsf_hifigan_onnx"
                                    && crate::pitch_editing::hifigan_tension_active_for_clip(
                                        entry, clip, start_sec,
                                    )
                            },
                        );

                        if needs_tension {
                            fallback_pcm =
                                crate::synth_clip_cache::get_latest_tension_rendered_pcm(&clip.id);
                        }

                        if fallback_pcm.is_none() {
                            if let Some((p, b)) =
                                crate::synth_clip_cache::get_latest_rendered_pcm(&clip.id)
                            {
                                fallback_pcm = Some(p);
                                fallback_breath = b;
                            }
                        }

                        if let Some(old_pcm) = fallback_pcm {
                            if debug {
                                eprintln!("[snapshot] clip_id={} exact hash missed, seamless fallback to PREVIOUS rendered PCM", clip.id);
                            }
                            // 即使是旧版缓存，我们也要将 needs_synthesis 设为 true，
                            // 这样下一次重新触发播放时，引擎才会识别到最新 Hash 未渲染而去重新渲染
                            (Some(old_pcm), fallback_breath, true)
                        } else {
                            // 连旧版本都没有（可能是这个 clip 第一次编辑）：
                            // 不再回退原声，避免出现“原始音频与处理后音频混播”残留问题。
                            // 统一进入静音等待，直到当前参数对应的渲染结果可用。
                            let state = crate::synth_clip_cache::get_clip_rendering_state(&clip.id);
                            let is_rendering = matches!(
                                state,
                                Some(crate::clip_rendering_state::ClipRenderingState::Rendering)
                            );

                            let pitch_analysis_ready = root_track_id
                                .as_ref()
                                .and_then(|root| {
                                    timeline.params_by_root_track.get(root).map(|entry| {
                                        crate::pitch_clip::get_or_compute_clip_pitch_midi_global(
                                            timeline,
                                            clip,
                                            root,
                                            entry.frame_period_ms.max(0.1),
                                        )
                                        .is_some()
                                    })
                                })
                                .unwrap_or(false);

                            if debug {
                                eprintln!(
                                    "[snapshot] clip_id={} cache missing, keep muted waiting render (ready={}, rendering={})",
                                    clip.id,
                                    pitch_analysis_ready,
                                    is_rendering
                                );
                            }
                            if is_rendering || pitch_analysis_ready {
                                eprintln!("[snapshot:WARN] clip_id={} hash={:#018x} cache_key found but rendered_pcm=None (rendering in progress, muting)", clip.id, key.param_hash);
                            }
                            (None, None, true)
                        }
                    } else {
                        (pcm, breath_noise, true)
                    }
                } else {
                    (None, None, false)
                }
            } else {
                (None, None, false)
            }
        };

        clips_out.push(EngineClip {
            clip_id: clip.id.clone(),
            track_id: clip.track_id.clone(),
            start_frame,
            length_frames,
            src: src_render,
            src_start_frame: src_start,
            src_end_frame: src_end,
            reversed: clip.reversed,
            playback_rate: playback_rate_render,
            local_src_offset_frames,
            repeat,
            fade_in_frames,
            fade_out_frames,
            gain,
            rendered_pcm,
            breath_noise_pcm,
            breath_curve,
            breath_curve_frame_period_ms,
            volume_curve,
            volume_curve_frame_period_ms,
            needs_synthesis,
        });
    }

    clips_out.sort_by_key(|c| c.start_frame);

    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        eprintln!(
            "AudioEngine: snapshot built: tracks={} clips_in_timeline={} clips_audible={} duration_frames={} sr={}",
            timeline.tracks.len(),
            timeline.clips.len(),
            clips_out.len(),
            duration_frames,
            out_rate
        );
        if let Some(c0) = clips_out.first() {
            eprintln!(
                "AudioEngine: first clip: start_frame={} len_frames={} src_start={:.1} src_end={:.1} gain={:.3} rate={:.3}",
                c0.start_frame,
                c0.length_frames,
                c0.src_start_frame,
                c0.src_end_frame,
                c0.gain,
                c0.playback_rate
            );
        }
    }

    let mut track_ids = Vec::new();
    let mut seen_track_ids = std::collections::HashSet::new();
    for clip in &clips_out {
        if seen_track_ids.insert(clip.track_id.clone()) {
            track_ids.push(clip.track_id.clone());
        }
    }

    EngineSnapshot {
        bpm,
        sample_rate: out_rate,
        duration_frames,
        track_ids: Arc::new(track_ids),
        clips: Arc::new(clips_out),
    }
}

pub(crate) fn build_snapshot_for_file(
    path: &Path,
    out_rate: u32,
    offset_sec: f64,
    cache: &Arc<Mutex<HashMap<(PathBuf, u32), ResampledStereo>>>,
) -> EngineSnapshot {
    let src = match get_resampled_stereo_cached(path, out_rate, cache) {
        Some(v) => v,
        None => return EngineSnapshot::empty(out_rate),
    };

    let offset_frames = (offset_sec.max(0.0) * out_rate as f64).round().max(0.0) as u64;
    let offset_frames = offset_frames.min(src.frames.saturating_sub(1) as u64);
    let available_frames = src.frames.saturating_sub(offset_frames as usize);
    let length_frames = available_frames.max(1) as u64;
    let src_end_frame = offset_frames
        .saturating_add(length_frames)
        .min(src.frames as u64);

    EngineSnapshot {
        bpm: 120.0,
        sample_rate: out_rate,
        duration_frames: length_frames,
        track_ids: Arc::new(vec!["__file_preview__".to_string()]),
        clips: Arc::new(vec![EngineClip {
            clip_id: "__file_preview__".to_string(),
            track_id: "__file_preview__".to_string(),
            start_frame: 0,
            length_frames,
            src,
            src_start_frame: offset_frames,
            src_end_frame,
            reversed: false,
            playback_rate: 1.0,
            local_src_offset_frames: 0,
            repeat: false,
            fade_in_frames: 0,
            fade_out_frames: 0,
            gain: 1.0,
            rendered_pcm: None,
            breath_noise_pcm: None,
            breath_curve: None,
            breath_curve_frame_period_ms: 5.0,
            volume_curve: None,
            volume_curve_frame_period_ms: 5.0,
            needs_synthesis: false,
        }]),
    }
}
