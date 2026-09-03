use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};

use super::byte_budget_cache::ByteBudgetCache;

use crate::state::{Clip, TimelineState, Track};

use super::io::{decode_resampled_stereo, get_resampled_stereo_cached, is_audio_path};
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
    // 消费窗口模型（与离线渲染一致）：
    // - 非 Loop：正放 [ss, ss+len·r)、倒放 [se−len·r, se)，clamp 到媒体内；
    //   域外部分由 local_src_offset_frames 的方向性前导静音表达。
    // - Loop：窗口字段只承载锚点相位，边界值不参与回绕数学（此处取值
    //   仅供"窗口退化跳过"守卫之外的通用路径，Loop 分支不会被其截断）。
    let (win_start, win_end) = crate::state::clip_playback_window_sec(clip);
    source_bounds_frames(win_start.max(0.0), win_end, src_total_frames, sr)
}

pub(crate) fn make_stretch_key(
    path: &Path,
    out_rate: u32,
    algorithm: crate::time_stretch::UserStretchAlgorithm,
    source_start: f64,
    source_end: f64,
    playback_rate: f64,
) -> StretchKey {
    StretchKey {
        path: path.to_path_buf(),
        out_rate,
        algorithm,
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
    stretch_cache: &Arc<Mutex<ByteBudgetCache<StretchKey, ResampledStereo>>>,
    app_handle: Option<&tauri::AppHandle>,
) {
    // 计算 track_gain，删除了无用的 bpm 和冗余的 audible_tracks
    let track_gain = compute_track_gains(&timeline.tracks);
    let stretch_algorithm = crate::time_stretch::resolved_user_external_stretch_algorithm();
    let runtime_stretch_algorithm = stretch_algorithm.to_runtime();

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
        let processor_handles_stretch = timeline
            .resolve_root_track_id(&clip.track_id)
            .and_then(|root| timeline.tracks.iter().find(|t| t.id == root))
            .map(|t| {
                let kind = crate::state::SynthPipelineKind::from_track_algo(&t.pitch_analysis_algo);
                crate::renderer::processor_handles_time_stretch(kind, t.compose_enabled)
            })
            .unwrap_or(false);
        let playback_rate = clip.playback_rate as f64;
        let playback_rate = if playback_rate.is_finite() && playback_rate > 0.0 {
            playback_rate
        } else {
            1.0
        };
        if processor_handles_stretch || (playback_rate - 1.0).abs() <= 1e-6 {
            continue;
        }
        let path = Path::new(source_path);
        if !is_audio_path(path) {
            continue;
        }

        // Loop（循环源）：拉伸对象是**整个文件**（回绕发生在完整媒体的拉伸
        // 版本上，与 build_snapshot 的换入键保持一致 —— 两处必须使用同一
        // `clip_loop_wrap_total_sec` 取值，否则键不匹配、拉伸结果永远无法命中）。
        // 非 Loop 拉伸窗口。
        let (job_start_sec, job_end_sec) = if clip.loop_enabled {
            (0.0f64, crate::state::clip_loop_wrap_total_sec(clip))
        } else {
            // 派生窗口：与 build_snapshot 换入键（下方 clip_source_bounds_frames
            // / 同一 effective 端点）保持成对，否则键不匹配永远无法命中。
            (
                clip.source_start_sec.max(0.0),
                crate::state::clip_effective_source_end_sec(clip),
            )
        };

        let key = make_stretch_key(
            path,
            out_rate,
            stretch_algorithm,
            job_start_sec,
            job_end_sec,
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
            algorithm: runtime_stretch_algorithm,
            source_start_sec: job_start_sec,
            source_end_sec: job_end_sec,
            playback_rate,
            clip_name,
            app_handle: app_handle.map(|h| std::sync::Arc::new(h.clone())),
        });
    }
}

pub(crate) fn build_snapshot(
    timeline: &TimelineState,
    out_rate: u32,
    cache: &Arc<Mutex<ByteBudgetCache<(PathBuf, u32), ResampledStereo>>>,
    stretch_cache: &Arc<Mutex<ByteBudgetCache<StretchKey, ResampledStereo>>>,
) -> EngineSnapshot {
    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
    let stretch_algorithm = crate::time_stretch::resolved_user_external_stretch_algorithm();
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
        let processor_handles_stretch =
            crate::pitch_editing::processor_should_handle_stretch(timeline, clip);

        let src = match get_resampled_stereo_cached(path, out_rate, cache) {
            Some(v) => {
                // ★ 运行时缓存有效性校验：对比 clip 记录的文件 mtime 与磁盘当前 mtime。
                // 若不一致说明文件已被外部替换，缓存的解码数据已过时，需同步重新解码。
                let cache_valid = clip.source_file_mtime.map_or(true, |expected_mtime| {
                    std::fs::metadata(path)
                        .ok()
                        .and_then(|m| m.modified().ok())
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_secs())
                        == Some(expected_mtime)
                });
                if cache_valid {
                    v
                } else {
                    // 文件已变更 → 同步重新解码并更新缓存
                    if debug {
                        log::warn!(
                            "[snapshot] clip_id={} source_file_mtime mismatch — forcing re-decode path={}",
                            clip.id,
                            path.display()
                        );
                    }
                    match decode_resampled_stereo(path, out_rate) {
                        Some(fresh) => {
                            let key = (path.to_path_buf(), out_rate);
                            if let Ok(mut map) = cache.lock() {
                                let pcm_bytes = fresh.pcm_bytes();
                                map.insert(key, fresh.clone(), pcm_bytes);
                            }
                            fresh
                        }
                        None => {
                            if debug {
                                log::error!("[snapshot] clip_id={} re-decode failed, using cached data as fallback", clip.id);
                            }
                            v
                        }
                    }
                }
            }
            None => {
                if debug {
                    log::warn!(
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
        // Loop（循环源）只依赖锚点 + 完整媒体缓冲（窗口字段仅承载锚点相位，
        // 甚至可能是 split 产生的环绕窗口），不能因窗口退化被跳过 ——
        // 否则实时渲染静音而离线导出有声。
        if !clip.loop_enabled && src_end.saturating_sub(src_start) == 0 {
            continue;
        }

        // ── Loop（循环源）────────────────────────────────────────────────────
        // 语义：对**完整媒体文件**做模运算回绕（对齐 REAPER Loop source）：
        //   正放 src(t) = floor_mod(source_start + t·rate, D)
        //   倒放 src(t) = floor_mod(source_end   − t·rate, D)
        // 锚点取进入媒体的一侧，使"启用 Loop 的瞬间可见内容保持连续"；
        // 负/超界锚点（向左延伸过的 Clip）经 floor_mod 正确环绕。
        // 拉伸 / Formant 分支换入的缓冲覆盖整个文件（或其拉伸版本），
        // 锚点按缓冲域等比换算后回绕语义不变。
        let decoded_total_frames = src.frames.max(1) as i64;
        // 媒体总时长（秒）：倒放锚点的 exclusive 末端需 clamp 到该值，
        // 与离线渲染（mixdown/render_single_clip 的 min(end, D)）保持一致，
        // 防止异常超界的 source_end 把锚点映射到错误的环绕相位。
        let decoded_dur_sec = decoded_total_frames as f64 / out_rate.max(1) as f64;

        // 非 Loop：消费窗口与媒体域完全无交集（整窗在媒体下方 / 上方）
        // → 纯静音，直接跳过。边界函数会把空窗口强造成 1 帧"可听"切片，
        // 不显式跳过会让全静音 Clip 产生幻影采样。
        if !clip.loop_enabled {
            let (win_start, win_end) = crate::state::clip_playback_window_sec(clip);
            if win_end <= 1e-9 || win_start >= decoded_dur_sec - 1e-9 {
                continue;
            }
        }

        let rev_anchor_sec = clip.source_end_sec.min(decoded_dur_sec);
        let mut loop_anchor_frame: Option<i64> = if clip.loop_enabled {
            let raw = if clip.reversed {
                (rev_anchor_sec * out_rate as f64).round() as i64 - 1
            } else {
                (clip.source_start_sec * out_rate as f64).round() as i64
            };
            Some(raw.rem_euclid(decoded_total_frames))
        } else {
            None
        };
        let repeat = clip.loop_enabled;

        // Loop 模式下换算锚点到指定缓冲长度的辅助闭包（拉伸域按 1/rate 缩放）。
        let rescale_anchor = |buf_frames: usize, rate_scale: f64| -> Option<i64> {
            let rate_scale = if rate_scale.is_finite() && rate_scale > 1e-6 {
                rate_scale
            } else {
                1.0
            };
            let raw = if clip.reversed {
                (rev_anchor_sec * out_rate as f64 / rate_scale).round() as i64 - 1
            } else {
                (clip.source_start_sec * out_rate as f64 / rate_scale).round() as i64
            };
            Some(raw.rem_euclid(buf_frames.max(1) as i64))
        };

        // Leading silence（前导静音）按**消费方向**取值（与离线渲染一致）：
        // - 正放：窗口起点越过媒体起点（ss<0）→ 前导静音；
        // - 倒放：窗口终点越过媒体末端（se>D）→ 前导静音；
        // - Loop：负 source_start 是环绕锚点，无前导静音。
        // 既有约定：负的 local offset = 先静音后内容。
        let local_src_offset_frames: i64 = if clip.loop_enabled {
            0
        } else if clip.reversed {
            let pr = playback_rate.max(1e-6);
            let decoded_dur_sec = decoded_total_frames as f64 / out_rate.max(1) as f64;
            // 用**原始** source_end（未 clamp）与媒体末端比较 —— 超出部分即前导静音。
            let pre_silence_sec = (clip.source_end_sec - decoded_dur_sec).max(0.0) / pr;
            if pre_silence_sec > 0.0 {
                -((pre_silence_sec * out_rate as f64).round().max(0.0) as i64)
            } else {
                0
            }
        } else if clip.source_start_sec.is_finite() && clip.source_start_sec < 0.0 {
            let pr = playback_rate.max(1e-6);
            let pre_silence_sec = (-clip.source_start_sec) / pr;
            let frames = (pre_silence_sec * out_rate as f64).round().max(0.0) as i64;
            -frames
        } else {
            0
        };

        // If the clip has formant morph enabled, build/use a clip-local preprocessed buffer first,
        // then feed that buffer into later stretch / processor stages.
        let formant_params = clip.formant_morph.as_ref().filter(|params| params.enabled);
        let mut src_render = src;
        let mut playback_rate_render = playback_rate;
        if let Some(params) = formant_params {
            // Loop：对整个文件做 Formant（自然顺序，方向由 mix 的锚点回绕处理，
            // 不再预反转）；非 Loop 保持原窗口切片 + 预反转行为。
            let slice_start = if clip.loop_enabled {
                0usize
            } else {
                (src_start as usize).saturating_mul(2)
            };
            let slice_end = if clip.loop_enabled {
                src_render.pcm.len()
            } else {
                (src_end as usize)
                    .saturating_mul(2)
                    .min(src_render.pcm.len())
            };
            let mut clip_pcm = src_render.pcm[slice_start..slice_end].to_vec();
            if clip.reversed && !clip.loop_enabled {
                crate::mixdown::reverse_interleaved_frames(&mut clip_pcm, 2);
            }

            let key = crate::formant_cache::make_formant_cache_key(
                &clip.id,
                path,
                out_rate,
                if clip.loop_enabled {
                    0.0
                } else {
                    // 非 Loop：消费窗口起点（正放=ss、倒放=se−len·r，clamp ≥0）
                    // —— 与离线渲染（mixdown / render_single_clip）的键成对。
                    crate::state::clip_playback_window_sec(clip).0.max(0.0)
                },
                if clip.loop_enabled {
                    // 与预计算（compute_formant_cache_entry_for_clip）使用同一
                    // 来源（优先 clip 元数据）—— 避免 wav 头时长与解码帧时长在
                    // 1ms 量化边界处错开键值，导致预计算永不命中、状态闪烁。
                    crate::state::clip_source_media_duration_sec(clip)
                        .unwrap_or_else(|| src_render.frames as f64 / out_rate as f64)
                } else {
                    // 消费窗口终点：正放派生（起点+长度×速率）、倒放为存储 se。
                    crate::state::clip_playback_window_sec(clip).1
                },
                clip.reversed && !clip.loop_enabled,
                // 实时域：完整文件自然顺序 / 窗口切片，绝非离线回绕平铺域。
                false,
                params,
            );
            match crate::formant_cache::get_or_compute_formant_audio(
                key, &clip_pcm, out_rate, params,
            ) {
                Ok(entry) => {
                    crate::formant_cache::formant_debug_log(format!(
                        "snapshot using formant clip_id={} frames={} diff={:.8} processor_handles_stretch={} playback_rate={:.4}",
                        clip.id,
                        entry.frames,
                        crate::formant_cache::average_abs_diff(&clip_pcm, entry.pcm_stereo.as_ref()),
                        processor_handles_stretch,
                        playback_rate,
                    ));
                    src_render = ResampledStereo {
                        sample_rate: entry.sample_rate,
                        frames: entry.frames,
                        pcm: entry.pcm_stereo,
                    };
                    src_start = 0;
                    src_end = src_render.frames as u64;
                    if clip.loop_enabled {
                        // Formant 不改变时长：锚点换算到同长缓冲。
                        loop_anchor_frame = rescale_anchor(src_render.frames, 1.0);
                    }
                    if !processor_handles_stretch && (playback_rate - 1.0).abs() > 1e-6 {
                        let target_frames = ((src_render.frames as f64) / playback_rate)
                            .round()
                            .max(2.0) as usize;
                        let stretched = crate::time_stretch::time_stretch_interleaved(
                            src_render.pcm.as_slice(),
                            2,
                            out_rate,
                            target_frames,
                            stretch_algorithm.to_runtime(),
                        );
                        src_render = ResampledStereo {
                            sample_rate: out_rate,
                            frames: target_frames,
                            pcm: Arc::new(stretched),
                        };
                        src_end = src_render.frames as u64;
                        playback_rate_render = 1.0;
                        if clip.loop_enabled {
                            // 拉伸后的缓冲覆盖 D/rate 秒：锚点按 1/rate 缩放。
                            loop_anchor_frame = rescale_anchor(src_render.frames, playback_rate);
                        }
                    }
                }
                Err(error) => {
                    crate::formant_cache::formant_debug_log(format!(
                        "snapshot formant error clip_id={} error={}",
                        clip.id, error
                    ));
                }
            }
        } else if !processor_handles_stretch && (playback_rate - 1.0).abs() > 1e-6 {
            // Loop（循环源）：拉伸对象是**整个文件**（回绕发生在完整媒体的
            // 拉伸版本上），缓存键相应取 [0, 文件时长]，与 schedule_stretch_jobs
            // 的任务键共用 `clip_loop_wrap_total_sec`（回退链一致才能命中）；
            // 非 Loop 维持窗口拉伸。
            let (key_start, key_end) = if clip.loop_enabled {
                (0.0f64, crate::state::clip_loop_wrap_total_sec(clip))
            } else {
                // 派生窗口：与上方 schedule_stretch_jobs 的生产者键成对。
                (
                    clip.source_start_sec.max(0.0),
                    crate::state::clip_effective_source_end_sec(clip),
                )
            };
            let key = make_stretch_key(
                path,
                out_rate,
                stretch_algorithm,
                key_start,
                key_end,
                playback_rate,
            );
            if let Ok(mut m) = stretch_cache.lock() {
                if let Some(stretched) = m.get(&key) {
                    src_render = stretched.clone();
                    src_start = 0;
                    src_end = src_render.frames as u64;
                    playback_rate_render = 1.0;
                    if clip.loop_enabled {
                        // 拉伸后的缓冲覆盖 D/rate 秒；锚点按 1/rate 缩放到缓冲域。
                        loop_anchor_frame = rescale_anchor(src_render.frames, playback_rate);
                    }
                }
            }
        }

        let fade_in_frames = (clip.effective_fade_in_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as u64;
        let fade_out_frames = (clip.effective_fade_out_sec().max(0.0) * out_rate as f64)
            .round()
            .max(0.0) as u64;
        // 形状化淡化查表：仅在对应长度有效时构建，混音端按 (shape,dir) 缓存命中。
        let fade_in_lut = if fade_in_frames > 0 {
            Some(crate::fade_curves::global_fade_lut(
                clip.fade_in_shape,
                clip.fade_in_dir,
                false,
            ))
        } else {
            None
        };
        let fade_out_lut = if fade_out_frames > 0 {
            Some(crate::fade_curves::global_fade_lut(
                clip.fade_out_shape,
                clip.fade_out_dir,
                true,
            ))
        } else {
            None
        };

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

        let (volume_curve, volume_curve_frame_period_ms, pan_curve, pan_curve_frame_period_ms) =
            processor_params
                .and_then(|(_, _, frame_period_ms, renderer_id, entry, _, _)| {
                    // vslib 通过自己的控制点消费 volume/pan（与合成输出一起缓存），
                    // mix 阶段跳过，避免二次应用；未开启 Compose 时按约定不生效。
                    if renderer_id == "vslib" {
                        return None;
                    }
                    let volume_curve =
                        crate::pitch_editing::common_volume_curve_for_clip(entry, clip)
                            .map(|curve| std::sync::Arc::new(curve.to_vec()));
                    let pan_curve = crate::pitch_editing::common_pan_curve_for_clip(entry, clip)
                        .map(|curve| std::sync::Arc::new(curve.to_vec()));
                    Some((volume_curve, frame_period_ms, pan_curve, frame_period_ms))
                })
                .unwrap_or((None, 5.0, None, 5.0));

        // ── 查询整 Clip 渲染缓存 ───────────────────────────────────────────
        // 改法 C+D：优先从 pending_rendered_keys 查找渲染线程传递的 cache_key，
        // 消除双重 hash 计算导致的不一致问题（采样率竞态、浮点精度差异等）。
        // 若 pending_rendered_keys 中无记录，回退到自行计算 hash（兼容非预渲染路径）。
        let (rendered_pcm, breath_noise_pcm, needs_synthesis) = {
            let needs_pitch_edit =
                crate::pitch_editing::does_clip_need_processor_render(timeline, clip, start_sec);

            debug_eprintln!(
                "[snapshot] clip_id={} needs_pitch_edit={}",
                clip.id, needs_pitch_edit
            );

            if needs_pitch_edit {
                // 优先从 pending_rendered_keys 查找渲染线程传递的 cache_key
                let pending_key = crate::synth_clip_cache::lookup_pending_rendered_key(&clip.id);

                let cache_key = if let Some(pk) = pending_key {
                    if debug {
                        log::warn!(
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
                            clip.formant_morph.as_ref().filter(|params| params.enabled),
                            None,
                            clip.source_file_mtime,
                            clip.loop_enabled,
                            (
                                (clip.source_start_sec * 1000.0).round() as i64,
                                (clip.source_end_sec * 1000.0).round() as i64,
                            ),
                        );
                        if debug {
                            log::warn!(
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
                                log::warn!(
                                    "[snapshot] clip_id={} tension_hash={:#018x} tension_cache_hit={}",
                                    clip.id, tension_hash, pcm.is_some()
                                );
                            }
                        }
                    }

                    if debug {
                        log::warn!(
                            "[snapshot] clip_id={} hash={:#018x} rendered_cache_hit={} needs_synthesis=true",
                            clip.id, key.param_hash, pcm.is_some()
                        );
                    }

                    if pcm.is_none() {
                        // 【优雅降级】：尝试获取该 Clip 最近一次成功的渲染结果作为过渡垫音
                        let mut fallback_pcm = None;
                        let mut fallback_breath = None;
                        let needs_breath = processor_params.map_or(
                            false,
                            |(_, _, _, renderer_id, entry, _, _)| {
                                renderer_id == "nsf_hifigan_onnx"
                                    && crate::pitch_editing::extra_param_enabled(
                                        &entry.extra_params,
                                        "breath_enabled",
                                    )
                            },
                        );

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
                            fallback_pcm = crate::synth_clip_cache::get_latest_tension_rendered_pcm(
                                &clip.id,
                                clip.active_take_id.as_deref(),
                            );
                        }

                        if fallback_pcm.is_none() {
                            if let Some((p, b)) = crate::synth_clip_cache::get_latest_rendered_pcm(
                                &clip.id,
                                clip.active_take_id.as_deref(),
                            ) {
                                fallback_pcm = Some(p);
                                fallback_breath = b;
                            }
                        }

                        // 气声开启时，绝不能回退到不含独立 breath stem 的旧渲染：
                        // 否则首次播放会听到“没有气声”的旧音频，第二次播放缓存就绪后才恢复。
                        if needs_breath && fallback_breath.is_none() {
                            fallback_pcm = None;
                        }

                        if let Some(old_pcm) = fallback_pcm {
                            if debug {
                                log::warn!("[snapshot] clip_id={} exact hash missed, seamless fallback to PREVIOUS rendered PCM", clip.id);
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
                                log::warn!(
                                    "[snapshot] clip_id={} cache missing, keep muted waiting render (ready={}, rendering={})",
                                    clip.id,
                                    pitch_analysis_ready,
                                    is_rendering
                                );
                            }
                            if is_rendering || pitch_analysis_ready {
                                debug_eprintln!("[snapshot:WARN] clip_id={} hash={:#018x} cache_key found but rendered_pcm=None (rendering in progress, muting)", clip.id, key.param_hash);
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
            // 非 Loop：Formant 缓冲已预反转，方向归零交给正向遍历；
            // Loop：缓冲保持自然顺序，倒放方向由 mix 的锚点回绕（anchor − f）
            // 体现 —— 此处若清零会把"倒放循环"错放成"从文件末端正向循环"。
            reversed: if clip.loop_enabled {
                clip.reversed
            } else {
                formant_params
                    .is_some()
                    .then_some(false)
                    .unwrap_or(clip.reversed)
            },
            playback_rate: playback_rate_render,
            local_src_offset_frames,
            repeat,
            loop_anchor_frame,
            fade_in_frames,
            fade_out_frames,
            fade_in_lut,
            fade_out_lut,
            gain,
            rendered_pcm,
            breath_noise_pcm,
            breath_curve,
            breath_curve_frame_period_ms,
            volume_curve,
            volume_curve_frame_period_ms,
            pan_curve,
            pan_curve_frame_period_ms,
            needs_synthesis,
        });
    }

    clips_out.sort_by_key(|c| c.start_frame);

    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        log::warn!(
            "AudioEngine: snapshot built: tracks={} clips_in_timeline={} clips_audible={} duration_frames={} sr={}",
            timeline.tracks.len(),
            timeline.clips.len(),
            clips_out.len(),
            duration_frames,
            out_rate
        );
        if let Some(c0) = clips_out.first() {
            log::warn!(
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

#[cfg(test)]
mod tests {
    use super::build_snapshot;
    use crate::audio_engine::byte_budget_cache::ByteBudgetCache;
    use crate::audio_engine::resource_manager::DecodeCache;
    use crate::audio_engine::types::{ResampledStereo, StretchKey};
    use crate::state::Clip;
    use std::path::PathBuf;
    use std::sync::{Arc, Mutex};

    #[test]
    fn make_stretch_key_distinguishes_algorithm() {
        let path = std::path::Path::new("demo.wav");
        let linear = super::make_stretch_key(
            path,
            48_000,
            crate::time_stretch::UserStretchAlgorithm::Linear,
            0.0,
            1.0,
            0.75,
        );
        let soundtouch = super::make_stretch_key(
            path,
            48_000,
            crate::time_stretch::UserStretchAlgorithm::Soundtouch,
            0.0,
            1.0,
            0.75,
        );
        assert_ne!(linear, soundtouch);
    }

    fn timeline_with_volume_curve() -> crate::state::TimelineState {
        let mut tl = crate::state::TimelineState::default();
        let root = tl.tracks[0].id.clone();
        tl.tracks[0].pitch_analysis_algo = crate::state::PitchAnalysisAlgo::NsfHifiganOnnx;
        tl.clips.push(Clip {
            takes: vec![],
            active_take_id: None,
            clip_playback_rate: 1.0,
            id: "clip-volume".to_string(),
            track_id: root.clone(),
            name: "volume clip".to_string(),
            start_sec: 0.0,
            length_sec: 0.5,
            color: "#ffffff".to_string(),
            source_path: Some("/tmp/hifishifter-volume-test.aiff".to_string()),
            source_path_relative: None,
            duration_sec: Some(0.5),
            duration_frames: Some(22_050),
            source_sample_rate: Some(44_100),
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: 0.5,
            playback_rate: 1.0,
            reversed: false,
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
        });
        let params = tl
            .params_by_root_track
            .entry(root)
            .or_default();
        params.extra_curves.insert("volume".to_string(), vec![0.25f32; 10]);
        tl
    }

    #[test]
    fn build_snapshot_attaches_volume_curve_to_rendered_and_raw_clips() {
        let tl = timeline_with_volume_curve();
        let out_rate = 44_100;
        let path = PathBuf::from("/tmp/hifishifter-volume-test.aiff");
        let source_path = path.clone();
        std::fs::write(&source_path, b"stub").expect("create source stub");

        // 预填充解码缓存，绕过磁盘 I/O；build_snapshot 只读缓存。
        // 缓冲至少覆盖 1s；测试 clip 只消费 0.5s。
        let pcm = vec![0.5f32; 44_100 * 2];
        let decoded = ResampledStereo {
            sample_rate: out_rate,
            frames: pcm.len() / 2,
            pcm: Arc::new(pcm),
        };
        let cache: Arc<Mutex<DecodeCache>> =
            Arc::new(Mutex::new(ByteBudgetCache::new(4, u64::MAX)));
        cache
            .lock()
            .unwrap()
            .insert((path.clone(), out_rate), decoded, 44_100 * 2 * 4);
        let stretch_cache: Arc<Mutex<ByteBudgetCache<StretchKey, ResampledStereo>>> =
            Arc::new(Mutex::new(ByteBudgetCache::new(4, u64::MAX)));

        let snap = build_snapshot(&tl, out_rate, &cache, &stretch_cache);
        std::fs::remove_file(&source_path).ok();
        assert_eq!(snap.clips.len(), 1);
        let engine_clip = &snap.clips[0];

        let volume = engine_clip
            .volume_curve
            .as_ref()
            .expect("volume curve must be attached");
        assert_eq!(volume.as_slice(), &[0.25f32; 10]);
        let fp = engine_clip.volume_curve_frame_period_ms;
        assert!(fp > 0.0 && fp.is_finite(), "invalid frame period {fp}");
        // volume 是混音级参数：即使 clip 不需要合成，也必须实时生效。
        assert!(!engine_clip.needs_synthesis, "plain clip should use source PCM");
        assert!(
            engine_clip.breath_curve.is_none(),
            "breath should not hijack plain clip volume"
        );
    }

}

pub(crate) fn build_snapshot_for_file(
    path: &Path,
    out_rate: u32,
    offset_sec: f64,
    cache: &Arc<Mutex<ByteBudgetCache<(PathBuf, u32), ResampledStereo>>>,
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
            loop_anchor_frame: None,
            fade_in_frames: 0,
            fade_out_frames: 0,
            fade_in_lut: None,
            fade_out_lut: None,
            gain: 1.0,
            rendered_pcm: None,
            breath_noise_pcm: None,
            breath_curve: None,
            breath_curve_frame_period_ms: 5.0,
            volume_curve: None,
            volume_curve_frame_period_ms: 5.0,
            pan_curve: None,
            pan_curve_frame_period_ms: 5.0,
            needs_synthesis: false,
        }]),
    }
}
