// 播放与渲染命令门面下的"播放 / 预渲染"实现。
//
// 主要内容：
// - `play_original`：带音高编辑时的前台 clip 级增量预渲染 + 实时混音；
// - `start_background_render` / `cancel_background_render` / `request_background_render`：
//   "后台预渲染"整轮渲染的启动、取消与重启调度；
// - `collect_clips_needing_render` / `render_single_clip` 等渲染辅助函数。
//
// 与其他模块的关系：
// - `commands.rs` 是本文件对外的唯一入口（`#[tauri::command]` 只允许出现在那里）；
// - `audio_engine/engine.rs` 在缓存失效时会读写本文件的 `BG_RENDER_*` 全局标志，
//   以中断并重启后台渲染；
// - `commands/project.rs` 在新建/打开工程时调用 `cancel_background_render`；
// - 渲染取消信号的抽象位于 `commands/render_cancel.rs`，
//   前台与后台渲染各用一套彼此隔离的令牌（详见该模块头注释）。

use crate::models::PlaybackStatePayload;
use crate::state::AppState;
use tauri::Emitter;
use tauri::Manager;
use tauri::State;

use super::common::{guard_json_command, PlaybackRenderingStateEvent};

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
/// `audio_engine/engine.rs` 的缓存失效处理设置此标志以中断旧渲染线程。
///
/// ★ 生命周期契约：置位与清除必须成对出现，否则标志会永久粘滞。
///   - 置位：统一通过 `commands::render_cancel::request_global_cancel()`
///     （单一入口，便于审计成对性）。调用点有 `cancel_background_render`
///     （仅当确实有后台渲染在跑），以及后台渲染循环的重启静默窗口晋升
///     （`request_bg_render_restart` 记录的重启请求挂起超过窗口后，由渲染
///     线程在 clip 边界调用）。请**不要**直接 `store(true, ..)`。
///     "新轮次不受历史残留影响"由 `BG_RENDER_GENERATION` 代数守卫负责，
///     见 `render_cancel.rs` 顶部说明。
///   - 清除：`start_background_render` 开头、后台渲染各条退出分支，
///     以及 `request_background_render` 的 disabled 分支（兜底）。
///
/// 历史上这里曾被无条件置位：由于本标志只在"后台预渲染"的路径上被清除，
/// 一旦后台预渲染未启用，打开工程后它便永远为 true，导致后续所有前台
/// `play_original` 预渲染在解码后的第一个检查点被中止（长音频渲染不出来）。
///
/// ★ 前台播放预渲染不得读取本标志 —— 它应使用
/// `commands::render_cancel::RenderCancelToken` 传入的私有令牌，
/// 由 `cancel_background_render` 通过 `cancel_all_foreground()` 显式通知。
pub(crate) static BG_RENDER_CANCEL: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 后台渲染重启标志。缓存失效 / 编辑发生时置位（见 [`request_bg_render_restart`]），
/// 渲染循环在 clip 边界仅当请求挂起超过静默窗口后才将其升级为实际取消；
/// 退出线程看到它为 true 时自动启动新一轮渲染。
pub(crate) static BG_RENDER_RESTART_NEEDED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 最近一次重启请求的时间戳（进程启动相对毫秒）。
/// 0 表示当前没有挂起的请求（`start_background_render` 入口会清零）。
pub(crate) static BG_RENDER_RESTART_REQUESTED_AT_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 重启请求的静默窗口（毫秒）：请求挂起超过该时长后，渲染循环才把它升级为
/// 实际取消并重启。
///
/// 背景：工程增量加载 / 连续编辑会以几十毫秒一次的频率使缓存失效并请求重启；
/// 旧的"失效即取消"实现导致渲染每轮只推进两三个 clip 就被打破（实测一次工程
/// 加载产生 202 次重启 / 4 分钟）。合流后，窗口内时间戳被持续刷新、当前轮不被
/// 打断，风暴结束后由退出路径串联的下一轮渲染最终状态。单次编辑最多延迟一个
/// 窗口才重启，对"后台"预渲染无感。
pub(crate) const BG_RENDER_RESTART_QUIET_WINDOW_MS: u64 = 500;

fn now_millis() -> u64 {
    static PROCESS_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    PROCESS_START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

/// 重启请求是否已"冷却"（挂起超过静默窗口，期间没有更新的请求）。
/// `last_requested_at_ms == 0`（无记录）视为已冷却，兜底保持旧的立即取消语义。
fn bg_render_restart_request_is_stale(last_requested_at_ms: u64, now_ms: u64) -> bool {
    now_ms.saturating_sub(last_requested_at_ms) >= BG_RENDER_RESTART_QUIET_WINDOW_MS
}

/// 请求重启后台渲染（合流版）。
///
/// 不立即取消在途渲染：仅置位重启标志并刷新请求时间戳。渲染循环在 clip
/// 边界发现请求挂起超过静默窗口后，才通过
/// `render_cancel::request_global_cancel()` 升级为实际取消，
/// 退出线程随后自动启动新一轮渲染。
///
/// 失效风暴期间时间戳被持续刷新，当前轮不会被反复打断；每轮完成后由退出
/// 路径串联下一轮，风暴结束后恰好剩一轮渲染最终状态。
pub(crate) fn request_bg_render_restart() {
    use std::sync::atomic::Ordering;
    BG_RENDER_RESTART_NEEDED.store(true, Ordering::Release);
    BG_RENDER_RESTART_REQUESTED_AT_MS.fetch_max(now_millis(), Ordering::Release);
}

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
        log::warn!("[play_original] called start_sec={start_sec}");
        // 在同一把锁的作用域内克隆时间线：分离读取时，并发 checkpoint
        // （先解锁后 bump 版本）会让旧 timeline 配上新版本号。
        let timeline = match state.timeline.lock() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        };
        let bpm = timeline.bpm;
        let playhead_sec = timeline.playhead_sec;
        if !(bpm.is_finite() && bpm > 0.0) {
            return serde_json::json!({"ok": false, "error": "invalid bpm"});
        }
        let start_sec = playhead_sec.max(0.0) + start_sec.max(0.0);

        // ── 渲染需求评估（与后台预渲染开关无关）─────────────────────────────
        // 开关只决定渲染的**触发时机**：启用 = 编辑即渲染（播放时通常已就绪）；
        // 关闭 = 播放时按需渲染。渲染途中的播放行为（原地等待 + 就绪自动播放）
        // 两者完全一致：音频回调遇到未就绪窗口即冻结，渲染线程发布结果后
        // 自动开始 / 继续播放。
        let clips_needing_render =
            collect_clips_needing_render(&timeline, state.audio_engine.sample_rate_hz());
        let need_prerender = !clips_needing_render.is_empty();
        let snapshot_has_pending = state.audio_engine.snapshot_has_pending_clips();
        log::warn!(
            "[play_original] clips_needing_render={} need_prerender={} snapshot_has_pending={}",
            clips_needing_render.len(),
            need_prerender,
            snapshot_has_pending
        );

        // 需要渲染但当前没有渲染 pass 在跑（后台预渲染关闭时的常态）→
        // 按需启动一轮：本次播放需要这些 Clip 的结果。已在运行的 pass 无需
        // 动作 —— 其待渲染列表本就覆盖当前所有待渲染 Clip。
        if need_prerender && !BG_RENDER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some(app) = state.app_handle.get() {
                ensure_render_pass_running(app);
            }
        }

        // ── 幂等：已在播放时绝不重启传输层 ──────────────────────────────────────
        // 每次 play 调用都会 seek 回锚点；若调用方重复触发（按钮 + 快捷键、
        // 双击、UI 自动重试），传输层会被反复拉回起播点 —— 表现正是用户报告的
        // "音频在放但播放光标不动（光标始终停在锚点）" 与 "重复触发播放/叠音"。
        // 因此已在播放（且目标为时间线）时直接返回：只确保渲染需求被覆盖，
        // 绝不再 seek / update_timeline / set_playing。
        if state.audio_engine.is_playing()
            && state
                .audio_engine
                .snapshot_state()
                .target
                .as_deref()
                == Some("original")
        {
            if need_prerender && !BG_RENDER_ACTIVE.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(app) = state.app_handle.get() {
                    ensure_render_pass_running(app);
                }
            }
            log::warn!("[play_original] already playing — idempotent no-op (no restart)");
            return serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec});
        }

        // ── 立即武装传输层 ─────────────────────────────────────────────────────
        // 覆盖起播窗口的 Clip 尚未就绪时，音频回调进入原地等待（保持播放态、
        // 位置冻结、静音输出）；渲染线程每完成一个 Clip 都会推送刷新引擎
        // 快照，等待随之解除并自动开始播放。
        state.audio_engine.seek_sec(start_sec);
        state.audio_engine.update_timeline(timeline);
        state.audio_engine.set_playing(true, Some("original"));
        serde_json::json!({"ok": true, "playing": "original", "start_sec": start_sec})
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
        log::warn!(
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
            log::warn!(
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
///
/// # 参数 `cancel`
/// 本轮渲染的取消令牌。各耗时阶段之间会通过它设置检查点，一旦请求取消就
/// 立即返回 `BG_RENDER_CANCELLED_ERR`。
///
/// ★ 不要在此函数内部直接读取全局 `BG_RENDER_CANCEL`：该标志由
/// `cancel_background_render`（新建/打开工程时必被调用）无条件置位，却只在
/// 后台预渲染的启动/收尾路径清除。若后台预渲染未启用，它会永久保持为 true，
/// 使前台 `play_original` 预渲染在解码后的第一个检查点就"失败"，
/// 表现为长音频渲染不出来、播放降级为原声。取消信号必须由调用方显式传入，
/// 详见 `commands/render_cancel.rs` 的模块头注释。
fn render_single_clip(
    timeline: &crate::state::TimelineState,
    clip: &crate::state::Clip,
    out_rate: u32,
    cancel: &crate::commands::render_cancel::RenderCancelToken,
) -> Result<RenderedClipOutput, String> {
    let source_path = clip
        .source_path
        .as_deref()
        .ok_or_else(|| "clip has no source_path".to_string())?;

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");

    // 阶段日志：渲染进度永久卡 0% 时，日志要能直接指出卡在哪一步。
    let stage_started = std::time::Instant::now();
    log::warn!(
        "[render] stage=begin clip_id={} rate={:.6} len_sec={:.3}",
        clip.id,
        clip.playback_rate,
        clip.length_sec
    );

    // 1. 解码源文件（走进程级解码缓存：同一源被多个 clip 引用、或同一 clip
    //    被反复编辑时不再重复读盘 + 解码，见 P1-3）。
    let decoded =
        crate::audio_utils::decode_audio_cached_interleaved(std::path::Path::new(source_path))?;
    let in_rate = decoded.sample_rate;
    let in_channels = decoded.channels;
    let pcm = decoded.pcm.clone();
    let in_channels_usize = in_channels as usize;
    let in_frames = pcm.len() / in_channels_usize;
    if in_frames < 2 {
        return Err("source audio too short".to_string());
    }
    if cancel.is_cancelled() {
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
    // 注意：clip_leading_silence_sec 内部已除以 playback_rate（state.rs:151-163），
    // 这里不能再除一次 —— 双重除法会把前导静音放大/缩小 1/rate 倍，
    // 使播放预渲染与 mixdown 导出、实时引擎三者不一致。
    let pre_silence_sec = crate::state::clip_leading_silence_sec(clip, Some(total_sec));
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
        crate::resample::resample_interleaved(&segment, in_channels_usize, in_rate, out_rate);

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

    if cancel.is_cancelled() {
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
    let processor_handles_stretch =
        crate::pitch_editing::processor_should_handle_stretch(timeline, clip);
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

    if cancel.is_cancelled() {
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

    let render_variant = |clip_variant: &crate::state::Clip| -> Result<Vec<f32>, String> {
        let mut rendered = segment.clone();
        log::warn!(
            "[render] stage=processor_begin clip_id={} elapsed_ms={}",
            clip_variant.id,
            stage_started.elapsed().as_millis()
        );
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
                    log::warn!(
                        "render_single_clip: pitch_edit applied to clip_id={}",
                        &clip_variant.id
                    );
                }
            }
            Ok(false) => {}
            Err(e) => {
                // ★ 处理器失败绝不能降级为"未处理的原始音频"：旧实现只打日志，
                // 于是 rendered 保持 source 原始内容，随后被当作渲染结果写入
                // 缓存并播放 —— 用户听到"完全没有任何算法参数的音频"，且该错误
                // 结果会被长期缓存。这里返回错误：本次 Clip 渲染失败、不写缓存，
                // 传输层继续原地等待，后续请求会重试（会话恢复后即成功）。
                return Err(format!(
                    "pitch processor failed for clip_id={}: {e}",
                    clip_variant.id
                ));
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

        Ok(rendered)
    };

    if !breath_enabled {
        if cancel.is_cancelled() {
            return Err(BG_RENDER_CANCELLED_ERR.to_string());
        }
        return Ok(RenderedClipOutput {
            rendered_stereo: render_variant(clip)?,
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
            log::warn!(
                "render_single_clip: breath_noise_cache HIT for clip_id={}, skipping second render_variant",
                clip.id
            );
        }

        let mut harmonic_only_clip = clip.clone();
        let mut harmonic_curves = merged_extra_curves.clone();
        harmonic_curves.insert("breath_gain".to_string(), vec![0.0; curve_len]);
        harmonic_only_clip.extra_params = Some(merged_extra_params.clone());
        harmonic_only_clip.extra_curves = Some(harmonic_curves);
        if cancel.is_cancelled() {
            return Err(BG_RENDER_CANCELLED_ERR.to_string());
        }
        let harmonic_only = render_variant(&harmonic_only_clip)?;

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
            log::warn!(
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
        log::warn!(
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
    // This ensures the subsequent render_variant(harmonic_only)? hits the cache
    // and only runs HiFiGAN, skipping HNSEP.
    if cancel.is_cancelled() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }
    // HNSEP 分离失败（模型缺失/推理错误）时降级为非 breath 渲染：外层已因
    // 气声跳过外部拉伸，硬错误会让整条 clip 无声等待，比"没有气声"严重得多。
    log::warn!(
        "[render] stage=hnsep_begin clip_id={} elapsed_ms={}",
        clip.id,
        stage_started.elapsed().as_millis()
    );
    let noise_mono = match crate::hnsep_onnx::infer_noise_mono(&clip.id, &mono, out_rate) {
        Ok(noise) => noise,
        Err(e) => {
            log::warn!(
                "render_single_clip: HNSEP failed for clip_id={}, falling back to non-breath render: {e}",
                clip.id
            );
            return Ok(RenderedClipOutput {
                rendered_stereo: render_variant(clip)?,
                breath_noise_stereo: None,
            });
        }
    };

    // Step 3: Render harmonic_only through ProcessorChain (HNSEP cache hits → HiFiGAN only)
    let mut harmonic_only_clip = clip.clone();
    let mut harmonic_curves = merged_extra_curves.clone();
    harmonic_curves.insert("breath_gain".to_string(), vec![0.0; curve_len]);
    harmonic_only_clip.extra_params = Some(merged_extra_params.clone());
    harmonic_only_clip.extra_curves = Some(harmonic_curves);
    if cancel.is_cancelled() {
        return Err(BG_RENDER_CANCELLED_ERR.to_string());
    }
    let harmonic_only = render_variant(&harmonic_only_clip)?;

    // Step 4: Convert noise mono to stereo, matching harmonic_only length
    let out_len = harmonic_only.len();
    let out_frames = out_len / 2;
    let noise_stereo: Vec<f32> = {
        let noise_mono_raw = noise_mono.as_ref();
        // 时间拉伸若由处理器内部完成（mel 域），谐波输出是**时间轴**长度，
        // 而 HNSEP 的噪声 stem 仍是**源速率**长度。必须对齐后再转立体声；
        // 对齐必须用与谐波一致的拉伸算法 —— 线性重采样会把气声的谱包络
        // 按 1/rate 缩放（慢放变闷、快放混叠），听感即"气声没有被正确拉伸"。
        let aligned = crate::renderer::chain::align_noise_stem_to_len(
            noise_mono_raw,
            out_rate,
            out_frames,
            crate::time_stretch::resolved_external_stretch_algorithm(),
        );
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
    // DAW 暂停语义：停止播放时把引擎当前可听位置（base + elapsed，即暂停点）
    // 写回 timeline.playhead_sec。播放期间该字段不会前进（只有显式 seek 会
    // 改），仍停留在本次播放的起始位置；若不回写，暂停后的任何编辑操作回灌
    // 全量快照都会把前端播放头拉回"播放起始位置"。停止（Stop）流程随后由
    // 前端显式 seek 回锚点覆盖，不受影响。
    // 必须在 stop() 之前取快照，且仅在确实有可听位置时写入——否则未播放时的
    // stop 调用（如录音收尾）会把 0 写进播放头。仅改字段、不做撤销检查点
    // （与 set_transport 的 playhead 分支一致：播放头不参与撤销）。
    //
    // 前端轮询存在至多一个周期（~33ms）+ 往返的滞后：最后一次采样之后音频
    // 仍在前进。因此把引擎的精确停止位置随响应返回（stopped_at_sec），
    // 前端在暂停时把视觉光标同步到该位置——否则视觉位置与后端记录的暂停点
    // 不一致，后续任何编辑回灌快照都会让光标再次右跳到真实位置。
    //
    // ★ 两类"有可听位置"的引擎状态都要覆盖：
    // 1. is_playing=true —— 常规播放中的暂停（base+elapsed 即暂停点）；
    // 2. is_playing=false 但 position>0 —— 后台预渲染遇未渲染 clip 时引擎
    //    自动暂停：is_playing 已被音频回调翻转为 false，但 base+position 冻结
    //    在真实停止点（handle_stop 之后 position 才会归零，二者可区分）。
    //    此时同样要把该位置写回并返回，前端暂停分支才能把光标对齐到精确
    //    冻结点（最后一次轮询采样略滞后）；停在"播放起始位置"的旧字段不动，
    //    编辑回灌快照也不会把光标拉回。
    let pb = state.audio_engine.snapshot_state();
    let audible_sec = if pb.is_playing || pb.position_sec > 1e-9 {
        Some(pb.base_sec + pb.position_sec)
    } else {
        None
    };
    let mut stopped_at_sec: Option<f64> = None;
    if let Some(paused_sec) = audible_sec.filter(|sec| sec.is_finite() && *sec >= 0.0) {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.playhead_sec = paused_sec;
        stopped_at_sec = Some(paused_sec);
    }
    state.audio_engine.stop();
    serde_json::json!({ "ok": true, "stopped_at_sec": stopped_at_sec })
}

pub(super) fn get_playback_state(state: State<'_, AppState>) -> PlaybackStatePayload {
    let pb = state.audio_engine.snapshot_state();
    PlaybackStatePayload {
        ok: true,
        is_playing: pb.is_playing,
        waiting_for_render: pb.waiting_for_render,
        target: pb.target,
        base_sec: pb.base_sec,
        position_sec: pb.position_sec,
        duration_sec: pb.duration_sec,
    }
}

// ─── 节拍器（Metronome） ──────────────────────────────────────────────────────

use crate::audio_engine::metronome::{
    build_click_schedule, build_tempo_segments, grid_step_beats, MetronomeConfig, MetronomeMode,
    MetronomeSound,
};

/// 由 UI 设置解析细分模式（参与响点表展开；引擎 RT 侧不感知模式）。
fn metronome_mode_from_settings(settings: &crate::config::UiSettings) -> MetronomeMode {
    match settings.metronome_mode.as_str() {
        "beat" => MetronomeMode::Beat,
        "bar" => MetronomeMode::Bar,
        _ => MetronomeMode::Grid,
    }
}

/// 由 UI 设置解析节拍器配置（设置持久化的唯一来源是 `UiSettings`）。
fn metronome_config_from_settings(settings: &crate::config::UiSettings) -> MetronomeConfig {
    MetronomeConfig {
        enabled: settings.metronome_enabled,
        gain: settings.metronome_gain.clamp(0.0, 1.0) as f32,
        accent_enabled: settings.metronome_accent,
        sound: match settings.metronome_sound.as_str() {
            "woodblock" => MetronomeSound::Woodblock,
            "beep" => MetronomeSound::Beep,
            _ => MetronomeSound::Click,
        },
    }
}

/// 把 UI 设置中的节拍器配置同步到引擎（不重建响点表）。
pub(super) fn sync_metronome_config(state: &State<'_, AppState>) {
    let config = metronome_config_from_settings(&state.ui_settings_snapshot());
    state.audio_engine.set_metronome(config);
}

/// 由当前工程（BPM / Tempo Map / 工程长度 / 吸附网格）、项目吸附设置
/// （Swing）与 UI 设置（细分模式）重建引擎的节拍器响点表。
///
/// 响点 = 时间标尺画出的网格线（逐段局部对齐，含每个变化点重对齐与
/// Swing 偏移），与小节线重合的响点标记重音 —— 与 Tempo Map 语义一致。
/// Grid 模式的步长取**项目**吸附网格（`ProjectState.grid_size`，与时间
/// 标尺同一来源）；`UiSettings.grid_size` 是没有任何 UI 路径写入的
/// 陈旧字段，不得使用。
///
/// 调用时机：播放启动（保证与实际出声内容一致）、Tempo Map / 网格 /
/// 拍号 / Swing / 节拍器设置变化。响点表构建为毫秒级，无需后台线程。
pub(super) fn refresh_metronome_schedule(state: &State<'_, AppState>) {
    let (bpm, tempo_map, project_sec) = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        (tl.bpm, tl.tempo_map.clone(), tl.project_sec)
    };
    let (beats_per_bar, denominator, project_grid_size) = {
        let p = state.project.lock().unwrap_or_else(|e| e.into_inner());
        (
            p.beats_per_bar,
            p.time_signature_denominator,
            p.grid_size.clone(),
        )
    };
    let settings = state.ui_settings_snapshot();
    // 展开地平线：工程末尾再多铺 2s，播放越过工程末尾时响点表不致耗尽。
    let horizon_sec = project_sec + 2.0;
    let segments = build_tempo_segments(
        bpm,
        tempo_map.as_deref(),
        beats_per_bar,
        denominator,
        horizon_sec,
    );
    let step = match metronome_mode_from_settings(&settings) {
        MetronomeMode::Grid => grid_step_beats(&project_grid_size).unwrap_or(1.0),
        MetronomeMode::Beat => 1.0,
        // 0 = 仅小节首（按各段拍号锚步进，见 build_click_schedule）。
        MetronomeMode::Bar => 0.0,
    };
    // Swing 与时间标尺同一来源：仅启用时作用于弱网格线的奇数格。
    let swing = if settings.timeline_snap.swing_enabled {
        settings.timeline_snap.swing_percent as f64
    } else {
        0.0
    };
    let clicks = build_click_schedule(
        &segments,
        step,
        state.audio_engine.sample_rate_hz(),
        horizon_sec,
        swing,
    );
    state
        .audio_engine
        .set_metronome_schedule(std::sync::Arc::new(clicks));
}

/// 设置节拍器（前端节拍器按钮 / 设置菜单 / 音量滑杆滚轮细调）。
///
/// 持久化走通用 `save_ui_settings` 通道（前端 updateMetronome 去抖调度）；
/// 本命令只负责写回设置缓存、把配置即时应用到引擎并按新模式重建响点表
/// （细分模式参与响点展开）。不在此处逐次落盘：滑杆 / 滚轮高频触发下，
/// 每次全量配置写盘（读-合并-写-备份 ≈ 8 次文件操作）既拖慢命令线程，
/// 也会与并发的其他设置保存互相踩踏。
pub(super) fn set_metronome(
    state: State<'_, AppState>,
    enabled: bool,
    gain: f64,
    mode: String,
    accent: bool,
    sound: String,
) -> serde_json::Value {
    let mut settings = (*state.ui_settings_snapshot()).clone();
    settings.metronome_enabled = enabled;
    settings.metronome_gain = gain.clamp(0.0, 1.0);
    settings.metronome_mode = match mode.as_str() {
        "beat" => "beat".to_string(),
        "bar" => "bar".to_string(),
        _ => "grid".to_string(),
    };
    settings.metronome_accent = accent;
    settings.metronome_sound = match sound.as_str() {
        "woodblock" => "woodblock".to_string(),
        "beep" => "beep".to_string(),
        _ => "click".to_string(),
    };
    state.store_ui_settings_cache(&settings);
    sync_metronome_config(&state);
    refresh_metronome_schedule(&state);
    serde_json::json!({"ok": true})
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
        log::warn!("[bg_render] already active, skipping");
        return serde_json::json!({"ok": true, "skipped": true, "reason": "already_active"});
    }
    // 清除可能残留的上一轮取消标志
    BG_RENDER_CANCEL.store(false, Ordering::Release);
    // 清除残留的重启请求（新一轮从干净状态开始；之后的失效会重新置位
    // 并刷新时间戳）。这样新轮次不会被上一轮遗留的请求立即晋升取消。
    BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);
    BG_RENDER_RESTART_REQUESTED_AT_MS.store(0, Ordering::Release);
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
            log::error!(
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

fn start_background_render_inner(
    app: tauri::AppHandle,
    render_generation: u64,
) -> serde_json::Value {
    use std::sync::atomic::Ordering;

    // Clone app before getting state (state borrows from the clone),
    // so the original app can be moved into the thread.
    let app_clone = app.clone();
    let state = app_clone.state::<AppState>();
    // 与 play_original 相同：clone 和版本号必须同锁域读取，避免旧 timeline
    // 配新版本号导致后台渲染覆盖更新的编辑。
    let (timeline, render_timeline_version) = {
        let guard = match state.timeline.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let version = state.timeline_version.load(Ordering::Acquire);
        (guard.clone(), version)
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
        log::warn!(
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

    log::warn!(
        "[bg_render] starting background render: {} clips, engine_sr={}, timeline_version={}",
        total,
        sr,
        render_timeline_version
    );

    // ★ 不在此处清空 pending_rendered_keys。
    // 旧实现每轮渲染开始一刀切清空全部 key，而快照的 rendered_pcm 解析依赖
    // "key → 缓存条目"：清空后任何快照重建都会把**已渲染**的 clip 视为未渲染，
    // 正在播放的传输层因此被重新静音冻结；渲染重启风暴（项目分批加载 /
    // 连续编辑）下形成"播放→静音冻结"的持续闪烁 —— 表现为音频断续、播放
    // 光标近乎原地停留。key 现在在**渲染失效**处按需移除
    //（invalidate_clip_all_caches / invalidate_clip_for_pitch_edit），语义精确，
    // 且跨轮持久，播放连续性不再被渲染重启打断。

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
            log::error!("[bg_render] render thread panicked; resetting flags");
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

/// 当前"播放关注点"（秒）：后台渲染据此为待渲染队列重排序。
///
/// - 引擎正在播放 → 实时传输位置（音频时钟），即用户**正在听 / 即将听到**
///   的位置；
/// - 未播放 → 时间线播放光标（用户当前定位处），即用户**最可能马上播放**
///   的位置。
///
/// 这样无论用户是在播放中等待，还是把光标停在工程中部再触发渲染，紧邻该
/// 位置的 clip 都会被优先渲染。
fn playback_priority_sec(app: &tauri::AppHandle) -> f64 {
    let state = app.state::<AppState>();
    let engine = state.audio_engine.clone();
    if engine.is_playing() {
        // 可听位置 = base + elapsed（与 stop_audio 的"暂停点"同语义）：
        // 时间线播放 base 恒为 0，即时间线绝对位置；文件播放时与 UI 展示的
        // 传输位置一致。只用 elapsed（position_frames）会丢掉文件播放的起点
        // 偏移，渲染优先级的"播放关注点"随之错位。
        let pb = engine.snapshot_state();
        (pb.base_sec + pb.position_sec).max(0.0)
    } else {
        match state.timeline.lock() {
            Ok(tl) => tl.playhead_sec.max(0.0),
            Err(poisoned) => poisoned.into_inner().playhead_sec.max(0.0),
        }
    }
}

/// 渲染优先级键：`(是否位于播放关注点之后, clip 起点)`，字典序升序。
///
/// - 组 0：**覆盖**关注点（正在听的 clip）或位于其之后（即将听到）；组内按
///   时间线起点升序 —— 覆盖当前位置的 clip 起点最小，因此最优先；
/// - 组 1：位于关注点之前的 clip（用户已听过 / 光标已越过），最后再渲染。
fn render_priority_key(start_sec: f64, length_sec: f64, anchor_sec: f64) -> (u8, f64) {
    let start = start_sec.max(0.0);
    let end = start + length_sec.max(0.0);
    let relevant = end > anchor_sec;
    (if relevant { 0 } else { 1 }, start)
}

/// 从**尚未开始**的条目中按实时播放位置择优取下一个待渲染 clip 的下标；
/// 无剩余条目时返回 `None`。
///
/// 只在 clip 边界（上一次渲染已完成 / 命中缓存）调用，因此正在进行的渲染
/// 永远不会被打断 —— 重排序只影响"接下来渲染谁"。
fn select_next_render_index(
    clips_to_render: &[ClipRenderInfo],
    done: &[bool],
    app: &tauri::AppHandle,
) -> Option<usize> {
    let anchor_sec = playback_priority_sec(app);
    clips_to_render
        .iter()
        .enumerate()
        .filter(|(index, _)| !done.get(*index).copied().unwrap_or(true))
        .min_by(|(_, a), (_, b)| {
            let ka = render_priority_key(a.clip.start_sec, a.clip.length_sec, anchor_sec);
            let kb = render_priority_key(b.clip.start_sec, b.clip.length_sec, anchor_sec);
            match ka.0.cmp(&kb.0) {
                std::cmp::Ordering::Equal => ka.1.total_cmp(&kb.1),
                other => other,
            }
        })
        .map(|(index, _)| index)
}

/// 后台渲染单轮主循环（在渲染线程上执行；由调用方负责 panic 隔离）。
///
/// 本轮渲染**不按固定顺序遍历**：每个 clip 边界都按实时播放位置动态择优
/// （见 [`select_next_render_index`]），使光标位置及其之后的 clip 优先渲染。
/// 已在渲染中的 clip 不会被打断 —— 重排序只作用于尚未开始的条目。
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

        // 归零推理耗时画像：分母是本轮的，分子也必须只含本轮 —— 否则被取消的
        // 上一轮残留会让 f_infer 虚高（见 render_profile 模块说明）。
        crate::render_profile::reset();

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
        // 后台预渲染继续跟踪全局 `BG_RENDER_CANCEL`：时间线编辑会置位它来
        // 中断本轮并触发重启（见 audio_engine/engine.rs 的失效处理）。
        let cancel_token = crate::commands::render_cancel::RenderCancelToken::background();

        // 待渲染队列的完成情况：按"实时播放位置"动态择优取下一个 clip
        // （见 `select_next_render_index`），已开始的渲染不会被打断 ——
        // 重新排序只作用于**尚未开始**的条目。
        let mut done = vec![false; clips_to_render.len()];
        loop {
            let next_index = select_next_render_index(clips_to_render, &done, &app);
            let Some(index) = next_index else {
                break;
            };
            done[index] = true;
            let clip_render_info = &clips_to_render[index];
            let clip_started_at = std::time::Instant::now();
            log::warn!(
                "[bg_render] clip {}/{} begin clip_id={} start_sec={:.3}",
                rendered_count + 1,
                total,
                clip_render_info.clip.id,
                clip_render_info.clip.start_sec
            );

            // 方案 A：重启请求静默窗口 —— 请求挂起超过窗口才升级为实际取消。
            // 失效风暴（工程加载 / 连续编辑）期间时间戳被持续刷新、当前轮
            // 不被打断；单次编辑最多延迟一个窗口重启。
            if BG_RENDER_RESTART_NEEDED.load(Ordering::Relaxed)
                && bg_render_restart_request_is_stale(
                    BG_RENDER_RESTART_REQUESTED_AT_MS.load(Ordering::Relaxed),
                    now_millis(),
                )
            {
                // 走 request_global_cancel 而非直接置位：置位只有这一个入口，
                // 便于审计"置位与清除成对"的生命周期契约（见 BG_RENDER_CANCEL）。
                crate::commands::render_cancel::request_global_cancel();
            }
            // 检查取消标志（用户在渲染中重新编辑参数时会设置）
            if BG_RENDER_CANCEL.load(Ordering::Relaxed) {
                log::warn!(
                    "[bg_render] cancel flag detected at clip {}/{}",
                    rendered_count,
                    total
                );
                cancelled = true;
                break;
            }
            // 每隔 32 个 clip 检查时间线版本是否已变更
            if rendered_count % 32 == 0 {
                let state = app.state::<AppState>();
                let changed =
                    state.timeline_version.load(Ordering::Acquire) != render_timeline_version;
                drop(state);
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
                    log::warn!(
                        "[bg_render][cache] HIT clip_id={} hash={:#018x}",
                        clip_render_info.clip.id,
                        clip_render_info.cache_key.param_hash
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
                    log::warn!(
                        "[bg_render][cache] MISS clip_id={} hash={:#018x}",
                        clip_render_info.clip.id,
                        clip_render_info.cache_key.param_hash
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
                match render_single_clip(
                    &timeline,
                    &clip_render_info.clip,
                    clip_render_info.sr,
                    &cancel_token,
                ) {
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
                        // 每个 clip 处理完毕后由循环末尾统一推送刷新
                        //（refresh_rendered_snapshot），等待中的传输层据此
                        // 恢复播放 —— 推送模型，无轮询、无版本号比对。
                    }
                    Err(e) => {
                        if e == BG_RENDER_CANCELLED_ERR {
                            cancelled = true;
                            break;
                        }
                        render_elapsed += render_started_at.elapsed();
                        log::error!(
                            "[bg_render] clip render failed: clip_id={} err={}",
                            clip_render_info.clip.id,
                            e
                        );
                        render_failed_count += 1;
                        // 该 clip 会一直保持未渲染（播放时静音等待），而前端此前
                        // 完全消费不到 `ClipRenderingState::Failed`，用户只看到
                        // "这段没声音"。这里补一条用户可见告警（见 P0-5）。
                        crate::render_warning::warn(
                            crate::render_warning::KIND_CLIP_RENDER_FAILED,
                            "A clip failed to render and will stay silent",
                            Some(&format!("clip_id={} err={e}", clip_render_info.clip.id)),
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
                        log::error!(
                            "[bg_render] tension render failed: clip_id={} err={}",
                            clip_render_info.clip.id,
                            e
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
            log::warn!(
                "[bg_render] clip {}/{} done clip_id={} elapsed_ms={}",
                rendered_count,
                total,
                clip_render_info.clip.id,
                clip_started_at.elapsed().as_millis()
            );

            // 本 clip 处理完毕（命中缓存或新渲染入库）→ 推送刷新引擎快照。
            // 这是"原地等待渲染"解除的唯一入口（见
            // `AudioEngine::refresh_rendered_snapshot`）：产出者主动发布，
            // 无需轮询 / 版本号比对 / 等待状态上报，等待中的传输层会在下一个
            // 音频块自动重新判定就绪性并继续播放。
            {
                let engine = app.state::<AppState>().audio_engine.clone();
                engine.refresh_rendered_snapshot();
            }
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
                log::warn!(
                    "[bg_render] cancelled by new edit (rendered {}/{}), restarting with fresh params...",
                    rendered_count, total
                );
                // 直接启动新一轮渲染，不发送中间完成事件，对用户无感
                start_background_render(app.clone());
                return;
            }

            // 真正取消（时间线版本变更等）：发出完成事件
            if cache_log {
                log::info!(
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

        // 定格本轮画像：分母用与下方 complete 日志相同的墙钟时间，保证
        // `f_infer` 与使用者从日志里手算的结果一致（见 §7 阻塞级问题 1）。
        let pass_elapsed = started_at.elapsed();
        let profile = crate::render_profile::finish_pass(pass_elapsed);

        if cache_log {
            log::warn!(
                "[bg_render][cache] DONE total={} hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} render_ms={:.2} tension_ms={:.2} inference_ms={:.2} inference_runs={} f_infer={:.3} total_ms={:.2}",
                total,
                cache_hit_count,
                cache_miss_count,
                render_success_count,
                render_failed_count,
                cache_probe_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
                tension_elapsed.as_secs_f64() * 1000.0,
                profile.inference_ms,
                profile.inference_runs,
                profile.inference_fraction,
                profile.total_ms
            );
        }
        // `f_infer` 无需环境变量即可见：它是 P1-2 唯一决定变量，默认输出能省掉
        // 一次"要开日志复现才拿得到"的往返（见 render_profile 模块说明）。
        log::warn!(
            "[bg_render] complete: {} clips, {} hit, {} miss, {} ok, {} fail in {:.2}s \
             (inference {:.2}ms x{}, f_infer={:.3}; chunked_clips={} multi_chunk_clips={} chunks_total={} multi_chunk_fraction={:.3})",
            total,
            cache_hit_count,
            cache_miss_count,
            render_success_count,
            render_failed_count,
            pass_elapsed.as_secs_f64(),
            profile.inference_ms,
            profile.inference_runs,
            profile.inference_fraction,
            profile.chunked_path_clips,
            profile.clips_with_multiple_chunks,
            profile.chunks_total,
            profile.multi_chunk_fraction
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
            log::warn!(
                "[bg_render] follow-up pass needed: {} clip(s) were not pitch-ready, {} clip(s) newly rendered in this pass",
                skipped_not_ready, render_success_count
            );
            start_background_render(app.clone());
            return;
        }
        if skipped_not_ready > 0 && render_success_count == 0 {
            log::warn!(
                "[bg_render] {} clip(s) still not pitch-ready but this pass made no new progress; waiting for pitch analysis completion instead of re-rendering",
                skipped_not_ready
            );
        }

        // 若完成时恰好有新编辑触发的重启请求，立即启动新一轮渲染
        if BG_RENDER_RESTART_NEEDED.swap(false, Ordering::AcqRel) {
            log::warn!("[bg_render] completed but restart was requested during finalization, starting new render");
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
/// 播放命令的按需渲染：**无论后台预渲染开关如何**，确保一轮渲染正在运行
/// （本次播放需要待渲染 Clip 的结果）。
///
/// 开关只决定"编辑是否自动触发渲染"；播放触发的渲染是传输层"原地等待 +
/// 就绪自动播放"契约的必要条件。已在运行的 pass 无需动作 —— 其待渲染
/// 列表本就覆盖当前所有待渲染 Clip；其后的缓存失效会经
/// `handle_update_timeline` 触发合流重启。
pub(crate) fn ensure_render_pass_running(app: &tauri::AppHandle) {
    use std::sync::atomic::Ordering;
    if BG_RENDER_ACTIVE.load(Ordering::Relaxed) {
        return;
    }
    // A fresh render request supersedes any stale restart marker.
    BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);
    let app = app.clone();
    std::thread::spawn(move || {
        let _ = start_background_render(app);
    });
}

/// 调用方会自我死锁 —— 命令线程永久阻塞，整个应用进入“未响应”。
/// 因此这里把真正的启动工作转移到新线程：本函数只做原子的状态检查与
/// 标记（无锁），渲染启动线程会等待时间线锁自然释放后再开始收集。
pub(crate) fn request_background_render(app: &tauri::AppHandle) -> serde_json::Value {
    use std::sync::atomic::Ordering;

    if !AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed) {
        // 后台预渲染未启用时，这里是不再有人清理全局取消标志的最后一道防线。
        // 置位它的 `cancel_background_render` 在打开/新建工程时必被调用，
        // 若此处不复位，标志会一直保持为 true（见该标志的生命周期注释）。
        if !BG_RENDER_ACTIVE.load(Ordering::Relaxed) {
            BG_RENDER_CANCEL.store(false, Ordering::Release);
        }
        return serde_json::json!({"ok": true, "skipped": true, "reason": "disabled"});
    }

    if BG_RENDER_ACTIVE.load(Ordering::Relaxed) {
        // 方案 A：不再立即取消在途渲染（失效风暴会被反复打断，每轮只能
        // 推进两三个 clip）。仅合流地记录重启请求，由渲染循环在 clip 边界
        // 按静默窗口决定是否升级为取消。
        request_bg_render_restart();
        debug_eprintln!(
            "[bg_render] caches invalidated while render active; restart requested (coalesced)"
        );
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

/// 取消当前正在运行的后台预渲染（如果有），并通知前台预渲染退出。
///
/// 本函数由 `new_project` / `open_project` **无条件**调用（前端对应 thunk 也会
/// 各调一次），因此"当时没有后台渲染在跑"是常态而非异常，必须按 `was_active`
/// 分流处理全局取消标志，详见下方注释。
pub(super) fn cancel_background_render(app: Option<&tauri::AppHandle>) -> serde_json::Value {
    use std::sync::atomic::Ordering;
    let was_active = BG_RENDER_ACTIVE.swap(false, Ordering::AcqRel);
    if was_active {
        // 让正在运行的渲染循环在下一个 clip 边界立刻退出，而不是继续把旧工程
        // 渲染完。走 request_global_cancel 以保持单一置位入口（见 render_cancel.rs）。
        crate::commands::render_cancel::request_global_cancel();
    } else {
        // ★ 没有任何后台渲染在跑时，绝不能留下取消信号。
        //
        // `BG_RENDER_CANCEL` 只在"后台预渲染"的启动（`start_background_render`
        // 开头）与收尾路径上被清除。而 `request_background_render` 在后台预渲染
        // 未启用时会走 disabled 分支直接返回、不清理标志。
        // 于是"打开工程（内部必调本函数）且后台预渲染关闭"会让标志永久为 true，
        // 之后每次前台 `play_original` 预渲染都在解码后的第一个检查点被中止，
        // 表现为"长音频渲染不出来、播放降级为原声"。
        // 因此这里必须主动复位，而不是简单地不置位。
        BG_RENDER_CANCEL.store(false, Ordering::Release);
    }
    // 清除重启标记，避免取消后被错误地按旧渲染状态自动重启。
    BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);
    BG_RENDER_PITCH_PENDING.store(false, Ordering::Release);
    // 递增代数，使旧渲染线程的收尾清理不再影响新的一轮渲染。
    BG_RENDER_GENERATION.fetch_add(1, Ordering::AcqRel);
    log::warn!("[bg_render] cancel requested, was_active={was_active}");
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
    use super::{
        bg_render_restart_request_is_stale, render_priority_key, should_follow_up_render,
        BG_RENDER_RESTART_QUIET_WINDOW_MS,
    };

    /// 渲染优先级：覆盖播放关注点的 clip 最优先，其后是位于关注点之后的
    /// clip（按时间线顺序），最后是关注点之前的 clip。前后台渲染共用此逻辑。
    #[test]
    fn render_priority_favours_clips_at_and_after_the_play_position() {
        // 播放关注点在 30s。
        let anchor = 30.0;

        // 覆盖关注点（20s~40s）。
        let covering = render_priority_key(20.0, 20.0, anchor);
        // 位于关注点之后（35s~45s）。
        let after = render_priority_key(35.0, 10.0, anchor);
        // 位于关注点之前（0s~10s）。
        let before = render_priority_key(0.0, 10.0, anchor);

        assert_eq!(
            covering.0, 0,
            "a clip covering the play position is top priority"
        );
        assert_eq!(after.0, 0, "clips after the play position are relevant");
        assert_eq!(before.0, 1, "clips before the play position are deferred");

        assert!(
            covering < after,
            "the clip being listened to precedes later clips"
        );
        assert!(
            after < before,
            "relevant clips precede already-passed clips"
        );

        // 关注点之后的两个 clip：时间线靠前者优先。
        let near = render_priority_key(31.0, 5.0, anchor);
        let far = render_priority_key(60.0, 5.0, anchor);
        assert!(near < far, "among upcoming clips the nearest one wins");

        // 边界：clip 恰好在关注点处结束（10s~30s）→ 已听过，延后。
        assert_eq!(
            render_priority_key(10.0, 20.0, anchor).0,
            1,
            "a clip ending exactly at the play position is not relevant"
        );
    }

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

    #[test]
    fn restart_request_not_stale_within_quiet_window() {
        // 请求刚发生（差值 < 窗口）→ 未冷却，不晋升取消。
        assert!(!bg_render_restart_request_is_stale(1_000, 1_000));
        assert!(!bg_render_restart_request_is_stale(
            1_000,
            1_000 + BG_RENDER_RESTART_QUIET_WINDOW_MS - 1
        ));
    }

    #[test]
    fn restart_request_stale_after_quiet_window() {
        // 挂起达到 / 超过窗口 → 已冷却，渲染循环应晋升为取消。
        assert!(bg_render_restart_request_is_stale(
            1_000,
            1_000 + BG_RENDER_RESTART_QUIET_WINDOW_MS
        ));
        assert!(bg_render_restart_request_is_stale(
            1_000,
            1_000 + BG_RENDER_RESTART_QUIET_WINDOW_MS * 10
        ));
    }

    #[test]
    fn restart_request_without_timestamp_defaults_to_stale() {
        // 时间戳为 0（无记录）时按"请求发生在进程启动时刻"计算：
        // 进程运行超过窗口 → 视为已冷却，兜底保持旧的立即取消语义。
        assert!(bg_render_restart_request_is_stale(
            0,
            BG_RENDER_RESTART_QUIET_WINDOW_MS
        ));
        // 进程刚启动、仍在首个窗口内 → 未冷却（窗口语义一致）。
        assert!(!bg_render_restart_request_is_stale(0, 50));
    }

    #[test]
    fn restart_request_clock_regression_never_stale() {
        // 时钟回退（now < last）→ 饱和减法得 0 → 未冷却，不会误触发取消。
        assert!(!bg_render_restart_request_is_stale(5_000, 4_999));
    }
}
