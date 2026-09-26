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
///   - 置位：**只能**通过 `commands::render_cancel::request_global_cancel()`
///     （它会同时推进纪元）。调用点有 `cancel_background_render`
///     （仅当确实有后台渲染在跑），以及后台渲染循环的重启静默窗口晋升
///     （`request_bg_render_restart` 记录的重启请求挂起超过窗口后，由渲染
///     线程在 clip 边界调用）。请**不要**直接 `store(true, ..)`，那样纪元
///     不推进，新轮次会把它当历史残留忽略。
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

/// 距渲染请求的去抖窗口结束还需等待多少毫秒（0 = 可以启动）。
///
/// 抽成纯函数是为了可测：调度线程本身无法在单测里稳定复现。
fn render_request_debounce_remaining_ms(last_requested_at_ms: u64, now_ms: u64) -> u64 {
    BG_RENDER_DEBOUNCE_MS.saturating_sub(now_ms.saturating_sub(last_requested_at_ms))
}

/// 请求重启后台渲染（合流版）。
///
/// 不立即取消在途渲染：仅置位重启标志并刷新请求时间戳。渲染循环在 clip
/// 边界发现请求挂起超过静默窗口后，才通过
/// `render_cancel::request_global_cancel()`（推进纪元）升级为实际取消，
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

/// 渲染请求去抖窗口（毫秒）。
///
/// 背景：音高分析完成、缓存失效、编辑提交都会请求渲染，而工程加载期间这类事件
/// 以几十毫秒一次的频率到来 —— 实测一次打开工程的前 1.5 s 内启动了 **61 轮**
/// pass，其中 53 轮只处理 6 个 clip 且毫无进展。每轮都要全量收集待渲染 Clip
/// （逐个计算渲染键，含曲线切片哈希）、扩容四个缓存并 spawn 线程，空转成本与
/// 真实渲染同阶。
///
/// 去抖把窗口内的多个请求合流成一次启动。窗口结束后到达的新请求会开启新的
/// 调度，因此不存在"请求被吞掉"的情况。
pub(crate) const BG_RENDER_DEBOUNCE_MS: u64 = 200;

/// 最近一次渲染请求的时间戳（进程启动相对毫秒）。
///
/// 与 [`BG_RENDER_RESTART_REQUESTED_AT_MS`] 是**两件事**：后者描述"在途 pass
/// 需要重启"（由渲染循环在 clip 边界消费），前者描述"还没有 pass 在跑，需要
/// 起一轮"（由调度线程消费）。混用会让两个静默窗口互相干扰。
static BG_RENDER_REQUESTED_AT_MS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// 是否已有调度线程在等待去抖窗口（用于合流同一窗口内的重复请求）。
static BG_RENDER_SCHEDULER_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

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
    // 【渲染输入稳定性卫兵（按根轨道）】所在根轨道的 `pitch_orig` 组装尚未
    // 收敛时跳过：组装在 clip 音高分析落地过程中会多次改写曲线（未就绪 span
    // 以零填充占位），而它是渲染键的输入 —— 未收敛时渲染的条目在收敛后必然
    // 键失效（播放时 miss 重渲染，同一 clip 渲两遍）。收敛判定见
    // root_pitch_assembly_pending；由 collect 侧按根记忆，避免逐 clip 重算。
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

/// 该根轨道的 `pitch_orig` 组装是否仍未收敛（渲染键还会漂移）。
///
/// 【为什么按根轨道而不是全局】clip 音高分析落地时，`maybe_schedule_pitch_orig`
/// 会用各 clip 的缓存逐次重组装整条根曲线（未就绪 span 以零填充占位），全部
/// 落地后的"全量命中"组装才会置位 `pitch_orig_key`（见 `maybe_schedule_pitch_orig`
/// 的分支说明）—— 它因此是"该根曲线已到终值、渲染键不再漂移"的权威信号。
/// 按根判定让**已收敛根轨道**的 clip 立即渲染（例如全缓存命中的工程打开即渲），
/// 不为其他根轨道的分析整体推迟；未收敛根的 clip 由完成回调链条逐根解锁。
///
/// 前置条件与 `build_pitch_job` 的"是否需要组装"判定保持一致（不含其昂贵的
/// mix timeline 构建）——该根根本没有组装任务时不视为 pending，否则会永久
/// 卡住渲染。
fn root_pitch_assembly_pending(
    timeline: &crate::state::TimelineState,
    root_track_id: &str,
) -> bool {
    let Some(track) = timeline.tracks.iter().find(|t| t.id == root_track_id) else {
        return false;
    };
    let has_active_midi_clip = timeline.clips.iter().any(|c| {
        timeline.resolve_root_track_id(&c.track_id).as_deref() == Some(root_track_id)
            && !c.muted
            && c.midi_note_data.is_some()
    });
    let currently_has_adjustment = timeline
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.has_pitch_adjustment_active)
        .unwrap_or(false);
    if !track.compose_enabled && !has_active_midi_clip && !currently_has_adjustment {
        return false;
    }
    if matches!(
        track.pitch_analysis_algo,
        crate::state::PitchAnalysisAlgo::None
    ) {
        return false;
    }
    timeline
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.pitch_orig_key.is_none())
        .unwrap_or(false)
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
        //
        // ── 起播等待期垫音抑制 ────────────────────────────────────────────────
        // 本次起播仍需渲染的 clip 整体登记"垫音抑制"：其当前渲染就绪前，快照
        // 不得回退旧版本渲染垫音 —— 起播行为与首渲染、与后台预渲染开关完全
        // 一致（就绪即播，未就绪诚实冻结），绝不让用户先听到上一版参数的
        // 结果再中途切换。条目在各 clip 当前渲染命中时由 build_snapshot 逐条
        // 解除，因此播放中段的参数编辑不受影响：那时的 miss 对应"上一版渲染
        // == 正在播放的内容"，垫音即零中断切换。必须在 UpdateTimeline 入队前
        // 登记，首个起播快照才会生效。
        crate::synth_clip_cache::set_pad_suppressed_clips(
            clips_needing_render.iter().map(|info| info.clip.id.clone()),
        );
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

    // 【持久化缓存读回】张力变体：内存未命中时回退磁盘（渲染线程内）。
    if crate::render_cache::enabled() {
        if let Some(loaded) = crate::render_cache::load_tension(&cache_key, out_rate) {
            let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(cache_key.clone(), loaded);
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
    // 【持久化】张力变体异步落盘。
    crate::render_cache::store_tension(&cache_key, &entry);
    let mut cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    cache.insert(cache_key.clone(), entry);
    Ok((Some(cache_key), true))
}

/// Clip 播放速率（非法值按 1.0 处理，与实时引擎口径一致）。
fn clip_playback_rate(clip: &crate::state::Clip) -> f64 {
    let rate = clip.playback_rate as f64;
    if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        1.0
    }
}

/// 构造整 Clip 渲染哈希输入。
///
/// ★ 所有需要计算渲染缓存键的位置（收集待渲染、气声噪声键、快照回退）都必须
/// 经由本函数：任何一处参数口径漂移都会让"写入的键"与"查询的键"不一致 ——
/// 轻则缓存永久 miss，重则跨参数误命中。
fn build_rendered_hash_input<'a>(
    clip: &'a crate::state::Clip,
    entry: &'a crate::state::TrackParamsState,
    renderer_id: &'a str,
    sr: u32,
    input_pitch_curve: Option<&'a [f32]>,
    compose_enabled: bool,
    scale_signature: &'a str,
) -> crate::synth_clip_cache::RenderedClipHashInput<'a> {
    let start_frame = (clip.start_sec.max(0.0) * sr as f64).round() as u64;
    let end_frame =
        start_frame + (clip.length_sec.max(0.0) * sr as f64).round().max(1.0) as u64;

    crate::synth_clip_cache::RenderedClipHashInput {
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

/// clip 的渲染材料：根轨道的参数与轨道本身。
struct ClipRenderMaterial<'a> {
    entry: &'a crate::state::TrackParamsState,
    track: &'a crate::state::Track,
}

/// 解析 clip 的渲染材料，并回答"这个 clip 当前是否需要渲染"。
///
/// ★ 这是该判定的**唯一实现**：收集待渲染（热路径）与 miss 归因诊断都必须经由
/// 它。两处各写一份必然漂移，而漂移的后果是"写入的键"与"查询的键"不一致 ——
/// 轻则永久 miss，重则跨参数误命中（见 `synth_clip_cache` 的模块契约）。
///
/// `find_track` 由调用方注入：热路径传预构建的 O(1) 查找表，诊断路径传线性查找
/// （每次 miss 至多一次，轨道数是常数级）。
fn resolve_render_material<'a>(
    timeline: &'a crate::state::TimelineState,
    clip: &crate::state::Clip,
    find_track: impl Fn(&str) -> Option<&'a crate::state::Track>,
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

/// 组装单个 clip 的渲染键输入（与收集、快照回退共用同一口径）。
fn rendered_hash_input_for_clip<'a>(
    timeline: &'a crate::state::TimelineState,
    clip: &'a crate::state::Clip,
    sr: u32,
    scale_signature: &'a str,
) -> Option<crate::synth_clip_cache::RenderedClipHashInput<'a>> {
    let material =
        resolve_render_material(timeline, clip, |id| timeline.tracks.iter().find(|t| t.id == id))?;
    let kind = crate::state::SynthPipelineKind::from_track_algo(&material.track.pitch_analysis_algo);
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

/// miss 归因：逐个排除最易漂移的输入后重算哈希，并探测磁盘上是否存在该变体。
///
/// 渲染键是不可逆的摘要，单看两个哈希无法知道"哪个输入变了"。但排除某项后重算
/// 若反而能在磁盘上找到条目，就唯一地指向该输入在两个会话之间发生了漂移 ——
/// 典型来源是音高分析在 GPU / CPU EP 之间切换导致 `pitch_orig` 出现 ULP 级差异、
/// 源文件因复制 / 同步而 mtime 变动。这是"整库静默失效"唯一可行的定位手段。
///
/// 仅在 `HIFISHIFTER_RENDER_CACHE_LOG=1` 下调用：每个 miss 要多算若干次哈希
/// （含曲线切片），不能进常规路径。
fn log_render_key_drift(timeline: &crate::state::TimelineState, clip: &crate::state::Clip, sr: u32) {
    use crate::synth_clip_cache::{compute_rendered_clip_hash_excluding, HashExclusions};

    let scale_signature = timeline.render_scale_signature();
    let Some(input) = rendered_hash_input_for_clip(timeline, clip, sr, &scale_signature) else {
        return;
    };
    for (label, exclusions) in HashExclusions::DIAGNOSTIC_CANDIDATES {
        let variant = compute_rendered_clip_hash_excluding(&input, exclusions);
        if crate::render_cache::contains_rendered(variant) {
            log::warn!(
                "[bg_render][cache] MISS attribution clip_id={} would_hit_if_excluded={} variant_hash={:#018x}",
                clip.id,
                label,
                variant
            );
        }
    }
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
    // 生效音阶签名在整个收集过程中不变，只需算一次（它遍历 Tempo Map）。
    let scale_signature = timeline.render_scale_signature();

    for clip in &timeline.clips {
        let Some(material) =
            resolve_render_material(timeline, clip, |id| tracks_by_id.get(id).copied())
        else {
            continue;
        };
        let kind = crate::state::SynthPipelineKind::from_track_algo(&material.track.pitch_analysis_algo);
        let renderer_id = crate::renderer::get_renderer(kind).id();

        // 渲染参数哈希：与渲染线程、快照回退共用同一份输入口径。
        let hash_input = build_rendered_hash_input(
            clip,
            material.entry,
            renderer_id,
            sr,
            None,
            material.track.compose_enabled,
            scale_signature.as_str(),
        );
        let param_hash = crate::synth_clip_cache::compute_rendered_clip_hash(&hash_input);
        let cache_key = crate::synth_clip_cache::RenderedClipCacheKey {
            clip_id: clip.id.clone(),
            param_hash,
        };

        if debug {
            log::warn!(
                "[collect_clips_needing_render] clip_id={} sr={} start_frame={} end_frame={} hash={:#018x}",
                clip.id, sr, hash_input.start_frame, hash_input.end_frame, param_hash
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

    // 1. 解码源文件
    let (in_rate, in_channels, pcm) =
        crate::audio_utils::decode_audio_f32_interleaved(std::path::Path::new(source_path))?;
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
        crate::mixdown::linear_resample_interleaved(&segment, in_channels_usize, in_rate, out_rate);

    // Loop 模式的倒放方向已由回绕索引体现，不再整体反转。
    if !loop_mode && clip.reversed {
        crate::mixdown::reverse_interleaved_frames(&mut segment, in_channels_usize);
    }

    // 4. 声道条件化（take 级 channel_mode 的唯一语义实现，与 mixdown 一致）：
    //    mono 源复制为双声道、stereo 源按模式取平面/交换/下混。
    let segment = crate::channel_mode::condition_take_channels(
        &segment,
        in_channels,
        clip.take_channel_mode(),
    );
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
            clip.channel_mode,
            // 本域输入已在上方做过声道条件化，与实时域（原始 stereo +
            // 混音时施加模式）必须用 preconditioned 判别隔离。
            true,
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

    let clip_root = timeline.resolve_root_track_id(&clip.track_id);
    let root_params = clip_root
        .as_ref()
        .and_then(|root| timeline.params_by_root_track.get(root));
    let effective_extra_params = clip
        .extra_params
        .as_ref()
        .or_else(|| root_params.map(|entry| &entry.extra_params));
    // 气声（HNSEP 分离 + noise stem）只由 NSF-HiGAN 渲染器消费：快照混音
    // （renderer_id == "nsf_hifigan_onnx" 才挂 breath_curve）与
    // track_requests_extra_processing 均按渲染器种类门控。轨道切换算法
    // （如 HiFiGAN → WORLD）不会清空 extra_params，这里若不门控，残留的
    // breath_enabled 会让 WORLD/vslib 渲染白跑一次 HNSEP 推理 —— 其 noise
    // stem 在混音侧永远不会被使用（World 声码器本身也不消费气声参数）。
    let breath_capable = clip_root
        .as_ref()
        .and_then(|root| timeline.tracks.iter().find(|track| &track.id == root))
        .map(|track| {
            matches!(
                crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo),
                crate::state::SynthPipelineKind::NsfHifiganOnnx
            )
        })
        .unwrap_or(false);
    let breath_enabled = breath_capable
        && effective_extra_params
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
                let scale_signature = timeline.render_scale_signature();
                let hash_input = build_rendered_hash_input(
                    clip,
                    entry,
                    renderer_id,
                    out_rate,
                    None,
                    track.compose_enabled,
                    scale_signature.as_str(),
                );
                let param_hash = crate::synth_clip_cache::compute_breath_noise_hash(&hash_input);
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
        {
            let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(entry) = cache.get(key) {
                return Some(entry.noise_stereo.clone());
            }
        }
        // 【持久化缓存读回】独立噪声 stem：内存未命中时回退磁盘（渲染线程内）。
        // 命中即回填内存缓存，省掉一次 HNSEP 分离推理。
        if !crate::render_cache::enabled() {
            return None;
        }
        let loaded = crate::render_cache::load_noise(key, out_rate)?;
        let noise = loaded.noise_stereo.clone();
        {
            let mut cache = crate::synth_clip_cache::global_breath_noise_cache()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            cache.insert(key.clone(), loaded);
        }
        Some(noise)
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

    // Step 1+2: 按有效声道数逐声道分离并预填 HNSEP 缓存。
    // - 等效单声道：对 (L+R)/2 下混分离一次，缓存键 channel 0 —— 与既有
    //   1-pass 优化完全一致（后续 render_variant 的链内分离命中缓存）。
    // - 真立体声：L/R 平面各自分离并预填各自声道位的缓存键（链内逐声道
    //   命中），噪声 stem 取两声道 noise 交错 —— 保证气声层也是真立体声。
    let fanout_channels = crate::channel_mode::effective_channels(
        clip.source_channels.unwrap_or(1).max(1),
        clip.take_channel_mode(),
    ) as usize;
    let noise_planes: Vec<std::sync::Arc<Vec<f32>>> = {
        if cancel.is_cancelled() {
            return Err(BG_RENDER_CANCELLED_ERR.to_string());
        }
        // HNSEP 分离失败（模型缺失/推理错误）时降级为非 breath 渲染：外层已因
        // 气声跳过外部拉伸，硬错误会让整条 clip 无声等待，比"没有气声"严重得多。
        log::warn!(
            "[render] stage=hnsep_begin clip_id={} channels={fanout_channels} elapsed_ms={}",
            clip.id,
            stage_started.elapsed().as_millis()
        );
        let extract_plane = |plane: usize| -> Vec<f32> {
            segment
                .chunks_exact(2)
                .map(|ch| ch[plane])
                .collect::<Vec<f32>>()
        };
        let mixdown_mono: Vec<f32> = segment
            .chunks_exact(2)
            .map(|ch| (ch[0] + ch[1]) * 0.5f32)
            .collect();
        let channels_mono: Vec<Vec<f32>> = if fanout_channels <= 1 {
            vec![mixdown_mono]
        } else {
            vec![extract_plane(0), extract_plane(1)]
        };
        let mut planes = Vec::with_capacity(channels_mono.len());
        for (ch_idx, ch_mono) in channels_mono.iter().enumerate() {
            match crate::hnsep_onnx::infer_noise_mono(&clip.id, ch_mono, out_rate, ch_idx as u16, clip.source_file_fingerprint) {
                Ok(noise) => planes.push(noise),
                Err(e) => {
                    log::warn!(
                        "render_single_clip: HNSEP failed for clip_id={} channel={ch_idx}, falling back to non-breath render: {e}",
                        clip.id
                    );
                    return Ok(RenderedClipOutput {
                        rendered_stereo: render_variant(clip)?,
                        breath_noise_stereo: None,
                    });
                }
            }
        }
        planes
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

    // Step 4: Convert noise stem(s) to stereo, matching harmonic_only length.
    // 等效单声道 → 对齐后复制双声道；真立体声 → 两声道分别对齐后交错。
    let out_len = harmonic_only.len();
    let out_frames = out_len / 2;
    let noise_stereo: Vec<f32> = {
        // 时间拉伸若由处理器内部完成（mel 域），谐波输出是**时间轴**长度，
        // 而 HNSEP 的噪声 stem 仍是**源速率**长度。必须对齐后再转立体声；
        // 对齐必须用与谐波一致的拉伸算法 —— 线性重采样会把气声的谱包络
        // 按 1/rate 缩放（慢放变闷、快放混叠），听感即"气声没有被正确拉伸"。
        let stretch_algo = crate::time_stretch::resolved_external_stretch_algorithm();
        let aligned: Vec<Vec<f32>> = noise_planes
            .iter()
            .map(|plane| {
                crate::renderer::chain::align_noise_stem_to_len(
                    plane.as_slice(),
                    out_rate,
                    out_frames,
                    stretch_algo,
                )
            })
            .collect();
        let mut stereo = Vec::with_capacity(out_len);
        if aligned.len() <= 1 {
            // Duplicate each mono sample to L/R channels
            if let Some(aligned_mono) = aligned.first() {
                for &s in aligned_mono.iter().take(out_frames) {
                    stereo.push(s);
                    stereo.push(s);
                }
            }
        } else {
            for f in 0..out_frames {
                stereo.push(aligned[0].get(f).copied().unwrap_or(0.0));
                stereo.push(aligned[1].get(f).copied().unwrap_or(0.0));
            }
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
        // 【持久化】噪声 stem 异步落盘：formant 变化时可跨会话复用，省一次 HNSEP。
        crate::render_cache::store_noise(&key, &entry);
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
    // 渲染输入稳定性卫兵（按根记忆收敛判定，见 root_pitch_assembly_pending）：
    // 未收敛根的 clip 跳过并计入 pending，由完成回调链条在收敛后补触发渲染。
    let mut root_settled: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    clips_to_render.retain(|info| {
        let root_key = timeline
            .resolve_root_track_id(&info.clip.track_id)
            .unwrap_or_default();
        let settled = *root_settled
            .entry(root_key.clone())
            .or_insert_with(|| !root_pitch_assembly_pending(&timeline, &root_key));
        settled && is_clip_pitch_analysis_ready(&timeline, &info.clip)
    });
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

    // 进度追踪：clip 级进度由渲染循环推进（对所有处理器一视同仁）；HiFiGAN
    // 推理 chunk / WORLD 合成块经 report_clip_progress 提供 clip 内细化。
    crate::renderer::progress::reset(total);

    let app_for_progress = app.clone();
    let progress_generation = render_generation;
    crate::renderer::progress::set_callback(Some(Box::new(move |progress: f64| {
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
        // 落盘准入计数基线：本轮结束时取差值，得到"这一轮里有多少产物通过
        // 准入、多少被拒以及为什么"。后台渲染会反复重启，用全局累计值当
        // 每轮结果会互相污染。
        let admission_before = crate::render_cache::admission_counters();

        let mut rendered_count = 0u32;
        let mut cache_hit_count = 0u32;
        let mut cache_miss_count = 0u32;
        let mut render_success_count = 0u32;
        let mut render_failed_count = 0u32;
        // 持久化缓存（磁盘）命中数：用于"本次打开省了多少"的反馈与统计。
        let mut disk_hit_count = 0u32;
        let mut disk_load_elapsed = std::time::Duration::ZERO;
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
                // 走 request_global_cancel 而非直接置位：需要推进纪元
                //（见 BG_RENDER_CANCEL 的生命周期契约）。
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

            // ── 持久化缓存读回（磁盘）──────────────────────────────────────────
            // 内存未命中时回退磁盘：命中即回填内存缓存并注册 pending key，
            // 本次直接跳过合成。磁盘 I/O 只允许发生在这里（渲染线程）——
            // 音频回调与 `build_snapshot` 绝不读盘。
            if base_entry.is_none() && crate::render_cache::enabled() {
                let disk_started_at = std::time::Instant::now();
                if let Some(entry) = crate::render_cache::load_rendered(
                    &clip_render_info.cache_key,
                    clip_render_info.sr,
                ) {
                    disk_load_elapsed += disk_started_at.elapsed();
                    disk_hit_count += 1;
                    if cache_log {
                        log::warn!(
                            "[bg_render][cache] DISK HIT clip_id={} hash={:#018x} load_ms={}",
                            clip_render_info.clip.id,
                            clip_render_info.cache_key.param_hash,
                            disk_started_at.elapsed().as_millis()
                        );
                    }
                    {
                        let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
                            .lock()
                            .unwrap_or_else(|e| e.into_inner());
                        cache.insert(clip_render_info.cache_key.clone(), entry.clone());
                    }
                    crate::synth_clip_cache::register_pending_rendered_key(
                        &clip_render_info.clip.id,
                        clip_render_info.cache_key.clone(),
                    );
                    pending_clip_ids_written.insert(clip_render_info.clip.id.clone());
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
                    base_entry = Some(entry);
                } else {
                    disk_load_elapsed += disk_started_at.elapsed();
                }
            }

            if base_entry.is_none() {
                cache_miss_count += 1;
                if cache_log {
                    log::warn!(
                        "[bg_render][cache] MISS clip_id={} hash={:#018x}",
                        clip_render_info.clip.id,
                        clip_render_info.cache_key.param_hash
                    );
                    // 归因：如果排除某个输入后反而能命中磁盘，就说明那个输入
                    // 在两次会话之间漂移了 —— 这是定位"整库静默失效"的唯一手段。
                    log_render_key_drift(timeline, &clip_render_info.clip, clip_render_info.sr);
                }
                if !rendering_started {
                    rendering_started = true;
                    // 首个未命中才点亮进度条（全命中不闪进度条）。此时可能已有
                    // 若干 clip 命中缓存，直接报告真实整体进度而非硬编码 0。
                    let _ = app.emit(
                        "playback_rendering_state",
                        PlaybackRenderingStateEvent {
                            active: true,
                            progress: Some(crate::renderer::progress::current_fraction()),
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

                        // 【持久化】渲染产物异步落盘（不阻塞渲染线程）。总开关、
                        // 片段时长下限、单条上限、磁盘保留空间等过滤在 store_* 内。
                        crate::render_cache::store_rendered(
                            &clip_render_info.cache_key,
                            &entry,
                        );

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
                                    progress: Some(
                                        crate::renderer::progress::current_fraction(),
                                    ),
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
            // clip 边界推进整体进度（clip 级是所有处理器的权威进度来源 ——
            // WORLD / vslib 没有推理 chunk 回调，clip 粒度即其唯一进度）；
            // 进度条已点亮才发射事件，缓存全命中的 pass 不闪进度条。
            crate::renderer::progress::advance_clip();
            if rendering_started {
                let _ = app.emit(
                    "playback_rendering_state",
                    PlaybackRenderingStateEvent {
                        active: true,
                        progress: Some(crate::renderer::progress::current_fraction()),
                        target: Some("background".to_string()),
                    },
                );
            }
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
            crate::renderer::progress::set_callback(None);
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

        if cache_log {
            log::warn!(
                "[bg_render][cache] DONE total={} hit={} disk_hit={} miss={} rendered_ok={} rendered_fail={} cache_probe_ms={:.2} disk_load_ms={:.2} render_ms={:.2} tension_ms={:.2} total_ms={:.2}",
                total,
                cache_hit_count,
                disk_hit_count,
                cache_miss_count,
                render_success_count,
                render_failed_count,
                cache_probe_elapsed.as_secs_f64() * 1000.0,
                disk_load_elapsed.as_secs_f64() * 1000.0,
                render_elapsed.as_secs_f64() * 1000.0,
                tension_elapsed.as_secs_f64() * 1000.0,
                started_at.elapsed().as_secs_f64() * 1000.0
            );
        }
        // 落盘准入结果（本轮增量）。`accepted` 是"通过准入并已投递写盘"的条数；
        // `stored` 是写盘线程真正落盘的条数（异步，可能略滞后于本轮）。
        let admission = crate::render_cache::admission_counters().since(&admission_before);
        let stored_this_pass = crate::render_cache::session_stored();
        log::warn!(
            "[bg_render] complete: {} clips, {} hit ({} from disk), {} miss, {} ok, {} fail in {:.2}s",
            total,
            cache_hit_count + disk_hit_count,
            disk_hit_count,
            cache_miss_count,
            render_success_count,
            render_failed_count,
            started_at.elapsed().as_secs_f64()
        );
        // 落盘准入是一类**静默失败**：产物进内存缓存、播放完全正常，只是永远
        // 不落盘 → 每次重开工程都要重新合成。只在真有拒绝时才打，且带上原因
        // 分解，使"命中率低"能在日志里直接定位到是哪个闸门在拦。
        if admission.accepted > 0 || admission.skipped > 0 {
            log::warn!(
                "[render_cache] pass admission: accepted={} skipped={} [{}] (stored_total={})",
                admission.accepted,
                admission.skipped,
                if admission.skipped > 0 {
                    admission.reason_summary()
                } else {
                    "none".to_string()
                },
                stored_this_pass
            );
        }

        // 渲染缓存命中汇总：前端据此在状态栏提示"本次打开复用了多少、省了多少"。
        // 用本轮真实渲染的平均耗时估算节省时间；全命中（无新渲染）时不估算。
        {
            let avg_render_ms = if render_success_count > 0 {
                render_elapsed.as_secs_f64() * 1000.0 / render_success_count as f64
            } else {
                0.0
            };
            let _ = app.emit(
                "render_cache_summary",
                serde_json::json!({
                    "diskHits": disk_hit_count,
                    "total": total,
                    "rendered": render_success_count,
                    "misses": cache_miss_count,
                    "savedMs": (avg_render_ms * disk_hit_count as f64).round() as u64,
                    "persisted": admission.accepted,
                    "skipped": admission.skipped,
                    "skippedByReason": crate::render_cache::SkipReason::ALL
                        .iter()
                        .filter_map(|reason| {
                            let count = admission.skipped_by_reason[reason.index()];
                            (count > 0).then_some(serde_json::json!({
                                "reason": reason.id(),
                                "count": count,
                            }))
                        })
                        .collect::<Vec<_>>(),
                }),
            );
        }

        // 旧代数线程完成时不得清理新一轮渲染的全局状态。
        if BG_RENDER_GENERATION.load(Ordering::Acquire) != render_generation {
            return;
        }

        crate::renderer::progress::set_callback(None);
        BG_RENDER_ACTIVE.store(false, Ordering::Release);
        BG_RENDER_CANCEL.store(false, Ordering::Release);

        // 第一轮可能因为音高分析尚未完成而跳过了部分 clip。
        // 音高分析完成后没有新的“缓存失效”事件，因此这里主动补一轮渲染，
        // 保证用户等待后台渲染进度结束后，所有需要渲染的 clip 都真正进入缓存。
        // ★ 补轮必须受"本轮是否有实际进展"约束（见 should_follow_up_render），
        // 否则会形成 100% CPU 的无限后台渲染循环，把整个应用拖到未响应。
        // 触发条件与 handle_update_timeline / handle_clip_pitch_ready 的规则
        // 一致：后台预渲染开启 → 补；关闭时仅当传输层在播放才补 —— 播放触发
        // 的按需渲染同样依赖被跳过 clip 的结果来解除原地等待，且补轮覆盖
        // "音高分析在本轮渲染途中完成"的窗口（此后不再有新的 ClipPitchReady
        // 事件把请求送回来）。
        let transport_playing = app.state::<AppState>().audio_engine.is_playing();
        if should_follow_up_render(skipped_not_ready, render_success_count)
            && (AUTO_BG_RENDER_ENABLED.load(Ordering::Relaxed) || transport_playing)
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
    schedule_render_pass(app.clone());
}

/// 请求一轮渲染，并把去抖窗口内的重复请求合流成一次启动。
///
/// ★ 为什么在**调用 `start_background_render` 之前**清除调度标志：
/// 若等启动完成后再清，等待窗口期间到达的新请求会因为"已有调度线程"而被合流
/// 掉，而此刻 `BG_RENDER_ACTIVE` 还是 false —— 既不进重启合流分支、也不进新
/// 启动分支，请求就真的丢了。提前清除则保证新请求一定会开启新的调度；与在途
/// 调度线程的并发竞争由 `start_background_render` 自身的 `BG_RENDER_ACTIVE`
/// 守卫收口，输掉竞争的一方补发重启请求，让在途 pass 收敛到更新的时间线。
fn schedule_render_pass(app: tauri::AppHandle) {
    use std::sync::atomic::Ordering;

    BG_RENDER_REQUESTED_AT_MS.fetch_max(now_millis(), Ordering::Release);
    if BG_RENDER_SCHEDULER_ACTIVE.swap(true, Ordering::AcqRel) {
        // 同一去抖窗口内已有调度线程在等：本次请求已被合流。
        return;
    }

    std::thread::spawn(move || {
        // 等到"最后一次请求之后静默满一个窗口"为止。循环而非单次 sleep：
        // 窗口内到达的新请求会刷新时间戳，需要重新计算剩余等待。
        loop {
            let last_requested_at = BG_RENDER_REQUESTED_AT_MS.load(Ordering::Acquire);
            let remaining = render_request_debounce_remaining_ms(last_requested_at, now_millis());
            if remaining == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(remaining));
        }
        BG_RENDER_SCHEDULER_ACTIVE.store(false, Ordering::Release);
        // A fresh render request supersedes any stale restart marker.
        BG_RENDER_RESTART_NEEDED.store(false, Ordering::Release);
        let started = start_background_render(app);
        // 并发竞争输给了另一个调度线程：在途 pass 覆盖的是它收集时刻的待渲染
        // 集合，可能落后于本次请求看到的时间线 —— 补发重启请求让它收敛。
        if started
            .get("skipped")
            .and_then(|value| value.as_bool())
            .unwrap_or(false)
        {
            request_bg_render_restart();
        }
    });
}

/// 请求一轮后台预渲染（编辑触发；受"后台预渲染"开关约束）。
///
/// ★ 线程安全约定（防死锁）：
/// `start_background_render` 内部会锁定 `state.timeline`，而本函数常被
/// “已经持有时间线锁”的调用方使用（如 `set_timeline_tempo_map` 的音阶
/// 变化分支）。std Mutex 不可重入，若在调用方线程上同步启动渲染，
/// 调用方会自我死锁 —— 命令线程永久阻塞，整个应用进入“未响应”。
/// 因此这里只做原子的状态检查与标记（无锁），真正的启动交给调度线程，
/// 它会在时间线锁自然释放后再收集待渲染集合。
///
/// 与 [`start_background_render`] 的分工：后者是"现在就起一轮"，本函数是
/// "在去抖窗口结束后起一轮"，并负责在已有 pass 在跑时合流为重启请求。
/// 返回值仅用于诊断，调用方普遍忽略。
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

    schedule_render_pass(app.clone());
    serde_json::json!({"ok": true, "starting": true, "debounced": true})
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
        // 渲染完。走 request_global_cancel 以同时推进纪元（见 render_cancel.rs）。
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
        bg_render_restart_request_is_stale, render_priority_key,
        render_request_debounce_remaining_ms, should_follow_up_render, BG_RENDER_DEBOUNCE_MS,
        BG_RENDER_RESTART_QUIET_WINDOW_MS,
    };

    /// 去抖窗口：请求刚发出时要等满一个窗口；期间被新请求刷新则重新计时；
    /// 静默满窗口后立即可启动。
    #[test]
    fn render_request_debounce_waits_for_a_quiet_window() {
        let t0 = 10_000;
        // 刚请求：需要等满整个窗口。
        assert_eq!(
            render_request_debounce_remaining_ms(t0, t0),
            BG_RENDER_DEBOUNCE_MS
        );
        // 窗口过半。
        assert_eq!(
            render_request_debounce_remaining_ms(t0, t0 + BG_RENDER_DEBOUNCE_MS / 2),
            BG_RENDER_DEBOUNCE_MS - BG_RENDER_DEBOUNCE_MS / 2
        );
        // 静默满窗口 → 可以启动。
        assert_eq!(
            render_request_debounce_remaining_ms(t0, t0 + BG_RENDER_DEBOUNCE_MS),
            0
        );
        // 超出窗口仍为 0（不得下溢成天文数字）。
        assert_eq!(
            render_request_debounce_remaining_ms(t0, t0 + BG_RENDER_DEBOUNCE_MS * 100),
            0
        );
        // 窗口内到达的新请求刷新时间戳 → 重新计满一个窗口（这就是"合流"）。
        let refreshed = t0 + BG_RENDER_DEBOUNCE_MS - 1;
        assert_eq!(
            render_request_debounce_remaining_ms(refreshed, refreshed),
            BG_RENDER_DEBOUNCE_MS
        );
        // 时钟回退（不应出现，但必须不 panic 且不无限等待）。
        assert_eq!(
            render_request_debounce_remaining_ms(t0, t0 - 5_000),
            BG_RENDER_DEBOUNCE_MS
        );
    }

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
