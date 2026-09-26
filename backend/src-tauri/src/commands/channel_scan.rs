//! 假立体声折叠的**可恢复扫描**：一次实现，两个触发点。
//!
//! # 为什么需要"可恢复"
//!
//! 旧实现把 v4→v5 的折叠做成 `open_project` 里的一次性同步迁移，判据是工程
//! 版本号。它有两处致命性质：
//!
//! - **一次性的**：打开瞬间没判成的 Take 停在 `channel_mode = 0`，工程一保存
//!   成 v5 就再没有第二次机会，而 0 会被当成"用户显式决定"永久尊重；
//! - **全量阻塞的**：在命令线程上按文件解码每一个立体声源，大工程是分钟级的
//!   UI 冻结 —— 越慢越容易撞上文件暂时不可读，漏判率随之上升。
//!
//! 现在判定是幂等的：结论连同上下文（内容指纹 / 策略签名 / 消费区间）落进
//! [`crate::channel_decision::ChannelDecisionRecord`]，只有"档案不存在或不权威"
//! 的 Take 才进入候选。读不到的记 `Pending`，**下次打开自动重试**。
//!
//! # 三阶段（与锁纪律对齐）
//!
//! 1. [`collect_targets`]（持锁，零 I/O）：快照出候选 Take 的最小字段；
//! 2. [`plan`]（**锁外**，会解码）：按源文件分组判定，产出落点；
//! 3. [`apply_planned`]（持锁，零 I/O）：按 id 写回，失效受影响的缓存。
//!
//! 后台触发点在 [`request_channel_scan`]：打开工程后跑一遍，跑完才请求后台
//! 预渲染 —— 顺序很重要，否则先渲染出来的结果会被随后的折叠全部失效掉。

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use tauri::{Emitter, Manager};

use crate::channel_decision::{self, ChannelDecisionRecord};
use crate::channel_policy::{self, AppliedResolution, ChannelResolution, ChannelScanOutcome};
use crate::config::ChannelImportPolicy;
use crate::state::{AppState, ClipTake};

/// 扫描世代号：工程切换时递增，在途的后台扫描据此丢弃结果。
///
/// 没有它，上一个工程的扫描线程会在新工程装载后把结论写进新工程的同 id
/// Take（工程文件里 clip/take id 复用是常态）。
static CHANNEL_SCAN_GENERATION: AtomicU64 = AtomicU64::new(0);

/// 是否有后台扫描在跑（避免重复起线程）。
static CHANNEL_SCAN_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 递增世代号，使在途的后台扫描作废。打开 / 新建工程时调用。
pub(crate) fn bump_generation() {
    CHANNEL_SCAN_GENERATION.fetch_add(1, Ordering::AcqRel);
}

fn current_generation() -> u64 {
    CHANNEL_SCAN_GENERATION.load(Ordering::Acquire)
}

/// 一个候选 Take 的最小快照（不克隆波形预览等大字段）。
#[derive(Debug, Clone)]
pub struct ScanTarget {
    pub clip_id: String,
    pub take_id: String,
    pub name: String,
    pub source_path: Option<String>,
    pub source_channels: Option<u16>,
    pub region: Option<(f64, f64)>,
    /// Take 上持久化的内容指纹（判定档案要比对的那一份）。
    pub fingerprint: Option<u64>,
    pub policy_sig: u64,
    /// 该 Take 当前的声道模式（用于判断写回是不是 no-op）。
    pub current_mode: i32,
    /// 当前档案（用于判断是否真的变了）。
    pub current_decision: Option<ChannelDecisionRecord>,
}

/// 持锁阶段：收集候选 Take。
///
/// `only_clip_ids` 为 `Some` 时只收这些 Clip（手动扫描的选区）；`None` = 整个
/// 工程。`include_settled` 为 `true` 时连"已定论"的 Take 也收（手动重新扫描：
/// 用户显式要求重判），否则只收 [`channel_decision::needs_auto_scan`] 认可的
/// 候选（后台扫描：只补没结论的）。
pub fn collect_targets(
    timeline: &crate::state::TimelineState,
    only_clip_ids: Option<&std::collections::HashSet<String>>,
    policy: &ChannelImportPolicy,
    include_settled: bool,
) -> Vec<ScanTarget> {
    let policy_sig = policy.detect_options().signature();
    let mut targets = Vec::new();
    for clip in &timeline.clips {
        if let Some(filter) = only_clip_ids {
            if !filter.contains(&clip.id) {
                continue;
            }
        }
        for take in &clip.takes {
            // 无音频源的 Take（MIDI / 空白 Clip）不属于声道折叠的适用范围：
            // 既不该被折叠，也不该在报告里冒充"读不到"。
            if take.source_path.is_none() {
                continue;
            }
            // 用户封印**无条件**跳过：连"手动重新扫描"也不得覆盖用户的选择
            //（`include_settled` 只放宽"已定论"这一条，不是放宽用户意图）。
            if take
                .channel_decision
                .is_some_and(channel_decision::ChannelDecisionRecord::is_user)
            {
                continue;
            }
            let region_q =
                crate::stereo_detect::quantize_region(channel_policy::take_consumption_region(take));
            if !include_settled
                && !channel_decision::needs_auto_scan(
                    take.channel_decision,
                    take.source_file_fingerprint,
                    policy_sig,
                    region_q,
                )
            {
                continue;
            }
            targets.push(ScanTarget {
                clip_id: clip.id.clone(),
                take_id: take.id.clone(),
                name: take.name.clone(),
                source_path: take.source_path.clone(),
                source_channels: take.source_channels,
                region: channel_policy::take_consumption_region(take),
                fingerprint: take.source_file_fingerprint,
                policy_sig,
                current_mode: take.channel_mode,
                current_decision: take.channel_decision,
            });
        }
    }
    targets
}

/// 一个候选的判定结果。
pub struct PlannedTarget {
    pub target: ScanTarget,
    pub outcome: ChannelScanOutcome,
    pub resolution: ChannelResolution,
    /// 判定证据；仅当 `want_details` 且结论为真立体声时有值。
    pub detail: Option<crate::stereo_detect::VerdictDetail>,
}

/// 锁外阶段：判定全部候选（**会解码**）。
///
/// 按源文件分组执行（`scan_sources_grouped`）：同一音频被多个 Take 引用是常态，
/// 分组后解码量与**文件数**成正比而不是引用数。
///
/// `want_details` 为真时，对结论为"真立体声"的 Take 补取一次证据（超差样本占比 /
/// 最大差）。那次查询**必定命中判定缓存**（分组扫描刚写入），因此不产生额外解码；
/// 后台扫描不需要证据，传 `false` 以免做无谓的缓存查找。
pub fn plan(
    targets: Vec<ScanTarget>,
    policy: &ChannelImportPolicy,
    want_details: bool,
) -> Vec<PlannedTarget> {
    let opts = policy.detect_options();
    let requests: Vec<channel_policy::ScanRequest<'_>> = targets
        .iter()
        .map(|target| channel_policy::ScanRequest {
            source_path: target.source_path.as_deref().map(Path::new),
            source_channels: target.source_channels,
            region: target.region,
        })
        .collect();
    let outcomes = channel_policy::scan_sources_grouped(&requests, policy);

    targets
        .into_iter()
        .enumerate()
        .map(|(index, target)| {
            let outcome = outcomes
                .get(index)
                .copied()
                .unwrap_or(ChannelScanOutcome::Pending);
            let ctx = channel_policy::DecisionContext {
                fingerprint: target.fingerprint,
                policy_sig: target.policy_sig,
                region_q: crate::stereo_detect::quantize_region(target.region),
            };
            let detail = if want_details && outcome == ChannelScanOutcome::TrueStereo {
                target.source_path.as_deref().map(|path| {
                    crate::stereo_detect::verdict_for_file_detailed(
                        Path::new(path),
                        target.region,
                        &opts,
                    )
                })
            } else {
                None
            };
            let resolution = outcome.resolution(policy, ctx);
            PlannedTarget {
                target,
                outcome,
                resolution,
                detail,
            }
        })
        .collect()
}

/// 写回结果统计。
#[derive(Debug, Default, Clone)]
pub struct ApplyStats {
    /// 真正被折叠（声道模式变化）的 Take 数。
    pub folded: usize,
    /// 只更新了判定档案的 Take 数。
    pub recorded: usize,
    /// 本次仍读不到的 Take 数。
    pub pending: usize,
    /// 模式确实被写入的 `(clip_id, take_id, mode)`，供扫描报告逐条回填
    /// `applied_mode` —— 判定说"该折叠"不等于"这次改了"（Take 可能已是目标模式）。
    pub applied: Vec<(String, String, i32)>,
}

/// 持锁阶段：把落点写回时间线（零 I/O）。
///
/// `expected_generation` 非 `None` 时先校验世代号，不匹配则整体放弃（在途的
/// 后台扫描发现工程已切换）。返回 `None` 表示被世代号拦下。
///
/// `with_undo` 决定是否留下撤销步：手动命令要（用户可撤销），后台迁移不要
/// （它是加载的一部分，不是用户的编辑）。撤销步**惰性**产生 —— 第一次真正
/// 写入模式前才快照，全部 no-op 时不留空步。
pub fn apply_planned(
    state: &AppState,
    planned: &[PlannedTarget],
    expected_generation: Option<u64>,
    with_undo: bool,
) -> Option<ApplyStats> {
    if let Some(expected) = expected_generation {
        if expected != current_generation() {
            return None;
        }
    }

    let mut stats = ApplyStats::default();
    // (clip_id, 是否 active take) —— 用于写回后的缓存失效与 formant 重调度。
    let mut mode_changed_clips: Vec<(String, bool)> = Vec::new();
    let mut any_record_change = false;

    {
        let mut timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        // 惰性撤销点：先判断这批里到底有没有东西要写。全 no-op 时不留空撤销步
        // （判定为"该折叠"的 Take 可能已处于目标模式，写回是 no-op）。
        // 借用在这里就结束，下面的可变遍历才能拿 `timeline`。
        if with_undo {
            let needs_checkpoint = {
                let snapshot: &crate::state::TimelineState = &timeline;
                planned.iter().any(|item| {
                    snapshot
                        .clips
                        .iter()
                        .find(|clip| clip.id == item.target.clip_id)
                        .and_then(|clip| {
                            clip.takes
                                .iter()
                                .find(|take| take.id == item.target.take_id)
                        })
                        .map(|take| {
                            channel_policy::resolution_changes(take, item.resolution)
                        })
                        .unwrap_or(false)
                })
            };
            if needs_checkpoint {
                state.checkpoint_timeline(&timeline, crate::state::HistoryOp::TakeChannelMode);
            }
        }

        for item in planned {
            if item.outcome.is_pending() {
                stats.pending += 1;
            }
            let Some(clip) = timeline
                .clips
                .iter_mut()
                .find(|clip| clip.id == item.target.clip_id)
            else {
                // Clip 已被删除（用户在扫描期间编辑）：跳过，不是错误。
                continue;
            };
            let is_active = clip.active_take_id.as_deref() == Some(item.target.take_id.as_str());
            let Some(take) = clip
                .takes
                .iter_mut()
                .find(|take| take.id == item.target.take_id)
            else {
                continue;
            };

            let applied: AppliedResolution =
                channel_policy::apply_resolution(take, item.resolution);
            if !applied.any() {
                continue;
            }
            if applied.mode_changed {
                stats.folded += 1;
                if let Some(mode) = item.resolution.mode {
                    stats
                        .applied
                        .push((clip.id.clone(), item.target.take_id.clone(), mode));
                }
                mode_changed_clips.push((clip.id.clone(), is_active));
            }
            if applied.record_changed {
                any_record_change = true;
                stats.recorded += 1;
            }

            // 声道模式是 active take 的内存投影：改完 Take 必须物化回投影，
            // 否则下一次 sync 会用旧投影覆盖（前端读的也是投影）。
            if is_active {
                let materialized = clip
                    .takes
                    .iter()
                    .find(|take| take.id == item.target.take_id)
                    .cloned();
                if let Some(take) = materialized {
                    take.apply_to_clip(clip);
                }
            }
        }

        if !mode_changed_clips.is_empty() {
            for (clip_id, is_active) in &mode_changed_clips {
                crate::commands::timeline::invalidate_take_related_caches(clip_id);
                if *is_active {
                    crate::commands::timeline::maybe_schedule_formant_rebuild(state, &timeline, clip_id);
                }
            }
            state.audio_engine.update_timeline(timeline.clone());
        } else if any_record_change {
            // 只记账也要让引擎看到新时间线（档案随 Take 进快照），但不必失效
            // 任何渲染缓存 —— 听感没有变化。
            state.audio_engine.update_timeline(timeline.clone());
        }
    }

    Some(stats)
}

/// 活跃 Take 的声道模式发生过变化的 Clip（后台扫描用它在锁外重调度音高分析）。
pub fn mode_changed_active_clips(planned: &[PlannedTarget], stats: &ApplyStats) -> Vec<String> {
    if stats.folded == 0 {
        return Vec::new();
    }
    // 只有真正改了模式的才需要重调度；`apply_planned` 已经按 id 匹配过，
    // 这里按同一判据重算一遍（模式不同的才是变化项）。
    planned
        .iter()
        .filter(|item| {
            item.resolution
                .mode
                .map(|mode| mode != item.target.current_mode)
                .unwrap_or(false)
        })
        .map(|item| item.target.clip_id.clone())
        .collect()
}

/// 后台扫描的进度事件负载。
fn emit_progress(app: &tauri::AppHandle, done: usize, total: usize, stats: &ApplyStats) {
    let _ = app.emit(
        "channel_scan_progress",
        serde_json::json!({
            "done": done,
            "total": total,
            "folded": stats.folded,
            "pending": stats.pending,
            "finished": done >= total,
        }),
    );
}

/// 请求一轮后台声道扫描（打开工程后调用）。
///
/// 跑完**才**请求后台预渲染：折叠会失效渲染缓存，反过来做会让先渲染出来的
/// 结果全部白做。没有候选时直接放行渲染，不留空转。
pub(crate) fn request_channel_scan(app: &tauri::AppHandle) {
    if CHANNEL_SCAN_ACTIVE.swap(true, Ordering::AcqRel) {
        // 已有一轮在跑：它会看到最新的时间线（世代号未变），本次合流。
        return;
    }
    let generation = current_generation();
    let app = app.clone();
    std::thread::spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_scan_pass(&app, generation);
        }));
        CHANNEL_SCAN_ACTIVE.store(false, Ordering::Release);
        if result.is_err() {
            log::error!("[channel_scan] background scan panicked; render request still issued");
        }
        // 无论扫描结果如何都要放行预渲染，否则一次失败会让渲染永远停摆。
        crate::commands::playback::request_background_render(&app);
    });
}

/// 单轮扫描：收集 → 判定 → 写回。世代号在写回前校验。
fn run_scan_pass(app: &tauri::AppHandle, generation: u64) {
    let state = match app.try_state::<AppState>() {
        Some(state) => state,
        None => return,
    };
    let policy = crate::config::channel_import_policy();

    let targets = {
        let timeline = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        collect_targets(&timeline, None, &policy, false)
    };
    let total = targets.len();
    if total == 0 {
        return;
    }
    log::info!("[channel_scan] scanning {total} take(s) needing a channel decision");
    emit_progress(app, 0, total, &ApplyStats::default());

    // 分批判定：每批完成后立刻写回并汇报进度。整批一次做完会让大工程在
    // 扫描期间完全没有反馈，而分批也让"工程中途被切换"能更早被发现。
    const BATCH: usize = 64;
    let mut done = 0usize;
    let mut total_stats = ApplyStats::default();
    for chunk in targets.chunks(BATCH) {
        let planned = plan(chunk.to_vec(), &policy, false);
        match apply_planned(&state, &planned, Some(generation), false) {
            Some(stats) => {
                total_stats.folded += stats.folded;
                total_stats.recorded += stats.recorded;
                total_stats.pending += stats.pending;
            }
            // 世代号不匹配：工程已切换，丢弃余下工作。
            None => {
                log::info!("[channel_scan] superseded by a project switch; aborting pass");
                return;
            }
        }
        done += chunk.len();
        emit_progress(app, done, total, &total_stats);
    }

    if total_stats.folded > 0 {
        log::info!(
            "[channel_scan] folded {} take(s) to mono ({} still unreadable)",
            total_stats.folded,
            total_stats.pending
        );
    }
}

/// 测试专用：等待在途的后台扫描结束。
#[cfg(test)]
pub(crate) fn wait_for_idle() {
    for _ in 0..600 {
        if !CHANNEL_SCAN_ACTIVE.load(Ordering::Acquire) {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// 把一个"用户显式设置声道模式"的封印写进 Take。
///
/// 这是判定档案存在的意义之一：用户一旦显式选择过，自动扫描必须永远放手。
/// 任何写入 `channel_mode` 的用户命令都应经过这里，而不是直接赋值。
pub fn seal_user_channel_mode(take: &mut ClipTake, mode: i32) {
    let mode = crate::channel_mode::TakeChannelMode::from_raw(mode).raw();
    take.channel_mode = mode;
    take.channel_decision = Some(ChannelDecisionRecord::user());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ChannelImportPolicy;

    /// 写一个临时 WAV；`identical` 为 true 时 L == R（假立体声）。
    fn write_wav(name: &str, identical: bool, seconds: u32) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("hifishifter_channel_scan_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(name);
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("wav");
        for i in 0..(44_100 * seconds) {
            let v = ((i as f32) * 0.01).sin() * 0.4;
            writer.write_sample(v).expect("l");
            writer
                .write_sample(if identical { v } else { -v })
                .expect("r");
        }
        writer.finalize().expect("finalize");
        path
    }

    /// 造一个单 Clip 单 Take 的时间线，Take 指向 `source`。
    fn timeline_with_take(source: Option<&std::path::Path>) -> crate::state::TimelineState {
        let mut tl = crate::state::TimelineState::default();
        let track = tl.tracks[0].id.clone();
        let clip_id = tl.add_clip(
            Some(track),
            Some("V".to_string()),
            Some(0.0),
            Some(1.0),
            source.map(|p| p.to_string_lossy().to_string()),
        );
        let clip = tl.clips.iter_mut().find(|c| c.id == clip_id).expect("clip");
        clip.sync_take_from_flat();
        for take in &mut clip.takes {
            take.channel_mode = 0;
            take.channel_decision = None;
        }
        tl
    }

    /// 走一遍"收集 → 判定 → 写回"，返回写回统计。
    fn run_once(
        state: &AppState,
        policy: &ChannelImportPolicy,
        generation: Option<u64>,
    ) -> Option<ApplyStats> {
        let targets = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            collect_targets(&tl, None, policy, false)
        };
        let planned = plan(targets, policy, false);
        apply_planned(state, &planned, generation, false)
    }

    fn take0(state: &AppState) -> ClipTake {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.clips[0].takes[0].clone()
    }

    #[test]
    fn sealed_user_choice_is_never_a_candidate() {
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/x.wav")));
        tl.clips[0].takes[0].channel_decision = Some(ChannelDecisionRecord::user());
        let targets = collect_targets(&tl, None, &policy, false);
        assert!(targets.is_empty(), "用户封印的 Take 不得进入候选");
        // 手动重扫（include_settled）也照样跳过。
        assert!(collect_targets(&tl, None, &policy, true).is_empty());
    }

    #[test]
    fn source_less_takes_are_out_of_scope() {
        let policy = ChannelImportPolicy::default();
        let tl = timeline_with_take(None);
        assert!(
            collect_targets(&tl, None, &policy, false).is_empty(),
            "无源 Take（MIDI / 空白 Clip）不属于声道折叠的适用范围"
        );
    }

    #[test]
    fn a_take_that_could_not_be_read_is_retried_on_the_next_pass() {
        // 这是"漏判不再永久化"的核心：第一次读不到 ⇒ 记 pending ⇒ 下一轮
        // 仍在候选里。旧实现会把这种 Take 永久留在 channel_mode = 0。
        let policy = ChannelImportPolicy::default();
        let missing = std::path::Path::new("C:/definitely/missing.wav");
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(missing));
        }

        let stats = run_once(&state, &policy, None).expect("apply");
        assert_eq!(stats.pending, 1);
        assert_eq!(stats.folded, 0);
        assert_eq!(take0(&state).channel_mode, 0);
        assert!(take0(&state).channel_decision.expect("record").is_pending());

        // 第二轮：仍然进候选（对比"已定论"的真立体声会被跳过）。
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            collect_targets(&tl, None, &policy, false).len(),
            1,
            "pending 的 Take 必须保持可重试"
        );
    }

    #[test]
    fn a_settled_verdict_is_not_rejudged() {
        // 已定论且上下文未变 ⇒ 零解码直接采用（跨会话复用）。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("settled_true.wav", false, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
        }

        let stats = run_once(&state, &policy, None).expect("apply");
        assert_eq!(stats.recorded, 1, "真立体声要落档案");
        assert_eq!(stats.folded, 0);

        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            collect_targets(&tl, None, &policy, false).is_empty(),
            "已定论的 Take 不该再进候选（否则每次打开都重解码）"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_fake_stereo_file_is_folded_and_recorded() {
        let policy = ChannelImportPolicy::default();
        let path = write_wav("scan_fake.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
        }

        let stats = run_once(&state, &policy, None).expect("apply");
        assert_eq!(stats.folded, 1);
        let take = take0(&state);
        assert_eq!(take.channel_mode, 2, "假立体声应折叠为 MonoMix");
        assert!(!take.channel_decision.expect("record").is_pending());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_stale_generation_discards_the_whole_batch() {
        // 工程切换后，在途扫描不得把结论写进新工程的同 id Take。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("stale_gen.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
        }

        let stale = current_generation().wrapping_sub(1);
        assert!(
            run_once(&state, &policy, Some(stale)).is_none(),
            "世代号不匹配必须整体放弃"
        );
        assert_eq!(take0(&state).channel_mode, 0, "不得写入任何东西");
        assert_eq!(take0(&state).channel_decision, None);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn trimming_the_consumption_region_invalidates_the_verdict() {
        // 判定只对消费区间负责：区间变了必须重判（否则 trim 出来的新片段
        // 会沿用旧区间得出的结论）。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("region_stale.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
        }
        run_once(&state, &policy, None).expect("first pass");

        {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            assert!(collect_targets(&tl, None, &policy, false).is_empty());
        }
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            tl.clips[0].source_start_sec = 0.25;
            tl.clips[0].source_end_sec = 0.75;
            tl.clips[0].sync_take_from_flat();
        }
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            collect_targets(&tl, None, &policy, false).len(),
            1,
            "消费区间变了 ⇒ 旧结论失效，必须重判"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn collect_targets_honours_a_clip_filter() {
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/x.wav")));
        // 再加一个 Clip。
        let track = tl.tracks[0].id.clone();
        tl.add_clip(
            Some(track),
            Some("W".to_string()),
            Some(2.0),
            Some(1.0),
            Some("C:/y.wav".to_string()),
        );
        let only_first: std::collections::HashSet<String> =
            [tl.clips[0].id.clone()].into_iter().collect();
        assert_eq!(
            collect_targets(&tl, Some(&only_first), &policy, false).len(),
            1
        );
    }
}
