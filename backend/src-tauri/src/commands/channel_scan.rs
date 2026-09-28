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

use crate::channel_decision;
use crate::channel_policy::{self, AppliedResolution, ChannelResolution, ChannelScanOutcome};
use crate::config::ChannelImportPolicy;
use crate::state::AppState;

/// 扫描世代号：工程切换时递增，在途的后台扫描据此丢弃结果。
///
/// 没有它，上一个工程的扫描线程会在新工程装载后把结论写进新工程的同 id
/// Take（工程文件里 clip/take id 复用是常态）。
static CHANNEL_SCAN_GENERATION: AtomicU64 = AtomicU64::new(0);

/// 是否有后台扫描在跑（避免重复起线程）。
static CHANNEL_SCAN_ACTIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// 扫描进行中又来了新请求：本轮结束后必须再跑一轮。
///
/// 不能把并发请求直接丢掉：打开工程触发的扫描可能已经取完快照，此时用户导入
/// 了新媒体（或重链了文件），那些新 Take 不在本轮候选里。丢掉请求就等于它们
/// 要等到下次打开工程才会被判 —— 与"可恢复"的承诺不符。
static CHANNEL_SCAN_REQUESTED_AGAIN: std::sync::atomic::AtomicBool =
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
}

/// 候选筛选的副产物：**为什么**没有候选。
///
/// "0 个 Take" 必须是可解释的，否则这个命令在出问题时就是一个黑箱：选区与后端
/// 对不上、选中的 Clip 确实没有音频源、全都已被用户封印 —— 三种完全不同的
/// 情况在界面上长得一模一样（都是 0），用户只能看到"功能什么都不做"。
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ScanEligibility {
    /// 工程里的 Clip 总数（不受选区筛选影响）。
    ///
    /// 用来区分两件完全不同的事：**工程本身是空的**，还是**选区与工程对不上**。
    /// 只看"命中 0 个"会把后者误报成前者、或反过来，并且都像是在怪用户的选区。
    pub project_clips: usize,
    /// 命中筛选条件的 Clip 数。
    pub matched_clips: usize,
    /// 这些 Clip 里被检查的 Take 总数。
    pub takes_seen: usize,
    /// 因**没有音频源**而跳过的 Take 数。
    pub skipped_no_source: usize,
    /// 因**可信的用户封印**而跳过的 Take 数（只有自动扫描会跳过）。
    pub skipped_user_seal: usize,
    /// 带着用户封印、但被**显式命令**纳入判定的 Take 数。
    pub overrode_user_seal: usize,
}

/// [`collect_targets`] 的结果：候选 + 筛选统计。
pub struct TargetSelection {
    pub targets: Vec<ScanTarget>,
    pub eligibility: ScanEligibility,
}


/// Take 的音频源路径 —— 与**渲染路径**读的是同一份数据。
///
/// # 为什么必须与渲染路径对齐
///
/// 引擎、音高编辑、混音一律读 `Clip.source_path`（active take 的同名投影，由
/// [`crate::state::ClipTake::apply_to_clip`] 维护）；而扫描过去只读 Take 自己的
/// 字段。两者一旦不同步（旧版序列化、归档快照、派生路径），就会出现最坏的一种
/// 不一致：**用户听得到声音，扫描却认为这个 Take 没有音频源** —— 于是"扫描
/// 假立体声并转换"静默地什么都不做，状态栏永远显示 0 个 Take。
///
/// 因此这里按渲染语义解析：Take 自己的路径优先；缺位时**只有 active Take**
/// 退回 Clip 投影（投影就是它的路径）。inactive Take 不能退回 —— 投影是 active
/// Take 的，借用它会把两个不同素材混为一谈。
fn take_audio_source(clip: &crate::state::Clip, take: &crate::state::ClipTake) -> Option<String> {
    let own = take
        .source_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty());
    if let Some(path) = own {
        return Some(path.to_string());
    }
    let is_active = clip.active_take_id.as_deref() == Some(take.id.as_str());
    if !is_active {
        return None;
    }
    clip.source_path
        .as_deref()
        .map(str::trim)
        .filter(|path| !path.is_empty())
        .map(str::to_string)
}

/// 持锁阶段：收集候选 Take。
///
/// `only_clip_ids` 为 `Some` 时只收这些 Clip（手动扫描的选区）；`None` = 整个
/// 工程。`include_settled` 为 `true` 时连"已定论"的 Take 也收（手动重新扫描：
/// 用户显式要求重判），否则只收 [`channel_decision::needs_auto_scan`] 认可的
/// 候选（后台扫描：只补没结论的）。
///
/// `respect_user_seal` 决定**用户的声道模式选择能不能否决这次扫描**：
/// - `true`（自动扫描）：带可信封印的 Take 一律跳过 —— 后台流程绝不该悄悄
///   推翻用户的选择；
/// - `false`（显式命令）：封印不否决，但会被记成 [`ScanEligibility::overrode_user_seal`]
///   以便如实汇报。用户此刻点的是"扫描假立体声并转换"，那本身就是一条更新的、
///   更具体的指令，不该被一条旧设置挡掉（而且折叠只在 L≈R 成立时才发生，
///   之后仍可一步撤销）。
pub fn collect_targets(
    timeline: &crate::state::TimelineState,
    only_clip_ids: Option<&std::collections::HashSet<String>>,
    policy: &ChannelImportPolicy,
    include_settled: bool,
    respect_user_seal: bool,
) -> TargetSelection {
    let policy_sig = policy.detect_options().signature();
    let mut targets = Vec::new();
    let mut eligibility = ScanEligibility {
        project_clips: timeline.clips.len(),
        ..ScanEligibility::default()
    };
    for clip in &timeline.clips {
        if let Some(filter) = only_clip_ids {
            if !filter.contains(&clip.id) {
                continue;
            }
        }
        eligibility.matched_clips += 1;

        // `takes` 为空时，**投影本身就是那个唯一的 Take**：与
        // `sync_take_from_flat` 的物化规则逐字段一致（同一个 id、同一份媒体
        // 信息）。不能把它当作"没有音频" —— 渲染路径照样会用投影里的路径出声，
        // 而扫描若因此跳过，就又回到"有声但扫不到"。
        let synthesized;
        let takes: &[crate::state::ClipTake] = if clip.takes.is_empty() {
            synthesized = [crate::state::ClipTake::from_clip(clip)];
            &synthesized
        } else {
            &clip.takes
        };

        for take in takes {
            eligibility.takes_seen += 1;
            // 音频源按**渲染语义**解析（见 `take_audio_source`）。
            let Some(source_path) = take_audio_source(clip, take) else {
                eligibility.skipped_no_source += 1;
                continue;
            };
            // 用户封印：自动扫描一律让路；显式命令不否决，但如实记账。
            let user_sealed = take
                .channel_decision
                .is_some_and(channel_decision::ChannelDecisionRecord::is_trusted_user_seal);
            if user_sealed {
                if respect_user_seal {
                    eligibility.skipped_user_seal += 1;
                    continue;
                }
                eligibility.overrode_user_seal += 1;
            }
            let region = channel_policy::take_consumption_region(take);
            let region_q = crate::stereo_detect::quantize_region(region);
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
                source_path: Some(source_path),
                source_channels: take.source_channels,
                region,
                fingerprint: take.source_file_fingerprint,
                policy_sig,
            });
        }
    }
    TargetSelection {
        targets,
        eligibility,
    }
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

        // 惰性撤销点：先判断这批里到底有没有**听感变化**。判定档案是内部记账，
        // 它从无到有不构成用户的一次编辑 —— 为它留撤销步会让用户按撤销时发现
        // "什么都没变"。同理，判定说"该折叠"但 Take 已是目标模式时也不留。
        // 借用在这里就结束，下面的可变遍历才能拿 `timeline`。
        if with_undo {
            let needs_checkpoint = {
                let snapshot: &crate::state::TimelineState = &timeline;
                planned.iter().any(|item| {
                    let Some(clip) = snapshot
                        .clips
                        .iter()
                        .find(|clip| clip.id == item.target.clip_id)
                    else {
                        return false;
                    };
                    // 与写回路径同一套解析：`takes` 为空时投影即唯一 Take。
                    // 判据不一致会让"即将发生的模式变化"漏掉撤销步。
                    let synthesized;
                    let take = match clip
                        .takes
                        .iter()
                        .find(|take| take.id == item.target.take_id)
                    {
                        Some(take) => Some(take),
                        None if clip.takes.is_empty() => {
                            synthesized = crate::state::ClipTake::from_clip(clip);
                            Some(&synthesized)
                        }
                        None => None,
                    };
                    take.map(|take| {
                        channel_policy::resolution_changes_mode(take, item.resolution)
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
            // 候选可能来自"投影即唯一 Take"（`takes` 为空，见 `collect_targets`）：
            // 写回前先按同一规则物化，否则下面的按 id 查找必然落空，折叠被静默
            // 丢弃 —— 那正是"报告里 0 个已折叠"的另一种成因。
            if clip.takes.is_empty() {
                clip.sync_take_from_flat();
            }
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
        // 已有一轮在跑：它可能已经取完快照，看不到这次的新 Take（导入 / 重链
        // 发生在它收集之后）。标记"还要再来一轮"，由本轮收尾时接手。
        CHANNEL_SCAN_REQUESTED_AGAIN.store(true, Ordering::Release);
        return;
    }
    let app = app.clone();
    std::thread::spawn(move || {
        loop {
            CHANNEL_SCAN_REQUESTED_AGAIN.store(false, Ordering::Release);
            let generation = current_generation();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_scan_pass(&app, generation);
            }));
            if result.is_err() {
                log::error!("[channel_scan] background scan panicked");
                // 出错时不再自旋重试（否则一次持续性故障会变成忙循环）。
                CHANNEL_SCAN_REQUESTED_AGAIN.store(false, Ordering::Release);
            }
            if !CHANNEL_SCAN_REQUESTED_AGAIN.swap(false, Ordering::AcqRel) {
                break;
            }
        }
        CHANNEL_SCAN_ACTIVE.store(false, Ordering::Release);
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
        collect_targets(&timeline, None, &policy, false, true)
    };
    let total = targets.targets.len();
    if total == 0 {
        return;
    }
    let targets = targets.targets;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel_decision::ChannelDecisionRecord;
    use crate::config::ChannelImportPolicy;
    use crate::state::ClipTake;

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
            collect_targets(&tl, None, policy, false, true)
        };
        let planned = plan(targets.targets, policy, false);
        apply_planned(state, &planned, generation, false)
    }

    fn take0(state: &AppState) -> ClipTake {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        tl.clips[0].takes[0].clone()
    }

    #[test]
    fn an_active_take_without_its_own_path_uses_the_clip_projection() {
        // 回归（"有声但扫不到"）：渲染/引擎一律读 `Clip.source_path`，而扫描过去
        // 只读 Take 自己的字段。两者一旦不同步（旧版序列化、归档快照、派生路径），
        // 就会出现最坏的不一致 —— 用户听得到声音，扫描却认为这个 Take 没有音频源，
        // 于是"扫描假立体声并转换"静默地什么都不做，状态栏永远 0 个 Take。
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/audio/x.wav")));
        // Take 侧的路径缺位，而 Clip 投影（= 渲染路径读的那份）仍在。
        tl.clips[0].takes[0].source_path = None;
        assert_eq!(
            tl.clips[0].source_path.as_deref(),
            Some("C:/audio/x.wav"),
            "前提：投影里仍有源（渲染路径看得到）"
        );

        let selection = collect_targets(&tl, None, &policy, true, true);
        assert_eq!(selection.targets.len(), 1, "投影里有源的 Take 必须仍是候选");
        assert_eq!(
            selection.targets[0].source_path.as_deref(),
            Some("C:/audio/x.wav"),
            "候选必须按渲染语义取到源路径"
        );
        assert_eq!(selection.eligibility.skipped_no_source, 0);
    }

    #[test]
    fn an_inactive_take_without_its_own_path_is_not_borrowed_from_the_active_one() {
        // 反方向同样重要：inactive Take 不能退回投影 —— 投影是 active Take 的
        // 路径，借用它会把两个完全不同的素材混为一谈（对错误的文件下折叠结论）。
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/audio/active.wav")));
        let active_id = {
            let clip = &mut tl.clips[0];
            clip.sync_take_from_flat();
            let active_id = clip.takes[0].id.clone();
            let mut inactive = clip.takes[0].clone();
            inactive.id = "take_inactive".into();
            inactive.source_path = None;
            clip.takes.push(inactive);
            active_id
        };
        let selection = collect_targets(&tl, None, &policy, true, true);
        assert_eq!(selection.targets.len(), 1, "只有 active Take 是候选");
        assert_eq!(selection.targets[0].take_id, active_id);
        assert_eq!(
            selection.eligibility.skipped_no_source, 1,
            "无源的 inactive Take 必须记成'没有音频源'"
        );
    }

    #[test]
    fn a_clip_with_no_take_records_is_still_scannable_and_folds() {
        // `takes` 为空时，投影本身就是那个唯一的 Take（与 `sync_take_from_flat`
        // 的物化规则一致）。不能当作"没有音频"：渲染路径照样用它出声。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("projected_only.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
            tl.clips[0].takes.clear();
            tl.clips[0].active_take_id = None;
            assert!(tl.clips[0].source_path.is_some(), "投影里仍有音频源");
        }

        let selection = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            collect_targets(&tl, None, &policy, true, true)
        };
        assert_eq!(selection.targets.len(), 1, "投影即唯一 Take，必须可被扫描");

        let planned = plan(selection.targets, &policy, true);
        let stats = apply_planned(&state, &planned, None, true).expect("apply");
        assert_eq!(stats.folded, 1, "写回前会物化 Take，折叠必须真正落地");
        {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(tl.clips[0].takes.len(), 1);
            assert_eq!(tl.clips[0].takes[0].channel_mode, 2);
            assert_eq!(tl.clips[0].channel_mode, 2, "投影也要同步");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_blank_source_path_counts_as_having_no_source() {
        // 空白路径不是"不可读"，而是"没有源"：过去它通过了 `is_none()` 检查，
        // 会被判成 Pending 反复重试，报告里也读不出真正的原因。
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/audio/z.wav")));
        tl.clips[0].takes[0].source_path = Some("   ".into());
        tl.clips[0].source_path = None;
        let selection = collect_targets(&tl, None, &policy, true, true);
        assert!(selection.targets.is_empty());
        assert_eq!(selection.eligibility.skipped_no_source, 1);
    }

    #[test]
    fn eligibility_explains_a_zero_candidate_result() {
        // "0 个 Take" 必须可解释：选区对不上 / 没有音频源 / 用户封印，三者在
        // 界面上曾经长得一模一样。
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/audio/a.wav")));
        let track = tl.tracks[0].id.clone();
        tl.add_clip(Some(track), Some("M".into()), Some(2.0), Some(1.0), None);
        let track = tl.tracks[0].id.clone();
        let id = tl.add_clip(
            Some(track),
            Some("S".into()),
            Some(4.0),
            Some(1.0),
            Some("C:/audio/s.wav".into()),
        );
        {
            let clip = tl.clips.iter_mut().find(|c| c.id == id).expect("clip");
            clip.sync_take_from_flat();
            clip.takes[0].channel_decision = Some(ChannelDecisionRecord::user(3));
        }

        let selection = collect_targets(&tl, None, &policy, true, true);
        assert_eq!(selection.eligibility.project_clips, 3);
        assert_eq!(selection.eligibility.matched_clips, 3);
        assert_eq!(selection.eligibility.takes_seen, 3);
        assert_eq!(selection.eligibility.skipped_no_source, 1);
        assert_eq!(selection.eligibility.skipped_user_seal, 1);
        assert_eq!(selection.targets.len(), 1, "只有一个真正可扫的 Take");

        // 选区与后端对不上时，matched_clips 为 0 —— 另一种完全不同的原因。
        let none: std::collections::HashSet<String> =
            ["does_not_exist".to_string()].into_iter().collect();
        let unmatched = collect_targets(&tl, Some(&none), &policy, true, true);
        assert_eq!(unmatched.eligibility.matched_clips, 0);
        assert_eq!(
            unmatched.eligibility.project_clips, 3,
            "工程非空 ⇒ 这是'范围对不上'而不是'工程是空的'"
        );
        assert!(unmatched.targets.is_empty());
    }

    #[test]
    fn the_explicit_scan_overrides_a_user_seal_but_the_automatic_one_does_not() {
        // 用户此刻点的是"扫描假立体声并转换"—— 那是一条更新的、更具体的指令，
        // 不该被一条旧的声道模式设置挡掉（否则命令会一声不吭地什么都不做）。
        // 而自动扫描（打开工程时的补漏）绝不该悄悄推翻用户的选择。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("sealed_explicit.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
            tl.clips[0].takes[0].channel_decision = Some(ChannelDecisionRecord::user(0));
        }

        let automatic = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            collect_targets(&tl, None, &policy, true, true)
        };
        assert!(automatic.targets.is_empty(), "自动扫描必须让路");
        assert_eq!(automatic.eligibility.skipped_user_seal, 1);

        let explicit = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            collect_targets(&tl, None, &policy, true, false)
        };
        assert_eq!(explicit.targets.len(), 1, "显式命令不被旧设置挡掉");
        assert_eq!(explicit.eligibility.overrode_user_seal, 1);
        assert_eq!(explicit.eligibility.skipped_user_seal, 0);

        // 而且折叠真的落地（一条撤销步可回退）。
        let planned = plan(explicit.targets, &policy, true);
        let stats = apply_planned(&state, &planned, None, true).expect("apply");
        assert_eq!(stats.folded, 1);
        assert_eq!(take0(&state).channel_mode, 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn sealed_user_choice_is_never_a_candidate() {
        let policy = ChannelImportPolicy::default();
        let mut tl = timeline_with_take(Some(std::path::Path::new("C:/x.wav")));
        tl.clips[0].takes[0].channel_decision = Some(ChannelDecisionRecord::user(0));
        let targets = collect_targets(&tl, None, &policy, false, true).targets;
        assert!(targets.is_empty(), "用户封印的 Take 不得进入候选");
        // 手动重扫（include_settled）也照样跳过。
        assert!(collect_targets(&tl, None, &policy, true, true).targets.is_empty());
    }

    #[test]
    fn the_explicit_scan_detects_even_when_the_import_policy_is_off() {
        // 回归：显式命令（右键"扫描假立体声并转换"）曾直接套用导入策略的 mode，
        // 于是"不自动转换声道"会把命令静默变成空操作 —— 用户点了却没有反应。
        let path = write_wav("explicit_off.wav", true, 1);
        let tl = timeline_with_take(Some(&path));
        let stored = ChannelImportPolicy {
            mode: "off".into(),
            ..Default::default()
        };

        // 自动路径：off 就是不判定（符合"不自动转换"的语义）。
        let auto = plan(collect_targets(&tl, None, &stored, false, true).targets, &stored, false);
        assert!(
            auto.iter().all(|p| p.resolution.mode.is_none()),
            "off 时自动扫描不得折叠"
        );

        // 显式命令：强制检测语义 ⇒ 假立体声必须被识别并折叠。
        let scan_policy = stored.for_explicit_scan();
        let explicit = plan(
            collect_targets(&tl, None, &scan_policy, true, true).targets,
            &scan_policy,
            false,
        );
        assert_eq!(explicit[0].outcome, ChannelScanOutcome::FakeStereo);
        assert_eq!(explicit[0].resolution.mode, Some(2), "显式扫描必须能折叠");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_explicit_scan_never_folds_true_stereo_under_always_mono() {
        // 回归：导入策略设为"全部转换为单声道"时，显式扫描曾跟着无差别折叠 ——
        // 一个叫"扫描假立体声"的命令不该有折叠真立体声的破坏力。
        let path = write_wav("explicit_forced.wav", false, 1);
        let tl = timeline_with_take(Some(&path));
        let stored = ChannelImportPolicy {
            mode: "alwaysMono".into(),
            ..Default::default()
        };
        let scan_policy = stored.for_explicit_scan();
        let planned = plan(
            collect_targets(&tl, None, &scan_policy, true, true).targets,
            &scan_policy,
            false,
        );
        assert_eq!(planned[0].outcome, ChannelScanOutcome::TrueStereo);
        assert_eq!(
            planned[0].resolution.mode, None,
            "显式扫描必须按检测结果行事，不得无差别折叠真立体声"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn source_less_takes_are_out_of_scope() {
        let policy = ChannelImportPolicy::default();
        let tl = timeline_with_take(None);
        assert!(
            collect_targets(&tl, None, &policy, false, true).targets.is_empty(),
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
            collect_targets(&tl, None, &policy, false, true).targets.len(),
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
            collect_targets(&tl, None, &policy, false, true).targets.is_empty(),
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
            assert!(collect_targets(&tl, None, &policy, false, true).targets.is_empty());
        }
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            tl.clips[0].source_start_sec = 0.25;
            tl.clips[0].source_end_sec = 0.75;
            tl.clips[0].sync_take_from_flat();
        }
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            collect_targets(&tl, None, &policy, false, true).targets.len(),
            1,
            "消费区间变了 ⇒ 旧结论失效，必须重判"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn manual_rescan_leaves_exactly_one_undo_step() {
        // 手动扫描是用户显式操作，必须可撤销；后台迁移则不留撤销步。这里覆盖
        // 撤销步路径本身（它要在**持有 timeline 锁**时快照，写错就是自我死锁）。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("manual_undo.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
        }
        let depth_before = state.history_depths().0;

        let targets = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            // 手动重扫连"已定论"的 Take 也一并重判。
            collect_targets(&tl, None, &policy, true, true).targets
        };
        let planned = plan(targets, &policy, true);
        let stats = apply_planned(&state, &planned, None, true).expect("apply");
        assert_eq!(stats.folded, 1);
        assert!(stats.applied.len() == 1, "报告要能逐条回填 applied_mode");
        assert_eq!(take0(&state).channel_mode, 2);
        assert!(
            state.history_depths().0 > depth_before,
            "手动扫描必须留下可撤销的一步"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn a_noop_pass_leaves_no_empty_undo_step() {
        // 撤销点是惰性的：判定说"该折叠"但 Take 已是目标模式时，写回是 no-op，
        // 不该因此多出一个空的撤销步（用户按撤销会发现"什么都没变"）。
        let policy = ChannelImportPolicy::default();
        let path = write_wav("noop_undo.wav", true, 1);
        let state = AppState::default();
        {
            let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            *tl = timeline_with_take(Some(&path));
            crate::state::TimelineState::populate_clip_file_metadata(&mut tl.clips[0]);
            // 已经是目标模式 + 档案已定论。
            tl.clips[0].takes[0].channel_mode = 2;
            tl.clips[0].channel_mode = 2;
        }
        let depth_before = state.history_depths().0;

        let targets = {
            let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
            collect_targets(&tl, None, &policy, true, true).targets
        };
        let planned = plan(targets, &policy, true);
        let stats = apply_planned(&state, &planned, None, true).expect("apply");
        assert_eq!(stats.folded, 0);
        assert_eq!(
            state.history_depths().0,
            depth_before,
            "全是 no-op 时不得留下空撤销步"
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
            collect_targets(&tl, Some(&only_first), &policy, false, true).targets.len(),
            1
        );
    }
}
