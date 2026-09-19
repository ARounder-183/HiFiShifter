//! DYN（动态）参数的原声电平基线：组装、归一化与调度。
//!
//! ## 语义
//!
//! `dyn_orig(t)` 是「本轨道组在这一时刻的原声电平」，以**倍率**表达：
//! `1.0` = 轨道组的参考电平（0 dB），`0.5` = 低于参考 6 dB。
//! 渲染增益由 `compute_dyn_gain(目标, 原声)` 求出，即用户绘制的是**绝对目标
//! 电平**（与 VocalShifter 的 DYN 同量纲）。
//!
//! ## 电平的物理口径（决定「目标 1.0」意味着什么）
//!
//! 混音链（`audio_engine/mix.rs`）per-clip 为
//! `out = raw × (vol × dyn × pan) × (clip增益 × 淡化)` 再求和 —— dyn 增益 g
//! 实际缩放的组信号是 `Σ raw_i × (clip增益_i × 淡化_i)`。要让「目标电平」被
//! 精确兑现，基线必须刻画**该信号在 dyn 增益作用点的电平**：
//!
//! ```text
//! 每 clip：analyze_clip_pitch_and_level（一次性解码 → 全量源音频逐帧 RMS）
//!      ↓ trim_and_resample_curve（窗口/Loop/倒放/rate 映射到 clip 可见区间）
//!      ↓ audible_i(t) = level_i(t) × clip增益_i   （淡化只作纳入门限，见下）
//! 根曲线：level(t) = sqrt(Σ audibleᵢ(t)²)   （能量域合成，与混音 RMS 同构）
//!      ↓ 除以参考电平（99 百分位）→ dyn_orig = clamp(level / ref, 0, 4)
//! ```
//!
//! 两条刻意的设计决策：
//! - **含静态 clip 增益**：用户听到的响度包含它；否则 clip 增益 0.5 的轨道上
//!   「画 1.0」实际只会得到参考电平的一半（目标语义失效），且虚线基线与波形
//!   显示（波形含 clip 增益）不同域、无法对齐比较。
//! - **不含淡化**：淡化是结构性的短时包络。基线不含它意味着「画平一段渐强/
//!   淡出」不会抹掉淡化本身 —— 淡化在 dyn 增益**下游**逐帧生效。淡化权重只用于
//!   判断「该帧是否可听」（= 0：前导静音 / clip 末尾之外 → 不纳入）。
//!
//! 注意：与音高融合的 `clip_weight_at_frame` **刻意不同源**。音高按
//! gain×fade 加权是"响的 clip 主导音高估计"的启发式；电平融合则是物理量合成，
//! 加权会破坏上述口径（此前把 gain/fade 当权重的实现里，单 clip 的增益在
//! 加权平均中被约掉、重叠区基线低估 √N 倍 —— 均已修正）。
//!
//! ## 数据流
//!
//! ## 与其他模块的关系
//! - 复用 `pitch_clip` 的 per-clip 缓存（`CachedClipPitch::level`），不新增解码；
//! - 复用 `pitch_analysis::schedule` 的调度入口（同一批后台任务同时产出音高与电平）；
//! - 产物写入 `TrackParamsState::dyn_orig`，由 `pitch_editing::dyn_orig_curve_for_clip`
//!   暴露给混音层，由 `commands::params` 暴露给前端画虚线基线。

use crate::state::{AppState, TimelineState};
use tauri::Emitter;

/// 参考电平的百分位（0..1）：取融合曲线中"最响的持续段落"作为 0 dB 基准。
///
/// 用百分位而非峰值，是为了让渐强/单帧尖峰不会把整条曲线压得过小；
/// 用百分位而非平均值，是为了让基准贴近"响的那一段"而不是"平均响度"
/// （后者会让曲线整体偏小，用户一画就撞上限）。
const REFERENCE_PERCENTILE: f64 = 0.99;

/// `dyn_orig` 的上限（与 DYN 参数值域一致）。
const DYN_ORIG_MAX: f32 = 4.0;

/// 组装根轨道的原声电平基线（同步，全部命中 per-clip 缓存时可用）。
///
/// 返回 `(curve, reference, all_cache_hit)`：
/// - `curve`：长度 = 工程参数帧数的 `dyn_orig`（归一化到参考电平）；
/// - `reference`：归一化用的参考电平（linear，供前端映射波形）；
/// - `all_cache_hit`：false 表示部分 clip 尚未分析完成，`curve` 是当前可得的部分。
pub(crate) fn assemble_dyn_orig_from_cache(
    tl: &TimelineState,
    root_track_id: &str,
) -> (Vec<f32>, f32, bool) {
    let fp = tl.frame_period_ms();
    let target_frames = tl.target_param_frames(fp);

    // 与音高组装完全一致的 clip 遍历顺序：下方轨道先写、上方轨道后写，
    // 同轨道按 z-order 递增 —— 保证两条曲线在重叠区间的取舍逻辑一致。
    let mut ordered_track_ids: Vec<&str> = Vec::new();
    for track in &tl.tracks {
        if tl.resolve_root_track_id(&track.id).as_deref() == Some(root_track_id) {
            ordered_track_ids.push(track.id.as_str());
        }
    }

    let mut track_clips: std::collections::HashMap<&str, Vec<&crate::state::Clip>> =
        std::collections::HashMap::new();
    for clip in &tl.clips {
        if tl.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            continue;
        }
        if clip.muted || clip.source_path.is_none() {
            continue;
        }
        track_clips
            .entry(clip.track_id.as_str())
            .or_default()
            .push(clip);
    }

    let mut ordered_clips: Vec<&crate::state::Clip> = Vec::new();
    for track_id in ordered_track_ids.iter().rev() {
        if let Some(clips) = track_clips.get(track_id) {
            ordered_clips.extend(clips.iter().copied());
        }
    }

    let mut clip_contributions: Vec<(usize, f64, Vec<f32>)> = Vec::new();
    let mut all_cache_hit = true;

    for clip in ordered_clips {
        let Some(cached) =
            crate::pitch_clip::get_clip_analysis_global(tl, clip, root_track_id, fp)
        else {
            all_cache_hit = false;
            continue;
        };
        if cached.level.is_empty() {
            continue;
        }

        let clip_len_sec = clip.length_sec.max(0.0);
        let clip_start_frame = ((clip.start_sec.max(0.0) * 1000.0) / fp).round().max(0.0) as usize;
        let clip_len_frames = ((clip_len_sec * 1000.0) / fp).round().max(0.0) as usize;
        let Some(clip_start_frame) = usize::checked_add(clip_start_frame, 0) else {
            continue;
        };
        if clip_start_frame >= target_frames {
            continue;
        }
        let write_len = clip_len_frames.min(target_frames - clip_start_frame);
        if write_len == 0 {
            continue;
        }

        // 全量源域 → clip 可见区间：与音高走同一条映射（窗口/Loop/倒放/rate）。
        let mapped = map_clip_curve(tl, clip, &cached.level, fp, clip_len_sec, clip_len_frames);
        if mapped.is_empty() {
            continue;
        }

        let media_total = crate::state::clip_source_media_duration_sec(clip);
        let pre_silence_sec = crate::state::clip_leading_silence_sec(clip, media_total);
        let clip_gain = (clip.gain.max(0.0) as f64).clamp(0.0, 4.0);

        // 淡化权重只作**纳入门限**（= 0：前导静音 / clip 末尾之外 → 不可听）；
        // 权重本身不进电平 —— 见文件头"电平的物理口径"。不可听帧贡献 0 能量。
        let mut audible_levels = Vec::with_capacity(write_len);
        for i in 0..write_len {
            let audible = match mapped.get(i) {
                Some(&v) if v.is_finite() && v > 0.0 => {
                    fade_weight_at(clip, i, fp, pre_silence_sec, clip_len_frames) > 0.0
                }
                _ => false,
            };
            audible_levels.push(if audible { mapped[i] } else { 0.0 });
        }
        clip_contributions.push((clip_start_frame, clip_gain, audible_levels));
    }

    // 能量域融合（无权重）：level(t) = sqrt(Σ (lᵢ × gainᵢ)²) —— 与混音里
    // "dyn 增益所作用的组信号的 RMS" 同构（不相关能量叠加近似）。
    let fused = fuse_level_energy(target_frames, clip_contributions);

    if !all_cache_hit && fused.iter().all(|&v| v <= 0.0) {
        // 一点数据都没有：返回全 0（= 无基线），reference 用默认 1.0 占位。
        return (vec![0.0; target_frames], 1.0, false);
    }

    let reference = percentile_reference(&fused, REFERENCE_PERCENTILE);
    if !(reference.is_finite() && reference > 0.0) {
        // 整组静音：无参考电平可归一化 → 全 0 基线（DYN 增益恒为 1）。
        return (vec![0.0; target_frames], 0.0, all_cache_hit);
    }

    let curve: Vec<f32> = fused
        .iter()
        .map(|&v| (v / reference).clamp(0.0, DYN_ORIG_MAX))
        .collect();
    (curve, reference, all_cache_hit)
}

/// 非零值的给定位百分位（用于选取参考电平）。
fn percentile_reference(curve: &[f32], percentile: f64) -> f32 {
    let mut finite: Vec<f32> = curve.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if finite.is_empty() {
        return 0.0;
    }
    finite.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((finite.len() as f64) * percentile).ceil() as usize;
    finite[idx.saturating_sub(1).min(finite.len() - 1)]
}

/// 能量域融合（**纯函数**，口径的唯一落点，供单测直接验证）：
///
/// `level(t) = sqrt(Σ (lᵢ(t) × gainᵢ)²)`
///
/// `entries` = `(clip 起始帧, clip 增益, 逐帧电平)`，不可听帧的电平传 0。
/// 两个关键性质（见文件头"电平的物理口径"）：
/// - 静态 clip 增益**乘进**电平（目标电平语义 = 可听响度）；
/// - 多 clip 重叠按**能量和**合成（无加权）—— 加权平均会把增益约掉、
///   并把重叠区基线低估 √N 倍。
fn fuse_level_energy(
    target_frames: usize,
    entries: Vec<(usize, f64, Vec<f32>)>,
) -> Vec<f32> {
    let mut energy = vec![0.0f64; target_frames];
    for (start, gain, levels) in entries {
        for (i, &level) in levels.iter().enumerate() {
            let dst = start + i;
            if dst >= target_frames || !(level > 0.0) {
                continue;
            }
            let audible = level as f64 * gain;
            energy[dst] += audible * audible;
        }
    }
    energy
        .iter()
        .map(|&e| {
            let v = e.sqrt();
            if v.is_finite() { v as f32 } else { 0.0 }
        })
        .collect()
}

/// 把全量源域的曲线映射到 clip 可见区间（帧域，长度 ≈ clip_len_frames）。
///
/// 与音高组装逐分支同构：Loop 走回绕、非 Loop 倒放走重定向窗口 + 翻转、
/// rate≈1 走窗口映射、其余走截取 + 重采样。音高侧的写法散落在 schedule.rs
/// 的 `assemble_pitch_orig_from_cache` 内联分支里；这里抽成一个函数是因为
/// 电平曲线需要完全相同的几何处理，但**不需要**音高特有的"0 = 无声"语义。
fn map_clip_curve(
    tl: &TimelineState,
    clip: &crate::state::Clip,
    full: &[f32],
    fp: f64,
    clip_len_sec: f64,
    clip_len_frames: usize,
) -> Vec<f32> {
    let pr = clip.playback_rate as f64;
    let pr_valid = if pr.is_finite() && pr > 0.0 { pr } else { 1.0 };
    let media_total = crate::state::clip_source_media_duration_sec(clip);

    // rate ≈ 1 且非 Loop：走带静音的窗口映射，与音频逐帧一致
    // （含 slip 前导静音 / 窗口越界），不走重采样以免引入插值涂抹。
    let rate_near_one = (pr_valid - 1.0).abs() <= 0.01;
    if rate_near_one && !clip.loop_enabled {
        let (win_start_sec, win_end_sec) = crate::state::clip_pitch_trim_window_sec(clip);
        let mut mapped = crate::pitch_clip::assemble_nonloop_pitch_from_window(
            full,
            fp,
            win_start_sec,
            win_end_sec,
            pr_valid,
            clip_len_frames,
        );
        if clip.reversed {
            mapped.reverse();
        }
        return mapped;
    }

    let (trim_src_start, trim_src_end) = crate::state::clip_pitch_trim_window_sec(clip);
    let mut resampled = crate::pitch_clip::trim_and_resample_curve(
        full,
        fp,
        trim_src_start,
        trim_src_end,
        pr_valid,
        clip_len_sec,
        clip.loop_enabled,
        media_total,
        clip.reversed && clip.loop_enabled,
    );
    // 非 Loop 倒放：升序窗口映射 + 输出整体翻转（与音高、mixdown 同一约定）。
    if clip.reversed && !clip.loop_enabled {
        resampled.reverse();
    }
    let _ = tl;
    resampled
}

/// 淡化包络在某 clip 内第 `i` 帧的权重（0..1）。
///
/// 与 `pitch_analysis::analysis::clip_weight_at_frame` 的淡化部分同一公式，
/// 但以帧为单位、不含 clip gain（gain 由调用方单独乘入）—— 电平融合与音高
/// 融合对"哪些帧算数"的判断必须一致，否则两条曲线在交叉淡化区会错位。
fn fade_weight_at(
    clip: &crate::state::Clip,
    i: usize,
    fp: f64,
    pre_silence_sec: f64,
    clip_total_frames: usize,
) -> f64 {
    let fps = 1000.0 / fp.max(0.1);
    let pre_silence_frames = (pre_silence_sec * fps).round().max(0.0) as usize;
    let local_in_clip = pre_silence_frames.saturating_add(i);
    if local_in_clip >= clip_total_frames {
        return 0.0;
    }
    // 淡出之前/前导静音期不计入。
    if local_in_clip < pre_silence_frames {
        return 0.0;
    }

    let mut g = 1.0f64;
    let fade_in_frames = (clip.effective_fade_in_sec().max(0.0) * fps).round().max(0.0);
    if fade_in_frames > 0.0 && (local_in_clip as f64) < fade_in_frames {
        g *= (local_in_clip as f64 / fade_in_frames).clamp(0.0, 1.0);
    }
    let fade_out_frames = (clip.effective_fade_out_sec().max(0.0) * fps).round().max(0.0);
    if fade_out_frames > 0.0 && (local_in_clip as f64) + fade_out_frames > clip_total_frames as f64 {
        let remain = clip_total_frames.saturating_sub(local_in_clip);
        g *= (remain as f64 / fade_out_frames).clamp(0.0, 1.0);
    }
    g
}

/// 组装并写回基线（共享实现），返回 `(changed, all_cache_hit)`。
///
/// 必须持有 timeline 写锁调用；两个入口（命令侧 / 引擎侧）共用它，
/// 保证"什么算命中、什么时候记 key"只有一处定义。
fn assemble_and_store(tl: &mut TimelineState, root_track_id: &str) -> (bool, bool) {
    let key = build_root_dyn_key(tl, root_track_id);
    let fp = tl.frame_period_ms();
    let target = tl.target_param_frames(fp);

    // key 命中且长度正确 → 无事可做。
    let up_to_date = tl
        .params_by_root_track
        .get(root_track_id)
        .map(|e| e.dyn_orig_key.as_deref() == Some(key.as_str()) && e.dyn_orig.len() == target)
        .unwrap_or(false);
    if up_to_date {
        return (false, true);
    }

    let (curve, reference, all_cache_hit) = assemble_dyn_orig_from_cache(tl, root_track_id);
    tl.ensure_params_for_root(root_track_id);
    let mut changed = false;
    if let Some(entry) = tl.params_by_root_track.get_mut(root_track_id) {
        changed = entry.dyn_orig != curve;
        entry.dyn_orig = curve;
        entry.dyn_orig_reference = reference;
        // 全部命中才记 key（与音高同一策略：部分命中时下一轮继续尝试组装）。
        entry.dyn_orig_key = all_cache_hit.then(|| key.clone());
    }
    (changed, all_cache_hit)
}

/// 从 per-clip 缓存组装根轨道的原声电平基线，并写回 `TrackParamsState`。
///
/// 与 `maybe_schedule_pitch_orig` 的差别：
/// - **不受 `compose_enabled` 限制** —— 动态是混音级参数，未开 Compose 也要生效；
/// - 只写 `dyn_orig` 系列字段，不触碰音高曲线；
/// - 曲线内容变化时 `emit("dyn_orig_updated")`（前端据此重取并重画虚线基线），
///   同时通过 `update_timeline` 让快照里的 `dyn_orig_curve` 立即换新。
///
/// **分析调度不在这里**：真正的后台分析由引擎的 `handle_update_timeline`
/// 经 `schedule_clip_pitch_jobs` 提交（该函数需要 worker 侧的 sender），
/// 完成后推送 `ClipPitchReady` 触发快照重建，本函数会被再次调用。
///
/// 返回 `true` 表示分析仍未完成（存在未命中的 clip 缓存）。
pub fn maybe_schedule_dyn_orig(state: &AppState, root_track_id: &str) -> bool {
    let (snapshot_source, all_cache_hit) = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let (changed, all_hit) = assemble_and_store(&mut tl, root_track_id);
        (changed.then(|| tl.clone()), all_hit)
    };

    if let Some(snapshot_source) = snapshot_source {
        // 基线变了 → 让快照里的 dyn_orig 曲线同步换新（实时播放立刻用新基线）。
        state.audio_engine.update_timeline(snapshot_source);
        if let Some(app) = state.app_handle.get() {
            let _ = app.emit(
                "dyn_orig_updated",
                crate::pitch_analysis::PitchOrigUpdatedEvent {
                    root_track_id: root_track_id.to_string(),
                },
            );
        }
    }

    // 缓存未全部命中 → 提交后台分析（worker 侧持有 sender）。
    //
    // 此前这里只做"组装"、不提交任务，于是首次打开动态面板时永远等不到
    // 分析结果（基线恒为占位值）。必须在锁外发起：命令层的调用点持有着
    // timeline 锁，`std::sync::Mutex` 不可重入。
    if !all_cache_hit {
        state
            .audio_engine
            .request_dyn_level_analysis();
    }

    !all_cache_hit
}

/// 引擎侧入口：在 `handle_update_timeline` 里组装基线，并在缓存未命中时
/// 提交后台分析（需要 worker 侧的 sender，因此与命令侧入口分开）。
///
/// 直接改传进来的 timeline（调用方随后 store 进 `last_timeline`），
/// 因此本次 snapshot 就会带上新基线 —— 无需再走一次 update 往返。
pub fn assemble_dyn_orig_for_engine(tl: &mut TimelineState, root_track_id: &str) -> bool {
    let (_changed, all_cache_hit) = assemble_and_store(tl, root_track_id);
    all_cache_hit
}

/// 原声基线（DYN）的缓存键。
///
/// 语义与 `build_root_pitch_key` 一致：任何会改变"逐帧电平"的时间线变化
/// （clip 几何、源文件签名、淡化、增益…）都必须产出一个不同的 key。
pub(crate) fn build_root_dyn_key(tl: &TimelineState, root_track_id: &str) -> String {
    let fp = tl.frame_period_ms().max(0.1);
    let mut hasher = blake3::Hasher::new();
    // 【v2】融合口径变更：clip 增益乘进电平、淡化只作纳入门限（见文件头）。
    // 旧 key 命中的基线是旧口径产物，必须整体重组一次。
    hasher.update(b"root_dyn_orig_v2");
    hasher.update(root_track_id.as_bytes());
    hasher.update(&crate::pitch_analysis::quantize_u32(fp, 1000.0).to_le_bytes());

    for clip in &tl.clips {
        if tl.resolve_root_track_id(&clip.track_id).as_deref() != Some(root_track_id) {
            continue;
        }
        hasher.update(clip.id.as_bytes());
        hasher.update(&[u8::from(clip.muted)]);
        hasher.update(&crate::pitch_analysis::quantize_i64(clip.start_sec, 1000.0).to_le_bytes());
        hasher.update(&crate::pitch_analysis::quantize_i64(clip.length_sec, 1000.0).to_le_bytes());
        hasher.update(
            &crate::pitch_analysis::quantize_i64(clip.source_start_sec, 1000.0).to_le_bytes(),
        );
        hasher.update(&crate::pitch_analysis::quantize_i64(clip.source_end_sec, 1000.0).to_le_bytes());
        hasher.update(&crate::pitch_analysis::quantize_u32(clip.playback_rate as f64, 1000.0).to_le_bytes());
        hasher.update(&crate::pitch_analysis::quantize_u32(clip.gain as f64, 1000.0).to_le_bytes());
        hasher.update(&[u8::from(clip.reversed), u8::from(clip.loop_enabled)]);
        hasher.update(
            &crate::pitch_analysis::quantize_u32(clip.effective_fade_in_sec(), 1000.0).to_le_bytes(),
        );
        hasher.update(
            &crate::pitch_analysis::quantize_u32(clip.effective_fade_out_sec(), 1000.0)
                .to_le_bytes(),
        );
        if let Some(path) = clip.source_path.as_deref() {
            hasher.update(path.as_bytes());
            let (len, mtime) = crate::pitch_analysis::file_sig(std::path::Path::new(path));
            hasher.update(&len.to_le_bytes());
            hasher.update(&mtime.to_le_bytes());
        }
    }

    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{Clip, TimelineState};

    /// 最小可用的 Clip（与 `pitch_editing::tests::make_clip` 逐字段同口径；
    /// `Clip` 未实现 `Default`，必须完整构造）。
    fn make_clip(id: &str, start: f64, len: f64) -> Clip {
        Clip {
            id: id.to_string(),
            takes: vec![],
            active_take_id: None,
            clip_playback_rate: 1.0,
            track_id: "root".to_string(),
            name: id.to_string(),
            start_sec: start,
            length_sec: len,
            color: "blue".to_string(),
            source_path: Some("a.wav".to_string()),
            source_path_relative: None,
            duration_sec: Some(len),
            duration_frames: None,
            source_sample_rate: Some(44_100),
            source_file_mtime: None,
            source_file_size: None,
            source_file_fingerprint: None,
            waveform_preview: None,
            pitch_range: None,
            gain: 1.0,
            muted: false,
            source_start_sec: 0.0,
            source_end_sec: len,
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
        }
    }

    /// 最小可用的 Track（`Track` 未实现 `Default`；借用 `add_track` 的默认值）。
    fn make_track(id: &str) -> crate::state::Track {
        let mut tl = TimelineState::default();
        tl.add_track(Some(id.to_string()), None, None);
        tl.tracks.into_iter().next().expect("track just added")
    }

    #[test]
    fn percentile_reference_uses_upper_percentile() {
        // 10 个值：最响的 10% 是 1.0。99 百分位取到接近最大但不受单点尖峰影响。
        let curve: Vec<f32> = vec![0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 1.0];
        let r = percentile_reference(&curve, 0.99);
        assert!((r - 1.0).abs() < 1e-6, "got {r}");
        // 单帧尖峰不会把参考抬到极点：99 百分位仍落在次高段。
        let curve: Vec<f32> = (0..100).map(|i| if i == 99 { 100.0 } else { 0.5 }).collect();
        let r = percentile_reference(&curve, 0.99);
        assert!((r - 0.5).abs() < 1e-6, "got {r}");
    }

    #[test]
    fn percentile_reference_handles_empty_and_silent() {
        assert_eq!(percentile_reference(&[], 0.99), 0.0);
        assert_eq!(percentile_reference(&[0.0, 0.0, 0.0], 0.99), 0.0);
    }

    #[test]
    fn dyn_key_changes_with_clip_geometry() {
        let mut tl = TimelineState::default();
        let track = make_track("root");
        tl.tracks = vec![track];
        tl.clips.push(make_clip("c1", 0.0, 1.0));
        let k1 = build_root_dyn_key(&tl, "root");

        // 仅移动位置也应改变键（窗口映射随之变化）。
        tl.clips[0].start_sec = 2.0;
        let k2 = build_root_dyn_key(&tl, "root");
        assert_ne!(k1, k2);

        // 增益变化同样改变键（融合权重依赖它）。
        tl.clips[0].gain = 0.5;
        let k3 = build_root_dyn_key(&tl, "root");
        assert_ne!(k2, k3);

        // 无变化时稳定。
        assert_eq!(k3, build_root_dyn_key(&tl, "root"));
    }

    /// 能量域融合的口径守护（文件头"电平的物理口径"的逐条断言）。
    ///
    /// 单 clip：基线 = 电平 × clip 增益（静态增益必须兑现到基线里 ——
    /// 旧实现把它当权重、被加权平均约掉，clip 增益 0.5 的轨道上"画 1.0"
    /// 只得到参考电平的一半）。
    #[test]
    fn fuse_energy_includes_clip_gain_and_overlaps_as_energy_sum() {
        use super::fuse_level_energy;
        // 单 clip，电平恒 0.5、增益 0.25 → fused = 0.125（不是 0.5）。
        let fused = fuse_level_energy(
            4,
            vec![(0usize, 0.25f64, vec![0.5f32, 0.5, 0.5, 0.5])],
        );
        for v in &fused {
            assert!((v - 0.125).abs() < 1e-6, "got {v}");
        }

        // 两个等电平 clip 完全重叠 → sqrt(2) × l × g（不是加权平均的 l×g）。
        let fused = fuse_level_energy(
            4,
            vec![
                (0usize, 1.0f64, vec![0.5f32, 0.5, 0.5, 0.5]),
                (0usize, 1.0f64, vec![0.5f32, 0.5, 0.5, 0.5]),
            ],
        );
        for v in &fused {
            assert!((v - (0.5f32 * std::f64::consts::SQRT_2 as f32)).abs() < 1e-6, "got {v}");
        }

        // 相邻不重叠：各自贡献自己的能量。
        let fused = fuse_level_energy(
            4,
            vec![
                (0usize, 1.0f64, vec![0.5f32, 0.5]),
                (2usize, 1.0f64, vec![1.0f32, 1.0]),
            ],
        );
        assert!((fused[0] - 0.5).abs() < 1e-6);
        assert!((fused[2] - 1.0).abs() < 1e-6);
        assert_eq!(fused.len(), 4);
    }
}
