use crate::state::AppState;
use tauri::State;

// ===================== param curves =====================

const CHILD_PITCH_OFFSET_CENTS_PREFIX: &str = "child_pitch_offset_cents@";
const CHILD_PITCH_OFFSET_DEGREES_PREFIX: &str = "child_pitch_offset_degrees@";
const CHILD_FORMANT_OFFSET_CENTS_PREFIX: &str = "child_formant_offset_cents@";
const CHILD_PITCH_OFFSET_CENTS_DEFAULT: f32 = 0.0;
const CHILD_PITCH_OFFSET_DEGREES_INTERNAL_DEFAULT: f32 = 0.0;
const CHILD_FORMANT_OFFSET_CENTS_DEFAULT: f32 = 0.0;
const CHILD_FORMANT_OFFSET_CENTS_RANGE: (f32, f32) = (-2400.0, 2400.0);

#[derive(Clone, Copy)]
enum ChildPitchOffsetParamMode {
    Cents,
    Degrees,
    Formant,
}

#[derive(Clone, Copy)]
struct ChildPitchOffsetParamSpec<'a> {
    mode: ChildPitchOffsetParamMode,
    track_id: &'a str,
}

fn parse_child_pitch_offset_param(param: &str) -> Option<ChildPitchOffsetParamSpec<'_>> {
    if let Some(track_id) = param.strip_prefix(CHILD_PITCH_OFFSET_CENTS_PREFIX) {
        if !track_id.is_empty() {
            return Some(ChildPitchOffsetParamSpec {
                mode: ChildPitchOffsetParamMode::Cents,
                track_id,
            });
        }
    }
    if let Some(track_id) = param.strip_prefix(CHILD_PITCH_OFFSET_DEGREES_PREFIX) {
        if !track_id.is_empty() {
            return Some(ChildPitchOffsetParamSpec {
                mode: ChildPitchOffsetParamMode::Degrees,
                track_id,
            });
        }
    }
    if let Some(track_id) = param.strip_prefix(CHILD_FORMANT_OFFSET_CENTS_PREFIX) {
        if !track_id.is_empty() {
            return Some(ChildPitchOffsetParamSpec {
                mode: ChildPitchOffsetParamMode::Formant,
                track_id,
            });
        }
    }
    None
}

fn resolve_child_pitch_offset_curve_default_value(
    timeline: &crate::state::TimelineState,
    param: &str,
) -> Option<f32> {
    let spec = parse_child_pitch_offset_param(param)?;
    let track = timeline
        .tracks
        .iter()
        .find(|track| track.id == spec.track_id)?;
    if track.parent_id.is_none() {
        return None;
    }

    match spec.mode {
        ChildPitchOffsetParamMode::Cents => Some(CHILD_PITCH_OFFSET_CENTS_DEFAULT),
        ChildPitchOffsetParamMode::Degrees => Some(CHILD_PITCH_OFFSET_DEGREES_INTERNAL_DEFAULT),
        ChildPitchOffsetParamMode::Formant => Some(CHILD_FORMANT_OFFSET_CENTS_DEFAULT),
    }
}

fn invalidate_rendered_clip_caches_for_child_track(
    timeline: &crate::state::TimelineState,
    child_track_id: &str,
) {
    let clip_ids: Vec<String> = timeline
        .clips
        .iter()
        .filter(|clip| clip.track_id == child_track_id)
        .map(|clip| clip.id.clone())
        .collect();

    if clip_ids.is_empty() {
        return;
    }

    {
        let mut cache = crate::synth_clip_cache::global_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for clip_id in &clip_ids {
            cache.invalidate(clip_id);
        }
    }
    {
        let mut tension_cache = crate::synth_clip_cache::global_tension_rendered_clip_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for clip_id in &clip_ids {
            tension_cache.invalidate(clip_id);
        }
    }
    {
        let mut noise_cache = crate::synth_clip_cache::global_breath_noise_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        for clip_id in &clip_ids {
            noise_cache.invalidate(clip_id);
        }
    }
    for clip_id in &clip_ids {
        crate::synth_clip_cache::remove_pending_rendered_key(clip_id);
    }
}

pub(super) fn resolve_extra_curve_default_value(
    kind: crate::state::SynthPipelineKind,
    param: &str,
) -> f32 {
    crate::renderer::automation_curve_default_value(kind, param).unwrap_or(0.0)
}

/// 参数在「无数据」时该填什么（用于区间填充、平移补位、前端请求越界帧）。
///
/// 与 `resolve_extra_curve_default_value` 的唯一差别是 `dyn`：它的描述符默认值
/// 1.0 表示「压平到参考电平」（一个真实动作），而「无数据」应表达为沿用原声。
/// 详见 `common_params::automation_curve_pad_value`。
fn resolve_extra_curve_pad_value(kind: crate::state::SynthPipelineKind, param: &str) -> f32 {
    crate::renderer::common_params::automation_curve_pad_value(kind, param)
}

fn resolve_param_reference_value(kind: crate::state::SynthPipelineKind, param: &str) -> f32 {
    match param {
        "pitch" => 0.0,
        "tension" => 0.0,
        // 写路径的参考值 = 填充值：对 `dyn` 是「沿用原声」哨兵 (−1)，而不是
        // 描述符默认值 1.0（那会把无数据帧解释成"压平到参考电平"）。
        _ => resolve_extra_curve_pad_value(kind, param),
    }
}

fn resolve_param_reference_value_with_timeline(
    timeline: &crate::state::TimelineState,
    kind: crate::state::SynthPipelineKind,
    param: &str,
) -> f32 {
    resolve_child_pitch_offset_curve_default_value(timeline, param)
        .unwrap_or_else(|| resolve_param_reference_value(kind, param))
}

fn resolve_param_reference_kind(param: &str) -> crate::models::ParamReferenceKind {
    match param {
        "pitch" | "dyn" => crate::models::ParamReferenceKind::SourceCurve,
        _ => crate::models::ParamReferenceKind::DefaultValue,
    }
}

fn resolve_extra_curve_frame_pair(
    curve: Option<&[f32]>,
    default_value: f32,
    idx: usize,
) -> (f32, f32) {
    let edit_value = curve
        .and_then(|values| values.get(idx))
        .copied()
        .unwrap_or(default_value);
    (default_value, edit_value)
}

fn resolve_static_param_default_value(kind: crate::state::SynthPipelineKind, param: &str) -> f64 {
    crate::renderer::static_enum_default_value(kind, param)
        .map(|value| value as f64)
        .unwrap_or(0.0)
}

pub(super) fn get_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    stride: Option<u32>,
    binary: Option<bool>,
    with_sentinel: Option<bool>,
) -> crate::models::ParamFramesPayload {
    if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
        log::warn!(
            "get_param_frames(track_id={}, param={}, start_frame={}, frame_count={}, stride={:?} binary={:?})",
            track_id, param, start_frame, frame_count, stride, binary
        );
    }
    // 二进制模式：orig/edit 以 Base64 单条返回，JSON 里不再展开成 number[]。
    let binary = binary.unwrap_or(false);
    let (root, fp, entry, compose_enabled, pitch_algo, param_reference_value, param_kind) = {
        let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

        let root = match tl.resolve_root_track_id(&track_id) {
            Some(id) => id,
            None => {
                return crate::models::ParamFramesPayload {
                    ok: false,
                    root_track_id: "".to_string(),
                    param,
                    frame_period_ms: tl.frame_period_ms(),
                    start_frame,
                    orig: vec![],
                    edit: vec![],
                    binary: None,
                    reference_kind: resolve_param_reference_kind("pitch"),
                    analysis_pending: None,
                    analysis_progress: None,
                    pitch_edit_user_modified: None,
                    pitch_edit_backend_available: None,
                    edit_sentinel: None,
                }
            }
        };

        tl.ensure_params_for_root(&root);
        let fp = tl.frame_period_ms();

        let track = tl.tracks.iter().find(|t| t.id == root);
        let compose_enabled = track.map(|t| t.compose_enabled).unwrap_or(false);
        let pitch_algo = track
            .map(|t| t.pitch_analysis_algo.clone())
            .unwrap_or_default();
        let kind = crate::state::SynthPipelineKind::from_track_algo(&pitch_algo);

        let entry = tl
            .params_by_root_track
            .get(&root)
            .cloned()
            .unwrap_or_default();

        let param_reference_value = resolve_param_reference_value_with_timeline(&tl, kind, &param);

        (
            root,
            fp,
            entry,
            compose_enabled,
            pitch_algo,
            param_reference_value,
            kind,
        )
    };

    let pitch_edit_user_modified = (param == "pitch").then_some(entry.pitch_edit_user_modified);

    let pitch_edit_backend_available = if param == "pitch" {
        let algo = crate::pitch_editing::PitchEditAlgorithm::from_track_algo(&pitch_algo);
        let available = match algo {
            crate::pitch_editing::PitchEditAlgorithm::WorldVocoder => {
                crate::world_vocoder::is_available()
            }
            crate::pitch_editing::PitchEditAlgorithm::NsfHifiganOnnx => {
                crate::nsf_hifigan_onnx::is_available()
            }
            #[cfg(feature = "vslib")]
            crate::pitch_editing::PitchEditAlgorithm::VocalShifterVslib => true,
            crate::pitch_editing::PitchEditAlgorithm::Bypass => true,
        };
        Some(available)
    } else {
        None
    };

    if param == "pitch" && !compose_enabled {
        return crate::models::ParamFramesPayload {
            ok: true,
            root_track_id: root,
            param,
            frame_period_ms: fp,
            start_frame,
            orig: vec![],
            edit: vec![],
            binary: None,
            reference_kind: resolve_param_reference_kind("pitch"),
            analysis_pending: None,
            analysis_progress: None,
            pitch_edit_user_modified,
            pitch_edit_backend_available,
            edit_sentinel: None,
        };
    }

    // Schedule pitch_orig analysis in background; return current cached curve immediately.
    let analysis_pending = if param == "pitch" {
        Some(crate::pitch_analysis::maybe_schedule_pitch_orig(
            &state, &root,
        ))
    } else {
        None
    };

    // 「动态」：先把原声电平基线组装进 state（同步路径只读已算好的 per-clip 缓存），
    // 再按普通 extra 曲线的口径读取 —— 下面是复制一份，因为 curve 需要重新读取。
    //
    // 前端请求 dyn 参数 == 动态面板正在显示这个轨道组。这里顺带登记该状态，
    // 使"面板已打开但用户还没落笔"时后台也会去算原声电平（虚线基线与 dB
    // 波形才能显示），并且**不依赖 `compose_enabled`** —— 动态是混音级参数。
    let dyn_analysis_pending = if param == "dyn" {
        if let Ok(mut roots) = crate::pitch_clip::dyn_panel_open_roots().lock() {
            roots.insert(root.clone());
        }
        Some(crate::pitch_analysis::maybe_schedule_dyn_orig(&state, &root))
    } else {
        None
    };
    let (entry, param_reference_value) = if param == "dyn" {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        let e = tl
            .params_by_root_track
            .get(&root)
            .cloned()
            .unwrap_or_default();
        (e, param_reference_value)
    } else {
        (entry, param_reference_value)
    };

    let start = start_frame as usize;
    let count = (frame_count as usize).max(1);
    let step = (stride.unwrap_or(1).max(1)) as usize;

    let mut orig = Vec::with_capacity(count);
    let mut edit = Vec::with_capacity(count);
    // dyn 的"未画"位图（仅 with_sentinel 时填充；其他参数恒 None）。
    let mut edit_sentinel: Option<Vec<bool>> = None;

    match param.as_str() {
        "pitch" => {
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let o = entry.pitch_orig.get(idx).copied().unwrap_or(0.0);
                let e_raw = entry.pitch_edit.get(idx).copied().unwrap_or(0.0);
                // Treat 0 as "unset" and fall back to orig.
                let e = if e_raw == 0.0 && o != 0.0 { o } else { e_raw };
                orig.push(o);
                edit.push(e);
            }
        }
        "tension" => {
            let reference_value = param_reference_value;
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let e = entry
                    .tension_edit
                    .get(idx)
                    .copied()
                    .unwrap_or(reference_value);
                orig.push(reference_value);
                edit.push(e);
            }
        }
        "dyn" => {
            // 动态（DYN）：`orig` = 原声电平基线（虚线），`edit` = 用户目标电平。
            //
            // 曲线里的 `DYN_FOLLOW_ORIG`(−1) 哨兵在**出口**解析成真实基线值：
            // 前端永远看不到负值，绘制/平滑/量化全都按普通 0..4 曲线处理。
            // 分析未就绪时两条都退化为描述符默认值 1.0（此时增益恒为 1）。
            //
            // 【哨兵位图】批量操作（平滑/量化/平均/拖拽提交/复制粘贴）走
            // "读-变换-写回"，若拿不到哨兵信息就会把**未画帧**物化成显式基线值
            // ——当下听感不变，但日后基线重分析时这些帧不再跟随（响度漂移）。
            // `with_sentinel` 请求方据此拿到逐帧"该帧是否未画"的位图，写回时
            // 把哨兵帧原样写回哨兵（pitch 的 0 哨兵在拖拽里已有同款约定）。
            let orig_curve = entry.dyn_orig.as_slice();
            let user_curve = entry.extra_curves.get("dyn").map(|v| v.as_slice());
            let fallback = resolve_extra_curve_default_value(param_kind, "dyn");
            let want_sentinel = with_sentinel.unwrap_or(false);
            let mut sentinels: Vec<bool> = Vec::new();
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let baseline = orig_curve
                    .get(idx)
                    .copied()
                    .filter(|v| v.is_finite() && *v > 0.0)
                    .unwrap_or(fallback);
                let user = user_curve.and_then(|c| c.get(idx)).copied();
                let is_unset = match user {
                    Some(v) => !(v.is_finite() && v >= 0.0), // 哨兵 / 非有限
                    None => true,                            // 曲线不存在或更短
                };
                if want_sentinel {
                    sentinels.push(is_unset);
                }
                let resolved = match user {
                    Some(v) if v.is_finite() && v >= 0.0 => v,
                    _ => baseline, // 哨兵 / 缺失 → 沿用原声
                };
                orig.push(baseline);
                edit.push(resolved);
            }
            if want_sentinel {
                edit_sentinel = Some(sentinels);
            }
        }
        _ => {
            // Extra automation curve: dashed orig should stay at the processor default,
            // while solid edit reflects the user-authored curve.
            let curve = entry.extra_curves.get(&param).map(|v| v.as_slice());
            let default_value = param_reference_value;
            for i in 0..count {
                let idx = start.saturating_add(i.saturating_mul(step));
                let (o, e) = resolve_extra_curve_frame_pair(curve, default_value, idx);
                orig.push(o);
                edit.push(e);
            }
        }
    }

    // dyn 的分析待定标志：与 pitch 共用 `analysis_pending` 字段（前端只关心
    // "这个参数是否还在后台分析"，语义一致）。
    let analysis_pending = analysis_pending.or(dyn_analysis_pending);

    crate::models::ParamFramesPayload {
        ok: true,
        root_track_id: root,
        param: param.clone(),
        frame_period_ms: fp,
        start_frame,
        binary: binary.then(|| encode_param_frames_binary(&orig, &edit)),
        orig: if binary { Vec::new() } else { orig },
        edit: if binary { Vec::new() } else { edit },
        reference_kind: resolve_param_reference_kind(&param),
        analysis_pending,
        analysis_progress: None,
        pitch_edit_user_modified,
        pitch_edit_backend_available,
        edit_sentinel,
        // 前端画 DYN 波形需要参考电平（把线性峰值换算成同样的倍率域）。
    }
}
/// 将 orig/edit 两组 f32 曲线编码为 Base64 二进制。
///
/// 协议：`[Header 8B][orig f32[count]][edit f32[count]]`，小端序。
///   - Header: magic `"PFB1"`（4B）+ count（u32 LE，两组长度相同）
///
/// 注意布局是**平面**（先整个 orig、再整个 edit），而不是 orig/edit 交错。
/// 平面布局下前端可以用 `new Float32Array(buffer, offset, count)` 直接建零拷贝
/// 视图，无需逐元素反交错。与前端 `paramFramesBinaryCodec.ts` 配套，
/// 改动任一侧必须同步另一侧。
fn encode_param_frames_binary(orig: &[f32], edit: &[f32]) -> String {
    use base64::Engine as _;

    debug_assert_eq!(orig.len(), edit.len(), "orig/edit length mismatch");
    let count = orig.len().min(edit.len());
    let orig = &orig[..count];
    let edit = &edit[..count];

    let mut bytes = Vec::with_capacity(8 + count * 8);
    bytes.extend_from_slice(b"PFB1");
    bytes.extend_from_slice(&(count as u32).to_le_bytes());
    for v in orig {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    for v in edit {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}


pub(super) fn set_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    values: Vec<f32>,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ParamCurve);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);
    let param_reference_value = resolve_param_reference_value_with_timeline(&tl, kind, &param);

    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    // For extra_curves we need separate handling; batch into known vs extra below.
    let is_extra_curve = !matches!(param.as_str(), "pitch" | "tension");
    if is_extra_curve {
        // Ensure the curve vector exists and is long enough.
        let curve = entry
            .extra_curves
            .entry(param.clone())
            .or_insert_with(Vec::new);
        let needed = start_frame as usize + values.len();
        let default_value = param_reference_value;
        if curve.len() < needed {
            curve.resize(needed, default_value);
        }
    }

    let dst = match param.as_str() {
        "pitch" => &mut entry.pitch_edit,
        "tension" => &mut entry.tension_edit,
        _ => {
            // Safety: we ensured extra_curves[&param] exists above.
            entry.extra_curves.get_mut(&param).unwrap()
        }
    };

    let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
    let extra_curve_default = is_extra_curve
        .then_some(param_reference_value)
        .unwrap_or(0.0);

    let start = start_frame as usize;
    let mut written = 0usize;
    let mut non_finite = 0usize;
    let mut clamped = 0usize;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut max_delta = 0.0f32;
    let mut prev_v: Option<f32> = None;
    for (i, v) in values.into_iter().enumerate() {
        let idx = start.saturating_add(i);
        if idx >= dst.len() {
            break;
        }

        let mut v = if v.is_finite() {
            v
        } else {
            non_finite += 1;
            if is_extra_curve {
                extra_curve_default
            } else {
                0.0
            }
        };

        if let Some(spec) = parse_child_pitch_offset_param(&param) {
            if matches!(spec.mode, ChildPitchOffsetParamMode::Formant) {
                let vv = v.clamp(
                    CHILD_FORMANT_OFFSET_CENTS_RANGE.0,
                    CHILD_FORMANT_OFFSET_CENTS_RANGE.1,
                );
                if vv != v {
                    clamped += 1;
                }
                v = vv;
            }
        }

        match param.as_str() {
            "pitch" => {
                // MIDI pitch. Keep 0 as "unset"; otherwise clamp into a reasonable range.
                if v != 0.0 {
                    let vv = v.clamp(1.0, 127.0);
                    if vv != v {
                        clamped += 1;
                    }
                    v = vv;
                }
            }
            "tension" => {
                // Tension is a UI parameter in [-100, 100].
                let vv = v.clamp(-100.0, 100.0);
                if vv != v {
                    clamped += 1;
                }
                v = vv;
            }
            "volume" | "hifigan_volume" => {
                // 音量：乘性增益，钳到描述符值域 0..2（±6 dB）。负值对音量无意义，
                // 一并钳到 0（全静音）。
                let vv = v.max(0.0).min(2.0);
                if vv != v {
                    clamped += 1;
                }
                v = vv;
            }
            "dyn" => {
                // 动态：正值是目标电平（0..1 倍率，见 common_params::DYN_PARAM
                // 的值域说明）；负值统一收敛到「沿用原声」哨兵 —— **不允许**前端
                // 把哨兵写成 −0.7 之类的中间值，只要符号为负就是同一个语义，
                // 值本身无意义。
                //
                // 上界必须与描述符值域**逐字一致**（0..1）：写路径曾独立钳到 2.0，
                // 于是前端能画出 1.0 以上的目标，而描述符、互转、轴刻度全都按
                // 0..1 处理 —— 同一份曲线在不同路径下含义不同。边界只有描述符
                // 一个真源，这里从它读。
                let vv = if v < 0.0 {
                    crate::renderer::common_params::DYN_FOLLOW_ORIG
                } else {
                    v.clamp(0.0, crate::renderer::common_params::DYN_VALUE_MAX)
                };
                if vv != v {
                    clamped += 1;
                }
                v = vv;
            }
            _ => {}
        }

        min_v = min_v.min(v);
        max_v = max_v.max(v);
        if let Some(p) = prev_v {
            max_delta = max_delta.max((v - p).abs());
        }
        prev_v = Some(v);

        dst[idx] = v;
        written += 1;
    }

    if debug {
        // This helps diagnose whether the frontend is sending invalid / extreme curves.
        log::warn!(
            "set_param_frames(param={param}, start_frame={start_frame}, len={}): non_finite={non_finite} clamped={clamped} min={min_v:.3} max={max_v:.3} max_delta={max_delta:.3}",
            written
        );
    }

    if param == "pitch" {
        entry.pitch_edit_user_modified = true;
    }

    // 写入动态后基线的组装前提可能刚成立（例如用户第一次画了 dyn），
    // 主动触发一次组装/调度，让用户画完就能听到正确的（而非 1.0 占位）结果。
    let dyn_touched = param == "dyn";
    let root_for_dyn = root.clone();

    if let Some(spec) = parse_child_pitch_offset_param(&param) {
        invalidate_rendered_clip_caches_for_child_track(&tl, spec.track_id);
    }

    // Ensure realtime playback reflects edits immediately.
    state.audio_engine.update_timeline(tl.clone());

    // ★ 必须先释放上面的 timeline 锁，再调用下面这条会**再次加锁**的路径。
    //
    // `std::sync::Mutex` 不可重入：`entry` 借自 `tl`，因此本函数直到 return
    // 都持有守卫；而 `maybe_schedule_dyn_orig` 内部会 `state.timeline.lock()`。
    // 在持锁状态下调用它 = 同线程二次加锁 = 永久挂死（用户观感即"画完动态线
    // 就崩溃"）。这也是为什么必须在锁外调用，而不是把 root 传进去。
    drop(tl);

    if dyn_touched {
        let _ = crate::pitch_analysis::maybe_schedule_dyn_orig(&state, &root_for_dyn);
    }

    serde_json::json!({"ok": true})
}

pub(super) fn restore_param_frames(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    start_frame: u32,
    frame_count: u32,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ParamRestore);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);
    let param_reference_value = resolve_param_reference_value_with_timeline(&tl, kind, &param);
    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    let start = start_frame as usize;
    let count = (frame_count as usize).max(1);

    match param.as_str() {
        "pitch" => {
            for i in 0..count {
                let idx = start.saturating_add(i);
                if idx >= entry.pitch_edit.len() {
                    break;
                }
                let o = entry.pitch_orig.get(idx).copied().unwrap_or(0.0);
                entry.pitch_edit[idx] = o;
            }

            // If the curve fully matches orig now, clear the user-modified flag.
            let len = entry.pitch_orig.len().min(entry.pitch_edit.len());
            entry.pitch_edit_user_modified = false;
            for i in 0..len {
                let o = entry.pitch_orig[i];
                let e = entry.pitch_edit[i];
                if (e.is_finite() && e > 0.0)
                    && (!(o.is_finite() && o > 0.0) || (e - o).abs() > 1e-3)
                {
                    entry.pitch_edit_user_modified = true;
                    break;
                }
            }
        }
        "tension" => {
            let reference_value = param_reference_value;
            for i in 0..count {
                let idx = start.saturating_add(i);
                if idx >= entry.tension_edit.len() {
                    break;
                }
                entry.tension_edit[idx] = reference_value;
            }
        }
        "dyn" => {
            // 「初始化」对动态 = 回到原声（写哨兵），而不是写描述符默认值 1.0。
            // 后者是把整段压平到参考电平，一个用户点"初始化"绝不想看到的响度巨变。
            let follow = crate::renderer::common_params::DYN_FOLLOW_ORIG;
            if let Some(curve) = entry.extra_curves.get_mut("dyn") {
                for i in 0..count {
                    let idx = start.saturating_add(i);
                    if idx >= curve.len() {
                        break;
                    }
                    curve[idx] = follow;
                }
            }
        }
        _ => {
            let default_value = param_reference_value;
            if let Some(curve) = entry.extra_curves.get_mut(&param) {
                for i in 0..count {
                    let idx = start.saturating_add(i);
                    if idx >= curve.len() {
                        break;
                    }
                    curve[idx] = default_value;
                }
            }
        }
    }

    if let Some(spec) = parse_child_pitch_offset_param(&param) {
        invalidate_rendered_clip_caches_for_child_track(&tl, spec.track_id);
    }

    // Ensure realtime playback reflects edits immediately.
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}

// ===================== 音量 ↔ 动态 互转（混音级参数） =====================

/// 互转方向（命令入参）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum MixConversionDirection {
    /// 音量 → 动态。
    VolumeToDyn,
    /// 动态 → 音量。
    DynToVolume,
}

/// 一帧的互转换算（**纯函数**，正确性的唯一落点，供单测直接验证）。
///
/// ## 为什么要基线补偿（等效性推导）
///
/// 渲染的最终增益是 `out = in × volume(t) × dyn_gain(t) × …`，其中
/// `dyn_gain = 目标电平 / 原声基线`。所以：
///
/// - **volume → dyn**：要满足 `1.0 × (target/orig) = v`，必须写
///   `target = v × orig`。直接搬 `target = v` 会得到 `v/orig` —— 不等效。
/// - **dyn → volume**：要满足 `vol = target/orig`（哨兵帧 target=orig → vol=1.0，
///   自动正确）。直接搬 `vol = target` 同样不等效，且会把"未画"的哨兵帧
///   物化成显式基线值，凭空改变那些帧的响度。
///
/// ## 无内容帧的约定
///
/// `orig < DYN_SILENCE_FLOOR`（−60 dBFS）时后端增益恒 1（绝不放大无内容帧），
/// 因此反向换算写 `vol = 1.0`（保持"不改变"），而不是一个巨大的 `target/orig` 商。
/// 注意门限远低于常见内容电平 —— 真实素材的安静段（−34…−55 dBFS）照样能互转，
/// 不会被误判成"保护帧"。
/// 换算结果超出目标参数值域时按各自值域钳制 —— 钳制帧的等效性破缺是
/// 已知的、被接受的（只发生在极端值组合下）。两个方向的不对称由此而来：
/// - dyn 值域 0..1（`> 1` 会削顶，无意义），故 **volume → dyn** 时任何
///   `v × orig > 1`（= 目标超过 0 dBFS）都被钳到 1.0，响度回落。这类帧
///   本质上应由音量参数承载。
/// - 音量值域 0..2（±6 dB），故 **dyn → volume** 时把安静段提升 +6 dB 以上
///   的帧（增益 > ×2）会被钳到 2.0。
///
/// # 参数
/// - `direction` 互转方向。
/// - `volume_value` 该帧 volume 曲线的原始值；`None` = 该帧无数据（曲线不存在或更短）。
/// - `dyn_raw` 该帧 dyn 曲线的原始值（**未解析哨兵**）；`None` = 无数据。
/// - `baseline` 该帧的原声基线（`dyn_orig`，调用方保证 > 0 且有限）。
///
/// # 返回
/// `(new_volume, new_dyn)` —— 两个字段都恒有值（选区内的每一帧都要么换算、
/// 要么归位到"无变化"），由调用方原样写回。
fn convert_mix_frame_value(
    direction: MixConversionDirection,
    volume_value: Option<f32>,
    dyn_raw: Option<f32>,
    baseline: f32,
) -> (f32, f32) {
    let sentinel = crate::renderer::common_params::DYN_FOLLOW_ORIG;
    // 目标侧值域从描述符读（唯一真源，见 common_params::DYN_VALUE_MAX）。
    let dyn_max = crate::renderer::common_params::DYN_VALUE_MAX;
    let vol_max = 2.0_f32; // 与 volume 描述符值域一致（0..2，±6 dB）
    match direction {
        MixConversionDirection::VolumeToDyn => {
            // 音量归位 1.0；动态接管（有数据的帧换算，无数据的帧表达"沿用原声"）。
            // 反解目标电平：使 `compute_dyn_gain(new_dyn, baseline) == v`。
            // 下限之上即 `v × 基线`（精确）；下限之下增益公式含"无内容淡出"，
            // 严格反解会要求超过值域的目标值，因此这里仍按 `v × 基线` 近似 ——
            // 该情形下转换后该帧会更**轻**（无内容是静音，方向安全），
            // 与 `volume_to_dyn_is_effect_equivalent` 用例声明的边界一致。
            let new_dyn = match volume_value {
                Some(v) if v.is_finite() && baseline > 0.0 => (v * baseline).clamp(0.0, dyn_max),
                _ => sentinel,
            };
            (1.0, new_dyn)
        }
        MixConversionDirection::DynToVolume => {
            // 动态归位哨兵（沿用原声）；音量接管。
            let new_volume = match dyn_raw {
                // 用户画过的帧：换算成等效音量 —— **直接取该帧的动态增益**（唯一真源
                // 是 `compute_dyn_gain`）。此处曾自己重算 `t / max(基线, 下限)`，
                // 一旦增益公式里加入别的因素（如"无内容淡出"）就会与音频分叉，
                // 违背"互转前后增益逐帧一致"的承诺。
                Some(t) if t.is_finite() && t >= 0.0 => {
                    crate::renderer::common_params::compute_dyn_gain(t, baseline)
                        .clamp(0.0, vol_max)
                }
                // 哨兵 / 无数据帧：原本就是"不改变" → 音量 1.0。
                _ => 1.0,
            };
            (new_volume, sentinel)
        }
    }
}

/// 一段选区的互转换算（纯函数）。
///
/// 逐帧取 `(volume, dyn_raw, baseline)` 并套用 [`convert_mix_frame_value`]；
/// 基线越界 / 无效（`dyn_orig` 缺该帧、非有限、≤ 0）的帧**写"无变化"值**
/// （volume=1.0、dyn=哨兵）并计入 `skipped` —— 输出与输入逐帧对齐，调用方
/// 无需关心跳帧位置。正常情况下基线长度 = 工程帧数、选区不会越出工程末端，
/// skipped 恒为 0；该分支只是不让异常数据错位写盘。
///
/// # 返回
/// `(volume_values, dyn_values, skipped)`：前两者长度恒等于 `count`。
fn compute_mix_conversion_range(
    direction: MixConversionDirection,
    volume_curve: Option<&[f32]>,
    dyn_curve: Option<&[f32]>,
    dyn_orig: &[f32],
    start: usize,
    count: usize,
) -> (Vec<f32>, Vec<f32>, usize) {
    let sentinel = crate::renderer::common_params::DYN_FOLLOW_ORIG;
    let mut volume_out = Vec::with_capacity(count);
    let mut dyn_out = Vec::with_capacity(count);
    let mut skipped = 0usize;
    for k in 0..count {
        let idx = start + k;
        let baseline = dyn_orig
            .get(idx)
            .copied()
            .filter(|v| v.is_finite() && *v > 0.0);
        match baseline {
            Some(b) => {
                let volume_value = volume_curve.and_then(|c| c.get(idx)).copied();
                let dyn_raw = dyn_curve.and_then(|c| c.get(idx)).copied();
                let (v, d) = convert_mix_frame_value(direction, volume_value, dyn_raw, b);
                volume_out.push(v);
                dyn_out.push(d);
            }
            None => {
                // 基线缺失：保持"无变化"，绝不信占位基线换算。
                skipped += 1;
                volume_out.push(1.0);
                dyn_out.push(sentinel);
            }
        }
    }
    (volume_out, dyn_out, skipped)
}

/// 音量 ↔ 动态 曲线互转命令（后端单事务）。
///
/// ## 为什么互转必须在后端做（前端纯搬迁方案已被证伪）
///
/// 1. **等效性需要基线**：`dyn_target = volume × orig` / `volume = dyn_target/orig`
///    都要逐帧的权威基线（`TrackParamsState.dyn_orig`，全分辨率），前端拿不到
///    （`get_param_frames` 的 edit 出口已解析哨兵，`orig` 只随窗口下发）。
/// 2. **曲线存在性只有后端知道**：volume 曲线哪些帧"有数据"取决于
///    `extra_curves` 的存在与长度，前端无从判断（会把无数据帧物化成显式值）。
/// 3. **原子性**：此前前端分两次 `set_param_frames`（先写目标、再归位源），
///    两次 `update_timeline` 之间存在"dyn 已生效而 volume 未归位"的快照窗口，
///    播放中表现为瞬时响度突跳。本命令单次加锁、单个撤销点、单次快照提交。
///
/// ## 基线未就绪
///
/// 互转前必须保证 `dyn_orig` 已按当前时间线组装完成（key 命中）。未命中时返回
/// `{ok:false, reason:"analysis_pending"}` 并顺带触发后台分析；前端在
/// `dyn_orig_updated` 后重试。绝不能按占位基线（1.0）换算 —— 那会把错误永久写进曲线。
pub(super) fn convert_mix_param(
    state: State<'_, AppState>,
    track_id: String,
    from: String,
    ranges: Vec<crate::commands::ConvertRange>,
) -> serde_json::Value {
    let direction = match from.as_str() {
        "volume" => MixConversionDirection::VolumeToDyn,
        "dyn" => MixConversionDirection::DynToVolume,
        _ => return serde_json::json!({"ok": false, "reason": "unsupported_param"}),
    };
    if ranges.is_empty() {
        return serde_json::json!({"ok": false, "reason": "empty_selection"});
    }

    let root = {
        let tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
        match tl.resolve_root_track_id(&track_id) {
            Some(id) => id,
            None => return serde_json::json!({"ok": false, "reason": "no_root"}),
        }
    };

    // 基线必须就绪（key 命中）。未命中则触发组装/调度后明确拒绝。
    let analysis_pending = crate::pitch_analysis::maybe_schedule_dyn_orig(&state, &root);
    if analysis_pending {
        return serde_json::json!({"ok": false, "reason": "analysis_pending"});
    }

    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    // 整批一个撤销点：Ctrl+Z 一次回退全部选区的换算与归位。
    state.checkpoint_timeline(&tl, crate::state::HistoryOp::ParamCurve);

    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "reason": "params_missing"});
    };

    // volume 曲线：新键优先，旧 `hifigan_volume` 兜底读取（只读，不写旧键）。
    let volume_snapshot: Vec<f32> = entry
        .extra_curves
        .get("volume")
        .or_else(|| entry.extra_curves.get("hifigan_volume"))
        .cloned()
        .unwrap_or_default();
    let dyn_snapshot: Vec<f32> = entry
        .extra_curves
        .get(crate::renderer::common_params::DYN_PARAM_ID)
        .cloned()
        .unwrap_or_default();
    let dyn_orig_snapshot: Vec<f32> = entry.dyn_orig.clone();

    let mut converted_frames = 0usize;
    let mut skipped_frames = 0usize;
    let mut volume_writes: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut dyn_writes: Vec<(usize, Vec<f32>)> = Vec::new();

    for range in &ranges {
        let start = range.start_frame as usize;
        let count = (range.frame_count as usize).max(1);
        let (volume_values, dyn_values, skipped) = compute_mix_conversion_range(
            direction,
            if volume_snapshot.is_empty() {
                None
            } else {
                Some(volume_snapshot.as_slice())
            },
            if dyn_snapshot.is_empty() {
                None
            } else {
                Some(dyn_snapshot.as_slice())
            },
            &dyn_orig_snapshot,
            start,
            count,
        );
        skipped_frames += skipped;
        // 选区段升序到达：相邻/重叠段合并（后到覆盖先到，与前端多段写入的
        // 确定性一致）；互不相交的段保持独立 —— **缝隙帧不写**（互转只作用于
        // 选区，缝隙里的曲线原样保留）。
        merge_write_segment(&mut volume_writes, start, &volume_values);
        merge_write_segment(&mut dyn_writes, start, &dyn_values);
        converted_frames += count - skipped;
    }

    // 写回：曲线不够长时用**该参数的"无变化"值**扩尾（volume=1.0、dyn=哨兵），
    // 避免凭空填入"压平到参考电平"之类的真实动作。
    for (start, values) in &volume_writes {
        let volume_curve = entry.extra_curves.entry("volume".to_string()).or_default();
        let needed = start + values.len();
        if volume_curve.len() < needed {
            volume_curve.resize(needed, 1.0);
        }
        for (k, v) in values.iter().enumerate() {
            volume_curve[start + k] = *v;
        }
    }
    for (start, values) in &dyn_writes {
        let dyn_curve = entry
            .extra_curves
            .entry(crate::renderer::common_params::DYN_PARAM_ID.to_string())
            .or_default();
        let needed = start + values.len();
        if dyn_curve.len() < needed {
            dyn_curve.resize(needed, crate::renderer::common_params::DYN_FOLLOW_ORIG);
        }
        for (k, v) in values.iter().enumerate() {
            dyn_curve[start + k] = *v;
        }
    }

    let root_for_dyn = root.clone();
    // Ensure realtime playback reflects the conversion immediately（单次快照提交）。
    state.audio_engine.update_timeline(tl.clone());
    drop(tl);
    // dyn 曲线可能刚获得第一个非哨兵值（volume→dyn 方向），与 set_param_frames
    // 的既有约定一致：锁外补一次组装/调度（基线已就绪时它是 no-op）。
    let _ = crate::pitch_analysis::maybe_schedule_dyn_orig(&state, &root_for_dyn);

    serde_json::json!({
        "ok": true,
        "convertedFrames": converted_frames,
        "skippedFrames": skipped_frames,
    })
}

/// 把一段写入并入写窗口列表：与最后一段**相接**（start == 旧段终点）则拼接，
/// 否则另起新段。段列表按写入顺序逐段应用到曲线 —— 同一帧被多段覆盖时
/// "后者胜"，行为确定。
///
/// 调用方保证 `start` 按升序到达（选区段升序）。选区段本就互不相交，因此
/// 现实路径只有"相接拼接"与"新段"两种；重叠输入不发生，无需在此处理。
fn merge_write_segment(writes: &mut Vec<(usize, Vec<f32>)>, start: usize, values: &[f32]) {
    if values.is_empty() {
        return;
    }
    if let Some(&(last_start, _)) = writes.last() {
        let last = &mut writes.last_mut().unwrap().1;
        let prev_end = last_start + last.len();
        if start == prev_end {
            last.extend_from_slice(values);
            return;
        }
    }
    writes.push((start, values.to_vec()));
}

pub(super) fn get_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
) -> crate::models::StaticParamValuePayload {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());

    let root = match tl.resolve_root_track_id(&track_id) {
        Some(id) => id,
        None => {
            return crate::models::StaticParamValuePayload {
                ok: false,
                root_track_id: String::new(),
                param,
                value: 0.0,
            }
        }
    };

    tl.ensure_params_for_root(&root);
    let kind = tl
        .tracks
        .iter()
        .find(|track| track.id == root)
        .map(|track| crate::state::SynthPipelineKind::from_track_algo(&track.pitch_analysis_algo))
        .unwrap_or(crate::state::SynthPipelineKind::WorldVocoder);
    let value = tl
        .params_by_root_track
        .get(&root)
        .and_then(|entry| entry.extra_params.get(&param).copied())
        .unwrap_or_else(|| resolve_static_param_default_value(kind, &param));

    crate::models::StaticParamValuePayload {
        ok: true,
        root_track_id: root,
        param,
        value,
    }
}

pub(super) fn set_static_param(
    state: State<'_, AppState>,
    track_id: String,
    param: String,
    value: f64,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    let do_checkpoint = checkpoint.unwrap_or(true);
    if do_checkpoint {
        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ParamStatic);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.ensure_params_for_root(&root);

    let Some(entry) = tl.params_by_root_track.get_mut(&root) else {
        return serde_json::json!({"ok": false, "error": "params missing"});
    };

    entry.extra_params.insert(param, value);
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}

/// "锁定参数线"：剪辑拉伸后把旧范围内的参数曲线时域映射到新范围。
///
/// 后端一次性完成 pitch（用户编辑过时）/ tension / 所有已存在的自动化曲线
/// （无论参数是否在 UI 激活）的批量映射，取代旧前端逐参数 get/set/restore
/// 的多请求流程（旧流程只覆盖 pitch+tension，遗漏其余全部参数曲线）。
///
/// 默认不产生独立撤销检查点：曲线映射与剪辑几何变更合并为同一撤销步
/// （与旧前端 set/restore(checkpoint=false) 的流程保持一致）。
pub(super) fn stretch_track_linked_params(
    state: State<'_, AppState>,
    track_id: String,
    mappings: Vec<crate::state::StretchLinkedRangeSec>,
    checkpoint: Option<bool>,
) -> serde_json::Value {
    let mut tl = state.timeline.lock().unwrap_or_else(|e| e.into_inner());
    if checkpoint.unwrap_or(false) {
        state.checkpoint_timeline(&tl, crate::state::HistoryOp::ParamStretch);
    }

    let Some(root) = tl.resolve_root_track_id(&track_id) else {
        return serde_json::json!({"ok": false});
    };
    tl.stretch_linked_params_in_root_range(&root, &mappings);

    // Ensure realtime playback reflects edits immediately.
    state.audio_engine.update_timeline(tl.clone());

    serde_json::json!({"ok": true})
}

#[cfg(test)]
mod mix_conversion_tests {
    use super::*;

    const GAIN: fn(f32, f32) -> f32 = crate::renderer::common_params::compute_dyn_gain;

    /// volume→dyn 的**最终效果等效**属性：转换前 `增益 = v`，转换后
    /// `增益 = compute_dyn_gain(new_dyn, baseline)`，两者逐帧相等
    /// （排除静音保护帧与钳制帧，两者在 convert_mix_frame_value 的文档中声明）。
    #[test]
    fn volume_to_dyn_is_effect_equivalent() {
        let baselines = [1.0f32, 0.5, 0.25, 0.8, 0.06];
        let volumes = [1.0f32, 0.5, 2.0, 0.0, 3.0];
        for baseline in baselines {
            for v in volumes {
                let (new_volume, new_dyn) =
                    convert_mix_frame_value(MixConversionDirection::VolumeToDyn, Some(v), None, baseline);
                assert_eq!(new_volume, 1.0, "源音量必须归位 1.0");
                let after = GAIN(new_dyn, baseline);
                if v * baseline <= 1.0 {
                    // dyn 值域内（0..1）：最终效果必须逐帧等效。
                    assert!(
                        (v - after).abs() < 1e-5,
                        "v={v} baseline={baseline} → dyn={new_dyn} gain={after}"
                    );
                } else {
                    // 超出 dyn 值域（v × baseline > 1.0 = 0 dBFS）：按文档化行为
                    // 钳到 1.0 —— 等效性破缺是 convert_mix_frame_value 文档
                    // 声明的已知边界（目标电平不允许表达"超过满量程"，
                    // 那在播出去之前就会被削顶）。
                    assert_eq!(new_dyn, 1.0, "v={v} baseline={baseline}");
                    assert!(after < v, "钳制帧增益必须小于原音量");
                }
            }
        }
    }

    /// dyn→volume 的最终效果等效属性：转换前 `增益 = GAIN(target, baseline)`，
    /// 转换后 `增益 = vol`。哨兵帧两端都是 1.0。
    #[test]
    fn dyn_to_volume_is_effect_equivalent() {
        let baselines = [1.0f32, 0.5, 0.25, 0.8];
        let targets = [1.0f32, 0.5, 2.0, 0.0];
        for baseline in baselines {
            for target in targets {
                let (new_volume, new_dyn) =
                    convert_mix_frame_value(MixConversionDirection::DynToVolume, None, Some(target), baseline);
                assert_eq!(new_dyn, crate::renderer::common_params::DYN_FOLLOW_ORIG);
                let before = GAIN(target, baseline);
                if before <= 2.0 {
                    // 音量值域内（0..2）：最终效果必须逐帧等效。
                    assert!(
                        (new_volume - before).abs() < 1e-5,
                        "target={target} baseline={baseline} → vol={new_volume} expected {before}"
                    );
                } else {
                    // 增益 > ×2（对安静段 +6 dB 以上的提升）：音量值域承载不了，
                    // 按文档化行为钳到 2.0（响度回落 —— 这类帧应由动态承载）。
                    assert_eq!(new_volume, 2.0, "target={target} baseline={baseline}");
                }
            }
        }
    }

    #[test]
    fn dyn_to_volume_follows_floor_and_clamp() {
        // 无内容帧（低于 −60 dBFS）：分母与 compute_dyn_gain 同款钳到下限，
        // 故换算结果有界（而非写成 1.0 的"保护"值）。
        let (vol, new_dyn) = convert_mix_frame_value(
            MixConversionDirection::DynToVolume,
            None,
            Some(1.0),
            crate::renderer::common_params::DYN_SILENCE_FLOOR * 0.5,
        );
        // t/下限 = 1000 → 被音量值域钳到 2。
        assert_eq!(vol, 2.0);
        assert_eq!(new_dyn, crate::renderer::common_params::DYN_FOLLOW_ORIG);

        // 超出了值域：同样钳到 2（与 compute_dyn_gain 的增益保持一致）。
        let (vol, _) = convert_mix_frame_value(
            MixConversionDirection::DynToVolume,
            None,
            Some(1.0),
            0.01,
        );
        assert_eq!(vol, 2.0);
        // 与增益公式逐帧一致（互转的定义就是"等效"）。
        let gain = crate::renderer::common_params::compute_dyn_gain(1.0, 0.01);
        assert_eq!(vol, gain.min(2.0));
    }

    /// ★ 回归：真实素材的安静段必须能互转（门限曾是 −26 dBFS，误伤 −34…−55 dBFS）。
    #[test]
    fn dyn_to_volume_converts_quiet_content() {
        // −40 dBFS 的素材画了目标 0.5 → 需 ×50，被音量值域钳到 2.0
        //（等效性破缺属文档化边界），但**绝不能**被当成保护帧写 1.0。
        let (vol, _) = convert_mix_frame_value(
            MixConversionDirection::DynToVolume,
            None,
            Some(0.5),
            0.01,
        );
        assert_eq!(vol, 2.0, "安静内容必须参与换算，而非被当作保护帧");

        // 值域内的正常换算：−40 dBFS 画目标 0.01（= 原声）→ ×1。
        let (vol, _) = convert_mix_frame_value(
            MixConversionDirection::DynToVolume,
            None,
            Some(0.01),
            0.01,
        );
        assert!((vol - 1.0).abs() < 1e-6, "got {vol}");
    }

    #[test]
    fn unset_frames_stay_unset() {
        // volume→dyn：volume 无数据的帧 → dyn 哨兵（沿用原声），绝不写 1.0
        // （那会把"未画"物化成"压平到参考电平"）。
        let (_, new_dyn) =
            convert_mix_frame_value(MixConversionDirection::VolumeToDyn, None, None, 0.7);
        assert_eq!(new_dyn, crate::renderer::common_params::DYN_FOLLOW_ORIG);
        // dyn→volume：哨兵帧 → 音量 1.0（原语义就是"不改变"）。
        let (new_volume, _) =
            convert_mix_frame_value(MixConversionDirection::DynToVolume, None, Some(-1.0), 0.7);
        assert_eq!(new_volume, 1.0);
        let (new_volume, _) =
            convert_mix_frame_value(MixConversionDirection::DynToVolume, None, None, 0.7);
        assert_eq!(new_volume, 1.0);
    }

    #[test]
    fn range_conversion_keeps_frame_alignment() {
        // 基线只有 3 帧有效，选区 5 帧：后 2 帧写"无变化"值、计入 skipped，
        // 输出与输入逐帧对齐（长度恒等于 count）。
        let volume = vec![0.5f32, 1.5, 1.0, 1.0, 1.0];
        // 基线取 0.8（0 dBFS 以下）而非 2.0 —— 后者会让 1.0 × 2.0 撞上 dyn
        // 值域上限、被钳制，掩盖"逐帧换算"这一本测试真正要守护的性质。
        let dyn_orig = vec![1.0f32, 0.5, 0.8];
        let (v_out, d_out, skipped) = compute_mix_conversion_range(
            MixConversionDirection::VolumeToDyn,
            Some(&volume),
            None,
            &dyn_orig,
            0,
            5,
        );
        assert_eq!(v_out.len(), 5);
        assert_eq!(d_out.len(), 5);
        assert_eq!(skipped, 2);
        // 前 3 帧按基线换算。
        assert!((d_out[0] - 0.5).abs() < 1e-6); // 0.5 × 1.0
        assert!((d_out[1] - 0.75).abs() < 1e-6); // 1.5 × 0.5
        assert!((d_out[2] - 0.8).abs() < 1e-6); // 1.0 × 0.8
        // 后 2 帧：volume 归位 1.0、dyn 哨兵。
        assert_eq!(v_out[3], 1.0);
        assert_eq!(d_out[3], crate::renderer::common_params::DYN_FOLLOW_ORIG);
        assert_eq!(d_out[4], crate::renderer::common_params::DYN_FOLLOW_ORIG);
    }

    /// volume→dyn 超出 0 dBFS 的目标必须钳到 1.0（>1 会削顶，无意义）。
    #[test]
    fn volume_to_dyn_clamps_target_at_full_scale() {
        // 把一段安静素材（0.1）抬到 1.0 需要 ×10 → 目标 1.0，正好是天花板；
        // 再大的音量（2.0）也只能得到 1.0，不能写出削顶之外的目标电平。
        let (_, loud) = convert_mix_frame_value(
            MixConversionDirection::VolumeToDyn,
            Some(10.0),
            None,
            0.1,
        );
        assert_eq!(loud, 1.0);
        let (_, loud) =
            convert_mix_frame_value(MixConversionDirection::VolumeToDyn, Some(2.0), None, 0.8);
        assert_eq!(loud, 1.0, "1.6 的目标必须钳到满量程 1.0");
        // 值域内的正常换算不受影响。
        let (_, ok) =
            convert_mix_frame_value(MixConversionDirection::VolumeToDyn, Some(0.5), None, 0.8);
        assert!((ok - 0.4).abs() < 1e-6);
    }

    #[test]
    fn range_conversion_dyn_to_volume_respects_sentinels() {
        // dyn 曲线：用户只画了 [0..2)，哨兵一帧、无数据一帧。
        let dyn_curve = vec![0.5f32, 2.0, -1.0];
        let dyn_orig = vec![1.0f32, 0.5, 0.5];
        let (v_out, d_out, skipped) = compute_mix_conversion_range(
            MixConversionDirection::DynToVolume,
            None,
            Some(&dyn_curve),
            &dyn_orig,
            0,
            3,
        );
        assert_eq!(skipped, 0);
        assert!((v_out[0] - 0.5).abs() < 1e-6); // 0.5 / 1.0
        assert_eq!(v_out[1], 2.0); // 2.0 / 0.5 = 增益 4 → 音量值域钳到 2
        assert_eq!(v_out[2], 1.0); // 哨兵帧 → 不改变
        for d in d_out {
            assert_eq!(d, crate::renderer::common_params::DYN_FOLLOW_ORIG);
        }
    }

    #[test]
    fn merge_write_segments_adjacent_gap_and_overlap() {
        let mut writes: Vec<(usize, Vec<f32>)> = Vec::new();
        merge_write_segment(&mut writes, 0, &[1.0, 1.0]);
        merge_write_segment(&mut writes, 2, &[2.0, 2.0]); // 相接 → 拼接
        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0], (0, vec![1.0, 1.0, 2.0, 2.0]));

        merge_write_segment(&mut writes, 10, &[3.0]); // 缝隙 → 新段（缝隙不写）
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[1], (10, vec![3.0]));

        merge_write_segment(&mut writes, 9, &[9.0, 8.0]); // 回跳（不发生）→ 新段
        assert_eq!(writes.len(), 3);
        assert_eq!(writes[2], (9, vec![9.0, 8.0]));
    }
}
