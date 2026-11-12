//! GAME 自动音符分块（OpenVPI Generative Adaptive MIDI Extractor）的 ONNX 推理。
//!
//! ## 为什么需要它
//!
//! 参数编辑器的“块”必须对应**音符事件**。电平 VAD 只能找到发声段，无法可靠区分
//! 长元音里的换字、滑音里的重新起音；这类边界需要音高与内容联合建模。GAME 是
//! SOME 的后继模型，用 encoder → D3PM segmenter → bd2dur → estimator 产出音符
//! 边界、持续时间、音高与浊音标记。
//!
//! ## 为什么不直接引用 HachiTune / dataset-tools
//!
//! HachiTune 是 AGPL-3.0，不能直接搬进本工程。OpenVPI 的 `dataset-tools` 里有
//! 可参考的 C++ `GameInfer`，但本工程的音频、缓存与模型路径都在 Rust 内核里。
//! 因此这里按 GAME 公开的 ONNX 张量契约重写推理管线，只复用现有 `ort` 封装。
//!
//! ## 输入输出
//!
//! 输入为 44.1 kHz 单声道 PCM。模型配置要求 `timestep = 0.01s`、每帧 441 个
//! 样本。输出是源域秒级的音符事件；静音与休止也会作为 `voiced = false` 的事件
//! 保留，供调用方在显示层决定是否画块，而不是把静音伪装成音节。
//!
//! ## 降级
//!
//! 模型缺失、加载失败或推理失败时返回错误，由调用方显式回退到既有 VAD。绝不能
//! 静默返回空区间，否则前端会误以为“没有音符”并把整个 Clip 当成一个块。

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// GAME 模型采样率。
pub const GAME_SAMPLE_RATE: u32 = 44_100;
/// GAME 模型每帧对应的时间，秒。
pub const GAME_TIMESTEP_SEC: f64 = 0.01;
/// 每帧样本数（44_100 × 0.01）。
pub const GAME_HOP_SAMPLES: usize = 441;
/// 单个模型切片的最长音频样本数，避免长素材一次性占用过多内存。
pub const GAME_MAX_CHUNK_SAMPLES: usize = 50 * GAME_SAMPLE_RATE as usize;
/// 模型算法标识；改变张量流程或后处理时必须升版本。
pub const GAME_ALGORITHM: &str = "game_d3pm_v1";
pub const GAME_ALGORITHM_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct GameConfig {
    pub samplerate: u32,
    pub timestep: f64,
    #[serde(default)]
    pub languages: Option<serde_json::Value>,
    #[serde(default = "default_loop")]
    pub r#loop: bool,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
}

fn default_loop() -> bool {
    true
}

fn default_embedding_dim() -> usize {
    128
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            samplerate: GAME_SAMPLE_RATE,
            timestep: GAME_TIMESTEP_SEC,
            languages: None,
            r#loop: true,
            embedding_dim: default_embedding_dim(),
        }
    }
}

/// GAME 输出的一个音符/休止事件（源域秒）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameNoteEvent {
    pub start_sec: f64,
    pub end_sec: f64,
    /// MIDI 音高；休止事件也保留模型估计值，但 `voiced=false` 时 UI 不应使用它。
    pub midi_note: f32,
    pub voiced: bool,
}

/// 一份 GAME 推理结果。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameAnalysis {
    pub algorithm: String,
    pub algorithm_version: u32,
    pub events: Vec<GameNoteEvent>,
}

impl GameAnalysis {
    pub fn new(events: Vec<GameNoteEvent>) -> Self {
        Self {
            algorithm: GAME_ALGORITHM.to_string(),
            algorithm_version: GAME_ALGORITHM_VERSION,
            events,
        }
    }
}

/// 解析模型目录中的 `config.json`。
pub fn read_config(model_dir: &Path) -> Result<GameConfig, String> {
    let path = model_dir.join("config.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|error| format!("failed to read GAME config {}: {error}", path.display()))?;
    let config: GameConfig = serde_json::from_str(&raw)
        .map_err(|error| format!("invalid GAME config {}: {error}", path.display()))?;
    if config.samplerate != GAME_SAMPLE_RATE {
        return Err(format!(
            "unsupported GAME sample rate {} (expected {})",
            config.samplerate, GAME_SAMPLE_RATE
        ));
    }
    if !(config.timestep.is_finite() && (config.timestep - GAME_TIMESTEP_SEC).abs() < 1e-9) {
        return Err(format!(
            "unsupported GAME timestep {} (expected {})",
            config.timestep, GAME_TIMESTEP_SEC
        ));
    }
    Ok(config)
}

fn required_models(dir: &Path) -> [(&'static str, PathBuf); 5] {
    [
        ("encoder.onnx", dir.join("encoder.onnx")),
        ("segmenter.onnx", dir.join("segmenter.onnx")),
        ("estimator.onnx", dir.join("estimator.onnx")),
        ("bd2dur.onnx", dir.join("bd2dur.onnx")),
        ("dur2bd.onnx", dir.join("dur2bd.onnx")),
    ]
}

/// 模型目录是否齐全（不加载 ONNX 会话，仅用于启动诊断）。
pub fn models_present(dir: &Path) -> bool {
    dir.join("config.json").is_file() && required_models(dir).iter().all(|(_, path)| path.is_file())
}

/// 对 PCM 做单声道、44.1 kHz 重采样；返回样本。
///
/// GAME 的 mel 前端在模型内部完成，调用方只负责提供单声道波形。重采样复用内核
/// 已有的线性重采样，避免再写一份 DSP。
pub fn prepare_mono_44k(
    interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
) -> Result<Vec<f32>, String> {
    if channels == 0 || interleaved.len() < channels {
        return Err("empty or invalid PCM".into());
    }
    let frames = interleaved.len() / channels;
    let mut mono = Vec::with_capacity(frames);
    for frame in 0..frames {
        let base = frame * channels;
        let mut sum = 0.0f32;
        for channel in 0..channels {
            sum += interleaved[base + channel];
        }
        mono.push(sum / channels as f32);
    }
    if sample_rate == GAME_SAMPLE_RATE {
        return Ok(mono);
    }
    // `linear_resample_interleaved` 接受交错多声道；单声道时 `channels=1` 仍然成立。
    let resampled =
        crate::mixdown::linear_resample_interleaved(&mono, 1, sample_rate, GAME_SAMPLE_RATE);
    Ok(resampled)
}

/// 把长音频按静音切成不超过模型上限的块。
///
/// 这里只负责**算力边界**，不是音符边界：每块可以独立跑模型，块与块之间按源域
/// 秒拼回去。找不到静音时在块上限处硬切，保证不会因为长素材耗尽内存。
pub fn split_waveform(waveform: &[f32]) -> Vec<(usize, usize)> {
    if waveform.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut start = 0usize;
    while start < waveform.len() {
        let end = (start + GAME_MAX_CHUNK_SAMPLES).min(waveform.len());
        if end == waveform.len() {
            out.push((start, end));
            break;
        }
        // 在末尾 10% 内找能量最低的 10ms 窗口，避免切在发声中间。
        let search_start = start + (end - start) * 9 / 10;
        let mut best = end;
        let mut best_energy = f64::INFINITY;
        let mut cursor = search_start;
        while cursor + GAME_HOP_SAMPLES <= end {
            let energy = waveform[cursor..cursor + GAME_HOP_SAMPLES]
                .iter()
                .map(|sample| (*sample as f64) * (*sample as f64))
                .sum::<f64>();
            if energy < best_energy {
                best_energy = energy;
                best = cursor + GAME_HOP_SAMPLES / 2;
            }
            cursor += GAME_HOP_SAMPLES;
        }
        if best <= start {
            best = end;
        }
        out.push((start, best));
        start = best;
    }
    out
}

/// GAME 的推理参数。默认值与 OpenVPI `GameInfer` 保持一致：
/// D3PM 8 步、边界阈值 0.2、边界局部极大半径 2 帧、浊音阈值 0.2。
#[derive(Debug, Clone, Copy)]
pub struct GameInferenceConfig {
    pub d3pm_steps: usize,
    pub seg_threshold: f32,
    pub seg_radius: i64,
    pub est_threshold: f32,
}

impl Default for GameInferenceConfig {
    fn default() -> Self {
        Self {
            d3pm_steps: 8,
            seg_threshold: 0.2,
            seg_radius: 2,
            est_threshold: 0.2,
        }
    }
}

/// 生成 D3PM 采样时间序列：从 0（纯噪声）到 1（干净数据）。
pub fn d3pm_timesteps(steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    (0..steps).map(|step| step as f32 / steps as f32).collect()
}

#[cfg(feature = "onnx")]
mod ort_impl {
    use super::*;
    use ort::session::Session;
    use ort::value::Tensor;
    use std::sync::{Mutex, OnceLock};

    struct GameSessions {
        encoder: Session,
        segmenter: Session,
        estimator: Session,
        bd2dur: Session,
    }

    impl GameSessions {
        fn load(dir: &Path) -> Result<Self, String> {
            let load = |name: &str| -> Result<Session, String> {
                let path = dir.join(name);
                let (session, _ep) = crate::vocoder_ort_session::build_ort_session(
                    &path,
                    crate::vocoder_ort_session::OrtSessionRole::Game,
                )?;
                Ok(session)
            };
            Ok(Self {
                encoder: load("encoder.onnx")?,
                segmenter: load("segmenter.onnx")?,
                estimator: load("estimator.onnx")?,
                bd2dur: load("bd2dur.onnx")?,
            })
        }
    }

    static SESSIONS: OnceLock<Mutex<Option<(PathBuf, GameSessions)>>> = OnceLock::new();

    fn sessions_for(dir: &Path) -> Result<(), String> {
        let slot = SESSIONS.get_or_init(|| Mutex::new(None));
        let mut guard = slot.lock().unwrap_or_else(|error| error.into_inner());
        if guard
            .as_ref()
            .is_some_and(|(cached_dir, _)| cached_dir == dir)
        {
            return Ok(());
        }
        let sessions = GameSessions::load(dir)?;
        *guard = Some((dir.to_path_buf(), sessions));
        Ok(())
    }

    fn with_sessions<T>(
        dir: &Path,
        f: impl FnOnce(&mut GameSessions) -> Result<T, String>,
    ) -> Result<T, String> {
        sessions_for(dir)?;
        let slot = SESSIONS
            .get()
            .ok_or("GAME model sessions are not initialized")?;
        let mut guard = slot.lock().unwrap_or_else(|error| error.into_inner());
        let (_, sessions) = guard.as_mut().ok_or("GAME model sessions are not loaded")?;
        f(sessions)
    }

    fn extract_bool(value: &ort::value::DynValue) -> Result<Vec<bool>, String> {
        let (_, data) = value
            .try_extract_tensor::<bool>()
            .map_err(|error| format!("extract GAME bool tensor: {error}"))?;
        Ok(data.to_vec())
    }

    fn extract_f32(value: &ort::value::DynValue) -> Result<Vec<f32>, String> {
        let (_, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|error| format!("extract GAME f32 tensor: {error}"))?;
        Ok(data.to_vec())
    }

    /// 跑一次完整 GAME 推理。`waveform` 必须是 44.1 kHz 单声道。
    pub fn infer_waveform(
        dir: &Path,
        waveform: &[f32],
        inference: GameInferenceConfig,
    ) -> Result<Vec<GameNoteEvent>, String> {
        if waveform.is_empty() {
            return Ok(Vec::new());
        }
        with_sessions(dir, |sessions| {
            let mut all = Vec::new();
            for (start, end) in split_waveform(waveform) {
                let chunk = &waveform[start..end];
                let chunk_events = infer_chunk(
                    &mut sessions.encoder,
                    &mut sessions.segmenter,
                    &mut sessions.bd2dur,
                    &mut sessions.estimator,
                    chunk,
                    start as f64 / GAME_SAMPLE_RATE as f64,
                    inference,
                )?;
                all.extend(chunk_events);
            }
            Ok(normalize_events(all))
        })
    }

    fn infer_chunk(
        encoder: &mut Session,
        segmenter: &mut Session,
        bd2dur: &mut Session,
        estimator: &mut Session,
        waveform: &[f32],
        chunk_start_sec: f64,
        inference: GameInferenceConfig,
    ) -> Result<Vec<GameNoteEvent>, String> {
        let duration = waveform.len() as f32 / GAME_SAMPLE_RATE as f32;
        let duration_tensor = Tensor::from_array(([1usize], vec![duration]))
            .map_err(|error| format!("build GAME duration tensor: {error}"))?;
        let waveform_tensor = Tensor::from_array((
            [1usize, waveform.len()],
            waveform.to_vec().into_boxed_slice(),
        ))
        .map_err(|error| format!("build GAME waveform tensor: {error}"))?;

        let input_names: Vec<String> = encoder
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        let mut encoder_inputs = Vec::with_capacity(input_names.len());
        for name in &input_names {
            match name.as_str() {
                "waveform" => encoder_inputs.push((
                    name.clone(),
                    Tensor::from_array((
                        [1usize, waveform.len()],
                        waveform.to_vec().into_boxed_slice(),
                    ))
                    .map_err(|error| format!("build GAME waveform input: {error}"))?,
                )),
                "duration" => encoder_inputs.push((name.clone(), duration_tensor.clone())),
                other => {
                    return Err(format!(
                        "unsupported GAME encoder input '{other}' (expected waveform/duration)"
                    ))
                }
            }
        }
        let encoder_outputs = encoder
            .run(vec![
                ("waveform".to_string(), waveform_tensor.upcast()),
                ("duration".to_string(), duration_tensor.upcast()),
            ])
            .map_err(|error| format!("GAME encoder failed: {error}"))?;
        let x_seg = extract_f32(
            encoder_outputs
                .get("x_seg")
                .ok_or("GAME encoder did not produce x_seg")?,
        )?;
        let x_est = extract_f32(
            encoder_outputs
                .get("x_est")
                .ok_or("GAME encoder did not produce x_est")?,
        )?;
        let mask_t = extract_bool(
            encoder_outputs
                .get("maskT")
                .ok_or("GAME encoder did not produce maskT")?,
        )?;
        let t = mask_t.len();
        if t == 0 || x_seg.len() % t != 0 || x_est.len() % t != 0 {
            return Err(format!(
                "GAME encoder shape mismatch: maskT={t}, x_seg={}, x_est={}",
                x_seg.len(),
                x_est.len()
            ));
        }
        let channels = x_seg.len() / t;

        let mut previous_boundaries = vec![false; t];
        let mut boundaries = previous_boundaries.clone();
        let segmenter_input_names: Vec<String> = segmenter
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        for step in d3pm_timesteps(inference.d3pm_steps) {
            let x_seg_tensor =
                Tensor::from_array(([1usize, t, channels], x_seg.clone().into_boxed_slice()))
                    .map_err(|error| format!("build GAME x_seg tensor: {error}"))?;
            let mask_tensor = Tensor::from_array(([1usize, t], mask_t.clone().into_boxed_slice()))
                .map_err(|error| format!("build GAME maskT tensor: {error}"))?;
            let known_tensor = Tensor::from_array(([1usize, t], vec![false; t].into_boxed_slice()))
                .map_err(|error| format!("build GAME known_boundaries tensor: {error}"))?;
            let previous_tensor =
                Tensor::from_array(([1usize, t], previous_boundaries.clone().into_boxed_slice()))
                    .map_err(|error| format!("build GAME prev_boundaries tensor: {error}"))?;
            let threshold_tensor = Tensor::from_array(((), vec![inference.seg_threshold]))
                .map_err(|error| format!("build GAME segmenter threshold: {error}"))?;
            let radius_tensor = Tensor::from_array(((), vec![inference.seg_radius]))
                .map_err(|error| format!("build GAME segmenter radius: {error}"))?;
            let t_tensor = Tensor::from_array(([1usize], vec![step]))
                .map_err(|error| format!("build GAME segmenter timestep: {error}"))?;
            let language_tensor = Tensor::from_array(([1usize], vec![0i64]))
                .map_err(|error| format!("build GAME segmenter language: {error}"))?;

            let mut inputs = Vec::with_capacity(segmenter_input_names.len());
            for name in &segmenter_input_names {
                let tensor = match name.as_str() {
                    "x_seg" => x_seg_tensor.clone().upcast(),
                    "maskT" => mask_tensor.clone().upcast(),
                    "known_boundaries" => known_tensor.clone().upcast(),
                    "prev_boundaries" => previous_tensor.clone().upcast(),
                    "threshold" => threshold_tensor.clone().upcast(),
                    "radius" => radius_tensor.clone().upcast(),
                    "t" => t_tensor.clone().upcast(),
                    "language" => language_tensor.clone().upcast(),
                    other => return Err(format!("unsupported GAME segmenter input '{other}'")),
                };
                inputs.push((name.clone(), tensor));
            }
            let outputs = segmenter
                .run(inputs)
                .map_err(|error| format!("GAME segmenter failed at t={step}: {error}"))?;
            let raw = extract_bool(
                outputs
                    .get("boundaries")
                    .ok_or("GAME segmenter did not produce boundaries")?,
            )?;
            if raw.len() != t {
                return Err(format!(
                    "GAME segmenter returned {} boundaries for {t} frames",
                    raw.len()
                ));
            }
            previous_boundaries = raw;
            boundaries = previous_boundaries.clone();
        }

        let boundary_tensor =
            Tensor::from_array(([1usize, t], boundaries.clone().into_boxed_slice()))
                .map_err(|error| format!("build GAME boundaries tensor: {error}"))?;
        let mask_tensor = Tensor::from_array(([1usize, t], mask_t.clone().into_boxed_slice()))
            .map_err(|error| format!("build GAME maskT tensor: {error}"))?;
        let bd_outputs = bd2dur
            .run(vec![
                ("boundaries".to_string(), boundary_tensor.upcast()),
                ("maskT".to_string(), mask_tensor.upcast()),
            ])
            .map_err(|error| format!("GAME bd2dur failed: {error}"))?;
        let durations = extract_f32(
            bd_outputs
                .get("durations")
                .ok_or("GAME bd2dur did not produce durations")?,
        )?;
        let mask_n = extract_bool(
            bd_outputs
                .get("maskN")
                .ok_or("GAME bd2dur did not produce maskN")?,
        )?;
        let n = durations.len().min(mask_n.len());
        if n == 0 {
            return Ok(Vec::new());
        }
        let mask_n_tensor =
            Tensor::from_array(([1usize, n], mask_n[..n].to_vec().into_boxed_slice()))
                .map_err(|error| format!("build GAME maskN tensor: {error}"))?;
        let x_est_tensor = Tensor::from_array(([1usize, t, channels], x_est.into_boxed_slice()))
            .map_err(|error| format!("build GAME x_est tensor: {error}"))?;
        let boundaries_tensor = Tensor::from_array(([1usize, t], boundaries.into_boxed_slice()))
            .map_err(|error| format!("build GAME estimator boundaries: {error}"))?;
        let mask_tensor = Tensor::from_array(([1usize, t], mask_t.into_boxed_slice()))
            .map_err(|error| format!("build GAME estimator maskT: {error}"))?;
        let est_threshold = Tensor::from_array(((), vec![inference.est_threshold]))
            .map_err(|error| format!("build GAME estimator threshold: {error}"))?;
        let est_outputs = estimator
            .run(vec![
                ("x_est".to_string(), x_est_tensor.upcast()),
                ("boundaries".to_string(), boundaries_tensor.upcast()),
                ("maskT".to_string(), mask_tensor.upcast()),
                ("maskN".to_string(), mask_n_tensor.upcast()),
                ("threshold".to_string(), est_threshold.upcast()),
            ])
            .map_err(|error| format!("GAME estimator failed: {error}"))?;
        let presence = extract_bool(
            est_outputs
                .get("presence")
                .ok_or("GAME estimator did not produce presence")?,
        )?;
        let scores = extract_f32(
            est_outputs
                .get("scores")
                .ok_or("GAME estimator did not produce scores")?,
        )?;
        Ok(notes_from_outputs(
            &durations,
            &presence,
            &scores,
            chunk_start_sec,
        ))
    }
}

#[cfg(feature = "onnx")]
pub use ort_impl::infer_waveform;

/// 把模型输出的帧级边界 / 帧级音符转换为源域秒事件。
///
/// `durations` 是模型给出的逐个音符持续时间（秒）；`presence` 表示是否浊音；
/// `scores` 是 MIDI 音高。`chunk_start_sec` 是该切片在源域中的起点。
pub fn notes_from_outputs(
    durations: &[f32],
    presence: &[bool],
    scores: &[f32],
    chunk_start_sec: f64,
) -> Vec<GameNoteEvent> {
    let mut out = Vec::new();
    let count = durations.len().min(scores.len());
    let mut cursor = chunk_start_sec.max(0.0);
    for index in 0..count {
        let duration = durations[index] as f64;
        if !(duration.is_finite() && duration > 0.0) {
            continue;
        }
        let start = cursor;
        let end = start + duration;
        cursor = end;
        out.push(GameNoteEvent {
            start_sec: start,
            end_sec: end,
            midi_note: scores[index],
            voiced: presence.get(index).copied().unwrap_or(false),
        });
    }
    out
}

/// 去掉零宽、重叠事件并排序；`voiced=false` 的休止事件也保留。
pub fn normalize_events(events: Vec<GameNoteEvent>) -> Vec<GameNoteEvent> {
    let mut out = Vec::new();
    for event in events {
        if !(event.end_sec > event.start_sec + 1e-9) {
            continue;
        }
        if let Some(previous) = out.last_mut() {
            let previous: &mut GameNoteEvent = previous;
            if event.start_sec < previous.end_sec - 1e-9 {
                // 不制造重叠：截断后一个事件的起点。
                if previous.end_sec <= event.end_sec {
                    continue;
                }
                // 前面的事件更长：保持前一个，丢弃后一个重叠段。
                continue;
            }
        }
        out.push(event);
    }
    out
}

/// 把音符事件转换成前端块需要的连续区间。
///
/// 只保留浊音事件；显示层需要连续覆盖时由调用方决定是否补休止。这里不补，
/// 因为静音不是音节，不能把休止伪装成参数块。
pub fn voiced_spans(events: &[GameNoteEvent]) -> Vec<(f64, f64, f32)> {
    let mut spans: Vec<(f64, f64, f32)> = events
        .iter()
        .filter(|event| event.voiced && event.end_sec.is_finite() && event.start_sec.is_finite())
        .map(|event| (event.start_sec, event.end_sec, event.midi_note))
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    spans
}

/// Full-source voiced note intervals. Persisted analysis and GAME baselines
/// must consume this; a Clip's audible window is only a projection filter.
pub fn full_source_voiced_spans(events: &[GameNoteEvent]) -> Vec<(f64, f64, f32)> {
    voiced_spans(events)
}

/// Display-only projection of full-source GAME events into an audible window.
///
/// This helper must never be used to persist analysis spans or GAME baselines:
/// clipping a note's source boundary destroys the identity that user edits,
/// reanalysis merges and truncated blocks rely on.
pub fn syllable_spans_for_window(
    events: &[GameNoteEvent],
    window_start_sec: f64,
    window_end_sec: f64,
) -> Vec<(f64, f64, f32)> {
    if !(window_start_sec.is_finite()
        && window_end_sec.is_finite()
        && window_end_sec > window_start_sec + 1e-12)
    {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (source_start, source_end, note) in full_source_voiced_spans(events) {
        let start = source_start.max(window_start_sec).min(window_end_sec);
        let end = source_end.max(window_start_sec).min(window_end_sec);
        if end > start + 1e-12 {
            out.push((start, end, note));
        }
    }
    out.sort_by(|a, b| a.0.total_cmp(&b.0));
    out
}

#[derive(Clone, PartialEq, Eq)]
struct GameFileStamp {
    path: PathBuf,
    bytes: u64,
    modified: Option<std::time::SystemTime>,
}

impl GameFileStamp {
    fn read(path: &Path) -> Result<Self, String> {
        let metadata = std::fs::metadata(path)
            .map_err(|error| format!("GAME source/model metadata {}: {error}", path.display()))?;
        Ok(Self {
            path: path.canonicalize().unwrap_or_else(|_| path.to_path_buf()),
            bytes: metadata.len(),
            modified: metadata.modified().ok(),
        })
    }
}

#[derive(Clone, PartialEq, Eq)]
struct GameSourceKey {
    source: GameFileStamp,
    models: Vec<GameFileStamp>,
    inference: (usize, u32, i64, u32),
}

impl GameSourceKey {
    fn read(source: &Path, models: &Path, inference: GameInferenceConfig) -> Result<Self, String> {
        let models = [
            "config.json",
            "encoder.onnx",
            "segmenter.onnx",
            "estimator.onnx",
            "bd2dur.onnx",
            "dur2bd.onnx",
        ]
        .iter()
        .map(|file| GameFileStamp::read(&models.join(file)))
        .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source: GameFileStamp::read(source)?,
            models,
            inference: (
                inference.d3pm_steps,
                inference.seg_threshold.to_bits(),
                inference.seg_radius,
                inference.est_threshold.to_bits(),
            ),
        })
    }
}

const GAME_CACHE_SOURCES: usize = 128;
const GAME_CACHE_EVENTS: usize = 100_000;

#[derive(Default)]
struct GameSourceCache {
    entries: std::collections::VecDeque<(GameSourceKey, Vec<GameNoteEvent>)>,
    event_count: usize,
}

impl GameSourceCache {
    fn get_or_compute(
        &mut self,
        key: GameSourceKey,
        compute: impl FnOnce() -> Result<Vec<GameNoteEvent>, String>,
    ) -> Result<Vec<GameNoteEvent>, String> {
        if let Some(index) = self.entries.iter().position(|(stored, _)| stored == &key) {
            let entry = self.entries.remove(index).unwrap();
            let result = entry.1.clone();
            self.entries.push_back(entry);
            return Ok(result);
        }
        let result = compute()?;
        if result.len() > GAME_CACHE_EVENTS {
            return Ok(result);
        }
        while self.entries.len() >= GAME_CACHE_SOURCES
            || self.event_count + result.len() > GAME_CACHE_EVENTS
        {
            let Some((_, evicted)) = self.entries.pop_front() else {
                break;
            };
            self.event_count -= evicted.len();
        }
        self.event_count += result.len();
        self.entries.push_back((key, result.clone()));
        Ok(result)
    }
}

/// Analyze a source once and reuse its events for all Clip windows/takes that
/// reference it. Cache only note events, with bounded memory and file/model stamps.
pub fn analyze_source_file(
    source_path: &Path,
    model_dir: &Path,
    inference: GameInferenceConfig,
) -> Result<Vec<GameNoteEvent>, String> {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<GameSourceCache>> =
        std::sync::OnceLock::new();
    let key = GameSourceKey::read(source_path, model_dir, inference)?;
    CACHE
        .get_or_init(|| std::sync::Mutex::new(GameSourceCache::default()))
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get_or_compute(key, || {
            analyze_source_file_uncached(source_path, model_dir, inference)
        })
}

/// 读取源音频并执行 GAME 分析，返回源域音符事件。
///
/// 解码与重采样都在 kernel 内完成，App 与插件可共用同一条路径。调用方负责
/// 在慢车道线程上调用；这里不写工程状态。
fn analyze_source_file_uncached(
    source_path: &Path,
    model_dir: &Path,
    inference: GameInferenceConfig,
) -> Result<Vec<GameNoteEvent>, String> {
    let config = read_config(model_dir)?;
    if !models_present(model_dir) {
        return Err(format!(
            "GAME model directory is incomplete: {}",
            model_dir.display()
        ));
    }
    if !model_dir.is_dir() {
        return Err(format!(
            "GAME model directory not found: {}",
            model_dir.display()
        ));
    }
    let (sample_rate, channels, interleaved) =
        crate::audio_utils::decode_audio_f32_interleaved(source_path)?;
    let mono = prepare_mono_44k(&interleaved, channels as usize, sample_rate)?;
    if mono.len() < GAME_HOP_SAMPLES {
        return Ok(Vec::new());
    }
    if config.samplerate != GAME_SAMPLE_RATE {
        return Err(format!(
            "GAME model sample rate {} is not supported",
            config.samplerate
        ));
    }
    #[cfg(feature = "onnx")]
    {
        infer_waveform(model_dir, &mono, inference)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = inference;
        Err("GAME inference is unavailable: the onnx feature is disabled".into())
    }
}

/// 模型目录的候选路径。与 FCPE / 声码器一致，优先使用已登记路径，再退回
/// 开发树与可执行文件同级目录。
pub fn guess_model_dir() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("resources/models/GAME"),
        manifest.join("../src-tauri/resources/models/GAME"),
        manifest.join("../../backend/src-tauri/resources/models/GAME"),
    ];
    for candidate in candidates {
        if models_present(&candidate) {
            return Some(candidate);
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let candidate = dir.join("models/GAME");
            if models_present(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

/// 显式指定模型目录（环境变量），供本机验证与自定义安装。
pub fn model_dir_from_env() -> Option<PathBuf> {
    let raw = std::env::var_os("HIFISHIFTER_GAME_MODEL_DIR")?;
    let raw = raw.to_string_lossy().trim().trim_matches('"').to_owned();
    if raw.is_empty() {
        return None;
    }
    let dir = PathBuf::from(raw);
    models_present(&dir).then_some(dir)
}

/// 运行时解析 GAME 模型目录：环境变量优先，再退回登记路径 / 开发树。
pub fn resolve_model_dir() -> Option<PathBuf> {
    model_dir_from_env()
        .or_else(|| crate::game_model_dir().map(PathBuf::from))
        .or_else(guess_model_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_event() -> GameNoteEvent {
        GameNoteEvent {
            start_sec: 0.0,
            end_sec: 2.0,
            midi_note: 60.0,
            voiced: true,
        }
    }

    fn cache_key(id: usize) -> GameSourceKey {
        GameSourceKey {
            source: GameFileStamp {
                path: PathBuf::from(format!("{id}.wav")),
                bytes: 1,
                modified: None,
            },
            models: Vec::new(),
            inference: (8, 0.2f32.to_bits(), 2, 0.2f32.to_bits()),
        }
    }

    #[test]
    fn source_cache_reuses_inference_across_clip_windows_without_sharing_mutable_results() {
        let mut cache = GameSourceCache::default();
        let mut calls = 0;
        let first = cache
            .get_or_compute(cache_key(1), || {
                calls += 1;
                Ok(vec![cache_event()])
            })
            .unwrap();
        assert_eq!(
            syllable_spans_for_window(&first, 0.0, 0.5),
            vec![(0.0, 0.5, 60.0)]
        );
        let mut second = cache
            .get_or_compute(cache_key(1), || {
                calls += 1;
                Ok(Vec::new())
            })
            .unwrap();
        assert_eq!(
            syllable_spans_for_window(&second, 1.0, 1.5),
            vec![(1.0, 1.5, 60.0)]
        );
        second.clear();
        assert_eq!(
            cache
                .get_or_compute(cache_key(1), || panic!("cached source was recomputed"))
                .unwrap()
                .len(),
            1
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn source_cache_invalidates_changed_source_models_and_inference_settings() {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let directory = std::env::temp_dir().join(format!(
            "hfs-game-cache-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let source = directory.join("audio.wav");
        std::fs::write(&source, [0u8; 4]).unwrap();
        for file in [
            "config.json",
            "encoder.onnx",
            "segmenter.onnx",
            "estimator.onnx",
            "bd2dur.onnx",
            "dur2bd.onnx",
        ] {
            std::fs::write(directory.join(file), [0u8]).unwrap();
        }
        let inference = GameInferenceConfig::default();
        let original = GameSourceKey::read(&source, &directory, inference).unwrap();
        std::fs::write(&source, [0u8; 8]).unwrap();
        let changed_source = GameSourceKey::read(&source, &directory, inference).unwrap();
        assert!(changed_source != original);
        std::fs::write(directory.join("encoder.onnx"), [0u8; 2]).unwrap();
        let changed_model = GameSourceKey::read(&source, &directory, inference).unwrap();
        assert!(changed_model != changed_source);
        let changed_options = GameSourceKey::read(
            &source,
            &directory,
            GameInferenceConfig {
                d3pm_steps: 4,
                ..inference
            },
        )
        .unwrap();
        assert!(changed_options != changed_model);
    }

    #[test]
    fn source_cache_bounds_retained_events_and_does_not_cache_failures() {
        let mut cache = GameSourceCache::default();
        assert!(cache
            .get_or_compute(cache_key(1), || Err("failed".into()))
            .is_err());
        assert!(cache.entries.is_empty());
        for id in 0..=GAME_CACHE_SOURCES {
            cache
                .get_or_compute(cache_key(id), || Ok(vec![cache_event()]))
                .unwrap();
        }
        assert_eq!(cache.entries.len(), GAME_CACHE_SOURCES);
        assert!(cache.event_count <= GAME_CACHE_EVENTS);
        let large = vec![cache_event(); GAME_CACHE_EVENTS + 1];
        cache.get_or_compute(cache_key(999), || Ok(large)).unwrap();
        assert!(!cache.entries.iter().any(|(key, _)| key == &cache_key(999)));
    }

    #[test]
    fn config_requires_the_games_timestep() {
        let dir = std::env::temp_dir().join(format!("hfs-game-config-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"samplerate":44100,"timestep":0.02,"embedding_dim":128}"#,
        )
        .unwrap();
        assert!(read_config(&dir).is_err());
    }

    #[test]
    fn prepare_mono_downsamples_and_resamples() {
        let stereo = vec![0.5f32, 0.5, -0.5, -0.5];
        let mono = prepare_mono_44k(&stereo, 2, GAME_SAMPLE_RATE).unwrap();
        assert_eq!(mono, vec![0.5, -0.5]);
    }

    #[test]
    fn split_waveform_never_exceeds_the_limit() {
        let samples = vec![0.01f32; GAME_MAX_CHUNK_SAMPLES * 2 + 10];
        let chunks = split_waveform(&samples);
        assert!(chunks.len() >= 3);
        assert!(chunks
            .iter()
            .all(|(start, end)| end > start && end - start <= GAME_MAX_CHUNK_SAMPLES));
    }

    #[test]
    fn events_are_source_domain_seconds() {
        let events = notes_from_outputs(&[0.2, 0.3], &[true, false], &[60.0, 0.0], 1.0);
        assert_eq!(events[0].start_sec, 1.0);
        assert!((events[0].end_sec - 1.2).abs() < 1e-6);
        assert!(!events[1].voiced);
        assert!((events[1].end_sec - 1.5).abs() < 1e-6);
    }

    #[test]
    fn only_voiced_events_are_projected_into_the_clip_window() {
        let events = vec![
            GameNoteEvent {
                start_sec: 0.0,
                end_sec: 0.5,
                midi_note: 60.0,
                voiced: false,
            },
            GameNoteEvent {
                start_sec: 0.5,
                end_sec: 1.5,
                midi_note: 62.0,
                voiced: true,
            },
            GameNoteEvent {
                start_sec: 1.5,
                end_sec: 2.5,
                midi_note: 64.0,
                voiced: true,
            },
        ];
        assert_eq!(
            syllable_spans_for_window(&events, 1.0, 2.0),
            vec![(1.0, 1.5, 62.0), (1.5, 2.0, 64.0)]
        );
    }

    #[test]
    fn d3pm_timesteps_cover_zero_but_never_one() {
        assert_eq!(d3pm_timesteps(4), vec![0.0, 0.25, 0.5, 0.75]);
    }

    #[test]
    #[ignore = "requires the GAME ONNX weights"]
    fn real_model_decodes_a_voiced_region_from_a_wav_file() {
        let dir = resolve_model_dir().expect("GAME model directory");
        let root = std::env::temp_dir().join(format!("hfs-game-real-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let wav_path = root.join("tone.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: GAME_SAMPLE_RATE,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
        for index in 0..GAME_SAMPLE_RATE as usize * 3 {
            let t = index as f32 / GAME_SAMPLE_RATE as f32;
            let amp = if (0.2..2.8).contains(&t) { 0.2 } else { 0.0 };
            let sample = amp * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            writer
                .write_sample((sample * i16::MAX as f32) as i16)
                .unwrap();
        }
        writer.finalize().unwrap();
        let events = analyze_source_file(&wav_path, &dir, GameInferenceConfig::default())
            .expect("GAME inference");
        assert!(events.iter().any(|event| event.voiced), "{events:?}");
        assert!(syllable_spans_for_window(&events, 0.0, 3.0)
            .iter()
            .any(|(start, end, _)| *end > *start + 0.1));
        for _ in 0..35 {
            assert_eq!(
                analyze_source_file(&wav_path, &dir, GameInferenceConfig::default())
                    .expect("cached GAME events"),
                events
            );
        }
    }

    /// 诊断用：逐个加载 GAME 子模型，显式要求 CoreML，打印实际 EP。
    ///
    /// 不参与默认测试：需要本机 GAME 权重与 macOS CoreML。用它回答“GAME
    /// 有没有真的走 CoreML”，而不是靠上层回退后的最终 EP 猜。
    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "requires the GAME ONNX weights and macOS CoreML"]
    fn coreml_ep_probe_for_each_game_submodel() {
        let dir = resolve_model_dir().expect("GAME model directory");
        for name in [
            "encoder.onnx",
            "segmenter.onnx",
            "estimator.onnx",
            "bd2dur.onnx",
            "dur2bd.onnx",
        ] {
            match crate::vocoder_ort_session::build_ort_session_with_ep(
                &dir.join(name),
                crate::vocoder_ort_session::OrtSessionRole::Game,
                Some("coreml"),
            ) {
                Ok((_session, ep)) => eprintln!("[game-coreml] {name} ep={ep}"),
                Err(error) => eprintln!("[game-coreml] {name} error={error}"),
            }
        }
    }

    /// 诊断用：逐个加载 GAME 子模型，显式要求 DirectML，打印实际 EP。
    ///
    /// 不参与默认测试：需要本机 GAME 权重与 Windows DirectML。保持独立进程运行，
    /// 这样某个 ORT/provider 组合发生原生崩溃时仍能先从输出定位到具体模型。
    #[cfg(feature = "onnx")]
    #[test]
    #[ignore = "requires the GAME ONNX weights and Windows DirectML"]
    fn directml_ep_probe_for_each_game_submodel() {
        let dir = resolve_model_dir().expect("GAME model directory");
        for name in [
            "encoder.onnx",
            "segmenter.onnx",
            "estimator.onnx",
            "bd2dur.onnx",
            "dur2bd.onnx",
        ] {
            eprintln!("[game-directml] probing {name}");
            match crate::vocoder_ort_session::build_ort_session_with_ep(
                &dir.join(name),
                crate::vocoder_ort_session::OrtSessionRole::Game,
                Some("directml"),
            ) {
                Ok((_session, ep)) => eprintln!("[game-directml] {name} ep={ep}"),
                Err(error) => eprintln!("[game-directml] {name} error={error}"),
            }
        }
    }
}
