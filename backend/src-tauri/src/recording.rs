use crate::config::RecordingSettings;
use crate::state::AppState;
use chrono::Local;
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::Emitter;

mod capture;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "windows")]
mod wasapi;

pub use capture::{AppAudioInfo, AudioDeviceInfo};
use capture::{CaptureContext, WriterMsg};

const METER_POLL_MS: u64 = 100;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingStartedInfo {
    pub start_sec: f64,
    pub output_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingFinishedInfo {
    pub start_sec: f64,
    pub duration_sec: f64,
    pub sample_rate: u32,
    pub channels: u16,
    pub peak: f32,
    pub output_path: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingStatePayload {
    pub active: bool,
    pub elapsed_sec: f64,
    pub level: f32,
    pub peak: f32,
    pub start_sec: Option<f64>,
    pub output_path: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordingMeterEvent {
    pub active: bool,
    pub elapsed_sec: f64,
    pub level: f32,
    pub peak: f32,
}

/// A recording session currently in progress.
pub struct ActiveRecording {
    start_sec: f64,
    output_path: PathBuf,
    started_at: Instant,
    stop_signal: Arc<AtomicBool>,
    level: Arc<AtomicU32>,
    peak: Arc<AtomicU32>,
    thread_join: Option<JoinHandle<Result<(), String>>>,
    writer_tx: Option<mpsc::Sender<WriterMsg>>,
    writer_join: Option<JoinHandle<Result<(), String>>>,
    meter_join: Option<JoinHandle<()>>,
}

impl Drop for ActiveRecording {
    fn drop(&mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);
        if let Some(join) = self.thread_join.take() {
            let _ = join.join();
        }
        if let Some(tx) = self.writer_tx.take() {
            let _ = tx.send(WriterMsg::Finish);
        }
        if let Some(join) = self.writer_join.take() {
            let _ = join.join();
        }
    }
}

pub fn load_settings(state: &AppState) -> RecordingSettings {
    if let Some(dir) = state.config_dir.get() {
        crate::config::load_recording_settings(dir)
    } else {
        RecordingSettings::default()
    }
}

pub fn save_settings(state: &AppState, settings: &RecordingSettings) -> RecordingSettings {
    let normalized = settings.normalized();
    if let Some(dir) = state.config_dir.get() {
        crate::config::save_recording_settings(dir, &normalized);
    }
    normalized
}

pub fn enumerate_devices() -> Vec<AudioDeviceInfo> {
    capture::enumerate_devices()
}

pub fn enumerate_applications() -> Vec<AppAudioInfo> {
    capture::enumerate_applications()
}

fn db_to_linear(db: f32) -> f32 {
    if db <= -60.0 {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

fn write_wav_thread(
    rx: mpsc::Receiver<WriterMsg>,
    path: &Path,
    channels: u16,
    sample_rate: u32,
    bit_depth: u32,
) -> Result<(), String> {
    use hound::{SampleFormat as HoundSampleFormat, WavSpec, WavWriter};

    let sample_format = if bit_depth == 32 {
        HoundSampleFormat::Float
    } else {
        HoundSampleFormat::Int
    };
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: bit_depth as u16,
        sample_format,
    };
    let mut writer =
        WavWriter::create(path, spec).map_err(|err| format!("recording_error_create_wav:{err}"))?;

    loop {
        match rx.recv() {
            Ok(WriterMsg::Data(chunk)) => {
                if bit_depth == 32 {
                    for sample in &chunk {
                        writer
                            .write_sample(sample.clamp(-1.0, 1.0))
                            .map_err(|err| format!("recording_error_write_wav:{err}"))?;
                    }
                } else if bit_depth == 16 {
                    for sample in &chunk {
                        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
                        writer
                            .write_sample(value)
                            .map_err(|err| format!("recording_error_write_wav:{err}"))?;
                    }
                } else {
                    let scale = ((1i64 << 23) - 1) as f32;
                    for sample in &chunk {
                        let value = (sample.clamp(-1.0, 1.0) * scale).round() as i32;
                        writer
                            .write_sample(value)
                            .map_err(|err| format!("recording_error_write_wav:{err}"))?;
                    }
                }
            }
            Ok(WriterMsg::Finish) | Err(_) => break,
        }
    }

    writer
        .finalize()
        .map_err(|err| format!("recording_error_finalize_wav:{err}"))?;
    Ok(())
}

fn unique_output_path(path: PathBuf) -> PathBuf {
    if !path.exists() {
        return path;
    }
    let parent = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("Recording")
        .to_string();
    let mut index = 1usize;
    loop {
        let candidate = parent.join(format!("{stem} ({index}).wav"));
        if !candidate.exists() {
            return candidate;
        }
        index += 1;
    }
}

fn resolve_output_path(state: &AppState, settings: &RecordingSettings) -> Result<PathBuf, String> {
    let template = if settings.path_template.trim().is_empty() {
        RecordingSettings::default().path_template
    } else {
        settings.path_template.trim().to_string()
    };
    let project_folder = crate::commands::project::resolve_project_folder_for_backup(state)
        .display()
        .to_string();
    let project_name = crate::commands::project::resolve_project_name_for_backup(state);
    let replaced = template
        .replace("<ProjectFolder>", &project_folder)
        .replace("<ProjectName>", &project_name);
    let (formatted, _fallback_used) =
        crate::commands::project::try_apply_time_format_with_fallback(&replaced, Local::now())
            .map_err(|_| "recording_error_invalid_time_format".to_string())?;
    let trimmed = formatted.trim();
    if trimmed.is_empty() {
        return Err("recording_error_empty_path".to_string());
    }

    let mut path = PathBuf::from(trimmed);
    if path.is_relative() {
        path = PathBuf::from(project_folder).join(path);
    }
    if path.extension().is_none() {
        path.set_extension("wav");
    }
    Ok(unique_output_path(path))
}

fn meter_loop(
    app_handle: Option<tauri::AppHandle>,
    stop_signal: Arc<AtomicBool>,
    level: Arc<AtomicU32>,
    peak: Arc<AtomicU32>,
    started_at: Instant,
) {
    loop {
        if stop_signal.load(Ordering::Relaxed) {
            break;
        }
        let event = RecordingMeterEvent {
            active: true,
            elapsed_sec: started_at.elapsed().as_secs_f64(),
            level: f32::from_bits(level.load(Ordering::Relaxed)),
            peak: f32::from_bits(peak.load(Ordering::Relaxed)),
        };
        if let Some(app) = &app_handle {
            let _ = app.emit("recording-meter", event);
        }
        std::thread::sleep(Duration::from_millis(METER_POLL_MS));
    }
}

pub fn start(state: &AppState, start_sec: f64) -> Result<RecordingStartedInfo, String> {
    let mut recording_guard = state
        .recording
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    if recording_guard.is_some() {
        return Err("recording_error_already_active".to_string());
    }

    let settings = load_settings(state);
    let output_path = resolve_output_path(state, &settings)?;
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|err| format!("recording_error_create_dir:{err}"))?;
        }
    }

    // Resolve the capture plan synchronously so common configuration errors
    // (unknown app, bad source) fail fast with a clean message.
    let plan = capture::CapturePlan::from_settings(&settings)?;

    let (writer_tx, writer_rx) = mpsc::channel();
    let writer_path = output_path.clone();
    let writer_channels = settings.channels;
    let writer_sample_rate = settings.sample_rate;
    let writer_bit_depth = settings.bit_depth;
    let writer_join = std::thread::spawn(move || {
        write_wav_thread(
            writer_rx,
            &writer_path,
            writer_channels,
            writer_sample_rate,
            writer_bit_depth,
        )
    });

    let stop_signal = Arc::new(AtomicBool::new(false));
    let level = Arc::new(AtomicU32::new(0.0f32.to_bits()));
    let peak = Arc::new(AtomicU32::new(0.0f32.to_bits()));
    let monitor_queue = if settings.monitor_enabled {
        Some(Arc::new(Mutex::new(VecDeque::new())))
    } else {
        None
    };
    let ctx = Arc::new(CaptureContext {
        tx: writer_tx.clone(),
        stop: stop_signal.clone(),
        level: level.clone(),
        peak: peak.clone(),
        monitor_queue,
        gain: db_to_linear(settings.input_gain_db),
        monitor_gain: db_to_linear(settings.monitor_gain_db),
    });

    // The capture stream is created, played and destroyed on a dedicated
    // thread because platform streams (WASAPI COM objects, PipeWire children,
    // cpal streams) are not Send.
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let thread_ctx = ctx.clone();
    let thread_join = std::thread::spawn(move || capture::run_capture(plan, thread_ctx, ready_tx));

    let readiness = ready_rx.recv_timeout(Duration::from_secs(8));
    match &readiness {
        Ok(Ok(())) => {}
        Ok(Err(_)) | Err(_) => {
            stop_signal.store(true, Ordering::Relaxed);
            let _ = writer_tx.send(WriterMsg::Finish);
            let _ = thread_join.join();
            let _ = writer_join.join();
            let _ = std::fs::remove_file(&output_path);
            let error = match readiness {
                Ok(Err(err)) => err,
                _ => "recording_error_start_timeout".to_string(),
            };
            return Err(error);
        }
    }

    let started_at = Instant::now();
    let app_handle = state.app_handle.get().cloned();
    let meter_stop = stop_signal.clone();
    let meter_level = level.clone();
    let meter_peak = peak.clone();
    let meter_join = std::thread::spawn(move || {
        meter_loop(app_handle, meter_stop, meter_level, meter_peak, started_at)
    });

    *recording_guard = Some(ActiveRecording {
        start_sec,
        output_path: output_path.clone(),
        started_at,
        stop_signal,
        level,
        peak,
        thread_join: Some(thread_join),
        writer_tx: Some(writer_tx),
        writer_join: Some(writer_join),
        meter_join: Some(meter_join),
    });

    Ok(RecordingStartedInfo {
        start_sec,
        output_path: output_path.display().to_string(),
    })
}

pub fn stop(state: &AppState) -> Result<RecordingFinishedInfo, String> {
    let mut recording_guard = state
        .recording
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let mut active = recording_guard
        .take()
        .ok_or_else(|| "recording_error_not_active".to_string())?;

    active.stop_signal.store(true, Ordering::Relaxed);
    let thread_result = match active.thread_join.take() {
        Some(join) => join
            .join()
            .map_err(|_| "recording_error_thread_panic".to_string())?,
        None => Ok(()),
    };
    if let Err(err) = thread_result {
        let _ = std::fs::remove_file(&active.output_path);
        return Err(err);
    }

    if let Some(tx) = active.writer_tx.take() {
        let _ = tx.send(WriterMsg::Finish);
    }

    let writer_result = match active.writer_join.take() {
        Some(join) => join
            .join()
            .map_err(|_| "recording_error_writer_panic".to_string())?,
        None => Ok(()),
    };
    if let Err(err) = writer_result {
        let _ = std::fs::remove_file(&active.output_path);
        return Err(err);
    }

    if let Some(join) = active.meter_join.take() {
        let _ = join.join();
    }

    let info = crate::audio_utils::try_read_wav_info(&active.output_path, 256)
        .ok_or_else(|| "recording_error_output_missing".to_string());
    let info = match info {
        Ok(info) => info,
        Err(err) => {
            let _ = std::fs::remove_file(&active.output_path);
            return Err(err);
        }
    };
    let peak = f32::from_bits(active.peak.load(Ordering::Relaxed));
    let channels = read_wav_channels(&active.output_path).unwrap_or(2);

    Ok(RecordingFinishedInfo {
        start_sec: active.start_sec,
        duration_sec: info.duration_sec,
        sample_rate: info.sample_rate,
        channels,
        peak,
        output_path: active.output_path.display().to_string(),
    })
}

fn read_wav_channels(path: &Path) -> Option<u16> {
    hound::WavReader::open(path)
        .ok()
        .map(|reader| reader.spec().channels)
}

pub fn current_state(state: &AppState) -> RecordingStatePayload {
    let recording_guard = state
        .recording
        .lock()
        .unwrap_or_else(|err| err.into_inner());
    let Some(active) = recording_guard.as_ref() else {
        return RecordingStatePayload {
            active: false,
            elapsed_sec: 0.0,
            level: 0.0,
            peak: 0.0,
            start_sec: None,
            output_path: None,
            error: None,
        };
    };
    RecordingStatePayload {
        active: true,
        elapsed_sec: active.started_at.elapsed().as_secs_f64(),
        level: f32::from_bits(active.level.load(Ordering::Relaxed)),
        peak: f32::from_bits(active.peak.load(Ordering::Relaxed)),
        start_sec: Some(active.start_sec),
        output_path: Some(active.output_path.display().to_string()),
        error: None,
    }
}

pub fn shutdown(state: &AppState) {
    if state
        .recording
        .lock()
        .map(|guard| guard.is_some())
        .unwrap_or(false)
    {
        let _ = stop(state);
    }
}
