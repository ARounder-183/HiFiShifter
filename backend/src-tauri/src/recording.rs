use crate::config::RecordingSettings;
use crate::state::AppState;
use chrono::Local;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tauri::Emitter;

const METER_POLL_MS: u64 = 100;
const MONITOR_QUEUE_MAX_SAMPLES: usize = 262_144;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub is_default: bool,
    pub is_loopback: bool,
}

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

enum WriterMsg {
    Data(Vec<f32>),
    Finish,
}

/// 一次进行中的录音会话。
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
    let host = cpal::default_host();
    let mut devices = Vec::new();

    devices.push(AudioDeviceInfo {
        id: "default".to_string(),
        name: "System Default".to_string(),
        kind: "default".to_string(),
        is_default: true,
        is_loopback: false,
    });

    if let Ok(inputs) = host.input_devices() {
        for device in inputs {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            devices.push(AudioDeviceInfo {
                id: format!("input:{name}"),
                name,
                kind: "input".to_string(),
                is_default: false,
                is_loopback: false,
            });
        }
    }

    // Windows WASAPI 会把输出设备用作输入流时透明开启 loopback；
    // 其他平台可能不支持，但保留选项以便用户尝试系统音频捕获。
    if let Ok(outputs) = host.output_devices() {
        for device in outputs {
            let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
            devices.push(AudioDeviceInfo {
                id: format!("loopback:{name}"),
                name,
                kind: "output".to_string(),
                is_default: false,
                is_loopback: true,
            });
        }
    }

    devices
}

fn resolve_device(settings: &RecordingSettings) -> Result<(cpal::Device, bool), String> {
    let host = cpal::default_host();
    if settings.source_device == "default" {
        return host
            .default_input_device()
            .map(|device| (device, false))
            .ok_or_else(|| "recording_error_no_default_input".to_string());
    }

    if let Some(name) = settings.source_device.strip_prefix("input:") {
        let devices = host
            .input_devices()
            .map_err(|_| "recording_error_enumeration".to_string())?;
        for device in devices {
            if let Ok(device_name) = device.name() {
                if device_name == name {
                    return Ok((device, false));
                }
            }
        }
        return Err("recording_error_device_not_found".to_string());
    }

    if let Some(name) = settings.source_device.strip_prefix("loopback:") {
        let devices = host
            .output_devices()
            .map_err(|_| "recording_error_enumeration".to_string())?;
        for device in devices {
            if let Ok(device_name) = device.name() {
                if device_name == name {
                    return Ok((device, true));
                }
            }
        }
        return Err("recording_error_loopback_not_found".to_string());
    }

    Err("recording_error_unknown_source".to_string())
}

fn pick_input_config(
    device: &cpal::Device,
    settings: &RecordingSettings,
    is_loopback: bool,
) -> Result<(cpal::StreamConfig, SampleFormat), String> {
    let want_rate = cpal::SampleRate(settings.sample_rate);
    let want_channels = settings.channels;
    let mut fallback: Option<(cpal::StreamConfig, SampleFormat)> = None;

    if is_loopback {
        if let Ok(default_config) = device.default_output_config() {
            let sample_format = default_config.sample_format();
            let config: cpal::StreamConfig = default_config.into();
            fallback = Some((config, sample_format));
        }
        if let Ok(ranges) = device.supported_output_configs() {
            for range in ranges {
                if range.channels() != want_channels {
                    continue;
                }
                if let Some(config) = range.try_with_sample_rate(want_rate) {
                    let sample_format = config.sample_format();
                    return Ok((config.into(), sample_format));
                }
            }
        }
    } else {
        if let Ok(default_config) = device.default_input_config() {
            let sample_format = default_config.sample_format();
            let default_channels = default_config.channels();
            let default_rate = default_config.sample_rate();
            let config: cpal::StreamConfig = default_config.into();
            if default_channels == want_channels && default_rate == want_rate {
                return Ok((config, sample_format));
            }
            fallback = Some((config, sample_format));
        }
        if let Ok(ranges) = device.supported_input_configs() {
            for range in ranges {
                if range.channels() != want_channels {
                    continue;
                }
                if let Some(config) = range.try_with_sample_rate(want_rate) {
                    let sample_format = config.sample_format();
                    return Ok((config.into(), sample_format));
                }
            }
        }
    }

    fallback.ok_or_else(|| "recording_error_no_supported_config".to_string())
}

fn db_to_linear(db: f32) -> f32 {
    if db <= -60.0 {
        0.0
    } else {
        10f32.powf(db / 20.0)
    }
}

fn send_to_writer_and_monitor(
    samples: &[f32],
    channels: usize,
    gain: f32,
    monitor_gain: f32,
    tx: &mpsc::Sender<WriterMsg>,
    level: &AtomicU32,
    peak: &AtomicU32,
    monitor_queue: &Option<Arc<Mutex<VecDeque<f32>>>>,
) {
    if samples.is_empty() {
        return;
    }

    let mut out: Vec<f32> = Vec::with_capacity(samples.len());
    let mut max = 0.0f32;
    for sample in samples {
        let value = (sample * gain).clamp(-1.0, 1.0);
        out.push(value);
        let magnitude = value.abs();
        if magnitude > max {
            max = magnitude;
        }
    }
    let _ = tx.send(WriterMsg::Data(out));

    level.store(max.to_bits(), Ordering::Relaxed);
    let old_peak = f32::from_bits(peak.load(Ordering::Relaxed));
    if max > old_peak {
        peak.store(max.to_bits(), Ordering::Relaxed);
    }

    if let Some(queue) = monitor_queue {
        if let Ok(mut q) = queue.lock() {
            if q.len() > MONITOR_QUEUE_MAX_SAMPLES {
                q.clear();
            }
            q.extend(samples.iter().map(|s| *s * gain * monitor_gain));
        }
    }

    let _ = channels;
}

fn handle_input_data(
    data: &cpal::Data,
    channels: usize,
    gain: f32,
    monitor_gain: f32,
    tx: &mpsc::Sender<WriterMsg>,
    level: &AtomicU32,
    peak: &AtomicU32,
    monitor_queue: &Option<Arc<Mutex<VecDeque<f32>>>>,
) {
    if let Some(samples) = data.as_slice::<f32>() {
        send_to_writer_and_monitor(
            samples,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<i8>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<u8>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<i16>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<u16>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<i32>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<u32>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample())
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<i64>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample() as f32)
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<u64>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample() as f32)
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else if let Some(samples) = data.as_slice::<f64>() {
        let converted: Vec<f32> = samples
            .iter()
            .map(|s| s.to_float_sample() as f32)
            .collect();
        send_to_writer_and_monitor(
            &converted,
            channels,
            gain,
            monitor_gain,
            tx,
            level,
            peak,
            monitor_queue,
        );
    } else {
        // 不支持的采样格式：静默处理，避免音频回调崩溃。
    }
}

fn fill_monitor_output<T: SizedSample + Sample + cpal::FromSample<f32>>(
    samples: &mut [T],
    queue: &Mutex<VecDeque<f32>>,
) {
    let len = samples.len();
    let mut chunk = Vec::with_capacity(len);
    if let Ok(mut q) = queue.lock() {
        let take = len.min(q.len());
        for _ in 0..take {
            chunk.push(q.pop_front().unwrap_or(0.0));
        }
    }
    for (index, sample) in samples.iter_mut().enumerate() {
        let value = if index < chunk.len() { chunk[index] } else { 0.0 };
        *sample = <f32 as cpal::Sample>::to_sample(value);
    }
}

fn build_monitor_stream(
    device: &cpal::Device,
    queue: Arc<Mutex<VecDeque<f32>>>,
) -> Result<cpal::Stream, String> {
    let (config, sample_format) = pick_monitor_config(device)?;
    let err_callback = |err: cpal::StreamError| {
        eprintln!("[recording] monitor stream error: {err}");
    };
    let stream = device
        .build_output_stream_raw(
            &config,
            sample_format,
            move |data: &mut cpal::Data, _: &cpal::OutputCallbackInfo| {
                if let Some(samples) = data.as_slice_mut::<f32>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<i8>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<u8>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<i16>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<u16>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<i32>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<u32>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<i64>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<u64>() {
                    fill_monitor_output(samples, &queue);
                } else if let Some(samples) = data.as_slice_mut::<f64>() {
                    fill_monitor_output(samples, &queue);
                }
            },
            err_callback,
            None,
        )
        .map_err(|_| "recording_error_build_monitor".to_string())?;
    stream
        .play()
        .map_err(|_| "recording_error_play_monitor".to_string())?;
    Ok(stream)
}

fn pick_monitor_config(
    device: &cpal::Device,
) -> Result<(cpal::StreamConfig, SampleFormat), String> {
    if let Ok(ranges) = device.supported_output_configs() {
        for range in ranges {
            if range.sample_format() != SampleFormat::F32 {
                continue;
            }
            let config = range
                .try_with_sample_rate(cpal::SampleRate(48_000))
                .or_else(|| Some(range.with_max_sample_rate()));
            if let Some(config) = config {
                return Ok((config.into(), SampleFormat::F32));
            }
        }
    }
    if let Ok(default_config) = device.default_output_config() {
        let sample_format = default_config.sample_format();
        return Ok((default_config.into(), sample_format));
    }
    Err("recording_error_no_monitor_config".to_string())
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
    let mut writer = WavWriter::create(path, spec)
        .map_err(|err| format!("recording_error_create_wav:{err}"))?;

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

    // 设备/配置解析在主线程完成，常见的“找不到设备/不支持配置”错误可以同步返回。
    let (device, is_loopback) = resolve_device(&settings)?;
    let (config, sample_format) = pick_input_config(&device, &settings, is_loopback)?;

    let (writer_tx, writer_rx) = mpsc::channel();
    let writer_path = output_path.clone();
    let writer_channels = config.channels;
    let writer_sample_rate = config.sample_rate.0;
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

    // 录音线程持有 cpal::Stream（Windows 上包含非 Send 的 COM 指针），
    // 因此流只能在独立线程中创建、播放并销毁，AppState 只保存 JoinHandle。
    let thread_stop = stop_signal.clone();
    let thread_level = level.clone();
    let thread_peak = peak.clone();
    let thread_writer_tx = writer_tx.clone();
    let thread_settings = settings.clone();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<(), String>>();
    let thread_join = std::thread::spawn(move || {
        let channels = config.channels as usize;
        let gain = db_to_linear(thread_settings.input_gain_db);
        let monitor_gain = db_to_linear(thread_settings.monitor_gain_db);

        let monitor_queue = if thread_settings.monitor_enabled {
            Some(Arc::new(Mutex::new(VecDeque::new())))
        } else {
            None
        };
        let mut monitor_stream = None;
        if thread_settings.monitor_enabled {
            let host = cpal::default_host();
            if let Some(output_device) = host.default_output_device() {
                if let Some(queue) = monitor_queue.clone() {
                    match build_monitor_stream(&output_device, queue) {
                        Ok(stream) => monitor_stream = Some(stream),
                        Err(err) => eprintln!("[recording] monitor unavailable: {err}"),
                    }
                }
            }
        }

        let tx_callback = thread_writer_tx.clone();
        let level_callback = thread_level.clone();
        let peak_callback = thread_peak.clone();
        let monitor_callback = monitor_queue.clone();
        let err_callback = |err: cpal::StreamError| {
            eprintln!("[recording] input stream error: {err}");
        };
        let stream = match device
            .build_input_stream_raw(
                &config,
                sample_format,
                move |data: &cpal::Data, _: &cpal::InputCallbackInfo| {
                    handle_input_data(
                        data,
                        channels,
                        gain,
                        monitor_gain,
                        &tx_callback,
                        &level_callback,
                        &peak_callback,
                        &monitor_callback,
                    );
                },
                err_callback,
                None,
            )
        {
            Ok(stream) => stream,
            Err(_) => {
                let _ = ready_tx.send(Err("recording_error_build_input".to_string()));
                return Err("recording_error_build_input".to_string());
            }
        };
        if let Err(_) = stream.play() {
            let _ = ready_tx.send(Err("recording_error_play_input".to_string()));
            return Err("recording_error_play_input".to_string());
        }
        let _ = ready_tx.send(Ok(()));

        // 阻塞等待停止信号，确保 Stream 在本线程中自然 Drop。
        while !thread_stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(10));
        }
        drop(stream);
        drop(monitor_stream);
        Ok(())
    });

    let readiness = ready_rx.recv_timeout(Duration::from_secs(5));
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
