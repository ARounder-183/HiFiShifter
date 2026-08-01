//! Recording capture engine.
//!
//! All capture paths converge on interleaved `f32` frames at the sample rate
//! and channel count requested in `RecordingSettings`:
//!
//! * Microphone / input devices use cpal.
//! * System-sound (loopback) capture on Windows uses a native WASAPI engine
//!   (see [`wasapi`]) that honors `AUDCLNT_BUFFERFLAGS_SILENT`; cpal's WASAPI
//!   loopback does not, which causes undefined buffer bytes (audible hiss) to
//!   be written whenever the endpoint is idle.
//! * Application audio capture on Windows uses the OS process-loopback API
//!   (`ActivateAudioInterfaceAsync` + `AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS`,
//!   Windows 10 build 20348+) with a session-muting fallback on older builds.
//! * On Linux, application capture is implemented through PipeWire's
//!   `pw-dump`/`pw-loopback` when available.
//! * On macOS the OS offers no per-application capture API, so that mode
//!   reports a localized error and system loopback relies on cpal (e.g. a
//!   BlackHole/Soundflower virtual device).

use crate::config::RecordingSettings;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Sample, SampleFormat, SizedSample};
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

pub const MONITOR_QUEUE_MAX_SAMPLES: usize = 262_144;

/// Messages sent from the capture thread to the WAV writer thread.
#[derive(Debug)]
pub enum WriterMsg {
    Data(Vec<f32>),
    Finish,
}

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
pub struct AppAudioInfo {
    /// Stable id for the selected application, e.g. `pid:1234`.
    pub id: String,
    /// Human-readable name shown in the UI.
    pub name: String,
    /// Executable/binary name used to re-match the app after a restart.
    pub process_name: String,
    pub pid: u32,
    /// Whether the app currently has an active audio session.
    pub is_active: bool,
}

/// Shared state between the capture thread, the WAV writer and the meter loop.
pub struct CaptureContext {
    pub tx: mpsc::Sender<WriterMsg>,
    pub stop: Arc<AtomicBool>,
    pub level: Arc<AtomicU32>,
    pub peak: Arc<AtomicU32>,
    pub monitor_queue: Option<Arc<Mutex<VecDeque<f32>>>>,
    pub gain: f32,
    pub monitor_gain: f32,
}

impl CaptureContext {
    /// Applies input gain, updates level/peak meters, feeds the monitor queue
    /// and forwards the chunk to the WAV writer.
    pub fn push(&self, samples: &[f32]) {
        if samples.is_empty() {
            return;
        }
        let mut out: Vec<f32> = Vec::with_capacity(samples.len());
        let mut max = 0.0f32;
        for sample in samples {
            let value = (sample * self.gain).clamp(-1.0, 1.0);
            out.push(value);
            let magnitude = value.abs();
            if magnitude > max {
                max = magnitude;
            }
        }
        let _ = self.tx.send(WriterMsg::Data(out));

        self.level.store(max.to_bits(), Ordering::Relaxed);
        let old_peak = f32::from_bits(self.peak.load(Ordering::Relaxed));
        if max > old_peak {
            self.peak.store(max.to_bits(), Ordering::Relaxed);
        }

        if let Some(queue) = &self.monitor_queue {
            if let Ok(mut q) = queue.lock() {
                if q.len() > MONITOR_QUEUE_MAX_SAMPLES {
                    q.clear();
                }
                q.extend(samples.iter().map(|s| *s * self.gain * self.monitor_gain));
            }
        }
    }
}

/// Fully resolved capture source.
#[derive(Debug, Clone)]
pub enum CapturePlan {
    Device {
        source: String,
        sample_rate: u32,
        channels: u16,
    },
    Loopback {
        device_id: String,
        sample_rate: u32,
        channels: u16,
    },
    Application {
        pid: u32,
        sample_rate: u32,
        channels: u16,
    },
}

impl CapturePlan {
    pub fn from_settings(settings: &RecordingSettings) -> Result<Self, String> {
        let sample_rate = settings.sample_rate;
        let channels = settings.channels;
        match settings.capture_mode.as_str() {
            "loopback" => Ok(CapturePlan::Loopback {
                device_id: settings.loopback_device.trim().to_string(),
                sample_rate,
                channels,
            }),
            "application" => {
                let pid = resolve_application_pid(settings)
                    .ok_or_else(|| "recording_error_app_not_found".to_string())?;
                Ok(CapturePlan::Application {
                    pid,
                    sample_rate,
                    channels,
                })
            }
            // Default ("device"), plus migration of legacy
            // `source_device = "loopback:<name>"` values saved by older builds.
            _ => {
                if let Some(name) = settings.source_device.strip_prefix("loopback:") {
                    Ok(CapturePlan::Loopback {
                        device_id: name.trim().to_string(),
                        sample_rate,
                        channels,
                    })
                } else {
                    Ok(CapturePlan::Device {
                        source: settings.source_device.clone(),
                        sample_rate,
                        channels,
                    })
                }
            }
        }
    }
}

fn parse_pid(id: &str) -> Option<u32> {
    let id = id.trim();
    if let Some(pid) = id.strip_prefix("pid:") {
        return pid.trim().parse().ok();
    }
    id.parse().ok()
}

/// Resolves the target PID for application capture. The saved PID is used
/// while it still matches the saved process name; if the app restarted (new
/// PID), the process name is used to find its current audio session.
fn resolve_application_pid(settings: &RecordingSettings) -> Option<u32> {
    let requested_pid = parse_pid(&settings.capture_app_id);
    let process_name = settings.capture_app_process.trim();
    let apps = enumerate_applications();
    if let Some(pid) = requested_pid {
        if let Some(app) = apps.iter().find(|app| app.pid == pid) {
            if process_name.is_empty() || app.process_name == process_name {
                return Some(pid);
            }
        }
    }
    if !process_name.is_empty() {
        if let Some(app) = apps.iter().find(|app| app.process_name == process_name) {
            return Some(app.pid);
        }
    }
    requested_pid
}

/// Runs the selected capture source until `ctx.stop` is set. Blocks the
/// calling thread; the WAV writer and monitor loop run on their own threads.
pub fn run_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    // Optional monitor: send the captured signal back to the default output
    // device so the user can hear what is being recorded.
    let mut monitor_stream = None;
    if ctx.monitor_queue.is_some() {
        let host = cpal::default_host();
        if let Some(output_device) = host.default_output_device() {
            if let Some(queue) = ctx.monitor_queue.clone() {
                match build_monitor_stream(&output_device, queue) {
                    Ok(stream) => monitor_stream = Some(stream),
                    Err(err) => eprintln!("[recording] monitor unavailable: {err}"),
                }
            }
        }
    }

    let result = match plan {
        CapturePlan::Device { .. } => run_device_capture(plan, ctx, ready_tx),
        CapturePlan::Loopback { .. } => run_loopback_capture(plan, ctx, ready_tx),
        CapturePlan::Application { .. } => run_application_capture(plan, ctx, ready_tx),
    };
    drop(monitor_stream);
    result
}

fn run_device_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let CapturePlan::Device {
        source,
        sample_rate,
        channels,
    } = plan
    else {
        unreachable!()
    };
    let device = resolve_input_device(&source)?;
    run_cpal_input(device, false, sample_rate, channels, ctx, ready_tx)
}

#[cfg(target_os = "windows")]
fn run_loopback_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    super::wasapi::run_loopback_capture(plan, ctx, ready_tx)
}

#[cfg(not(target_os = "windows"))]
fn run_loopback_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let CapturePlan::Loopback {
        device_id,
        sample_rate,
        channels,
    } = plan
    else {
        unreachable!()
    };
    let device = resolve_loopback_device(&device_id)?;
    run_cpal_input(device, true, sample_rate, channels, ctx, ready_tx)
}

#[cfg(target_os = "windows")]
fn run_application_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    super::wasapi::run_app_capture(plan, ctx, ready_tx)
}

#[cfg(target_os = "linux")]
fn run_application_capture(
    plan: CapturePlan,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    super::linux::run_app_capture(plan, ctx, ready_tx)
}

#[cfg(target_os = "macos")]
fn run_application_capture(
    _plan: CapturePlan,
    _ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let _ = ready_tx.send(Err("recording_error_app_capture_unsupported".to_string()));
    Err("recording_error_app_capture_unsupported".to_string())
}

#[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
fn run_application_capture(
    _plan: CapturePlan,
    _ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let _ = ready_tx.send(Err("recording_error_app_capture_unsupported".to_string()));
    Err("recording_error_app_capture_unsupported".to_string())
}

/// cpal-based capture used for microphones everywhere and for loopback on
/// platforms without a native loopback engine.
fn run_cpal_input(
    device: cpal::Device,
    is_loopback: bool,
    sample_rate: u32,
    channels: u16,
    ctx: Arc<CaptureContext>,
    ready_tx: mpsc::Sender<Result<(), String>>,
) -> Result<(), String> {
    let (config, sample_format) =
        pick_input_config(&device, sample_rate, channels, is_loopback)?;
    let channels_in = config.channels as usize;
    let channels_out = channels as usize;
    let resampler = if config.sample_rate.0 != sample_rate {
        Some(RateConverter::new(
            config.sample_rate.0,
            sample_rate,
            channels_out,
            channels_out,
        )?)
    } else {
        None
    };
    let resampler = Arc::new(Mutex::new(resampler));
    let err_callback = |err: cpal::StreamError| {
        eprintln!("[recording] input stream error: {err}");
    };
    let capture_ctx = ctx.clone();
    let capture_resampler = resampler.clone();
    let stream = device
        .build_input_stream_raw(
            &config,
            sample_format,
            move |data: &cpal::Data, _: &cpal::InputCallbackInfo| {
                let samples = decode_input_data(data, channels_in, channels_out);
                if samples.is_empty() {
                    return;
                }
                let mut guard = capture_resampler
                    .lock()
                    .unwrap_or_else(|err| err.into_inner());
                if let Some(converter) = guard.as_mut() {
                    if let Ok(out) = converter.push_interleaved(&samples, channels_out) {
                        capture_ctx.push(&out);
                    }
                } else {
                    capture_ctx.push(&samples);
                }
            },
            err_callback,
            None,
        )
        .map_err(|_| "recording_error_build_input".to_string())?;
    stream
        .play()
        .map_err(|_| "recording_error_play_input".to_string())?;
    let _ = ready_tx.send(Ok(()));

    while !ctx.stop.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(10));
    }
    drop(stream);
    // Flush whatever is still sitting in the resampler so the final ~20 ms of
    // audio is not lost.
    if let Ok(mut guard) = resampler.lock() {
        if let Some(converter) = guard.as_mut() {
            let out = converter.flush_interleaved(channels_out, channels_out);
            if !out.is_empty() {
                ctx.push(&out);
            }
        }
    }
    Ok(())
}

fn resolve_input_device(source: &str) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    if source == "default" {
        return host
            .default_input_device()
            .ok_or_else(|| "recording_error_no_default_input".to_string());
    }
    let name = source
        .strip_prefix("input:")
        .ok_or_else(|| "recording_error_unknown_source".to_string())?;
    let devices = host
        .input_devices()
        .map_err(|_| "recording_error_enumeration".to_string())?;
    for device in devices {
        if let Ok(device_name) = device.name() {
            if device_name == name {
                return Ok(device);
            }
        }
    }
    Err("recording_error_device_not_found".to_string())
}

#[cfg(not(target_os = "windows"))]
fn resolve_loopback_device(device_id: &str) -> Result<cpal::Device, String> {
    let host = cpal::default_host();
    if device_id.is_empty() || device_id == "default" || device_id == "loopback:default" {
        return host
            .default_output_device()
            .ok_or_else(|| "recording_error_loopback_not_found".to_string());
    }
    let name = device_id.strip_prefix("loopback:").unwrap_or(device_id);
    let devices = host
        .output_devices()
        .map_err(|_| "recording_error_enumeration".to_string())?;
    for device in devices {
        if let Ok(device_name) = device.name() {
            if device_name == name {
                return Ok(device);
            }
        }
    }
    Err("recording_error_loopback_not_found".to_string())
}

fn pick_input_config(
    device: &cpal::Device,
    sample_rate: u32,
    channels: u16,
    is_loopback: bool,
) -> Result<(cpal::StreamConfig, SampleFormat), String> {
    let want_rate = cpal::SampleRate(sample_rate);
    let mut fallback: Option<(cpal::StreamConfig, SampleFormat)> = None;

    if is_loopback {
        if let Ok(default_config) = device.default_output_config() {
            let sample_format = default_config.sample_format();
            let config: cpal::StreamConfig = default_config.into();
            fallback = Some((config, sample_format));
        }
        if let Ok(ranges) = device.supported_output_configs() {
            for range in ranges {
                if range.channels() != channels {
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
            if default_channels == channels && default_rate == want_rate {
                return Ok((config, sample_format));
            }
            fallback = Some((config, sample_format));
        }
        if let Ok(ranges) = device.supported_input_configs() {
            for range in ranges {
                if range.channels() != channels {
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

fn decode_input_data(data: &cpal::Data, channels_in: usize, channels_out: usize) -> Vec<f32> {
    let decoded: Option<Vec<f32>> = if let Some(samples) = data.as_slice::<f32>() {
        Some(samples.to_vec())
    } else if let Some(samples) = data.as_slice::<i8>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<u8>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<i16>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<u16>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<i32>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<u32>() {
        Some(samples.iter().map(|s| s.to_float_sample()).collect())
    } else if let Some(samples) = data.as_slice::<i64>() {
        Some(samples.iter().map(|s| s.to_float_sample() as f32).collect())
    } else if let Some(samples) = data.as_slice::<u64>() {
        Some(samples.iter().map(|s| s.to_float_sample() as f32).collect())
    } else if let Some(samples) = data.as_slice::<f64>() {
        Some(samples.iter().map(|s| s.to_float_sample() as f32).collect())
    } else {
        // Unsupported sample format: stay silent to avoid feeding garbage.
        None
    };
    decoded
        .map(|samples| convert_channels(&samples, channels_in, channels_out))
        .unwrap_or_default()
}

/// Channel count conversion (1/2 in and out; 3+ sources downmix to stereo).
pub(super) fn convert_channels(
    samples: &[f32],
    channels_in: usize,
    channels_out: usize,
) -> Vec<f32> {
    let channels_in = channels_in.max(1);
    let channels_out = channels_out.max(1);
    let frames = samples.len() / channels_in;
    let mut out = Vec::with_capacity(frames * channels_out);
    for frame in 0..frames {
        let base = frame * channels_in;
        match (channels_in, channels_out) {
            (_, 1) => {
                let sum: f32 = samples[base..base + channels_in].iter().sum();
                out.push((sum / channels_in as f32).clamp(-1.0, 1.0));
            }
            (1, 2) => {
                let value = samples[base].clamp(-1.0, 1.0);
                out.push(value);
                out.push(value);
            }
            (2, 2) => {
                out.push(samples[base].clamp(-1.0, 1.0));
                out.push(samples[base + 1].clamp(-1.0, 1.0));
            }
            (n, 2) => {
                // Standard ITU downmix: FL, FR, FC*0.707, BL*0.707, BR*0.707,
                // LFE*0.5 (channel 3 = FC, 4 = LFE, 5 = BL, 6 = BR, 7 = SL,
                // 8 = SR in the common 5.1/7.1 layouts).
                let get = |index: usize| {
                    if index < n {
                        samples[base + index]
                    } else {
                        0.0
                    }
                };
                let left = get(0) + 0.7071 * get(2) + 0.7071 * get(4) + 0.7071 * get(6)
                    + 0.5 * get(3);
                let right = get(1) + 0.7071 * get(2) + 0.7071 * get(5) + 0.7071 * get(7)
                    + 0.5 * get(3);
                out.push(left.clamp(-1.0, 1.0));
                out.push(right.clamp(-1.0, 1.0));
            }
            _ => {
                // Rare layout (e.g. requested >2 channels): pass through the
                // first `channels_out` channels.
                for index in 0..channels_out.min(channels_in) {
                    out.push(samples[base + index].clamp(-1.0, 1.0));
                }
            }
        }
    }
    out
}

/// High-quality rate conversion used when the capture device cannot deliver
/// the requested sample rate (cpal paths and the WASAPI fallback).
pub(super) struct RateConverter {
    resampler: rubato::SincFixedIn<f32>,
    buffers: Vec<VecDeque<f32>>,
    channels_out: usize,
}

impl RateConverter {
    pub(super) fn new(
        rate_in: u32,
        rate_out: u32,
        channels_in: usize,
        channels_out: usize,
    ) -> Result<Self, String> {
        let params = rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: rubato::SincInterpolationType::Nearest,
            window: rubato::WindowFunction::BlackmanHarris2,
        };
        let ratio = rate_out as f64 / rate_in as f64;
        let resampler = rubato::SincFixedIn::<f32>::new(ratio, 2.0, params, 1024, channels_in)
            .map_err(|e| format!("recording_error_resampler:{e}"))?;
        Ok(Self {
            resampler,
            buffers: vec![VecDeque::new(); channels_in],
            channels_out,
        })
    }

    pub(super) fn push_interleaved(
        &mut self,
        samples: &[f32],
        channels_in: usize,
    ) -> Result<Vec<f32>, String> {
        for (index, sample) in samples.iter().enumerate() {
            self.buffers[index % channels_in].push_back(*sample);
        }
        let mut out = Vec::new();
        loop {
            let needed = rubato::Resampler::input_frames_next(&self.resampler);
            if needed == 0 || self.buffers.iter().any(|buffer| buffer.len() < needed) {
                break;
            }
            let mut input = Vec::with_capacity(channels_in);
            for buffer in self.buffers.iter_mut() {
                input.push(buffer.drain(..needed).collect::<Vec<f32>>());
            }
            let wave = rubato::Resampler::process(&mut self.resampler, &input, None)
                .map_err(|e| format!("recording_error_resampler:{e}"))?;
            let out_frames = wave[0].len();
            for frame in 0..out_frames {
                for channel in 0..self.channels_out {
                    out.push(wave[channel][frame]);
                }
            }
        }
        Ok(out)
    }

    /// Zero-pads the remaining input to a full chunk so the filter delay is
    /// flushed and the last real samples are converted.
    pub(super) fn flush_interleaved(
        &mut self,
        channels_in: usize,
        channels_out: usize,
    ) -> Vec<f32> {
        let needed = rubato::Resampler::input_frames_next(&self.resampler);
        if needed == 0 {
            return Vec::new();
        }
        for buffer in &mut self.buffers {
            while buffer.len() < needed {
                buffer.push_back(0.0);
            }
        }
        let mut input = Vec::with_capacity(channels_in);
        for buffer in self.buffers.iter_mut() {
            input.push(buffer.drain(..needed).collect::<Vec<f32>>());
        }
        let mut out = Vec::new();
        if let Ok(wave) = rubato::Resampler::process(&mut self.resampler, &input, None) {
            let out_frames = wave[0].len();
            for frame in 0..out_frames {
                for channel in 0..channels_out {
                    out.push(wave[channel][frame]);
                }
            }
        }
        out
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

fn pick_monitor_config(device: &cpal::Device) -> Result<(cpal::StreamConfig, SampleFormat), String> {
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

/// Enumerates every selectable recording source: the system-default input,
/// all input devices, and all loopback (output) devices.
pub fn enumerate_devices() -> Vec<AudioDeviceInfo> {
    let mut devices = Vec::new();
    devices.push(AudioDeviceInfo {
        id: "default".to_string(),
        name: "System Default".to_string(),
        kind: "default".to_string(),
        is_default: true,
        is_loopback: false,
    });

    if let Ok(host_inputs) = cpal::default_host().input_devices() {
        for device in host_inputs {
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

    #[cfg(target_os = "windows")]
    {
        devices.extend(super::wasapi::enumerate_loopback_devices());
    }
    #[cfg(not(target_os = "windows"))]
    {
        devices.extend(enumerate_cpal_loopback_devices());
    }
    devices
}

#[cfg(not(target_os = "windows"))]
fn enumerate_cpal_loopback_devices() -> Vec<AudioDeviceInfo> {
    let mut devices = Vec::new();
    devices.push(AudioDeviceInfo {
        id: "loopback:default".to_string(),
        name: "System Default Output".to_string(),
        kind: "output".to_string(),
        is_default: true,
        is_loopback: true,
    });
    if let Ok(host_outputs) = cpal::default_host().output_devices() {
        for device in host_outputs {
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

/// Enumerates applications that currently emit audio (platform-dependent).
pub fn enumerate_applications() -> Vec<AppAudioInfo> {
    #[cfg(target_os = "windows")]
    {
        return super::wasapi::enumerate_applications();
    }
    #[cfg(target_os = "linux")]
    {
        return super::linux::enumerate_applications();
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        Vec::new()
    }
}
