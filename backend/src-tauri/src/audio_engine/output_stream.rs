//! 输出 stream 工厂。
//!
//! 每次设备切换都重新创建一套 RT scratch（混音缓冲 / 快照过渡 / 电平 / 节拍器
//! 发声槽）；旧 stream drop 时一起释放。这样设备重建不需要把旧闭包里的 &mut
//! 状态移出，稳态也没有额外分配。

use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::Arc;

use arc_swap::ArcSwap;
use cpal::traits::DeviceTrait;

use super::metronome::{MetronomeRt, MetronomeVoices};
use super::mix::{
    render_callback_f32, render_callback_i16, render_callback_u16, RtFault,
    SnapshotTransitionState, TrackMeterBus, TrackMeterScratch, U16_SILENCE,
};
use super::types::{EngineCommand, EngineSnapshot};

pub(crate) struct OutputStreamArgs {
    pub device: cpal::Device,
    pub config: cpal::StreamConfig,
    pub sample_format: cpal::SampleFormat,
    pub channels: usize,
    pub snapshot: Arc<ArcSwap<EngineSnapshot>>,
    pub is_playing: Arc<AtomicBool>,
    pub play_start_wait: Arc<AtomicBool>,
    pub position_frames: Arc<AtomicU64>,
    pub duration_frames: Arc<AtomicU64>,
    pub meter_bus: Arc<TrackMeterBus>,
    pub metro: Arc<MetronomeRt>,
    /// 回传给 worker 的命令通道：RT 侧的 `err_fn` 用它请求**强制重建**流。
    ///
    /// 【为什么必须在流工厂里拿通道】流死掉这件事只有 cpal 的 err_fn 知道，
    /// 而那里不持有任何引擎句柄。把它一路传进来比自己维护一张"已死流"表要
    /// 可靠：错误发生的那一刻就发出重建请求，不依赖轮询或身份比对。
    pub tx: std::sync::mpsc::Sender<EngineCommand>,
}

pub(crate) fn build_output_stream(args: OutputStreamArgs) -> Result<cpal::Stream, String> {
    let OutputStreamArgs {
        device,
        config,
        sample_format,
        channels,
        snapshot,
        is_playing,
        play_start_wait,
        position_frames,
        duration_frames,
        meter_bus,
        metro,
        tx,
    } = args;

    let mut scratch_mix: Vec<f32> = Vec::new();
    let mut scratch_mix_fade_from: Vec<f32> = Vec::new();
    let mut snapshot_transition = SnapshotTransitionState::default();
    let err_fn = |bus: Arc<TrackMeterBus>, tx: std::sync::mpsc::Sender<EngineCommand>| {
        move |_err: cpal::Error| {
            bus.record_rt_fault(RtFault::StreamError);
            // 【为什么在这里就发重建请求】流已死，但"设备身份"可能一点没变
            // （例如被独占后回退到同一台设备的同一格式）。只靠 watcher 的
            // 身份比对永远等不到重建，用户听到的是"一直在播、一直没有声"。
            // `force: true` 跳过那次判等。
            let _ = tx.send(EngineCommand::DefaultOutputDeviceChanged { force: true });
        }
    };

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let meter_bus_for_cb = meter_bus.clone();
            let meter_bus_for_err = meter_bus.clone();
            let mut meter_scratch = TrackMeterScratch::default();
            let metro = metro.clone();
            let mut metro_voices = MetronomeVoices::default();
            device.build_output_stream(
                config,
                move |data: &mut [f32], _| {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        render_callback_f32(
                            data,
                            channels,
                            &snapshot,
                            is_playing.as_ref(),
                            play_start_wait.as_ref(),
                            position_frames.as_ref(),
                            duration_frames.as_ref(),
                            &mut scratch_mix,
                            &mut scratch_mix_fade_from,
                            &mut snapshot_transition,
                            &mut meter_scratch,
                            &meter_bus_for_cb,
                            metro.as_ref(),
                            &mut metro_voices,
                        );
                    }));
                    if r.is_err() {
                        meter_bus_for_cb.record_rt_fault(RtFault::PanicF32);
                        data.fill(0.0);
                        is_playing.store(false, std::sync::atomic::Ordering::Relaxed);
                        play_start_wait.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                err_fn(meter_bus_for_err, tx.clone()),
                None,
            )
        }
        cpal::SampleFormat::I16 => {
            let meter_bus_for_cb = meter_bus.clone();
            let meter_bus_for_err = meter_bus.clone();
            let mut meter_scratch = TrackMeterScratch::default();
            let metro = metro.clone();
            let mut metro_voices = MetronomeVoices::default();
            device.build_output_stream(
                config,
                move |data: &mut [i16], _| {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        render_callback_i16(
                            data,
                            channels,
                            &snapshot,
                            is_playing.as_ref(),
                            play_start_wait.as_ref(),
                            position_frames.as_ref(),
                            duration_frames.as_ref(),
                            &mut scratch_mix,
                            &mut scratch_mix_fade_from,
                            &mut snapshot_transition,
                            &mut meter_scratch,
                            &meter_bus_for_cb,
                            metro.as_ref(),
                            &mut metro_voices,
                        );
                    }));
                    if r.is_err() {
                        meter_bus_for_cb.record_rt_fault(RtFault::PanicI16);
                        data.fill(0);
                        is_playing.store(false, std::sync::atomic::Ordering::Relaxed);
                        play_start_wait.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                err_fn(meter_bus_for_err, tx.clone()),
                None,
            )
        }
        cpal::SampleFormat::U16 => {
            let meter_bus_for_cb = meter_bus.clone();
            let meter_bus_for_err = meter_bus.clone();
            let mut meter_scratch = TrackMeterScratch::default();
            let metro = metro.clone();
            let mut metro_voices = MetronomeVoices::default();
            device.build_output_stream(
                config,
                move |data: &mut [u16], _| {
                    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        render_callback_u16(
                            data,
                            channels,
                            &snapshot,
                            is_playing.as_ref(),
                            play_start_wait.as_ref(),
                            position_frames.as_ref(),
                            duration_frames.as_ref(),
                            &mut scratch_mix,
                            &mut scratch_mix_fade_from,
                            &mut snapshot_transition,
                            &mut meter_scratch,
                            &meter_bus_for_cb,
                            metro.as_ref(),
                            &mut metro_voices,
                        );
                    }));
                    if r.is_err() {
                        meter_bus_for_cb.record_rt_fault(RtFault::PanicU16);
                        data.fill(U16_SILENCE);
                        is_playing.store(false, std::sync::atomic::Ordering::Relaxed);
                        play_start_wait.store(false, std::sync::atomic::Ordering::Relaxed);
                    }
                },
                err_fn(meter_bus_for_err, tx.clone()),
                None,
            )
        }
        _ => return Err("unsupported output sample format".to_string()),
    };

    stream.map_err(|error| format!("failed to build output stream: {error}"))
}
