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
use super::types::EngineSnapshot;

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
    } = args;

    let mut scratch_mix: Vec<f32> = Vec::new();
    let mut scratch_mix_fade_from: Vec<f32> = Vec::new();
    let mut snapshot_transition = SnapshotTransitionState::default();
    let err_fn = |bus: Arc<TrackMeterBus>| {
        move |_err: cpal::Error| bus.record_rt_fault(RtFault::StreamError)
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
                err_fn(meter_bus_for_err),
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
                err_fn(meter_bus_for_err),
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
                err_fn(meter_bus_for_err),
                None,
            )
        }
        _ => return Err("unsupported output sample format".to_string()),
    };

    stream.map_err(|error| format!("failed to build output stream: {error}"))
}
