// CLAP 音频与 transport 适配的行为夹具。
use super::*;
use crate::abi::{clap_event_transport_t, clap_event_header_t, transport_flags};
use hifishifter_plugin_core::format::frame_is_trustworthy;

fn transport(flags: u32, seconds: f64) -> clap_event_transport_t {
    clap_event_transport_t {
        header: clap_event_header_t { size: 0, time: 0, space_id: 0, type_: 9, flags: 0 },
        flags,
        song_pos_beats: 0,
        song_pos_seconds: (seconds * CLAP_SECTIME_FACTOR) as i64,
        tempo: 120.0,
        tempo_inc: 0.0,
        loop_start_beats: 0,
        loop_end_beats: 0,
        loop_start_seconds: 0,
        loop_end_seconds: 0,
        bar_start: 0,
        bar_number: 0,
        tsig_num: 4,
        tsig_denom: 4,
    }
}

#[test]
fn a_stopped_host_does_not_infer_playing_from_steady_time() {
    set_sample_rate(48_000.0);
    set_render_mode(0);
    let event = transport(transport_flags::HAS_SECONDS_TIMELINE, 2.0);
    let process = clap_process_t { steady_time: 96_000, frames_count: 128, transport: &event,
        audio_inputs: std::ptr::null(), audio_outputs: std::ptr::null_mut(),
        audio_inputs_count: 0, audio_outputs_count: 0, in_events: std::ptr::null(), out_events: std::ptr::null() };
    let frame = to_transport_frame(&process);
    assert_eq!(frame.playing, Some(false));
    assert_eq!(frame.project_time_samples, 96_000);
    assert!(frame_is_trustworthy(&frame));
}

#[test]
fn a_host_without_transport_is_not_trusted() {
    set_sample_rate(48_000.0);
    let process = clap_process_t { steady_time: 100, frames_count: 128, transport: std::ptr::null(),
        audio_inputs: std::ptr::null(), audio_outputs: std::ptr::null_mut(),
        audio_inputs_count: 0, audio_outputs_count: 0, in_events: std::ptr::null(), out_events: std::ptr::null() };
    let frame = to_transport_frame(&process);
    assert_eq!(frame.playing, None);
    assert!(!frame.position_trusted);
    assert!(!frame_is_trustworthy(&frame));
}

#[test]
fn offline_render_is_only_reported_when_the_host_says_so() {
    set_sample_rate(48_000.0);
    set_render_mode(0);
    assert!(!to_transport_frame(&clap_process_t { steady_time: 0, frames_count: 0, transport: std::ptr::null(),
        audio_inputs: std::ptr::null(), audio_outputs: std::ptr::null_mut(), audio_inputs_count: 0,
        audio_outputs_count: 0, in_events: std::ptr::null(), out_events: std::ptr::null() }).offline);
    set_render_mode(crate::abi::CLAP_RENDER_OFFLINE);
    assert!(to_transport_frame(&clap_process_t { steady_time: 0, frames_count: 0, transport: std::ptr::null(),
        audio_inputs: std::ptr::null(), audio_outputs: std::ptr::null_mut(), audio_inputs_count: 0,
        audio_outputs_count: 0, in_events: std::ptr::null(), out_events: std::ptr::null() }).offline);
    set_render_mode(0);
}

#[test]
fn constant_mask_never_replaces_reading_the_input() {
    let mut left = vec![1.0_f32; 4];
    let mut right = vec![2.0_f32; 4];
    let mut in_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut out_left = vec![0.0_f32; 4];
    let mut out_right = vec![0.0_f32; 4];
    let mut out_planes = [out_left.as_mut_ptr(), out_right.as_mut_ptr()];
    let input = clap_audio_buffer_t { data32: in_planes.as_mut_ptr(), data64: std::ptr::null_mut(),
        channel_count: 2, latency: 0, constant_mask: u64::MAX };
    let output = clap_audio_buffer_t { data32: out_planes.as_mut_ptr(), data64: std::ptr::null_mut(),
        channel_count: 2, latency: 0, constant_mask: 0 };
    let process = clap_process_t { steady_time: 0, frames_count: 4, transport: std::ptr::null(),
        audio_inputs: &input, audio_outputs: &output as *const _ as *mut _,
        audio_inputs_count: 1, audio_outputs_count: 1, in_events: std::ptr::null(), out_events: std::ptr::null() };
    assert!(unsafe { process_block(&process) });
    assert_eq!(out_left, vec![1.0; 4]);
    assert_eq!(out_right, vec![2.0; 4]);
}

#[test]
fn f64_is_silenced_but_still_succeeds() {
    let mut left = vec![1.0_f64; 4];
    let mut planes = [left.as_mut_ptr()];
    let input = clap_audio_buffer_t { data32: std::ptr::null_mut(), data64: planes.as_mut_ptr(),
        channel_count: 1, latency: 0, constant_mask: 0 };
    let output = clap_audio_buffer_t { data32: std::ptr::null_mut(), data64: planes.as_mut_ptr(),
        channel_count: 1, latency: 0, constant_mask: 0 };
    let process = clap_process_t { steady_time: 0, frames_count: 4, transport: std::ptr::null(),
        audio_inputs: &input, audio_outputs: &output as *const _ as *mut _,
        audio_inputs_count: 1, audio_outputs_count: 1, in_events: std::ptr::null(), out_events: std::ptr::null() };
    assert!(unsafe { process_block(&process) });
    assert_eq!(left, vec![0.0; 4]);
}
