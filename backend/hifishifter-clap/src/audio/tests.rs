// CLAP 音频与 transport 适配的行为夹具。
use super::*;
use crate::abi::{
    clap_audio_buffer_t, clap_event_header_t, clap_event_transport_t, transport_flags,
};

fn transport(flags: u32, seconds: f64) -> clap_event_transport_t {
    clap_event_transport_t {
        header: clap_event_header_t {
            size: 0,
            time: 0,
            space_id: 0,
            type_: 9,
            flags: 0,
        },
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
    let state = AudioState::default();
    state.set_sample_rate(48_000.0);
    state.set_render_mode(0);
    let event = transport(transport_flags::HAS_SECONDS_TIMELINE, 2.0);
    let process = clap_process_t {
        steady_time: 96_000,
        frames_count: 128,
        transport: &event,
        audio_inputs: std::ptr::null(),
        audio_outputs: std::ptr::null_mut(),
        audio_inputs_count: 0,
        audio_outputs_count: 0,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    let context = state
        .block_context(&process)
        .expect("transport 在场 ⇒ 一定有上下文");
    assert!(!context.playing, "steady_time 前进不能推断出“在播”");
    assert_eq!(context.project_time_samples, 96_000);
    assert!(
        context.continuous.is_none(),
        "steady_time 是回调稳态计数，不是工程连续时间"
    );
    assert!(context.music.is_none(), "宿主只给了秒轴");
}

#[test]
fn a_host_without_transport_gives_no_context_instead_of_position_zero() {
    // 【为什么这条是硬判据】空 transport 曾经被翻译成 "project_time_samples = 0"，
    // 于是“宿主什么都没说”被渲染层读成“这一块在工程第 0 帧”—— 离线块会从快照
    // 开头取音频。现在它必须与 VST3 的 process_context == nullptr 同一条路径。
    let state = AudioState::default();
    state.set_sample_rate(48_000.0);
    let process = clap_process_t {
        steady_time: 100,
        frames_count: 128,
        transport: std::ptr::null(),
        audio_inputs: std::ptr::null(),
        audio_outputs: std::ptr::null_mut(),
        audio_inputs_count: 0,
        audio_outputs_count: 0,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    assert!(
        state.block_context(&process).is_none(),
        "空 transport 必须映射成“没有时间事实”"
    );
}

#[test]
fn offline_render_is_only_reported_when_the_host_says_so() {
    let state = AudioState::default();
    state.set_sample_rate(48_000.0);
    state.set_render_mode(0);
    assert!(!state.is_offline());
    state.set_render_mode(crate::abi::CLAP_RENDER_OFFLINE);
    assert!(state.is_offline());
    state.set_render_mode(0);
    assert!(!state.is_offline());
}

#[test]
fn constant_mask_never_replaces_reading_the_input() {
    let mut left = vec![1.0_f32; 4];
    let mut right = vec![2.0_f32; 4];
    let mut in_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut out_left = vec![0.0_f32; 4];
    let mut out_right = vec![0.0_f32; 4];
    let mut out_planes = [out_left.as_mut_ptr(), out_right.as_mut_ptr()];
    let input = clap_audio_buffer_t {
        data32: in_planes.as_mut_ptr(),
        data64: std::ptr::null_mut(),
        channel_count: 2,
        latency: 0,
        constant_mask: u64::MAX,
    };
    let output = clap_audio_buffer_t {
        data32: out_planes.as_mut_ptr(),
        data64: std::ptr::null_mut(),
        channel_count: 2,
        latency: 0,
        constant_mask: 0,
    };
    let process = clap_process_t {
        steady_time: 0,
        frames_count: 4,
        transport: std::ptr::null(),
        audio_inputs: &input,
        audio_outputs: &output as *const _ as *mut _,
        audio_inputs_count: 1,
        audio_outputs_count: 1,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    assert!(unsafe { process_block(&process) });
    assert_eq!(out_left, vec![1.0; 4]);
    assert_eq!(out_right, vec![2.0; 4]);
}

#[test]
fn f64_is_silenced_but_still_succeeds() {
    let mut left = vec![1.0_f64; 4];
    let mut planes = [left.as_mut_ptr()];
    let input = clap_audio_buffer_t {
        data32: std::ptr::null_mut(),
        data64: planes.as_mut_ptr(),
        channel_count: 1,
        latency: 0,
        constant_mask: 0,
    };
    let output = clap_audio_buffer_t {
        data32: std::ptr::null_mut(),
        data64: planes.as_mut_ptr(),
        channel_count: 1,
        latency: 0,
        constant_mask: 0,
    };
    let process = clap_process_t {
        steady_time: 0,
        frames_count: 4,
        transport: std::ptr::null(),
        audio_inputs: &input,
        audio_outputs: &output as *const _ as *mut _,
        audio_inputs_count: 1,
        audio_outputs_count: 1,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    assert!(unsafe { process_block(&process) });
    assert_eq!(left, vec![0.0; 4]);
}

#[test]
fn two_instances_keep_independent_sample_rates_and_render_modes() {
    // 【为什么这条是硬判据】此前采样率/render/transport 都是进程级 static：
    // 同一宿主进程里第二个实例 activate 会覆盖第一个，工程位置换算与离线
    // 标记全部串台。多实例是 DAW 的常态，不是边界情况。
    let realtime = AudioState::default();
    realtime.set_sample_rate(48_000.0);
    assert!(realtime.set_render_mode(crate::abi::CLAP_RENDER_REALTIME));

    let offline = AudioState::default();
    offline.set_sample_rate(96_000.0);
    assert!(offline.set_render_mode(crate::abi::CLAP_RENDER_OFFLINE));

    assert_eq!(realtime.sample_rate(), 48_000.0);
    assert!(!realtime.is_offline());
    assert_eq!(offline.sample_rate(), 96_000.0);
    assert!(offline.is_offline());
    // 【为什么看 is_offline】观察位现在只有一份（捕获存储）；本结构只回答"此刻是不是
    // 离线渲染模式" —— 那正是共享捕获路径按流分账时读的那个事实。
}

#[test]
fn in_place_buffers_are_passed_through_without_cross_channel_corruption() {
    // 输入与输出共用同一块内存：若实现"读一路就写一路"，右声道会读到被
    // 左声道覆盖的数据。这里让两个声道各用不同的波形，逐样本比对。
    let mut left = vec![1.0_f32, 2.0, 3.0, 4.0];
    let mut right = vec![10.0_f32, 20.0, 30.0, 40.0];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let buffer = clap_audio_buffer_t {
        data32: planes.as_mut_ptr(),
        data64: std::ptr::null_mut(),
        channel_count: 2,
        latency: 0,
        constant_mask: 0,
    };
    let process = clap_process_t {
        steady_time: 0,
        frames_count: 4,
        transport: std::ptr::null(),
        audio_inputs: &buffer,
        audio_outputs: &buffer as *const _ as *mut _,
        audio_inputs_count: 1,
        audio_outputs_count: 1,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    assert!(unsafe { process_block(&process) });
    assert_eq!(left, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(right, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn a_beats_only_transport_still_yields_music_tempo_and_loop() {
    // 【为什么用拍轴】REAPER 的 CLAP transport 实测只给 HAS_BEATS_TIMELINE + HAS_TEMPO。
    // 只认秒轴的实现会让这类宿主上 project_time_samples 恒为 0（"捕获按钮永远灰着"），
    // 因此拍轴 + 速度必须换算成同一个工程位置。
    let state = AudioState::default();
    state.set_sample_rate(48_000.0);
    let mut event = transport(
        transport_flags::HAS_BEATS_TIMELINE
            | transport_flags::HAS_TEMPO
            | transport_flags::HAS_TIME_SIGNATURE
            | transport_flags::IS_PLAYING
            | transport_flags::IS_LOOP_ACTIVE,
        0.0,
    );
    event.song_pos_beats = (8.0 * crate::abi::CLAP_BEATTIME_FACTOR) as i64;
    event.loop_start_beats = (4.0 * crate::abi::CLAP_BEATTIME_FACTOR) as i64;
    event.loop_end_beats = (12.0 * crate::abi::CLAP_BEATTIME_FACTOR) as i64;
    let process = clap_process_t {
        steady_time: 0,
        frames_count: 128,
        transport: &event,
        audio_inputs: std::ptr::null(),
        audio_outputs: std::ptr::null_mut(),
        audio_inputs_count: 0,
        audio_outputs_count: 0,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    let context = state
        .block_context(&process)
        .expect("beats 轴也是 transport");
    assert!(context.playing);
    assert_eq!(context.music, Some(8.0));
    assert_eq!(context.tempo, Some(120.0));
    // 8 拍 @ 120 BPM = 4 s ⇒ 4 × 48 kHz
    assert_eq!(context.project_time_samples, 192_000);
    assert_eq!(context.time_signature, Some((4, 4)));
    assert!(context.cycle_valid && context.cycle_active);
    assert_eq!(context.cycle_start_music, 4.0);
    assert_eq!(context.cycle_end_music, 12.0);
}

#[test]
fn an_unsupported_output_layout_is_reported_before_any_rendering() {
    // 端口表只声明一条输出；多出来的总线不会被 AudioBlock 覆盖，因此必须整体拒绝
    // 并把每一条总线清零（与 VST3 的 UnsupportedFormat 同一结局）。
    let mut left = [9.0_f32; 4];
    let mut right = [9.0_f32; 4];
    let mut extra = [9.0_f32; 4];
    let mut first = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut second = [extra.as_mut_ptr()];
    let mut buses = [
        clap_audio_buffer_t {
            data32: first.as_mut_ptr(),
            data64: std::ptr::null_mut(),
            channel_count: 2,
            latency: 0,
            constant_mask: 0,
        },
        clap_audio_buffer_t {
            data32: second.as_mut_ptr(),
            data64: std::ptr::null_mut(),
            channel_count: 1,
            latency: 0,
            constant_mask: 0,
        },
    ];
    let process = clap_process_t {
        steady_time: 0,
        frames_count: 4,
        transport: std::ptr::null(),
        audio_inputs: std::ptr::null(),
        audio_outputs: buses.as_mut_ptr(),
        audio_inputs_count: 0,
        audio_outputs_count: 2,
        in_events: std::ptr::null(),
        out_events: std::ptr::null(),
    };
    assert!(!output_layout_supported(&process));
    assert!(unsafe { silence_outputs(&process) });
    assert_eq!(left, [0.0; 4]);
    assert_eq!(right, [0.0; 4]);
    assert_eq!(extra, [0.0; 4]);
}
