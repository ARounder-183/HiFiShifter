//! `vst3.rs` 的音频边界测试（缓冲布局、清零、快照透传）。从内联模块纯搬到这里。

use super::*;
use crate::audio_abi::{AudioBusBuffers, ProcessData};
use crate::sync::MutexExt;

/// 让文档上另有一个实例持 playback 角色。
///
/// 【为什么必须要它】"透传"的判据现在是**文档级**的：文档上没有任何 playback 实例时，
/// 唯一的编辑实例会**接管渲染**（见 `host_audio_state_tests::
/// a_sole_editor_instance_takes_over_rendering`），因为实测 Cubase 15 正是"只给一个
/// 编辑实例、于是插件永远不发声"的形态。本文件的用例要测的是"编辑实例与播放实例**并存**时的
/// 透传"（REAPER 的形态），所以必须把播放实例摆出来。
fn with_a_playback_sibling(
    model: &crate::ara::model::ModelHandle,
) -> std::sync::Arc<crate::render::extension::ExtensionOwner> {
    let owner = std::sync::Arc::new(crate::render::extension::ExtensionOwner::default());
    owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .expect("playback sibling must bind");
    owner
}

/// SDK纯editor renderer透传宿主音频，不能把已含fade/gain的输入换成自己的旧快照。
#[test]
fn editor_only_renderer_preserves_host_fades_in_all_process_modes_without_allocations() {
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let _playback = with_a_playback_sibling(&model);
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 0,
            left: vec![0.9; 4],
            right: vec![0.9; 4],
            _reservation: None,
        })
        .unwrap();
    for mode in [0, 1, 2] {
        let mut source_left = [0.0, 0.1, 0.2, 0.3];
        let mut source_right = [0.3, 0.2, 0.1, 0.0];
        let mut input_planes = [source_left.as_mut_ptr(), source_right.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: input_planes.as_mut_ptr(),
        };
        let mut left = [9.0; 5];
        let mut right = [9.0; 5];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 3,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        // context可选/停播也必须透传；preview为空时不能凭播放门禁覆盖宿主输入。
        let mut data = ProcessData {
            process_mode: mode,
            num_samples: 4,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            ..Default::default()
        };
        crate::test_allocator::begin();
        let result =
            unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
        let allocations = crate::test_allocator::end();
        assert_eq!(result, K_RESULT_OK);
        assert_eq!(allocations, 0);
        assert_eq!(left, [0.0, 0.1, 0.2, 0.3, 9.0]);
        assert_eq!(right, [0.3, 0.2, 0.1, 0.0, 9.0]);
        assert_eq!(output.silence_flags, 0);
    }
}

/// 相同bus/相同plane的in-place不能先清零；silent或inactive输入则明确补零。
// 测试故意把plane置空来模拟inactive输入，赋值本身不被读取。
#[allow(unused_assignments)]
#[test]
fn editor_only_passthrough_handles_in_place_silence_and_inactive_planes() {
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::EDITOR_RENDERER,
            None,
        )
        .unwrap();
    let _playback = with_a_playback_sibling(&model);
    let mut left = [0.05, 0.1, 123.0];
    let mut right = [0.2, 0.3, 456.0];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut data = ProcessData {
        num_samples: 2,
        num_inputs: 1,
        num_outputs: 1,
        inputs: &raw mut bus,
        outputs: &raw mut bus,
        ..Default::default()
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(left, [0.05, 0.1, 123.0]);
    assert_eq!(right, [0.2, 0.3, 456.0]);
    bus.silence_flags = 1;
    planes[1] = std::ptr::null_mut();
    let mut out_left = [9.0; 3];
    let mut out_right = [9.0; 3];
    let mut out_planes = [out_left.as_mut_ptr(), out_right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: out_planes.as_mut_ptr(),
    };
    data.inputs = &raw mut bus;
    data.outputs = &raw mut output;
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(out_left, [0.0, 0.0, 9.0]);
    assert_eq!(out_right, [0.0, 0.0, 9.0]);
    assert_eq!(output.silence_flags, 3);
    bus.channel_buffers = std::ptr::null_mut();
    out_left = [7.0; 3];
    data.inputs = &raw mut bus;
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_INVALID_ARGUMENT
    );
    assert_eq!(out_left, [7.0; 3]);
}

/// 真正 process 在有效快照下输出 PCM，且观察区间不分配或释放任何内存。
#[test]
fn process_outputs_the_snapshot_without_allocating_or_deallocating() {
    let mut processor = Processor::create().unwrap();
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 10,
            left: vec![0.1, 0.2],
            right: vec![0.3, 0.4],
            _reservation: None,
        })
        .unwrap();
    let mut left = [8.0_f32; 3];
    let mut right = [8.0_f32; 3];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 44100.0,
        project_time_samples: 11,
        ..Default::default()
    };
    let mut data = ProcessData {
        num_samples: 2,
        num_outputs: 1,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    crate::test_allocator::begin();
    // SAFETY: 实际 processor 音频子对象和完整 SDK 缓冲都在回调期间存活。
    let result =
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    let allocations = crate::test_allocator::end();
    assert_eq!(result, K_RESULT_OK);
    assert_eq!(left, [0.2, 0.0, 8.0]);
    assert_eq!(right, [0.4, 0.0, 8.0]);
    assert_eq!(bus.silence_flags, 0);
    assert_eq!(allocations, 0);
}

/// 真实SDK入口验证离线setup等完整后台发布；普通setup不受doc事务/worker阻塞。
#[test]
fn offline_setup_waits_for_current_snapshot_and_reprepares_after_edit_revision_changes() {
    use std::sync::mpsc;
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let pcm = document.edit_sources.lock_or_recover()["ara://source"].clone();
    document
        .sources
        .lock_or_recover()
        .insert("ara://source".into(), pcm);
    let mut processor = Processor::create().unwrap();
    processor.extension_owner = owners[0].clone();
    // SAFETY: Box保留稳定处理器地址到离线setup测试线程join。
    let audio = unsafe { audio_ptr(&mut *processor) };
    let held = document.transaction.lock_or_recover();
    owners[0].prepare();
    let mut normal = ProcessSetup {
        process_mode: 0,
        symbolic_sample_size: 0,
        max_samples_per_block: 1024,
        sample_rate: 44100.,
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.setup_processing)(audio, &raw mut normal) },
        K_RESULT_OK,
        "普通setup不能等待doc/推理"
    );
    let pointer = audio as usize;
    let (done, rx) = mpsc::channel();
    let job = std::thread::spawn(move || {
        let mut setup = ProcessSetup {
            process_mode: 2,
            symbolic_sample_size: 0,
            max_samples_per_block: 1024,
            sample_rate: 44100.,
        };
        // SAFETY: 主测试保留稳定Box/接口/owner到join；此线程模拟SDK的非实时setup调用。
        let result =
            unsafe { (AUDIO_VTBL.setup_processing)(pointer as *mut c_void, &raw mut setup) };
        done.send(result).unwrap();
    });
    assert!(
        rx.recv_timeout(std::time::Duration::from_millis(20))
            .is_err(),
        "不能在后台未完成时成功返回"
    );
    drop(held);
    assert_eq!(
        rx.recv_timeout(std::time::Duration::from_secs(3)).unwrap(),
        K_RESULT_OK
    );
    job.join().unwrap();
    assert!(owners[0]
        .snapshots
        .iter()
        .all(|snapshot| snapshot.is_ready()));
    {
        let _transaction = document.transaction.lock_or_recover();
        let mut edits = document.edits.lock_or_recover();
        edits.revision += 1;
        let mut track = document.timeline.lock_or_recover().as_ref().unwrap().tracks[0].clone();
        track.volume = 0.5;
        edits.tracks = vec![track];
    }
    let mut setup = ProcessSetup {
        process_mode: 2,
        symbolic_sample_size: 0,
        max_samples_per_block: 1024,
        sample_rate: 44100.,
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.setup_processing)(audio, &raw mut setup) },
        K_RESULT_OK
    );
    let mut left = [9_f32; 5];
    let mut right = [9_f32; 5];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        sample_rate: 44100.,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 2,
        num_samples: 4,
        num_outputs: 1,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio, (&raw mut data).cast()) },
        K_RESULT_OK
    );
    document.close();
    for (actual, want) in left[..4].iter().zip([0.05, 0.1, 0.15, 0.2]) {
        assert!((*actual - want).abs() < 1e-6);
    }
    assert_eq!(left, right);
    assert_eq!(left[4], 9.);
}
/// 离线缺快照/上下文必须显式失败；实时仍安全静音，全部回调保持零分配。
#[test]
fn offline_missing_snapshot_is_failure_not_successful_silence() {
    let mut processor = Processor::create().unwrap();
    let mut left = [9_f32; 4];
    let mut right = [9_f32; 4];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        sample_rate: 44100.,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 2,
        num_samples: 4,
        num_outputs: 1,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    crate::test_allocator::begin();
    let missing =
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    data.process_context = std::ptr::null_mut();
    let no_context =
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    data.process_mode = 0;
    let realtime =
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    let allocations = crate::test_allocator::end();
    assert_eq!(
        (missing, no_context, realtime),
        (K_RESULT_FALSE, K_RESULT_FALSE, K_RESULT_OK)
    );
    assert_eq!(allocations, 0);
    assert_eq!(left, [0.; 4]);
    assert_eq!(right, [0.; 4]);
}
/// 用真实接口子对象调用 process，避免测试绕过产品 vtable。
fn run(data: *mut ProcessData) -> TResult {
    let mut processor = Processor::create().unwrap();
    // SAFETY: 本测试持有处理器与 SDK 同布局数据，调用期间缓冲存活。
    unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), data.cast()) }
}

/// 停播重复process同一帧必须静音；离线导出没有kPlaying也必须保留音频。
#[test]
fn idle_transport_is_silent_but_offline_export_still_reads_snapshot() {
    let mut processor = Processor::create().unwrap();
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 0,
            left: vec![0.25; 4],
            right: vec![0.5; 4],
            _reservation: None,
        })
        .unwrap();
    let mut left = [9_f32; 4];
    let mut right = [9_f32; 4];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        sample_rate: 44100.,
        ..Default::default()
    };
    let mut data = ProcessData {
        num_samples: 4,
        num_outputs: 1,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    for _ in 0..3 {
        assert_eq!(
            unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
            K_RESULT_OK
        );
        assert_eq!(left, [0.; 4], "停播不能循环输出光标位置的快照块");
        assert_eq!(right, [0.; 4]);
    }
    data.process_mode = 2;
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(left, [0.25; 4]);
    assert_eq!(right, [0.5; 4]);
}

/// 宿主有效item mute即时门控真正VST3输出；同文档另一region不被静音，解除后复用原快照。
#[test]
fn effective_item_mute_projects_to_gui_and_gates_real_process_without_inference() {
    let (model, owners, _ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let host = crate::host::MockHost::new();
    host.set_value("D_POSITION", 0.);
    host.set_value("D_LENGTH", 4. / 44100.);
    host.set_value("B_MUTE", 1.);
    unsafe {
        owners[0].bind_host_adapter(host.context());
    }
    owners[0].refresh_host_transport();
    let workspace = document.workspace_timeline().unwrap();
    assert!(workspace.clips[0].muted);
    assert!(!workspace.clips[1].muted);
    assert!(!owners[1].host_item_muted(), "静音只能影响直接绑定region");
    let mut processor = Processor::create().unwrap();
    processor.extension_owner = owners[0].clone();
    owners[0].snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 0,
            left: vec![0.25; 4],
            right: vec![0.5; 4],
            _reservation: None,
        })
        .unwrap();
    let mut left = [9_f32; 5];
    let mut right = [9_f32; 5];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 44100.,
        ..Default::default()
    };
    let mut data = ProcessData {
        num_samples: 4,
        num_outputs: 1,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    for mode in [0, 2] {
        data.process_mode = mode;
        crate::test_allocator::begin();
        let result =
            unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
        let allocations = crate::test_allocator::end();
        assert_eq!(result, K_RESULT_OK);
        assert_eq!(allocations, 0);
        assert_eq!(&left[..4], &[0.; 4]);
        assert_eq!(&right[..4], &[0.; 4]);
        assert_eq!((left[4], right[4]), (9., 9.));
        assert_eq!(bus.silence_flags, 3);
    }
    host.set_value("B_MUTE", 0.);
    host.set_value("B_MUTE_ACTUAL", 1.);
    owners[0].refresh_host_transport();
    assert!(
        !document.workspace_timeline().unwrap().clips[0].muted,
        "solo覆盖后的解除不丢原始mute"
    );
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(&left[..4], &[0.25; 4]);
    assert_eq!(&right[..4], &[0.5; 4]);
    document.close();
}

/// 空实现留下旧音频；写错块长则越界覆盖尾哨兵。
#[test]
fn process_clears_exactly_the_requested_frames() {
    let mut left = [0.75_f32, 0.75, 123.0];
    let mut right = [-0.5_f32, -0.5, 456.0];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut data = ProcessData {
        num_samples: 2,
        num_outputs: 1,
        outputs: &raw mut output,
        ..Default::default()
    };
    assert_eq!(run(&raw mut data), K_RESULT_OK);
    assert_eq!(left, [0.0, 0.0, 123.0]);
    assert_eq!(right, [0.0, 0.0, 456.0]);
    assert_eq!(output.silence_flags, 3);
}

/// 非法宿主数据不能被空实现伪报成功。
#[test]
fn process_rejects_invalid_counts_and_missing_bus_storage() {
    assert_eq!(run(std::ptr::null_mut()), K_INVALID_ARGUMENT);
    for data in [
        ProcessData {
            num_samples: -1,
            ..Default::default()
        },
        ProcessData {
            num_outputs: -1,
            ..Default::default()
        },
        ProcessData {
            num_inputs: -1,
            ..Default::default()
        },
        ProcessData {
            num_samples: 1,
            num_outputs: 1,
            ..Default::default()
        },
        ProcessData {
            num_samples: 1,
            num_inputs: 1,
            ..Default::default()
        },
    ] {
        let mut data = data;
        assert_eq!(run(&raw mut data), K_INVALID_ARGUMENT);
    }
}

/// 零帧参数 flush 与无总线合法，不应解引用空地址。
#[test]
fn process_accepts_zero_frame_flush_and_no_output() {
    let mut flush = ProcessData {
        num_outputs: 1,
        ..Default::default()
    };
    assert_eq!(run(&raw mut flush), K_RESULT_OK);
    let mut empty = ProcessData {
        num_samples: 32,
        ..Default::default()
    };
    assert_eq!(run(&raw mut empty), K_RESULT_OK);
}

/// 64 位样本不支持处理，但**不能**让宿主把插件踢出链路：按实际宽度静音、返回 OK，
/// 且绝不按 f32 解引用 f64 平面（那会写出破坏的样本）。
#[test]
fn process_silences_sample64_instead_of_leaving_the_chain() {
    let mut samples = [0.25_f64, 0.5];
    let mut planes = [samples.as_mut_ptr(), samples.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr().cast(),
    };
    let mut data = ProcessData {
        symbolic_sample_size: 1,
        num_samples: 2,
        num_outputs: 1,
        outputs: &raw mut output,
        ..Default::default()
    };
    assert_eq!(run(&raw mut data), K_RESULT_OK);
    assert_eq!(samples, [0.0, 0.0]);
    assert_eq!(output.silence_flags, 3);
}

/// SDK 允许 inactive plane 为 null，但有通道时 plane 数组不能缺失。
#[test]
fn process_skips_inactive_planes_but_rejects_missing_arrays() {
    let mut samples = [1.0_f32, 2.0];
    let mut planes = [samples.as_mut_ptr(), std::ptr::null_mut()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut data = ProcessData {
        num_samples: 2,
        num_outputs: 1,
        outputs: &raw mut output,
        ..Default::default()
    };
    assert_eq!(run(&raw mut data), K_RESULT_OK);
    assert_eq!(samples, [0.0, 0.0]);
    assert_eq!(output.silence_flags, 3);
    output.channel_buffers = std::ptr::null_mut();
    data.outputs = &raw mut output;
    assert_eq!(run(&raw mut data), K_INVALID_ARGUMENT);
}

/// 所有输入形状先校验，非法输入不能导致输出缓冲被部分写入。
#[test]
fn process_rejects_invalid_input_before_touching_output() {
    // 三个 plane：下面的用例会声明 3 声道输入（宽总线现在被接受）。
    let mut input_planes = [
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ];
    for (channels, missing_array, expected) in [
        (-1, false, K_INVALID_ARGUMENT),
        // 单声道是**合法**总线形状（人声轨常见）；此前被拒 ⇒ 宿主静默旁路插件。
        (1, false, K_RESULT_OK),
        (2, true, K_INVALID_ARGUMENT),
        (2, false, K_RESULT_OK),
        (0, true, K_RESULT_OK),
        // 宽于立体声的输入**接受**：拒绝会让宿主把插件踢出插槽（5.1 / 7.1 轨道）。
        // 我们只读前两个声道，其余不参与处理。
        (3, false, K_RESULT_OK),
    ] {
        let mut left = [0.25_f32, 0.5];
        let mut right = [0.75_f32, 1.0];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: channels,
            silence_flags: 0,
            channel_buffers: if missing_array {
                std::ptr::null_mut()
            } else {
                input_planes.as_mut_ptr()
            },
        };
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        let mut data = ProcessData {
            num_samples: 2,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            ..Default::default()
        };
        assert_eq!(run(&raw mut data), expected);
        if expected != K_RESULT_OK {
            assert_eq!(left, [0.25, 0.5]);
            assert_eq!(right, [0.75, 1.0]);
            assert_eq!(output.silence_flags, 0);
        } else {
            assert_eq!(left, [0.0, 0.0]);
            assert_eq!(right, [0.0, 0.0]);
        }
    }
}

/// 未实现多总线或非 stereo 时不能假装完成协商。
#[test]
fn bus_negotiation_rejects_unsupported_arrangements() {
    let mut processor = Processor::create().unwrap();
    let mut stereo = 3_u64;
    let mut mono = 1_u64;
    let mut surround = 0x3F_u64;
    // SAFETY: 真实处理器和布局数组在协商期间存活。
    unsafe {
        let this = audio_ptr(&mut *processor);
        assert_eq!(
            audio_set_bus_arrangements(this, &raw mut stereo, 1, &raw mut stereo, 1),
            K_RESULT_OK
        );
        // 单声道入/出、以及单声道入 + 立体声出都接受：人声轨单声道极常见，此前
        // 一律拒绝会让宿主把插件踢出链路。
        assert_eq!(
            audio_set_bus_arrangements(this, &raw mut mono, 1, &raw mut mono, 1),
            K_RESULT_OK
        );
        assert_eq!(
            audio_set_bus_arrangements(this, &raw mut mono, 1, &raw mut stereo, 1),
            K_RESULT_OK
        );
        assert_eq!(
            audio_set_bus_arrangements(this, &raw mut stereo, 0, &raw mut stereo, 1),
            K_RESULT_FALSE
        );
        assert_eq!(
            audio_set_bus_arrangements(this, std::ptr::null_mut(), 1, &raw mut stereo, 1),
            K_INVALID_ARGUMENT
        );
        // `getBusInfo` 必须如实回吐协商后的声道数。此前无条件写死 2 ⇒ 单声道轨道上
        // 宿主按 2 声道组织缓冲，而插件只填/读 1 声道（对宿主撒谎）。最后一次成功
        // 协商是 mono 入 + stereo 出，故输入 1、输出 2。
        let component = (&mut *processor) as *mut Processor as *mut c_void;
        let mut input_info = std::mem::zeroed::<BusInfo>();
        assert_eq!(
            component_get_bus_info(component, 0, 0, 0, &raw mut input_info),
            K_RESULT_OK
        );
        assert_eq!(
            input_info.channel_count, 1,
            "协商为单声道输入 ⇒ 报告 1 声道"
        );
        let mut output_info = std::mem::zeroed::<BusInfo>();
        assert_eq!(
            component_get_bus_info(component, 0, 1, 0, &raw mut output_info),
            K_RESULT_OK
        );
        assert_eq!(
            output_info.channel_count, 2,
            "协商为立体声输出 ⇒ 报告 2 声道"
        );

        // 环绕声**接受**（放在最后，避免影响上面的"最后一次成功协商"断言）：
        // 此前拒绝，而多数宿主在 `kResultFalse` 时把插件**踢出插槽** ——
        // 5.1 / 7.1 轨道上用户看到的是"插件不见了"，没有任何提示。
        // 现在的取舍是"部分工作、留在插槽里"：内部只处理前两个声道
        // （见 `PlaybackSnapshot::copy_block` / `audio_abi::clear_outputs`）。
        assert_eq!(
            audio_set_bus_arrangements(this, &raw mut surround, 1, &raw mut surround, 1),
            K_RESULT_OK
        );
    }
}

/// 非法 setup 不能让后续回调使用无效采样率或块尺寸。
#[test]
fn setup_rejects_invalid_rate_block_and_format() {
    let mut processor = Processor::create().unwrap();
    // SAFETY: 真实处理器和 setup POD 在调用期间存活。
    unsafe {
        let this = audio_ptr(&mut *processor);
        assert_eq!(
            audio_setup_processing(this, std::ptr::null_mut()),
            K_INVALID_ARGUMENT
        );
        for (rate, block, format, expected) in [
            (44100.0, 512, 0, K_RESULT_OK),
            (f64::NAN, 512, 0, K_INVALID_ARGUMENT),
            (0.0, 512, 0, K_INVALID_ARGUMENT),
            (44100.0, -1, 0, K_INVALID_ARGUMENT),
            // 非 32-bit 浮点现在**接受**：process 侧按实际宽度静音并留在链路里
            // （见 `silence_unsupported_sample_size`）。此前 `kResultFalse` 会让宿主
            // 把插件整条踢出链路。
            (44100.0, 512, 1, K_RESULT_OK),
            // 96k 现在**接受**（内部按 44.1/48k 出快照，输出线性重采样）—— 此前
            // 返回 `kResultFalse` 会让宿主把插件整条踢出链路。
            (96000.0, 512, 0, K_RESULT_OK),
            (192000.0, 512, 0, K_RESULT_OK),
            // 低于 44.1k 现在也**接受**（接受域与 `render::source` 的 8k 下限对齐）：
            // 内部快照仍按 44.1/48k 出，输出走线性重采样（降采样、有镜像，记警告）。
            // 此前返回 `kResultFalse` 会让宿主把插件整条踢出链路。
            (8000.0, 512, 0, K_RESULT_OK),
            (22050.0, 512, 0, K_RESULT_OK),
            // 超出源读取域的率仍然拒绝：重采样比例会大到没有意义。
            (4000.0, 512, 0, K_RESULT_FALSE),
            (768000.0, 512, 0, K_RESULT_FALSE),
        ] {
            let mut setup = ProcessSetup {
                process_mode: 0,
                symbolic_sample_size: format,
                max_samples_per_block: block,
                sample_rate: rate,
            };
            assert_eq!(audio_setup_processing(this, &raw mut setup), expected);
        }
    }
}
