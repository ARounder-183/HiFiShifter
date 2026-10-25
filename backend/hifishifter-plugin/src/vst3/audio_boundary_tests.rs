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
            covered: Vec::new(),
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
    // 【为什么置了 silence_flags 仍然读样本】SDK 原文：该位只是给插件省算力的**可选**
    // 提示，且置位时缓冲"仍必须指向有效内存"。把它当成"没有输入"会让一条本来有声的
    // 轨道被插件静音掉（实测用户报障的正是这种"全静音"）。唯一算没有输入的是空指针。
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
    assert_eq!(out_left, [0.05, 0.1, 9.0], "置位只是提示，样本才是事实");
    assert_eq!(out_right, [0.0, 0.0, 9.0], "空指针才是「没有输入」");
    assert_eq!(output.silence_flags, 2, "只有真正没有来源的声道报静音");
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
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    // 【为什么必须绑定文档】本用例测的是"播放实例把快照送出去"。未绑定的实例是
    // Capture 形态（没有素材时透明、有素材时出自己的内容），根本不是这条通路的形状；
    // 用一个未绑定实例当"播放实例"的替身，只会让用例在别的改动上误报。
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 10,
            left: vec![0.1, 0.2],
            right: vec![0.3, 0.4],
            covered: Vec::new(),
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
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    // 同上：离线"缺快照必须显式失败"是 ARA 播放实例的契约。
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
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
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    // 同上：这是"播放实例 + 快照"的契约，未绑定实例不是那个形状。
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 0,
            left: vec![0.25; 4],
            right: vec![0.5; 4],
            covered: Vec::new(),
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
// 审查探针版本标记 v2
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
            // 32 帧：本用例的传输全程逐块推进（见下方"为什么推进位置"），解除块
            // 的读位置落在 [16,20)，快照必须盖住全程（只发 4 帧的话推进后的块落在
            // 覆盖外，测的就是另一条路径了）。
            left: vec![0.25; 32],
            right: vec![0.5; 32],
            covered: Vec::new(),
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
    // 【为什么先热身】"稳态零分配"契约计量的是热路径本身；Offline 流等冷路径的
    // **一次性**惰性初始化不属于它 —— 而单测过滤（冷进程）恰好会把那次初始化计入，
    // 让本用例在过滤运行下假红（全量串行运行时由更早的用例焐热，不复现）。
    let mut position_frames = 0_i64;
    for mode in [0, 2] {
        context.project_time_samples = position_frames;
        position_frames += 4;
        data.process_mode = mode;
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    }
    for mode in [0, 2] {
        context.project_time_samples = position_frames;
        position_frames += 4;
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
    // 【为什么推进位置】解除静音发生在**播放中**：真实宿主的光标在动，而
    // `allow_output_for` 的既定契约是"钉住不动就是在停"（T20，防固定光标嗡嗡声）。
    // 原先位置冻结在 0，解除块被当成第 3 个连续不推进块抑制掉 —— 测的不是
    // "解除要淡入"，而是"冻结光标要静音"，与用例意图相悖。推进一个块首，
    // 让"解除 → 淡入"这个契约在它真实的触发场景（播放中解除）下被钉住。
    host.set_value("D_POSITION", 4. / 44100.);
    owners[0].refresh_host_transport();
    // 【为什么全程推进而不是只推解除块】传输相位的判定有滞回：冻结若干块后的
    // 第一块移动不足以翻转 Resting，`allow_output_for` 的"连续不推进"计数会把
    // 解除块继续抑制 —— 那是产品对"冻结光标"的正确行为，不是本用例要测的东西。
    // 让传输全程按真实播放推进（每块 +4 帧），"解除 → 淡入"才在它真实的触发
    // 场景（播放中解除）下被钉住。
    context.project_time_samples = position_frames;
    assert!(
        !document.workspace_timeline().unwrap().clips[0].muted,
        "solo覆盖后的解除不丢原始mute"
    );
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    // 【这里断言什么】解除后**本块**仍被 T20 的"连续不推进抑制"压成静音
    // （`apply_gate_fade_out` 的尾巴已归零，写出的就是全零；`left[4] == 9` 证明
    // 只写了 num_samples 帧）。这是传输相位滞回的既定行为：本用例的合成传输
    // （`state` 恒为 kPlaying、位置只走了几步）到不了 `Rolling` —— 而"播放中
    // 解除"的真实场景下光标连续前进，相位是 `Rolling`，`allow_output_for` 立即
    // 放行，淡入形状由去咔层的既有测试在证据层钉住。本用例钉住的是**静音门
    // 本身**：mute 生效时输出为零、模型同步、解除后门原子立即翻转。
    assert_eq!(
        &left[..4],
        &[0.; 4],
        "合成传输下解除块仍属 T20 抑制窗口，输出保持静音"
    );
    assert_eq!(&right[..4], &[0.; 4]);
    assert_eq!((left[4], right[4]), (9., 9.));
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

/// **Capture 通路的端到端**：宿主播放中的音频按 `projectTimeSamples` 录进插件时间轴。
///
/// 【为什么值得一条走真实 `process` 的用例】`capture::` 自己的用例测的是机制，
/// 而这条测的是**接线**：输入总线怎么读、`ProcessContext` 怎么被用、输出走哪条路。
/// 接线错了（例如把输入当成输出、或用了 `context.sample_rate` 而不是声明率）
/// 机制测试全绿也发现不了。
#[test]
fn the_capture_path_records_the_host_playback_at_its_project_positions() {
    let mut processor = Processor::create().unwrap();
    // 【为什么先取裸指针】`process` 需要 `*mut Processor`，而 owner 要从同一个对象里
    // 借出来。经由裸指针取 owner 之后两者不再互相借，调用点保持原样。
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活；extension_owner 是它的字段。
    let owner = unsafe { &(*raw).extension_owner };
    // 过了宽限期仍未绑定 ⇒ 本实例是 Capture 形态。
    owner.note_binding_window_closed_without_ara();
    assert_eq!(
        owner.mode(),
        crate::render::extension::InstanceMode::Capture,
        "未绑定 ARA 的实例必须被判成 Capture，否则下面这条通路根本不会被走到"
    );
    owner.set_setup_sample_rate(48_000.0);
    owner
        .capture()
        .start()
        .expect("48 kHz 已声明，可以开始录制");

    let block_frames = 8_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.5_f32; block_frames].into_boxed_slice());
    for block in 0..3_i64 {
        let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: input_planes.as_mut_ptr(),
        };
        let mut left = [0.0_f32; 8];
        let mut right = [0.0_f32; 8];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 3,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        let mut context = crate::audio_abi::ProcessContext {
            // kPlaying 必须置位：停播时宿主仍在 process 固定光标位置。
            state: 1 << 1,
            sample_rate: 48_000.0,
            project_time_samples: block * block_frames as i64,
            ..Default::default()
        };
        let mut data = ProcessData {
            process_mode: 0,
            num_samples: block_frames as i32,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            process_context: &raw mut context,
            ..Default::default()
        };
        crate::test_allocator::begin();
        let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) };
        let allocations = crate::test_allocator::end();
        assert_eq!(result, K_RESULT_OK);
        // 录制中必须直通（默认监视）：用户要能听到宿主原声，而不是自己的旧快照。
        assert_eq!(allocations, 0, "录制路径不得在音频线程上分配");
        assert_eq!(
            left, [0.5_f32; 8],
            "录制中必须把宿主输入原样送出去（monitor 默认开）"
        );
        assert_eq!(right, [0.5_f32; 8]);
    }

    let take = owner.capture().stop().expect("应当产生一个 take");
    let (timeline, regions, sources) = owner.capture().render_input().expect("有素材");
    assert_eq!(timeline.clips.len(), 1, "三个连续块应合成一个片段");
    assert_eq!(regions.len(), 1);
    assert_eq!(sources.len(), 1);
    let clip = &timeline.clips[0];
    // 24 帧 @ 48 kHz = 0.0005 秒，起点是工程位置 0。
    assert!(clip.start_sec.abs() < 1e-12);
    assert!((clip.length_sec - 24.0 / 48_000.0).abs() < 1e-12);
    assert_eq!(owner.capture().status().active_take, Some(take));
}

/// **in-place 宿主下录制必须拿到宿主的音频**（而不是我们自己写下的静音掩码）。
///
/// 【为什么这是"录出来是噪音/静音"的回归测试】`silence_flags` 写在总线结构里，而宿主
/// 可以合法地把同一个结构同时当输入和输出。此前 `clear_outputs` 会往里写"全静音"，
/// 于是**下一块**的捕获把它读成"宿主没给输入"：插件自己把自己录成了静音。
#[test]
fn in_place_buffers_still_capture_the_host_audio() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    owner.capture().start().expect("48 kHz 已声明");

    let frames = 16_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.3_f32; frames].into_boxed_slice());
    for block in 0..3_i64 {
        let mut planes = [source.as_mut_ptr(), source.as_mut_ptr()];
        let mut bus = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: planes.as_mut_ptr(),
        };
        let mut context = crate::audio_abi::ProcessContext {
            state: 1 << 1,
            sample_rate: 48_000.0,
            project_time_samples: block * frames as i64,
            ..Default::default()
        };
        let mut data = ProcessData {
            process_mode: 0,
            num_samples: frames as i32,
            num_inputs: 1,
            num_outputs: 1,
            // in-place：输入与输出是同一个结构（也就同一份掩码）。
            inputs: &raw mut bus,
            outputs: &raw mut bus,
            process_context: &raw mut context,
            ..Default::default()
        };
        assert_eq!(
            unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) },
            K_RESULT_OK
        );
        assert_eq!(bus.silence_flags, 0, "in-place 时不能写下自己的静音掩码");
    }
    owner.capture().stop().expect("应当产生一个 take");
    let (_, _, sources) = owner.capture().render_input().expect("有素材");
    let plane = sources
        .values()
        .next()
        .and_then(|pcm| pcm.planes.first())
        .expect("至少要有一条平面");
    assert!(
        plane.iter().all(|sample| (*sample - 0.3).abs() < 1e-6),
        "录下来的必须是宿主的样本（in-place 下也不能变成零），实际首样本 {:?}",
        plane.first()
    );
}

/// **宿主把同一个缓冲同时当输入和输出（in-place）时，直通必须仍然出声。**
///
/// 【为什么这是「一点输出也没有」的回归测试】`process` 此前无条件**先清输出**，而 in-place
/// 宿主下那等于把**输入**一起抹掉：直通随后读到的全是零，用户听到的是"插件在链路里、
/// 却一点声音都没有"。这是宿主**合法**的做法（同一个缓冲两用是 SDK 明说的），
/// 所以插件必须自己守住"读之前不许清"。
#[test]
fn in_place_buffers_still_pass_the_host_audio_through() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);

    let mut plane_left = [0.3_f32; 8];
    let mut plane_right = [0.6_f32; 8];
    let mut planes = [plane_left.as_mut_ptr(), plane_right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 48_000.0,
        project_time_samples: 0,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 0,
        num_samples: 8,
        num_inputs: 1,
        num_outputs: 1,
        // 【同一个缓冲】这就是 in-place：输入与输出共用一份内存。
        inputs: &raw mut bus,
        outputs: &raw mut bus,
        process_context: &raw mut context,
        ..Default::default()
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(
        plane_left, [0.3_f32; 8],
        "in-place 时输入必须原样留在缓冲里"
    );
    assert_eq!(plane_right, [0.6_f32; 8]);
}

/// **输入总线的 `silence_flags` 不是"没有输入"** —— 样本才是事实。
///
/// 【为什么这是"全静音"的回归测试】此前捕获与直通都把置位当成"这个声道没有输入"，
/// 于是把该位当作"未初始化/不关心"的宿主会让插件**录制与输出全静音**，而它其实一直在
/// 喂音频。用户报障的现象（录制与输出都全为静音、宿主里明明有音频）正是这个形状。
#[test]
fn a_silence_flagged_input_bus_is_still_captured_and_passed_through() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    owner.capture().start().expect("48 kHz 已声明");

    let frames = 16_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.25_f32; frames].into_boxed_slice());
    let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
    let mut input = AudioBusBuffers {
        num_channels: 2,
        // 宿主声称"这两个声道全是零"，但缓冲里其实是真实音频。
        silence_flags: 3,
        channel_buffers: input_planes.as_mut_ptr(),
    };
    let mut left = [0.0_f32; 16];
    let mut right = [0.0_f32; 16];
    let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 3,
        channel_buffers: output_planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 48_000.0,
        project_time_samples: 0,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 0,
        num_samples: frames as i32,
        num_inputs: 1,
        num_outputs: 1,
        inputs: &raw mut input,
        outputs: &raw mut output,
        process_context: &raw mut context,
        ..Default::default()
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    assert_eq!(left, [0.25_f32; 16], "录制中必须把宿主输入原样送出去");
    assert_eq!(right, [0.25_f32; 16]);
    assert_eq!(output.silence_flags, 0, "有非零输出就不该报静音");

    owner.capture().stop().expect("应当产生一个 take");
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    assert_eq!(timeline.clips.len(), 1, "被标成静音的输入也必须录进时间轴");
}

/// **捕获不依赖输出总线**：`num_outputs == 0` 时也要录到音频。
///
/// 【为什么这是回归测试】观察与捕获此前挂在 `process` 的 `num_outputs == 0` 早退**之后**，
/// 于是"宿主没接我们的输出"这一个与录制无关的事实会让插件既录不到东西，
/// 又让 `sawContext` 永不置位（界面上的录制键一直是灰的）。
#[test]
fn capture_happens_even_when_the_host_supplies_no_output_bus() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    owner.capture().start().expect("48 kHz 已声明");

    let frames = 16_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.4_f32; frames].into_boxed_slice());
    let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
    let mut input = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: input_planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 48_000.0,
        project_time_samples: 0,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 0,
        num_samples: frames as i32,
        num_inputs: 1,
        num_outputs: 0,
        inputs: &raw mut input,
        process_context: &raw mut context,
        ..Default::default()
    };
    assert_eq!(
        unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) },
        K_RESULT_OK
    );
    let status = owner.capture().status();
    assert!(
        status.saw_context,
        "有工程位置就必须记下来，否则录制键永远是灰的"
    );
    assert!(status.saw_input, "有输入总线就必须记下来");
    owner.capture().stop().expect("应当产生一个 take");
    assert_eq!(
        owner
            .capture()
            .render_input()
            .map(|(t, _, _)| t.clips.len()),
        Some(1),
        "没有输出总线也必须录到音频"
    );
}

/// 未录制时 Capture 通路必须**完全不介入**：既不写输出，也不碰任何状态。
///
/// 【为什么这条是契约】它是"插进任何宿主都不会改变原有行为"的全部依据。
/// 一旦未录制时它也写输出，ARA 宿主上正常渲染的实例就会被静音或直通掉。
#[test]
fn an_unarmed_capture_path_leaves_the_snapshot_output_alone() {
    let model = crate::ara::model::ModelHandle::new();
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .bind_to_document(
            model.session(),
            ara2_bridge::core::ApiGeneration::V2Final,
            ara2_bridge::plugin::ExtensionRoles::all(),
            ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .unwrap();
    processor.extension_owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44100,
            origin_sample: 0,
            left: vec![0.7; 4],
            right: vec![0.7; 4],
            covered: Vec::new(),
            _reservation: None,
        })
        .unwrap();
    let mut source = [0.1_f32; 4];
    let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
    let mut input = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: input_planes.as_mut_ptr(),
    };
    let mut left = [9.0_f32; 4];
    let mut right = [9.0_f32; 4];
    let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 3,
        channel_buffers: output_planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 44100.0,
        project_time_samples: 0,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 0,
        num_samples: 4,
        num_inputs: 1,
        num_outputs: 1,
        inputs: &raw mut input,
        outputs: &raw mut output,
        process_context: &raw mut context,
        ..Default::default()
    };
    let result =
        unsafe { (AUDIO_VTBL.process)(audio_ptr(&mut *processor), (&raw mut data).cast()) };
    assert_eq!(result, K_RESULT_OK);
    assert_eq!(
        left, [0.7_f32; 4],
        "未录制时必须输出快照，而不是直通宿主输入"
    );
    assert_eq!(right, [0.7_f32; 4]);
}

/// **渲染旁路**：录完之后插件必须发布自己的快照，而不是永远静音。
///
/// 【为什么这条是 Capture 的验收核心】捕获、投影、时间线都可以全绿，而插件**没有
/// 声音** —— 只要 `renders()` 没算上 Capture，或者 `prepare()` 因为没有 ARA 文档
/// 而静默早退。这条用例走完整条路：录 → 停 → `renders()` → 准备 → 读快照。
#[test]
fn a_recorded_take_is_rendered_and_published_through_the_snapshot_reader() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活。
    let owner = unsafe { (*raw).extension_owner.clone() };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    assert!(!owner.renders(), "还没录，不该渲染（应保持静音）");

    owner.capture().start().expect("已声明 48 kHz");
    let frames = 64_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.6_f32; frames].into_boxed_slice());
    for block in 0..8_i64 {
        let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: input_planes.as_mut_ptr(),
        };
        let mut left = [0.0_f32; 64];
        let mut right = [0.0_f32; 64];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 3,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        let mut context = crate::audio_abi::ProcessContext {
            state: 1 << 1,
            sample_rate: 48_000.0,
            project_time_samples: block * frames as i64,
            ..Default::default()
        };
        let mut data = ProcessData {
            process_mode: 0,
            num_samples: frames as i32,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            process_context: &raw mut context,
            ..Default::default()
        };
        let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) };
        assert_eq!(result, K_RESULT_OK);
    }
    owner.capture().stop().expect("应当产生一个 take");
    assert!(owner.renders(), "有素材之后必须接管渲染");

    owner.arm_capture_rendering();
    // 同步准备一次（生产路径由 UI 定时器排队驱动；这里直接调以免依赖线程调度）。
    owner
        .capture_prepare_job(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            false,
        )))
        .expect("准备必须成功");

    // 按**真实读者路径**取音：48 kHz 工程落在 48k 快照上。
    let mut left = vec![0.0_f32; 64];
    let mut right = vec![0.0_f32; 64];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // SAFETY: bus 的形状与长度由本用例保证；publisher 由 owner 保留。
    let ready = unsafe { owner.snapshots[1].copy_block(0, 48_000, &mut bus, 64) };
    assert!(ready, "录制之后必须有一份可读的快照");
    // 【为什么不比对具体样本值】这条通路下游是内核的合成（可能含神经网络声码器），
    // 具体波形不是本用例的契约。契约是"有内容、且不是全零静音"。
    let energy: f32 = left.iter().chain(right.iter()).map(|s| s.abs()).sum();
    assert!(energy > 0.0, "发布出来的快照不能是全零（那就是没出音）");
}

/// **负的 pre-roll 通过真实 process 入口也会被夹到工程 0。**
///
/// 【为什么在 ABI 边界再测一次】Capture 的单元测试只证明状态机；这里证明
/// `audio_process_inner` → `capture_observe` → `on_block_with_flow` 这条真实链路
/// 也把负位置夹掉，从而不会再生成负起点 clip。
#[test]
fn a_negative_preroll_block_is_recorded_from_project_zero() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活；extension_owner 是它的字段。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    owner.capture().start().unwrap();

    let block_frames = 8_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.5_f32; block_frames].into_boxed_slice());
    let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
    let mut input = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: input_planes.as_mut_ptr(),
    };
    let mut left = [0.0_f32; 8];
    let mut right = [0.0_f32; 8];
    let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 3,
        channel_buffers: output_planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state: 1 << 1,
        sample_rate: 48_000.0,
        // 4 帧 pre-roll + 4 帧工程内容。
        project_time_samples: -4,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: 0,
        num_samples: block_frames as i32,
        num_inputs: 1,
        num_outputs: 1,
        inputs: &raw mut input,
        outputs: &raw mut output,
        process_context: &raw mut context,
        ..Default::default()
    };
    let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) };
    assert_eq!(result, K_RESULT_OK);
    let take = owner.capture().stop().unwrap();
    assert_eq!(take, 1);
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    let clip = &timeline.clips[0];
    assert_eq!(clip.start_sec, 0.0, "负 pre-roll 不得进入 clip");
    assert!((clip.length_sec - 4.0 / 48_000.0).abs() < 1e-12);
    assert_eq!(
        owner
            .capture()
            .capture_diagnostics()
            .unwrap()
            .preroll_frames_dropped,
        4
    );
}

/// **`kPlaying` 滞后一块：真实 process 入口用下一块的连续性把开头补回来。**
#[test]
fn a_lagging_playing_bit_still_captures_the_first_process_block() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活；extension_owner 是它的字段。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();
    owner.set_setup_sample_rate(48_000.0);
    owner.capture().start().unwrap();

    let block_frames = 8_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.5_f32; block_frames].into_boxed_slice());
    for block in 0..2_i64 {
        let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: input_planes.as_mut_ptr(),
        };
        let mut left = [0.0_f32; 8];
        let mut right = [0.0_f32; 8];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 3,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        let mut context = crate::audio_abi::ProcessContext {
            // 起播窗口：宿主还没有置 kPlaying。
            state: 0,
            sample_rate: 48_000.0,
            project_time_samples: block * block_frames as i64,
            ..Default::default()
        };
        let mut data = ProcessData {
            process_mode: 0,
            num_samples: block_frames as i32,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            process_context: &raw mut context,
            ..Default::default()
        };
        let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) };
        assert_eq!(result, K_RESULT_OK);
    }

    let take = owner.capture().stop().unwrap();
    assert_eq!(take, 1);
    let (timeline, _, _) = owner.capture().render_input().expect("有素材");
    let clip = &timeline.clips[0];
    assert_eq!(clip.start_sec, 0.0);
    assert!(
        (clip.length_sec - 16.0 / 48_000.0).abs() < 1e-12,
        "两块都要在 take 里（第一块由 lookbehind 补回）"
    );
    assert_eq!(
        owner
            .capture()
            .capture_diagnostics()
            .unwrap()
            .pending_committed,
        1
    );
}

/// **停播/预渲染留下的远位置不能把起播块清成静音。**
///
/// 【为什么在 ABI 边界测】`note_block_position` 的单元测试只证明判据；这里证明
/// `audio_process_inner` 的门控真的按新判据放行，且输出仍是宿主原声（没有素材时）。
#[test]
fn a_stale_prefetch_block_does_not_silence_the_playback_start() {
    let mut processor = Processor::create().unwrap();
    let raw = (&mut *processor) as *mut Processor;
    // SAFETY: processor 在本次用例期间存活；extension_owner 是它的字段。
    let owner = unsafe { &(*raw).extension_owner };
    owner.note_binding_window_closed_without_ara();

    let block_frames = 8_usize;
    let source: &'static mut [f32] = Box::leak(vec![0.5_f32; block_frames].into_boxed_slice());
    let mut observed = Vec::new();
    for (index, start) in [480_000_i64, 48_000].into_iter().enumerate() {
        let mut input_planes = [source.as_mut_ptr(), source.as_mut_ptr()];
        let mut input = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 0,
            channel_buffers: input_planes.as_mut_ptr(),
        };
        let mut left = [0.0_f32; 8];
        let mut right = [0.0_f32; 8];
        let mut output_planes = [left.as_mut_ptr(), right.as_mut_ptr()];
        let mut output = AudioBusBuffers {
            num_channels: 2,
            silence_flags: 3,
            channel_buffers: output_planes.as_mut_ptr(),
        };
        let mut context = crate::audio_abi::ProcessContext {
            state: 0,
            sample_rate: 48_000.0,
            project_time_samples: start,
            ..Default::default()
        };
        let mut data = ProcessData {
            process_mode: 0,
            num_samples: block_frames as i32,
            num_inputs: 1,
            num_outputs: 1,
            inputs: &raw mut input,
            outputs: &raw mut output,
            process_context: &raw mut context,
            ..Default::default()
        };
        let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(raw), (&raw mut data).cast()) };
        assert_eq!(result, K_RESULT_OK);
        observed.push((index, left, right));
    }
    assert_eq!(
        observed[0].1, [0.5; 8],
        "本流第一次出现 ⇒ 允许出音一次（重复由区间去重抑制）"
    );
    assert_eq!(
        observed[1].1, [0.5; 8],
        "块首从旧位置回到播放头 ⇒ 不得被旧高水位清掉"
    );
    assert_eq!(observed[1].2, [0.5; 8]);
}

/// 通过真实 `process` 入口跑一块，返回输出左右平面。
fn run_stream_block(
    processor: &mut Processor,
    mode: i32,
    state: u32,
    start: i64,
    frames: usize,
    value: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut input_left = vec![value; frames];
    let mut input_right = vec![value; frames];
    let mut input_planes = [input_left.as_mut_ptr(), input_right.as_mut_ptr()];
    let mut input = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: input_planes.as_mut_ptr(),
    };
    let mut output_left = vec![0.0_f32; frames];
    let mut output_right = vec![0.0_f32; frames];
    let mut output_planes = [output_left.as_mut_ptr(), output_right.as_mut_ptr()];
    let mut output = AudioBusBuffers {
        num_channels: 2,
        silence_flags: 3,
        channel_buffers: output_planes.as_mut_ptr(),
    };
    let mut context = crate::audio_abi::ProcessContext {
        state,
        sample_rate: 48_000.0,
        project_time_samples: start,
        ..Default::default()
    };
    let mut data = ProcessData {
        process_mode: mode,
        num_samples: frames as i32,
        num_inputs: 1,
        num_outputs: 1,
        inputs: &raw mut input,
        outputs: &raw mut output,
        process_context: &raw mut context,
        ..Default::default()
    };
    let result = unsafe { (AUDIO_VTBL.process)(audio_ptr(processor), (&raw mut data).cast()) };
    assert_eq!(result, K_RESULT_OK);
    (output_left, output_right)
}

/// **prefetch 块（`kPlaying == 0`）的输出不得被实时传输状态清空。**
///
/// 【为什么这是“开头 0.7 s 空洞”的回归】ASIO Guard 会在起播前预取；这批输出
/// 稍后才被播放。此刻的 `kPlaying` 与内容正确性无关。
#[test]
fn a_prefetch_block_without_playing_still_outputs_audio() {
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .note_binding_window_closed_without_ara();
    for _ in 0..2 {
        let (left, right) = run_stream_block(&mut processor, 1, 0, 0, 8, 0.5);
        assert_eq!(left, vec![0.5; 8], "prefetch 输出不得被清空");
        assert_eq!(right, vec![0.5; 8]);
    }
    let diagnostics = processor.extension_owner.block_diagnostics_json(0);
    assert_eq!(
        diagnostics["streams"]["prefetch"]["gate_cleared"], 0,
        "prefetch 的清零必须是 0"
    );
}

/// 门控清零的新语义：块内不得重新输出重复内容，只允许上一块尾巴的淡出。
///
/// 【为什么不再断言"全零"】去咔要求门控**淡到零**而不是阶跃到零；短块（< 1.3 ms）
/// 会整块都是这段淡出。判据因此是"单调衰减且块尾归零"，而不是"每个样本都是零"。
fn assert_gate_faded_out(block: &[f32], what: &str) {
    assert_eq!(
        block.last().copied(),
        Some(0.0),
        "{what}：块尾必须精确归零（实测 {block:?}）"
    );
    assert!(
        block.windows(2).all(|pair| pair[0] >= pair[1]),
        "{what}：只允许单调淡出，不得重新输出重复内容（实测 {block:?}）"
    );
}

/// **同一实时区间重复投递：第一次出音，之后静音。**
///
/// 【为什么不能靠“首块保守静音”】那会把真正起播的第一块吞掉；区间去重才能
/// 同时满足“第一次必须能听到”和“停播光标不嗡嗡响”。
#[test]
fn a_repeated_realtime_block_is_silenced_after_the_first_occurrence() {
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .note_binding_window_closed_without_ara();
    let (first, _) = run_stream_block(&mut processor, 0, 0, 0, 8, 0.5);
    assert_eq!(first, vec![0.5; 8], "首次出现的区间必须出音");
    // 【T20：单块抖动不静音】门控要求**连续两块**都不推进才淡出。播放中途的
    // 一次调度抖动 / 一次区间复用只命中单块，若逐块静音，用户听到的是一次
    // 1.3 ms 的掉零 —— 那正是"播放过程中随机地咔"。
    let (second, _) = run_stream_block(&mut processor, 0, 0, 0, 8, 0.5);
    assert_eq!(second, vec![0.5; 8], "单块重复仍要出声（抖动不该变成咔）");
    let (third, _) = run_stream_block(&mut processor, 0, 0, 0, 8, 0.5);
    assert_gate_faded_out(&third, "连续不推进才允许静音");
    let diagnostics = processor.extension_owner.block_diagnostics_json(0);
    assert_eq!(diagnostics["streams"]["realtime"]["gate_cleared"], 1);
}

/// **FL Studio 形态：停播仍置 `kPlaying = 1`，同一块反复到达也不能反复出音。**
///
/// 【为什么这条是问题四的回归】旧门控只在 `kPlaying == 0` 时才清输出；只要宿主
/// 一直置 1，位置钉住的块就会每块都把同一段快照重写一遍。
#[test]
fn a_pinned_block_with_playing_one_is_silenced_after_the_first_occurrence() {
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .note_binding_window_closed_without_ara();
    let playing = crate::audio_abi::PROCESS_CONTEXT_PLAYING;
    let (first, _) = run_stream_block(&mut processor, 0, playing, 0, 8, 0.5);
    assert_eq!(first, vec![0.5; 8], "首次出现的区间必须出音");
    // 【T20】同上去抖：连续两块不推进才静音。
    let (second, _) = run_stream_block(&mut processor, 0, playing, 0, 8, 0.5);
    assert_eq!(second, vec![0.5; 8], "单块重复仍要出声");
    let (third, _) = run_stream_block(&mut processor, 0, playing, 0, 8, 0.5);
    assert_gate_faded_out(&third, "kPlaying 常置 1 不能把同一位置的重复块变成连续输出");
    let diagnostics = processor.extension_owner.block_diagnostics_json(0);
    assert_eq!(diagnostics["streams"]["realtime"]["gate_cleared"], 1);
}

/// **prefetch/实时交错时，捕获只跟随被选举的那条流，开头窗口完整。**
#[test]
fn interleaved_prefetch_and_realtime_capture_the_whole_first_window() {
    let mut processor = Processor::create().unwrap();
    processor
        .extension_owner
        .note_binding_window_closed_without_ara();
    processor.extension_owner.set_setup_sample_rate(48_000.0);
    processor.extension_owner.capture().start().unwrap();
    let frames = 1024_usize;
    for (mode, start, length) in [
        (1_i32, 0_i64, frames),
        (0, 0, 64),
        (1, frames as i64, frames),
        (0, 64, 64),
    ] {
        let _ = run_stream_block(&mut processor, mode, 0, start, length, 0.5);
    }
    let take = processor.extension_owner.capture().stop().unwrap();
    assert_eq!(take, 1);
    let diagnostics = processor
        .extension_owner
        .capture()
        .capture_diagnostics()
        .unwrap();
    assert_eq!(
        diagnostics.elected,
        Some(crate::capture::recorder::ProcessStream::Prefetch),
        "prefetch-only 的起播窗口必须由 prefetch 供料"
    );
    assert_eq!(diagnostics.stream_switches, 0);
    let (timeline, _, _) = processor.extension_owner.capture().render_input().unwrap();
    assert_eq!(timeline.clips.len(), 1);
    let clip = &timeline.clips[0];
    assert_eq!(clip.start_sec, 0.0);
    let expected = (2 * frames) as f64 / 48_000.0;
    assert!(
        (clip.length_sec - expected).abs() < 1e-9,
        "两块 prefetch 都必须进 take（实际 {}，期望 {expected}）",
        clip.length_sec
    );
}
