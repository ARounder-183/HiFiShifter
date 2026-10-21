//! Transfer 输出仲裁与"形态由素材证明"的回归。
//!
//! 【为什么这一组必须有】用户报障的全部现象（播放没输出、时间轴看不到内容、播放光标不动）
//! 都指向同一类失效：**某个环节没准备好，而插件把它表现成了静音**。这里把"什么情况下
//! 该出什么声"钉成一张真值表，让"静音"只剩"确实没有素材"一个含义。

use super::*;
use crate::transfer::InputBlock;

/// 一个"宿主把本插件当普通插入"的 owner：从未绑定、形态未定、窗口也没开过。
///
/// 这正是报障现场的形态：宿主把 VST3 当普通效果器插入，用户不开插件窗口，直接按播放。
fn insert_owner() -> Arc<ExtensionOwner> {
    let owner = Arc::new(ExtensionOwner::default());
    owner.set_setup_sample_rate(48_000.0);
    owner
}

/// 录一段并 promote 成一个 take。
fn record(owner: &Arc<ExtensionOwner>, frames: usize) {
    owner.transfer().start().unwrap();
    let samples: &'static [f32] = Box::leak(vec![0.25_f32; frames].into_boxed_slice());
    for index in 0..2_i64 {
        owner.transfer().on_block(InputBlock {
            start: index * frames as i64,
            frames,
            sample_rate: 48_000.0,
            planes: [Some(samples), Some(samples)],
            playing: true,
        });
    }
    owner.transfer().stop().unwrap();
}

/// 一份可发布的最小快照（处理结果的替身）。
fn processed_snapshot() -> crate::render::snapshot::PlaybackSnapshot {
    crate::render::snapshot::PlaybackSnapshot {
        sample_rate: 48_000,
        origin_sample: 0,
        left: vec![0.1, 0.2],
        right: vec![0.1, 0.2],
        covered: Vec::new(),
        _reservation: None,
    }
}

/// **关着窗口、形态从未落定的会话，也必须能合成并发布处理结果。**
///
/// 【为什么这条是"永远是直通"的正面验收】处理结果的唯一生产者此前只在两处出现：
/// `prepare_job` 的 Transfer 分流（条件 `mode() == Transfer`），以及会话 actor。
/// 而形态由 `note_editor_tick` 落定 —— 它的唯一调用点是 WebView 的 20ms 定时器；
/// 会话则只在 `webview.rs` 的命令分发里惰性创建。**两者都要求窗口开着。**
/// 于是宿主把插件当普通插入、用户录完关掉窗口再播放时，一条处理结果都没合成过，
/// 仲裁落到 `PassThrough`：用户听到的永远是直通。
///
/// 【为什么不跑真神经合成】本用例钉的是"分流与产出链通不通"，不是合成质量。
/// `transfer_prepare_job` 走的是与生产同一份 `RenderInput::render`。
#[test]
fn a_closed_window_session_still_reaches_the_processed_output() {
    let owner = insert_owner();
    // 前提：形态确实没落定（没有 `setActive`、没有 `note_editor_tick`）。
    assert_eq!(owner.mode(), InstanceMode::Undetermined);
    record(&owner, 256);
    // `setupProcessing` 装载合成队列；不碰任何 UI 定时器。
    owner.clone().arm_transfer_rendering();
    owner
        .transfer_prepare_job(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(
            false,
        )))
        .expect("有素材就必须能准备出处理结果");
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "关着窗口、形态未落定也得出处理结果 —— 否则播放时永远是直通"
    );
}

/// **T3：读游标跳变是纯判据：没有上一块、位置相同、小于一个淡化带都不算跳变。**
///
/// 【为什么单独钉判据】它决定"这一块要不要整块回退宿主原声"。三条边界各自都会被
/// 真实宿主碰到：会话刚开始（没有上一块）、停播光标（位置相同）、以及窗口收口时
/// 只差几帧（小于一个淡化带，被淡化吸收）。
#[test]
fn the_read_position_jump_is_a_pure_judgement() {
    let owner = insert_owner();
    let evidence = BlockEvidence {
        stream: crate::transfer::capture::ProcessStream::Realtime,
        position: 48_000,
        frames: 256,
        playing: true,
        recording: false,
        playing_rising: false,
        recording_rising: false,
        phase: crate::transfer::capture::TransportPhase::Rolling,
        had_previous: true,
        allow_output: true,
        start_window: false,
        repeated_nonzero: false,
        split: BlockSplit::whole(48_000, 256),
    };
    // ① 没有上一块 ⇒ 没有跳变。
    assert_eq!(owner.read_position_jump(Some(evidence)), None);
    // ② 上一块没写出过内容 ⇒ 仍然没有（`tail` 里是零）。
    owner.note_read_position_for_test(0);
    assert_eq!(owner.read_position_jump(Some(evidence)), None);
    // ③ 上一块写出过、位置相同 ⇒ 不是跳变。
    owner.mark_emitted_for_test();
    owner.note_read_position_for_test(48_000);
    assert_eq!(owner.read_position_jump(Some(evidence)), None);
    // ④ 只差几帧（小于一个淡化带）⇒ 被淡化吸收，不是跳变。
    owner.note_read_position_for_test(47_990);
    assert_eq!(owner.read_position_jump(Some(evidence)), None);
    // ⑤ 差 1 秒 ⇒ 跳变，且幅度可读。
    owner.note_read_position_for_test(1);
    assert_eq!(owner.read_position_jump(Some(evidence)), Some(47_999));

    // ⑥ 反向跳变同样算（回退 = 重播已听内容）。
    owner.note_read_position_for_test(96_000);
    assert_eq!(owner.read_position_jump(Some(evidence)), Some(48_000));
}
/// **T-A1-1：连续推进不是跳变。**
///
/// 【为什么这是"起播 0.4~0.7 s 一次咔 + 随机咔"的主回归】旧判据拿本块起点减**上一块
/// 起点**，而连续播放时那个差恒等于**块长**（256、512…）。于是每一块都被判成"读游标
/// 跳变" ⇒ 整块回退宿主原声 ⇒ 上一块是合成结果、这一块是干声，中间只有 64 帧混合。
/// 用户听到的就是那一下咔。
#[test]
fn a_contiguous_block_advance_is_not_a_jump() {
    let owner = insert_owner();
    let evidence = BlockEvidence {
        stream: crate::transfer::capture::ProcessStream::Realtime,
        position: 48_256,
        frames: 256,
        playing: true,
        recording: false,
        playing_rising: false,
        recording_rising: false,
        phase: crate::transfer::capture::TransportPhase::Rolling,
        had_previous: true,
        allow_output: true,
        start_window: false,
        repeated_nonzero: false,
        split: BlockSplit::whole(48_256, 256),
    };
    owner.mark_emitted_for_test();
    owner.note_read_position_for_test(48_000);
    owner.note_read_frames_for_test(256);
    assert_eq!(
        owner.read_position_jump(Some(evidence)),
        None,
        "紧接上一块的推进不是跳变"
    );
    // 但真的错位仍然要判出来：预期续接 48_256，本块却从 96_000 开始。
    let jumped = BlockEvidence {
        position: 96_000,
        split: BlockSplit::whole(96_000, 256),
        ..evidence
    };
    assert_eq!(owner.read_position_jump(Some(jumped)), Some(47_744));
}

/// **T-S1-4: Raw ↔ Processed 用 20 ms 等功率交叉，而不是 1.3 ms 淡入。**
///
/// Why this replaced the old "always fade in" rule: Raw (the captured tape) and
/// Processed (the vocoder output) are two *processings of the same material*. They are
/// highly correlated, so an equal-power crossfade does not stack two audible sources -
/// it is a transition. The old rule forced a 64-frame fade-in instead, which only hides
/// the step and leaves an audible timbre jump. That jump lands exactly 0.4-0.7 s after
/// the transport starts when the render finishes mid-playback.
///
/// A passthrough on either side is a genuinely unrelated signal (the host dry signal),
/// so that case must still fade in rather than crossfade.
#[test]
fn a_rung_change_between_unrelated_signals_fades_in() {
    let code = TransferOutput::code as fn(TransferOutput) -> u8;
    let raw = code(TransferOutput::Raw);
    let processed = code(TransferOutput::Processed);
    let stale = code(TransferOutput::StaleProcessed);
    let passthrough = code(TransferOutput::PassThrough);
    // Two processings of the same take: a wide equal-power crossfade.
    assert_eq!(ExtensionOwner::seam(processed, raw), Seam::WideCrossfade);
    assert_eq!(ExtensionOwner::seam(raw, processed), Seam::WideCrossfade);
    assert_eq!(ExtensionOwner::seam(stale, raw), Seam::WideCrossfade);
    // A passthrough on either side is a different source: fade in.
    assert_eq!(ExtensionOwner::seam(passthrough, processed), Seam::FadeIn);
    assert_eq!(ExtensionOwner::seam(processed, passthrough), Seam::FadeIn);
    // Same content family (a re-render of the same take): plain crossfade.
    assert_eq!(ExtensionOwner::seam(processed, stale), Seam::Crossfade);
    assert_eq!(ExtensionOwner::seam(raw, raw), Seam::Crossfade);
}

/// **Raw ↔ Processed 的交叉淡化必须是 20 ms，其余接缝仍是 1.3 ms。**
///
/// Why pin the width: the whole point of `Seam::WideCrossfade` is that 64 frames is too
/// short for two *correlated but differently timbred* signals. If the width ever falls
/// back to the default, the timbre jump returns and nothing else would notice.
#[test]
fn a_wide_crossfade_is_twenty_milliseconds() {
    let owner = insert_owner();
    let state = &owner.declick[ProcessStream::Realtime.index()];
    owner.arm_seam(state, Seam::WideCrossfade);
    assert_eq!(
        state.fade_len.load(Ordering::Acquire) as usize,
        OUTPUT_WIDE_CROSSFADE_FRAMES
    );
    assert!(
        OUTPUT_WIDE_CROSSFADE_FRAMES >= 15 * 48,
        "20 ms at 48 kHz is the floor for two different timbres"
    );
    owner.arm_seam(state, Seam::Crossfade);
    assert_eq!(
        state.fade_len.load(Ordering::Acquire) as usize,
        OUTPUT_DECLICK_FRAMES
    );
    owner.arm_seam(state, Seam::FadeIn);
    assert_eq!(
        state.fade_len.load(Ordering::Acquire) as usize,
        OUTPUT_DECLICK_FRAMES
    );
}
/// **分流判据必须是"未绑定 + 有素材"，而不是 `mode()`。**
///
/// 【为什么单独钉这一条】上面那条走的是 `arm_transfer_rendering` 之后的完整链路；
/// 这里单独钉 `prepare_job` 的分流本身，免得将来有人把判据改回 `mode()` 而只有
/// 集成用例能发现。
#[test]
fn the_prepare_split_does_not_depend_on_the_binding_mode() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.clone().arm_transfer_rendering();
    assert_eq!(
        owner.mode(),
        InstanceMode::Undetermined,
        "前提：形态没落定（窗口没开过）"
    );
    let before = owner.prepare_request_count();
    owner.prepare();
    assert!(
        owner.prepare_request_count() > before,
        "有素材就必须排队准备，与形态无关"
    );
}

/// **素材本身就是"这个实例是 Transfer"的证明。**
///
/// 【为什么这是回归测试】形态判定此前只由编辑器窗口的 20ms 定时器写入，且要等
/// `setActive` 之后 5 秒。宿主把插件当普通插入时用户很可能**根本没开窗口**，于是
/// `mode()` 永远是 `Undetermined` ⇒ `renders()` 恒假 ⇒ 一条快照都不发布 ⇒
/// 录制正常、界面正常、**播放时全静音**。素材是比时钟更早、更硬的证据。
#[test]
fn material_alone_makes_an_unbound_instance_transfer() {
    let owner = insert_owner();
    assert_eq!(
        owner.mode(),
        InstanceMode::Undetermined,
        "从未 setActive、从未绑定：形态确实还没落定"
    );
    assert!(!owner.renders(), "还没有素材时不发布快照（那是诚实的）");
    record(&owner, 256);
    assert!(
        owner.renders(),
        "有素材就必须渲染 —— 否则没开过窗口的会话里播放全静音"
    );
    assert!(
        matches!(
            owner.host_backend(),
            crate::host::adapter::HostBackend::Transfer(_)
        ),
        "有素材的未绑定实例必须是 Transfer 后端，而不是退成 ARA-only"
    );
}

/// **处理结果缺失时必须置位"该合成一次"。**
///
/// 【为什么这是"合成被饿死"的兜底】合成此前只由用户动作触发。一旦那条调度链出事
/// （票据自锁、actor 停摆），就没有任何东西会把结果补出来，而界面照样显示素材已就绪。
/// 音频线程只置一个原子；actor 心跳据此补排一次渲染（见 `render_wanted`）。
#[test]
fn a_missing_processed_result_asks_the_actor_to_render() {
    let owner = insert_owner();
    record(&owner, 256);
    assert!(
        !owner.take_render_wanted(),
        "前提：还没有任何东西请求过合成"
    );
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "原始磁带可播、但处理结果缺失"
    );
    assert!(
        owner.take_render_wanted(),
        "结果缺失 ⇒ 必须请求一次合成，否则 actor 停摆时永远没人补"
    );
    // 处理结果就绪之后不再请求。
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_revision
        .store(owner.transfer.revision(), Ordering::Release);
    assert_eq!(owner.transfer_output(), TransferOutput::Processed);
    assert!(
        !owner.take_render_wanted(),
        "结果已就绪时不得再请求合成（否则会无谓地反复重算）"
    );
}

/// **输出仲裁的真值表：有素材之后，任何中间状态都不允许是静音。**
#[test]
fn the_output_arbiter_degrades_instead_of_going_silent() {
    let owner = insert_owner();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "没有素材时插件必须**透明** —— 静音会让「插上插件这条轨就哑了」"
    );
    record(&owner, 256);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PendingSynthesis,
        "素材有了但两份都还没发布：输出仍是宿主原声，但必须**如实报成等待合成** \
         （与「没有素材」共用 `passthrough` 会让「合成从没被调度」完全不可见）"
    );
    assert!(
        owner.take_render_wanted(),
        "发现素材没有对应结果时必须置位「该合成一次」，否则 actor 停摆时永远没人补"
    );
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "处理结果还没合成好时必须能听到**原始磁带**"
    );
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_revision
        .store(owner.transfer.revision(), Ordering::Release);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "处理结果就绪且属于当前 take 时优先听它"
    );
    owner.transfer().start().unwrap();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "武装之后、宿主还没播的那一段是**待命**：没有任何素材在产生，因此没有\
         「用旧内容盖住新演奏」的风险。此时仍须输出处理结果 —— 否则用户画完参数线\
         听不到任何变化（「参数线不生效」）"
    );
    // 宿主开始播放 ⇒ 立刻回到直通，否则捕获会把我们自己的输出录进去
    // （in-place 宿主里输入与输出是同一块缓冲）。
    owner.note_block_position(0, 480);
    owner.note_block_position(480, 480);
    owner.transfer().observe(&crate::transfer::InputBlock {
        start: 480,
        frames: 480,
        sample_rate: 48_000.0,
        planes: [None, None],
        playing: true,
    });
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Recording,
        "宿主一开始播放就必须直通，而不是播上一次的快照"
    );
    owner.transfer().cancel();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "取消之后回到处理结果（素材还在）"
    );
}

/// **还没有素材时一律透传**，而不是把整条轨静音。
///
/// 【为什么这条是"没声音"的回归测试】此前只有"形态已落定为 Transfer"才透传，于是
/// 窗口没开过、宽限期未过的会话（`Undetermined`）在**任何素材之前**都是静音 ——
/// 用户把插件插到一条有声音的轨道上，得到的是"整条轨哑了"。
/// 真正的契约是"不要覆盖一个正常渲染的 ARA 实例的输出"，而那由 `is_bound()` /
/// `bound_once` 守卫负责（见下一条用例）。
#[test]
fn an_instance_without_material_is_transparent_regardless_of_mode() {
    let owner = insert_owner();
    assert_eq!(
        owner.mode(),
        InstanceMode::Undetermined,
        "窗口没开过、宽限期未过"
    );
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "形态未定也不是把这条轨静音的理由"
    );
    owner.note_binding_window_closed_without_ara();
    assert_eq!(owner.mode(), InstanceMode::Transfer);
    assert!(!owner.renders(), "没有素材就没什么可发布的");
    assert_eq!(owner.transfer_output(), TransferOutput::PassThrough);
}

/// **换 take 之后绝不播上一份的原始磁带；但上一版处理结果比宿主原声更接近真相。**
///
/// 【为什么这一条改了】此前新 take 落到 `PassThrough`：两份发布都还没跟上时要求"透传
/// 宿主原声"。但那段空档里用户刚录完，听到的是**与自己素材无关的那一轨** —— 听起来
/// 有声音，于是"编辑没生效"反而更难发现。上一版处理结果是**这份素材的上一版**，
/// 且 `copy_block_layered` 仍按区间取输入总线，区间外不会静音。
///
/// 【为什么原始磁带那一半的判据不变】`raw_tape_ready()` 仍按 take 号比对 —— 换 take
/// 后播上一份原始磁带是"听到了上一次录制的内容"，那是真正必须挡住的。
#[test]
fn a_new_take_invalidates_both_publishers() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_revision
        .store(owner.transfer.revision(), Ordering::Release);
    assert_eq!(owner.transfer_output(), TransferOutput::Processed);

    // 又录一遍：take 号前进，两份发布都还停留在上一个 take。
    record(&owner, 256);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::StaleProcessed,
        "新 take 的合成还没跑完时，上一版处理结果好过宿主原声"
    );
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "原始磁带跟上新 take 之后就该听到它 —— 它比上一版处理结果新"
    );
}

/// **从未发布过处理结果时不凭空造一档**：那时只能报"等待合成"。
///
/// 【为什么这条必须单独钉住】`StaleProcessed` 的判据是"发布器非空"。若它在
/// `has_material()` 为真时无条件生效，第一次录制之后、第一次合成之前就会凭空出现
/// 一个"上一版结果"，而那时根本不存在任何一版。
#[test]
fn no_published_result_never_invents_a_stale_one() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "原始磁带先到位"
    );
    // 撤掉原始磁带、且从未发布过处理结果 ⇒ 只能是"等待合成"。
    owner.raw_tape.clear();
    owner.raw_tape_revision.store(0, Ordering::Release);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PendingSynthesis,
        "没有发布过任何结果时不得凭空造一档（既不是 StaleProcessed，也不是 Raw）"
    );
}

/// 清空之后原始磁带必须被撤掉（否则仲裁会以为还有素材可播）。
#[test]
fn clearing_the_takes_retracts_the_raw_tape() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    let revision = owner.transfer.revision();
    assert_eq!(owner.raw_tape_revision.load(Ordering::Acquire), revision);
    assert!(owner.raw_tape.is_ready());

    // 幂等：素材没变就不重复拷贝 PCM。
    owner.sync_raw_tape();
    assert_eq!(owner.raw_tape_revision.load(Ordering::Acquire), revision);

    owner.transfer().clear();
    owner.sync_raw_tape();
    assert_eq!(owner.raw_tape_revision.load(Ordering::Acquire), 0);
    assert!(!owner.raw_tape.is_ready(), "没有 take 就不该留下指针");
    assert!(!owner.renders());
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "素材清空之后回到「透明」，而不是继续播一份已经不存在的原始磁带"
    );
}

/// **ARA 实例的仲裁完全不受 Transfer 改动影响。**
///
/// 【为什么必须钉住】`transfer_stage` 在 `process` 里对所有实例都被调用（包括绑定了
/// ARA 的）。若"没有素材就透传"这条规则也套到 ARA 实例上，ARA 宿主里插件会开始输出
/// **宿主原声**而不是处理结果 —— 那是整条 ARA 通路静默失效。
#[test]
fn a_bound_instance_stays_on_the_ara_path() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    assert!(owner.is_bound(), "夹具的实例是绑定在 ARA 文档上的");
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Untouched,
        "绑定的实例永远走 ARA 快照通路，哪怕它恰好有一份 Transfer 素材"
    );
    model.session().close();
}

/// 录制中的输出选项：默认监听（直通宿主原声），关掉则静音。
#[test]
fn the_recording_monitor_option_is_reachable() {
    let owner = insert_owner();
    assert!(owner.transfer().options().monitor, "默认监听");
    owner
        .transfer()
        .set_options(crate::transfer::TransferOptions {
            monitor: false,
            ..owner.transfer().options()
        });
    assert!(!owner.transfer().options().monitor);
}

/// 按音频线程的真实顺序喂两个块位置，回答"守卫认为宿主在不在推进位置"。
///
/// 【为什么走生产代码而不是自己算】判据长在 `note_block_position` 里；测试自己再算
/// 一遍就只是测了自己。
fn transport_running_after(
    owner: &Arc<ExtensionOwner>,
    previous: (i64, usize),
    current: (i64, usize),
) -> bool {
    owner.note_block_position(previous.0, previous.1);
    owner.note_block_position(current.0, current.1);
    owner.transfer_transport_running()
}

/// **起播的第一个块不能被吞掉。**
///
/// 【为什么这是"播放后约 1 秒没声音"的回归测试】停播清输出是对的，但判据此前是
/// "连着两步向前"。它隐含要求宿主把一次起播切成至少两个块 —— 宿主没有这个义务：
/// 块长由宿主决定（实测上限 6 秒）。于是一个 1 秒的起播块只算"一步"，**整块 1 秒
/// 被清成静音**。判据改成"块尾是否越过上一块的块尾"之后，一个块就足以证明时间轴在走。
#[test]
fn the_first_blocks_of_playback_are_not_swallowed() {
    let owner = insert_owner();
    // 第一个块：本流的首次出现 ⇒ 允许出音一次（重复由 last_emitted 抑制）。
    owner.note_block_position(0, 480);
    assert!(
        owner.transfer_transport_running(),
        "本流第一块必须能出音一次，而不是保守清空"
    );
    // 停播：同一个块被反复 process ⇒ 块尾不动 ⇒ 不前进 ⇒ 清输出是对的。
    assert!(
        !transport_running_after(&owner, (0, 480), (0, 480)),
        "位置钉住不动就是在停"
    );
    // 起播：块尾前进 ⇒ 立刻出声，不再需要第二个块确认。
    assert!(
        transport_running_after(&owner, (0, 480), (480, 480)),
        "块尾前进就是证据，不得多等一块"
    );
}

/// **一个长块也必须算前进。**
///
/// 【为什么单独钉住】这正是用户报障的现场：宿主把起播压成一个约 1 秒的块。
/// 只看块头时它只算"一步"；块尾把整块的长度算进来，于是立刻成立。
#[test]
fn a_single_long_block_at_playback_start_is_not_swallowed() {
    let owner = insert_owner();
    owner.note_block_position(0, 512); // 停播时的光标块
    owner.note_block_position(0, 48_000); // 起播：宿主一次给 1 秒
    assert!(
        owner.transfer_transport_running(),
        "一个长块也必须算前进，否则整秒会被清空"
    );
}

/// **停播时向后 seek，代价至多一块音频**。
///
/// 【为什么不再要求"完全不放声"】预渲染/ASIO Guard 的块在门控看来就是"位置向后
/// 跳"；如果向后一律判停，起播窗口会在"清/不清"之间反复切（用户听到的正是开头
/// 卡顿）。一次 seek 的一声是 DAW 里预期内的瞬态，与第七轮已经接受的向前 seek
/// 代价对称。
#[test]
fn a_backward_seek_costs_at_most_one_block() {
    let owner = insert_owner();
    owner.note_block_position(96_000, 480);
    assert!(
        owner.note_block_position(48_000, 480),
        "块首移动就是推进证据；不能因为方向向后就清掉起播/预渲染块"
    );
    // 之后的连续前进仍然能正常识别。
    assert!(owner.note_block_position(48_480, 480));
}

/// **停播时向前拖一次播放头，代价是至多一块音频。**
///
/// 【为什么接受这个代价】另一个方向的错法是把**整块**（可达数秒）静音。一次 seek
/// 出一声是 DAW 里预期内的行为，而整秒无声不是。判据一旦能分辨"钉住"与"前进"，
/// 就不该再为单次跳变牺牲一个块。
#[test]
fn a_forward_jump_while_stopped_costs_at_most_one_block() {
    let owner = insert_owner();
    // 位置钉在 0 很久（停播），然后用户把光标拖到 10 秒处：块尾前进一次。
    owner.note_block_position(0, 480);
    owner.note_block_position(0, 480);
    owner.note_block_position(480_000, 480);
    assert!(owner.transfer_transport_running(), "前进一次即算在走");
    // 下一块钉在新位置 ⇒ 立刻回到静音，不会持续出声。
    assert!(
        !transport_running_after(&owner, (480_000, 480), (480_000, 480)),
        "钉住不动 = 停播"
    );
}

/// 真的停播时**仍然**清输出：位置钉住不动。
///
/// 【为什么这条必须与上一条同时钉住】只放宽不放守会让 REAPER 停播时把同一小段音频
/// 反复送出去 —— 听起来是一个嗡嗡声，比静音更糟。
#[test]
fn a_really_stopped_host_is_not_treated_as_running() {
    let owner = insert_owner();
    assert!(
        !transport_running_after(&owner, (96_000, 480), (96_000, 480)),
        "钉住不动 = 停播"
    );
}

/// **ARA 实例的守卫恒为假**：它的输出语义由 ARA 通路决定。
///
/// 【为什么必须钉住】这个守卫写在 `process` 里所有实例都会经过的地方。若它对绑定实例
/// 返回真，"停播即静音"这条既有契约就在 ARA 宿主里静默失效 —— 那是整条 ARA 通路的
/// 输出行为变化，而不是一次修 bug。
#[test]
fn a_bound_ara_instance_keeps_its_output() {
    let (model, owner, _identity) = crate::editor::session::tests::fixture();
    assert!(owner.is_bound());
    assert!(
        !transport_running_after(&owner, (0, 48_000), (48_000, 48_000)),
        "绑定的实例必须保持『停播即静音』的既有行为"
    );
    model.session().close();
}

/// **停播/预渲染留下的旧高水位不能压住起播块。**
///
/// 【为什么这条是用户报障的回归】旧实现比较“本块块尾是否越过上一块块尾”。
/// 宿主预渲染的块可以落在播放头**之后**；起播后播放头还没追上它时，每一块都被
/// 判成“没前进”，于是开头约 0.7 秒被清成静音/卡顿。新判据只看当前块与上一块：
/// 块首从旧位置移到播放头 ⇒ 立刻是推进。
#[test]
fn a_stale_prefetch_position_does_not_suppress_the_start() {
    let owner = insert_owner();
    owner.note_block_position(480_000, 512); // 预渲染/停播留下的远位置
    assert!(
        owner.note_block_position(48_000, 512),
        "块首回到播放头就是推进；不能等播放头越过旧高水位"
    );
}

/// **乱序/重复块不再让门控在“清/不清”之间抖动。**
#[test]
fn out_of_order_blocks_do_not_toggle_the_gate() {
    let owner = insert_owner();
    owner.note_block_position(48_000, 512);
    for (start, frames) in [(48_512, 512), (48_000, 512), (48_512, 512)] {
        assert!(
            owner.note_block_position(start, frames),
            "块首变化就是推进证据；乱序不能把整块清掉"
        );
    }
}

/// **停播时 `kPlaying == 0` 且块首钉住：仍然判停。**
#[test]
fn a_pinned_block_with_the_same_geometry_is_still_stopped() {
    let owner = insert_owner();
    owner.note_block_position(48_000, 512);
    assert!(
        !owner.note_block_position(48_000, 512),
        "同头同长 = 停播证据（唯一的例外是起播第一块，捕获侧用 lookbehind 补）"
    );
}

/// **武装 + `kPlaying` 滞后的起播：输出立刻按“录制中”直通，而不是旧处理结果。**
#[test]
fn an_armed_lagging_start_does_not_play_the_previous_result() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_revision
        .store(owner.transfer.revision(), Ordering::Release);
    owner.transfer().start().unwrap();
    // 起播：位置前进，但宿主还没置 kPlaying。
    owner.note_block_position(0, 512);
    owner.note_block_position(512, 512);
    owner.transfer().observe(&crate::transfer::InputBlock {
        start: 512,
        frames: 512,
        sample_rate: 48_000.0,
        planes: [None, None],
        playing: false,
    });
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Recording,
        "位置流在推进 ⇒ 不能再把上一版处理结果盖在新演奏上"
    );
    owner.transfer().cancel();
}

/// **待命时重复块最终回到处理结果。**
#[test]
fn armed_standby_returns_to_the_processed_result() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_revision
        .store(owner.transfer.revision(), Ordering::Release);
    owner.transfer().start().unwrap();
    owner.note_block_position(0, 512);
    assert!(!owner.note_block_position(0, 512), "钉住 = 停播");
    owner.transfer().observe(&crate::transfer::InputBlock {
        start: 0,
        frames: 512,
        sample_rate: 48_000.0,
        planes: [None, None],
        playing: false,
    });
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "待命不是录制：重复块必须回到处理结果（否则参数线又不生效）"
    );
    owner.transfer().cancel();
}

/// **块级 trace 能区分“kPlaying 滞后但位置在推进”的起播窗口。**
#[test]
fn the_block_trace_records_a_lagging_start() {
    let owner = insert_owner();
    owner.init_block_trace();
    owner.transfer().note_project_rate(48_000.0);
    owner.transfer().start().unwrap();
    for start in [0_i64, 512] {
        owner.note_block_trace(
            0,
            0,
            ProcessStream::Realtime,
            start,
            start,
            512,
            false,
            TransportPhase::Rolling,
            false,
            false,
            1,
            true,
            true,
            crate::transfer::TransferBlockOutcome::NotPlaying.code(),
            false,
        );
    }
    let diagnostics = owner.block_diagnostics_json(8);
    assert_eq!(diagnostics["blocks_seen"], 2);
    assert_eq!(diagnostics["blocks_flow_without_playing"], 2);
    let trace = diagnostics["trace"].as_array().expect("trace array");
    assert_eq!(trace.len(), 2);
    assert_eq!(trace[0]["start"], 0);
    assert_eq!(trace[0]["playing"], false);
    assert_eq!(trace[0]["flow"], true);
    assert_eq!(trace[1]["start"], 512);
    assert!(owner.take_trace_wanted(), "可疑窗口必须请求一次落盘");
    assert!(!owner.take_trace_wanted(), "同一次窗口不得重复请求");
    owner.transfer().cancel();
}

/// **prefetch 的移动不能替实时流的重复块作“在前进”的证。**
///
/// 【为什么这是“开头小段反复播放”的最小回归】两条流共用 `last_block_start` 时，
/// prefetch 每移动一次，实时流就会被判一次“前进”，门控随即放行同一小段。
#[test]
fn prefetch_movement_does_not_unlock_a_repeated_realtime_block() {
    let owner = insert_owner();
    assert!(
        owner
            .note_stream_position(ProcessStream::Realtime, 48_000, 512)
            .0
    );
    assert!(
        owner
            .note_stream_position(ProcessStream::Prefetch, 480_000, 32_768)
            .0
    );
    let (flow, had) = owner.note_stream_position(ProcessStream::Realtime, 48_000, 512);
    assert!(had);
    assert!(
        !flow,
        "同一个实时块在 prefetch 移动之后仍然是“重复”，不得被解锁"
    );
    assert!(!owner.transfer_transport_running());
}

/// **起播窗口内的静音不能被门控清掉。**
///
/// 【为什么这是问题一的输出侧回归】窗口内的前段可能是静音；按
/// `allow_output == false` 清掉它，内容出现时就是一次零→内容的阶跃。
/// `repeated_nonzero` 仍然必须阻止 FL 停播继续出声。
#[test]
fn the_start_window_keeps_the_output_open_across_pinned_silence() {
    let owner = insert_owner();
    let evidence = BlockEvidence {
        stream: ProcessStream::Realtime,
        position: 0,
        frames: 512,
        playing: false,
        recording: false,
        playing_rising: false,
        recording_rising: false,
        phase: TransportPhase::Resting,
        had_previous: true,
        allow_output: false,
        start_window: true,
        repeated_nonzero: false,
        split: BlockSplit::whole(0, 512),
    };
    assert!(
        owner.transfer_should_emit(Some(evidence), false),
        "窗口内静音必须放行，否则内容出现时是零→内容的阶跃"
    );
    let repeated = BlockEvidence {
        repeated_nonzero: true,
        ..evidence
    };
    assert!(
        !owner.transfer_should_emit(Some(repeated), false),
        "FL 停播的重复非零内容仍然必须被门控"
    );
}

/// **门控清零后的第一块必须淡入，而不是原样输出。**
#[test]
fn a_gate_clear_arms_a_fade_in_on_resume() {
    let owner = insert_owner();
    let frames = 64_usize;
    let mut first = vec![0.5_f32; frames];
    let mut second = vec![0.5_f32; frames];
    let mut planes = [first.as_mut_ptr(), second.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // 先建立"上一块输出的尾巴"。
    unsafe { owner.apply_output_declick(ProcessStream::Realtime, &mut bus, frames) };
    // 门控清零：写尾巴→0，并武装下一次淡入。
    unsafe { owner.apply_gate_fade_out(ProcessStream::Realtime, &mut bus, frames) };
    assert_eq!(first[frames - 1], 0.0);
    // 恢复：整块内容为 0.5，前 64 帧必须从 0 单调升到 0.5。
    for sample in first.iter_mut().chain(second.iter_mut()) {
        *sample = 0.5;
    }
    unsafe { owner.apply_output_declick(ProcessStream::Realtime, &mut bus, frames) };
    assert!(
        first.windows(2).all(|pair| pair[0] <= pair[1]),
        "恢复后的前 64 帧必须单调淡入"
    );
    assert!(
        first[0] < 0.1 && (first[frames - 1] - 0.5).abs() < 1e-6,
        "第一帧接近 0、最后一帧到完整幅度（实测 {:?}）",
        (first[0], first[frames - 1])
    );
}

/// **prefetch/offline 的输出永远不被传输状态门控。**
#[test]
fn prefetch_and_offline_never_look_stopped() {
    let owner = insert_owner();
    for stream in [ProcessStream::Prefetch, ProcessStream::Offline] {
        let evidence = BlockEvidence {
            stream,
            position: 0,
            frames: 512,
            playing: false,
            recording: false,
            playing_rising: false,
            recording_rising: false,
            phase: TransportPhase::Rolling,
            had_previous: true,
            allow_output: false,
            start_window: false,
            repeated_nonzero: false,
            split: BlockSplit::whole(0, 512),
        };
        assert!(
            owner.transfer_should_emit(Some(evidence), false),
            "{stream:?} 的输出稍后才被播放，不能因为此刻 kPlaying=0 被清空"
        );
    }
    let realtime = BlockEvidence {
        stream: ProcessStream::Realtime,
        position: 0,
        frames: 512,
        playing: false,
        recording: false,
        playing_rising: false,
        recording_rising: false,
        phase: TransportPhase::Rolling,
        had_previous: true,
        allow_output: false,
        start_window: false,
        repeated_nonzero: false,
        split: BlockSplit::whole(0, 512),
    };
    assert!(!owner.transfer_should_emit(Some(realtime), false));
    assert!(owner.transfer_should_emit(
        Some(BlockEvidence {
            allow_output: true,
            ..realtime
        }),
        false
    ));
}

/// **面板读数按流取 OR：prefetch 的 0 不能清掉实时流的 1。**
#[test]
fn prefetch_playing_zero_does_not_clear_realtime_playing() {
    let owner = insert_owner();
    let samples: &'static [f32] = Box::leak(vec![0.25_f32; 8].into_boxed_slice());
    let realtime = InputBlock {
        start: 0,
        frames: 8,
        sample_rate: 48_000.0,
        planes: [Some(samples), Some(samples)],
        playing: true,
    };
    owner
        .transfer()
        .observe_with_stream(&realtime, ProcessStream::Realtime);
    let prefetch = InputBlock {
        playing: false,
        ..realtime
    };
    owner
        .transfer()
        .observe_with_stream(&prefetch, ProcessStream::Prefetch);
    assert!(
        owner.transfer().host_playing(),
        "prefetch 的 kPlaying=0 不得覆盖实时流的 kPlaying=1"
    );
    assert!(!owner.transfer().capture_stream_playing(), "尚未选举捕获流");
}

/// **位置钉住：第一块之后进入 Resting，块长抖动不能把它重新判成前进。**
#[test]
fn a_pinned_stream_enters_resting_after_the_first_block() {
    let owner = insert_owner();
    assert!(
        owner
            .note_stream_position(ProcessStream::Realtime, 0, 512)
            .0,
        "首块允许一次"
    );
    assert!(
        !owner
            .note_stream_position(ProcessStream::Realtime, 0, 512)
            .0,
        "同一块首的第二次出现是 Resting"
    );
    assert!(
        !owner
            .note_stream_position(ProcessStream::Realtime, 0, 480)
            .0,
        "停播时块长抖动不是位置推进"
    );
}

/// **停播块之后出现一个显著更大的块：只允许一次扩展，然后回到 Resting。**
#[test]
fn a_pinned_start_that_grows_once_is_allowed_then_rests() {
    let owner = insert_owner();
    owner.note_stream_position(ProcessStream::Realtime, 0, 512);
    assert!(
        !owner
            .note_stream_position(ProcessStream::Realtime, 0, 512)
            .0
    );
    assert!(
        owner
            .note_stream_position(ProcessStream::Realtime, 0, 32_768)
            .0,
        "显著更大的块是起播候选，允许一次"
    );
    assert!(
        !owner
            .note_stream_position(ProcessStream::Realtime, 0, 32_768)
            .0,
        "扩展只允许一次，重复块必须 Resting"
    );
}

/// **真正播放：每个块首都在前进，永远不进入 Resting。**
#[test]
fn a_real_playback_never_rests() {
    let owner = insert_owner();
    for index in 0..8_i64 {
        assert!(
            owner
                .note_stream_position(ProcessStream::Realtime, index * 512, 512)
                .0,
            "连续前进的第 {index} 块不能 Resting"
        );
    }
}

/// **工程位置钉住、continuous 前进：有效位置按增量跟随。**
#[test]
fn a_pinned_project_position_follows_continuous_time() {
    use crate::audio_abi::ProcessContext;
    let owner = insert_owner();
    let mut context = ProcessContext {
        state: crate::audio_abi::PROCESS_CONTEXT_CONTINUOUS_VALID,
        sample_rate: 48_000.0,
        project_time_samples: 0,
        continuous_time_samples: 0,
        ..Default::default()
    };
    let first = owner.note_stream_block(ProcessStream::Realtime, &context, 512, false, false);
    assert_eq!(first.position, 0);
    context.continuous_time_samples = 512;
    let second = owner.note_stream_block(ProcessStream::Realtime, &context, 512, false, false);
    assert_eq!(second.position, 512);
    assert_eq!(
        PositionSource::from_code(
            owner.streams[ProcessStream::Realtime.index()]
                .last_source
                .load(Ordering::Acquire)
        ),
        Some(PositionSource::Continuous)
    );
    context.continuous_time_samples = 1_024;
    let third = owner.note_stream_block(ProcessStream::Realtime, &context, 512, false, false);
    assert_eq!(third.position, 1_024);
    assert_eq!(third.phase, TransportPhase::Rolling);
}

/// **短循环：块首相同但 continuous 前进 ⇒ cycle_wrap，强制 Rolling。**
#[test]
fn a_short_cycle_wrap_is_not_resting() {
    use crate::audio_abi::ProcessContext;
    let owner = insert_owner();
    let mut context = ProcessContext {
        state: crate::audio_abi::PROCESS_CONTEXT_CYCLE_ACTIVE
            | crate::audio_abi::PROCESS_CONTEXT_CYCLE_VALID
            | crate::audio_abi::PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID
            | crate::audio_abi::PROCESS_CONTEXT_TEMPO_VALID
            | crate::audio_abi::PROCESS_CONTEXT_CONTINUOUS_VALID,
        sample_rate: 1_000.0,
        project_time_samples: 0,
        continuous_time_samples: 0,
        project_time_music: 0.0,
        cycle_start_music: 0.0,
        cycle_end_music: 1.0,
        tempo: 60.0,
        ..Default::default()
    };
    let first = owner.note_stream_block(ProcessStream::Realtime, &context, 1_000, false, false);
    assert_eq!(first.phase, TransportPhase::Start);
    context.continuous_time_samples = 1_000;
    let second = owner.note_stream_block(ProcessStream::Realtime, &context, 1_000, false, false);
    assert!(second.cycle_wrap, "continuous 前进而块首相同就是回卷");
    assert_eq!(second.phase, TransportPhase::Rolling);
}

/// **块跨越循环终点：切成两段，后半段回到 cycle start。**
#[test]
fn a_block_crossing_the_cycle_end_is_split() {
    use crate::audio_abi::ProcessContext;
    let owner = insert_owner();
    let context = ProcessContext {
        state: crate::audio_abi::PROCESS_CONTEXT_CYCLE_ACTIVE
            | crate::audio_abi::PROCESS_CONTEXT_CYCLE_VALID
            | crate::audio_abi::PROCESS_CONTEXT_PROJECT_TIME_MUSIC_VALID
            | crate::audio_abi::PROCESS_CONTEXT_TEMPO_VALID,
        sample_rate: 1_000.0,
        project_time_samples: 3_900,
        project_time_music: 3.9,
        cycle_start_music: 0.0,
        cycle_end_music: 4.0,
        tempo: 60.0,
        ..Default::default()
    };
    let split = owner.block_split(&context, 200, 1_000.0, 3_900);
    assert_eq!(split.count, 2);
    assert_eq!(split.parts[0].position, 3_900);
    assert_eq!(split.parts[0].frames, 100);
    assert_eq!(split.parts[1].position, 0);
    assert_eq!(split.parts[1].input_offset, 100);
    assert_eq!(split.parts[1].frames, 100);
}

/// 在会话时间线上写一条用户曲线（Transfer 形态下"处理"的全部内容）。
fn seed_transfer_curve(owner: &Arc<ExtensionOwner>) -> Arc<crate::editor::session::EditorSession> {
    let session = owner.transfer_session().expect("transfer session");
    {
        let mut timeline = session.timeline.lock_or_recover();
        timeline.tracks.clear();
        timeline.tracks.push(hifishifter_kernel::state::Track {
            id: "transfer-track".into(),
            name: "Transfer".into(),
            parent_id: None,
            order: 0,
            muted: false,
            solo: false,
            volume: 1.0,
            compose_enabled: true,
            pitch_analysis_algo: Default::default(),
            color: String::new(),
        });
        let mut params = hifishifter_kernel::state::TrackParamsState::default();
        params.pitch_edit = vec![61.0, 63.0];
        params.pitch_edit_user_modified = true;
        timeline
            .params_by_root_track
            .insert("transfer-track".into(), params);
    }
    session
}

/// **非 ARA 的曲线必须进组件 state。**
///
/// 【为什么这是本轮最重要的回归之一】用户报"保存工程后重开，参数全丢"。根因是
/// `encode_state` 在没有 ARA 文档时序列化 `self.edits` —— 一个从未被 Transfer
/// 路径写过的默认字段。这条钉住真正的权威（会话时间线）确实被存了进去。
#[test]
fn transfer_curves_are_saved_with_the_component_state() {
    let owner = insert_owner();
    record(&owner, 256);
    let session = seed_transfer_curve(&owner);

    let bytes = owner.encode_state().expect("save succeeds");
    let state = crate::component_state::ComponentState::decode(&bytes).unwrap();
    let edits = crate::editor::transfer_edits::TransferEdits::decode(
        state
            .transfer_edits
            .as_deref()
            .expect("Transfer 编辑必须随组件 state 保存"),
    )
    .unwrap();
    assert_eq!(
        edits.params_by_root_track["transfer-track"].pitch_edit,
        vec![61.0, 63.0]
    );
    assert!(edits.params_by_root_track["transfer-track"].pitch_edit_user_modified);
    session.close();
}

/// **恢复时没有会话也不能丢**：字节块先暂存在 owner 上，创建会话时补投。
///
/// 【为什么必须有这一条】`setState` 到达时用户往往从没开过编辑器窗口 —— 那一刻没有
/// 会话可投递。若那条路直接返回，恢复就静默丢失（与修复前的症状完全一样）。
#[test]
fn a_saved_transfer_curve_reaches_the_session_created_after_restore() {
    let source = insert_owner();
    record(&source, 256);
    let session = seed_transfer_curve(&source);
    let bytes = source.encode_state().expect("save succeeds");
    session.close();

    let target = insert_owner();
    assert!(
        target.transfer_session_if_initialized().is_none(),
        "前提：目标实例还没有会话"
    );
    target.restore_state(&bytes).expect("restore succeeds");
    let restored = target
        .transfer_session()
        .expect("session is created on demand");
    assert_eq!(
        restored.staged_transfer_edit_roots(),
        vec!["transfer-track".to_string()],
        "恢复的曲线必须在会话创建时补投"
    );
    restored.close();
}

/// **没开过窗口时保存不能失败。**
///
/// 【为什么必须钉住】`encode_state` 会把 `Err` 变成 `K_RESULT_FALSE`，宿主据此提示
/// "插件存档失败"。用户没开窗口只是"没有曲线"，不是错误。
#[test]
fn saving_without_an_open_editor_succeeds_and_has_no_transfer_edits() {
    let owner = insert_owner();
    record(&owner, 256);
    let bytes = owner
        .encode_state()
        .expect("save succeeds without a session");
    let state = crate::component_state::ComponentState::decode(&bytes).unwrap();
    assert!(
        state.transfer_edits.is_none(),
        "没有会话 ⇒ 没有用户曲线可存，但清单仍然要在"
    );
    assert!(state.transfer.is_some(), "take 清单必须仍然随存档走");
}

/// 一个"位置钉住、音频在流动"的宿主上下文（VEGAS 起播窗口的形状）。
fn pinned_context(position: i64) -> crate::audio_abi::ProcessContext {
    crate::audio_abi::ProcessContext {
        state: crate::audio_abi::PROCESS_CONTEXT_PLAYING,
        sample_rate: 48_000.0,
        project_time_samples: position,
        ..Default::default()
    }
}

/// **位置钉住但音频在流动 ⇒ 位置由本插件按块长推进。**
///
/// 【为什么这是问题一/二的回归】VEGAS 起播窗口里 `projectTimeSamples` 不动，而输入
/// 已经在按块长流动。判据若只看位置，这些块会被判成 `Resting`：捕获拒绝写入、输出
/// 被清空。这条钉住"内容变化 ⇒ 位置前进"。
#[test]
fn a_pinned_start_with_live_audio_advances_the_position() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    let mut seen = Vec::new();
    for index in 0..5_u64 {
        // 每块内容不同 ⇒ 音频在流动。
        owner.streams[stream.index()].note_activity(0x1000 + index);
        let block = owner.note_stream_block(stream, &context, 256, index == 0, false);
        seen.push((block.position, block.phase));
    }
    assert_eq!(seen[0].0, 1_000, "首块落在宿主报的位置上");
    assert_eq!(seen[1].0, 1_256, "第二块必须前进一个块长");
    assert_eq!(seen[2].0, 1_512);
    assert_eq!(seen[3].0, 1_768);
    assert_eq!(seen[4].0, 2_024);
    assert_eq!(
        PositionSource::from_code(
            owner.streams[stream.index()]
                .last_source
                .load(Ordering::Acquire)
        ),
        Some(PositionSource::Synthesized)
    );
    for (position, phase) in &seen[1..] {
        assert_eq!(
            *phase,
            TransportPhase::Rolling,
            "位置 {position} 必须被当成在走，否则捕获与输出都会被拒绝"
        );
    }
    assert_eq!(
        owner.synthesized_blocks.load(Ordering::Acquire),
        5,
        "起播窗口是**区间**：上升沿之后的 5 块都按块长推进，包括首块"
    );
}

/// **停播时同一份内容反复到达仍然判 `Resting`。**
///
/// 【为什么这条必须同时存在】放宽"位置不动 ⇒ 停播"的方向正好是"停播时不得出声"。
/// 若只按"有信号"判流动，FL Studio 停播时会把同一段非零内容判成播放，用户又会听到
/// 反复播放的一小段。判据必须是"内容**在变**"。
#[test]
fn a_pinned_silent_repeat_stays_resting() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    for index in 0..20_u64 {
        // 同一份内容、**没有**播放上升沿 ⇒ 这是停播重复，不是起播窗口。
        owner.streams[stream.index()].note_activity(0x2000);
        let block = owner.note_stream_block(stream, &context, 256, false, false);
        if index > 0 {
            assert_eq!(
                block.phase,
                TransportPhase::Resting,
                "第 {index} 块的同一份内容不得被判成在走"
            );
            assert_eq!(block.position, 1_000, "位置必须保持钉住");
        }
    }
    assert_eq!(owner.synthesized_blocks.load(Ordering::Acquire), 0);
}

/// **起播窗口里的静音/持续音也必须按块长推进。**
///
/// 【为什么这是问题一的回归】VEGAS 在钉住窗口里可能先送 0.4 s 静音，再送内容。
/// 只按"内容变化"判流动会把这 0.4 s 当成停播；resync 时整段缺失，时间线在 0.66 s
/// 处被切成两个 clip。上升沿给出的一次性窗口必须覆盖它。
#[test]
fn a_pinned_silent_start_advances_the_position() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    let mut seen = Vec::new();
    for index in 0..40_u64 {
        // 同一份内容、且是全零：没有"内容变化"证据，但上升沿锁存必须推进。
        owner.streams[stream.index()].note_activity(0x6000);
        owner.streams[stream.index()].note_block_silence(true);
        let block = owner.note_stream_block(stream, &context, 256, index == 0, false);
        seen.push(block.position);
    }
    assert_eq!(seen[0], 1_000, "首块落在宿主报的位置上");
    assert_eq!(
        seen[39],
        1_000 + 39 * 256,
        "上升沿之后 1 秒内的静音块必须按块长推进"
    );
    assert!(
        owner.streams[stream.index()]
            .pinned_advance_blocks
            .load(Ordering::Acquire)
            > 0,
        "必须记录锁存推进的块数"
    );
    assert_eq!(
        PositionSource::from_code(
            owner.streams[stream.index()]
                .last_source
                .load(Ordering::Acquire)
        ),
        Some(PositionSource::Synthesized)
    );
}

/// **起播窗口里"前段静音 + 后段内容"整段都要按块长推进。**
///
/// 【为什么这是问题一 1a 的回归】VEGAS 的窗口约 0.66 s：前 0.4 s 静音、后 0.26 s 内容。
/// 旧的逐块锁存只保护静音块；内容一变化，`was_stationary` 清零与
/// `SYNTHESIZED_STALL_BLOCKS` 撤销立刻重新生效，把已记账的 0.4 s 抹掉 ——
/// 用户听到的就是"0.4 s 处咔"与"0.66 s 断成两个 clip"。区间锁存必须让两段连起来。
#[test]
fn a_window_spanning_silence_and_live_audio_advances_the_whole_window() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    let mut seen = Vec::new();
    // 20 块静音（0.4 s @ 256/48k 约 20 块）+ 20 块内容 = 40 块 / 10240 帧 ≈ 0.21 s。
    // 小于 1 s 窗口，所以 40 块全在窗口内；这条钉的是"静音段与内容段连起来"。
    for index in 0..40_u64 {
        if index < 20 {
            // 前 20 块静音：内容指纹相同、且标记为静音。
            owner.streams[stream.index()].note_activity(0x7000);
            owner.streams[stream.index()].note_block_silence(true);
        } else {
            // 后 20 块内容在变。
            owner.streams[stream.index()].note_activity(0x7000 + index);
            owner.streams[stream.index()].note_block_silence(false);
        }
        let block = owner.note_stream_block(stream, &context, 256, index == 0, false);
        seen.push(block.position);
    }
    for (index, position) in seen.iter().enumerate() {
        assert_eq!(
            *position,
            1_000 + index as i64 * 256,
            "第 {index} 块必须落在记账位置上（窗口是区间，不因内容从静音转为有声而中断）"
        );
    }
    let geometry = &owner.streams[stream.index()];
    assert_eq!(
        geometry.pinned_window_open_count.load(Ordering::Acquire),
        1,
        "一次起播只开一个窗口"
    );
    assert_eq!(
        geometry.pinned_window_closed_by.load(Ordering::Acquire),
        0,
        "40 块还在 1 s 窗口内：窗口尚未收口，因此没有收口原因"
    );
    assert!(
        geometry.pinned_window_open.load(Ordering::Acquire),
        "窗口仍然开着"
    );
    assert_eq!(
        geometry.pinned_advance_frames_used.load(Ordering::Acquire),
        40 * 256,
        "窗口内推进的帧数必须覆盖整个窗口"
    );
    assert_eq!(
        owner.synthesized_blocks.load(Ordering::Acquire),
        40,
        "40 块全部按块长记账：静音段与内容段都不能被抹掉"
    );
}

/// **T12：第一块是静音时，它的帧数也必须计入前导静音。**
///
/// 【为什么这是问题一的回归】VEGAS 的窗口从**第一块**开始就钉住位置。旧判据
/// `had && raw_start == last_raw` 在第一块上恒假（`had == false`），于是第一块的
/// 静音一次都没被累积 —— 窗口前段的静音要么全丢、要么要多两块才立起来，
/// 表现为"0.4 s 附近浮动"的一次源切换。
#[test]
fn the_first_silent_block_accumulates_the_lead_in() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    owner.streams[stream.index()].note_activity(0x9000);
    owner.streams[stream.index()].note_block_silence(true);
    let first = owner.note_stream_block(stream, &context, 256, true, false);
    assert_eq!(first.preceding_silent_frames, 0, "本块之前没有任何静音");

    // 第二块仍静音：它读到的 `preceding_silent_frames` 必须包含第一块的 256 帧。
    owner.streams[stream.index()].note_activity(0x9000);
    owner.streams[stream.index()].note_block_silence(true);
    let second = owner.note_stream_block(stream, &context, 256, false, false);
    assert_eq!(
        second.preceding_silent_frames, 256,
        "第一块（had == false）的静音必须被累积"
    );
}

/// **T12：没有输入平面（宿主标了 silence_flags）与全零平面在沉默判据上是同一件事。**
///
/// 【为什么必须钉住】`note_block_silence` 旧入参要求 `block.has_audio()`，于是
/// "宿主标了静音"被当成"没有证据" —— 整段起播窗口的静音一次都记不下来。
#[test]
fn a_block_without_audio_planes_counts_as_silent() {
    let block = crate::transfer::capture::InputBlock {
        start: 0,
        frames: 8,
        sample_rate: 48_000.0,
        planes: [None, None],
        playing: true,
    };
    assert!(block_is_silent(&block), "空平面必须被判成静音");
    let zeros: &'static [f32] = Box::leak(vec![0.0_f32; 8].into_boxed_slice());
    let block = crate::transfer::capture::InputBlock {
        start: 0,
        frames: 8,
        sample_rate: 48_000.0,
        planes: [Some(zeros), Some(zeros)],
        playing: true,
    };
    assert!(block_is_silent(&block), "全零平面同样判成静音");
}

/// **窗口一旦见到"逐块完全相同的非零内容"就立刻作废。**
///
/// 【为什么这条与上一条并存】FL Studio 停播时 `kPlaying` 仍为 1、内容逐块相同。
/// 若窗口不看内容就把同一段重复判成播放，停播光标会继续出声。这也是唯一能与
/// VEGAS 起播窗口区分的证据：VEGAS 送的是静音，FL 送的是同一份非零内容。
#[test]
fn a_repeated_nonzero_block_closes_the_window_immediately() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    let mut positions = Vec::new();
    for index in 0..20_u64 {
        // 同一份非零内容：`unchanged_run` 会累加，`last_block_silent` 为 false。
        owner.streams[stream.index()].note_activity(0x8000);
        owner.streams[stream.index()].note_block_silence(false);
        let block = owner.note_stream_block(stream, &context, 256, index == 0, false);
        positions.push(block.position);
    }
    assert_eq!(positions[0], 1_000, "首块落在宿主报的位置上");
    for (index, position) in positions.iter().enumerate().skip(1) {
        assert_eq!(
            *position, 1_000,
            "第 {index} 块的位置必须保持钉住：重复内容当场收口窗口"
        );
    }
    let geometry = &owner.streams[stream.index()];
    assert_eq!(
        geometry.pinned_window_closed_by.load(Ordering::Acquire),
        super::PINNED_WINDOW_REPEATED_CONTENT,
        "窗口必须因重复内容而收口"
    );
    assert_eq!(
        owner.synthesized_blocks.load(Ordering::Acquire),
        1,
        "只有起播那一块被记账，之后的重复内容不得继续推进"
    );
}

/// **从停播转入播放时，停播期间累积的帧数必须清零。**
///
/// 【为什么必须有这一条】宿主可以一直报 `kPlaying` 却钉住位置（停播）。那些重复块
/// 会累积"已送来的帧数"。若不清零，用户一按播放，第一块就会落到停播时长的后面 ——
/// 音频与工程位置整体错位。
#[test]
fn a_stationary_stretch_does_not_offset_the_real_start() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let context = pinned_context(1_000);
    // 先停 40 块（同一份内容）—— 这一段**没有**播放上升沿，它只是停播重复。
    for _ in 0..40_u64 {
        owner.streams[stream.index()].note_activity(0x3000);
        owner.note_stream_block(stream, &context, 256, false, false);
    }
    let mut seen = Vec::new();
    for index in 0..3_u64 {
        owner.streams[stream.index()].note_activity(0x4000 + index);
        // 真正的起播上升沿在第一块流动块上。
        let block = owner.note_stream_block(stream, &context, 256, index == 0, false);
        seen.push(block.position);
    }
    assert_eq!(
        seen,
        vec![1_000, 1_256, 1_512],
        "真实起播必须从宿主报的位置开始，不能被停播期间的重复块推后"
    );
}

/// **输出源切换必须去咔：第一块原样，第二块不得阶跃。**
///
/// 【为什么这是问题一 1c 的回归】咔声位置不固定的三个来源之一是"上一轮播放收在
/// `Raw`、这一轮首块是 `Processed`"。那种场合 `primed` 逻辑会整段跳过去咔，
/// 用户听到一次硬切。这条钉住"种类变化就淡化"。
#[test]
fn a_source_kind_change_on_the_first_block_still_fades() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    // 【T-S1-4：块长必须覆盖整条淡化带】Raw → Processed 是**同一段素材的两种处理**，
    // 走 20 ms 等功率交叉。用 64 帧的块去验一条 960 帧的淡化，"最后一帧到达新值"
    // 这条断言必然不成立 —— 那不是回归，是断言写错了对象。这里用与淡化带同宽的块，
    // 让"从旧值走到新值"在一屏之内可见。
    let frames = super::OUTPUT_WIDE_CROSSFADE_FRAMES;
    // 缓冲必须活到 `apply_output_declick` 返回之后：clang 里的临时 Vec 会在闭包
    // 返回时被释放，指针立刻变野指针（实测 STATUS_STACK_BUFFER_OVERRUN）。
    let mut first_left = vec![0.5_f32; frames];
    let mut first_right = vec![0.5_f32; frames];
    let mut first_planes = [first_left.as_mut_ptr(), first_right.as_mut_ptr()];
    let mut first = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: first_planes.as_mut_ptr(),
    };

    // 第一块：真首块（种类 4）。`primed_kind` 从 0 变成 4 ⇒ 原样输出。
    owner
        .output_token[stream.index()]
        .store(4, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut first, frames) };
    assert_eq!(
        first_left,
        vec![0.5; frames],
        "真首块必须原样输出：没有上一块可淡"
    );

    // 第二块：换到 Processed（种类 2），内容也完全不同。
    let mut second_left = vec![1.0_f32; frames];
    let mut second_right = vec![1.0_f32; frames];
    let mut second_planes = [second_left.as_mut_ptr(), second_right.as_mut_ptr()];
    let mut second = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: second_planes.as_mut_ptr(),
    };
    owner
        .output_token[stream.index()]
        .store(2, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut second, frames) };
    // 开头必须还在旧值附近（从 0.5 走向 1.0），不能第一帧就是 1.0。
    assert!(
        second_left[0] < 0.9,
        "源切换的第一帧必须留在旧值附近（实测 {}）",
        second_left[0]
    );
    assert!(
        second_left[frames - 1] > 0.9,
        "淡化的最后一帧必须到达新值（实测 {}）",
        second_left[frames - 1]
    );
    // 等功率：中点两侧各约 0.707，且不得出现线性混合那种能量凹陷。
    let middle = second_left[frames / 2];
    assert!(
        (0.35..0.95).contains(&middle),
        "中点必须落在两条信号的等功率交叠区（实测 {middle}）"
    );
}

/// **本流从未去过咔、但上一轮停在另一种源上时，首块也要淡化。**
///
/// 【为什么单独一条】`primed_kind` 的语义是"本流已知的最后种类"。上一轮播放收在
/// `Raw`，`primed_kind` 停在 4；这一轮第一块就是 `Processed`（种类 2）——
/// `first_kind != kind` 分支必须触发淡化，而不是走"真首块原样输出"。
#[test]
fn a_carryover_source_kind_still_triggers_a_fade() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let frames = super::OUTPUT_DECLICK_FRAMES;
    let mut tail_left = vec![0.5_f32; frames];
    let mut tail_right = vec![0.5_f32; frames];
    let mut tail_planes = [tail_left.as_mut_ptr(), tail_right.as_mut_ptr()];
    let mut tail = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: tail_planes.as_mut_ptr(),
    };

    // 上一轮：输出过 Raw（种类 4），给 tail 留下 0.5。
    owner
        .output_token[stream.index()]
        .store(4, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut tail, frames) };
    assert_eq!(tail_left, vec![0.5; frames], "上一轮首块原样输出");

    // 这一轮：首块直接是 Processed（种类 2），内容 1.0。
    let mut left = vec![1.0_f32; frames];
    let mut right = vec![1.0_f32; frames];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    owner
        .output_token[stream.index()]
        .store(2, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut output, frames) };
    assert!(
        left[0] < 0.9,
        "上一轮停在另一种源上时，本轮首块必须淡化（实测 {}）",
        left[0]
    );
}

/// **"透传宿主原声 → 处理结果"必须淡入，而不是把两段内容叠在一起。**
///
/// 【为什么与上一条不同】`PendingSynthesis` / `PassThrough` 输出的是宿主原声，
/// `Processed` 输出的是神经声码器结果。两者内容根本不同源，交叉淡化会把两段不同
/// 内容叠加（听感是"两声并发"）。那种场合改为对结果做一次线性淡入：旧信号置零后
/// `old * (1 - t) + current * t` 退化成 `current * t`。
#[test]
fn a_passthrough_to_processed_switch_fades_in_instead_of_crossfading() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    let frames = super::OUTPUT_DECLICK_FRAMES;
    let mut host_left = vec![0.8_f32; frames];
    let mut host_right = vec![0.8_f32; frames];
    let mut host_planes = [host_left.as_mut_ptr(), host_right.as_mut_ptr()];
    let mut host = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: host_planes.as_mut_ptr(),
    };

    // 第一块：透传宿主原声（种类 5 = PendingSynthesis）。
    owner
        .output_token[stream.index()]
        .store(5, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut host, frames) };
    assert_eq!(host_left, vec![0.8; frames], "透传首块原样输出");

    // 第二块：处理结果（种类 2），内容 0.3。
    let mut left = vec![0.3_f32; frames];
    let mut right = vec![0.3_f32; frames];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut output = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    owner
        .output_token[stream.index()]
        .store(2, std::sync::atomic::Ordering::Release);
    unsafe { owner.apply_output_declick(stream, &mut output, frames) };

    // 淡入：不应出现"宿主原声 × (1-t) + 结果 × t"的叠加值。
    // 叠加会在 t≈0.5 处给出约 0.55，而纯淡入给出约 0.15。
    let mid = left[frames / 2];
    assert!(
        mid < 0.25,
        "透传到处理结果必须淡入而不是叠加（中点实测 {mid}）"
    );
    assert_eq!(
        left[frames - 1],
        0.3,
        "淡入结束必须精确落在新值上（实测 {}）",
        left[frames - 1]
    );
}

/// **T18: an identity mismatch during restore is pending, not a failure.**
///
/// Why this matters to the user: when a project is reopened the host's graph may not be
/// rebuilt yet, so the saved track identities cannot be matched. That used to be reported as
/// "the saved parameters could not be read back" - a terminal-sounding message that sends the
/// user looking for settings that would have come back on the next edit anyway.
///
/// The strict reconcile contract is untouched (guessing an owner would attach parameters to
/// the wrong track); only how its failure is classified changed.
#[test]
fn an_identity_mismatch_during_restore_is_reported_as_pending() {
    // The three identity errors `state_channel` can produce.
    for message in [
        "ARA edit identity missing; cannot restore by session track number",
        "ARA edit identity missing or ambiguous in rebuilt host graph",
        "ARA saved edit identities are ambiguous; multiple records target one host track",
    ] {
        assert_eq!(
            ExtensionOwner::classify_restore_identity_error(message),
            RestoreIdentityVerdict::AwaitingHostGraph,
            "a missing host identity is a 'not rebuilt yet' case: {message}"
        );
    }
    // Everything else stays fatal - a broken archive must still be reported as such.
    for message in ["edit revision exhausted", "unsupported state version"] {
        assert_eq!(
            ExtensionOwner::classify_restore_identity_error(message),
            RestoreIdentityVerdict::Fatal,
            "{message}"
        );
    }
}
/// **门控清零必须是"淡到零"，而不是阶跃到零。**
///
/// 【为什么这是问题一的直接回归】`vst3.rs` 曾在门控分支里"先去咔、再
/// `clear_outputs`"：去咔把前 64 帧写好，紧接着的清零又把它们抹掉 —— 输出仍然是
/// 一次纯阶跃。`apply_gate_fade_out` 自己写满整块，调用方因此不再清零。
#[test]
fn the_emit_gate_fades_to_zero_instead_of_stepping() {
    let owner = insert_owner();
    let stream = ProcessStream::Realtime;
    // 块长 8（< 64）：门控必须在**本块内**淡到零，不能把非零尾巴留下。
    let frames = 8;
    let mut left = [1.0_f32; 8];
    let mut right = [1.0_f32; 8];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    // 先让上一块输出满幅信号，把尾巴存进去（用同一块长，保证尾巴真实）。
    unsafe { owner.apply_output_declick(stream, &mut bus, frames) };
    left.fill(1.0);
    right.fill(1.0);
    unsafe { owner.apply_gate_fade_out(stream, &mut bus, frames) };
    assert!(
        left[0] > 0.0 && left[0] < 1.0,
        "首帧必须是上一块尾巴的淡出起点（实测 {}）",
        left[0]
    );
    assert!(
        left[0] > left[1] && left[1] > left[2],
        "淡出必须单调下降（实测 {:?}）",
        &left[..4]
    );
    assert_eq!(left[frames - 1], 0.0, "块内必须精确落到零");
}

/// **预取块不因 `kPlaying == 0` 被静音（bound 实例同样适用）。**
///
/// 【为什么必须覆盖 bound】REAPER 的 ARA 播放渲染器就在预取流上工作；它会把输出
/// 缓存下来稍后播放。按 `kPlaying` 清零等于往缓存里写静音 —— 用户听到的是
/// "有概率完全没声音"。实时流仍保持"停播即静音"。
#[test]
fn a_bound_instance_does_not_silence_a_prefetch_block_without_k_playing() {
    let owner = insert_owner();
    owner.roles.store(
        ara2_bridge::plugin::ExtensionRoles::PLAYBACK_RENDERER.bits(),
        std::sync::atomic::Ordering::Release,
    );
    let prefetch = BlockEvidence {
        stream: ProcessStream::Prefetch,
        position: 0,
        frames: 512,
        playing: false,
        recording: false,
        playing_rising: false,
        recording_rising: false,
        phase: TransportPhase::Resting,
        had_previous: true,
        allow_output: false,
        start_window: false,
        repeated_nonzero: false,
        split: BlockSplit::whole(0, 512),
    };
    assert!(
        owner.transfer_should_emit(Some(prefetch), false),
        "预取输出稍后才被播放，不能因为此刻 kPlaying=0 被清空"
    );
    let realtime = BlockEvidence {
        stream: ProcessStream::Realtime,
        ..prefetch
    };
    assert!(
        !owner.transfer_should_emit(Some(realtime), false),
        "实时流的光标重复块仍然必须静音"
    );
    assert!(
        owner.transfer_should_emit(Some(realtime), true),
        "真实播放时实时流照常出声"
    );
}

/// **快照就绪探测必须查"实际会被读取的那一份"。**
///
/// 【为什么是 48 kHz】`snapshot_index_for_rate` 对 48 kHz 及倍率返回 1。此前
/// `transfer_stage` 固定查 `snapshots[0]`，于是"48 kHz 那份还没就绪"会被判成
/// 就绪 ⇒ `layered == false` ⇒ 预清后逐样本写零。
#[test]
fn the_snapshot_readiness_probe_uses_the_publisher_that_is_read() {
    let owner = insert_owner(); // setup_rate = 48 kHz
    assert_eq!(owner.snapshot_index_for_rate(48_000.0), 1);
    // 只发布 44.1 kHz 那一份：48 kHz 工程实际读的是索引 1，必须报"未就绪"。
    owner.snapshots[0]
        .publish(crate::render::snapshot::PlaybackSnapshot {
            sample_rate: 44_100,
            origin_sample: 0,
            left: vec![0.1, 0.2],
            right: vec![0.1, 0.2],
            covered: Vec::new(),
            _reservation: None,
        })
        .unwrap();
    assert!(
        !owner.processed_snapshot_ready(),
        "48 kHz 工程读索引 1，索引 0 就绪不等于实际发布器就绪"
    );
    owner.set_setup_sample_rate(44_100.0);
    assert!(
        owner.processed_snapshot_ready(),
        "44.1 kHz 工程读索引 0，就绪探测必须跟着 setup 率走"
    );
}

/// **恢复失败必须变成可读状态，而不是一行日志。**
#[test]
fn a_failed_restore_is_reported_not_swallowed() {
    let owner = insert_owner();
    assert!(!owner.restore_failed(), "从未恢复过时不是失败");
    owner.note_restore_pending();
    assert!(!owner.restore_failed(), "待恢复不是故障");
    owner.note_restore_failed("fade ownership metadata pending");
    assert!(owner.restore_failed());
    let (status, error, applied) = owner.restore_report();
    assert_eq!(status, super::RESTORE_FAILED);
    assert_eq!(error.as_deref(), Some("fade ownership metadata pending"));
    assert_eq!(applied, 0, "失败不能计入已应用");
}

/// **ARA 实例首次发布快照必须改变输出 token。**
///
/// 【为什么是问题一的 ARA 版本】bound 实例的仲裁恒为 `Untouched`；token 只含"种类"
/// 时，快照从无到有（此前逐样本写零）不会触发去咔，用户听到的是从零到有声的硬切。
#[test]
fn an_ara_snapshot_publish_changes_the_output_token() {
    let owner = insert_owner();
    let before = owner.output_token_for(TransferOutput::Untouched);
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner.snapshots[1].publish(processed_snapshot()).unwrap();
    let after = owner.output_token_for(TransferOutput::Untouched);
    assert_ne!(
        before, after,
        "快照发布必须改变 token，否则 ARA 的接缝拿不到 1.3 ms 淡化"
    );
}
