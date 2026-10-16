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
        .transfer_prepare_job(std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)))
        .expect("有素材就必须能准备出处理结果");
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "关着窗口、形态未落定也得出处理结果 —— 否则播放时永远是直通"
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
        TransferOutput::PassThrough,
        "素材有了但两份都还没发布：透传宿主原声，不是静音"
    );
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "处理结果还没合成好时必须能听到**原始磁带**"
    );
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
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
    owner.note_block_position(0);
    owner.note_block_position(480);
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
    assert_eq!(owner.mode(), InstanceMode::Undetermined, "窗口没开过、宽限期未过");
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

/// **从未发布过处理结果时不凭空造一档**：那时只能透传。
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
    // 撤掉原始磁带、且从未发布过处理结果 ⇒ 只能是透传。
    owner.raw_tape.clear();
    owner.raw_tape_revision.store(0, Ordering::Release);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "没有发布过任何结果时不得凭空造一档"
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
    owner.transfer().set_options(crate::transfer::TransferOptions {
        monitor: false,
        ..owner.transfer().options()
    });
    assert!(!owner.transfer().options().monitor);
}

/// 按音频线程的真实顺序喂两个块位置，回答"守卫认为宿主在不在推进位置"。
///
/// 【为什么走生产代码而不是自己算】判据长在 `note_block_position` 里；测试自己再算
/// 一遍就只是测了自己。
fn transport_running_after(owner: &Arc<ExtensionOwner>, previous: i64, current: i64) -> bool {
    owner.note_block_position(previous);
    owner.note_block_position(current);
    owner.transfer_transport_running()
}

/// **起播的头几块不能被吞掉。**
///
/// 【为什么这是本轮"播放后约 1 秒没声音"的回归测试】停播清输出是对的，但那个分支此前
/// 只看 `kPlaying` —— 而它与 `projectTimeSamples` 不是原子更新的。宿主从"停止"切到
/// "播放"的那几块里位置已经在走、播放位可能还没置上，于是**每次播放都吞掉开头**。
///
/// 判据必须是"位置在不在前进"：真在播就必然前进，钉住不动就是在停。
#[test]
fn the_first_blocks_of_playback_are_not_swallowed() {
    let owner = insert_owner();
    // 停播：位置钉在 0，反复 process ⇒ 不前进 ⇒ 清输出是对的。
    owner.note_block_position(0);
    assert!(
        !owner.transfer_transport_running(),
        "第一个块没有可比对象，必须保守判成没在前进"
    );
    assert!(
        !transport_running_after(&owner, 0, 0),
        "位置钉住不动就是在停"
    );
    // 起播：位置开始走，而播放位（此刻还没置上）不参与判断。
    //
    // 【为什么现在是"连着两步"才成立】"位置变了"与"宿主在播"不是一件事：停播状态下
    // 移动播放头也会变一次，而那一次足以放出一块**新位置**的音频 —— 用户报障的
    // "很短、内容不对的一声，之后约 1 秒无声，换位置也一样"正是这样来的。
    // 位置**连续向前**才是真在播的可核实证据。代价是多等一块（≈10 ms），
    // 而它换掉的是"每次移动光标都漏一小块"。
    assert!(
        !transport_running_after(&owner, 0, 480),
        "单次跳变不是证据：seek / 循环回卷 / 宿主重排都会跳一次"
    );
    // 此时上一块是 480、已累计一步。再走一块 ⇒ 两步 ⇒ 判定在播。
    // （不能重复喂 480：位置钉住会清零累计 —— 那正是"停播"的定义。）
    owner.note_block_position(960);
    assert!(
        owner.transfer_transport_running(),
        "连着两步向前后就必须照常输出，否则每次播放都吞掉开头"
    );
}

/// **后退不算"前进"**：停播时把播放头往前挪一次，不该放出一块音频。
///
/// 【为什么这条必须存在】旧的判据是 `previous != start`，方向无关。于是停播状态下
/// 移动播放头会放出一小块新位置的音频，而用户此刻并没有在播放 —— 那正是报障里
/// "很短、内容不对"的那一声。
#[test]
fn a_backward_seek_does_not_unlock_a_block() {
    let owner = insert_owner();
    owner.note_block_position(96_000);
    owner.note_block_position(48_000);
    assert!(
        !owner.transfer_transport_running(),
        "位置倒退不是播放：那是一次 seek 或循环回卷"
    );
    // 之后的连续前进仍然能正常识别。
    owner.note_block_position(48_480);
    owner.note_block_position(48_960);
    assert!(owner.transfer_transport_running());
}

/// **停播时移动一次播放头 ⇒ 仍然静音**（旧实现会放出一块）。
#[test]
fn a_single_jump_while_stopped_stays_silent() {
    let owner = insert_owner();
    // 位置钉在 0 很久（停播），然后用户把光标拖到 10 秒处：只跳了一次。
    owner.note_block_position(0);
    owner.note_block_position(0);
    owner.note_block_position(480_000);
    assert!(
        !owner.transfer_transport_running(),
        "一次跳变后再无前进 ⇒ 仍在停播，不得放出一块"
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
        !transport_running_after(&owner, 96_000, 96_000),
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
        !transport_running_after(&owner, 0, 48_000),
        "绑定的实例必须保持『停播即静音』的既有行为"
    );
    model.session().close();
}
