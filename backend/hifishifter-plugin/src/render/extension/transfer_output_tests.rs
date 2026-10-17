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
            512,
            false,
            true,
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

/// **prefetch/offline 的输出永远不被传输状态门控。**
#[test]
fn prefetch_and_offline_never_look_stopped() {
    let owner = insert_owner();
    for stream in [ProcessStream::Prefetch, ProcessStream::Offline] {
        let evidence = BlockEvidence {
            stream,
            start: 0,
            frames: 512,
            playing: false,
            recording: false,
            flow: false,
            had_previous: true,
            allow_output: false,
        };
        assert!(
            owner.transfer_should_emit(Some(evidence)),
            "{stream:?} 的输出稍后才被播放，不能因为此刻 kPlaying=0 被清空"
        );
    }
    let realtime = BlockEvidence {
        stream: ProcessStream::Realtime,
        start: 0,
        frames: 512,
        playing: false,
        recording: false,
        flow: false,
        had_previous: true,
        allow_output: false,
    };
    assert!(!owner.transfer_should_emit(Some(realtime)));
    assert!(owner.transfer_should_emit(Some(BlockEvidence {
        allow_output: true,
        ..realtime
    })));
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
