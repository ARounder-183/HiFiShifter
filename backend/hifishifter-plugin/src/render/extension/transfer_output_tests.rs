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
        TransferOutput::Untouched,
        "形态未定且没有素材时不介入输出 —— 宿主可能正要绑定 ARA"
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
        .transfer_ready_take
        .store(owner.transfer.active_take_id().unwrap(), Ordering::Release);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Processed,
        "处理结果就绪且属于当前 take 时优先听它"
    );
    owner.transfer().start().unwrap();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Recording,
        "录制中走直通/静音，而不是播上一次的快照"
    );
    owner.transfer().cancel();
}

/// **形态落定为 Transfer 且还没有素材时透传**，而不是把整条轨静音。
///
/// 【为什么单列这一条】"插入插件 → 整条轨没声音"是用户最容易误判成"插件坏了"的表现。
/// 但这条只在**确知是 Transfer**时才允许（形态未定时宿主可能正要绑定 ARA，
/// 那时输出归快照通路管）。
#[test]
fn a_known_transfer_instance_is_transparent_before_the_first_recording() {
    let owner = insert_owner();
    owner.note_binding_window_closed_without_ara();
    assert_eq!(owner.mode(), InstanceMode::Transfer);
    assert!(!owner.renders(), "没有素材就没什么可发布的");
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "还没录制时插件必须是透明的"
    );
}

/// **换 take 之后绝不播上一份**：两份发布都按 take 号比对。
#[test]
fn a_new_take_invalidates_both_publishers() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    owner.snapshots[0].publish(processed_snapshot()).unwrap();
    owner
        .transfer_ready_take
        .store(owner.transfer.active_take_id().unwrap(), Ordering::Release);
    assert_eq!(owner.transfer_output(), TransferOutput::Processed);

    // 又录一遍：take 号前进，两份发布都还停留在上一个 take。
    record(&owner, 256);
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::PassThrough,
        "新 take 的两份发布都还没跟上 ⇒ 透传宿主原声，而不是播上一个 take 的内容"
    );
    owner.sync_raw_tape();
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Raw,
        "原始磁带跟上新 take 之后就该听到它"
    );
}

/// 清空之后原始磁带必须被撤掉（否则仲裁会以为还有素材可播）。
#[test]
fn clearing_the_takes_retracts_the_raw_tape() {
    let owner = insert_owner();
    record(&owner, 256);
    owner.sync_raw_tape();
    let take = owner.transfer.active_take_id().unwrap();
    assert_eq!(owner.raw_tape_take.load(Ordering::Acquire), take);
    assert!(owner.raw_tape.is_ready());

    // 幂等：take 没变就不重复拷贝 PCM。
    owner.sync_raw_tape();
    assert_eq!(owner.raw_tape_take.load(Ordering::Acquire), take);

    owner.transfer().clear();
    owner.sync_raw_tape();
    assert_eq!(owner.raw_tape_take.load(Ordering::Acquire), 0);
    assert!(!owner.raw_tape.is_ready(), "没有 take 就不该留下指针");
    assert!(!owner.renders());
    assert_eq!(
        owner.transfer_output(),
        TransferOutput::Untouched,
        "素材清空之后回到「不介入」，而不是继续播一份已经不存在的原始磁带"
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
