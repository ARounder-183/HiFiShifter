//! 用真实 ARA 扩展回调验证 renderer 分配通知与两种释放顺序，不伪造宿主模型数据。

use ara2_bridge::core::ApiGeneration;
use ara2_bridge::plugin::{ExtensionBinding, ExtensionRoles};
use std::sync::{Arc, Mutex};

/// 分配观察器事件：`(role bits, 区域键列表)`。
type AssignmentEvents = Arc<Mutex<Vec<(i32, Vec<usize>)>>>;

/// editor sequence 也是实际分配输入，新增/删除必须通知而非只有 region 通知。
#[test]
fn editor_sequences_notify_without_affecting_playback_roles() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let captured = events.clone();
    let (binding, _lease) = ExtensionBinding::new_with_renderer_observer(
        ApiGeneration::V2Final,
        ExtensionRoles::all(),
        ExtensionRoles::EDITOR_RENDERER,
        ExtensionRoles::EDITOR_RENDERER,
        Arc::new(move |role, regions, sequences| {
            captured
                .lock()
                .unwrap()
                .push((role.bits(), regions.to_vec(), sequences.to_vec()))
        }),
    )
    .unwrap();
    let mut identity = 0_u8;
    let key = (&raw mut identity) as usize;
    // SAFETY: 真实扩展接口及 sequence 不透明身份在全部回调期间存活。
    unsafe {
        let instance = &*binding.as_raw();
        let api = &*instance.editorRendererInterface;
        api.addRegionSequence.unwrap()(instance.editorRendererRef, key as *mut _);
        api.removeRegionSequence.unwrap()(instance.editorRendererRef, key as *mut _);
    }
    assert_eq!(
        *events.lock().unwrap(),
        [(2, vec![], vec![key]), (2, vec![], vec![])]
    );
}

/// 观察器重入查询不能卡住分配锁；锁若未释放，超时后报告明确失败。
#[test]
fn observer_runs_after_assignment_locks_are_released() {
    let target = Arc::new(Mutex::new(None::<std::sync::Weak<ExtensionBinding>>));
    let copied = target.clone();
    let (binding, _lease) = ExtensionBinding::new_with_assignment_observer(
        ApiGeneration::V2Final,
        ExtensionRoles::all(),
        ExtensionRoles::PLAYBACK_RENDERER,
        ExtensionRoles::PLAYBACK_RENDERER,
        Arc::new(move |_, _| {
            let binding = copied.lock().unwrap().as_ref().unwrap().upgrade().unwrap();
            let (send, receive) = std::sync::mpsc::channel();
            let query = std::thread::spawn(move || {
                send.send(binding.assignment_counts()).unwrap();
            });
            assert_eq!(
                receive
                    .recv_timeout(std::time::Duration::from_secs(2))
                    .unwrap(),
                (1, 0)
            );
            query.join().unwrap();
        }),
    )
    .unwrap();
    let binding = Arc::new(binding);
    *target.lock().unwrap() = Some(Arc::downgrade(&binding));
    binding
        .add_playback_region(ExtensionRoles::PLAYBACK_RENDERER, 123)
        .unwrap();
}

/// 两个 playback renderer 的宿主分配必须分别通知，remove 后不能留下区域。
#[test]
fn native_assignments_notify_each_renderer_independently() {
    let events_a = Arc::new(Mutex::new(Vec::new()));
    let events_b = Arc::new(Mutex::new(Vec::new()));
    let make = |events: AssignmentEvents| {
        ExtensionBinding::new_with_assignment_observer(
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            ExtensionRoles::PLAYBACK_RENDERER,
            Arc::new(move |role, keys| events.lock().unwrap().push((role.bits(), keys.to_vec()))),
        )
        .unwrap()
    };
    let (binding_a, _lease_a) = make(events_a.clone());
    let (binding_b, _lease_b) = make(events_b.clone());
    let mut identity_a = 0_u8;
    let mut identity_b = 0_u8;
    let key_a = (&raw mut identity_a) as usize;
    let key_b = (&raw mut identity_b) as usize;
    // SAFETY: 扩展及 lease 存活；区域键是只作为不透明身份传递的独立地址。
    unsafe {
        let a = &*binding_a.as_raw();
        let b = &*binding_b.as_raw();
        let api_a = &*a.playbackRendererInterface;
        let api_b = &*b.playbackRendererInterface;
        api_a.addPlaybackRegion.unwrap()(a.playbackRendererRef, key_a as *mut _);
        api_b.addPlaybackRegion.unwrap()(b.playbackRendererRef, key_b as *mut _);
        api_a.removePlaybackRegion.unwrap()(a.playbackRendererRef, key_a as *mut _);
    }
    assert_eq!(*events_a.lock().unwrap(), [(1, vec![key_a]), (1, vec![])]);
    assert_eq!(*events_b.lock().unwrap(), [(1, vec![key_b])]);
}

/// controller 先销毁时接口内存仍被 companion 保留，但不得再改变分配。
#[test]
fn controller_first_teardown_tombstones_native_assignment_calls() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let copy = events.clone();
    let (binding, lease) = ExtensionBinding::new_with_assignment_observer(
        ApiGeneration::V2Final,
        ExtensionRoles::all(),
        ExtensionRoles::PLAYBACK_RENDERER,
        ExtensionRoles::PLAYBACK_RENDERER,
        Arc::new(move |_, keys| copy.lock().unwrap().push(keys.to_vec())),
    )
    .unwrap();
    let mut identity = 0_u8;
    lease.destroy();
    // SAFETY: binding 保留接口 storage，销毁 lease 只撤销控制器访问。
    unsafe {
        let instance = &*binding.as_raw();
        let api = &*instance.playbackRendererInterface;
        api.addPlaybackRegion.unwrap()(instance.playbackRendererRef, (&raw mut identity).cast());
    }
    assert_eq!(binding.assignment_counts(), (0, 0));
    assert!(events.lock().unwrap().is_empty());
}

/// companion 先释放时 controller lease 必须继续保留接口内存到最后一次宿主调用。
#[test]
fn companion_first_teardown_keeps_native_storage_alive() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let copy = events.clone();
    let (binding, lease) = ExtensionBinding::new_with_assignment_observer(
        ApiGeneration::V2Final,
        ExtensionRoles::all(),
        ExtensionRoles::PLAYBACK_RENDERER,
        ExtensionRoles::PLAYBACK_RENDERER,
        Arc::new(move |_, keys| copy.lock().unwrap().push(keys.to_vec())),
    )
    .unwrap();
    let raw = binding.as_raw();
    drop(binding);
    assert!(lease.storage_is_alive());
    let mut identity = 0_u8;
    let key = (&raw mut identity) as usize;
    // SAFETY: lease 在调用完成前保留完整 allocation。
    unsafe {
        let instance = &*raw;
        let api = &*instance.playbackRendererInterface;
        api.addPlaybackRegion.unwrap()(instance.playbackRendererRef, key as *mut _);
    }
    assert_eq!(*events.lock().unwrap(), [vec![key]]);
    drop(lease);
}
