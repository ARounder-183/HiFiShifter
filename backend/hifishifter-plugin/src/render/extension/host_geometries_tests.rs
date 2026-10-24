//! 多 region 几何匹配的纯函数回归。

use crate::sync::MutexExt;

/// 把某条 region 调成与夹具 item 相容的窗口。
///
/// 夹具的 item 几何是 `D_POSITION=1 / D_LENGTH=4 / D_STARTOFFS=0 / D_PLAYRATE=0.5`。
fn align_with_fixture_item(region: &mut crate::ara::AraPlaybackRegion) {
    region.start_in_playback_time = 1.;
    region.duration_in_playback_time = 4.;
    region.start_in_modification_time = 0.;
    region.duration_in_modification_time = 2.;
}

fn fixture_host() -> &'static crate::host::MockHost {
    let host = crate::host::MockHost::new();
    host.enable_media();
    host.clear_markers();
    host.inventory_enabled.set(true);
    host
}

/// 一组 region 里能对上的那些被绑定，对不上的保持未绑定（不猜、不兜底）。
#[test]
fn only_the_regions_matching_a_host_item_are_bound() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    let first = (&*ids[0] as *const u8) as u64;
    let second = (&*ids[1] as *const u8) as u64;
    // 夹具只有一个 item，先让它与 first 相容；second 留在原位（对不上）。
    {
        let mut regions = document.regions.lock_or_recover();
        align_with_fixture_item(regions.get_mut(&first).unwrap());
    }
    // 把第二条 region 也分配给同一个 owner：这才是"一个实例管一组 region"。
    let raw = owner.binding.lock_or_recover().as_ref().unwrap().as_raw();
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            second as *mut _,
        );
    }
    let host = fixture_host();
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    let bound = owner.host_geometries().expect("至少一条 region 必须绑定");
    assert_eq!(bound.len(), 1, "对不上的 region 不得被绑定");
    assert_eq!(bound[0].region_key, first);
    assert!(bound[0].geometry.item_id.starts_with('{'));
    document.close();
}

/// 两条 region 都指向同一个 item ⇒ 只保留**一个**认领者，绝不留下两个指向同一
/// 对象的绑定。
///
/// 【为什么不"两个都丢"】旧实现把重复认领的候选**全部**丢弃，于是"同一位置两个
/// 几何全等的 item"会让两个 clip 一起消失（用户报障："片段凭空没了"）。现在按
/// region key 的稳定顺序保留第一个认领者：仍然满足"一个 item 只归一个 region"
/// （后续写回不会互相覆盖），但不会一次丢掉全部。
#[test]
fn two_regions_claiming_one_item_keep_exactly_one_owner() {
    let (model, owners, ids) = crate::editor::session::tests::workspace_fixture();
    let document = model.session();
    let owner = &owners[0];
    let first = (&*ids[0] as *const u8) as u64;
    let second = (&*ids[1] as *const u8) as u64;
    {
        let mut regions = document.regions.lock_or_recover();
        align_with_fixture_item(regions.get_mut(&first).unwrap());
        align_with_fixture_item(regions.get_mut(&second).unwrap());
    }
    let raw = owner.binding.lock_or_recover().as_ref().unwrap().as_raw();
    unsafe {
        let ext = &*raw;
        ((*ext.playbackRendererInterface).addPlaybackRegion.unwrap())(
            ext.playbackRendererRef,
            second as *mut _,
        );
    }
    let host = fixture_host();
    unsafe {
        owner.bind_host_adapter(host.context());
    }
    let bound = owner
        .host_geometries()
        .expect("重复认领必须留下恰好一个绑定，不能整组丢弃");
    assert_eq!(bound.len(), 1, "一个 item 只能归一个 region");
    assert!(
        bound[0].region_key == first || bound[0].region_key == second,
        "保留的必须是其中一个真实认领者"
    );
    document.close();
}

// ── 绑定读数的去重与退避（2026-10-12 刷屏缺陷的回归）───────────────────────────
//
// 【为什么这一组必须存在】用户报障：REAPER 挂后台 35 分钟，`multi-region owner` /
// `no unique host item` 两条日志输出了 16,769 次，把三个 8MB 轮转件全部写满。根因是
// 这两条日志**无条件**输出，而 folder 父轨上"2 个 region 对 1 个候选 item"是结构性
// 不匹配、100ms 内不会自愈。去重与退避各自挡掉这条路径的一半：去重挡日志，退避挡
// 同一节奏上的**完整宿主工程枚举**（35 分钟约 22,000 次）。

/// 造一份"分配了但一条都没绑上"的读数。
fn unbound_report(candidates: usize) -> super::BindingReport {
    super::BindingReport {
        assigned: vec![1, 2],
        bound: Vec::new(),
        unmatched: vec![1, 2],
        dropped: 0,
        candidates,
    }
}

/// **不变量：同一份绑定读数只输出一次。**
///
/// 反向断言（候选数变化必须穿透）同样重要：去重最常见的失败模式是把真实变化一起吞掉，
/// 那样用户改了工程却在日志里看不到任何新读数。
#[test]
fn a_repeated_binding_reading_is_not_reported_twice() {
    let owner = super::ExtensionOwner::default();
    let report = unbound_report(1);
    assert!(owner.binding_report_is_new(&report), "第一次必须输出");
    for _ in 0..20 {
        assert!(
            !owner.binding_report_is_new(&report),
            "同一读数不得重复输出 —— 这正是刷屏的成因"
        );
    }
    let changed = unbound_report(2);
    assert!(
        owner.binding_report_is_new(&changed),
        "候选数变了就是新读数，去重不得吞掉真实变化"
    );
}

/// **不变量：失败 → 恢复 → 同样的失败，第二条必须输出。**
///
/// 【为什么要单独钉】去重状态如果不随"恢复"清空，用户改好工程再改坏就再也看不到读数 ——
/// 那比刷屏更难查（日志里"什么都没发生"）。
#[test]
fn a_recurring_failure_after_recovery_is_reported_again() {
    let owner = super::ExtensionOwner::default();
    let failure = unbound_report(1);
    assert!(owner.binding_report_is_new(&failure));
    let recovered = super::BindingReport {
        assigned: vec![1, 2],
        bound: vec![1, 2],
        unmatched: Vec::new(),
        dropped: 0,
        candidates: 2,
    };
    assert!(
        !owner.binding_report_is_new(&recovered),
        "全部绑上不是需要报告的读数"
    );
    assert!(
        owner.binding_report_is_new(&failure),
        "恢复后再次失败必须重新报告"
    );
}

/// **不变量：重复认领（dropped>0）也算需要报告的读数。**
#[test]
fn a_dropped_claim_is_reported_even_without_unmatched_regions() {
    let owner = super::ExtensionOwner::default();
    let dropped = super::BindingReport {
        assigned: vec![1, 2],
        bound: vec![1],
        unmatched: Vec::new(),
        dropped: 1,
        candidates: 1,
    };
    assert!(owner.binding_report_is_new(&dropped));
    assert!(!owner.binding_report_is_new(&dropped));
}

/// **不变量：结构性失败必须退避且封顶。**
///
/// 【为什么封顶是硬要求】没有上限就等于把 10Hz 换成"永不枚举"：宿主稍后自己修好了
/// 这个不匹配（例如用户把 item 拖到匹配位置但没推进代次），插件必须还能自己发现。
#[test]
fn structural_binding_failures_back_off_and_cap() {
    let owner = super::ExtensionOwner::default();
    assert_eq!(
        owner.enumeration_interval(),
        super::HOST_ENUMERATION_INTERVAL,
        "无失败时保持 100ms"
    );
    let failure: Result<Vec<crate::host::geometry::BoundHostGeometry>, super::GeometryFailure> =
        Err(super::GeometryFailure::Structural(
            super::BINDING_NO_UNIQUE_ITEM,
        ));
    // 第一次失败：记录代次，档位保持 0 ⇒ 下一次重试仍是 100ms（不牺牲首次重试速度）。
    owner.note_geometry_outcome(&failure, Some(7));
    assert_eq!(
        owner.enumeration_interval(),
        super::HOST_ENUMERATION_INTERVAL
    );
    let mut previous = std::time::Duration::ZERO;
    for _ in 0..20 {
        owner.note_geometry_outcome(&failure, Some(7));
        let delay = owner.enumeration_interval();
        assert!(delay >= previous, "退避不得回退");
        assert!(
            delay <= std::time::Duration::from_secs(4),
            "退避必须封顶，否则宿主自己修好了也发现不了"
        );
        previous = delay;
    }
    assert_eq!(
        previous,
        std::time::Duration::from_secs(4),
        "上限必须是 4 秒"
    );
}

/// **不变量：宿主几何代次一变就立刻回到全速。**
///
/// 【为什么这是核心】退避不得牺牲响应速度：用户拖完 item 必须立刻看到新几何。这条是
/// 唯一的保证 —— 去掉它，用户改完工程要等 4 秒。
#[test]
fn a_host_change_resets_the_backoff_immediately() {
    let owner = super::ExtensionOwner::default();
    let failure: Result<Vec<crate::host::geometry::BoundHostGeometry>, super::GeometryFailure> =
        Err(super::GeometryFailure::Structural(
            super::BINDING_NO_UNIQUE_ITEM,
        ));
    for _ in 0..20 {
        owner.note_geometry_outcome(&failure, Some(7));
    }
    assert_eq!(
        owner.enumeration_interval(),
        std::time::Duration::from_secs(4)
    );
    owner.note_geometry_outcome(&failure, Some(8));
    assert_eq!(
        owner.enumeration_interval(),
        super::HOST_ENUMERATION_INTERVAL,
        "工程真变了必须立刻回到 100ms"
    );
}

/// **不变量：状态变化类失败不退避。**
///
/// 授权撤销 / 文档关闭要么很快自愈、要么已经是终态 —— 对它们退避只会拖慢恢复。
#[test]
fn non_structural_failures_do_not_back_off() {
    let owner = super::ExtensionOwner::default();
    for error in [
        "host geometry authorization revoked",
        "host extension unavailable",
        "host project changed during cached geometry refresh",
    ] {
        let failure: Result<Vec<crate::host::geometry::BoundHostGeometry>, super::GeometryFailure> =
            Err(error.into());
        for _ in 0..10 {
            owner.note_geometry_outcome(&failure, Some(1));
        }
        assert_eq!(
            owner.enumeration_interval(),
            super::HOST_ENUMERATION_INTERVAL,
            "{error} 是状态变化，不是结构性问题"
        );
    }
}

/// **不变量：一次成功求解清零全部退避。**
#[test]
fn a_successful_solve_resets_the_backoff() {
    let owner = super::ExtensionOwner::default();
    let failure: Result<Vec<crate::host::geometry::BoundHostGeometry>, super::GeometryFailure> =
        Err(super::GeometryFailure::Structural(
            super::BINDING_NO_UNIQUE_ITEM,
        ));
    for _ in 0..20 {
        owner.note_geometry_outcome(&failure, Some(1));
    }
    assert!(owner.enumeration_interval() > super::HOST_ENUMERATION_INTERVAL);
    owner.note_geometry_outcome(&Ok(Vec::new()), Some(1));
    assert_eq!(
        owner.enumeration_interval(),
        super::HOST_ENUMERATION_INTERVAL
    );
}
