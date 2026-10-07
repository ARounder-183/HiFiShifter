//! `TimelineState::root_track_map` 与 `resolve_root_track_id` 的等价性对拍。
//!
//! 【为什么值得一个专门的对拍】`root_track_map` 会被塞进**按 clip** 的热循环
//! （`build_snapshot` / `handle_update_timeline` / `schedule_stretch_jobs`）
//! 替换掉逐次调用的 `resolve_root_track_id`。一旦两者在某个边界上分叉，症状是
//! "某些子轨道的参数突然不生效"这类极难归因的错误 —— 逐例对拍是唯一可靠的
//! 安全网。
//!
//! 放在 `tests/` 而不是 `model.rs` 的 `#[cfg(test)]` 里：`root_track_map` 与
//! `resolve_root_track_id` 都是公开 API，从 crate 外部就能完整覆盖，无需改动
//! 那个已经上万行的测试模块。

use hifishifter_kernel::state::{TimelineState, Track};

/// 用 `(id, parent_id)` 边表造一份只含轨道的时间轴。
fn timeline_with_track_edges(edges: &[(&str, Option<&str>)]) -> TimelineState {
    let mut tl = TimelineState::default();
    let template: Track = tl
        .tracks
        .first()
        .cloned()
        .expect("default TimelineState carries a root track to clone as a template");
    tl.tracks.clear();
    for (id, parent) in edges {
        let mut track = template.clone();
        track.id = (*id).to_string();
        track.parent_id = parent.map(|p| p.to_string());
        tl.tracks.push(track);
    }
    tl
}

/// 消费方的等价写法：查不到就退回 id 自己（`resolve_root_track_id` 对未知 id
/// 正是这么返回的）。
fn lookup(tl: &TimelineState, id: &str) -> Option<String> {
    let roots = tl.root_track_map();
    Some(roots.get(id).copied().unwrap_or(id).to_string())
}

/// 无环的各种树形都必须逐例一致。
#[test]
fn root_track_map_matches_resolve_root_track_id() {
    let cases: &[&[(&str, Option<&str>)]] = &[
        // 单根
        &[("a", None)],
        // 两个根
        &[("a", None), ("b", None)],
        // 三层链
        &[("a", None), ("b", Some("a")), ("c", Some("b"))],
        // 两棵树 + 交错顺序
        &[
            ("a", None),
            ("b", None),
            ("c", Some("b")),
            ("d", Some("c")),
            ("e", Some("a")),
        ],
        // 父不存在：链到此为止（返回自己）
        &[("a", None), ("b", Some("missing"))],
        // parent_id 为空串：视作根
        &[("a", None), ("b", Some(""))],
        // 子先于父出现（前向引用）
        &[("child", Some("parent")), ("parent", None)],
        // 深链
        &[
            ("t0", None),
            ("t1", Some("t0")),
            ("t2", Some("t1")),
            ("t3", Some("t2")),
            ("t4", Some("t3")),
        ],
        // 深链的**逆序**声明：构表时先遇到叶子
        &[
            ("t4", Some("t3")),
            ("t3", Some("t2")),
            ("t2", Some("t1")),
            ("t1", Some("t0")),
            ("t0", None),
        ],
    ];

    for edges in cases {
        let tl = timeline_with_track_edges(edges);
        let roots = tl.root_track_map();
        for (id, _) in edges.iter() {
            assert_eq!(
                Some(roots.get(id).copied().unwrap_or(id).to_string()),
                tl.resolve_root_track_id(id),
                "mismatch for id={id} edges={edges:?}"
            );
        }
    }
}

/// 未知 id 与空 id 的契约：映射查不到，`resolve_root_track_id` 分别是
/// "返回自己"与 `None` —— 消费方必须用 `unwrap_or(id)` 补齐。
#[test]
fn unknown_and_empty_ids_keep_the_original_contract() {
    let tl = timeline_with_track_edges(&[("a", None), ("b", Some("a"))]);
    let roots = tl.root_track_map();

    assert_eq!(tl.resolve_root_track_id("unknown"), Some("unknown".into()));
    assert_eq!(roots.get("unknown").copied(), None);
    assert_eq!(lookup(&tl, "unknown"), Some("unknown".into()));

    assert_eq!(tl.resolve_root_track_id(""), None);
    assert_eq!(roots.get("").copied(), None);
}

/// 环必须**终止**，且落点是环内成员。
///
/// 【为什么不与 `resolve_root_track_id` 对拍】它在环上靠 `safety > 2048` 停下，
/// 落点取决于循环次数的奇偶，本身没有可依赖的语义。这里只要求"不死循环"与
/// "落点是树内成员"。
#[test]
fn cycles_terminate_and_land_inside_the_tree() {
    let cases: &[&[(&str, Option<&str>)]] = &[
        &[("a", Some("a"))],
        &[("a", Some("b")), ("b", Some("a"))],
        &[("a", None), ("b", Some("c")), ("c", Some("b"))],
    ];

    for edges in cases {
        let tl = timeline_with_track_edges(edges);
        let roots = tl.root_track_map();
        for (id, _) in edges.iter() {
            let root = roots.get(id).copied().expect("cycle member must be mapped");
            assert!(
                edges.iter().any(|(member, _)| member == &root),
                "root {root} must be a member of the tree for id={id}"
            );
        }
    }
}
