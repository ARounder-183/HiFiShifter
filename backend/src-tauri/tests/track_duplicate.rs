//! "Clone track" ordering regression tests (integration target). Runs with
//! plain `cargo test` (helpers from `backend_lib::__test_internals`):
//!
//! ```text
//! cargo test --test track_duplicate
//! ```
//!
//! Behavioral contract: a cloned track must sit **immediately after the
//! cloned source** —
//! - child tracks: the clone becomes the next sibling of the source;
//! - root tracks (with subtree): the cloned subtree is inserted right
//!   after the source root track, preserving the other roots' order.

use backend_lib::__test_internals::TimelineState;

/// 显示顺序 = to_payload() 的轨道顺序（后端按树形 DFS 生成，每层按 order 排序）。
fn display_order(tl: &TimelineState) -> Vec<String> {
    tl.to_payload().tracks.into_iter().map(|t| t.id).collect()
}

fn find_child_ids(tl: &TimelineState, parent: &str) -> Vec<String> {
    let payload = tl.to_payload();
    match payload.tracks.iter().find(|t| t.id == parent) {
        Some(t) => t.child_track_ids.clone().unwrap_or_default(),
        None => vec![],
    }
}

#[test]
fn duplicate_root_is_inserted_right_after_source_root() {
    let mut tl = TimelineState::default();
    // 默认状态已有根轨道 track_main；再补两个根 R1 / R2。
    let r1 = tl.add_track(Some("R1".into()), None, None);
    let r2 = tl.add_track(Some("R2".into()), None, None);

    let copy_id = tl.duplicate_track(&r1);
    assert_eq!(copy_id.len(), 1, "无子树的根克隆只产生一个轨道");
    let copy = copy_id[0].clone();

    let order = display_order(&tl);
    assert_eq!(
        order,
        vec!["track_main".to_string(), r1, copy, r2],
        "克隆根必须紧贴源根之后，其余根保持相对顺序"
    );
}

#[test]
fn duplicate_root_subtree_keeps_structure_after_source() {
    let mut tl = TimelineState::default();
    let root = tl.add_track(Some("R".into()), None, None);
    let child = tl.add_track(Some("K".into()), Some(root.clone()), None);
    let tail = tl.add_track(Some("Z".into()), None, None);

    let copies = tl.duplicate_track(&root);
    assert_eq!(copies.len(), 2, "根 + 一个子轨道 = 克隆出两个轨道");
    let copy_root = copies[0].clone();
    let copy_child_id = copies[1].clone();
    let copy_child = copy_child_id.clone();

    let order = display_order(&tl);
    assert_eq!(
        order,
        vec![
            "track_main".to_string(),
            root.clone(),
            child.clone(),
            copy_root.clone(),
            copy_child,
            tail,
        ],
        "克隆子树必须整体插在源根之后、尾部根之前"
    );

    // 克隆子树结构正确：copy_root 的子级是 copy_child。
    assert_eq!(find_child_ids(&tl, &copy_root), vec![copy_child_id]);
    // 原子树未被移动。
    assert_eq!(find_child_ids(&tl, &root), vec![child]);
}

#[test]
fn duplicate_child_becomes_next_sibling_of_source() {
    let mut tl = TimelineState::default();
    let root = tl.add_track(Some("R".into()), None, None);
    let c1 = tl.add_track(Some("C1".into()), Some(root.clone()), None);
    let c2 = tl.add_track(Some("C2".into()), Some(root.clone()), None);

    let copy_id = tl.duplicate_track(&c1);
    assert_eq!(copy_id.len(), 1);
    let copy = copy_id[0].clone();

    let children = find_child_ids(&tl, &root);
    assert_eq!(
        children,
        vec![c1.clone(), copy.clone(), c2],
        "克隆的子轨道必须是源轨道的同级下一个兄弟"
    );

    // 显示顺序中克隆紧跟源子轨道（父轨道之后）。
    let order = display_order(&tl);
    let c1_pos = order.iter().position(|id| id == &c1).unwrap();
    assert_eq!(
        order.get(c1_pos + 1).map(String::as_str),
        Some(copy.as_str())
    );
}

#[test]
fn duplicate_track_to_places_clone_at_requested_root_position() {
    // “复制拖动”：拖拽根轨道到根列表最前面放置克隆。
    let mut tl = TimelineState::default();
    let r1 = tl.add_track(Some("R1".into()), None, None);
    let r2 = tl.add_track(Some("R2".into()), None, None);

    let copies = tl.duplicate_track_to(&r1, None, 0);
    assert_eq!(copies.len(), 1);
    let copy = copies[0].clone();

    assert_eq!(
        display_order(&tl),
        vec![copy, "track_main".to_string(), r1.clone(), r2],
        "克隆必须落在用户拖放的位置（根列表 index 0），源轨道保持原位"
    );
    // 源轨道仍在原位且未被改动。
    assert_eq!(find_child_ids(&tl, &r1), Vec::<String>::new());
}

#[test]
fn duplicate_track_to_supports_nesting_copy_under_another_root() {
    // “复制拖动”嵌套：把 R1 的克隆放入 R2 之下。
    let mut tl = TimelineState::default();
    let r1 = tl.add_track(Some("R1".into()), None, None);
    let r2 = tl.add_track(Some("R2".into()), None, None);

    let copies = tl.duplicate_track_to(&r1, Some(r2.clone()), 0);
    assert_eq!(copies.len(), 1);
    let copy = copies[0].clone();

    assert_eq!(
        find_child_ids(&tl, &r2),
        vec![copy.clone()],
        "克隆必须成为目标父级的子轨道"
    );
    // 源轨道保持原位（仍是无子的根轨道）。
    assert_eq!(find_child_ids(&tl, &r1), Vec::<String>::new());
    let order = display_order(&tl);
    assert_eq!(
        order,
        vec!["track_main".to_string(), r1, r2.clone(), copy],
        "显示顺序：Main、R1、R2、R2 下的克隆"
    );
}
