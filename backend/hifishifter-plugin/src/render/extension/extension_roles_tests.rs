//! ARA 角色分配的契约回归。
//!
//! 【为什么单独一组】ARA SDK 对 `ARAPlugInExtensionInstance` 的契约是**双向**的：
//!
//! > Plug-ins **must provide all interfaces that have been requested by the host through
//! > the role assignment**, and suppress interfaces explicitly excluded by the roles — e.g.
//! > if the host did not assign kARAEditorRendererRole even it was known,
//! > editorRendererInterface will be NULL.
//!
//! 少给一个宿主**已分配**的角色接口，宿主拿到的就是 NULL —— 而它会在真正用到的那一刻
//! 崩在自己的代码里。Cubase 15 实测：分配 `[editor,view]`，我们交出的
//! `editorViewInterface` 是 NULL（`supported` 里漏了 `EDITOR_VIEW`），崩点就在
//! `IEditController::createView` 之后 7ms 的 `araservice.dll` 内 —— 那时宿主正要按
//! 契约用 view 角色通知选区（"Selection and hiding of host entities is communicated to
//! editor view plug-ins"）。

use super::*;
use ara2_bridge::core::ApiGeneration;

/// **不变量：宿主分配的角色必须全部有接口，未分配的角色必须保持 NULL。**
#[test]
fn every_assigned_role_is_served_and_unassigned_ones_are_suppressed() {
    // 夹具自带的 owner 已经绑定过，这里要一个**未绑定**的实例（重复绑定会被拒）。
    let (model, _bound_owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    // Cubase 实测的分配：editor renderer + editor view（playback 由另一个实例承担）。
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Draft,
            ExtensionRoles::all(),
            ExtensionRoles::EDITOR_RENDERER | ExtensionRoles::EDITOR_VIEW,
            None,
        )
        .expect("绑定必须成功");
    assert!(!raw.is_null());
    // SAFETY: 记录是紧凑结构；整份拷出来再读（对 packed 字段取引用是 UB）。
    let instance = unsafe { raw.read_unaligned() };
    assert!(
        !instance.editorRendererInterface.is_null(),
        "已分配的 editor renderer 角色必须给出接口"
    );
    assert!(
        !instance.editorViewInterface.is_null(),
        "已分配的 editor view 角色必须给出接口 —— 宿主打开编辑器时要用它通知选区"
    );
    assert!(
        !instance.editorViewRef.is_null(),
        "view 角色的 ref 必须与接口一起给出"
    );
    // 契约的另一半：宿主没分配的角色必须为 NULL，不得凭空给出。
    assert!(
        instance.playbackRendererInterface.is_null(),
        "未分配 playback 时不得给出接口"
    );
    assert!(instance.playbackRendererRef.is_null());
    document.close();
}

/// **不变量：只分配 playback 时，editor / view 角色仍必须是 NULL。**
///
/// 【为什么必须钉住反向】放宽 `supported` 时最容易犯的错是"顺手把所有角色都填上" ——
/// 那会让宿主以为我们承担了它没有要求的角色。
#[test]
fn a_playback_only_assignment_does_not_claim_editor_roles() {
    let (model, _bound_owner, _identity) = crate::editor::session::tests::fixture();
    let document = model.session();
    let owner = Arc::new(ExtensionOwner::default());
    let raw = owner
        .bind_to_document(
            document.clone(),
            ApiGeneration::V2Final,
            ExtensionRoles::all(),
            ExtensionRoles::PLAYBACK_RENDERER,
            None,
        )
        .expect("绑定必须成功");
    assert!(!raw.is_null());
    // SAFETY: 同上，紧凑记录整份拷出。
    let instance = unsafe { raw.read_unaligned() };
    assert!(!instance.playbackRendererInterface.is_null());
    assert!(!instance.playbackRendererRef.is_null());
    assert!(
        instance.editorRendererInterface.is_null(),
        "未分配 editor renderer 时不得给出接口"
    );
    assert!(
        instance.editorViewInterface.is_null(),
        "未分配 editor view 时不得给出接口"
    );
    document.close();
}
