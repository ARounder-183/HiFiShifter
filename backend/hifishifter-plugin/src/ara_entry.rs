//! 控制器身份感知的 ARA entry；复用 audited C++ shim 和 companion 绑定，不手写 ARA COM vtable。

use crate::render::{document::DocumentSession, extension::ExtensionOwner};
use crate::sync::MutexExt;
use ara2_bridge::companion::vst3::ffi::*;
use ara2_bridge::companion::{CompanionFactory, CompanionProcessorBinding, CompanionRoles};
use ara2_bridge::core::AraError;
use ara2_bridge::plugin::ExtensionRoles;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

/// 处理器级 companion 绑定的**可复位**容器。
///
/// 【为什么需要容器】`CompanionProcessorBinding::bind` 是一次性的：companion 只把
/// `lifecycle.bound` 写成 `true`，`CompanionControllerBinding` 的 `Drop` 也不复位它。
/// 而 VST3 处理器实例会跨工程存活 —— 宿主重开工程时复用同一个 FX 槽，于是第二次
/// `bind` 必然返回 `Err("companion processor is already bound")`。`ara_entry::bind`
/// 把它误当成"宿主重复绑定"，返回 null 扩展实例，插件从此永远不绑：既有 ARA 数据与
/// 新导入音频都不加载，只有删掉插件再添加才好（那会建一个全新的 `Processor`）。
///
/// 【为什么可以换掉整个对象】`bound` 是 `CompanionProcessorBinding` 内部的状态，而
/// `CompanionControllerBinding` 自己持有同一份 `Arc<BindingState>`。丢弃处理器侧的
/// binding 不会释放控制器侧的墓碑状态，也不会打断宿主已经拿到的扩展实例；换一个新
/// 对象就是把它重置回"从未绑定"。这比改外部 crate 小得多，也不依赖上游发布新版。
struct CompanionSlot {
    factory: CompanionFactory<'static>,
    supported_roles: CompanionRoles,
    binding: Mutex<Option<CompanionProcessorBinding<'static>>>,
}

impl CompanionSlot {
    fn new(
        factory: CompanionFactory<'static>,
        supported_roles: CompanionRoles,
    ) -> Result<Self, AraError> {
        let binding =
            CompanionProcessorBinding::new([factory.clone()], supported_roles)?;
        Ok(Self {
            factory,
            supported_roles,
            binding: Mutex::new(Some(binding)),
        })
    }

    /// 用同一批 factory / roles 换一个全新的绑定对象。
    ///
    /// 【为什么必须保持 factory 与 roles 不变】它们决定宿主看到的关联与角色集合；
    /// 重开工程不是"换产品形态"，只是"换文档"。
    fn reset(&self) -> Result<(), AraError> {
        let fresh = CompanionProcessorBinding::new([self.factory.clone()], self.supported_roles)?;
        *self.binding.lock_or_recover() = Some(fresh);
        Ok(())
    }

    /// 在**当前**绑定上查询工厂（`get_factory` 回调）。
    fn factory_raw(&self, class_name: &str) -> *const c_void {
        self.binding
            .lock_or_recover()
            .as_ref()
            .and_then(|binding| binding.factory_for_id(class_name))
            .map_or(std::ptr::null(), |factory| factory.as_raw().cast())
    }

    /// 初始绑定的生命周期探针（仅测试）。
    ///
    /// 【为什么探针要记在创建时】复位会换掉整个 `CompanionProcessorBinding`，
    /// 而复位后旧对象已经被丢弃 —— 想观察"宿主的实际销毁是否把控制器侧一并
    /// tombstone"就只能持有创建时那一份的弱引用。
    #[cfg(test)]
    fn lifetime_probe(&self) -> Option<ara2_bridge::companion::CompanionLifetimeProbe<'static>> {
        self.binding
            .lock_or_recover()
            .as_ref()
            .map(|binding| binding.lifetime_probe())
    }
}

struct Context {
    companion: CompanionSlot,
    class_name: String,
    owner: Arc<ExtensionOwner>,
}

/// 唯一 Rust owning native COM 引用；查询返回的引用另由宿主释放。
pub(crate) struct HostEntry {
    interface: NonNull<c_void>,
    /// 初始 companion 绑定的生命周期探针（仅测试使用）。
    ///
    /// 【为什么在创建时抓一份】Context 归原生 shim 所有，Rust 侧只留接口指针；
    /// 测试要观察"真实销毁是否把控制器侧一并 tombstone"，就必须在创建时留下弱引用。
    /// 复位会换掉绑定对象，而这一份探针仍然指向被宿主实际用过的那个状态。
    #[cfg(test)]
    probe: Option<ara2_bridge::companion::CompanionLifetimeProbe<'static>>,
}

impl HostEntry {
    /// 让原生绑定回调携带实际 controller，而不是只通知两个 role 参数。
    pub fn new(
        factory: CompanionFactory<'static>,
        supported_roles: CompanionRoles,
        class_name: &str,
        owner: Arc<ExtensionOwner>,
    ) -> Result<Self, AraError> {
        let companion = CompanionSlot::new(factory, supported_roles)?;
        if companion
            .binding
            .lock_or_recover()
            .as_ref()
            .is_none_or(|binding| binding.factory_for_id(class_name).is_none())
        {
            return Err(AraError::InvalidArgument("unknown factory association"));
        }
        #[cfg(test)]
        let probe = companion.lifetime_probe();
        let context = Box::into_raw(Box::new(Context {
            companion,
            class_name: class_name.into(),
            owner,
        }))
        .cast();
        let callbacks = Ara2Vst3PluginEntryCallbacks {
            context,
            get_factory: Some(get_factory),
            bind: Some(bind),
            drop: Some(destroy),
        };
        let mut interface = std::ptr::null_mut();
        // SAFETY: shim 成功时接管 context，最后 COM release 回调唯一释放。
        let result = unsafe { ara2_vst3_plugin_entry_create(&callbacks, &raw mut interface) };
        if result != ARA2_VST3_OK {
            // SAFETY: 创建失败时 shim 未接管 context。
            unsafe { drop(Box::from_raw(context.cast::<Context>())) };
            return Err(AraError::Peer("native entry creation failed"));
        }
        Ok(Self {
            interface: NonNull::new(interface).ok_or(AraError::Abi("null native entry"))?,
            #[cfg(test)]
            probe,
        })
    }
    /// 初始 companion 绑定的生命周期探针（测试）。
    #[cfg(test)]
    pub(crate) fn lifetime_probe(
        &self,
    ) -> Option<&ara2_bridge::companion::CompanionLifetimeProbe<'static>> {
        self.probe.as_ref()
    }
    pub fn as_raw(&self) -> *mut c_void {
        self.interface.as_ptr()
    }
    pub fn query_interface(&self, kind: Ara2Vst3InterfaceKind) -> Result<*mut c_void, AraError> {
        let mut output = std::ptr::null_mut();
        // SAFETY: 本 adapter 保留原生 owning 引用到同步查询结束。
        if unsafe { ara2_vst3_query_interface(self.as_raw(), kind, &raw mut output) }
            != ARA2_VST3_OK
            || output.is_null()
        {
            return Err(AraError::Peer("native entry query failed"));
        }
        Ok(output)
    }
}
impl Drop for HostEntry {
    fn drop(&mut self) {
        let mut remaining = 0;
        // SAFETY: 消耗本 adapter 唯一 owning COM 引用，其他宿主引用不受影响。
        unsafe { ara2_vst3_release(self.as_raw(), &raw mut remaining) };
    }
}

unsafe extern "C" fn get_factory(context: *mut c_void) -> *const c_void {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: 原生对象保留 Context 到最终 release。
        let context = unsafe { &*context.cast::<Context>() };
        context.companion.factory_raw(&context.class_name)
    }))
    .unwrap_or(std::ptr::null())
}
/// 把 ARA 角色位展开成名字，供绑定日志直接读数。
///
/// 【为什么按名字打印】`assigned=0x6` 要读者自己换算位掩码 —— 用户报障时不会算，也
/// 不该让他算。而"少了 PLAYBACK_RENDERER"正是"拿不到音频"最直接的线索，必须一眼可见。
/// 越界的位进不来：`CompanionRoles::from_bits` 对未知位返回 `None`，调用方已经提前返回。
fn role_names(roles: i32) -> String {
    let mut names = Vec::new();
    if roles & CompanionRoles::PLAYBACK_RENDERER.bits() != 0 {
        names.push("playback");
    }
    if roles & CompanionRoles::EDITOR_RENDERER.bits() != 0 {
        names.push("editor");
    }
    if roles & CompanionRoles::EDITOR_VIEW.bits() != 0 {
        names.push("view");
    }
    format!("[{}]", names.join(","))
}

unsafe extern "C" fn bind(
    context: *mut c_void,
    controller: *mut c_void,
    known: i32,
    assigned: i32,
) -> *const c_void {
    catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: Context 由 shim 保留，宿主按 ARA/VST3 契约提供完整实例记录。
        let context = unsafe { &*context.cast::<Context>() };
        if controller.is_null() {
            return std::ptr::null();
        }
        let Some(known_roles) = CompanionRoles::from_bits(known) else {
            return std::ptr::null();
        };
        let Some(assigned_roles) = CompanionRoles::from_bits(assigned) else {
            return std::ptr::null();
        };
        // ARAVST3.h 与 shim 的实参是 ARADocumentControllerRef；它不透明，禁止解引用。
        let Some(document) = DocumentSession::lookup(controller as usize) else {
            return std::ptr::null();
        };
        let Some(generation) = *document.generation.lock_or_recover() else {
            return std::ptr::null();
        };
        // SAFETY: lookup 已确认来自本产品的仍存活文档，guard 将在实际 document close 撤销。
        //
        // 【为什么要一个重试循环】companion 的 `bound` 是**一次性**的，而处理器实例
        // 跨工程存活。重开工程时宿主复用同一 FX 槽 ⇒ 同一个处理器 ⇒ 第二次 bind 必然
        // 被 "already bound" 拒绝。此时如果上一份文档已经关闭，正确动作是**换掉整个
        // companion 绑定对象**再试一次，而不是把 null 交给宿主。
        //
        // 【为什么最多两次】一次正常尝试 + 一次复位后的重试。复位后仍然失败说明不是
        // "遗留绑定"这一种成因，继续循环只会反复拆建、把现场搅得更难诊断。
        let mut attempt = 0;
        let companion = loop {
            let bound = {
                let slot = context.companion.binding.lock_or_recover();
                let Some(binding) = slot.as_ref() else {
                    return std::ptr::null();
                };
                // SAFETY: 调用方保证 controller 是本处理器工厂给出的活引用。
                unsafe { binding.bind(controller.cast(), known_roles, assigned_roles) }
            };
            match bound {
                Ok(companion) => break companion,
                Err(AraError::InvalidState("companion processor is already bound")) => {
                    // 第一优先：本实例仍有**活的**绑定（宿主重复 bindDocumentToExtension）。
                    // 幂等复用既有扩展实例：不新建 companion，也就不需要释放旧 companion。
                    if let Some(raw) = context.owner.existing_binding_raw() {
                        log::warn!(
                            "[ara] duplicate bindDocumentToExtension; reusing the existing \
                             extension instance (document={})",
                            document.id
                        );
                        return raw.cast();
                    }
                    // 走到这里说明 binding 已被 `document_closed()` 关闭。从未绑定过的
                    // 实例撞上 "already bound" 是真异常，不能复位（复位会把它变成"看起来
                    // 绑过"的状态，掩盖真相）。
                    if !context.owner.was_bound_before() {
                        log::error!(
                            "[ara] companion reports already bound but this instance never \
                             bound; refusing to reset a binding we do not own"
                        );
                        return std::ptr::null();
                    }
                    if attempt > 0 {
                        log::error!(
                            "[ara] companion still reports already bound after a reset; \
                             handing the host a null extension instance"
                        );
                        return std::ptr::null();
                    }
                    attempt += 1;
                    // 【为什么在这里复位是安全的】上一份文档已经关闭（
                    // `document_closed()` 已清掉 assignments/document，并撤销了 lease）。
                    // 旧的 `CompanionControllerBinding` 仍被 owner 的 `binding` 持有，
                    // 它自己握有 `Arc<BindingState>` —— 丢弃处理器侧对象不会让它悬空。
                    if let Err(error) = context.companion.reset() {
                        log::error!("[ara] companion reset failed: {error}");
                        return std::ptr::null();
                    }
                    context.owner.note_companion_reset();
                    log::info!(
                        "[ara] reset the one-shot companion binding so document {} can bind \
                         after the previous project closed",
                        document.id
                    );
                }
                Err(error) => {
                    log::warn!("[ara] companion bind failed: {error}");
                    return std::ptr::null();
                }
            }
        };
        // 宿主对本实例的角色决定：`assigned` 少掉 playback 是"拿不到音频"的直接线索，
        // 因此按名字打印，并显式标注缺失（见方案 Task 3.3）。
        let playback = if assigned & CompanionRoles::PLAYBACK_RENDERER.bits() != 0 {
            "assigned"
        } else {
            "NOT assigned"
        };
        log::info!(
            "[ara] bind document={} roles known={} assigned={} (playback {playback})",
            document.id,
            role_names(known),
            role_names(assigned)
        );
        let document_id = document.id;
        match context.owner.bind_to_document(
            document,
            generation,
            ExtensionRoles::from_bits_truncate(known),
            ExtensionRoles::from_bits_truncate(assigned),
            Some(companion),
        ) {
            Ok(raw) => {
                // 【为什么把接口清单记下来】ARA 契约要求"宿主**分配**的角色必须给出接口"
                // （`ARAPlugInExtensionInstance` 的文档原文）。少一个接口，宿主拿到的就是
                // NULL —— 而它会在真正用到的那一刻崩在自己的代码里。Cubase 15 实测：
                // view 角色漏了接口，崩点就在打开编辑器时 `araservice.dll` 内读空指针，
                // 现场完全看不出与插件有关。这一行把"我们到底给了宿主什么"变成可读事实。
                //
                // SAFETY: 记录由 owner 持有到实例销毁，本调用期间有效；紧凑结构整份拷出
                // 再读字段（对 packed 字段取引用是 UB）。
                let instance = unsafe { raw.read_unaligned() };
                log::info!(
                    "[ara] extension instance: playback={} editor={} view={}",
                    !instance.playbackRendererInterface.is_null(),
                    !instance.editorRendererInterface.is_null(),
                    !instance.editorViewInterface.is_null()
                );
                raw.cast()
            }
            Err(error) => {
                // 静默返回 null 等于把"宿主崩在自己代码里"留给下一个人查。
                log::error!(
                    "[ara] binding this instance to document {document_id} failed: {error}; \
                     handing the host a null extension instance"
                );
                std::ptr::null()
            }
        }
    }))
    .unwrap_or(std::ptr::null())
}
unsafe extern "C" fn destroy(context: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: native shim 只在最终 owning COM release 时调用一次。
        unsafe { drop(Box::from_raw(context.cast::<Context>())) };
    }));
}

#[cfg(test)]
mod role_name_tests {
    use super::*;

    /// 角色日志的**唯一职责**是让"少了哪一个角色"一眼可见 —— 用户报障时不会去换算
    /// 位掩码，也不该让他算。因此顺序与缺失都要钉住。
    #[test]
    fn roles_are_named_so_the_missing_one_is_visible() {
        assert_eq!(role_names(0), "[]");
        assert_eq!(
            role_names(CompanionRoles::PLAYBACK_RENDERER.bits()),
            "[playback]"
        );
        // 实测里最常见的那一行：三种角色全部声明（known）。
        assert_eq!(
            role_names(CompanionRoles::all().bits()),
            "[playback,editor,view]"
        );
        // 而宿主只分配 editor + view：playback 的缺失必须直接读出来。
        assert_eq!(
            role_names(CompanionRoles::EDITOR_RENDERER.bits() | CompanionRoles::EDITOR_VIEW.bits()),
            "[editor,view]"
        );
    }
}
