//! 控制器身份感知的 ARA entry；复用 audited C++ shim 和 companion 绑定，不手写 ARA COM vtable。

use crate::render::{document::DocumentSession, extension::ExtensionOwner};
use ara2_bridge::companion::vst3::ffi::*;
use ara2_bridge::companion::{CompanionProcessorBinding, CompanionRoles};
use ara2_bridge::core::AraError;
use ara2_bridge::plugin::ExtensionRoles;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::Arc;

struct Context {
    processor: CompanionProcessorBinding<'static>,
    class_name: String,
    owner: Arc<ExtensionOwner>,
}

/// 唯一 Rust owning native COM 引用；查询返回的引用另由宿主释放。
pub(crate) struct HostEntry {
    interface: NonNull<c_void>,
}

impl HostEntry {
    /// 让原生绑定回调携带实际 controller，而不是只通知两个 role 参数。
    pub fn new(
        processor: CompanionProcessorBinding<'static>,
        class_name: &str,
        owner: Arc<ExtensionOwner>,
    ) -> Result<Self, AraError> {
        if processor.factory_for_id(class_name).is_none() {
            return Err(AraError::InvalidArgument("unknown factory association"));
        }
        let context = Box::into_raw(Box::new(Context {
            processor,
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
        })
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
        context
            .processor
            .factory_for_id(&context.class_name)
            .map_or(std::ptr::null(), |factory| factory.as_raw().cast())
    }))
    .unwrap_or(std::ptr::null())
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
        let Some(generation) = *document.generation.lock().unwrap() else {
            return std::ptr::null();
        };
        // SAFETY: lookup 已确认来自本产品的仍存活文档，guard 将在实际 document close 撤销。
        let Ok(companion) = (unsafe {
            context
                .processor
                .bind(controller.cast(), known_roles, assigned_roles)
        }) else {
            return std::ptr::null();
        };
        log::info!(
            "[ara] bind document={} known={known:#x} assigned={assigned:#x}",
            document.id
        );
        context
            .owner
            .bind_to_document(
                document,
                generation,
                ExtensionRoles::from_bits_truncate(known),
                ExtensionRoles::from_bits_truncate(assigned),
                Some(companion),
            )
            .map_or(std::ptr::null(), |raw| raw.cast())
    }))
    .unwrap_or(std::ptr::null())
}
unsafe extern "C" fn destroy(context: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: native shim 只在最终 owning COM release 时调用一次。
        unsafe { drop(Box::from_raw(context.cast::<Context>())) };
    }));
}
