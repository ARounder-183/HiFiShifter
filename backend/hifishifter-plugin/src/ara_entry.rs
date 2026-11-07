//! 控制器身份感知的 ARA entry；复用 audited C++ shim 和 companion 绑定，不手写 ARA COM vtable。
//!
//! 【绑定逻辑不在这里】角色校验、文档查找、代数协商检查、companion 复位与幂等复用全在
//! [`crate::ara::binder::AraBinder`] —— CLAP 外壳调的是**同一个方法**。本文件只剩 VST3 的
//! COM 外壳：把 shim 的三个回调转成对 binder 的调用，并持有唯一的 owning COM 引用。

use crate::ara::binder::AraBinder;
use crate::render::extension::ExtensionOwner;
use ara2_bridge::companion::vst3::ffi::*;
use ara2_bridge::companion::{CompanionFactory, CompanionRoles};
use ara2_bridge::core::AraError;
use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::Arc;

struct Context {
    binder: AraBinder,
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
        let binder = AraBinder::new(factory, supported_roles, class_name, owner)?;
        #[cfg(test)]
        let probe = binder.lifetime_probe();
        let context = Box::into_raw(Box::new(Context { binder })).cast();
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
        context.binder.factory_raw()
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
        context.binder.bind(controller, known, assigned)
    }))
    .unwrap_or(std::ptr::null())
}

unsafe extern "C" fn destroy(context: *mut c_void) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: native shim 只在最终 owning COM release 时调用一次。
        unsafe { drop(Box::from_raw(context.cast::<Context>())) };
    }));
}
