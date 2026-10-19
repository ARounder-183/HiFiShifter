//! Per-factory ARA initialization state.

use crate::factory::ControllerCreator;
use ara2_bridge_core::{ApiGeneration, AraError};
use ara2_bridge_sys::{
    ARAAssertFunction, ARADocumentControllerHostInstance, ARADocumentControllerInstance,
    ARADocumentProperties, ARAFactory,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

#[derive(Clone, Copy, Debug)]
struct Selection {
    generation: ApiGeneration,
    assert_address: usize,
}

#[derive(Clone, Copy, Debug)]
struct SharedAssertion {
    address: usize,
    users: usize,
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn assertions() -> &'static Mutex<BTreeMap<ApiGeneration, SharedAssertion>> {
    static ASSERTIONS: OnceLock<Mutex<BTreeMap<ApiGeneration, SharedAssertion>>> = OnceLock::new();
    ASSERTIONS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn acquire_assertion(generation: ApiGeneration, address: usize) -> Result<(), AraError> {
    let mut assertions = lock(assertions());
    match assertions.get_mut(&generation) {
        Some(shared) if shared.address != address => Err(AraError::InvalidState(
            "factories using one API generation must share the assert-function address",
        )),
        Some(shared) => {
            shared.users = shared.users.checked_add(1).ok_or(AraError::InvalidState(
                "assert-function user count overflow",
            ))?;
            Ok(())
        }
        None => {
            assertions.insert(generation, SharedAssertion { address, users: 1 });
            Ok(())
        }
    }
}

fn release_assertion(selection: Selection) {
    let mut assertions = lock(assertions());
    let remove = match assertions.get_mut(&selection.generation) {
        Some(shared) if shared.address == selection.assert_address && shared.users > 1 => {
            shared.users -= 1;
            false
        }
        Some(shared) if shared.address == selection.assert_address => true,
        _ => false,
    };
    if remove {
        assertions.remove(&selection.generation);
    }
}

/// Shared state reached by both the safe entry API and its C callback trampolines.
pub(crate) struct CallbackState {
    lowest: ApiGeneration,
    highest: ApiGeneration,
    selection: Mutex<Option<Selection>>,
    /// 最近一次成功协商的代数；`uninitialize` **不清**。
    ///
    /// 【为什么需要它】ARA SDK 允许 initialize/uninitialize 在模块不卸载的情况下
    /// 多次配对调用；但真实宿主（尤其工程加载/关闭路径）可能只做一半：工程关闭时
    /// `uninitializeARA`，重开时不再 `initializeARAWithConfiguration`。此时
    /// `createDocumentControllerWithDocument` 会因为 `selection == None` 返回 null，
    /// 宿主随后拿不到任何文档控制器 —— 界面"等待宿主音频"。
    /// 保留最近一次代数让控制器创建在这种情况下仍能继续；真正的模块卸载会销毁
    /// 整个 `CallbackState`，不存在跨模块使用陈旧代数的风险。
    last_generation: Mutex<Option<ApiGeneration>>,
    create_controller: Option<Arc<ControllerCreator>>,
    factory: AtomicPtr<ARAFactory>,
}

impl CallbackState {
    pub(crate) fn new(
        lowest: ApiGeneration,
        highest: ApiGeneration,
        create_controller: Option<Arc<ControllerCreator>>,
    ) -> Self {
        Self {
            lowest,
            highest,
            selection: Mutex::new(None),
            last_generation: Mutex::new(None),
            create_controller,
            factory: AtomicPtr::new(std::ptr::null_mut()),
        }
    }

    pub(crate) fn attach_factory(&self, factory: *const ARAFactory) {
        self.factory.store(factory.cast_mut(), Ordering::Release);
    }

    pub(crate) fn create_document_controller(
        &self,
        host: *const ARADocumentControllerHostInstance,
        properties: *const ARADocumentProperties,
    ) -> Result<*const ARADocumentControllerInstance, AraError> {
        let generation = self
            .generation()
            .ok_or(AraError::InvalidState("factory is not initialized"))?;
        let create = self
            .create_controller
            .as_ref()
            .ok_or(AraError::Unsupported(
                "factory has no document-controller constructor",
            ))?;
        let factory = self.factory.load(Ordering::Acquire).cast_const();
        if factory.is_null() {
            return Err(AraError::InvalidState(
                "factory ABI storage is not attached",
            ));
        }
        create(generation, factory, host, properties)
    }

    pub(crate) fn initialize(
        &self,
        generation: ApiGeneration,
        assert_address: *mut ARAAssertFunction,
    ) -> Result<(), AraError> {
        // 【本地补丁 8】宿主没提供 assert 指针时**容忍**，而不是拒绝。
        //
        // 规范要求宿主总是提供这个指针（"It must be always provided by the host, but can
        // point to NULL to suppress debugging in release versions"），但拒绝的代价极不对称：
        // 拒绝是静默的（C ABI 返回 void），随后 `createDocumentControllerWithDocument` 会
        // 把 **null** 交给宿主，而宿主不检查它 —— 它拿着 null 写字段就崩在自己的代码里
        // （Cubase 15 实测：`araservice.dll` 空指针 + 0xB0 写入，0xc0000005/[1, 0xb0]）。
        //
        // 而这个地址在本库里**只被记录与比较，从不被调用**（见 `acquire_assertion` /
        // `release_assertion`），所以取 0 的唯一后果是"没有装调试断言钩子"。
        // 把宿主的轻微不规范变成崩溃是不可接受的，所以这里记一条警告后继续。
        if assert_address.is_null() {
            log::warn!(
                "[ara] the host provided no assert-function address; continuing without the \
                 debug assertion hook instead of rejecting the configuration"
            );
        }
        if generation < self.lowest || generation > self.highest {
            return Err(AraError::InvalidArgument(
                "API generation is outside the factory range",
            ));
        }
        if !generation.supported_on_target() {
            return Err(AraError::Unsupported(
                "API generation is unavailable on this target",
            ));
        }

        let mut selection = lock(&self.selection);
        if let Some(existing) = selection.as_ref() {
            // 【为什么同配置重复初始化是幂等的】宿主可能在每次工程加载时重新
            // initialize，却不保证先 uninitialize。规格允许重复调用；把它当错误
            // 会让后续 `createDocumentController` 仍然能用（selection 还在），
            // 但日志会持续报"被拒"，掩盖真正的问题。同代数/同 assert 地址直接
            // 返回 Ok；不同配置才拒绝（那才是真正的协议冲突）。
            if existing.generation == generation
                && existing.assert_address == assert_address as usize
            {
                return Ok(());
            }
            return Err(AraError::InvalidState("factory is already initialized"));
        }
        let selected = Selection {
            generation,
            assert_address: assert_address as usize,
        };
        acquire_assertion(generation, selected.assert_address)?;
        *selection = Some(selected);
        *lock(&self.last_generation) = Some(generation);
        Ok(())
    }

    pub(crate) fn uninitialize(&self) -> Result<(), AraError> {
        let mut selection = lock(&self.selection);
        let selected = selection
            .take()
            .ok_or(AraError::InvalidState("factory is not initialized"))?;
        release_assertion(selected);
        // 刻意保留 `last_generation`：见字段说明。模块卸载时整个 state 析构，
        // 不存在跨模块误用。
        Ok(())
    }

    pub(crate) fn generation(&self) -> Option<ApiGeneration> {
        if let Some(selection) = lock(&self.selection).as_ref() {
            return Some(selection.generation);
        }
        let fallback = *lock(&self.last_generation);
        if fallback.is_some() {
            use std::sync::atomic::AtomicBool;
            static REPORTED: AtomicBool = AtomicBool::new(false);
            if !REPORTED.swap(true, Ordering::AcqRel) {
                log::warn!(
                    "[ara] creating a document controller after the host uninitialized the \
                     factory; using the last negotiated API generation instead of handing the \
                     host a null controller"
                );
            }
        }
        fallback
    }
}

impl Drop for CallbackState {
    fn drop(&mut self) {
        if let Some(selection) = *self
            .selection
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
        {
            release_assertion(selection);
        }
    }
}

/// Independent initialization handle for one exported ARA factory.
pub struct PluginEntry {
    pub(crate) state: std::sync::Arc<CallbackState>,
}

impl PluginEntry {
    pub(crate) fn new(state: std::sync::Arc<CallbackState>) -> Self {
        Self { state }
    }

    /// Initializes this factory for the selected released API generation.
    pub fn initialize(
        &self,
        generation: ApiGeneration,
        assert_address: *mut ARAAssertFunction,
    ) -> Result<(), AraError> {
        self.state.initialize(generation, assert_address)
    }

    /// Balances a successful initialization of this factory.
    pub fn uninitialize(&self) -> Result<(), AraError> {
        self.state.uninitialize()
    }

    /// Returns the generation currently selected for this factory.
    pub fn generation(&self) -> Option<ApiGeneration> {
        self.state.generation()
    }
}
