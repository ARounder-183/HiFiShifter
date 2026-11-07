//! 格式无关的 ARA 实例绑定：VST3 与 CLAP 共用**同一份**决策逻辑。
//!
//! 【为什么在这里】VST3 走 `ARAVST3.h` 的插件入口点、CLAP 走 `ARACLAP.h` 的
//! `bindToDocumentController`，两者的 **ABI 外壳**不同，但绑定那一刻要做的判断完全相同：
//! 角色位是否合法、控制器是不是本产品仍活着的文档、代数是否协商过、companion 的一次性
//! 绑定要不要复位、重复绑定同一份文档要不要幂等复用、失败时该留哪条痕迹。
//!
//! 这些判断此前只写在 `ara_entry.rs` 里。CLAP 侧要么复制一份（两份必然漂移），要么
//! 绕开（行为分叉）。现在只有这一份：两个外壳各自把格式回调转成对 [`AraBinder::bind`]
//! 的一次调用，其余一律不重复实现。

use crate::render::{document::DocumentSession, extension::ExtensionOwner};
use crate::sync::MutexExt;
use ara2_bridge::companion::{
    CompanionControllerBinding, CompanionFactory, CompanionProcessorBinding, CompanionRoles,
};
use ara2_bridge::core::AraError;
use ara2_bridge::plugin::ExtensionRoles;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

/// 处理器级 companion 绑定的**可复位**容器。
///
/// 【为什么需要容器】`CompanionProcessorBinding::bind` 是一次性的：companion 只把
/// `lifecycle.bound` 写成 `true`，`CompanionControllerBinding` 的 `Drop` 也不复位它。
/// 而处理器实例会跨工程存活 —— 宿主重开工程时复用同一个实例槽位，于是第二次
/// `bind` 必然返回 `Err("companion processor is already bound")`。把它误当成"宿主重复
/// 绑定"并返回 null 扩展实例，插件从此永远不绑：既有 ARA 数据与新导入音频都不加载，
/// 只有删掉插件再添加才好（那会建一个全新的处理器）。
///
/// 【为什么绑定要**惰性**创建】companion 的 `bind` 要求"当前线程 == 创建绑定的线程"。
/// 处理器可能在工程加载线程上创建、在 UI/模型线程上被 bind；在创建实例时立刻创建会把
/// `model_thread` 钉在错误的线程上。第一次 `bind` 时再创建，捕获的正是宿主实际使用的
/// 模型线程；每次复位后在新的 bind 线程上重建。
pub struct CompanionSlot {
    factory: CompanionFactory<'static>,
    supported_roles: CompanionRoles,
    binding: Mutex<Option<CompanionProcessorBinding<'static>>>,
}

impl CompanionSlot {
    /// 以一组稳定的工厂关联与支持的 ARA 角色建立容器。
    pub fn new(
        factory: CompanionFactory<'static>,
        supported_roles: CompanionRoles,
    ) -> Result<Self, AraError> {
        Ok(Self {
            factory,
            supported_roles,
            binding: Mutex::new(None),
        })
    }

    /// 在当前线程上（惰性）创建绑定并执行一次 bind。
    fn bind(
        &self,
        controller: *mut c_void,
        known_roles: CompanionRoles,
        assigned_roles: CompanionRoles,
    ) -> Result<CompanionControllerBinding<'static>, AraError> {
        let mut slot = self.binding.lock_or_recover();
        if slot.is_none() {
            *slot = Some(CompanionProcessorBinding::new(
                [self.factory.clone()],
                self.supported_roles,
            )?);
        }
        let binding = slot.as_ref().expect("companion binding was just created");
        // SAFETY: 调用方保证 controller 是本处理器工厂给出的活引用。
        unsafe { binding.bind(controller.cast(), known_roles, assigned_roles) }
    }

    /// 丢弃当前绑定对象，下一次 `bind` 会在新的线程上重建。
    ///
    /// 【为什么必须保持 factory 与 roles 不变】它们决定宿主看到的关联与角色集合；
    /// 重开工程不是"换产品形态"，只是"换文档"。
    fn reset(&self) {
        *self.binding.lock_or_recover() = None;
    }

    /// `get_factory` 回调：工厂指针本身就是进程级稳定的，不依赖当前绑定对象。
    pub fn factory_raw(&self) -> *const c_void {
        self.factory.as_raw().cast()
    }

    /// 当前绑定的生命周期探针（仅测试）。
    #[cfg(test)]
    fn lifetime_probe(&self) -> Option<ara2_bridge::companion::CompanionLifetimeProbe<'static>> {
        let mut slot = self.binding.lock_or_recover();
        if slot.is_none() {
            *slot =
                CompanionProcessorBinding::new([self.factory.clone()], self.supported_roles).ok();
        }
        slot.as_ref().map(|binding| binding.lifetime_probe())
    }
}

/// 一个插件实例的 ARA 绑定器：companion 槽位 + 扩展所有权。
///
/// 【为什么不带格式】两个外壳都持有它，回调签名各写各的（VST3 的 COM shim / CLAP 的
/// `ClapAraPluginExtension`），但**绑定到哪份文档、给不给、给完记什么**只有这里一套。
pub struct AraBinder {
    companion: CompanionSlot,
    owner: Arc<ExtensionOwner>,
}

impl AraBinder {
    /// 建立绑定器。
    ///
    /// `class_name` 是宿主用来把"主工厂"与"音频处理器"配成一对的那个名字：VST3 是
    /// 处理器类的 `PClassInfo.name`，CLAP 是 descriptor 的插件 ID。名字不匹配说明宿主
    /// 拿到的关联根本不是本产品的 —— 那必须当场拒绝，而不是绑上一份说不清来历的文档。
    pub fn new(
        factory: CompanionFactory<'static>,
        supported_roles: CompanionRoles,
        class_name: &str,
        owner: Arc<ExtensionOwner>,
    ) -> Result<Self, AraError> {
        let companion = CompanionSlot::new(factory, supported_roles)?;
        if companion.factory.id() != class_name {
            return Err(AraError::InvalidArgument("unknown factory association"));
        }
        Ok(Self { companion, owner })
    }

    /// `get_factory` 的实现：本实例对应的 ARA 工厂指针。
    pub fn factory_raw(&self) -> *const c_void {
        self.companion.factory_raw()
    }

    /// 本实例的扩展所有权（CLAP 侧需要它取文档/形态读数）。
    pub fn owner(&self) -> &Arc<ExtensionOwner> {
        &self.owner
    }

    /// 绑定到一份文档控制器；返回宿主将持有的 `ARAPlugInExtensionInstance`。
    ///
    /// 失败一律返回 null 并留痕：静默返回 null 等于把"宿主崩在自己代码里"留给下一个人查。
    pub fn bind(&self, controller: *mut c_void, known: i32, assigned: i32) -> *const c_void {
        if controller.is_null() {
            log::error!("[ara] bindDocumentToExtension called with a null controller reference");
            return std::ptr::null();
        }
        let Some(known_roles) = CompanionRoles::from_bits(known) else {
            log::error!("[ara] bind rejected: unknown known-roles bits {known:#x}");
            return std::ptr::null();
        };
        let Some(assigned_roles) = CompanionRoles::from_bits(assigned) else {
            log::error!("[ara] bind rejected: unknown assigned-roles bits {assigned:#x}");
            return std::ptr::null();
        };
        // ARAVST3.h 与 ARACLAP.h 的实参都是 ARADocumentControllerRef；它不透明，禁止解引用。
        let Some(document) = DocumentSession::lookup(controller as usize) else {
            // 【为什么要把 live keys 打出来】"控制器不在表里"可能是 create 失败、
            // initialize 状态丢失，或宿主传了另一份 ref。没有这行日志，现场只能看到
            // 一个 null 扩展实例，无法区分三种完全不同的处置。
            log::error!(
                "[ara] bind target controller key={:#x} is not a live document; live keys={:?}; \
                 bound_once={} has_binding={} closed={}",
                controller as usize,
                DocumentSession::live_controller_keys(),
                self.owner.was_bound_before(),
                self.owner.has_binding(),
                self.owner.is_closed()
            );
            return std::ptr::null();
        };
        let Some(generation) = *document.generation.lock_or_recover() else {
            log::error!(
                "[ara] bind rejected: document {} has no negotiated ARA generation \
                 (createDocumentController did not complete)",
                document.id
            );
            return std::ptr::null();
        };
        log::info!(
            "[ara] bind request: document={} controller={:#x} owner_bound_document={:?} \
             closed={} has_binding={}",
            document.id,
            controller as usize,
            self.owner.bound_document_id(),
            self.owner.is_closed(),
            self.owner.has_binding()
        );
        // SAFETY: lookup 已确认来自本产品的仍存活文档，guard 将在实际 document close 撤销。
        //
        // 【为什么要一个重试循环】companion 的 `bound` 是**一次性**的，而处理器实例
        // 跨工程存活。重开工程时宿主复用同一实例槽位 ⇒ 同一个处理器 ⇒ 第二次 bind 必然
        // 被 "already bound" 拒绝。此时正确动作是**换掉整个 companion 绑定对象**再试一次，
        // 而不是把 null 交给宿主。
        //
        // 【为什么最多两次】一次正常尝试 + 一次复位后的重试。复位后仍然失败说明不是
        // "遗留绑定"这一种成因，继续循环只会反复拆建、把现场搅得更难诊断。
        let mut attempt = 0;
        let companion = loop {
            match self.companion.bind(controller, known_roles, assigned_roles) {
                Ok(companion) => break companion,
                Err(AraError::InvalidState("companion processor is already bound")) => {
                    // 第一优先：本实例仍**活在同一份文档**上（宿主重复 bindDocumentToExtension）。
                    // 只有文档身份也相同才允许幂等复用；旧文档还活着时把它的指针交给
                    // 新文档，会让新文档永远收不到 renderer 分配。
                    if let Some(raw) = self.owner.existing_binding_raw_for(document.id) {
                        log::warn!(
                            "[ara] duplicate bindDocumentToExtension for the same live document {}; \
                             reusing the existing extension instance",
                            document.id
                        );
                        return raw.cast();
                    }
                    // 走到这里说明 binding 要么属于另一份文档，要么已被
                    // `document_closed()` 关闭。从未绑定过的实例撞上 "already bound"
                    // 是真异常，不能复位（复位会把它变成"看起来绑过"的状态，掩盖真相）。
                    if !self.owner.was_bound_before() && !self.owner.has_binding() {
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
                    // 【为什么在这里复位是安全的】旧的 `CompanionControllerBinding` 仍被
                    // owner 的 `binding` / retired 列表持有，它自己握有 `Arc<BindingState>`
                    // —— 丢弃处理器侧对象不会让它悬空。
                    self.companion.reset();
                    self.owner.note_companion_reset();
                    log::info!(
                        "[ara] reset the one-shot companion binding so document {} can bind \
                         (previous document {:?}, closed={})",
                        document.id,
                        self.owner.bound_document_id(),
                        self.owner.is_closed()
                    );
                }
                Err(error) => {
                    log::error!("[ara] companion bind failed: {error}");
                    return std::ptr::null();
                }
            }
        };
        // 宿主对本实例的角色决定：`assigned` 少掉 playback 是"拿不到音频"的直接线索，
        // 因此按名字打印，并显式标注缺失。
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
        match self.owner.bind_to_document(
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
                log::error!(
                    "[ara] binding this instance to document {document_id} failed: {error}; \
                     handing the host a null extension instance"
                );
                std::ptr::null()
            }
        }
    }

    /// 初始 companion 绑定的生命周期探针（测试）。
    #[cfg(test)]
    pub fn lifetime_probe(&self) -> Option<ara2_bridge::companion::CompanionLifetimeProbe<'static>> {
        self.companion.lifetime_probe()
    }
}

/// 把 ARA 角色位展开成名字，供绑定日志直接读数。
///
/// 【为什么按名字打印】`assigned=0x6` 要读者自己换算位掩码 —— 用户报障时不会算，也
/// 不该让他算。而"少了 PLAYBACK_RENDERER"正是"拿不到音频"最直接的线索，必须一眼可见。
/// 越界的位进不来：`CompanionRoles::from_bits` 对未知位返回 `None`，调用方已经提前返回。
pub fn role_names(roles: i32) -> String {
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

#[cfg(test)]
mod tests {
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
        assert_eq!(
            role_names(CompanionRoles::all().bits()),
            "[playback,editor,view]"
        );
        assert_eq!(
            role_names(CompanionRoles::EDITOR_RENDERER.bits() | CompanionRoles::EDITOR_VIEW.bits()),
            "[editor,view]"
        );
    }
}
