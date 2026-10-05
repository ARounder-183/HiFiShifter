//! 进程级 ARA 运行时：ARA 工厂 + companion 关联。
//!
//! 【为什么是进程级单例】ARA 的工厂在整个进程生命周期内必须稳定存在（宿主会一直
//! 拿它查接口），而 VST3 模块可能被多次 `initialize`/`terminate`。把工厂泄漏到进程
//! 结束是最简单且与 ARA 契约一致的做法的 —— 探针阶段已经用同样形状跑通。
//!
//! 与探针版的差别：模型换成产品模型（累积 `AraDocument` 而不是计数），日志走 `log`。

use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::core::AraError;
use ara2_bridge::core::PlaybackTransformationFlags;
use ara2_bridge::plugin::{
    Factory, FactoryBuilder, FactoryCapabilities, PluginBuilder, PluginEntry,
};
use std::sync::OnceLock;

/// 把 `Factory` 以只读方式跨线程共享的包装。
///
/// `Factory` 内部是 ARA 回调的常驻 backing，ARA 工厂契约本身就要求它在进程内稳定可读；
/// 这里只读取它（取 generation / 包 companion 关联），不修改，因此这条 `Sync` 是安全的。
pub struct StaticFactory(&'static Factory);

// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Send for StaticFactory {}
// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Sync for StaticFactory {}

impl StaticFactory {
    /// 借出该工厂的初始化入口（用于读取协商到的 ARA 版本）。
    pub fn entry(&self) -> &PluginEntry {
        self.0.entry()
    }
}

/// 进程级运行时：泄漏到进程结束的 ARA 工厂 + companion 关联。
pub struct Runtime {
    /// ARA 工厂。
    pub factory: StaticFactory,
    /// companion 层看到的关联（VST3 主工厂与处理器都用它）。
    pub companion: CompanionFactory<'static>,
}

impl Runtime {
    /// 建立工厂与 companion 关联。
    fn init() -> Result<Runtime, AraError> {
        let factory = FactoryBuilder::new(crate::FACTORY_ID, crate::ARCHIVE_ID)
            .display(
                crate::CLASS_NAME,
                "HiFiShifter",
                "https://example.invalid",
                crate::VERSION,
            )
            // 仅线性保调已实现；tempo反映与content fade须通过完整音频门后再协商。
            .capabilities(
                FactoryCapabilities::default().with_playback_transformations(
                    PlaybackTransformationFlags::TIMESTRETCH,
                ),
            )
            // 每个文档控制器拿到**自己的一份**模型：一份模型对应一份 ARA 文档。
            // 共享一份会让两份文档的累积互相污染（设计 §4.1：v1 是"一实例一编辑轨"）。
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                let playback=session.clone();
                PluginBuilder::new(model).controller_identity(move |key| session.register(key))
                    .host_playback(move |handle| {crate::log_line(&format!("ARA host playback control available={}",handle.is_some()));*playback.playback.lock().unwrap()=handle;}).build()
            })
            .build()?;
        let factory: &'static Factory = Box::leak(Box::new(factory));
        // SAFETY: 工厂被泄漏到进程结束，`as_raw` 指向的 ARAFactory 与工厂同寿。
        let companion = unsafe { CompanionFactory::from_raw(crate::CLASS_NAME, &*factory.as_raw())? };
        crate::log_line("ARA factory built; companion association ready");
        Ok(Runtime {
            factory: StaticFactory(factory),
            companion,
        })
    }
}

/// 进程级唯一的运行时；首次访问时初始化。
static RUNTIME: OnceLock<Option<Runtime>> = OnceLock::new();

/// 取运行时；初始化失败时返回 `None`（原因写进日志）。
///
/// 【为什么失败也缓存】`OnceLock::get_or_init` 只跑一次。宿主可能因为插件不可用而反复
/// 查询，每次都重试构建工厂既没意义也会刷日志。
pub fn runtime() -> Option<&'static Runtime> {
    RUNTIME
        .get_or_init(|| match Runtime::init() {
            Ok(runtime) => Some(runtime),
            Err(error) => {
                log::error!("[ara] runtime init failed: {error:?}");
                None
            }
        })
        .as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;
    /// 宿主实际读取的native工厂不能把尚未实现的tempo/content fade协商给插件。
    #[test]
    fn task38a_native_factory_advertises_only_implemented_linear_timestretch() {
        let runtime=Runtime::init().unwrap();
        let raw=unsafe {&*runtime.factory.0.as_raw()};
        assert_eq!(raw.supportedPlaybackTransformationFlags as u32,PlaybackTransformationFlags::TIMESTRETCH.bits());
    }
}
