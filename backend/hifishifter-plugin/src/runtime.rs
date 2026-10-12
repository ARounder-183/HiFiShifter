//! 进程级 ARA 运行时：ARA 工厂 + companion 关联。
//!
//! 【为什么是进程级单例】ARA 的工厂在整个进程生命周期内必须稳定存在（宿主会一直
//! 拿它查接口），而 VST3 模块可能被多次 `initialize`/`terminate`。把工厂泄漏到进程
//! 结束是最简单且与 ARA 契约一致的做法的 —— 探针阶段已经用同样形状跑通。
//!
//! 与探针版的差别：模型换成产品模型（累积 `AraDocument` 而不是计数），日志走 `log`。

use crate::sync::MutexExt;
use ara2_bridge::companion::CompanionFactory;
use ara2_bridge::core::ApiGeneration;
use ara2_bridge::core::AraError;
use ara2_bridge::core::PlaybackTransformationFlags;
use ara2_bridge::plugin::{Factory, FactoryBuilder, FactoryCapabilities, PluginBuilder};
use std::sync::OnceLock;

/// 把 `Factory` 以只读方式跨线程共享的包装。
///
/// `Factory` 内部是 ARA 回调的常驻 backing，ARA 工厂契约本身就要求它在进程内稳定可读；
/// 这里只读取它（取 generation / 包 companion 关联），不修改，因此这条 `Sync` 是安全的。
pub struct StaticFactory(
    // ARA 工厂句柄；只被 cfg(test) 的 ABI 断言读取（`factory_raw`），产品构建里
    // 字段本身不直接引用。此前还有一个 `entry()` 访问器，无任何调用方，已删除。
    #[allow(dead_code)] &'static Factory,
);

impl StaticFactory {
    /// 只读的 ARA 工厂记录指针。
    ///
    /// 【为什么需要它】ARA 契约要求处理器类的 `PClassInfo.name` 与
    /// `ARAFactory.plugInName` 一致，宿主据此把 ARA 主工厂与音频处理器配成一对
    /// （`ARAVST3.h` 的 `IMainFactory` 说明）。`vst3::factory_declaration_tests`
    /// 要读回 `plugInName` 才能把这条不变量钉住 —— 漂移的症状是"插件能加载但从不
    /// 进入 ARA"，没有断言就无从发现。
    ///
    /// 【为什么只读】工厂由 `OnceLock` 持有并泄漏到进程结束，这里只暴露核对元数据
    /// 所需的只读指针；能改状态的访问器不该存在。
    ///
    /// 【为什么 `cfg(test)`】唯一的调用方是 `#[cfg(test)]` 的
    /// `vst3::factory_declaration_tests`；不带这条属性时它在产品构建里"从未使用"，
    /// `clippy --all-targets -D warnings` 直接红。与 `ee9d95cd` 对同类只读访问器的
    /// 处理一致。
    #[cfg(test)]
    pub(crate) fn factory_raw(&self) -> *const ara2_bridge::sys::ARAFactory {
        self.0.as_raw()
    }
}

// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Send for StaticFactory {}
// SAFETY: 见 StaticFactory 文档注释。
unsafe impl Sync for StaticFactory {}

/// 进程级运行时：泄漏到进程结束的 ARA 工厂 + companion 关联。
pub struct Runtime {
    /// ARA 工厂。
    // 只被 cfg(test) 的 ABI 断言读取；产品构建里由 `OnceLock` 持有并泄漏到进程结束。
    #[allow(dead_code)]
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
                "https://github.com/ARounder-183/HiFiShifter",
                crate::VERSION,
            )
            // 【为什么最低代数必须是 2.0 **Draft**，不能是 2.0 Final】ARA SDK 的
            // `ARAAPIGeneration` 注释把**每一代由哪些宿主使用**写得很清楚：
            //
            //   kARAAPIGeneration_2_0_Draft = 3,
            //     "supported by Studio One, Logic Pro, Cubase/Nuendo, Cakewalk,
            //      REAPER, Melodyne, ReVoice Pro, ..."
            //   kARAAPIGeneration_2_0_Final = 4,
            //     "supported by Pro Tools"
            //
            // 也就是说 2.0 Draft 是**真实宿主在用的**一代，Cubase/Nuendo 明确在列。
            // 而代数协商失败是**静默**的：`initializeARAWithConfiguration` 的 C ABI
            // 返回 void，宿主无从得知我们拒了它。后果不是"退回旧语义"，而是
            // `createDocumentControllerWithDocument` 直接回 **null** 交给宿主 ——
            // 宿主不会检查这个 null（Cubase 15 实测就是拿着它写字段，在
            // `araservice.dll` 内以空指针 + 0xB0 崩溃，异常 0xc0000005 参数 [1, 0xb0]）。
            //
            // 所以"宿主请求的代数"必须全部落在我们声明的区间里。区间取
            // [2.0 Draft, 2.0 Final]：
            //   * 下界 2.0 Draft —— 覆盖上表里那些真实宿主（含 Cubase/Nuendo）；
            //   * 上界 2.0 Final —— 我们**实测验收过**、且真正履行其契约的一代
            //     （2.3 要求通过 `ARAModelUpdateControllerInterface` 发状态变更通知，
            //     本插件从不调用它，见下方 `capabilities` 与工厂声明测试）；
            //   * 不含 2.X Draft（"used during 2.x development"，不是出货代）。
            // 库对本区间的支持是完整的：2.0 Draft 走 `generation < V2Final` 的 vtable
            // 分支（45 个回调，含 head/tail 扩展），而我们**没有**启用任何
            // ARA-2-Final-专属的控制器能力（`processing_algorithms` / `licensing` /
            // `audio_file_chunks` / `signal_preservation` 一个都没用），因此不会撞上
            // 库的"能力需要 2.0 Final"拒绝。
            .generations(ApiGeneration::V2Draft, ApiGeneration::V2Final)
            // 原kernel处理委托的边界包络，head/tail无额外延伸；tempo-reflecting仍不广告。
            .capabilities(
                FactoryCapabilities::default().with_playback_transformations(
                    PlaybackTransformationFlags::TIMESTRETCH
                        | PlaybackTransformationFlags::CONTENT_FADES,
                ),
            )
            // 每个文档控制器拿到**自己的一份**模型：一份模型对应一份 ARA 文档。
            // 共享一份会让两份文档的累积互相污染（设计 §4.1：v1 是"一实例一编辑轨"）。
            .document_controller(|| {
                // 【为什么记这一行】这是 ARA 握手里**唯一由宿主主动发起**、且能证明
                // "代数协商通过了"的步骤：库在调用本闭包之前先校验协商结果。日志里
                // 缺了它，就说明请求在更早的地方被静默拒掉（代数落在区间外 / 宿主没
                // 提供 assert 指针），宿主随后拿到的是一个 null 控制器。
                log::info!("[ara] host requested a document controller");
                let model = crate::ara::model::ModelHandle::new();
                let head_tail = model.head_tail();
                let session = model.session();
                let playback = session.clone();
                let builder = PluginBuilder::new(model);
                // 适配器创建失败时降级：不注册快照能力，其余照常（见 `publish_head_tail`）。
                let builder = match head_tail {
                    Some(adapter) => builder.realtime_head_tail(adapter),
                    None => builder,
                };
                builder
                    .controller_identity(move |key| session.register(key))
                    .host_playback(move |handle| {
                        crate::log_line(&format!(
                            "ARA host playback control available={}",
                            handle.is_some()
                        ));
                        *playback.playback.lock_or_recover() = handle;
                    })
                    .build()
            })
            .build()?;
        let factory: &'static Factory = Box::leak(Box::new(factory));
        // SAFETY: 工厂被泄漏到进程结束，`as_raw` 指向的 ARAFactory 与工厂同寿。
        let companion =
            unsafe { CompanionFactory::from_raw(crate::CLASS_NAME, &*factory.as_raw())? };
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
    /// 原生工厂仅广告线性拉伸与本应用的委托渐变，不广告未实现的tempo-reflecting。
    #[test]
    fn native_factory_advertises_linear_stretch_and_owned_fades_without_tempo_reflection() {
        let runtime = Runtime::init().unwrap();
        let raw = unsafe { &*runtime.factory.0.as_raw() };
        assert_eq!(
            raw.supportedPlaybackTransformationFlags as u32,
            (PlaybackTransformationFlags::TIMESTRETCH | PlaybackTransformationFlags::CONTENT_FADES)
                .bits()
        );
    }
}
