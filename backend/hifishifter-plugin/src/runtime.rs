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

/// 工厂能力集合：只广告我们**真正履行**的东西。
/// 【为什么没有 CONTENT_FADES】ARA 契约（`ARAInterface.h:993-996`）规定广告这一位意味着
/// **宿主把区域边界的淡化职责交给插件**。但我们并不渲染宿主委托的淡化：两个 fade 位只被
/// 记录与显示，唯一的音频侧消费者是一条**拒绝**路径（`render/snapshot.rs` 在
/// `allow_fades == false` 时报 `UnsupportedTransform`），而 head/tail 快照恒为 `(0,0)`。
/// 广告它等于承诺一件做不到的事 —— 宿主不再自己打淡化，边界就会有咔哒。
///
/// 【为什么不能"实现它"】ARA 只给"头/尾是否有基于内容的淡化"两个布尔，**不给**淡化长度
/// 与形状（`LOST_FIELDS::FadeLength` / `FadeShapeAndCurvature` 是真正拿不到的）。
/// 没有这两个量，插件只能猜一条淡化曲线，那比让宿主用它自己那条正确的曲线更糟。
///
/// 【为什么这里没有可分析内容类型】`analyzeableContentTypes` 保持为空。曾有一个
/// `HIFISHIFTER_ARA_PROBE_ANALYSIS` 探针用来验证"Cubase 是否因为插件广告不出素材的
/// 音乐时值而不交事件"，该假设已被日志否证：整个会话里宿主**一次都没有**调用
/// `getAudioSourceContentGrade` / `isAudioSourceContentAvailable` /
/// `createAudioSourceContentReader`，也没有发起任何分析请求 —— 它压根没问过内容。
/// 真正的成因是 `createAudioModification` 把宿主传来的 null hostRef 当成了硬错误。
/// 探针因此删除，不留一个已经答完的问题的答案。
fn factory_capabilities() -> FactoryCapabilities {
    FactoryCapabilities::default().with_playback_transformations(
        PlaybackTransformationFlags::TIMESTRETCH | PlaybackTransformationFlags::REFLECT_TEMPO,
    )
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
            // 【为什么必须广告 `REFLECT_TEMPO`】宿主在**创建每条 playback region** 时给出
            // transformation flags，而库会拿它跟这里声明的集合做子集校验：
            //
            //   if !self.factory_playback_transformations.contains(requested) { return null_mut(); }
            //
            // 校验不过就**静默**回 null —— 宿主拿不到 region，于是既没有 clip、也不会
            // 调用任何 renderer 的 addPlaybackRegion。实测 Cubase 15 正是如此：
            // `sources=1 regionSequences=1 playbackRegions=0`，编辑器里一条片段都没有，
            // 而 source / region sequence 的创建路径**没有**这个旗标闸门，所以它们照常到达。
            //
            // Cubase 的 ARA region 是跟随工程速度拉伸的，于是带上
            // `kARAPlaybackTransformationTimestretchReflectingTempo`（1<<1）。我们此前只
            // 广告 `TIMESTRETCH | CONTENT_FADES`（漏了这一位），于是**每一条** region
            // 都被丢掉。
            //
            // 【为什么广告它是诚实的】SDK 对这个旗标的定义是"区分拉伸是严格线性、还是反映
            // 音乐上下文与素材内容之间的速度关系"。本插件的拉伸比例正是取自 region 声明的
            // `durationInModificationTime / durationInPlaybackTime` —— 对 tempo-reflecting
            // region 而言，那个比值**就是**宿主算出的速度关系，所以按它渲染正是这一位所要求
            // 的结果。我们**不**支持的是非线性 stretch markers，那在 ARA 里由别处表达，且
            // 本插件会显式报出"不支持非线性拉伸"。
            //
            // 【为什么不能继续拒绝】拒绝的代价不是"退回线性"，而是**整条片段消失**（宿主
            // 拿到 null region）。这与本插件"逐 region 显式降级、不静默丢弃"的纪律相悖 ——
            // 模型侧早已按这个纪律写好（`log_unknown_transformation_flags` 会留痕、映射层
            // 用时长比值），只有工厂这一句声明在反着来。
            //
            // 原kernel处理委托的边界包络，head/tail无额外延伸。
            .capabilities(factory_capabilities())
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
                // 【变更通知通道】宿主只保存"它知道变过"的部分，而插件此前**从未**通知过
                // 任何持久状态变化（`mark_*` 在生产代码里出现次数为 0）。没有这条，
                // 只存增量的宿主会丢数据（`ARAInterface.h:2098-2099` 等三处重申）。
                // 队列在 `notifyModelUpdates` 里由桥接层统一投递，这里只取出句柄。
                session.set_update_emitter(builder.update_emitter());
                // 适配器创建失败时降级：不注册快照能力，其余照常（见 `publish_head_tail`）。
                let builder = match head_tail {
                    Some(adapter) => builder.realtime_head_tail(adapter),
                    None => builder,
                };
                builder
                    // 【为什么在这里注册存档】宿主保存/重载工程时经 ARA 的
                    // `storeDocumentToArchive` / `beginRestoringDocumentFromArchive` 携带插件
                    // 状态。此前这条能力**从未注册**，四个存档回调全部以 `Unsupported` 告终 ——
                    // 宿主每次存盘都被告知"插件存档失败"，重载时状态全丢。provider 绑定本闭包
                    // 刚创建的文档会话，因此不存在"该服务哪个文档"的歧义。
                    // 【为什么是 partial 而不是 persistence】`PartialPersistence` 同时
                    // 覆盖整份与筛选两种存档；只注册整份时，任何带筛选的宿主保存都会拿到
                    // `kARAFalse`，宿主据此认为"插件存档失败"并在增量保存时丢编辑。
                    .partial_persistence(crate::ara::persistence::ArchivePersistence::new(&session))
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
        // 【为什么把对外声明留痕】宿主是否愿意把事件内容交给本插件，读的正是这几个字段。
        // 实测 Cubase 15 建了源与轨道序列却不建任何 region，而日志里看不出我们声明了什么，
        // 于是每次都只能靠推测。`ARAFactory` 是紧凑结构，按值读出字段再打印。
        {
            // SAFETY: 工厂泄漏到进程结束，指针在整个进程期内有效；read_unaligned 接受任意对齐。
            let raw = unsafe { std::ptr::read_unaligned(factory.as_raw()) };
            let lowest = raw.lowestSupportedApiGeneration;
            let highest = raw.highestSupportedApiGeneration;
            let transformations = raw.supportedPlaybackTransformationFlags;
            let analyzeable = raw.analyzeableContentTypesCount;
            log::info!(
                "[ara] factory advertised: generations={lowest}..={highest} \
                 transformations={transformations:#x} analyzeableContentTypes={analyzeable}"
            );
        }
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

    /// **不变量：默认不广告任何"可分析内容类型"。**
    ///
    /// 【为什么钉住】广告一个我们不实现的能力是**假声明**（与早先"宣称 ARA 2.3 变更
    /// 通知"同类）。
    #[test]
    fn the_default_capabilities_advertise_no_analysis() {
        let runtime = Runtime::init().unwrap();
        // SAFETY: 工厂泄漏到进程结束；紧凑结构整份拷出来再读字段。
        let raw = unsafe { std::ptr::read_unaligned(runtime.factory.0.as_raw()) };
        let analyzeable = raw.analyzeableContentTypesCount;
        let transformations = raw.supportedPlaybackTransformationFlags;
        assert_eq!(analyzeable, 0, "不得广告任何可分析内容类型");
        assert_eq!(
            transformations as u32,
            (PlaybackTransformationFlags::TIMESTRETCH | PlaybackTransformationFlags::REFLECT_TEMPO)
                .bits()
        );
    }

    /// **不变量：广告的 transformation 集合必须恰好是"我们能履行的那些"。**
    ///
    /// 【为什么这是"片段凭空消失"级回归】`create_playback_region` 会用广告集合校验宿主
    /// 旗标。此前的行为是不通过就**静默**回 null region —— 宿主拿不到 region，于是既没有
    /// clip、也不会调用任何 renderer 的 addPlaybackRegion（实测 Cubase 15：
    /// `sources=1 regionSequences=1 playbackRegions=0`）。现在那条路径已改为**降级接受
    /// 并留痕**，所以"少广告一位"不再等于"丢一条片段"。
    ///
    /// `REFLECT_TEMPO`（1<<1）是 Cubase 的 ARA region 必带的一位（它的拉伸跟随工程速度）。
    /// 本插件的拉伸比例取自 region 声明的时长比值，对 tempo-reflecting region 而言那正是
    /// 宿主算出的速度关系，因此广告它是诚实的。
    ///
    /// `CONTENT_FADES` **不广告**：广告它意味着宿主把边界淡化的职责交给插件，而我们不渲染
    /// 宿主委托的淡化（ARA 只给两个布尔，不给长度与形状 —— `LOST_FIELDS` 里那两项是真的
    /// 拿不到）。不广告 ⇒ 宿主继续用它自己那条正确的淡化曲线。
    #[test]
    fn native_factory_advertises_exactly_the_transformations_it_honours() {
        let runtime = Runtime::init().unwrap();
        let raw = unsafe { &*runtime.factory.0.as_raw() };
        let advertised = PlaybackTransformationFlags::from_bits_retain(
            raw.supportedPlaybackTransformationFlags as u32,
        );
        let honoured =
            PlaybackTransformationFlags::TIMESTRETCH | PlaybackTransformationFlags::REFLECT_TEMPO;
        assert_eq!(
            advertised.bits(),
            honoured.bits(),
            "广告集合必须恰好是能履行的那些位（CONTENT_FADES 不履行，因此不广告）"
        );
        // 宿主按契约只会设置广告过的位；即便越界，create 路径也只降级不丢弃。
        assert!(advertised.contains(honoured));
    }
}
