//! 宿主握手回归：**代数协商 → 文档控制器**。
//!
//! 【为什么单独一组】这条链上有三个"静默失败"叠在一起，任一处出问题都表现为
//! "宿主崩在自己的代码里"：
//!
//! 1. `initializeARAWithConfiguration` 的 C ABI 返回 **void** —— 我们拒了宿主，
//!    宿主无从得知；
//! 2. 库把拒绝的结果折叠成 `createDocumentControllerWithDocument` 返回 **null**
//!    （`factory.rs` 的 `.ok()` + `unwrap_or(null)`）；
//! 3. 宿主**不检查**这个 null —— 它会拿着它写字段。
//!
//! Cubase 15 实测就是这样：插入为 ARA 插件后，`araservice.dll` 内以空指针 + 0xB0
//! 写入崩溃（异常 `0xc0000005`，参数 `[1, 0xb0]`，两次 dump 同一偏移），而我们的
//! 模块既不在调用栈上、也不在崩溃进程的模块表里。
//!
//! 本组用例按**宿主的方式**驱动真实工厂（不是调库的内部函数），把"宿主请求的每一代
//! 都必须拿到真实控制器"钉死。

use ara2_bridge::core::ApiGeneration;
use ara2_bridge::plugin::{Factory, FactoryBuilder, PluginBuilder};
use ara2_bridge::sys::{ARADocumentControllerInstance, ARADocumentProperties};

/// 生产配置里声明的代数区间（从**运行时工厂记录**读回，与 `runtime.rs` 同源）。
///
/// 【为什么不写字面量】这组用例的全部意义是"声明区间要覆盖宿主会请求的代数"。把区间
/// 抄成字面量，`runtime.rs` 一改窄用例就会假通过 —— 那正是这次要防的回归。
fn advertised_range() -> (ApiGeneration, ApiGeneration) {
    let runtime = crate::runtime::runtime().expect("进程级 ARA 运行时必须可建");
    // SAFETY: 工厂泄漏到进程结束；`ARAFactory` 是紧凑结构，整份拷出来再读字段。
    let raw = unsafe { std::ptr::read_unaligned(runtime.factory.factory_raw()) };
    // 按值读出（对 packed 字段取引用是 UB，见 `factory_declaration_tests` 的同款说明）。
    let lowest = raw.lowestSupportedApiGeneration;
    let highest = raw.highestSupportedApiGeneration;
    (
        ApiGeneration::try_from_raw(lowest).expect("生产工厂声明了未知的最低代数"),
        ApiGeneration::try_from_raw(highest).expect("生产工厂声明了未知的最高代数"),
    )
}

/// 造一个与生产同配置（同代数区间）的工厂，供本组用例逐个代数驱动。
///
/// 【为什么不用 `runtime()` 的单例工厂】`initialize` 对一个工厂只能成功一次
/// （`factory is already initialized`），而本组要逐个代数各驱动一次。
fn probe_factory() -> &'static Factory {
    let (lowest, highest) = advertised_range();
    Box::leak(Box::new(
        FactoryBuilder::new("org.hfs.handshake.probe", "org.hfs.handshake.probe.archive")
            .display(
                crate::CLASS_NAME,
                "HiFiShifter",
                "https://github.com/ARounder-183/HiFiShifter",
                crate::VERSION,
            )
            .generations(lowest, highest)
            .document_controller(|| {
                let model = crate::ara::model::ModelHandle::new();
                let session = model.session();
                PluginBuilder::new(model)
                    .controller_identity(move |key| session.register(key))
                    .build()
            })
            .build()
            .expect("探针工厂必须可建"),
    ))
}

/// 按宿主的方式请求一个文档控制器，返回原始指针（null 表示我们把它交给了宿主）。
fn request_document_controller(factory: &Factory) -> *const ARADocumentControllerInstance {
    let mut fixture = crate::test_host::HostFixture::new(vec![]);
    let host = fixture.instance();
    let properties = ARADocumentProperties {
        structSize: std::mem::size_of::<ARADocumentProperties>(),
        name: c"handshake".as_ptr(),
    };
    // SAFETY: fixture / properties 在本次调用期间存活；工厂 backing 与进程同寿。
    unsafe {
        (factory
            .raw_copy()
            .createDocumentControllerWithDocument
            .expect("工厂必须导出文档控制器构造器"))(&host, &properties)
    }
}

/// 归还一个由本组用例创建的控制器（不归还会把文档留在进程级登记表里）。
fn release_controller(raw: *const ARADocumentControllerInstance) {
    // SAFETY: 调用方保证 raw 是本次创建、尚未销毁的控制器；原工厂返回完整 packed 实例。
    let controller = unsafe { raw.read_unaligned() };
    // SAFETY: 控制器由本次调用创建，销毁入口存在。
    unsafe {
        ((*controller.documentControllerInterface)
            .destroyDocumentController
            .expect("控制器必须导出销毁入口"))(controller.documentControllerRef)
    };
}

/// **不变量：真实宿主会请求的每一代 ARA 2，都必须拿到一个真实控制器。**
///
/// 【为什么是这两代】ARA SDK 的 `ARAAPIGeneration` 注释把每一代的使用者写明了：
/// 2.0 **Draft**（3）列出 Studio One / Logic Pro / **Cubase·Nuendo** / Cakewalk /
/// REAPER / Melodyne；2.0 **Final**（4）列出 Pro Tools。Cubase 15 实测崩在这条路上。
#[test]
fn every_host_generation_in_our_range_yields_a_real_controller() {
    for generation in [ApiGeneration::V2Draft, ApiGeneration::V2Final] {
        let factory = probe_factory();
        factory
            .entry()
            .initialize(generation, crate::test_host::assert_address())
            .unwrap_or_else(|error| {
                panic!("{generation:?} 被拒了：{error} —— 宿主拿不到文档控制器，会拿着 null 继续写")
            });
        let raw = request_document_controller(factory);
        assert!(
            !raw.is_null(),
            "{generation:?} 必须拿到真实控制器：null 会被宿主直接写穿（Cubase 实测即如此）"
        );
        release_controller(raw);
        factory.entry().uninitialize().expect("工厂必须可注销");
    }
}

/// **不变量：宿主没提供 assert 指针时，也必须拿到真实控制器。**
///
/// 【为什么必须容忍】规范要求宿主总是提供该指针，但拒绝的代价极不对称：拒绝是静默的
/// （C ABI 返回 void），随后交给宿主的是 **null** 控制器，宿主拿着它写字段就崩在自己的
/// 代码里。而该地址在本库**只被记录与比较、从不被调用**，取 0 的唯一后果是"没装调试
/// 断言钩子"。把宿主的轻微不规范变成崩溃是不可接受的。
#[test]
fn a_host_without_an_assert_address_still_gets_a_real_controller() {
    let factory = probe_factory();
    factory
        .entry()
        .initialize(ApiGeneration::V2Final, std::ptr::null_mut())
        .expect("缺少 assert 指针不得导致拒绝 —— 拒绝的后果是把 null 交给宿主");
    let raw = request_document_controller(factory);
    assert!(
        !raw.is_null(),
        "缺 assert 指针时仍必须拿到真实控制器，而不是 null"
    );
    release_controller(raw);
    factory.entry().uninitialize().expect("工厂必须可注销");
}

/// **不变量：代数落在声明区间之外时，宿主拿到的是 null —— 这就是崩溃的来源。**
///
/// 【为什么必须把这个失败模式钉住】它不是"优雅降级"，而是把 `null` 交给宿主。这条
/// 用例的价值在于把"区间必须够宽"从一句注释变成一个可执行的断言：一旦有人再把下界
/// 收窄，`every_host_generation_in_our_range_yields_a_real_controller` 会先红，
/// 这条则解释红在哪。
#[test]
fn a_generation_outside_our_range_hands_the_host_a_null_controller() {
    let factory = probe_factory();
    let (lowest, _highest) = advertised_range();
    // ARA 1.0 Final：结构上不属于我们声明的 ARA 2 区间。
    let outside = ApiGeneration::V1Final;
    assert!(
        outside < lowest,
        "本用例假定 ARA 1.0 落在声明区间之外；区间变了就要重选一个越界代数"
    );
    let rejected = factory
        .entry()
        .initialize(outside, crate::test_host::assert_address());
    assert!(rejected.is_err(), "区间外的代数必须被拒（这正是问题所在）");
    let raw = request_document_controller(factory);
    assert!(
        raw.is_null(),
        "初始化被拒之后，`createDocumentControllerWithDocument` 会回 null 给宿主"
    );
}
