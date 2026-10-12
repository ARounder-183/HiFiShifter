//! VST3 工厂的类别声明回归。
//!
//! 【为什么单独一组】三个入口（`getClassInfo` / `getClassInfo2` /
//! `getClassInfoUnicode`）此前各写一份字面量。ARA 契约要求处理器类的
//! `PClassInfo.name` 与 `ARAFactory.plugInName` 一致 —— 宿主据此把 ARA 主工厂与音频
//! 处理器配成一对（`ARAVST3.h` 的 `IMainFactory` 说明）。任何一处漂移都会让宿主配不上
//! 对，而症状是"插件能加载但从不进入 ARA"，极难定位（Cubase 报障正是这一类）。
//!
//! 现在三者共用 [`super::class_declaration`]，这组用例把"同源"这件事钉住。

use super::*;
use ara2_bridge::companion::vst3::ffi::ara2_vst3_release;
use ara2_bridge::core::ApiGeneration;
use std::ffi::CStr;

/// 读一个 `[c_char; N]` 字段（Windows 上 `c_char` 是 `i8`，必须先按字节重解释）。
fn cstr_field(field: &[std::ffi::c_char]) -> String {
    // SAFETY: 调用方传入定长缓冲，其地址与长度都有效。
    let bytes = unsafe { std::slice::from_raw_parts(field.as_ptr().cast::<u8>(), field.len()) };
    CStr::from_bytes_until_nul(bytes)
        .expect("字段必须以 NUL 结尾")
        .to_str()
        .expect("字段必须是 UTF-8")
        .to_owned()
}

/// 读一个 `[u16; N]` 字段（UTF-16）。
fn wide_field(field: &[u16]) -> String {
    let end = field.iter().position(|unit| *unit == 0).unwrap_or(0);
    String::from_utf16(&field[..end]).expect("字段必须是合法 UTF-16")
}

/// 三个入口读出的同一类必须完全一致 —— 包括 UTF-16 与 UTF-8 之间的往返。
#[test]
fn every_class_info_entry_agrees_on_the_same_declaration() {
    for index in 0..=2 {
        let declaration = class_declaration(index).expect("三个类别都必须有声明");
        // SAFETY: 三个结构体都是 POD，先清零再交给入口填写。
        let mut ascii: PClassInfo = unsafe { std::mem::zeroed() };
        let mut ascii2: PClassInfo2 = unsafe { std::mem::zeroed() };
        let mut wide: PClassInfoW = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { factory_get_class_info(std::ptr::null_mut(), index, &mut ascii) },
            K_RESULT_OK
        );
        assert_eq!(
            unsafe { factory_get_class_info2(std::ptr::null_mut(), index, &mut ascii2) },
            K_RESULT_OK
        );
        assert_eq!(
            unsafe {
                factory_get_class_info_unicode(
                    std::ptr::null_mut(),
                    index,
                    (&mut wide as *mut PClassInfoW).cast(),
                )
            },
            K_RESULT_OK
        );
        assert_eq!(ascii.cid, declaration.cid);
        assert_eq!(ascii2.cid, declaration.cid);
        assert_eq!(wide.cid, declaration.cid);
        // category / name 在三个入口里逐字节/逐码元一致。
        assert_eq!(cstr_field(&ascii.category), declaration.category);
        let name = cstr_field(&ascii.name);
        assert_eq!(name, declaration.name);
        assert_eq!(
            wide_field(&wide.name),
            name,
            "UTF-16 与 UTF-8 两个入口的类名必须一致，否则宿主配不上 ARA 主工厂与处理器"
        );
        // `PClassInfo` 是精简版：只有 cid / category / name。
        assert_eq!(ascii.cardinality, K_MANY_INSTANCES);
        assert_eq!(ascii2.cardinality, K_MANY_INSTANCES);
        assert_eq!(wide.cardinality, K_MANY_INSTANCES);
        // 扩展字段在两个"富"入口里也必须一致。
        assert_eq!(
            cstr_field(&ascii2.sub_categories),
            declaration.sub_categories
        );
        assert_eq!(cstr_field(&wide.sub_categories), declaration.sub_categories);
        assert_eq!(ascii2.class_flags, declaration.class_flags);
        assert_eq!(wide.class_flags, declaration.class_flags);
        assert_eq!(cstr_field(&ascii2.vendor), declaration.vendor);
        assert_eq!(wide_field(&wide.vendor), declaration.vendor);
        assert_eq!(cstr_field(&ascii2.version), declaration.version);
        assert_eq!(wide_field(&wide.version), declaration.version);
    }
}

/// 处理器类的名字必须等于 ARA 工厂对外声明的 `plugInName`。
///
/// 【为什么这是 ARA 配对的关键】`ARAVST3.h` 明确要求两者一致，宿主据此把
/// `ARA::IMainFactory` 与 `IAudioProcessor` 配成一对。这条断言把"改了一个忘了另一个"
/// 变成测试期就能看见的失败。
#[test]
fn the_processor_class_name_matches_the_ara_factory_plug_in_name() {
    let runtime = crate::runtime::runtime().expect("进程级 ARA 运行时必须可建");
    // SAFETY: 工厂泄漏到进程结束，指针在整个用例期间有效。
    let factory = unsafe { &*runtime.factory.factory_raw() };
    // SAFETY: `plugInName` 是工厂持有的 NUL 结尾 C 字符串。
    let plug_in_name = unsafe { CStr::from_ptr(factory.plugInName) }
        .to_str()
        .expect("plugInName 必须是合法 UTF-8");
    let declaration = class_declaration(0).expect("处理器类别必须存在");
    assert_eq!(
        declaration.name, plug_in_name,
        "处理器类名与 ARA 工厂 plugInName 不一致 ⇒ 宿主无法把两者配成一对"
    );
}

/// ARA 主工厂的类别串必须来自适配器（与 shim 的常量同源）。
#[test]
fn the_main_factory_category_comes_from_the_companion_shim() {
    let declaration = class_declaration(1).expect("主工厂类别必须存在");
    // SAFETY: 适配器返回常驻 NUL 结尾 C 字符串。
    let expected = unsafe { CStr::from_ptr(ara2_vst3_main_factory_category()) };
    assert_eq!(
        declaration.category.as_bytes(),
        expected.to_bytes(),
        "类别串漂移会让宿主找不到 ARA 主工厂"
    );
    assert_eq!(declaration.category, "ARA Main Factory Class");
}

/// `IPluginFactory3` 必须可查询。
///
/// 【为什么钉住】Cubase 实测会主动查询 `IPluginFactory3`
/// （`factory queryInterface miss iid=ABA2554523C1574E9B12291036878931`）。回
/// `K_NO_INTERFACE` 等于对宿主说"这个工厂没有你认识的那版接口"。
#[test]
fn plugin_factory3_is_queryable() {
    let factory = get_plugin_factory();
    let mut out = std::ptr::null_mut();
    let iid = uid_guid(IID_IPLUGIN_FACTORY3);
    // SAFETY: factory 是常驻单例，out 是可写输出槽。
    let result = unsafe { factory_query_interface(factory, iid.as_ptr(), &mut out) };
    assert_eq!(result, K_RESULT_OK, "IPluginFactory3 必须命中");
    assert_eq!(out, factory, "必须返回工厂对象本身");
    // SAFETY: 归还上面多拿的那一份引用。
    unsafe { factory_release(factory) };

    // 未实现的接口仍必须明确回 `K_NO_INTERFACE`（放宽不得变成"什么都接受"）。
    let mut miss = std::ptr::null_mut();
    let unknown = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF,
        0x00,
    ];
    let result = unsafe { factory_query_interface(factory, unknown.as_ptr(), &mut miss) };
    assert_eq!(result, K_NO_INTERFACE);
    assert!(miss.is_null());
}

/// **不变量：对外宣称的 ARA 代数必须是我们真正履行的契约。**
///
/// 【为什么这是崩溃级回归】库的默认最高代数是 **2.3 Final**，而 2.3 的全部含义是一句
/// 能力承诺（ARA SDK 的 `ARAAPIGeneration` 注释）：插件会通过
/// `ARAModelUpdateControllerInterface` 发送持久状态变更通知，宿主因此可以**只保存变化
/// 的部分**。本插件从不调用那套接口 —— 照默认值宣称 2.3，等于让宿主改用自己的增量保存
/// 与状态记账路径，而通知永远不来。
///
/// 实测：Cubase 15 在读完 `ARAFactory`（即选定代数）之后、创建文档控制器之前，于
/// `araservice.dll` 内以空指针 + 0xB0 写入崩溃（`0xc0000005`，参数 `[1, 0xb0]`）；
/// 而唯一验收过的宿主 REAPER 协商的是 2.0 Final（`apiGeneration=V2Final`）。
#[test]
fn the_advertised_ara_generation_is_the_one_we_actually_honour() {
    let runtime = crate::runtime::runtime().expect("进程级 ARA 运行时必须可建");
    // `ARAFactory` 是**紧凑（packed）**结构（ARA SDK 在 Windows 上按 1 字节对齐生成），
    // 因此不能直接对字段取引用（E0793）。整份拷出来再读，与库的 `raw_copy` 同一手法。
    // SAFETY: 工厂泄漏到进程结束，指针在整个用例期间有效；read_unaligned 接受任意对齐。
    let factory = unsafe { std::ptr::read_unaligned(runtime.factory.factory_raw()) };
    // 按值读出：对 packed 字段**取引用**是未定义行为（E0793），`assert_eq!` 会取引用，
    // 所以先把两个值拷进局部变量。
    let lowest = factory.lowestSupportedApiGeneration;
    let highest = factory.highestSupportedApiGeneration;
    assert_eq!(
        lowest,
        ApiGeneration::V2Final.as_raw(),
        "最低代数必须是实测验收过的那一代"
    );
    assert_eq!(
        highest,
        ApiGeneration::V2Final.as_raw(),
        "最高代数一旦高于 2.0 Final，就是在宣称我们并不实现的 2.3 变更通知契约"
    );
}

/// **不变量：`createInstance` 只交付宿主确实请求的接口。**
///
/// 【为什么必须钉住】VST3 契约要求：请求的 IID 不受支持时必须销毁刚建的对象并回
/// `kNoInterface`。此前主工厂这条分支无条件交出主接口 —— 宿主若问的是别的接口，就会
/// 拿着**错误的 vtable** 调用方法，故障点落在宿主自己的代码里（那种崩溃现场看不出与
/// 插件有关，只会看到宿主的模块）。
#[test]
fn the_main_factory_instance_honours_the_requested_iid() {
    let factory = get_plugin_factory();
    let cid = FACTORY_CID.as_ptr();

    // 不受支持的 IID ⇒ 明确拒绝，且不交出任何指针。
    let unknown = [
        0x11u8, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF, 0x00,
    ];
    let mut rejected = std::ptr::null_mut();
    let result = unsafe {
        factory_create_instance(
            factory,
            cid.cast(),
            unknown.as_ptr().cast(),
            &mut rejected,
        )
    };
    assert_eq!(result, K_NO_INTERFACE, "不受支持的 IID 必须回 kNoInterface");
    assert!(rejected.is_null(), "拒绝时不得交出对象");

    // 受支持：FUnknown（基类）与 ARA::IMainFactory 都必须成功交付。
    let main_factory = ara_iid_words(Ara2Vst3InterfaceKind::MainFactory)
        .expect("适配器必须声明 IMainFactory 的 IID");
    for words in [IID_FUNKNOWN, main_factory] {
        let iid = uid_guid(words);
        let mut out = std::ptr::null_mut();
        let result = unsafe {
            factory_create_instance(factory, cid.cast(), iid.as_ptr().cast(), &mut out)
        };
        assert_eq!(result, K_RESULT_OK, "受支持的 IID 必须交付");
        assert!(!out.is_null());
        // 交付出去的那一份引用由宿主（这里就是本测试）负责释放。
        let mut remaining = 0;
        // SAFETY: out 是本用例刚刚拿到的 owning COM 引用。
        unsafe { ara2_vst3_release(out, &mut remaining) };
    }
    // SAFETY: 归还 `get_plugin_factory` 多拿的那一份工厂引用。
    unsafe { factory_release(factory) };
}
