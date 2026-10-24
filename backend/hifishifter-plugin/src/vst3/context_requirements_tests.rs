//! `IProcessContextRequirements` 的 ABI 与语义回归。
//!
//! 【为什么单独一组】这个接口有一处**极易写错且症状极难归因**的地方：
//! SDK 的原声明是
//!
//! ```cpp
//! class IProcessContextRequirements : public FUnknown
//! ```
//!
//! —— 它**不**继承 `IAudioProcessor`。所以它的 vtable 只有 4 个槽
//! （`queryInterface` / `addRef` / `release` / `getProcessContextRequirements`），
//! 而不是"`IAudioProcessor` 的 11 个槽 + 1"。
//!
//! 若按后者实现，宿主会在**槽 3** 上调 `getProcessContextRequirements`，而那里是
//! `setBusArrangements`：宿主拿到一个 `tresult` 当位掩码用，于是"要哪些字段"这件事
//! 从根上错掉，而插件侧没有任何错误、没有任何日志 —— 症状只会是"宿主有时不发
//! tempo / 拍号"，与真正的原因隔着一整层。
//!
//! 因此这组用例把**槽位布局**本身钉死，而不只是测返回值。

use super::*;
use std::sync::atomic::Ordering;

/// 子对象指针必须真的指向对象内嵌的 vtbl 字段，且能反解回拥有者。
fn context_requirements_ptr(processor: *mut Processor) -> *mut c_void {
    let mut out = std::ptr::null_mut();
    // SAFETY: processor 是活动对象基址；IID 与输出槽在本调用期间存活。
    let result = unsafe {
        component_query_interface(
            processor.cast(),
            uid_guid(IID_IPROCESS_CONTEXT_REQUIREMENTS).as_ptr().cast(),
            &mut out,
        )
    };
    assert_eq!(result, K_RESULT_OK, "必须命中 IProcessContextRequirements");
    assert!(!out.is_null(), "命中就必须给出真实子对象指针");
    out
}

/// **布局契约**：`getProcessContextRequirements` 必须落在第 4 个槽（下标 3）。
///
/// 这条断言是整组用例的核心 —— 它把"接口直接继承 `FUnknown`"这一 SDK 事实
/// 编码成一条机器可查的约束。任何"顺手把方法追加到 `AudioProcessorVtbl`"的改动
/// 都会让它失败，而不是让宿主在运行时拿到 `setBusArrangements` 的返回值。
#[test]
fn the_requirements_method_occupies_the_fourth_slot_of_an_funknown_vtable() {
    assert_eq!(
        std::mem::offset_of!(ProcessContextRequirementsVtbl, base),
        0,
        "FUnknown 的三个槽必须位于 vtbl 开头"
    );
    assert_eq!(
        std::mem::offset_of!(
            ProcessContextRequirementsVtbl,
            get_process_context_requirements
        ),
        3 * std::mem::size_of::<*const std::ffi::c_void>(),
        "getProcessContextRequirements 必须是第 4 个槽（下标 3）—— \
         它继承 FUnknown 而非 IAudioProcessor，槽 3 不是 setBusArrangements"
    );
}

/// 通过真实 `queryInterface` 拿到子对象，并按宿主的调用方式读出掩码。
///
/// 【为什么走 `queryInterface` 而不是直接读静态表】宿主拿到的是指针，
/// 不是我们的静态符号。只有走完整条路（QI → 解引用 vtbl → 调槽 3）才能证明
/// "宿主实际会调用到的那个函数"是对的。
#[test]
fn the_host_reads_the_mask_through_the_real_query_interface_path() {
    let mut processor = Processor::create().expect("测试环境必须能建处理器");
    let base = (&mut *processor) as *mut Processor;
    let subobject = context_requirements_ptr(base);

    // SAFETY: subobject 是对象内嵌的 vtbl 字段地址，其 vtbl 是常驻静态表。
    let table = unsafe { &**(subobject as *const *const ProcessContextRequirementsVtbl) };
    // SAFETY: 槽 3 已由布局用例证明是 getProcessContextRequirements；对象存活。
    let mask = unsafe { (table.get_process_context_requirements)(subobject) };

    assert_eq!(
        mask,
        process_context_requirements(),
        "宿主读到的掩码必须与声明一致"
    );
    assert!(mask > 0, "返回 0 等于对宿主说'我什么都不要'");

    // 反解回拥有者：子对象必须与基址不同（否则说明我们返回了对象首地址，
    // 那样宿主调槽 3 会打到 IComponent::initialize）。
    assert_ne!(
        subobject as usize, base as usize,
        "子对象不能与组件基址重合"
    );
    // SAFETY: subobject 由本对象生成，反解得到同一对象。
    assert_eq!(
        unsafe { base_from_context_requirements(subobject) } as usize,
        base as usize,
        "必须能反解回正确的拥有者"
    );

    // 引用计数共享：子对象上的 release 必须落在拥有者上。
    // 拥有者用 `AtomicI32`（COM 计数是有符号的），接口方法回 `u32`，比较前统一。
    let before = unsafe { (*base).refcount.load(Ordering::Acquire) as u32 };
    // SAFETY: 子对象与拥有者共享同一个原子计数。
    let after = unsafe { (table.base.add)(subobject) };
    assert_eq!(after, before + 1, "子对象 addRef 必须落在拥有者的计数上");
    // SAFETY: 归还上面这一次引用。
    assert_eq!(unsafe { (table.base.release)(subobject) }, before);
}

/// 掩码必须包含插件**真正读取**的每一位，逐位说明消费点。
///
/// 【为什么按位断言而不是比一个魔数】魔数断言在"有人改了一位"时只报
/// "期望 1247，实际 1246"，读者仍要去数位；按位断言直接指出**丢的是哪一项**，
/// 而每一项都对应一处具体消费点。
#[test]
fn the_mask_covers_every_field_the_plugin_consumes() {
    let mask = process_context_requirements();

    // kPlaying / kCycleActive：`audio_process_inner` 的停播判据与 Capture 录制的前提。
    // 【没有这一位，kPlaying 是未定义的，不是"不准"】—— SDK 只在插件声明
    // kNeedTransportState 时才保证 kPlaying 有意义。
    assert_ne!(
        mask & NEED_TRANSPORT_STATE,
        0,
        "kPlaying 是停播判据与 Capture 录制的前提，必须声明"
    );
    // kTempoValid / kTimeSigValid：`TransportClock::tempo()` / `time_signature()` 的唯一来源。
    assert_ne!(mask & NEED_TEMPO, 0, "TransportClock 的 BPM 依赖这一位");
    assert_ne!(
        mask & NEED_TIME_SIGNATURE,
        0,
        "TransportClock 的拍号依赖这一位"
    );
    // kSystemTimeValid / kContTimeValid：诊断读数。
    assert_ne!(mask & NEED_SYSTEM_TIME, 0, "诊断读数依赖 systemTime");
    assert_ne!(
        mask & NEED_CONTINUOUS_TIME_SAMPLES,
        0,
        "循环回卷的旁证依赖 continuousTimeSamples"
    );
    // 音乐位置与循环：Capture 的区间对齐与"只录循环区间"。
    assert_ne!(mask & NEED_PROJECT_TIME_MUSIC, 0);
    assert_ne!(mask & NEED_BAR_POSITION_MUSIC, 0);
    assert_ne!(mask & NEED_CYCLE_MUSIC, 0);

    // 反向：不消费的位**不应**被声明。多要一位的代价是宿主为每个音频块多算一份，
    // 且没有任何日志能证明这件事 —— 属于"不可检测的开销"。
    const NEED_SAMPLES_TO_NEXT_CLOCK: u32 = 1 << 5;
    const NEED_CHORD: u32 = 1 << 8;
    const NEED_FRAME_RATE: u32 = 1 << 9;
    assert_eq!(mask & NEED_CHORD, 0, "和弦识别按块计算，我们没有消费者");
    assert_eq!(
        mask & NEED_SAMPLES_TO_NEXT_CLOCK,
        0,
        "MIDI 时钟分辨率无消费者"
    );
    assert_eq!(mask & NEED_FRAME_RATE, 0, "SMPTE 帧率无消费者");
}

/// 两个实例必须拿到**各自**的子对象指针。
///
/// 【为什么值得一条用例】用共享静态对象实现会通过上面所有断言，
/// 但两个实例会拿到同一地址 —— 一旦将来子对象需要携带实例状态
/// （例如按实例上报不同的需求集），那就是一个静默的串台。
#[test]
fn two_instances_do_not_share_one_subobject() {
    let mut first = Processor::create().expect("测试环境必须能建处理器");
    let mut second = Processor::create().expect("测试环境必须能建处理器");
    let a = context_requirements_ptr((&mut *first) as *mut Processor);
    let b = context_requirements_ptr((&mut *second) as *mut Processor);
    assert_ne!(a as usize, b as usize, "子对象身份必须与实例一一对应");
}
