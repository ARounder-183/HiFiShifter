//! 发给宿主的 IID 字节必须与平台的 `INLINE_UID` 展开逐字节一致。
//!
//! 【为什么值得单独一个文件】这一组断言钉的是**跨平台约定**，不是本模块的实现细节：
//! 任何一次"顺手把 `platform_iid` 改成 `uid_guid`"的改动，都会让 Linux / macOS 上
//! 所有"插件 → 宿主"的 `queryInterface` 静默失败（宿主回 null 指针，日志只剩一行
//! `available=false`），而 Windows 上一切正常 —— 是最难在 CI 里被发现的一类回归。

use super::*;

/// 期望值逐字来自 `INLINE_UID (0x79655E36, 0x77EE4267, 0xA573FEF7, 0x4912C27C)`
/// （`reaper_vst3_interfaces.h` 的 `IReaperHostApplication`）。
///
/// 【为什么拿它当样本】它是"拿不到就整条 ARA 身份链断掉"的那个接口；而且这两个字节串
/// 已经在现场被独立观测过两次：REAPER 的二进制里只有每字大端那一串，插件的 mock 宿主
/// 实验里发出去的则是 COM 布局那一串。
#[test]
fn the_platform_iid_matches_the_inline_uid_macro() {
    const WORDS: [u32; 4] = [0x79655E36, 0x77EE4267, 0xA573FEF7, 0x4912C27C];

    #[cfg(windows)]
    const EXPECTED: [u8; 16] = [
        0x36, 0x5E, 0x65, 0x79, // 第 1 字小端
        0xEE, 0x77, 0x67, 0x42, // 第 2 字拆成两个 u16
        0xA5, 0x73, 0xFE, 0xF7, // 第 3 字大端
        0x49, 0x12, 0xC2, 0x7C, // 第 4 字大端
    ];
    #[cfg(not(windows))]
    const EXPECTED: [u8; 16] = [
        0x79, 0x65, 0x5E, 0x36, // 每字大端
        0x77, 0xEE, 0x42, 0x67, //
        0xA5, 0x73, 0xFE, 0xF7, //
        0x49, 0x12, 0xC2, 0x7C, //
    ];

    assert_eq!(platform_iid(WORDS), EXPECTED);
    assert_eq!(
        iid_layout(),
        if cfg!(windows) {
            "com-guid"
        } else {
            "big-endian"
        }
    );
}

/// 兜底用的"另一种布局"必须真的是另一种，且与 `platform_iid` 互为镜像。
#[test]
fn the_alternate_iid_is_the_other_layout_of_the_same_words() {
    const WORDS: [u32; 4] = [0x58E595CC, 0xDB2D4969, 0x8B6AAF8C, 0x36A664E5];

    #[cfg(windows)]
    let expected = uid_be(WORDS);
    #[cfg(not(windows))]
    let expected = uid_guid(WORDS);

    assert_eq!(alternate_iid(WORDS), expected);
    assert_ne!(platform_iid(WORDS), alternate_iid(WORDS));
}

/// 类 ID **不能**跟着平台变：它们已经写进用户工程，换布局等于让所有实例"插件丢失"。
///
/// 【为什么在这里钉】`platform_iid` 的存在很容易被误读成"所有 TUID 都该走它"。
/// 这条用例把边界画出来：只有"发给宿主去比较的接口 IID"跟着平台走。
#[test]
fn class_ids_stay_in_one_layout_on_every_platform() {
    assert_eq!(
        PROCESSOR_CID,
        uid_guid([0x9C3A_4F17, 0x1E8B, 0x4D02, 0xB7A1_3C6E])
    );
    assert_eq!(
        CONTROLLER_CID,
        uid_guid([0x9C3A_4F18, 0x1E8B, 0x4D02, 0xB7A1_3C6E])
    );
    assert_eq!(
        FACTORY_CID,
        uid_guid([0x9C3A_4F19, 0x1E8B, 0x4D02, 0xB7A1_3C6E])
    );
}
