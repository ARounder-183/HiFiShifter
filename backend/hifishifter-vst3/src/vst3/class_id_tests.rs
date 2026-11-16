//! 类 ID 布局与平台无关；接口 IID 才跟平台走。
//!
//! 【为什么从 `iid_layout_tests` 拆出来】类 ID（`PROCESSOR_CID` 等）与接口 IID 的
//! 布局规则不同：前者是我们自己造的 16 字节身份、宿主只做"原样保存、原样回传"，
//! 不能跟平台变；后者才经 `platform_iid` 展开。壳提取后本文件随
//! `hifishifter-vst3` 壳走（类 ID 是壳的产物身份）。

use super::*;

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
