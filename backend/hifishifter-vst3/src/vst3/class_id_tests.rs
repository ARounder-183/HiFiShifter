//! 类 ID 的编码选择与**宿主文本形式**。
//!
//! 【为什么与接口 IID 分开钉】接口 IID 是"发给宿主去逐字节比较"的，必须与平台 SDK 的
//! `INLINE_UID` 展开（`platform_iid`）一致；类 ID 是"写进工程、下次还要认回来"的
//! **身份**，判据是"同一个身份在两边文本化出来的字符串必须相同"。这两件事恰好选了同一
//! 个布局函数，但**理由不同、可以分开变** —— 所以各有一条用例把选择钉死。谁"顺手把类
//! ID 改成 `uid_guid`"，工程就跨不了平台；谁"顺手把类 ID 改成另一种布局"，Windows 存量
//! 工程就全部作废。

use super::*;
use hifishifter_plugin::vst3_abi::{uid_be, uid_guid};

/// 宿主的文本化规则（与 REAPER 实测一致）：Windows 走 COM/GUID，其它平台走逐字大端。
fn host_text(cid: &Tuid) -> String {
    if cfg!(windows) {
        let word = |range: std::ops::Range<usize>| -> u32 {
            let mut bytes = [0u8; 4];
            bytes.copy_from_slice(&cid[range]);
            u32::from_le_bytes(bytes)
        };
        format!(
            "{:08X}{:04X}{:04X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}{:02X}",
            word(0..4),
            u16::from_le_bytes([cid[4], cid[5]]),
            u16::from_le_bytes([cid[6], cid[7]]),
            cid[8],
            cid[9],
            cid[10],
            cid[11],
            cid[12],
            cid[13],
            cid[14],
            cid[15]
        )
    } else {
        cid.iter().map(|byte| format!("{byte:02X}")).collect()
    }
}

/// 类 ID 按平台的 `INLINE_UID` 编码**同一组字**。
#[test]
fn class_ids_are_encoded_per_platform_from_one_word_set() {
    assert_eq!(PROCESSOR_CID, platform_iid(PROCESSOR_WORDS));
    assert_eq!(CONTROLLER_CID, platform_iid(CONTROLLER_WORDS));
    assert_eq!(FACTORY_CID, platform_iid(FACTORY_WORDS));
    if cfg!(windows) {
        assert_eq!(PROCESSOR_CID, uid_guid(PROCESSOR_WORDS));
    } else {
        assert_eq!(PROCESSOR_CID, uid_be(PROCESSOR_WORDS));
    }
}

/// **Windows 侧字节一个都不许动**：那是存量工程与现役产物已经在用的身份。
///
/// 0.2.0 的 Windows 产物把类 ID 冻结成了 `uid_guid` 字节；`platform_iid` 在 Windows 上
/// 恰好就是 `uid_guid`，所以这次改动对 Windows **完全无感**（缓存、工程、实例都不用动）。
/// 这条断言就是"无感"的证据。
#[test]
fn the_windows_bytes_stay_exactly_what_every_saved_project_uses() {
    if cfg!(windows) {
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
}

/// **回归本体**：每个平台都必须把处理器身份文本化成同一串。
///
/// 这串是 REAPER 写进 `<VST … {…}>` 与 `reaper-vstplugins64.ini` 的东西，也是
/// "macOS 存的工程能不能在 Windows 上认出插件"的**唯一判据**。它同时是 Windows 存量
/// 工程里已经在用的那一串 —— 所以这条用例一旦变红，说明有人同时破坏了跨平台与向后兼容。
#[test]
fn every_platform_derives_the_same_textual_identity() {
    assert_eq!(
        host_text(&PROCESSOR_CID),
        "9C3A4F1700001E8B00004D02B7A13C6E"
    );
    // 控制器 / 主工厂是同族的另两个值：必须互不相同，否则宿主会把三类混为一谈。
    assert_ne!(host_text(&PROCESSOR_CID), host_text(&CONTROLLER_CID));
    assert_ne!(host_text(&PROCESSOR_CID), host_text(&FACTORY_CID));
    assert_ne!(host_text(&CONTROLLER_CID), host_text(&FACTORY_CID));
}
