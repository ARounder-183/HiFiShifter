//! 宿主中文音频对象ID定向回归：保存原字节，不放宽输出ASCII及非法输入防线。
use ara2_bridge::core::{
    ApiGeneration, AraBool, AudioModificationProperties, AudioSourceProperties,
};
use std::ffi::CString;

/// 只替换安全夹具的宿主ID字段，模拟REAPER真实非ASCII输入，而非绕过FFI属性复制。
#[test]
fn source_and_modification_host_ids_preserve_utf8_identity() {
    let source =
        AudioSourceProperties::new(None, "source-ascii", 8820, 44100., 2, AraBool::new(false))
            .unwrap();
    let guard = source.as_ffi(ApiGeneration::V2Final).unwrap();
    let modification = AudioModificationProperties::new(None, "mod-ascii").unwrap();
    let modification_guard = modification.as_ffi();
    for text in ["E:\\音源\\ha.wav", "中文 ID 😀 组合e\u{301}"] {
        let id = CString::new(text).unwrap();
        // SAFETY: raw结构/嵌套name与id都有有效存储，声明尺寸来自真实生成的FFI guard。
        let mut raw = unsafe { std::ptr::read(guard.as_ref().as_ptr()) };
        raw.persistentID = id.as_ptr();
        let owned = unsafe { AudioSourceProperties::copy_from_ffi(&raw) }.unwrap();
        assert_eq!(owned.persistent_id().as_bytes(), text.as_bytes());
        let mut raw_mod = unsafe { std::ptr::read(modification_guard.as_ref().as_ptr()) };
        raw_mod.persistentID = id.as_ptr();
        let owned_mod = unsafe { AudioModificationProperties::copy_from_ffi(&raw_mod) }.unwrap();
        assert_eq!(owned_mod.persistent_id(), text);
        assert!(
            AudioSourceProperties::new(None, text, 8820, 44100., 2, AraBool::new(false)).is_err(),
            "本插件生成ID仍遵守ASCII规范"
        );
        assert!(AudioModificationProperties::new(None, text).is_err());
    }
}
/// null/空串/无效UTF-8仍拒绝，不能为了宿主兼容制造虚假source身份。
#[test]
fn invalid_host_persistent_ids_are_still_rejected() {
    let source =
        AudioSourceProperties::new(None, "valid", 1, 44100., 1, AraBool::new(false)).unwrap();
    let guard = source.as_ffi(ApiGeneration::V2Final).unwrap();
    for bytes in [vec![0], vec![0xff, 0]] {
        let mut raw = unsafe { std::ptr::read(guard.as_ref().as_ptr()) };
        raw.persistentID = bytes.as_ptr().cast();
        assert!(unsafe { AudioSourceProperties::copy_from_ffi(&raw) }.is_err());
    }
    let mut raw = unsafe { std::ptr::read(guard.as_ref().as_ptr()) };
    raw.persistentID = std::ptr::null();
    assert!(unsafe { AudioSourceProperties::copy_from_ffi(&raw) }.is_err());
}
