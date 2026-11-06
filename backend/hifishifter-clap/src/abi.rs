//! CLAP 1.1.9 的精简 ABI 声明。
//!
//! 【为什么是手写声明】`ara2-bridge-companion` 只 checked-in 了 ARA-over-CLAP 的结构，
//! 不含完整的 `clap_process_t` / `clap_plugin_t`。这里按 `third_party/sdk/clap` 的
//! 1.1.9 头文件逐字段复刻，且只声明本产物实际消费的部分；每个 `#[repr(C)]` 的
//! 字段顺序由 `tests/clap_layout.rs` 与头文件常量共同钉住。
//! 【禁止外部 SDK】不链接 CLAP 的 C 库：CLAP 是纯头文件 ABI，宿主按 `clap_entry` 加载。

#![allow(non_camel_case_types)]
use std::ffi::{c_char, c_void};

/// 静态 C 字符串指针的 Sync 包装。
///
/// 【为什么需要】CLAP 的 descriptor / entry 是进程级 `static`，Rust 要求它 Sync；
/// 里面只有指向 `.rodata` 的裸指针与函数指针，宿主按 ABI 承诺只读。
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct StaticCStr(pub *const c_char);
// SAFETY: 见 StaticCStr 文档；这些指针从不改写，且跨线程只读。
unsafe impl Sync for StaticCStr {}
unsafe impl Send for StaticCStr {}

/// 把静态 NUL 结尾字节串转成 C 字符串指针（const 上下文可用）。
pub const fn cstr(bytes: &'static [u8]) -> *const c_char {
    bytes.as_ptr().cast()
}

macro_rules! sync_c_table {
    ($($ty:ty),+ $(,)?) => { $(
        // SAFETY: 表内只有 host 只读的函数指针与静态数据，见模块头说明。
        unsafe impl Sync for $ty {}
        unsafe impl Send for $ty {}
    )+ };
}
sync_c_table!(
    clap_plugin_descriptor_t,
    clap_plugin_t,
    clap_plugin_factory_t,
    clap_plugin_entry_t,
    clap_host_t,
    clap_process_t,
    clap_audio_buffer_t,
    clap_event_transport_t,
);

/// CLAP 1.1.9 语义版本。
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct clap_version_t {
    pub major: u32,
    pub minor: u32,
    pub revision: u32,
}
/// 本产物编译时所依据的 CLAP 版本。
pub const CLAP_VERSION: clap_version_t = clap_version_t {
    major: 1,
    minor: 1,
    revision: 9,
};

/// `clap_process_status`：成功并继续处理。
pub const CLAP_PROCESS_CONTINUE: i32 = 1;
/// `clap_process_status`：失败，输出缓冲必须丢弃。
pub const CLAP_PROCESS_ERROR: i32 = 0;
/// 离线 render 模式。
pub const CLAP_RENDER_OFFLINE: i32 = 1;
/// 实时 render 模式。
pub const CLAP_RENDER_REALTIME: i32 = 0;

/// CLAP transport 标志（events.h）。
pub mod transport_flags {
    pub const HAS_TEMPO: u32 = 1 << 0;
    pub const HAS_BEATS_TIMELINE: u32 = 1 << 1;
    pub const HAS_SECONDS_TIMELINE: u32 = 1 << 2;
    pub const HAS_TIME_SIGNATURE: u32 = 1 << 3;
    pub const IS_PLAYING: u32 = 1 << 4;
    pub const IS_RECORDING: u32 = 1 << 5;
    pub const IS_LOOP_ACTIVE: u32 = 1 << 6;
    pub const IS_WITHIN_PRE_ROLL: u32 = 1 << 7;
}

/// CLAP 事件头。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct clap_event_header_t {
    pub size: u32,
    pub time: u32,
    pub space_id: u16,
    pub type_: u16,
    pub flags: u32,
}
/// `CLAP_EVENT_TRANSPORT`。
pub const CLAP_EVENT_TRANSPORT: u16 = 9;
/// CLAP 定点时间；头文件为 64 位带符号定点值。
pub type clap_beattime = i64;
/// CLAP 秒时间；头文件为 64 位带符号定点值。
pub type clap_sectime = i64;
/// 固定点换标（fixedpoint.h: 1<<31）。
pub const CLAP_SECTIME_FACTOR: f64 = 2_147_483_648.0;

/// `clap_event_transport_t`。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct clap_event_transport_t {
    pub header: clap_event_header_t,
    pub flags: u32,
    pub song_pos_beats: clap_beattime,
    pub song_pos_seconds: clap_sectime,
    pub tempo: f64,
    pub tempo_inc: f64,
    pub loop_start_beats: clap_beattime,
    pub loop_end_beats: clap_beattime,
    pub loop_start_seconds: clap_sectime,
    pub loop_end_seconds: clap_sectime,
    pub bar_start: clap_beattime,
    pub bar_number: i32,
    pub tsig_num: u16,
    pub tsig_denom: u16,
}

/// `clap_audio_buffer_t`。
#[repr(C)]
pub struct clap_audio_buffer_t {
    pub data32: *mut *mut f32,
    pub data64: *mut *mut f64,
    pub channel_count: u32,
    pub latency: u32,
    pub constant_mask: u64,
}

/// `clap_process_t`。只声明到本产物读取的字段；事件表的布局不需要暴露。
#[repr(C)]
pub struct clap_process_t {
    pub steady_time: i64,
    pub frames_count: u32,
    pub transport: *const clap_event_transport_t,
    pub audio_inputs: *const clap_audio_buffer_t,
    pub audio_outputs: *mut clap_audio_buffer_t,
    pub audio_inputs_count: u32,
    pub audio_outputs_count: u32,
    pub in_events: *const c_void,
    pub out_events: *const c_void,
}

/// `clap_plugin_descriptor_t`。
#[repr(C)]
pub struct clap_plugin_descriptor_t {
    pub clap_version: clap_version_t,
    pub id: *const c_char,
    pub name: *const c_char,
    pub vendor: *const c_char,
    pub url: *const c_char,
    pub manual_url: *const c_char,
    pub support_url: *const c_char,
    pub version: *const c_char,
    pub description: *const c_char,
    pub features: *const *const c_char,
}

/// `clap_host_t`（只消费版本与 get_extension，后面字段按布局占位）。
#[repr(C)]
pub struct clap_host_t {
    pub clap_version: clap_version_t,
    pub host_data: *mut c_void,
    pub name: *const c_char,
    pub vendor: *const c_char,
    pub url: *const c_char,
    pub version: *const c_char,
    pub get_extension:
        Option<unsafe extern "C" fn(*const clap_host_t, *const c_char) -> *const c_void>,
    pub request_restart: Option<unsafe extern "C" fn(*const clap_host_t)>,
    pub request_process: Option<unsafe extern "C" fn(*const clap_host_t)>,
    pub request_callback: Option<unsafe extern "C" fn(*const clap_host_t)>,
}

/// `clap_plugin_t`。
#[repr(C)]
pub struct clap_plugin_t {
    pub desc: *const clap_plugin_descriptor_t,
    pub plugin_data: *mut c_void,
    pub init: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
    pub destroy: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
    pub activate: Option<unsafe extern "C" fn(*const clap_plugin_t, f64, u32, u32) -> bool>,
    pub deactivate: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
    pub start_processing: Option<unsafe extern "C" fn(*const clap_plugin_t) -> bool>,
    pub stop_processing: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
    pub reset: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
    pub process: Option<unsafe extern "C" fn(*const clap_plugin_t, *const clap_process_t) -> i32>,
    pub get_extension:
        Option<unsafe extern "C" fn(*const clap_plugin_t, *const c_char) -> *const c_void>,
    pub on_main_thread: Option<unsafe extern "C" fn(*const clap_plugin_t)>,
}

/// `clap_plugin_factory_t`。
#[repr(C)]
pub struct clap_plugin_factory_t {
    pub get_plugin_count: Option<unsafe extern "C" fn(*const clap_plugin_factory_t) -> u32>,
    pub get_plugin_descriptor: Option<
        unsafe extern "C" fn(*const clap_plugin_factory_t, u32) -> *const clap_plugin_descriptor_t,
    >,
    pub create_plugin: Option<
        unsafe extern "C" fn(
            *const clap_plugin_factory_t,
            *const clap_host_t,
            *const c_char,
        ) -> *const clap_plugin_t,
    >,
}

/// `clap_plugin_entry_t`。
#[repr(C)]
pub struct clap_plugin_entry_t {
    pub clap_version: clap_version_t,
    pub init: Option<unsafe extern "C" fn(*const c_char) -> bool>,
    pub deinit: Option<unsafe extern "C" fn()>,
    pub get_factory: Option<unsafe extern "C" fn(*const c_char) -> *const c_void>,
}

/// 标准插件工厂 ID。
pub const CLAP_PLUGIN_FACTORY_ID: &[u8] = b"clap.plugin-factory\0";
/// `clap.state` 扩展 ID。
pub const CLAP_EXT_STATE: &[u8] = b"clap.state\0";
/// `clap.render` 扩展 ID。
pub const CLAP_EXT_RENDER: &[u8] = b"clap.render\0";
/// `clap.audio-ports` 扩展 ID。
pub const CLAP_EXT_AUDIO_PORTS: &[u8] = b"clap.audio-ports\0";
/// `clap.gui` 扩展 ID。
pub const CLAP_EXT_GUI: &[u8] = b"clap.gui\0";
/// 阶段 2 启用 ARA 时才声明 `ara:supported`；Capture-only 外壳绝不谎报能力。
pub const FEATURE_ARA_SUPPORTED: &[u8] = b"ara:supported\0";
/// 插件分类：音频效果器。
pub const FEATURE_AUDIO_EFFECT: &[u8] = b"audio-effect\0";
/// 插件分类：音高修正。
pub const FEATURE_PITCH_CORRECTION: &[u8] = b"pitch-correction\0";
/// 插件分类：立体声。
pub const FEATURE_STEREO: &[u8] = b"stereo\0";
/// 稳定 descriptor ID。
pub const PLUGIN_ID: &[u8] = b"com.hifishifter.clap\0";
/// 产品名。
pub const PLUGIN_NAME: &[u8] = b"HiFiShifter\0";
/// 厂商名。
pub const VENDOR: &[u8] = b"HiFiShifter\0";
/// 项目 URL。
pub const URL: &[u8] = b"https://github.com/ARounder-183/HiFiShifter\0";
/// 描述。
pub const DESCRIPTION: &[u8] = b"HiFiShifter CLAP plug-in (Capture / ARA)\0";

/// descriptor ID 的 UTF-8 形式（ARA companion 配对键）。
pub const PLUGIN_ID_STR: &str = "com.hifishifter.clap";

/// 静态函数/指针数组的 Sync 包装。
#[repr(transparent)]
pub struct StaticPtrArray(pub [*const c_char; 4]);
// SAFETY: 只读静态表，见模块头。
unsafe impl Sync for StaticPtrArray {}
unsafe impl Send for StaticPtrArray {}

/// 由 build.rs 从产品版本单一来源注入的静态 C 字符串。
pub const PRODUCT_VERSION_CSTR: &[u8] = concat!(env!("HFS_PRODUCT_VERSION"), "\0").as_bytes();
