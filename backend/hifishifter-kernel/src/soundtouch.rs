//! SoundTouch DLL FFI 封装
//!
//! 基于 SoundTouch Windows DLL (`SoundTouchDLL_x64.dll`) 的音频时间拉伸模块。
//! 仅处理“保音高的时间拉伸”，因此固定使用：
//! - `rate = 1.0`
//! - `pitch = 1.0`
//! - `tempo = 1.0 / time_ratio`

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use libloading::Library;

type Handle = *mut c_void;

// ── 运行期加载，不再是 `#[link]` 链接期依赖 ─────────────────────────────────
//
// 【为什么必须改】`#[link(name = "SoundTouchDLL")]` 会在宿主 LoadLibrary 的那一刻就
// 要求解析这个 DLL，而 Windows 只按**宿主目录**搜索（不是插件模块目录）。VST3 靠
// loader shim 的 LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR + 同目录拷贝解决，独立 App 把 DLL
// 放在 exe 旁边；CLAP 是公共目录里的单文件模块，不能往那儿塞同名 DLL。于是历史上
// 只能给 CLAP 整个关掉 `hifishifter-plugin` 的默认 features —— 连带把 `onnx` 也关掉，
// `fcpe_onnx` 退化成 stub：捕获后音高分析只产出电平、F0 恒为空，原线不出现、渲染门禁
// pending 60 s 后超时。三个格式编出三份不同代码，正是"唯一性"被破坏的根源。
//
// 【统一后的事实】所有格式编译**同一份**模块、同一份降级逻辑；差异只剩"DLL 在不在"
// 这个部署事实：在就加载、不在就 `is_available() = false`，`time_stretch` 已有的
// Signalsmith / 线性降级照常兜住（默认算法本来就是 Signalsmith，静态链接、恒可用）。
//
// 【为什么只接受绝对路径】裸文件名会走 OS 默认搜索顺序（CWD 与 PATH 排在 System32
// 之前），用户从下载目录打开工程时同目录同名 DLL 会被抢先加载（CWE-426/427，与
// gpu_info.rs 的 NVML 同一条纪律）。候选只来自显式环境变量或本模块自身所在目录。
type FnCreate = unsafe extern "C" fn() -> Handle;
type FnVoidHandle = unsafe extern "C" fn(Handle);
type FnGetVersion = unsafe extern "C" fn() -> *const c_char;
type FnSetFloat = unsafe extern "C" fn(Handle, f32);
type FnSetUint = unsafe extern "C" fn(Handle, c_uint) -> c_int;
type FnFlush = unsafe extern "C" fn(Handle) -> c_int;
type FnPutSamples = unsafe extern "C" fn(Handle, *const f32, c_uint) -> c_int;
type FnNumSamples = unsafe extern "C" fn(Handle) -> c_uint;
type FnReceiveSamples = unsafe extern "C" fn(Handle, *mut f32, c_uint) -> c_uint;

struct Api {
    // 必须常驻：下面所有函数指针都指向这块映射，Library 一旦 drop 全部悬空。
    _library: Library,
    create: FnCreate,
    destroy: FnVoidHandle,
    get_version: FnGetVersion,
    set_rate: FnSetFloat,
    set_tempo: FnSetFloat,
    set_pitch: FnSetFloat,
    set_channels: FnSetUint,
    set_sample_rate: FnSetUint,
    flush: FnFlush,
    put_samples: FnPutSamples,
    clear: FnVoidHandle,
    num_samples: FnNumSamples,
    receive_samples: FnReceiveSamples,
}

// Library 是 Send + Sync，函数指针是 Copy，因此 Api 可安全放进进程级静态。
static API: OnceLock<Option<Api>> = OnceLock::new();

fn api() -> Option<&'static Api> {
    API.get_or_init(load_api).as_ref()
}

macro_rules! sym {
    ($lib:expr, $ty:ty, $name:literal) => {
        match unsafe { $lib.get::<$ty>(concat!($name, "\0").as_bytes()) } {
            Ok(symbol) => *symbol,
            Err(error) => {
                log::warn!("[soundtouch] missing symbol {}: {error}", $name);
                return None;
            }
        }
    };
}

fn load_api() -> Option<Api> {
    let name = runtime_library_name();
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("HIFISHIFTER_SOUNDTOUCH_DLL") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(dir) = module_dir() {
        candidates.push(dir.join(name));
        // 单文件 CLAP 的私有资源目录：将来若要随插件分发这份 DLL 放这里，
        // 不往公共 CLAP 根里塞通用名（会和其它插件的同名 DLL 冲突）。
        candidates.push(dir.join("HiFiShifter-resources").join(name));
    }
    let mut loaded = None;
    for path in &candidates {
        if !path.is_file() {
            continue;
        }
        if let Ok(library) = unsafe { Library::new(path) } {
            loaded = Some(library);
            break;
        }
    }
    let lib = loaded?;
    Some(Api {
        create: sym!(lib, FnCreate, "soundtouch_createInstance"),
        destroy: sym!(lib, FnVoidHandle, "soundtouch_destroyInstance"),
        get_version: sym!(lib, FnGetVersion, "soundtouch_getVersionString"),
        set_rate: sym!(lib, FnSetFloat, "soundtouch_setRate"),
        set_tempo: sym!(lib, FnSetFloat, "soundtouch_setTempo"),
        set_pitch: sym!(lib, FnSetFloat, "soundtouch_setPitch"),
        set_channels: sym!(lib, FnSetUint, "soundtouch_setChannels"),
        set_sample_rate: sym!(lib, FnSetUint, "soundtouch_setSampleRate"),
        flush: sym!(lib, FnFlush, "soundtouch_flush"),
        put_samples: sym!(lib, FnPutSamples, "soundtouch_putSamples"),
        clear: sym!(lib, FnVoidHandle, "soundtouch_clear"),
        num_samples: sym!(lib, FnNumSamples, "soundtouch_numSamples"),
        receive_samples: sym!(lib, FnReceiveSamples, "soundtouch_receiveSamples"),
        _library: lib,
    })
}

/// 本模块所在 cdylib / 可执行文件的目录。
///
/// 【为什么用自身函数地址】`current_exe()` 在插件里返回的是**宿主**路径。取本模块内
/// 一个函数的地址反查装载模块，是 VST3 / CLAP / App 三种形态都成立的唯一办法
/// （与 `editor::resources::module_path` 同一条纪律）。
fn module_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        use windows::Win32::Foundation::HMODULE;
        use windows::Win32::System::LibraryLoader::{
            GetModuleFileNameW, GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        };
        let mut module = HMODULE::default();
        let ok = unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                    | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                windows::core::PCWSTR(module_dir as *const () as *const u16),
                &mut module,
            )
        };
        if ok.is_err() {
            return None;
        }
        let mut buffer = vec![0_u16; 32768];
        let count = unsafe { GetModuleFileNameW(Some(module), &mut buffer) } as usize;
        if count == 0 || count >= buffer.len() {
            return None;
        }
        PathBuf::from(String::from_utf16_lossy(&buffer[..count]))
            .parent()
            .map(Path::to_path_buf)
    }
    #[cfg(unix)]
    {
        let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
        let found =
            unsafe { libc::dladdr(module_dir as *const () as *const c_void, info.as_mut_ptr()) };
        if found == 0 {
            return None;
        }
        let info = unsafe { info.assume_init() };
        if info.dli_fname.is_null() {
            return None;
        }
        PathBuf::from(
            unsafe { CStr::from_ptr(info.dli_fname) }
                .to_string_lossy()
                .as_ref(),
        )
        .parent()
        .map(Path::to_path_buf)
    }
    #[cfg(not(any(target_os = "windows", unix)))]
    {
        None
    }
}

/// 保持所有原调用点一字不改：wrapper 从 `api()` 取函数指针；DLL 不可用时返回安全
/// 空值，调用方本来就检查 `handle.is_null()` / 返回码，最终落到已有的降级分支。
#[allow(non_snake_case)]
mod ffi {
    use super::*;

    pub unsafe fn soundtouch_createInstance() -> Handle {
        api().map_or(std::ptr::null_mut(), |a| (a.create)())
    }
    pub unsafe fn soundtouch_destroyInstance(handle: Handle) {
        if let Some(a) = api() {
            (a.destroy)(handle)
        }
    }
    pub unsafe fn soundtouch_getVersionString() -> *const c_char {
        api().map_or(std::ptr::null(), |a| (a.get_version)())
    }
    pub unsafe fn soundtouch_setRate(handle: Handle, value: f32) {
        if let Some(a) = api() {
            (a.set_rate)(handle, value)
        }
    }
    pub unsafe fn soundtouch_setTempo(handle: Handle, value: f32) {
        if let Some(a) = api() {
            (a.set_tempo)(handle, value)
        }
    }
    pub unsafe fn soundtouch_setPitch(handle: Handle, value: f32) {
        if let Some(a) = api() {
            (a.set_pitch)(handle, value)
        }
    }
    pub unsafe fn soundtouch_setChannels(handle: Handle, value: c_uint) -> c_int {
        api().map_or(0, |a| (a.set_channels)(handle, value))
    }
    pub unsafe fn soundtouch_setSampleRate(handle: Handle, value: c_uint) -> c_int {
        api().map_or(0, |a| (a.set_sample_rate)(handle, value))
    }
    pub unsafe fn soundtouch_flush(handle: Handle) -> c_int {
        api().map_or(0, |a| (a.flush)(handle))
    }
    pub unsafe fn soundtouch_putSamples(
        handle: Handle,
        samples: *const f32,
        num_samples: c_uint,
    ) -> c_int {
        api().map_or(0, |a| (a.put_samples)(handle, samples, num_samples))
    }
    pub unsafe fn soundtouch_clear(handle: Handle) {
        if let Some(a) = api() {
            (a.clear)(handle)
        }
    }
    pub unsafe fn soundtouch_numSamples(handle: Handle) -> c_uint {
        api().map_or(0, |a| (a.num_samples)(handle))
    }
    pub unsafe fn soundtouch_receiveSamples(
        handle: Handle,
        out_buffer: *mut f32,
        max_samples: c_uint,
    ) -> c_uint {
        api().map_or(0, |a| (a.receive_samples)(handle, out_buffer, max_samples))
    }
}
use ffi::*;

#[derive(Debug, Clone)]
pub enum SoundTouchError {
    RuntimeUnavailable(String),
    InvalidConfig(&'static str),
    ProcessingFailed(&'static str),
}

impl Display for SoundTouchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuntimeUnavailable(name) => {
                write!(f, "SoundTouch runtime unavailable: {name}")
            }
            Self::InvalidConfig(msg) => write!(f, "SoundTouch invalid config: {msg}"),
            Self::ProcessingFailed(msg) => write!(f, "SoundTouch processing failed: {msg}"),
        }
    }
}

impl std::error::Error for SoundTouchError {}

pub fn runtime_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "SoundTouchDLL.dll"
    } else if cfg!(target_os = "macos") {
        "libSoundTouchDLL.dylib"
    } else {
        "libSoundTouchDLL.so"
    }
}

pub fn import_library_name() -> &'static str {
    "SoundTouchDLL.lib"
}

pub fn normalize_output_len(mut output: Vec<f32>, channels: usize, out_frames: usize) -> Vec<f32> {
    let wanted = out_frames.saturating_mul(channels);
    output.resize(wanted, 0.0);
    output.truncate(wanted);
    output
}

/// DLL 是否真的已加载。
///
/// 【为什么不再是恒真】链接期依赖时代 `#[link]` 保证 DLL 一定在（不在进程根本起不来）；
/// 改成运行期加载之后，"不可用"是一个**正常**的部署事实（单文件 CLAP 旁没有这份
/// DLL）。必须如实回报，`time_stretch` 才能落到 Signalsmith 降级，而不是拿一个
/// 空 handle 去跑 SoundTouch 再失败。
pub fn is_available() -> bool {
    api().is_some()
}

fn tempo_from_time_ratio(time_ratio: f64) -> f32 {
    let safe_ratio = if time_ratio.is_finite() && time_ratio > 1.0e-6 {
        time_ratio
    } else {
        1.0
    };
    (1.0 / safe_ratio) as f32
}

fn version_string() -> Option<String> {
    let ptr = unsafe { soundtouch_getVersionString() };
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .trim()
            .to_string(),
    )
}

struct SoundTouchState {
    handle: Handle,
    channels: usize,
}

unsafe impl Send for SoundTouchState {}

impl SoundTouchState {
    fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, SoundTouchError> {
        if !is_available() {
            return Err(SoundTouchError::RuntimeUnavailable(
                runtime_library_name().to_string(),
            ));
        }
        if channels == 0 {
            return Err(SoundTouchError::InvalidConfig("channels == 0"));
        }
        let handle = unsafe { soundtouch_createInstance() };
        if handle.is_null() {
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_createInstance returned null",
            ));
        }

        let init_result = unsafe {
            let ch_ok = soundtouch_setChannels(handle, channels as c_uint);
            let sr_ok = soundtouch_setSampleRate(handle, sample_rate.max(1) as c_uint);
            soundtouch_setRate(handle, 1.0);
            soundtouch_setPitch(handle, 1.0);
            soundtouch_setTempo(handle, tempo_from_time_ratio(time_ratio));
            (ch_ok, sr_ok)
        };

        if init_result.0 == 0 || init_result.1 == 0 {
            unsafe { soundtouch_destroyInstance(handle) };
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_setChannels/sampleRate failed",
            ));
        }

        Ok(Self { handle, channels })
    }

    fn reset(&mut self, time_ratio: f64) -> Result<(), SoundTouchError> {
        unsafe {
            soundtouch_clear(self.handle);
            soundtouch_setRate(self.handle, 1.0);
            soundtouch_setPitch(self.handle, 1.0);
            soundtouch_setTempo(self.handle, tempo_from_time_ratio(time_ratio));
        }
        Ok(())
    }

    fn put_samples(&mut self, input_interleaved: &[f32]) -> Result<(), SoundTouchError> {
        let in_frames = input_interleaved.len() / self.channels.max(1);
        if in_frames == 0 {
            return Ok(());
        }
        let ret = unsafe {
            soundtouch_putSamples(self.handle, input_interleaved.as_ptr(), in_frames as c_uint)
        };
        if ret == 0 {
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_putSamples failed",
            ));
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), SoundTouchError> {
        let ret = unsafe { soundtouch_flush(self.handle) };
        if ret == 0 {
            return Err(SoundTouchError::ProcessingFailed("soundtouch_flush failed"));
        }
        Ok(())
    }

    fn drain_available(
        &mut self,
        out: &mut Vec<f32>,
        max_frames_per_read: usize,
    ) -> Result<(), SoundTouchError> {
        let mut temp = vec![0.0f32; max_frames_per_read.max(1) * self.channels];
        loop {
            let available = unsafe { soundtouch_numSamples(self.handle) as usize };
            if available == 0 {
                break;
            }
            let want_frames = available.min(max_frames_per_read.max(1));
            let got = unsafe {
                soundtouch_receiveSamples(self.handle, temp.as_mut_ptr(), want_frames as c_uint)
                    as usize
            };
            if got == 0 {
                break;
            }
            out.extend_from_slice(&temp[..got * self.channels]);
        }
        Ok(())
    }
}

impl Drop for SoundTouchState {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        unsafe { soundtouch_destroyInstance(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

pub struct RealtimeStretcher {
    inner: SoundTouchState,
    channels: usize,
    out_buffer: Vec<f32>,
    /// Read cursor into `out_buffer`; draining from the front of a large
    /// buffer with `Vec::drain` is O(remaining) per chunk, so we advance a
    /// cursor and only compact when the buffer is fully consumed.
    out_read_pos: usize,
}

unsafe impl Send for RealtimeStretcher {}

impl RealtimeStretcher {
    pub fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, String> {
        let inner =
            SoundTouchState::new(sample_rate, channels, time_ratio).map_err(|e| e.to_string())?;
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            if let Some(version) = version_string() {
                log::warn!(
                    "[SoundTouch] Created realtime stretcher: sr={} ch={} ratio={:.6} version={}",
                    sample_rate,
                    channels,
                    time_ratio,
                    version
                );
            }
        }
        Ok(Self {
            inner,
            channels,
            out_buffer: Vec::with_capacity(4096),
            out_read_pos: 0,
        })
    }

    pub fn reset(&mut self, time_ratio: f64) -> Result<(), String> {
        self.out_buffer.clear();
        self.out_read_pos = 0;
        self.inner.reset(time_ratio).map_err(|e| e.to_string())
    }

    pub fn process_interleaved(
        &mut self,
        input_interleaved: &[f32],
        final_chunk: bool,
    ) -> Result<(), String> {
        if input_interleaved.is_empty() {
            if final_chunk {
                self.inner.flush().map_err(|e| e.to_string())?;
                self.inner
                    .drain_available(&mut self.out_buffer, 1024)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        self.inner
            .put_samples(input_interleaved)
            .map_err(|e| e.to_string())?;
        self.inner
            .drain_available(&mut self.out_buffer, 1024)
            .map_err(|e| e.to_string())?;
        if final_chunk {
            self.inner.flush().map_err(|e| e.to_string())?;
            self.inner
                .drain_available(&mut self.out_buffer, 1024)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn retrieve_interleaved_into(
        &mut self,
        out_interleaved: &mut Vec<f32>,
        max_frames: usize,
    ) -> Result<usize, String> {
        let remaining = self.out_buffer.len() - self.out_read_pos;
        if remaining == 0 || max_frames == 0 {
            return Ok(0);
        }
        let avail_frames = remaining / self.channels.max(1);
        let take_frames = avail_frames.min(max_frames);
        let take_samples = take_frames * self.channels;
        let end = self.out_read_pos + take_samples;
        out_interleaved.extend_from_slice(&self.out_buffer[self.out_read_pos..end]);
        self.out_read_pos = end;
        if self.out_read_pos == self.out_buffer.len() {
            self.out_buffer.clear();
            self.out_read_pos = 0;
        }
        Ok(take_frames)
    }
}

pub fn try_time_stretch_interleaved_offline(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    if input_interleaved.is_empty() || channels == 0 {
        return Ok(vec![]);
    }
    let in_frames = input_interleaved.len() / channels;
    if in_frames < 2 {
        return Ok(normalize_output_len(
            input_interleaved.to_vec(),
            channels,
            out_frames_hint.max(in_frames),
        ));
    }

    let mut state =
        SoundTouchState::new(sample_rate, channels, time_ratio).map_err(|e| e.to_string())?;
    state
        .put_samples(input_interleaved)
        .map_err(|e| e.to_string())?;
    state.flush().map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(out_frames_hint.max(1) * channels);
    state
        .drain_available(&mut out, 4096)
        .map_err(|e| e.to_string())?;

    let target_frames = if out_frames_hint > 0 {
        out_frames_hint
    } else {
        ((in_frames as f64) * time_ratio).round().max(1.0) as usize
    };
    Ok(normalize_output_len(out, channels, target_frames))
}

pub fn try_time_stretch_interleaved_realtime(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    let mut stretcher = RealtimeStretcher::new(sample_rate, channels, time_ratio)?;
    stretcher.process_interleaved(input_interleaved, true)?;
    let mut out = Vec::with_capacity(out_frames_hint.max(1) * channels);
    loop {
        let got = stretcher.retrieve_interleaved_into(&mut out, 4096)?;
        if got == 0 {
            break;
        }
    }
    Ok(normalize_output_len(out, channels, out_frames_hint))
}

#[cfg(test)]
mod tests {
    use super::{normalize_output_len, runtime_library_name, SoundTouchError};

    #[test]
    fn normalize_output_len_pads_short_output() {
        let out = normalize_output_len(vec![1.0, 2.0, 3.0, 4.0], 2, 4);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_output_len_truncates_long_output() {
        let out = normalize_output_len(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn missing_runtime_error_is_clear() {
        let err = SoundTouchError::RuntimeUnavailable(runtime_library_name().to_string());
        assert!(err.to_string().contains(runtime_library_name()));
    }

    #[test]
    fn runtime_library_name_is_pinned() {
        let name = runtime_library_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "SoundTouchDLL.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(name, "libSoundTouchDLL.dylib");
        } else {
            assert_eq!(name, "libSoundTouchDLL.so");
        }
    }

    /// **SoundTouch 是可选运行期依赖，不再是链接期 `#[link]`。**
    ///
    /// 【为什么这条是那次「捕获后没有音高原线」的架构回归】链接期依赖时代它恒真，
    /// 于是 CLAP 只能关掉整个默认 feature 集来避开 DLL 解析，连带把 `onnx` 也关掉。
    /// 现在 `is_available()` 如实反映"DLL 有没有被加载"，且查询本身不碰 CWD / PATH：
    /// 找不到就是 false，`time_stretch` 走 Signalsmith，插件照常工作。
    #[test]
    fn availability_is_a_runtime_fact_not_a_link_time_guarantee() {
        // 不依赖构建产物是否恰好在模块目录：两次查询结果必须一致（OnceLock 只解析一次），
        // 且查询绝不 panic / 绝不因 CWD 里的同名文件而改变。
        let first = super::is_available();
        assert_eq!(super::is_available(), first);
        // 无论 DLL 在不在，FFI wrapper 都必须安全：create 返回空 handle，
        // 计数类返回 0，调用方的既有判据会落到降级分支，而不是 UB。
        let handle = unsafe { super::ffi::soundtouch_createInstance() };
        if !first {
            assert!(handle.is_null(), "DLL 不可用时不得凭空造出句柄");
        }
    }
}
