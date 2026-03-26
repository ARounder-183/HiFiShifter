//! VST3 COM 接口辅助模块。
//!
//! 封装 `vst3` crate 提供的 Steinberg COM 接口，
//! 提供类型安全的 VST3 插件加载、初始化、音频处理、状态管理和 GUI 操作。

use std::ffi::c_void;
use std::ptr;

use vst3::ComPtr;
use vst3::Steinberg::*;
use vst3::Steinberg::Vst::*;

// ─── 常量定义 ────────────────────────────────────────────────────────────────

/// VST3 32 位浮点采样标志。
const K_SAMPLE_32: i32 = 0; // kSample32
/// VST3 实时处理模式。
const K_REALTIME: i32 = 0; // kRealtime
/// VST3 离线处理模式。
#[allow(dead_code)]
const K_OFFLINE: i32 = 1; // kOffline
/// 音频媒体类型。
const K_AUDIO: i32 = 0; // MediaTypes::kAudio
/// 事件媒体类型。
#[allow(dead_code)]
const K_EVENT: i32 = 1; // MediaTypes::kEvent
/// 输入方向。
const K_INPUT: i32 = 0; // BusDirections::kInput
/// 输出方向。
const K_OUTPUT: i32 = 1; // BusDirections::kOutput

// ─── IBStream 内存实现 ────────────────────────────────────────────────────────

/// 内存中的 IBStream 实现（用于 getState/setState）。
///
/// VST3 的状态序列化通过 IBStream 接口进行，
/// 此结构体提供一个基于 Vec<u8> 的内存流实现。
#[repr(C)]
pub struct MemoryStream {
    vtbl: *const IBStreamVtbl,
    ref_count: std::sync::atomic::AtomicI32,
    data: std::sync::Mutex<MemoryStreamData>,
}

struct MemoryStreamData {
    buffer: Vec<u8>,
    position: i64,
}

impl MemoryStream {
    /// 创建空的内存流。
    pub fn new() -> Box<Self> {
        let stream = Box::new(Self {
            vtbl: &MEMORY_STREAM_VTBL,
            ref_count: std::sync::atomic::AtomicI32::new(1),
            data: std::sync::Mutex::new(MemoryStreamData {
                buffer: Vec::new(),
                position: 0,
            }),
        });
        stream
    }

    /// 从已有数据创建内存流。
    pub fn from_data(data: Vec<u8>) -> Box<Self> {
        let stream = Box::new(Self {
            vtbl: &MEMORY_STREAM_VTBL,
            ref_count: std::sync::atomic::AtomicI32::new(1),
            data: std::sync::Mutex::new(MemoryStreamData {
                buffer: data,
                position: 0,
            }),
        });
        stream
    }

    /// 获取流中的数据。
    pub fn get_data(&self) -> Vec<u8> {
        let data = self.data.lock().unwrap();
        data.buffer.clone()
    }

    /// 将 Box<Self> 转换为原始 IBStream 指针。
    pub fn into_raw(self: Box<Self>) -> *mut IBStream {
        Box::into_raw(self) as *mut IBStream
    }
}

// IBStream VTable 实现
static MEMORY_STREAM_VTBL: IBStreamVtbl = IBStreamVtbl {
    base: FUnknownVtbl {
        queryInterface: memory_stream_query_interface,
        addRef: memory_stream_add_ref,
        release: memory_stream_release,
    },
    read: memory_stream_read,
    write: memory_stream_write,
    seek: memory_stream_seek,
    tell: memory_stream_tell,
};

unsafe extern "system" fn memory_stream_query_interface(
    this: *mut FUnknown,
    _iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    // 简化实现：返回自身
    *obj = this as *mut c_void;
    memory_stream_add_ref(this);
    kResultOk
}

unsafe extern "system" fn memory_stream_add_ref(this: *mut FUnknown) -> u32 {
    let stream = &*(this as *const MemoryStream);
    let prev = stream.ref_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    (prev + 1) as u32
}

unsafe extern "system" fn memory_stream_release(this: *mut FUnknown) -> u32 {
    let stream = &*(this as *const MemoryStream);
    let prev = stream.ref_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    if prev == 1 {
        // 引用计数归零，释放内存
        drop(Box::from_raw(this as *mut MemoryStream));
        return 0;
    }
    (prev - 1) as u32
}

unsafe extern "system" fn memory_stream_read(
    this: *mut IBStream,
    buffer: *mut c_void,
    num_bytes: i32,
    num_bytes_read: *mut i32,
) -> tresult {
    let stream = &*(this as *const MemoryStream);
    let mut data = stream.data.lock().unwrap();
    let pos = data.position as usize;
    let available = data.buffer.len().saturating_sub(pos);
    let to_read = (num_bytes as usize).min(available);

    if to_read > 0 {
        ptr::copy_nonoverlapping(
            data.buffer[pos..].as_ptr(),
            buffer as *mut u8,
            to_read,
        );
    }
    data.position += to_read as i64;

    if !num_bytes_read.is_null() {
        *num_bytes_read = to_read as i32;
    }
    kResultOk
}

unsafe extern "system" fn memory_stream_write(
    this: *mut IBStream,
    buffer: *mut c_void,
    num_bytes: i32,
    num_bytes_written: *mut i32,
) -> tresult {
    let stream = &*(this as *const MemoryStream);
    let mut data = stream.data.lock().unwrap();
    let pos = data.position as usize;
    let write_end = pos + num_bytes as usize;

    // 扩展 buffer 如有必要
    if write_end > data.buffer.len() {
        data.buffer.resize(write_end, 0);
    }

    ptr::copy_nonoverlapping(
        buffer as *const u8,
        data.buffer[pos..].as_mut_ptr(),
        num_bytes as usize,
    );
    data.position = write_end as i64;

    if !num_bytes_written.is_null() {
        *num_bytes_written = num_bytes;
    }
    kResultOk
}

unsafe extern "system" fn memory_stream_seek(
    this: *mut IBStream,
    pos: i64,
    mode: i32,
    result: *mut i64,
) -> tresult {
    let stream = &*(this as *const MemoryStream);
    let mut data = stream.data.lock().unwrap();
    let new_pos = match mode {
        0 => pos,                                  // kIBSeekSet
        1 => data.position + pos,                  // kIBSeekCur
        2 => data.buffer.len() as i64 + pos,       // kIBSeekEnd
        _ => return kInvalidArgument,
    };
    if new_pos < 0 {
        return kInvalidArgument;
    }
    data.position = new_pos;
    if !result.is_null() {
        *result = new_pos;
    }
    kResultOk
}

unsafe extern "system" fn memory_stream_tell(
    this: *mut IBStream,
    pos: *mut i64,
) -> tresult {
    let stream = &*(this as *const MemoryStream);
    let data = stream.data.lock().unwrap();
    if !pos.is_null() {
        *pos = data.position;
    }
    kResultOk
}

// ─── VST3 实例封装 ──────────────────────────────────────────────────────────

// ─── IPlugFrame 最小实现 ─────────────────────────────────────────────────────
//
// VST3 规范要求宿主在 IPlugView::attached() 前通过 setFrame() 传入 IPlugFrame。
// 许多插件在没有 IPlugFrame 的情况下不渲染 GUI（白屏）。
// 这里提供一个最小实现：resizeView 直接返回 kResultOk。

/// IPlugFrame 最小实现的 VTable。
#[cfg(target_os = "windows")]
static SIMPLE_PLUG_FRAME_VTBL: IPlugFrameVtbl = IPlugFrameVtbl {
    base: FUnknownVtbl {
        queryInterface: simple_plug_frame_query_interface,
        addRef: simple_plug_frame_add_ref,
        release: simple_plug_frame_release,
    },
    resizeView: simple_plug_frame_resize_view,
};

/// 简化的 IPlugFrame 宿主实现。
///
/// 使用引用计数管理生命周期。`resizeView` 回调中调整宿主窗口大小
/// 以配合插件请求的尺寸变化。
#[repr(C)]
#[cfg(target_os = "windows")]
pub struct SimplePlugFrame {
    vtbl: *const IPlugFrameVtbl,
    ref_count: std::sync::atomic::AtomicI32,
    /// 宿主窗口句柄，用于在 resizeView 中调整窗口大小。
    hwnd: *mut c_void,
}

#[cfg(target_os = "windows")]
unsafe impl Send for SimplePlugFrame {}

#[cfg(target_os = "windows")]
impl SimplePlugFrame {
    /// 创建新的 IPlugFrame 实例，关联到指定的宿主窗口句柄。
    pub fn new(hwnd: *mut c_void) -> Box<Self> {
        Box::new(Self {
            vtbl: &SIMPLE_PLUG_FRAME_VTBL,
            ref_count: std::sync::atomic::AtomicI32::new(1),
            hwnd,
        })
    }

    /// 将 Box<Self> 转换为 IPlugFrame 原始指针。
    pub fn into_raw(self: Box<Self>) -> *mut IPlugFrame {
        Box::into_raw(self) as *mut IPlugFrame
    }
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn simple_plug_frame_query_interface(
    this: *mut FUnknown,
    iid: *const TUID,
    obj: *mut *mut c_void,
) -> tresult {
    if obj.is_null() {
        return K_INVALID_ARGUMENT;
    }

    // 检查是否请求 IPlugFrame 或 FUnknown
    let requested_iid = &*iid;
    let plug_frame_iid = &IPlugFrame_iid;
    let funknown_iid = &FUnknown_iid;

    if requested_iid == plug_frame_iid || requested_iid == funknown_iid {
        *obj = this as *mut c_void;
        simple_plug_frame_add_ref(this);
        kResultOk
    } else {
        *obj = ptr::null_mut();
        K_NO_INTERFACE
    }
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn simple_plug_frame_add_ref(this: *mut FUnknown) -> u32 {
    let frame = &*(this as *const SimplePlugFrame);
    frame
        .ref_count
        .fetch_add(1, std::sync::atomic::Ordering::SeqCst) as u32
        + 1
}

#[cfg(target_os = "windows")]
unsafe extern "system" fn simple_plug_frame_release(this: *mut FUnknown) -> u32 {
    let frame = &*(this as *const SimplePlugFrame);
    let prev = frame
        .ref_count
        .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    if prev <= 1 {
        // 引用计数归零，释放
        drop(Box::from_raw(this as *mut SimplePlugFrame));
        return 0;
    }
    (prev - 1) as u32
}

/// resizeView 回调：插件请求改变编辑器大小时，调整宿主窗口。
#[cfg(target_os = "windows")]
unsafe extern "system" fn simple_plug_frame_resize_view(
    this: *mut IPlugFrame,
    _view: *mut IPlugView,
    new_size: *mut ViewRect,
) -> tresult {
    if new_size.is_null() {
        return K_INVALID_ARGUMENT;
    }

    let frame = &*(this as *const SimplePlugFrame);
    let rect = &*new_size;
    let new_w = rect.right - rect.left;
    let new_h = rect.bottom - rect.top;

    eprintln!(
        "[vst3_com::IPlugFrame] resizeView requested: {}x{}",
        new_w, new_h
    );

    // 调整宿主窗口客户区大小
    // 使用与 gui.rs 中一致的 RECT 结构体定义（4 个 i32 字段）
    #[repr(C)]
    struct RECT {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    extern "system" {
        fn SetWindowPos(
            hWnd: *mut c_void,
            hWndInsertAfter: *mut c_void,
            X: i32,
            Y: i32,
            cx: i32,
            cy: i32,
            uFlags: u32,
        ) -> i32;
        fn AdjustWindowRectEx(
            lpRect: *mut RECT,
            dwStyle: u32,
            bMenu: i32,
            dwExStyle: u32,
        ) -> i32;
        fn GetWindowLongW(hWnd: *mut c_void, nIndex: i32) -> i32;
    }

    const GWL_STYLE: i32 = -16;
    const SWP_NOMOVE: u32 = 0x0002;
    const SWP_NOZORDER: u32 = 0x0004;

    let style = GetWindowLongW(frame.hwnd, GWL_STYLE) as u32;
    let mut adj_rect = RECT {
        left: 0,
        top: 0,
        right: new_w,
        bottom: new_h,
    };
    AdjustWindowRectEx(&mut adj_rect, style, 0, 0);
    let adj_w = adj_rect.right - adj_rect.left;
    let adj_h = adj_rect.bottom - adj_rect.top;

    SetWindowPos(
        frame.hwnd,
        ptr::null_mut(),
        0,
        0,
        adj_w,
        adj_h,
        SWP_NOMOVE | SWP_NOZORDER,
    );

    kResultOk
}

// ─── 辅助常量 ──
#[cfg(target_os = "windows")]
#[allow(non_upper_case_globals)]
const K_INVALID_ARGUMENT: tresult = -1; // Steinberg::kInvalidArgument
#[cfg(target_os = "windows")]
#[allow(non_upper_case_globals)]
const K_NO_INTERFACE: tresult = -2; // Steinberg::kNoInterface

/// VST3 插件实例的 COM 接口集合。
///
/// 持有一个 VST3 插件的所有必要 COM 接口指针，
/// 通过 `vst3` crate 的 `ComPtr` 管理引用计数。
pub struct Vst3Instance {
    /// 保持 DLL 加载状态。
    pub _lib: libloading::Library,
    /// IPluginFactory — 插件工厂。
    pub factory: ComPtr<IPluginFactory>,
    /// IComponent — 插件组件（初始化/状态/总线配置）。
    pub component: ComPtr<IComponent>,
    /// IAudioProcessor — 音频处理器。
    pub processor: ComPtr<IAudioProcessor>,
    /// IEditController — 编辑控制器（参数管理/UI 创建）。
    /// 部分插件的 IComponent 和 IEditController 是同一对象。
    pub controller: Option<ComPtr<IEditController>>,
    /// 当前打开的编辑器视图（持有引用以便窗口关闭时调用 removed()）。
    pub editor_view: Option<ComPtr<IPlugView>>,
    /// 当前关联的 IPlugFrame 原始指针（编辑器打开期间保持有效）。
    /// 通过 SimplePlugFrame 的引用计数管理生命周期。
    #[cfg(target_os = "windows")]
    pub plug_frame_ptr: Option<*mut IPlugFrame>,
    /// 是否已初始化并激活。
    pub active: bool,
    /// 已配置的采样率。
    pub configured_sample_rate: f64,
    /// 已配置的块大小。
    pub configured_block_size: i32,
}

// SAFETY: Vst3Instance 通过外部 Mutex 保护，同一时间只有一个线程访问。
unsafe impl Send for Vst3Instance {}

impl Vst3Instance {
    /// 从 DLL 路径加载 VST3 插件并完成 COM 初始化。
    ///
    /// 执行完整的 VST3 初始化流程：
    /// 1. 加载 DLL → GetPluginFactory
    /// 2. IPluginFactory::getClassInfo → 找到音频处理器组件
    /// 3. IPluginFactory::createInstance → IComponent
    /// 4. IComponent::initialize
    /// 5. QueryInterface → IAudioProcessor
    /// 6. IAudioProcessor::setupProcessing
    /// 7. IComponent::setActive(true)
    /// 8. IAudioProcessor::setProcessing(true)
    pub fn load(
        module_path: &std::path::Path,
        sample_rate: f64,
        block_size: i32,
    ) -> Result<(Self, String, String, u32, u32), String> {
        unsafe {
            // 1. 加载 DLL
            let lib = libloading::Library::new(module_path)
                .map_err(|e| format!("VST3 load library failed: {}", e))?;

            // 获取 GetPluginFactory 函数
            type GetFactoryFn = unsafe extern "system" fn() -> *mut c_void;
            let get_factory: libloading::Symbol<GetFactoryFn> = lib
                .get(b"GetPluginFactory\0")
                .map_err(|e| format!("VST3 GetPluginFactory not found: {}", e))?;

            let factory_raw = get_factory();
            if factory_raw.is_null() {
                return Err("VST3 GetPluginFactory returned null".to_string());
            }

            // 将 raw ptr 包装为 ComPtr<IPluginFactory>
            // GetPluginFactory 返回的指针已有一个引用计数，ComPtr::from_raw 接管所有权
            let factory: ComPtr<IPluginFactory> =
                ComPtr::from_raw(factory_raw as *mut IPluginFactory)
                    .ok_or("Failed to wrap IPluginFactory pointer")?;

            // 2. 遍历类信息，找到音频处理器组件
            let class_count = factory.countClasses();
            if class_count <= 0 {
                return Err("VST3 plugin factory has no classes".to_string());
            }

            let mut component_cid: Option<[i8; 16]> = None;
            let mut plugin_name = String::new();
            let mut plugin_vendor = String::new();

            for i in 0..class_count {
                let mut info: PClassInfo = std::mem::zeroed();
                if factory.getClassInfo(i, &mut info) == kResultOk {
                    // 检查 category 是否为 "Audio Module Class"
                    let cat = std::ffi::CStr::from_ptr(info.category.as_ptr())
                        .to_string_lossy()
                        .to_string();
                    if cat.contains("Audio Module Class") || cat.contains("Audio") || i == 0 {
                        component_cid = Some(info.cid);
                        plugin_name = std::ffi::CStr::from_ptr(info.name.as_ptr())
                            .to_string_lossy()
                            .to_string();
                        // 尝试 IPluginFactory2 获取更多信息
                        if let Some(factory2) = factory.cast::<IPluginFactory2>() {
                            let mut info2: PClassInfo2 = std::mem::zeroed();
                            if factory2.getClassInfo2(i, &mut info2) == kResultOk {
                                plugin_vendor = std::ffi::CStr::from_ptr(info2.vendor.as_ptr())
                                    .to_string_lossy()
                                    .to_string();
                                // 如果 factory2 有更好的名称，使用它
                                let name2 = std::ffi::CStr::from_ptr(info2.name.as_ptr())
                                    .to_string_lossy()
                                    .to_string();
                                if !name2.is_empty() {
                                    plugin_name = name2;
                                }
                            }
                        }
                        break;
                    }
                }
            }

            let component_cid = component_cid
                .ok_or("No audio processor component class found in VST3 plugin")?;

            // 3. 创建 IComponent 实例
            let mut component_raw: *mut c_void = ptr::null_mut();
            let cid_ptr = component_cid.as_ptr() as FIDString;
            let iid_ptr = IComponent_iid.as_ptr() as FIDString;

            let result = factory.createInstance(cid_ptr, iid_ptr, &mut component_raw);
            if result != kResultOk || component_raw.is_null() {
                return Err(format!(
                    "VST3 createInstance for IComponent failed: {}",
                    result
                ));
            }

            let component: ComPtr<IComponent> =
                ComPtr::from_raw(component_raw as *mut IComponent)
                    .ok_or("Failed to wrap IComponent pointer")?;

            // 4. IComponent::initialize（传入 null context — 简化宿主）
            let init_result = component.initialize(ptr::null_mut());
            if init_result != kResultOk {
                eprintln!(
                    "[vst3_com] IComponent::initialize returned {}, continuing...",
                    init_result
                );
            }

            // 5. QueryInterface → IAudioProcessor
            let processor: ComPtr<IAudioProcessor> = component
                .cast::<IAudioProcessor>()
                .ok_or("VST3 IComponent does not implement IAudioProcessor")?;

            // 6. 获取总线信息
            let num_audio_inputs = component.getBusCount(K_AUDIO, K_INPUT);
            let num_audio_outputs = component.getBusCount(K_AUDIO, K_OUTPUT);

            // 获取输入/输出通道数
            let mut num_in_channels: u32 = 2;
            let mut num_out_channels: u32 = 2;

            if num_audio_inputs > 0 {
                let mut bus_info: BusInfo = std::mem::zeroed();
                if component.getBusInfo(K_AUDIO, K_INPUT, 0, &mut bus_info) == kResultOk
                {
                    num_in_channels = bus_info.channelCount as u32;
                    // 激活输入总线
                    component.activateBus(K_AUDIO, K_INPUT, 0, 1);
                }
            }

            if num_audio_outputs > 0 {
                let mut bus_info: BusInfo = std::mem::zeroed();
                if component.getBusInfo(K_AUDIO, K_OUTPUT, 0, &mut bus_info) == kResultOk
                {
                    num_out_channels = bus_info.channelCount as u32;
                    // 激活输出总线
                    component.activateBus(K_AUDIO, K_OUTPUT, 0, 1);
                }
            }

            // 确保至少有 1 个通道
            num_in_channels = num_in_channels.max(1);
            num_out_channels = num_out_channels.max(1);

            // 7. setupProcessing
            let mut setup = ProcessSetup {
                processMode: K_REALTIME,
                symbolicSampleSize: K_SAMPLE_32,
                maxSamplesPerBlock: block_size,
                sampleRate: sample_rate,
            };
            let setup_result = processor.setupProcessing(&mut setup);
            if setup_result != kResultOk {
                eprintln!(
                    "[vst3_com] IAudioProcessor::setupProcessing returned {}, continuing...",
                    setup_result
                );
            }

            // 8. setActive(true)
            let active_result = component.setActive(1);
            if active_result != kResultOk {
                eprintln!(
                    "[vst3_com] IComponent::setActive(true) returned {}",
                    active_result
                );
            }

            // 9. setProcessing(true)
            let processing_result = processor.setProcessing(1);
            if processing_result != kResultOk {
                eprintln!(
                    "[vst3_com] IAudioProcessor::setProcessing(true) returned {}",
                    processing_result
                );
            }

            // 10. 尝试获取 IEditController
            // 注意：cast (QueryInterface) 成功意味着 controller 与 component 是同一对象，
            // 此时不应再次调用 initialize（已经在 IComponent 阶段初始化过了）。
            let controller = component.cast::<IEditController>();

            // 获取 PFactoryInfo 用于 vendor 回退
            if plugin_vendor.is_empty() {
                let mut factory_info: PFactoryInfo = std::mem::zeroed();
                if factory.getFactoryInfo(&mut factory_info) == kResultOk {
                    plugin_vendor = std::ffi::CStr::from_ptr(factory_info.vendor.as_ptr())
                        .to_string_lossy()
                        .to_string();
                }
            }

            let instance = Vst3Instance {
                _lib: lib,
                factory,
                component,
                processor,
                controller,
                editor_view: None,
                #[cfg(target_os = "windows")]
                plug_frame_ptr: None,
                active: true,
                configured_sample_rate: sample_rate,
                configured_block_size: block_size,
            };

            Ok((instance, plugin_name, plugin_vendor, num_in_channels, num_out_channels))
        }
    }

    /// 处理音频数据。
    ///
    /// 构造 ProcessData 并调用 IAudioProcessor::process。
    pub fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>]) {
        if !self.active {
            // 未激活时 passthrough
            for (out_ch, in_ch) in outputs.iter_mut().zip(inputs.iter()) {
                let len = out_ch.len().min(in_ch.len());
                out_ch[..len].copy_from_slice(&in_ch[..len]);
            }
            return;
        }

        let num_samples = outputs.first().map(|ch| ch.len()).unwrap_or(0);
        if num_samples == 0 {
            return;
        }

        unsafe {
            // 准备输入缓冲区
            let mut input_bufs: Vec<Vec<f32>> = inputs.to_vec();
            let mut input_ptrs: Vec<*mut f32> =
                input_bufs.iter_mut().map(|ch| ch.as_mut_ptr()).collect();

            // 准备输出缓冲区
            let mut output_bufs: Vec<Vec<f32>> =
                outputs.iter().map(|ch| vec![0.0f32; ch.len()]).collect();
            let mut output_ptrs: Vec<*mut f32> =
                output_bufs.iter_mut().map(|ch| ch.as_mut_ptr()).collect();

            // 构造 AudioBusBuffers
            let mut input_bus = AudioBusBuffers {
                numChannels: input_ptrs.len() as i32,
                silenceFlags: 0,
                __field0: std::mem::zeroed(),
            };
            // channelBuffers32 是联合体的第一个字段
            input_bus.__field0.channelBuffers32 = if input_ptrs.is_empty() {
                ptr::null_mut()
            } else {
                input_ptrs.as_mut_ptr()
            };

            let mut output_bus = AudioBusBuffers {
                numChannels: output_ptrs.len() as i32,
                silenceFlags: 0,
                __field0: std::mem::zeroed(),
            };
            output_bus.__field0.channelBuffers32 = if output_ptrs.is_empty() {
                ptr::null_mut()
            } else {
                output_ptrs.as_mut_ptr()
            };

            // 构造 ProcessData
            // 注意：inputs/outputs 字段类型是 *mut AudioBusBuffers（原始指针）
            let mut process_data = ProcessData {
                processMode: K_REALTIME,
                symbolicSampleSize: K_SAMPLE_32,
                numSamples: num_samples as i32,
                numInputs: if input_ptrs.is_empty() { 0 } else { 1 },
                numOutputs: if output_ptrs.is_empty() { 0 } else { 1 },
                inputs: &mut input_bus as *mut AudioBusBuffers,
                outputs: &mut output_bus as *mut AudioBusBuffers,
                inputParameterChanges: ptr::null_mut(),
                outputParameterChanges: ptr::null_mut(),
                inputEvents: ptr::null_mut(),
                outputEvents: ptr::null_mut(),
                processContext: ptr::null_mut(),
            };

            let result = self.processor.process(&mut process_data);
            if result != kResultOk {
                // 处理失败，passthrough
                for (out_ch, in_ch) in outputs.iter_mut().zip(inputs.iter()) {
                    let len = out_ch.len().min(in_ch.len());
                    out_ch[..len].copy_from_slice(&in_ch[..len]);
                }
                return;
            }

            // 复制输出
            for (out_ch, buf) in outputs.iter_mut().zip(output_bufs.iter()) {
                let len = out_ch.len().min(buf.len());
                out_ch[..len].copy_from_slice(&buf[..len]);
            }
        }
    }

    /// 重新配置采样率和块大小。
    ///
    /// 需要先 deactivate → setupProcessing → activate。
    pub fn reconfigure(&mut self, sample_rate: f64, block_size: i32) {
        if !self.active {
            return;
        }

        unsafe {
            // 停止处理
            let _ = self.processor.setProcessing(0);
            let _ = self.component.setActive(0);

            // 重新配置
            let mut setup = ProcessSetup {
                processMode: K_REALTIME,
                symbolicSampleSize: K_SAMPLE_32,
                maxSamplesPerBlock: block_size,
                sampleRate: sample_rate,
            };
            let _ = self.processor.setupProcessing(&mut setup);

            // 重新激活
            let _ = self.component.setActive(1);
            let _ = self.processor.setProcessing(1);

            self.configured_sample_rate = sample_rate;
            self.configured_block_size = block_size;
        }
    }

    /// 获取插件状态（chunk 数据）。
    ///
    /// 通过 IComponent::getState 将状态保存到内存流。
    pub fn get_state(&self) -> Option<Vec<u8>> {
        unsafe {
            let stream = MemoryStream::new();
            let stream_ptr = stream.into_raw();

            let result = self.component.getState(stream_ptr);
            if result == kResultOk {
                let stream_ref = &*(stream_ptr as *const MemoryStream);
                let data = stream_ref.get_data();
                // 释放 stream（减少引用计数）
                memory_stream_release(stream_ptr as *mut FUnknown);
                if data.is_empty() {
                    None
                } else {
                    Some(data)
                }
            } else {
                memory_stream_release(stream_ptr as *mut FUnknown);
                None
            }
        }
    }

    /// 恢复插件状态（chunk 数据）。
    ///
    /// 通过 IComponent::setState 从内存流恢复状态。
    pub fn set_state(&mut self, data: &[u8]) -> Result<(), String> {
        unsafe {
            let stream = MemoryStream::from_data(data.to_vec());
            let stream_ptr = stream.into_raw();

            let result = self.component.setState(stream_ptr);
            memory_stream_release(stream_ptr as *mut FUnknown);

            if result == kResultOk {
                // 也尝试恢复 controller 状态
                if let Some(ref controller) = self.controller {
                    let stream2 = MemoryStream::from_data(data.to_vec());
                    let stream2_ptr = stream2.into_raw();
                    let _ = controller.setComponentState(stream2_ptr);
                    memory_stream_release(stream2_ptr as *mut FUnknown);
                }
                Ok(())
            } else {
                Err(format!("VST3 setState failed: {}", result))
            }
        }
    }

    /// 获取参数快照。
    ///
    /// 通过 IEditController 遍历所有参数，获取其归一化值。
    pub fn get_params_snapshot(&self) -> std::collections::HashMap<u32, f32> {
        let mut params = std::collections::HashMap::new();

        if let Some(ref controller) = self.controller {
            unsafe {
                let count = controller.getParameterCount();
                for i in 0..count {
                    let mut info: ParameterInfo = std::mem::zeroed();
                    if controller.getParameterInfo(i, &mut info) == kResultOk {
                        let value = controller.getParamNormalized(info.id);
                        params.insert(info.id, value as f32);
                    }
                }
            }
        }

        params
    }

    /// 获取编辑器窗口推荐尺寸。
    ///
    /// 通过 IEditController::createView 获取 IPlugView，
    /// 然后调用 IPlugView::getSize。
    pub fn editor_size(&self) -> (u32, u32) {
        if let Some(ref controller) = self.controller {
            unsafe {
                // "editor" 是标准视图类型
                let editor_str = b"editor\0";
                let view_ptr =
                    controller.createView(editor_str.as_ptr() as FIDString);
                if !view_ptr.is_null() {
                    let view = ComPtr::<IPlugView>::from_raw(view_ptr);
                    if let Some(ref view) = view {
                        let mut rect: ViewRect = std::mem::zeroed();
                        if view.getSize(&mut rect) == kResultOk {
                            let w = (rect.right - rect.left).max(100) as u32;
                            let h = (rect.bottom - rect.top).max(100) as u32;
                            return (w, h);
                        }
                    }
                }
            }
        }
        (800, 600)
    }

    /// 将 VST3 编辑器 GUI 附着到指定窗口句柄。
    ///
    /// 在 Windows 上使用 kPlatformTypeHWND。
    /// 遵循 VST3 规范的完整流程：
    ///   1. createView("editor") → IPlugView
    ///   2. setFrame(IPlugFrame) — 许多插件无 frame 不渲染
    ///   3. attached(hwnd, "HWND")
    ///   4. onSize(rect) — 触发首次绘制
    ///
    /// 成功后将 IPlugView 保存到 `editor_view` 字段，以便后续调用 `detach_editor` 清理。
    #[cfg(target_os = "windows")]
    pub fn attach_editor(&mut self, hwnd: *mut c_void) -> Result<(), String> {
        if let Some(ref controller) = self.controller {
            unsafe {
                let editor_str = b"editor\0";
                let view_ptr =
                    controller.createView(editor_str.as_ptr() as FIDString);
                if view_ptr.is_null() {
                    return Err("VST3 plugin has no editor view".to_string());
                }

                let view = ComPtr::<IPlugView>::from_raw(view_ptr)
                    .ok_or("Failed to wrap IPlugView pointer")?;

                // 检查平台支持
                let platform_type = kPlatformTypeHWND;
                let supported = view.isPlatformTypeSupported(platform_type);
                if supported != kResultOk {
                    eprintln!(
                        "[vst3_com] isPlatformTypeSupported(HWND) returned {}, trying anyway...",
                        supported
                    );
                }

                // ── 步骤 1：创建并设置 IPlugFrame ──
                // VST3 规范要求 attached() 前必须 setFrame()，
                // 许多插件在没有 IPlugFrame 的情况下白屏不渲染。
                let frame = SimplePlugFrame::new(hwnd);
                let frame_ptr = frame.into_raw();
                let set_frame_result = view.setFrame(frame_ptr);
                if set_frame_result != kResultOk {
                    eprintln!(
                        "[vst3_com] IPlugView::setFrame returned {} (non-fatal, continuing)",
                        set_frame_result
                    );
                } else {
                    eprintln!("[vst3_com] IPlugFrame set successfully");
                }
                // 保存 frame 指针以保持生命周期
                self.plug_frame_ptr = Some(frame_ptr);

                // ── 步骤 2：附着到窗口 ──
                let attach_result = view.attached(hwnd, platform_type);
                if attach_result != kResultOk {
                    // 清理 frame
                    simple_plug_frame_release(frame_ptr as *mut FUnknown);
                    self.plug_frame_ptr = None;
                    return Err(format!(
                        "VST3 IPlugView::attached failed: {}",
                        attach_result
                    ));
                }

                eprintln!("[vst3_com] VST3 editor attached to HWND {:?}", hwnd);

                // ── 步骤 3：发送 onSize 通知触发首次渲染 ──
                // 许多 VST3 插件在 attached() 后需要一次 onSize() 才会开始绘制
                let mut rect: ViewRect = std::mem::zeroed();
                if view.getSize(&mut rect) == kResultOk {
                    let on_size_result = view.onSize(&mut rect);
                    eprintln!(
                        "[vst3_com] onSize({}x{}) returned {}",
                        rect.right - rect.left,
                        rect.bottom - rect.top,
                        on_size_result
                    );
                }

                // 保存 view 引用，以便窗口关闭时调用 removed()
                self.editor_view = Some(view);
                Ok(())
            }
        } else {
            Err("VST3 plugin has no IEditController".to_string())
        }
    }

    /// 将 VST3 编辑器 GUI 附着到指定窗口句柄（非 Windows 平台桩实现）。
    #[cfg(not(target_os = "windows"))]
    pub fn attach_editor(&mut self, _hwnd: *mut c_void) -> Result<(), String> {
        Err("VST3 editor GUI is currently only supported on Windows".to_string())
    }

    /// 从窗口分离 VST3 编辑器 GUI。
    ///
    /// 调用 `IPlugView::removed()` 通知插件编辑器已从窗口分离，
    /// 然后释放 IPlugView 和 IPlugFrame 引用。
    pub fn detach_editor(&mut self) {
        if let Some(view) = self.editor_view.take() {
            unsafe {
                // 先通知 view 移除
                let _ = view.removed();
                // 清除 frame（setFrame(null)）
                let _ = view.setFrame(ptr::null_mut());
            }
            eprintln!("[vst3_com] VST3 editor detached");
            // ComPtr drop 时自动 release view
        }

        // 释放 IPlugFrame
        #[cfg(target_os = "windows")]
        if let Some(frame_ptr) = self.plug_frame_ptr.take() {
            unsafe {
                simple_plug_frame_release(frame_ptr as *mut FUnknown);
            }
            eprintln!("[vst3_com] IPlugFrame released");
        }
    }
}

impl Drop for Vst3Instance {
    fn drop(&mut self) {
        // 先分离编辑器
        self.detach_editor();
        unsafe {
            if self.active {
                let _ = self.processor.setProcessing(0);
                let _ = self.component.setActive(0);
                self.active = false;
            }
            let _ = self.component.terminate();
        }
    }
}

// ─── 扫描辅助 ────────────────────────────────────────────────────────────────

/// 从 VST3 DLL 提取插件元数据（不保留实例）。
///
/// 加载 DLL → 获取 IPluginFactory → 读取类信息 → 释放。
/// 比完整初始化更轻量，适合扫描场景。
pub fn probe_vst3_metadata(
    module_path: &std::path::Path,
) -> Option<Vst3ProbeResult> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        probe_vst3_metadata_inner(module_path)
    }));

    match result {
        Ok(opt) => opt,
        Err(_) => {
            eprintln!(
                "[vst3_com] Plugin panicked during probe, skipping: {}",
                module_path.display()
            );
            None
        }
    }
}

/// 扫描结果。
pub struct Vst3ProbeResult {
    pub name: String,
    pub vendor: String,
    pub category: String,
    pub is_instrument: bool,
    pub num_inputs: u32,
    pub num_outputs: u32,
}

fn probe_vst3_metadata_inner(
    module_path: &std::path::Path,
) -> Option<Vst3ProbeResult> {
    unsafe {
        let lib = libloading::Library::new(module_path).ok()?;

        type GetFactoryFn = unsafe extern "system" fn() -> *mut c_void;
        let get_factory: libloading::Symbol<GetFactoryFn> =
            lib.get(b"GetPluginFactory\0").ok()?;

        let factory_raw = get_factory();
        if factory_raw.is_null() {
            return None;
        }

        let factory: ComPtr<IPluginFactory> =
            ComPtr::from_raw(factory_raw as *mut IPluginFactory)?;

        let class_count = factory.countClasses();
        if class_count <= 0 {
            return None;
        }

        let mut name = String::new();
        let mut vendor = String::new();
        let mut category = "Effect".to_string();
        let mut is_instrument = false;

        // 获取工厂信息
        let mut factory_info: PFactoryInfo = std::mem::zeroed();
        if factory.getFactoryInfo(&mut factory_info) == kResultOk {
            vendor = std::ffi::CStr::from_ptr(factory_info.vendor.as_ptr())
                .to_string_lossy()
                .to_string();
        }

        // 遍历类信息
        for i in 0..class_count {
            let mut info: PClassInfo = std::mem::zeroed();
            if factory.getClassInfo(i, &mut info) != kResultOk {
                continue;
            }

            let cat = std::ffi::CStr::from_ptr(info.category.as_ptr())
                .to_string_lossy()
                .to_string();

            // 查找第一个 Audio Module Class
            if !cat.contains("Audio Module Class") && !cat.contains("Audio") && i != 0 {
                continue;
            }

            name = std::ffi::CStr::from_ptr(info.name.as_ptr())
                .to_string_lossy()
                .to_string();

            // 尝试 IPluginFactory2 获取更详细信息
            if let Some(factory2) = factory.cast::<IPluginFactory2>() {
                let mut info2: PClassInfo2 = std::mem::zeroed();
                if factory2.getClassInfo2(i, &mut info2) == kResultOk {
                    let name2 = std::ffi::CStr::from_ptr(info2.name.as_ptr())
                        .to_string_lossy()
                        .to_string();
                    if !name2.is_empty() {
                        name = name2;
                    }
                    let vendor2 = std::ffi::CStr::from_ptr(info2.vendor.as_ptr())
                        .to_string_lossy()
                        .to_string();
                    if !vendor2.is_empty() {
                        vendor = vendor2;
                    }

                    let sub_cat = std::ffi::CStr::from_ptr(info2.subCategories.as_ptr())
                        .to_string_lossy()
                        .to_string();
                    if sub_cat.contains("Instrument") || sub_cat.contains("Synth") {
                        is_instrument = true;
                        category = "Synth".to_string();
                    } else if sub_cat.contains("Fx") {
                        category = "Effect".to_string();
                    } else if !sub_cat.is_empty() {
                        category = sub_cat.split('|').next().unwrap_or("Effect").to_string();
                    }
                }
            }

            // 尝试创建实例来获取通道信息（轻量尝试）
            let mut component_raw: *mut c_void = ptr::null_mut();
            let cid_ptr = info.cid.as_ptr() as FIDString;
            let iid_ptr = IComponent_iid.as_ptr() as FIDString;
            let mut num_inputs: u32 = 2;
            let mut num_outputs: u32 = 2;

            if factory.createInstance(cid_ptr, iid_ptr, &mut component_raw) == kResultOk
                && !component_raw.is_null()
            {
                if let Some(comp) =
                    ComPtr::<IComponent>::from_raw(component_raw as *mut IComponent)
                {
                    let _ = comp.initialize(ptr::null_mut());

                    // 读取总线通道数
                    if comp.getBusCount(K_AUDIO, K_INPUT) > 0 {
                        let mut bus_info: BusInfo = std::mem::zeroed();
                        if comp.getBusInfo(K_AUDIO, K_INPUT, 0, &mut bus_info) == kResultOk
                        {
                            num_inputs = (bus_info.channelCount as u32).max(1);
                        }
                    }
                    if comp.getBusCount(K_AUDIO, K_OUTPUT) > 0 {
                        let mut bus_info: BusInfo = std::mem::zeroed();
                        if comp.getBusInfo(K_AUDIO, K_OUTPUT, 0, &mut bus_info) == kResultOk
                        {
                            num_outputs = (bus_info.channelCount as u32).max(1);
                        }
                    }

                    let _ = comp.terminate();
                    // ComPtr drop 时自动 release
                }
            }

            return Some(Vst3ProbeResult {
                name,
                vendor,
                category,
                is_instrument,
                num_inputs,
                num_outputs,
            });
        }

        // 没有找到合适的类
        None
    }
}
