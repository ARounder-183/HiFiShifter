//! 单个 VST 插件实例的统一封装。
//!
//! `VstPluginInstance` 将 VST2 和 VST3 后端统一封装为一致的音频处理接口，
//! 支持 process（音频处理）、参数访问、chunk 序列化、GUI 窗口打开等操作。

use std::collections::HashMap;
use std::path::PathBuf;

use super::VstFormat;

// ─── VST2 Host 回调 ────────────────────────────────────────────────────────

/// 简化的 VST2 Host 回调实现。
#[cfg(feature = "vst")]
pub struct SimpleVst2Host;

#[cfg(feature = "vst")]
impl vst2::host::Host for SimpleVst2Host {
    fn automate(&self, _index: i32, _value: f32) {
        // 参数自动化回调：当插件内部修改了参数时调用
    }

    fn get_plugin_id(&self) -> i32 {
        0
    }
}

// ─── VST 插件后端 ──────────────────────────────────────────────────────────

/// VST 插件后端，区分 VST2 和 VST3 的具体实现。
pub enum VstPluginBackend {
    /// VST2 后端（使用 `vst2` crate — alias of `vst`）。
    #[cfg(feature = "vst")]
    Vst2 {
        plugin: Box<dyn vst2::plugin::Plugin>,
        _host: std::sync::Arc<std::sync::Mutex<SimpleVst2Host>>,
    },

    /// VST3 后端（使用 `libloading` 直接 FFI）。
    #[cfg(feature = "vst")]
    Vst3 {
        _lib: libloading::Library,
        factory_ptr: *mut std::ffi::c_void,
        component_ptr: *mut std::ffi::c_void,
        processor_ptr: *mut std::ffi::c_void,
        initialized: bool,
    },

    /// 桩后端（VST feature 未启用时使用）。
    #[cfg(not(feature = "vst"))]
    Stub,
}

// SAFETY: VST 插件实例在单线程中处理音频，
// 我们通过 Mutex 确保同一时间只有一个线程访问实例。
unsafe impl Send for VstPluginBackend {}

// ─── 统一插件实例 ──────────────────────────────────────────────────────────

/// 统一的 VST 插件实例封装。
pub struct VstPluginInstance {
    /// 插件显示名称。
    pub name: String,
    /// 插件厂商名称。
    pub vendor: String,
    /// 插件格式。
    pub format: VstFormat,
    /// 插件文件路径。
    pub path: PathBuf,
    /// 当前采样率。
    pub sample_rate: f32,
    /// 当前块大小。
    pub block_size: usize,
    /// 输入通道数。
    pub num_inputs: u32,
    /// 输出通道数。
    pub num_outputs: u32,
    /// 是否旁通。
    pub bypassed: bool,
    /// 后端实现。
    pub backend: VstPluginBackend,
}

impl VstPluginInstance {
    /// 处理音频数据。
    ///
    /// `inputs`: 输入通道的 PCM 缓冲区切片（每个通道一个 Vec）。
    /// `outputs`: 输出通道的 PCM 缓冲区切片。
    pub fn process(&mut self, inputs: &[Vec<f32>], outputs: &mut [Vec<f32>]) {
        if self.bypassed {
            for (out_ch, in_ch) in outputs.iter_mut().zip(inputs.iter()) {
                let len = out_ch.len().min(in_ch.len());
                out_ch[..len].copy_from_slice(&in_ch[..len]);
            }
            return;
        }

        match &mut self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                let num_samples = outputs.first().map(|ch| ch.len()).unwrap_or(0);
                if num_samples == 0 {
                    return;
                }

                // 准备缓冲区
                let mut input_bufs: Vec<Vec<f32>> = inputs.to_vec();
                let mut output_bufs: Vec<Vec<f32>> = outputs
                    .iter()
                    .map(|ch| vec![0.0f32; ch.len()])
                    .collect();

                let mut in_ptrs: Vec<*mut f32> =
                    input_bufs.iter_mut().map(|ch| ch.as_mut_ptr()).collect();
                let mut out_ptrs: Vec<*mut f32> =
                    output_bufs.iter_mut().map(|ch| ch.as_mut_ptr()).collect();

                let buffer = unsafe {
                    vst2::buffer::AudioBuffer::from_raw(
                        input_bufs.len(),
                        output_bufs.len(),
                        in_ptrs.as_mut_ptr(),
                        out_ptrs.as_mut_ptr(),
                        num_samples,
                    )
                };
                plugin.process(&buffer);

                // 复制输出
                for (out_ch, buf) in outputs.iter_mut().zip(output_bufs.iter()) {
                    let len = out_ch.len().min(buf.len());
                    out_ch[..len].copy_from_slice(&buf[..len]);
                }
            }

            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => {
                // VST3 处理：暂时 passthrough
                for (out_ch, in_ch) in outputs.iter_mut().zip(inputs.iter()) {
                    let len = out_ch.len().min(in_ch.len());
                    out_ch[..len].copy_from_slice(&in_ch[..len]);
                }
            }

            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {
                for (out_ch, in_ch) in outputs.iter_mut().zip(inputs.iter()) {
                    let len = out_ch.len().min(in_ch.len());
                    out_ch[..len].copy_from_slice(&in_ch[..len]);
                }
            }
        }
    }

    /// 设置采样率。
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        match &mut self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                plugin.suspend();
                plugin.set_sample_rate(sample_rate);
                plugin.resume();
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => {}
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {}
        }
    }

    /// 设置块大小。
    pub fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size;
        match &mut self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                plugin.suspend();
                plugin.set_block_size(block_size as i64);
                plugin.resume();
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => {}
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {}
        }
    }

    /// 获取插件 chunk 数据（预设序列化）。
    pub fn get_chunk(&self) -> Option<String> {
        match &self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                // vst crate 0.4 使用 get_bank_data 返回 Vec<u8>
                let data = plugin.get_parameter_object().get_bank_data();
                data.map(|bytes| {
                    use base64::Engine;
                    base64::engine::general_purpose::STANDARD.encode(&bytes)
                })
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => None,
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => None,
        }
    }

    /// 从 base64 编码的 chunk 数据恢复插件状态。
    pub fn set_chunk(&mut self, chunk_base64: &str) -> Result<(), String> {
        use base64::Engine;
        let data = base64::engine::general_purpose::STANDARD
            .decode(chunk_base64)
            .map_err(|e| format!("Base64 decode failed: {}", e))?;

        match &mut self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                plugin.get_parameter_object().load_bank_data(&data);
                Ok(())
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => {
                Err("VST3 chunk restore not implemented yet".to_string())
            }
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {
                let _ = data;
                Err("VST feature is not enabled".to_string())
            }
        }
    }

    /// 获取参数快照。
    pub fn get_params_snapshot(&self) -> HashMap<u32, f32> {
        let mut params = HashMap::new();
        match &self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                let info = plugin.get_info();
                for i in 0..info.parameters {
                    params.insert(i as u32, plugin.get_parameter(i));
                }
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => {}
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {}
        }
        params
    }

    /// 获取编辑器窗口推荐尺寸。
    ///
    /// VST2 插件尝试从 editor 获取实际尺寸，没有 editor 或获取失败时返回默认值。
    pub fn editor_size(&self) -> (u32, u32) {
        match &self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                if let Some(editor) = plugin.get_editor() {
                    let (w, h) = editor.size();
                    if w > 0 && h > 0 {
                        return (w as u32, h as u32);
                    }
                }
                (800, 600)
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 { .. } => (800, 600),
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => (800, 600),
        }
    }
}

impl Drop for VstPluginInstance {
    fn drop(&mut self) {
        match &mut self.backend {
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst2 { plugin, .. } => {
                plugin.suspend();
            }
            #[cfg(feature = "vst")]
            VstPluginBackend::Vst3 {
                initialized, ..
            } => {
                if *initialized {
                    *initialized = false;
                }
            }
            #[cfg(not(feature = "vst"))]
            VstPluginBackend::Stub => {}
        }
    }
}
