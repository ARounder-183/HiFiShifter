//! VST 插件宿主模块
//!
//! 提供 VST2 (`.dll`) 和 VST3 (`.vst3`) 插件的加载、管理、扫描和 GUI 窗口支持。
//! 通过 `VstProcessingStage` 将 VST 效果器无缝接入 `ProcessorChain`。
//!
//! 模块结构：
//! - `plugin_host`: VST2/VST3 插件加载与卸载
//! - `plugin_instance`: 单个插件实例封装
//! - `scanner`: 插件扫描器（搜索系统 VST 目录）
//! - `gui`: 原生窗口创建（Windows HWND / macOS NSView）
//! - `stage`: `VstProcessingStage` 实现 `ProcessingStage` trait

pub mod gui;
pub mod plugin_host;
pub mod plugin_instance;
pub mod scanner;
pub mod stage;
#[cfg(feature = "vst")]
pub mod vst3_com;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};

use serde::{Deserialize, Serialize};

// ─── 插件格式枚举 ───────────────────────────────────────────────────────────

/// VST 插件格式。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VstFormat {
    /// VST2 (classic `.dll` / `.vst`)
    Vst2,
    /// VST3 (`.vst3` bundle)
    Vst3,
}

// ─── 插件描述符（扫描结果） ──────────────────────────────────────────────────

/// 扫描到的 VST 插件描述信息（不含运行时实例）。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VstPluginDescriptor {
    /// 插件唯一标识符（SHA-256 of path + name）。
    pub uid: String,
    /// 插件显示名称。
    pub name: String,
    /// 插件厂商名称。
    pub vendor: String,
    /// 插件格式（VST2 / VST3）。
    pub format: VstFormat,
    /// 插件文件路径。
    pub path: PathBuf,
    /// 插件类别（Effect / Instrument / ...）。
    pub category: String,
    /// 是否为乐器插件（VSTi）。
    pub is_instrument: bool,
    /// 插件版本字符串。
    pub version: String,
    /// 输入通道数。
    pub num_inputs: u32,
    /// 输出通道数。
    pub num_outputs: u32,
}

// ─── 插件实例状态（序列化用） ────────────────────────────────────────────────

/// 单个 VST 插件实例在轨道 FX 链中的持久化状态。
/// 用于项目保存/加载时的序列化。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VstPluginState {
    /// 引用的插件 UID（对应 `VstPluginDescriptor::uid`）。
    pub plugin_uid: String,
    /// 插件路径（冗余保存，用于跨机器恢复）。
    pub plugin_path: PathBuf,
    /// 插件格式。
    pub format: VstFormat,
    /// 插件 chunk 数据（base64 编码的预设状态）。
    pub chunk_data: Option<String>,
    /// 是否被旁通（bypass）。
    pub bypassed: bool,
    /// 插件参数快照（参数 index → 值），作为 chunk 不可用时的备选。
    pub params: HashMap<u32, f32>,
}

// ─── 轨道 FX 链配置 ─────────────────────────────────────────────────────────

/// 每条轨道的 VST FX 链配置（持久化）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VstChainConfig {
    /// 有序的插件列表（按信号流顺序）。
    pub plugins: Vec<VstPluginState>,
}

// ─── 全局插件注册表 ─────────────────────────────────────────────────────────

/// 全局 VST 插件注册表，管理已扫描的插件列表和运行时实例。
pub struct VstPluginRegistry {
    /// 已扫描的插件描述符列表。
    pub descriptors: RwLock<Vec<VstPluginDescriptor>>,
    /// 活跃的插件实例池（key = 实例 ID）。
    pub instances: Mutex<HashMap<String, Arc<Mutex<plugin_instance::VstPluginInstance>>>>,
    /// 打开的编辑器窗口（key = 实例 ID）。
    pub editor_windows: Mutex<HashMap<String, gui::VstEditorWindow>>,
    /// 自定义 VST 扫描路径（用户配置）。
    pub custom_scan_paths: RwLock<Vec<PathBuf>>,
    /// 扫描状态。
    pub scan_in_progress: std::sync::atomic::AtomicBool,
}

impl Default for VstPluginRegistry {
    fn default() -> Self {
        Self {
            descriptors: RwLock::new(Vec::new()),
            instances: Mutex::new(HashMap::new()),
            editor_windows: Mutex::new(HashMap::new()),
            custom_scan_paths: RwLock::new(Vec::new()),
            scan_in_progress: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

impl VstPluginRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 通过 UID 查找插件描述符。
    pub fn find_descriptor(&self, uid: &str) -> Option<VstPluginDescriptor> {
        let descs = self.descriptors.read().unwrap_or_else(|e| e.into_inner());
        descs.iter().find(|d| d.uid == uid).cloned()
    }

    /// 列出所有已扫描的效果器插件（排除乐器插件）。
    pub fn list_effects(&self) -> Vec<VstPluginDescriptor> {
        let descs = self.descriptors.read().unwrap_or_else(|e| e.into_inner());
        descs.iter().filter(|d| !d.is_instrument).cloned().collect()
    }

    /// 列出所有已扫描插件。
    pub fn list_all(&self) -> Vec<VstPluginDescriptor> {
        let descs = self.descriptors.read().unwrap_or_else(|e| e.into_inner());
        descs.clone()
    }

    /// 注销一个插件实例（关闭时调用）。
    pub fn remove_instance(&self, instance_id: &str) {
        let mut instances = self.instances.lock().unwrap_or_else(|e| e.into_inner());
        instances.remove(instance_id);
    }
}
