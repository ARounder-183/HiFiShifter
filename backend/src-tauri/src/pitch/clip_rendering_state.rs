//! Clip 渲染状态跟踪机制
//!
//! 支持 clip 级别的渲染状态跟踪，用于实现增量渲染和动态暂停/恢复机制。

use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicU32, Ordering},
    Mutex,
};

/// Clip 渲染状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ClipRenderingState {
    /// 等待渲染（尚未开始）
    #[allow(dead_code)]
    Pending,
    /// 正在渲染中
    Rendering,
    /// 渲染完成，可以播放
    Ready,
    /// 渲染失败
    Failed,
}

/// Clip 渲染状态信息
#[derive(Debug, Clone)]
pub struct ClipRenderingInfo {
    /// 当前渲染状态
    pub state: ClipRenderingState,
    /// 渲染进度（0.0 - 1.0）
    #[allow(dead_code)]
    pub progress: f32,
    /// 错误信息（如果状态为 Failed）
    #[allow(dead_code)]
    pub error: Option<String>,
    /// 渲染开始时间戳（用于超时检测）
    #[allow(dead_code)]
    pub start_time: Option<std::time::Instant>,
}

/// Clip 渲染状态管理器
pub struct ClipRenderingStateManager {
    /// Clip ID 到渲染状态的映射
    states: HashMap<String, ClipRenderingInfo>,
    /// 渲染超时时间（秒）
    #[allow(dead_code)]
    timeout_seconds: u32,
    /// 状态变更计数器（用于检测状态变化）
    change_counter: AtomicU32,
}

impl ClipRenderingStateManager {
    /// 创建新的状态管理器
    pub fn new(timeout_seconds: u32) -> Self {
        Self {
            states: HashMap::new(),
            timeout_seconds,
            change_counter: AtomicU32::new(0),
        }
    }

    /// 设置 clip 的渲染状态
    pub fn set_state(
        &mut self,
        clip_id: &str,
        state: ClipRenderingState,
        progress: f32,
        error: Option<String>,
    ) {
        let now = std::time::Instant::now();
        let start_time = match state {
            ClipRenderingState::Rendering => Some(now),
            _ => None,
        };

        let info = ClipRenderingInfo {
            state,
            progress: progress.clamp(0.0, 1.0),
            error,
            start_time,
        };

        self.states.insert(clip_id.to_string(), info);
        self.change_counter.fetch_add(1, Ordering::Relaxed);
    }

    /// 获取 clip 的渲染状态
    pub fn get_state(&self, clip_id: &str) -> Option<&ClipRenderingInfo> {
        self.states.get(clip_id)
    }

    /// 检查 clip 是否就绪（Ready 状态）
    #[allow(dead_code)]
    pub fn is_ready(&self, clip_id: &str) -> bool {
        self.states
            .get(clip_id)
            .map(|info| info.state == ClipRenderingState::Ready)
            .unwrap_or(false)
    }

    /// 检查 clip 是否正在渲染中（Rendering 状态）
    #[allow(dead_code)]
    pub fn is_rendering(&self, clip_id: &str) -> bool {
        self.states
            .get(clip_id)
            .map(|info| info.state == ClipRenderingState::Rendering)
            .unwrap_or(false)
    }

    /// 检查 clip 是否失败（Failed 状态）
    #[allow(dead_code)]
    pub fn is_failed(&self, clip_id: &str) -> bool {
        self.states
            .get(clip_id)
            .map(|info| info.state == ClipRenderingState::Failed)
            .unwrap_or(false)
    }

    /// 检查 clip 是否等待渲染（Pending 状态）
    #[allow(dead_code)]
    pub fn is_pending(&self, clip_id: &str) -> bool {
        self.states
            .get(clip_id)
            .map(|info| info.state == ClipRenderingState::Pending)
            .unwrap_or(false)
    }

    /// 获取所有渲染中的 clip ID 列表
    #[allow(dead_code)]
    pub fn get_rendering_clips(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, info)| info.state == ClipRenderingState::Rendering)
            .map(|(clip_id, _)| clip_id.clone())
            .collect()
    }

    /// 获取所有就绪的 clip ID 列表
    #[allow(dead_code)]
    pub fn get_ready_clips(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, info)| info.state == ClipRenderingState::Ready)
            .map(|(clip_id, _)| clip_id.clone())
            .collect()
    }

    /// 获取所有失败的 clip ID 列表
    #[allow(dead_code)]
    pub fn get_failed_clips(&self) -> Vec<String> {
        self.states
            .iter()
            .filter(|(_, info)| info.state == ClipRenderingState::Failed)
            .map(|(clip_id, _)| clip_id.clone())
            .collect()
    }

    /// 清理超时的渲染任务
    pub fn cleanup_timeouts(&mut self) -> Vec<String> {
        let now = std::time::Instant::now();
        let timeout_duration = std::time::Duration::from_secs(self.timeout_seconds as u64);

        let mut timed_out = Vec::new();

        for (clip_id, info) in &mut self.states {
            if info.state == ClipRenderingState::Rendering {
                if let Some(start_time) = info.start_time {
                    if now.duration_since(start_time) > timeout_duration {
                        info.state = ClipRenderingState::Failed;
                        info.error = Some("渲染超时".to_string());
                        timed_out.push(clip_id.clone());
                    }
                }
            }
        }

        if !timed_out.is_empty() {
            self.change_counter.fetch_add(1, Ordering::Relaxed);
        }

        timed_out
    }

    /// 获取状态变更计数器（用于检测状态变化）
    #[allow(dead_code)]
    pub fn get_change_counter(&self) -> u32 {
        self.change_counter.load(Ordering::Relaxed)
    }

    /// 清除指定 clip 的状态
    pub fn remove_state(&mut self, clip_id: &str) -> bool {
        if self.states.remove(clip_id).is_some() {
            self.change_counter.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// 清除所有状态
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.states.clear();
        self.change_counter.fetch_add(1, Ordering::Relaxed);
    }
}

/// 全局 clip 渲染状态管理器
static GLOBAL_CLIP_RENDERING_STATE: std::sync::OnceLock<Mutex<ClipRenderingStateManager>> =
    std::sync::OnceLock::new();

/// 获取全局 clip 渲染状态管理器
pub fn global_clip_rendering_state() -> &'static Mutex<ClipRenderingStateManager> {
    GLOBAL_CLIP_RENDERING_STATE.get_or_init(|| {
        Mutex::new(ClipRenderingStateManager::new(30)) // 默认30秒超时
    })
}

/// Clip 渲染状态事件（用于前端显示）
#[derive(Debug, Clone, serde::Serialize)]
pub struct ClipRenderingStateEvent {
    /// Clip ID
    pub clip_id: String,
    /// 渲染状态
    pub state: ClipRenderingState,
    /// 渲染进度（0.0 - 1.0）
    pub progress: f32,
    /// 错误信息（如果有）
    pub error: Option<String>,
}

impl ClipRenderingStateEvent {
    /// 创建新的状态事件
    #[allow(dead_code)]
    pub fn new(
        clip_id: String,
        state: ClipRenderingState,
        progress: f32,
        error: Option<String>,
    ) -> Self {
        Self {
            clip_id,
            state,
            progress: progress.clamp(0.0, 1.0),
            error,
        }
    }
}

/// 将 ClipRenderingState 转换为字符串表示
impl std::fmt::Display for ClipRenderingState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClipRenderingState::Pending => write!(f, "pending"),
            ClipRenderingState::Rendering => write!(f, "rendering"),
            ClipRenderingState::Ready => write!(f, "ready"),
            ClipRenderingState::Failed => write!(f, "failed"),
        }
    }
}
