use std::fs;
use std::path::Path;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct AppConfig {
    #[serde(default)]
    recent: Vec<String>,
}

/// 从 config dir 读取最近工程列表；读取失败时返回空列表。
pub fn load_recent(config_dir: &Path) -> Vec<String> {
    let path = config_dir.join("app_config.json");
    let Ok(data) = fs::read_to_string(&path) else {
        return Vec::new();
    };
    let Ok(cfg) = serde_json::from_str::<AppConfig>(&data) else {
        return Vec::new();
    };
    cfg.recent
}

/// 将最近工程列表写入 config dir；写入失败时静默忽略。
pub fn save_recent(config_dir: &Path, recent: &[String]) {
    let path = config_dir.join("app_config.json");
    let cfg = AppConfig {
        recent: recent.to_vec(),
    };
    if let Ok(data) = serde_json::to_string_pretty(&cfg) {
        let _ = fs::write(&path, data);
    }
}

// ─── 外部 Resampler 注册表持久化 ──────────────────────────────────────────────

const RESAMPLER_REGISTRY_FILE: &str = "resampler_registry.json";

/// 从 config dir 加载已注册的外部 Resampler 列表；读取失败时返回空注册表。
pub fn load_resampler_registry(config_dir: &Path) -> crate::state::ResamplerRegistry {
    let path = config_dir.join(RESAMPLER_REGISTRY_FILE);
    let Ok(data) = fs::read_to_string(&path) else {
        return crate::state::ResamplerRegistry::default();
    };
    let Ok(mut registry) = serde_json::from_str::<crate::state::ResamplerRegistry>(&data) else {
        eprintln!("[config] 解析 resampler_registry.json 失败，使用空注册表");
        return crate::state::ResamplerRegistry::default();
    };
    // 刷新可用性：检查 exe 文件是否仍然存在
    registry.refresh_availability();
    registry
}

/// 将外部 Resampler 注册表写入 config dir；写入失败时静默忽略。
pub fn save_resampler_registry(config_dir: &Path, registry: &crate::state::ResamplerRegistry) {
    let path = config_dir.join(RESAMPLER_REGISTRY_FILE);
    if let Ok(data) = serde_json::to_string_pretty(registry) {
        let _ = fs::write(&path, data);
    }
}
