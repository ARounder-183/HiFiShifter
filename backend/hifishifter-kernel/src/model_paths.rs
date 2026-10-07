//! ONNX 模型文件的位置（进程级、只写一次）。
//!
//! 【为什么在内核里】声码器与 FCPE 都在内核侧，它们启动时要按路径去加载模型；
//! 而"模型装在哪"是**宿主**才知道的事 —— 打包后的 app 从 `resource_dir` 里找，
//! 开发时从工作目录找，将来的插件则是另一套规则。
//!
//! 所以这里只提供**一个只写一次的登记处**：宿主在启动时把路径登记进来，
//! 内核在用时读。这样内核既不需要认识 Tauri 的 `path()`，也不需要在编译期知道
//! 任何目录布局。

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static NSF_HIFIGAN_MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
static HNSEP_MODEL_DIR: OnceLock<PathBuf> = OnceLock::new();
static FCPE_ONNX_PATH: OnceLock<PathBuf> = OnceLock::new();

/// NSF-HiFiGAN 模型目录（含 `pc_nsf_hifigan.onnx` 与 `config.json`）。
pub fn nsf_hifigan_model_dir() -> Option<&'static Path> {
    NSF_HIFIGAN_MODEL_DIR.get().map(|p| p.as_path())
}

/// hnsep（气声分离）模型目录。
pub fn hnsep_model_dir() -> Option<&'static Path> {
    HNSEP_MODEL_DIR.get().map(|p| p.as_path())
}

/// FCPE 音高检测模型文件路径。
pub fn fcpe_onnx_path() -> Option<&'static Path> {
    FCPE_ONNX_PATH.get().map(|p| p.as_path())
}

/// 登记 NSF-HiFiGAN 模型目录。返回 `false` 表示已经登记过（**不覆盖**）。
pub fn set_nsf_hifigan_model_dir(path: PathBuf) -> bool {
    NSF_HIFIGAN_MODEL_DIR.set(path).is_ok()
}

/// 登记 hnsep 模型目录。返回 `false` 表示已经登记过（**不覆盖**）。
pub fn set_hnsep_model_dir(path: PathBuf) -> bool {
    HNSEP_MODEL_DIR.set(path).is_ok()
}

/// 登记 FCPE 模型路径。返回 `false` 表示已经登记过（**不覆盖**）。
pub fn set_fcpe_onnx_path(path: PathBuf) -> bool {
    FCPE_ONNX_PATH.set(path).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 登记之后必须读得回来。
    #[test]
    fn a_registered_path_reads_back() {
        // 这三个登记处是**进程级单例**，而测试是并行跑的：用只写一次不覆盖的语义，
        // 就不会因为并行而互相踩（先登记者胜，后登记者返回 false）。
        let p = PathBuf::from("C:/models/fcpe/fcpe.onnx");
        let first = set_fcpe_onnx_path(p.clone());
        assert_eq!(fcpe_onnx_path(), Some(p.as_path()));
        // 第二次登记必须失败且不覆盖。
        let second = set_fcpe_onnx_path(PathBuf::from("C:/other.onnx"));
        assert_ne!(first, second);
        assert_eq!(fcpe_onnx_path(), Some(p.as_path()));
    }
}
