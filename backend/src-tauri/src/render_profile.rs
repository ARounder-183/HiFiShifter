/*
 * render_profile.rs - 渲染耗时画像：把"推理耗时"从"总渲染耗时"里分离出来。
 *
 * 目的（回答 §7 阻塞级开放问题 1）：
 * `f_infer = 推理耗时 / 总渲染耗时` 是 **P1-2 收益的唯一决定变量**
 * （加速比 `S = 1 / (f_infer + (1 - f_infer) / W)`）。在拿到这个数字之前，
 * "并行化推理"到底是高收益还是白费力气无法判断；而方案本身也写明
 * "若 `f_infer > 60%`，应下调 P1-2 优先级、上调 P1-1"。
 *
 * 为什么需要新增计数器：渲染循环原本已经分别统计了
 * `cache_probe / render / tension / total`（见 `playback.rs` 的
 * `[bg_render][cache] DONE` 日志），但 `render` 把**推理**与处理器链
 * （共振峰 / 拉伸 / 重采样等 CPU 工作）混在一起，无法据此得到 `f_infer`。
 * 本模块只补上缺失的那一项：ORT 会话执行时长。
 *
 * 计数方式与线程安全：
 * - 只在**每次 `session.run`** 时累加（按 chunk / 按 clip，不是按样本），
 *   因此即使被实时线程调用也不会引入可测量的开销；
 * - 内存序一律 `Relaxed`：这是统计量，不参与任何数据同步，不需要
 *   happens-before 关系；
 * - 进程级全局：一轮渲染的主循环在单线程上跑，跨轮次用 `reset()` 归零。
 */

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

/// 本轮累计的 ORT 会话执行时长（微秒）。u64 微秒可表示约 58 万年，不会溢出。
static INFERENCE_MICROS: AtomicU64 = AtomicU64::new(0);

/// 本轮累计的 `session.run` 次数（用于区分"单次很慢"与"次数很多"）。
static INFERENCE_RUNS: AtomicU64 = AtomicU64::new(0);

/// 本轮进入 `CHUNK_MAX_FRAMES` 分块路径的 clip 数。
///
/// 采集它的目的（回答 §7 非阻塞问题 5 / 支撑 P0-4 的取舍）：P0-4「取消检查点
/// 下沉到 chunk 边界」的收益完全取决于**有多少 clip 真的被切成多个分块**。
/// 此前该结论是**推断**出来的（`CHUNK_MAX_FRAMES = 4096` 帧 ≈ 47.5 s，而
/// README 推荐短切片工作流 → 多数 clip 应只有 1 块）。有了这个计数，"多数如此"
/// 就从推断变成可核验的事实，取舍不必再靠论证站住。
static CHUNKED_PATH_CLIPS: AtomicU64 = AtomicU64::new(0);

/// 上述 clip 中分块数 > 1 的数量。它 > 0 时，P0-4 才可能有收益。
static CLIPS_WITH_MULTIPLE_CHUNKS: AtomicU64 = AtomicU64::new(0);

/// 本轮分块总数（含每块的重推理）。
static CHUNKS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// 最近一次完整渲染轮次的画像，供诊断命令读取（无需解析日志）。
static LAST_PASS: Mutex<Option<RenderPassProfile>> = Mutex::new(None);

#[derive(Debug, Clone, Copy, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderPassProfile {
    /// 本轮 ORT 推理总耗时（毫秒）。
    pub inference_ms: f64,
    /// 本轮 `session.run` 调用次数。
    pub inference_runs: u64,
    /// 本轮总耗时（毫秒，由渲染循环提供的分母）。
    pub total_ms: f64,
    /// `inference_ms / total_ms`，即方案中的 `f_infer`。
    pub inference_fraction: f64,
    /// 进入 `CHUNK_MAX_FRAMES` 分块路径的 clip 数。
    pub chunked_path_clips: u64,
    /// 其中被切成多个分块的 clip 数（P0-4 收益的必要条件）。
    pub clips_with_multiple_chunks: u64,
    /// 分块总数。
    pub chunks_total: u64,
    /// `clips_with_multiple_chunks / chunked_path_clips`；无样本时为 0。
    pub multi_chunk_fraction: f64,
}

/// 记录一次 ORT 会话执行耗时。**只应在真正的 `session.run` 调用点使用。**
///
/// 无 onnx 构建下没有 ORT 会话，因此没有任何调用点 —— 这是结构性的，不是遗漏
/// （该构建里 `f_infer` 恒为 0，画像仍可读，只是分子始终为 0）。
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) fn record_inference(elapsed: Duration) {
    INFERENCE_MICROS.fetch_add(elapsed.as_micros() as u64, Ordering::Relaxed);
    INFERENCE_RUNS.fetch_add(1, Ordering::Relaxed);
}

/// 记录一次分块路径调用的分块数（每个 clip 一次）。
///
/// 只应由 `nsf_hifigan_onnx` 的 `CHUNK_MAX_FRAMES` 路径调用 —— mel-stretch 路径
/// 按秒分块、由另一个常量控制，混在一起会让统计失去意义。
///
/// 无 onnx 构建下没有声码器分块路径，因此没有调用点（结构性，非遗漏）。
#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
pub(crate) fn record_chunking(chunks: usize) {
    CHUNKED_PATH_CLIPS.fetch_add(1, Ordering::Relaxed);
    CHUNKS_TOTAL.fetch_add(chunks as u64, Ordering::Relaxed);
    if chunks > 1 {
        CLIPS_WITH_MULTIPLE_CHUNKS.fetch_add(1, Ordering::Relaxed);
    }
}

/// 开始新一轮渲染：归零累计器。
///
/// 必须在渲染主循环入口调用，否则上一轮（可能因取消而中止）的残留会污染本轮
/// 的 `f_infer` —— 分母是本轮的、分子却含上一轮的，比值会虚假偏高。
pub(crate) fn reset() {
    INFERENCE_MICROS.store(0, Ordering::Relaxed);
    INFERENCE_RUNS.store(0, Ordering::Relaxed);
    CHUNKED_PATH_CLIPS.store(0, Ordering::Relaxed);
    CLIPS_WITH_MULTIPLE_CHUNKS.store(0, Ordering::Relaxed);
    CHUNKS_TOTAL.store(0, Ordering::Relaxed);
}

/// 一轮渲染结束：用渲染循环提供的总耗时定格画像，并返回它。
///
/// `total` 由调用方给出而非本模块自行计时 —— 本模块只看得到推理，看不到
/// 本轮从开始到结束的墙钟时间（其中包含快照构建、缓存探测、取消检查等）。
pub(crate) fn finish_pass(total: Duration) -> RenderPassProfile {
    let inference_us = INFERENCE_MICROS.load(Ordering::Relaxed);
    let total_ms = total.as_secs_f64() * 1000.0;
    let inference_ms = inference_us as f64 / 1000.0;

    let chunked_clips = CHUNKED_PATH_CLIPS.load(Ordering::Relaxed);
    let multi_chunk_clips = CLIPS_WITH_MULTIPLE_CHUNKS.load(Ordering::Relaxed);

    let profile = RenderPassProfile {
        inference_ms,
        inference_runs: INFERENCE_RUNS.load(Ordering::Relaxed),
        total_ms,
        // 总耗时为 0（极短渲染 / 测试）时比值无意义，按 0 处理而不是 NaN。
        inference_fraction: if total_ms > 0.0 {
            (inference_ms / total_ms).clamp(0.0, 1.0)
        } else {
            0.0
        },
        chunked_path_clips: chunked_clips,
        clips_with_multiple_chunks: multi_chunk_clips,
        chunks_total: CHUNKS_TOTAL.load(Ordering::Relaxed),
        multi_chunk_fraction: if chunked_clips > 0 {
            multi_chunk_clips as f64 / chunked_clips as f64
        } else {
            0.0
        },
    };

    if let Ok(mut last) = LAST_PASS.lock() {
        *last = Some(profile);
    }
    profile
}

/// 最近一次完整轮次的画像（若从未完成过一轮则为 `None`）。
pub(crate) fn last_pass() -> Option<RenderPassProfile> {
    LAST_PASS.lock().ok().and_then(|l| *l)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 分块统计：区分"全部单块"与"存在多块" —— 前者直接否掉 P0-4 的收益前提。
    #[test]
    fn chunking_statistics_distinguish_single_from_multi_chunk() {
        reset();
        // 3 个 clip：两个单块、一个两块。
        record_chunking(1);
        record_chunking(2);
        record_chunking(1);

        let profile = finish_pass(Duration::from_millis(100));
        assert_eq!(profile.chunked_path_clips, 3);
        assert_eq!(profile.clips_with_multiple_chunks, 1);
        assert_eq!(profile.chunks_total, 4);
        assert!((profile.multi_chunk_fraction - 1.0 / 3.0).abs() < 1e-9);

        // 全部单块 → 比例为 0 → P0-4 无收益（这正是本计数的用途）。
        reset();
        record_chunking(1);
        record_chunking(1);
        let profile = finish_pass(Duration::from_millis(100));
        assert_eq!(profile.clips_with_multiple_chunks, 0);
        assert_eq!(profile.multi_chunk_fraction, 0.0);
    }

    /// 累计 → 定格 → 比值 的完整链路，以及总耗时为 0 时的降级。
    #[test]
    fn accumulates_and_reports_fraction() {
        // 本模块是进程级全局，用例内部自洽地 reset 即可（不依赖其它用例状态）。
        reset();
        record_inference(Duration::from_millis(300));
        record_inference(Duration::from_millis(100));

        let profile = finish_pass(Duration::from_millis(1000));
        assert_eq!(profile.inference_runs, 2);
        assert!((profile.inference_ms - 400.0).abs() < 1e-6, "got {}", profile.inference_ms);
        assert!(
            (profile.inference_fraction - 0.4).abs() < 1e-9,
            "f_infer should be 0.4, got {}",
            profile.inference_fraction
        );
        assert_eq!(last_pass().map(|p| p.inference_runs), Some(2));
    }

    /// 总耗时为 0 时不得产生 NaN（极短渲染 / 测试路径）。
    #[test]
    fn zero_total_does_not_produce_nan() {
        reset();
        record_inference(Duration::from_micros(500));
        let profile = finish_pass(Duration::ZERO);
        assert!(profile.inference_fraction.is_finite());
        assert_eq!(profile.inference_fraction, 0.0);
    }

    /// reset 必须真正归零 —— 否则被取消的上一轮会抬高本轮的 f_infer。
    #[test]
    fn reset_clears_accumulators() {
        reset();
        record_inference(Duration::from_millis(50));
        reset();
        let profile = finish_pass(Duration::from_millis(100));
        assert_eq!(profile.inference_ms, 0.0);
        assert_eq!(profile.inference_runs, 0);
    }

    /// 比值被钳制在 [0,1]：若调用方给的分母小于实际推理耗时（例如取消发生在
    /// 计时的边界上），不得报告 >100% 的占比。
    #[test]
    fn fraction_is_clamped() {
        reset();
        record_inference(Duration::from_millis(500));
        let profile = finish_pass(Duration::from_millis(100));
        assert!(profile.inference_fraction <= 1.0);
    }
}
