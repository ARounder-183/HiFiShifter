//! 渲染耗时统计：**零成本**地回答「这台机器到底慢不慢」。
//!
//! 【为什么需要它】诊断包里的 `benchmark.json` 跑的是**合成**推理（1024 帧、5 轮、
//! 空 mel），它回答「ORT 在这台机器上能跑多快」。而用户抱怨的是「拖动卡」——
//! 那是**真实渲染**（真实 mel、真实缓存命中率、真实音频时长）的耗时。
//! 两者可以相差一个数量级，而且合成基准在移动端根本跑不起（见
//! `run_benchmark_smoke` 的文件头）。
//!
//! 后台渲染本来就在逐 clip 记录 `elapsed_ms`（只记"慢"的那几条）。这里把同一个
//! 数字顺手放进一个有界环形缓冲：**不做任何额外测量**，只是把已经算出来的值留下
//! 来。诊断包读它得到 p50 / p95，比合成基准更贴近用户体感。
//!
//! 【为什么是有界环形缓冲】渲染是长跑：一次工程可能渲染上千个 clip。留下全部会
//! 让诊断包随工程规模膨胀；留下最近 256 次足以给出稳定的分位数，且内存恒定。

use std::sync::Mutex;

/// 环形缓冲容量。256 次 ≈ 一个中等工程的完整渲染量。
const CAPACITY: usize = 256;

static SAMPLES_MS: Mutex<Vec<f64>> = Mutex::new(Vec::new());

/// 记录一次真实渲染耗时（毫秒）。超过容量的最旧样本被丢弃。
pub fn record_render_ms(elapsed_ms: f64) {
    if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
        return;
    }
    let Ok(mut samples) = SAMPLES_MS.lock() else {
        return;
    };
    if samples.len() >= CAPACITY {
        samples.remove(0);
    }
    samples.push(elapsed_ms);
}

/// 取一份分位数快照（诊断包用）。
///
/// @returns `samples` 为样本数；样本不足时 `p50` / `p95` 为 `null` —— 用 0 冒充
///   「很快」会误导排查。
pub fn snapshot() -> serde_json::Value {
    let samples: Vec<f64> = SAMPLES_MS
        .lock()
        .map(|guard| guard.clone())
        .unwrap_or_default();
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let percentile = |p: f64| -> Option<f64> {
        if sorted.is_empty() {
            return None;
        }
        let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted.get(index).copied()
    };
    serde_json::json!({
        "samples": sorted.len(),
        "capacity": CAPACITY,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": sorted.last().copied(),
    })
}

/// 仅测试用：清空样本。
#[cfg(test)]
pub fn reset_for_tests() {
    if let Ok(mut samples) = SAMPLES_MS.lock() {
        samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_ignore_empty_and_clamp_at_capacity() {
        reset_for_tests();
        let empty = snapshot();
        assert_eq!(empty["samples"], 0);
        assert!(empty["p50"].is_null());

        for value in 1..=300 {
            record_render_ms(value as f64);
        }
        let filled = snapshot();
        // 容量是 256：只留最近 256 个（45..=300）。
        assert_eq!(filled["samples"], CAPACITY);
        assert_eq!(filled["max"], 300.0);
        // 分位数单调。
        let p50 = filled["p50"].as_f64().unwrap();
        let p95 = filled["p95"].as_f64().unwrap();
        assert!(p50 <= p95);
        assert!((45.0..=300.0).contains(&p50));
    }

    #[test]
    fn invalid_samples_are_dropped() {
        reset_for_tests();
        record_render_ms(f64::NAN);
        record_render_ms(-1.0);
        record_render_ms(f64::INFINITY);
        assert_eq!(snapshot()["samples"], 0);
    }
}
