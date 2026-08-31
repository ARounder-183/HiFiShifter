/*
 * formant_morph/analysis.rs - Clip 源共振峰统计（供前端可视化）。
 *
 * 主要内容：
 * - FormantAnalysisSummary：统计 F1/F2（检出帧中位数）+ 稀疏轨迹 +
 *   浊音帧占比 + 诊断消息。
 * - analyze_clip_formants：mono PCM → 分析域降采样 → LPC → 候选筛选 →
 *   统计。**与 DSP 主流程共用同一套 lpc/track 代码**，保证
 *   "UI 显示的位置 == 算法实际认定的位置"（否则诊断会误导）。
 *
 * 与其他模块的关系：
 * - 只做纯计算（无 I/O、无缓存）；解码/缓存由 formant_cache.rs 负责，
 *   Tauri 命令封装在 commands/formant.rs。
 *
 * 维护说明：
 * - 本模块的阈值筛选与 DSP 完全同源，若调整 lpc/track 的筛选参数，
 *   此处显示结果会自动跟随，无需同步维护。
 */

use super::decimator::Decimator;
use super::{lpc, track, ANALYSIS_FRAME_SEC, ANALYSIS_HOP_SEC, ANALYSIS_TARGET_RATE, LPC_ORDER};

/// 轨迹稀疏采样的最大点数（控制 IPC 载荷大小）。
const TRACK_MAX_POINTS: usize = 64;

/// 源共振峰分析结果。
#[derive(Debug, Clone)]
pub struct FormantAnalysisSummary {
    /// 统计意义上的源 F1（检出帧中位数，Hz；无检出为 0）。
    pub source_f1_hz: f32,
    /// 统计意义上的源 F2（Hz；无检出为 0）。
    pub source_f2_hz: f32,
    /// 稀疏轨迹 (t_norm ∈ [0,1], f1_hz, f2_hz)，仅含检出帧，按时间升序。
    pub track: Vec<(f32, f32, f32)>,
    /// 检出候选的分析帧占比 [0,1]。过低说明素材不适合做共振峰调整。
    pub voiced_ratio: f32,
    /// 诊断消息："source_too_short" / "no_voiced_frames"。
    pub message: Option<&'static str>,
}

/// 分析一段 mono PCM 的源共振峰。
///
/// 流程：
/// 1. 采样率/长度校验（不足两个分析帧 → source_too_short）。
/// 2. 降采样到分析域 → 逐帧 LPC → 求根 → 候选筛选（与 DSP 同源）。
/// 3. 统计中位数 F1/F2、浊音占比，轨迹下采样到 ≤TRACK_MAX_POINTS。
pub fn analyze_clip_formants(mono: &[f32], sr: u32) -> FormantAnalysisSummary {
    let empty = |msg| FormantAnalysisSummary {
        source_f1_hz: 0.0,
        source_f2_hz: 0.0,
        track: Vec::new(),
        voiced_ratio: 0.0,
        message: Some(msg),
    };

    let Some(mut decimator) = Decimator::new(sr, ANALYSIS_TARGET_RATE as u32) else {
        return empty("source_too_short");
    };
    let analysis_rate = decimator.analysis_rate();
    let analysis = decimator.process(mono);
    let frame_len = (analysis_rate * ANALYSIS_FRAME_SEC).round() as usize;
    let hop = (analysis_rate * ANALYSIS_HOP_SEC).round() as usize;
    if analysis.len() < frame_len * 2 {
        return empty("source_too_short");
    }

    // 逐帧分析（与 DSP 主流程完全同源）
    let mut detected: Vec<(f32, f32, f32)> = Vec::new(); // (t_sec, f1, f2)
    let mut total_frames = 0usize;
    let mut start = 0usize;
    while start + frame_len <= analysis.len() {
        total_frames += 1;
        if let Some(lpc_res) = lpc::analyze_frame(&analysis[start..start + frame_len], analysis_rate, LPC_ORDER) {
            if let Some(roots) = lpc::poly_roots(&lpc::coeffs_to_monic(&lpc_res.coeffs)) {
                let poles = lpc::roots_to_poles(&roots, analysis_rate);
                if let Some(cand) =
                    track::select_f1_f2(&poles, &lpc_res.coeffs, analysis_rate, lpc_res.residual_ratio)
                {
                    let t = (start + frame_len / 2) as f32 / analysis_rate;
                    detected.push((t, cand.f1.freq_hz, cand.f2.freq_hz));
                }
            }
        }
        start += hop;
    }

    if detected.is_empty() || total_frames == 0 {
        return empty("no_voiced_frames");
    }

    // 中位数统计（F1/F2 独立取中位数）
    let median = |mut vals: Vec<f32>| -> f32 {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        vals[vals.len() / 2]
    };
    let f1s: Vec<f32> = detected.iter().map(|d| d.1).collect();
    let f2s: Vec<f32> = detected.iter().map(|d| d.2).collect();

    // 轨迹下采样到 ≤ TRACK_MAX_POINTS，t 归一化到 [0,1]
    let total_sec = analysis.len() as f32 / analysis_rate;
    let stride = detected.len().div_ceil(TRACK_MAX_POINTS);
    let track: Vec<(f32, f32, f32)> = detected
        .iter()
        .step_by(stride)
        .map(|(t, f1, f2)| (t / total_sec, *f1, *f2))
        .collect();

    FormantAnalysisSummary {
        source_f1_hz: median(f1s),
        source_f2_hz: median(f2s),
        track,
        voiced_ratio: detected.len() as f32 / total_frames as f32,
        message: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 48_000;

    /// 合成稳态元音（并联共振器 + 脉冲串），与 DSP 测试同一构造。
    fn synth_vowel(f0: f32, f1: f32, f2: f32, sr: u32, secs: f32) -> Vec<f32> {
        let sr = sr as f32;
        let n = (sr * secs) as usize;
        let period = (sr / f0).round() as usize;
        let r1 = (-(std::f32::consts::PI * 80.0) / sr).exp();
        let t1 = 2.0 * std::f32::consts::PI * f1 / sr;
        let r2 = (-(std::f32::consts::PI * 100.0) / sr).exp();
        let t2 = 2.0 * std::f32::consts::PI * f2 / sr;
        let mut y1 = [0.0_f32; 2];
        let mut y2 = [0.0_f32; 2];
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let x = if i % period == 0 { 1.0 } else { 0.0 };
            let o1 = x + 2.0 * r1 * t1.cos() * y1[0] - r1 * r1 * y1[1];
            y1[1] = y1[0];
            y1[0] = o1;
            let o2 = x + 2.0 * r2 * t2.cos() * y2[0] - r2 * r2 * y2[1];
            y2[1] = y2[0];
            y2[0] = o2;
            out.push((o1 + o2) * 0.15);
        }
        out
    }

    #[test]
    fn analyzes_synthetic_vowel_formants() {
        let signal = synth_vowel(150.0, 800.0, 1_200.0, SR, 0.5);
        let summary = analyze_clip_formants(&signal, SR);
        assert_eq!(summary.message, None);
        assert!(
            (summary.source_f1_hz - 800.0).abs() / 800.0 < 0.10,
            "F1 统计 {} 偏离真值过大",
            summary.source_f1_hz
        );
        assert!(
            (summary.source_f2_hz - 1_200.0).abs() / 1_200.0 < 0.10,
            "F2 统计 {} 偏离真值过大",
            summary.source_f2_hz
        );
        assert!(summary.voiced_ratio > 0.8, "voiced_ratio={}", summary.voiced_ratio);
        assert!(!summary.track.is_empty());
        assert!(summary.track.len() <= TRACK_MAX_POINTS);
        // t 归一化严格递增
        for w in summary.track.windows(2) {
            assert!(w[1].0 > w[0].0);
        }
    }

    #[test]
    fn silence_reports_no_voiced_frames() {
        let signal = vec![0.0_f32; SR as usize]; // 1s 静音
        let summary = analyze_clip_formants(&signal, SR);
        assert_eq!(summary.message, Some("no_voiced_frames"));
        assert!((summary.voiced_ratio - 0.0).abs() < 1e-6);
        assert!(summary.track.is_empty());
    }

    #[test]
    fn too_short_input_reports_source_too_short() {
        let summary = analyze_clip_formants(&[0.1_f32; 100], SR);
        assert_eq!(summary.message, Some("source_too_short"));
    }
}
