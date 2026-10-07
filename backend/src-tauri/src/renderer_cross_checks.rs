//! 渲染侧与混音侧采样器的一致性检查（跨 app / 内核边界）。
//!
//! 【为什么这一条住在 app】它断言"同一曲线、同一时刻，导出侧采样器（内核的
//! `renderer::chain`）与预览侧采样器（app 的 `audio_engine::mix`）给出相同的值"。
//! 两个被测对象分处边界两侧，所以判据只能住在能同时看见两边的地方 —— 也就是 app。
//! 单看任一侧的测试都无法发现"两侧各自自洽、彼此不同"这一缺陷形态。

/// **预览与导出必须一致**：同一曲线、同一时刻，渲染侧采样器
/// （本模块，导出走它）与混音侧采样器（`audio_engine::mix`，预览走它）
/// 必须给出相同的值 —— 包括越界区段。
///
/// 这是本模块越界语义修复的**跨模块判据**：单看任一侧的测试都无法发现
/// "两侧各自自洽、彼此不同"这一缺陷形态。
///
/// 注意两个采样器的入参口径不同：
/// - 本模块收**绝对秒**；
/// - 混音侧收 `abs_frame`，它是**绝对采样点序号**（不是曲线帧号），
///   内部按 `abs_frame / sample_rate` 换成秒（见 `mix.rs` 的
///   `volume_curve_samples_at_timeline_absolute_frame`）。
/// 因此这里用同一个「绝对秒」推出两者的入参，而不是直接传同一个整数。
#[test]
fn sample_curve_agrees_with_preview_mixer() {
    let fp = 5.0;
    let sr = 44_100u32;
    let curves: Vec<Option<Vec<f32>>> = vec![
        // 末值非默认值 —— 最能暴露 hold-last vs default 的分歧
        Some(vec![0.0, 0.25, 0.5]),
        // 单点曲线
        Some(vec![0.75]),
        // 全默认值
        Some(vec![1.0, 1.0, 1.0]),
        // 零值末点
        Some(vec![1.0, 0.0]),
        None,
        Some(vec![]),
    ];

    // 覆盖曲线内、末点、末点之后、以及很远的位置（单位：曲线帧）
    for &curve_frame in &[0.0f64, 1.0, 2.0, 3.0, 4.0, 10.0, 100.0, 10_000.0] {
        let abs_sec = curve_frame * fp / 1000.0;
        // 同一时刻换算成绝对采样点（四舍五入到最近的采样点）
        let abs_frame = (abs_sec * sr as f64).round() as u64;

        for curve in &curves {
            let export = hifishifter_kernel::renderer::chain::sample_curve_at_abs_sec(
                curve.as_deref(),
                abs_sec,
                fp,
                1.0,
            );
            let preview = crate::audio_engine::mix::sample_automation_curve(
                curve.as_deref(),
                abs_frame,
                sr,
                fp,
                1.0,
            );
            assert!(
                (export - preview).abs() < 1e-3,
                "preview/export disagree at {abs_sec}s (frame {abs_frame}) \
                     for curve {curve:?}: preview={preview}, export={export}"
            );
        }
    }
}
