//! Loop（循环源）纯函数回归测试（集成测试目标）。
//!
//! 为什么放在 tests/ 而不是模块内 `#[cfg(test)]`：lib 单元测试 harness 在
//! Windows 上因缺少内嵌 ComCtl32 v6 清单无法启动（tauri_build 只给 bin 目标
//! 嵌清单）。本文件经由 build.rs 的 `/MANIFEST:EMBED` link-arg 获得清单，
//! 可以在 Windows 上直接运行：
//!
//! ```text
//! cargo test --features __test-internals --test loop_semantics
//! ```
//!
//! 同类单元测试也保留在 pitch_clip.rs / state.rs 的 `#[cfg(test)]` 模块中，
//! 在 Linux/CI 等无此限制的环境下照常执行。

use backend_lib::__test_internals::trim_and_resample_midi;

/// Loop + 媒体时长未知 + 环绕窗口（start > end，split 产生）：
/// 不得落入"空窗口"提前返回 —— 退化为整条缓存曲线回绕，输出非空且相位与
/// 逐帧 floor_mod 参考一致。
#[test]
fn loop_wrapped_window_without_media_duration_does_not_collapse_to_empty() {
    // 环绕窗口 [3.5, 3.0)（start > end），缓存曲线 100 帧（1s @10ms）。
    let full: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let out = trim_and_resample_midi(
        &full,
        10.0,
        3.5,  // source_start_sec（> end）
        3.0,  // source_end_sec
        1.0,  // playback_rate
        2.0,  // clip_timeline_len_sec → target = 200 帧
        true, // loop_enabled
        None, // 媒体时长未知
        false,
    );
    assert_eq!(out.len(), 200, "curve must cover the clip length");
    assert!(out.iter().any(|&v| v > 0.0), "curve must not be empty");

    // 与逐帧 floor_mod 参考对拍：idx = floor_mod(anchor + i, N)。
    let anchor = ((3.5f64 * 1000.0) / 10.0).round() as i64; // 350
    for (i, v) in out.iter().enumerate() {
        let expect = full[(anchor + i as i64).rem_euclid(100) as usize];
        assert!((v - expect).abs() < 1e-6, "frame {i}: {v} != {expect}");
    }
}

/// Loop + 已知媒体时长：逐帧 floor_mod(anchor ± round(i·rate), N) 映射正确
/// （正放，负锚点环绕），且长度等于 clip 时间线帧数。
#[test]
fn loop_with_media_duration_maps_per_frame_floor_mod() {
    // 媒体 1s（100 帧 @10ms），锚点 -0.25s（负值环绕到尾部一侧）。
    let full: Vec<f32> = (0..100).map(|i| (i * 7) as f32 % 13.0).collect();
    let out = trim_and_resample_midi(&full, 10.0, -0.25, 1.0, 1.0, 1.5, true, Some(1.0), false);
    assert_eq!(out.len(), 150);
    let anchor = ((-0.25f64 * 1000.0) / 10.0).round() as i64; // -25
    for (i, v) in out.iter().enumerate() {
        let idx = (anchor + i as i64).rem_euclid(100);
        assert!((v - full[idx as usize]).abs() < 1e-6, "frame {i}");
    }
}

/// 非 Loop 行为回归：rate≈1 且 target 超出窗口时按既有约定 clamp 到窗口长度。
#[test]
fn non_loop_rate_near_one_clamps_target_to_window() {
    // 窗口 [0.5, 1.5) → 100 帧；clip 长度 2s → target 200 帧；
    // 非 Loop + rate≈1 → 输出应被 clamp 为 trimmed.len()=100。
    let full: Vec<f32> = (0..300).map(|i| i as f32).collect();
    let out = trim_and_resample_midi(&full, 10.0, 0.5, 1.5, 1.0, 2.0, false, None, false);
    assert_eq!(out.len(), 100, "clamped to window length");
    for (i, v) in out.iter().enumerate() {
        assert!((v - full[50 + i]).abs() < 1e-6, "frame {i} content preserved");
    }
}
