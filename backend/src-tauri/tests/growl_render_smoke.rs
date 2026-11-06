//! 真实模型冒烟：怒音曲线必须真的改变送往声码器的谐波支。
//!
//! 这不是全量导出回归，而是防止"参数已经接进 UI/渲染键，却在 DSP 分支里
//! 被吞掉"的假接线。测试只在模型可用时运行；CI/开发机没有模型时跳过。

use hifishifter_kernel::renderer::chain::hifigan_chain;
use hifishifter_kernel::renderer::{ClipProcessContext, ClipProcessor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SEGMENT_SEC: f64 = 1.0;
const SAMPLE_RATE: u32 = 40_000;
const FRAME_MS: f64 = 5.0;

fn repo_model_dir() -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()?
        .join("src-tauri/resources/models");
    dir.join("nsf_hifigan/pc_nsf_hifigan_coreml.onnx")
        .is_file()
        .then_some(dir)
}

fn sine(freq: f64, seconds: f64, sr: u32) -> Vec<f32> {
    let n = (seconds * sr as f64) as usize;
    (0..n)
        .map(|i| (0.35 * (2.0 * std::f64::consts::PI * freq * i as f64 / sr as f64).sin()) as f32)
        .collect()
}

#[test]
fn growl_curve_changes_the_hifigan_output() {
    let Some(models) = repo_model_dir() else {
        eprintln!("skip: app-bundled nsf_hifigan model not present");
        return;
    };
    hifishifter_kernel::model_store::resolve_and_register(&models)
        .expect("bundled model dir should register");
    if !hifishifter_kernel::nsf_hifigan_onnx::is_available()
        || !hifishifter_kernel::hnsep_onnx::is_available()
    {
        eprintln!("skip: NF-HiFiGAN/HNSEP runtime unavailable");
        return;
    }

    let pcm = sine(220.0, SEGMENT_SEC, SAMPLE_RATE);
    let frame_count = ((SEGMENT_SEC * 1000.0 / FRAME_MS).ceil() as usize) + 4;
    let midi = vec![57.0f32; frame_count];
    let pitch_edit = vec![0.0f32; frame_count];
    let mut params = HashMap::new();
    params.insert("breath_enabled".to_string(), 1.0);

    let dry = hifigan_chain()
        .process(&ClipProcessContext {
            mono_pcm: &pcm,
            channel_index: 0,
            source_fingerprint: Some(0x5EED),
            sample_rate: SAMPLE_RATE,
            clip_start_sec: 0.0,
            seg_start_sec: 0.0,
            seg_end_sec: SEGMENT_SEC,
            frame_period_ms: FRAME_MS,
            pitch_edit: &pitch_edit,
            clip_midi: &midi,
            playback_rate: 1.0,
            out_frames: pcm.len(),
            clip_id: "growl-smoke-dry",
            extra_curves: &HashMap::new(),
            extra_params: &params,
        })
        .expect("dry render should succeed");

    let growl = vec![80.0f32; frame_count];
    let curves = HashMap::from([("hifigan_growl".to_string(), growl)]);
    let wet = hifigan_chain()
        .process(&ClipProcessContext {
            mono_pcm: &pcm,
            channel_index: 0,
            source_fingerprint: Some(0x5EED),
            sample_rate: SAMPLE_RATE,
            clip_start_sec: 0.0,
            seg_start_sec: 0.0,
            seg_end_sec: SEGMENT_SEC,
            frame_period_ms: FRAME_MS,
            pitch_edit: &pitch_edit,
            clip_midi: &midi,
            playback_rate: 1.0,
            out_frames: pcm.len(),
            clip_id: "growl-smoke-wet",
            extra_curves: &curves,
            extra_params: &params,
        })
        .expect("growl render should succeed");

    assert_eq!(dry.len(), wet.len(), "输出长度必须一致");
    let max_diff = dry
        .iter()
        .zip(wet.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff > 1e-3,
        "怒音曲线必须改变声码器输出；实测最大差 {max_diff}"
    );
}
