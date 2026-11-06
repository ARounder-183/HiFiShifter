//! 真实人声试听渲染：用项目自己的 `hifigan_chain()` 处理同一段 PCM，
//! 分别输出干声与怒音，供人工 A/B。默认 `#[ignore]`，不进入常规测试。
//!
//! 运行：
//! ```text
//! HIFISHIFTER_GROWL_INPUT=/path/to.wav \
//! HIFISHIFTER_GROWL_OUTPUT_DIR=/tmp/growl \
//! cargo test -p HiFiShifter --test growl_real_vocal_ab -- --ignored --nocapture
//! ```

use hifishifter_kernel::renderer::chain::hifigan_chain;
use hifishifter_kernel::renderer::{ClipProcessContext, ClipProcessor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SR: u32 = 40_000;
const START_SEC: f64 = 15.0;
const END_SEC: f64 = 27.0;
const FRAME_MS: f64 = 5.0;

#[test]
#[ignore = "requires models + a user-provided vocal file; run manually for listening"]
fn render_real_vocal_growl_ab() {
    let input = std::env::var("HIFISHIFTER_GROWL_INPUT")
        .map(PathBuf::from)
        .expect("set HIFISHIFTER_GROWL_INPUT");
    let out_dir = std::env::var("HIFISHIFTER_GROWL_OUTPUT_DIR")
        .map(PathBuf::from)
        .expect("set HIFISHIFTER_GROWL_OUTPUT_DIR");
    std::fs::create_dir_all(&out_dir).expect("create output dir");

    let models = Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/models");
    hifishifter_kernel::model_store::resolve_and_register(&models)
        .expect("bundled models should register");
    assert!(hifishifter_kernel::nsf_hifigan_onnx::is_available());
    assert!(hifishifter_kernel::hnsep_onnx::is_available());

    let (sr, channels, pcm) =
        hifishifter_kernel::audio_utils::decode_audio_f32_interleaved(&input).expect("decode");
    if channels > 1 {
        let mono = pcm
            .chunks_exact(channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect();
        return render_mono(mono, sr, &out_dir);
    }
    let mono = pcm;
    render_mono(mono, sr.max(1), &out_dir);
}

fn render_mono(input: Vec<f32>, sr: u32, out_dir: &Path) {
    let start = (START_SEC * sr as f64) as usize;
    let end = (END_SEC * sr as f64) as usize;
    let seg = if start < end && end <= input.len() {
        input[start..end].to_vec()
    } else {
        input
    };
    let pcm: Vec<f32> = if sr != SR {
        // 极简线性重采样，仅用于试听；真实引擎有自己的重采样器。
        let out_len = (seg.len() as f64 * SR as f64 / sr as f64).round() as usize;
        (0..out_len)
            .map(|i| {
                let t =
                    i as f64 * (seg.len().saturating_sub(1)) as f64 / (out_len.max(2) - 1) as f64;
                let i0 = t as usize;
                let i1 = (i0 + 1).min(seg.len().saturating_sub(1));
                seg[i0] + (seg[i1] - seg[i0]) * (t - i0 as f64) as f32
            })
            .collect()
    } else {
        seg
    };

    let frame_count = ((pcm.len() as f64 / SR as f64 * 1000.0 / FRAME_MS).ceil() as usize) + 8;
    let midi: Vec<f32> =
        hifishifter_kernel::fcpe_onnx::infer_f0_hz_f32(&pcm, SR, FRAME_MS, 55.0, 900.0)
            .expect("fcpe f0")
            .into_iter()
            .map(|hz| {
                if hz > 0.0 {
                    (69.0 + 12.0 * (hz / 440.0).log2()) as f32
                } else {
                    0.0
                }
            })
            .collect();
    let mut midi_padded = midi;
    midi_padded.resize(frame_count.max(8), 0.0);
    let pitch_edit = vec![0.0f32; midi_padded.len()];
    let params = HashMap::from([("breath_enabled".to_string(), 1.0f64)]);
    let chain = hifigan_chain();

    let render = |name: &str, growl: Option<Vec<f32>>| {
        let curves = growl
            .map(|g| HashMap::from([("hifigan_growl".to_string(), g)]))
            .unwrap_or_default();
        let out = chain
            .process(&ClipProcessContext {
                mono_pcm: &pcm,
                channel_index: 0,
                source_fingerprint: Some(0xA11CE),
                sample_rate: SR,
                clip_start_sec: 0.0,
                seg_start_sec: 0.0,
                seg_end_sec: pcm.len() as f64 / SR as f64,
                frame_period_ms: FRAME_MS,
                pitch_edit: &pitch_edit,
                clip_midi: &midi_padded,
                playback_rate: 1.0,
                out_frames: pcm.len(),
                clip_id: name,
                extra_curves: &curves,
                extra_params: &params,
            })
            .expect("render");
        write_wav(&out_dir.join(format!("{name}.wav")), &out);
        sf_report(name, &out);
    };

    render("dry", None);
    render("growl_medium", Some(vec![65.0f32; frame_count.max(8)]));
    render("growl_strong", Some(vec![95.0f32; frame_count.max(8)]));
}

fn write_wav(path: &Path, mono: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SR,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
    for &v in mono {
        let q = (v.clamp(-1.0, 1.0) * ((1i32 << 23) - 1) as f32).round() as i32;
        writer.write_sample(q).expect("write sample");
    }
    writer.finalize().expect("finalize wav");
}

fn sf_report(name: &str, mono: &[f32]) {
    let rms = (mono.iter().map(|v| v * v).sum::<f32>() / mono.len().max(1) as f32).sqrt();
    let peak = mono.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    eprintln!(
        "{name}: {} samples, rms {:.2} dB, peak {:.2} dB",
        mono.len(),
        20.0 * rms.max(1e-12).log10(),
        20.0 * peak.max(1e-12).log10()
    );
}
