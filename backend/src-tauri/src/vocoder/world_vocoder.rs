use std::sync::OnceLock;
// Direct FFI bindings to statically-linked WORLD library

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DioOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub channels_in_octave: f64,
    pub frame_period: f64, // msec
    pub speed: i32,
    pub allowed_range: f64,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HarvestOption {
    pub f0_floor: f64,
    pub f0_ceil: f64,
    pub frame_period: f64, // msec
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CheapTrickOption {
    pub q1: f64,
    pub f0_floor: f64,
    pub fft_size: i32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct D4COption {
    pub threshold: f64,
}

// External C functions from statically-linked WORLD library
extern "C" {
    pub fn Dio(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const DioOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    pub fn InitializeDioOption(option: *mut DioOption);
    pub fn GetSamplesForDIO(fs: i32, x_length: i32, frame_period: f64) -> i32;
    pub fn StoneMask(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        refined_f0: *mut f64,
    );
    pub fn Harvest(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const HarvestOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    pub fn InitializeHarvestOption(option: *mut HarvestOption);
    pub fn GetSamplesForHarvest(fs: i32, x_length: i32, frame_period: f64) -> i32;
    pub fn CheapTrick(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        option: *const CheapTrickOption,
        spectrogram: *mut *mut f64,
    );
    pub fn InitializeCheapTrickOption(fs: i32, option: *mut CheapTrickOption);
    pub fn GetFFTSizeForCheapTrick(fs: i32, option: *const CheapTrickOption) -> i32;
    pub fn D4C(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        fft_size: i32,
        option: *const D4COption,
        aperiodicity: *mut *mut f64,
    );
    pub fn InitializeD4COption(option: *mut D4COption);
    pub fn Synthesis(
        f0: *const f64,
        f0_length: i32,
        spectrogram: *const *const f64,
        aperiodicity: *const *const f64,
        fft_size: i32,
        frame_period: f64,
        fs: i32,
        y_length: i32,
        y: *mut f64,
    );

    // ── 流式合成（WorldSynthesizer / synthesisrealtime.h）──────────────────
    pub fn InitializeSynthesizer(
        fs: i32,
        frame_period: f64,
        fft_size: i32,
        buffer_size: i32,
        number_of_pointers: i32,
        synth: *mut WorldSynthesizerRaw,
    );
    pub fn AddParameters(
        f0: *mut f64,
        f0_length: i32,
        spectrogram: *mut *mut f64,
        aperiodicity: *mut *mut f64,
        synth: *mut WorldSynthesizerRaw,
    ) -> i32;
    pub fn RefreshSynthesizer(synth: *mut WorldSynthesizerRaw);
    pub fn DestroySynthesizer(synth: *mut WorldSynthesizerRaw);
    pub fn IsLocked(synth: *mut WorldSynthesizerRaw) -> i32;
    pub fn Synthesis2(synth: *mut WorldSynthesizerRaw) -> i32;
}

// ── WorldSynthesizer 不透明句柄 ────────────────────────────────────────────────
//
// WorldSynthesizer 结构体内部含有大量裸指针和 FFT 状态，Rust 侧不需要直接访问字段，
// 只需保证内存布局足够大即可。我们用一个足够大的字节数组作为不透明存储，
// 实际大小由 C++ 侧的 sizeof(WorldSynthesizer) 决定。
//
// 精确计算（x86_64，MSVC ABI）：
//   基本字段（fs/frame_period/buffer_size/...到 randn_state 结束）：约 192 字节
//   RandnState（4×uint32）：16 字节
//   MinimumPhaseAnalysis（含 2 个 fft_plan，每个 72 字节）：176 字节
//   InverseRealFFT（含 1 个 fft_plan）：96 字节
//   ForwardRealFFT（含 1 个 fft_plan）：96 字节
//   合计：约 560 字节
//
// 原来预留 512 字节不足（差 ~48 字节），导致 InitializeSynthesizer 写入越界，
// 引发 STATUS_ACCESS_VIOLATION。现扩大到 1024 字节，留足安全余量。
//
// 注意：该结构体**只能**通过 Box::new(zeroed()) 分配在堆上，
// 绝不能在栈上创建（避免栈溢出和移动后指针失效）。
#[repr(C)]
pub struct WorldSynthesizerRaw {
    _opaque: [u8; 1024],
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum WorldF0Method {
    Dio,
    Harvest,
}

fn world_f0_method() -> WorldF0Method {
    static METHOD: OnceLock<WorldF0Method> = OnceLock::new();
    *METHOD.get_or_init(|| {
        match std::env::var("HIFISHIFTER_WORLD_F0")
            .ok()
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("dio") => WorldF0Method::Dio,
            Some("harvest") => WorldF0Method::Harvest,
            _ => WorldF0Method::Harvest,
        }
    })
}

pub fn is_available() -> bool {
    // With static linking, WORLD functions are always available
    true
}

fn ratio_from_semitones(semitones: f64) -> f64 {
    crate::pitch_editing::semitone_to_ratio(semitones)
}

fn clamp11(x: f64) -> f64 {
    x.clamp(-1.0, 1.0)
}

fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
}

const WORLD_DRY_SILENCE_RMS: f64 = 0.003;

fn blend_unvoiced_regions_with_silence_gate(
    out: &mut [f64],
    dry: &[f64],
    voiced: &[bool],
    fp: f64,
    fs: i32,
) {
    static SILENCE_RMS: OnceLock<f64> = OnceLock::new();
    let silence_rms = *SILENCE_RMS.get_or_init(|| {
        env_f64("HIFISHIFTER_WORLD_DRY_SILENCE_RMS")
            .unwrap_or(WORLD_DRY_SILENCE_RMS)
            .max(0.0)
    });
    blend_unvoiced_regions_with_silence_gate_impl(out, dry, voiced, fp, fs, silence_rms);
}

fn blend_unvoiced_regions_with_silence_gate_impl(
    out: &mut [f64],
    dry: &[f64],
    voiced: &[bool],
    fp: f64,
    fs: i32,
    silence_rms: f64,
) {
    if out.is_empty() || dry.is_empty() || voiced.is_empty() {
        return;
    }

    let fade_ms = 10.0f64;
    let fade_samples = ((fade_ms / 1000.0) * (fs.max(1) as f64)).round().max(0.0) as usize;
    let frame_samples = ((fp.max(0.1) / 1000.0) * (fs.max(1) as f64))
        .round()
        .max(1.0) as usize;

    let mut dry_allowed = vec![false; voiced.len()];
    for (fi, allowed) in dry_allowed.iter_mut().enumerate() {
        if voiced[fi] {
            continue;
        }
        let start = fi.saturating_mul(frame_samples).min(dry.len());
        let end = ((fi + 1).saturating_mul(frame_samples)).min(dry.len());
        if start >= end {
            continue;
        }
        let mut energy = 0.0f64;
        for &sample in &dry[start..end] {
            energy += sample * sample;
        }
        let rms = (energy / (end - start) as f64).sqrt();
        *allowed = rms >= silence_rms;
    }

    let mut w_prev = 0.0f64;
    let mut ramp_left = 0usize;
    let mut ramp_from = 0.0f64;
    let mut ramp_to = 0.0f64;

    for si in 0..out.len().min(dry.len()) {
        let t_ms = (si as f64) * 1000.0 / (fs.max(1) as f64);
        let fi = (t_ms / fp.max(0.1)).floor().max(0.0) as usize;
        let target_w = if fi < voiced.len() && voiced[fi] {
            1.0f64
        } else if fi < dry_allowed.len() && dry_allowed[fi] {
            0.0f64
        } else {
            1.0f64
        };

        if ramp_left == 0 && (target_w - w_prev).abs() > 1e-9 && fade_samples > 0 {
            ramp_left = fade_samples;
            ramp_from = w_prev;
            ramp_to = target_w;
        }

        let w = if ramp_left > 0 {
            let k = (fade_samples - ramp_left) as f64 / (fade_samples as f64);
            ramp_left = ramp_left.saturating_sub(1);
            ramp_from + (ramp_to - ramp_from) * k.clamp(0.0, 1.0)
        } else {
            target_w
        };

        if ramp_left == 0 {
            w_prev = target_w;
        }

        let wet = out[si];
        let dry_sample = dry[si];
        out[si] = wet * w + dry_sample * (1.0 - w);
    }
}

fn cleanup_f0_inplace(f0: &mut [f64], frame_period_ms: f64, f0_floor: f64, f0_ceil: f64) {
    if f0.is_empty() {
        return;
    }

    // 1) Clamp to a reasonable range and sanitize NaN/inf.
    for hz in f0.iter_mut() {
        if !hz.is_finite() || *hz < 0.0 {
            *hz = 0.0;
            continue;
        }
        if *hz > 0.0 {
            *hz = hz.clamp(f0_floor.max(1.0), f0_ceil.max(f0_floor.max(1.0)));
        }
    }

    // 2) Fill short unvoiced gaps inside voiced regions.
    // This reduces analysis/synthesis instability ("gargling") when f0 flickers to 0 for a few frames.
    // Default: 15ms; set HIFISHIFTER_WORLD_F0_GAP_MS=0 to disable.
    static GAP_MS: OnceLock<f64> = OnceLock::new();
    let gap_ms = *GAP_MS.get_or_init(|| env_f64("HIFISHIFTER_WORLD_F0_GAP_MS").unwrap_or(15.0));
    if gap_ms <= 0.0 {
        return;
    }
    let fp = frame_period_ms.max(0.1);
    let max_gap_frames = ((gap_ms / fp).round() as isize).max(1) as usize;

    let mut i = 0usize;
    while i < f0.len() {
        if f0[i] > 0.0 {
            i += 1;
            continue;
        }

        let start = i;
        while i < f0.len() && f0[i] <= 0.0 {
            i += 1;
        }
        let end = i; // [start, end) is unvoiced
        let gap_len = end - start;
        if gap_len == 0 || gap_len > max_gap_frames {
            continue;
        }
        if start == 0 || end >= f0.len() {
            continue;
        }

        let left = f0[start - 1];
        let right = f0[end];
        if !(left > 0.0 && right > 0.0) {
            continue;
        }

        // Linear interpolate across the gap.
        for k in 0..gap_len {
            let t = (k + 1) as f64 / (gap_len + 1) as f64;
            f0[start + k] = left + (right - left) * t;
        }
    }
}

fn compute_f0_with_positions_dio_stonemask(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if x.is_empty() {
        return Ok((vec![], vec![]));
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let x_len: i32 = x
        .len()
        .try_into()
        .map_err(|_| "WORLD: input too long".to_string())?;

    let samples = unsafe { GetSamplesForDIO(fs, x_len, fp) };
    if samples <= 0 {
        return Ok((vec![], vec![]));
    }

    let mut option = DioOption {
        f0_floor: 71.0,
        f0_ceil: 800.0,
        channels_in_octave: 2.0,
        frame_period: fp,
        speed: 1,
        allowed_range: 0.1,
    };
    unsafe { InitializeDioOption(&mut option as *mut DioOption) };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = vec![0.0f64; samples as usize];
    let mut f0 = vec![0.0f64; samples as usize];

    unsafe {
        Dio(
            x.as_ptr(),
            x_len,
            fs,
            &option as *const DioOption,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }

    let mut refined = vec![0.0f64; samples as usize];
    unsafe {
        StoneMask(
            x.as_ptr(),
            x_len,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            samples,
            refined.as_mut_ptr(),
        );
    }

    Ok((temporal_positions, refined))
}

fn compute_f0_with_positions_harvest(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if x.is_empty() {
        return Ok((vec![], vec![]));
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let x_len: i32 = x
        .len()
        .try_into()
        .map_err(|_| "WORLD: input too long".to_string())?;

    let samples = unsafe { GetSamplesForHarvest(fs, x_len, fp) };
    if samples <= 0 {
        return Ok((vec![], vec![]));
    }

    let mut option = HarvestOption {
        f0_floor: 71.0,
        f0_ceil: 800.0,
        frame_period: fp,
    };
    unsafe { InitializeHarvestOption(&mut option as *mut HarvestOption) };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = vec![0.0f64; samples as usize];
    let mut f0 = vec![0.0f64; samples as usize];

    unsafe {
        Harvest(
            x.as_ptr(),
            x_len,
            fs,
            &option as *const HarvestOption,
            temporal_positions.as_mut_ptr(),
            f0.as_mut_ptr(),
        );
    }

    // Note: StoneMask is primarily recommended for DIO. Keep Harvest output as-is.
    Ok((temporal_positions, f0))
}

fn vocode_one(
    x_f64: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    abs_time_start_sec: f64,
    semitone_at_time: &impl Fn(f64) -> f64,
) -> Result<Vec<f64>, String> {
    if x_f64.is_empty() {
        return Ok(vec![]);
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let (temporal_positions, mut f0) = match world_f0_method() {
        WorldF0Method::Harvest => {
            compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil).or_else(|_e| {
                // Fallback to DIO if Harvest symbols or runtime fail.
                compute_f0_with_positions_dio_stonemask(x_f64, fs, fp, f0_floor, f0_ceil)
            })?
        }
        WorldF0Method::Dio => compute_f0_with_positions_dio_stonemask(
            x_f64, fs, fp, f0_floor, f0_ceil,
        )
        .or_else(|_e| {
            // Fallback to Harvest if DIO fails.
            compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil)
        })?,
    };

    cleanup_f0_inplace(&mut f0, fp, f0_floor, f0_ceil);

    let f0_len_i32: i32 = f0
        .len()
        .try_into()
        .map_err(|_| "WORLD: f0 too long".to_string())?;

    if f0.is_empty() {
        // Nothing voiced; passthrough.
        return Ok(x_f64.to_vec());
    }

    // Precompute voiced flags. WORLD uses 0 Hz for unvoiced.
    let voiced: Vec<bool> = f0.iter().map(|&hz| hz > 0.0).collect();

    // Create shifted f0.
    let mut shifted_f0 = vec![0.0f64; f0.len()];
    for i in 0..f0.len() {
        let hz = f0[i];
        if hz > 0.0 {
            let t = temporal_positions.get(i).copied().unwrap_or(0.0);
            let abs_t = abs_time_start_sec + t;
            let semitones = semitone_at_time(abs_t);
            let r = ratio_from_semitones(semitones);
            shifted_f0[i] = hz * r;
        }
    }

    // CheapTrick options.
    let mut ct_opt = CheapTrickOption {
        q1: -0.15,
        f0_floor: f0_floor.max(20.0),
        fft_size: 0,
    };
    unsafe { InitializeCheapTrickOption(fs, &mut ct_opt as *mut CheapTrickOption) };
    ct_opt.f0_floor = f0_floor.max(20.0);

    let fft_size = unsafe { GetFFTSizeForCheapTrick(fs, &ct_opt as *const CheapTrickOption) };
    if fft_size <= 0 {
        return Err("WORLD: invalid fft_size".to_string());
    }
    ct_opt.fft_size = fft_size;

    let spec_bins = (fft_size as usize / 2) + 1;

    // Allocate spectrogram and aperiodicity as 2D arrays.
    let mut spectrogram = vec![0.0f64; f0.len() * spec_bins];
    let mut sp_ptrs: Vec<*mut f64> = spectrogram
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        CheapTrick(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_len_i32,
            &ct_opt as *const CheapTrickOption,
            sp_ptrs.as_mut_ptr(),
        );
    }

    let mut d4c_opt = D4COption { threshold: 0.85 };
    unsafe { InitializeD4COption(&mut d4c_opt as *mut D4COption) };

    let mut aperiodicity = vec![0.0f64; f0.len() * spec_bins];
    let mut ap_ptrs: Vec<*mut f64> = aperiodicity
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        D4C(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_len_i32,
            fft_size,
            &d4c_opt as *const D4COption,
            ap_ptrs.as_mut_ptr(),
        );
    }

    // Synthesis.
    let y_length: i32 = x_f64
        .len()
        .try_into()
        .map_err(|_| "WORLD: output too long".to_string())?;
    let mut y = vec![0.0f64; x_f64.len()];

    unsafe {
        Synthesis(
            shifted_f0.as_ptr(),
            f0_len_i32,
            sp_ptrs.as_ptr() as *const *const f64,
            ap_ptrs.as_ptr() as *const *const f64,
            fft_size,
            fp,
            fs,
            y_length,
            y.as_mut_ptr(),
        );
    }

    // Blend vocoded output with original for unvoiced / aperiodic regions.
    // This significantly reduces the typical "sand/noise" artifacts after pitch edits,
    // especially on fricatives/breath sounds where WORLD vocoding is brittle.
    // NOTE: This is not a low-pass on the control curve; it is voiced/unvoiced gating.
    let mut out = y;
    if !voiced.is_empty() {
        let debug = std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1");
        if debug {
            let voiced_n = voiced.iter().filter(|&&b| b).count();
            let ratio = (voiced_n as f64) / (voiced.len().max(1) as f64);
            eprintln!(
                "WORLD vocoder: voiced_ratio={:.3} f0_len={} fp_ms={:.3}",
                ratio,
                voiced.len(),
                fp
            );
        }
        blend_unvoiced_regions_with_silence_gate(&mut out, x_f64, &voiced, fp, fs);
    }

    Ok(out)
}

fn vocode_one_streaming(
    x_f64: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    abs_time_start_sec: f64,
    semitone_at_time: &impl Fn(f64) -> f64,
    synth: &mut crate::streaming_world::StreamingWorldSynthesizer,
) -> Result<Vec<f64>, String> {
    if x_f64.is_empty() {
        return Ok(vec![]);
    }

    let fp = if frame_period_ms.is_finite() && frame_period_ms > 0.1 {
        frame_period_ms
    } else {
        5.0
    };

    let (temporal_positions, mut f0) = match world_f0_method() {
        WorldF0Method::Harvest => {
            compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil).or_else(|_e| {
                compute_f0_with_positions_dio_stonemask(x_f64, fs, fp, f0_floor, f0_ceil)
            })?
        }
        WorldF0Method::Dio => {
            compute_f0_with_positions_dio_stonemask(x_f64, fs, fp, f0_floor, f0_ceil)
                .or_else(|_e| compute_f0_with_positions_harvest(x_f64, fs, fp, f0_floor, f0_ceil))?
        }
    };

    cleanup_f0_inplace(&mut f0, fp, f0_floor, f0_ceil);

    let f0_len_i32: i32 = f0
        .len()
        .try_into()
        .map_err(|_| "WORLD: f0 too long".to_string())?;

    if f0.is_empty() {
        return Ok(x_f64.to_vec());
    }

    let voiced: Vec<bool> = f0.iter().map(|&hz| hz > 0.0).collect();

    // 应用音高偏移
    let mut shifted_f0 = vec![0.0f64; f0.len()];
    for i in 0..f0.len() {
        let hz = f0[i];
        if hz > 0.0 {
            let t = temporal_positions.get(i).copied().unwrap_or(0.0);
            let abs_t = abs_time_start_sec + t;
            let semitones = semitone_at_time(abs_t);
            let r = ratio_from_semitones(semitones);
            shifted_f0[i] = hz * r;
        }
    }

    // CheapTrick 频谱分析
    let mut ct_opt = CheapTrickOption {
        q1: -0.15,
        f0_floor: f0_floor.max(20.0),
        fft_size: 0,
    };
    unsafe { InitializeCheapTrickOption(fs, &mut ct_opt as *mut CheapTrickOption) };
    ct_opt.f0_floor = f0_floor.max(20.0);

    let fft_size = unsafe { GetFFTSizeForCheapTrick(fs, &ct_opt as *const CheapTrickOption) };
    if fft_size <= 0 {
        return Err("WORLD: invalid fft_size".to_string());
    }
    ct_opt.fft_size = fft_size;

    let spec_bins = (fft_size as usize / 2) + 1;

    let mut spectrogram = vec![0.0f64; f0.len() * spec_bins];
    let mut sp_ptrs: Vec<*mut f64> = spectrogram
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        CheapTrick(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_len_i32,
            &ct_opt as *const CheapTrickOption,
            sp_ptrs.as_mut_ptr(),
        );
    }

    // D4C 非周期性分析
    let mut d4c_opt = D4COption { threshold: 0.85 };
    unsafe { InitializeD4COption(&mut d4c_opt as *mut D4COption) };

    let mut aperiodicity = vec![0.0f64; f0.len() * spec_bins];
    let mut ap_ptrs: Vec<*mut f64> = aperiodicity
        .chunks_exact_mut(spec_bins)
        .map(|row| row.as_mut_ptr())
        .collect();

    unsafe {
        D4C(
            x_f64.as_ptr(),
            x_f64
                .len()
                .try_into()
                .map_err(|_| "WORLD: input too long".to_string())?,
            fs,
            temporal_positions.as_ptr(),
            f0.as_ptr(),
            f0_len_i32,
            fft_size,
            &d4c_opt as *const D4COption,
            ap_ptrs.as_mut_ptr(),
        );
    }

    // 流式合成：将帧推入 WorldSynthesizer，取出合成 PCM
    // 若合成器被锁定（环形缓冲区满），先取出已合成样本再推入
    // 注意：push_frames 会取走数据所有权，防止 Synthesis2 访问悬空指针
    if synth.is_locked() {
        let _ = synth.pull_samples();
    }

    let pushed = synth.push_frames(
        shifted_f0.clone(),
        spectrogram.clone(),
        aperiodicity.clone(),
    );
    if !pushed {
        // 推入失败（缓冲区满），先取出再重试
        let _ = synth.pull_samples();
        synth.push_frames(
            shifted_f0.clone(),
            spectrogram.clone(),
            aperiodicity.clone(),
        );
    }

    let y_f64_raw = synth.pull_samples();

    // 若流式合成输出长度不足（合成器需要更多帧才能输出），
    // 回退到批量 Synthesis 保证输出长度正确
    let y_f64 = if y_f64_raw.len() >= x_f64.len() {
        y_f64_raw[..x_f64.len()].to_vec()
    } else {
        // 流式输出不足，用批量合成补全
        let y_length: i32 = x_f64
            .len()
            .try_into()
            .map_err(|_| "WORLD: output too long".to_string())?;
        let mut y = vec![0.0f64; x_f64.len()];
        unsafe {
            Synthesis(
                shifted_f0.as_ptr(),
                f0_len_i32,
                sp_ptrs.as_ptr() as *const *const f64,
                ap_ptrs.as_ptr() as *const *const f64,
                fft_size,
                fp,
                fs,
                y_length,
                y.as_mut_ptr(),
            );
        }
        y
    };

    // voiced/unvoiced 混合（与 vocode_one 相同逻辑）
    let mut out = y_f64;
    if !voiced.is_empty() {
        blend_unvoiced_regions_with_silence_gate(&mut out, x_f64, &voiced, fp, fs);
    }

    Ok(out)
}

/// WORLD vocoder pitch shift, chunked to avoid huge memory usage.
///
/// `start_sec` is the absolute timeline time for `mono_pcm[0]`.
/// 使用流式 `WorldSynthesizer` 跨块保持合成相位连续性，消除块边界的相位跳变。
pub fn vocode_pitch_shift_chunked<F>(
    mono_pcm: &[f32],
    sample_rate: u32,
    start_sec: f64,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
    semitone_at_time: F,
) -> Result<Vec<f32>, String>
where
    F: Fn(f64) -> f64,
{
    if mono_pcm.is_empty() {
        return Ok(vec![]);
    }

    let sr = sample_rate.max(1) as i32;

    let total_frames = mono_pcm.len();

    let chunk_sec = 6.0f64;
    let overlap_sec = 0.10f64;

    let chunk_len = (chunk_sec * (sample_rate as f64)).round().max(1.0) as usize;
    let overlap_len = (overlap_sec * (sample_rate as f64)).round().max(0.0) as usize;

    // 预先计算 fft_size，用于初始化流式合成器
    let fft_size = {
        let mut ct_opt = CheapTrickOption {
            q1: -0.15,
            f0_floor: f0_floor.max(20.0),
            fft_size: 0,
        };
        unsafe { InitializeCheapTrickOption(sr, &mut ct_opt) };
        ct_opt.f0_floor = f0_floor.max(20.0);
        unsafe { GetFFTSizeForCheapTrick(sr, &ct_opt) }
    };

    // 流式合成器：跨 chunk 共享，保持相位连续性
    // buffer_size = 512 样本（约 11ms @ 44100Hz），number_of_pointers = 32 个槽位
    let synth_buffer_size = 512usize;
    let synth_pointers = 32usize;
    let mut streaming_synth = crate::streaming_world::StreamingWorldSynthesizer::new(
        sample_rate,
        frame_period_ms,
        fft_size,
        synth_buffer_size,
        synth_pointers,
    );

    let mut out = vec![0.0f32; total_frames];

    // 预先在循环外部分配好内存池，避免在 Chunk 循环中疯狂申请堆内存
    let mut x_f64 = Vec::with_capacity(chunk_len + overlap_len * 2);

    let mut pos = 0usize;
    while pos < total_frames {
        let chunk_start = pos;
        let chunk_end = (pos + chunk_len).min(total_frames);

        let pad_start = chunk_start.saturating_sub(overlap_len);
        let pad_end = (chunk_end + overlap_len).min(total_frames);

        let x = &mono_pcm[pad_start..pad_end];
        x_f64.clear(); // 清空上次的脏数据，但复用分配好的物理内存容量

        let mut mean = 0.0f64;
        for &v in x {
            mean += v as f64;
        }
        mean /= x.len().max(1) as f64;

        let mut max_abs = 0.0f64;
        for &v in x {
            let vv = (v as f64) - mean;
            let a = vv.abs();
            if a.is_finite() && a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs.is_finite() && max_abs > 1.0 {
            (1.0 / max_abs).clamp(0.0, 1.0)
        } else {
            1.0
        };

        for &v in x {
            let vv = ((v as f64) - mean) * scale;
            x_f64.push(clamp11(vv));
        }

        let abs_time_start_sec = start_sec + (pad_start as f64) / (sample_rate as f64);

        // 使用流式合成器处理当前块
        let y_f64 = vocode_one_streaming(
            &x_f64,
            sr,
            frame_period_ms,
            f0_floor,
            f0_ceil,
            abs_time_start_sec,
            &semitone_at_time,
            &mut streaming_synth,
        )?;

        if y_f64.len() != x_f64.len() {
            return Err("WORLD: chunk output length mismatch".to_string());
        }

        let central_start = chunk_start - pad_start;

        let dst_start = chunk_start;
        let dst_end = chunk_end;

        for i in 0..(dst_end - dst_start) {
            let src_idx = central_start + i;
            let v = clamp11(y_f64[src_idx]) as f32;
            let dst_idx = dst_start + i;

            if overlap_len > 0 && chunk_start > 0 && dst_idx < chunk_start + overlap_len {
                let t = (dst_idx - chunk_start) as f32 / overlap_len as f32;
                let angle = t.clamp(0.0, 1.0) * std::f32::consts::FRAC_PI_2;
                let w_curr = angle.sin();
                let w_prev = angle.cos();
                let prev_val = out[dst_idx];
                out[dst_idx] = prev_val * w_prev + v * w_curr;
            } else {
                out[dst_idx] = v;
            }
        }

        pos = chunk_end;
    }

    Ok(out)
}
