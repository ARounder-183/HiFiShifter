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

// External C functions from statically-linked WORLD library
extern "C" {
    fn Dio(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const DioOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    fn InitializeDioOption(option: *mut DioOption);
    fn GetSamplesForDIO(fs: i32, x_length: i32, frame_period: f64) -> i32;
    fn Harvest(
        x: *const f64,
        x_length: i32,
        fs: i32,
        option: *const HarvestOption,
        temporal_positions: *mut f64,
        f0: *mut f64,
    );
    fn InitializeHarvestOption(option: *mut HarvestOption);
    fn GetSamplesForHarvest(fs: i32, x_length: i32, frame_period: f64) -> i32;
    fn StoneMask(
        x: *const f64,
        x_length: i32,
        fs: i32,
        temporal_positions: *const f64,
        f0: *const f64,
        f0_length: i32,
        refined_f0: *mut f64,
    );
}

pub fn is_available() -> bool {
    // With static linking, WORLD functions are always available
    true
}

pub fn compute_f0_hz_harvest(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<Vec<f64>, String> {
    if x.is_empty() {
        return Ok(vec![]);
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
        return Ok(vec![]);
    }

    // 在栈上开辟空间，由 C 库负责初始化
    let mut option = unsafe {
        let mut opt = std::mem::MaybeUninit::uninit();
        InitializeHarvestOption(opt.as_mut_ptr());
        opt.assume_init()
    };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = Vec::with_capacity(samples as usize);
    let mut f0 = Vec::with_capacity(samples as usize);
    unsafe {
        temporal_positions.set_len(samples as usize);
        f0.set_len(samples as usize);
    }

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

    // Harvest already produces stable F0; keep StoneMask for Dio only.
    Ok(f0)
}

pub fn compute_f0_hz_dio_stonemask(
    x: &[f64],
    fs: i32,
    frame_period_ms: f64,
    f0_floor: f64,
    f0_ceil: f64,
) -> Result<Vec<f64>, String> {
    if x.is_empty() {
        return Ok(vec![]);
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
        return Ok(vec![]);
    }

    let mut option = unsafe {
        let mut opt = std::mem::MaybeUninit::uninit();
        InitializeDioOption(opt.as_mut_ptr());
        opt.assume_init()
    };
    option.frame_period = fp;
    if f0_floor.is_finite() && f0_floor > 0.0 {
        option.f0_floor = f0_floor;
    }
    if f0_ceil.is_finite() && f0_ceil > 0.0 {
        option.f0_ceil = f0_ceil;
    }

    let mut temporal_positions = Vec::with_capacity(samples as usize);
    let mut f0 = Vec::with_capacity(samples as usize);
    unsafe {
        temporal_positions.set_len(samples as usize);
        f0.set_len(samples as usize);
    }

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

    let mut refined = Vec::with_capacity(samples as usize);
    unsafe {
        refined.set_len(samples as usize);
    }
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

    Ok(refined)
}
