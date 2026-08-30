//! SoundTouch DLL FFI 封装
//!
//! 基于 SoundTouch Windows DLL (`SoundTouchDLL_x64.dll`) 的音频时间拉伸模块。
//! 仅处理“保音高的时间拉伸”，因此固定使用：
//! - `rate = 1.0`
//! - `pitch = 1.0`
//! - `tempo = 1.0 / time_ratio`

use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::fmt::{Display, Formatter};

type Handle = *mut c_void;

#[link(name = "SoundTouchDLL")]
extern "C" {
    fn soundtouch_createInstance() -> Handle;
    fn soundtouch_destroyInstance(handle: Handle);
    fn soundtouch_getVersionString() -> *const c_char;
    fn soundtouch_setRate(handle: Handle, new_rate: f32);
    fn soundtouch_setTempo(handle: Handle, new_tempo: f32);
    fn soundtouch_setPitch(handle: Handle, new_pitch: f32);
    fn soundtouch_setChannels(handle: Handle, num_channels: c_uint) -> c_int;
    fn soundtouch_setSampleRate(handle: Handle, sample_rate: c_uint) -> c_int;
    fn soundtouch_flush(handle: Handle) -> c_int;
    fn soundtouch_putSamples(handle: Handle, samples: *const f32, num_samples: c_uint) -> c_int;
    #[allow(dead_code)]
    fn soundtouch_clear(handle: Handle);
    fn soundtouch_numSamples(handle: Handle) -> c_uint;
    #[allow(dead_code)]
    fn soundtouch_isEmpty(handle: Handle) -> c_int;
    fn soundtouch_receiveSamples(
        handle: Handle,
        out_buffer: *mut f32,
        max_samples: c_uint,
    ) -> c_uint;
}

#[derive(Debug, Clone)]
pub enum SoundTouchError {
    RuntimeUnavailable(String),
    InvalidConfig(&'static str),
    ProcessingFailed(&'static str),
}

impl Display for SoundTouchError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RuntimeUnavailable(name) => {
                write!(f, "SoundTouch runtime unavailable: {name}")
            }
            Self::InvalidConfig(msg) => write!(f, "SoundTouch invalid config: {msg}"),
            Self::ProcessingFailed(msg) => write!(f, "SoundTouch processing failed: {msg}"),
        }
    }
}

impl std::error::Error for SoundTouchError {}

pub fn runtime_library_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "SoundTouchDLL.dll"
    } else if cfg!(target_os = "macos") {
        "libSoundTouchDLL.dylib"
    } else {
        "libSoundTouchDLL.so"
    }
}

#[allow(dead_code)]
pub fn import_library_name() -> &'static str {
    "SoundTouchDLL.lib"
}

pub fn normalize_output_len(mut output: Vec<f32>, channels: usize, out_frames: usize) -> Vec<f32> {
    let wanted = out_frames.saturating_mul(channels);
    output.resize(wanted, 0.0);
    output.truncate(wanted);
    output
}

pub fn is_available() -> bool {
    true
}

fn tempo_from_time_ratio(time_ratio: f64) -> f32 {
    let safe_ratio = if time_ratio.is_finite() && time_ratio > 1.0e-6 {
        time_ratio
    } else {
        1.0
    };
    (1.0 / safe_ratio) as f32
}

fn version_string() -> Option<String> {
    let ptr = unsafe { soundtouch_getVersionString() };
    if ptr.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .trim()
            .to_string(),
    )
}

struct SoundTouchState {
    handle: Handle,
    channels: usize,
}

unsafe impl Send for SoundTouchState {}

impl SoundTouchState {
    fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, SoundTouchError> {
        if !is_available() {
            return Err(SoundTouchError::RuntimeUnavailable(
                runtime_library_name().to_string(),
            ));
        }
        if channels == 0 {
            return Err(SoundTouchError::InvalidConfig("channels == 0"));
        }
        let handle = unsafe { soundtouch_createInstance() };
        if handle.is_null() {
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_createInstance returned null",
            ));
        }

        let init_result = unsafe {
            let ch_ok = soundtouch_setChannels(handle, channels as c_uint);
            let sr_ok = soundtouch_setSampleRate(handle, sample_rate.max(1) as c_uint);
            soundtouch_setRate(handle, 1.0);
            soundtouch_setPitch(handle, 1.0);
            soundtouch_setTempo(handle, tempo_from_time_ratio(time_ratio));
            (ch_ok, sr_ok)
        };

        if init_result.0 == 0 || init_result.1 == 0 {
            unsafe { soundtouch_destroyInstance(handle) };
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_setChannels/sampleRate failed",
            ));
        }

        Ok(Self { handle, channels })
    }

    #[allow(dead_code)]
    fn reset(&mut self, time_ratio: f64) -> Result<(), SoundTouchError> {
        unsafe {
            soundtouch_clear(self.handle);
            soundtouch_setRate(self.handle, 1.0);
            soundtouch_setPitch(self.handle, 1.0);
            soundtouch_setTempo(self.handle, tempo_from_time_ratio(time_ratio));
        }
        Ok(())
    }

    fn put_samples(&mut self, input_interleaved: &[f32]) -> Result<(), SoundTouchError> {
        let in_frames = input_interleaved.len() / self.channels.max(1);
        if in_frames == 0 {
            return Ok(());
        }
        let ret = unsafe {
            soundtouch_putSamples(self.handle, input_interleaved.as_ptr(), in_frames as c_uint)
        };
        if ret == 0 {
            return Err(SoundTouchError::ProcessingFailed(
                "soundtouch_putSamples failed",
            ));
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), SoundTouchError> {
        let ret = unsafe { soundtouch_flush(self.handle) };
        if ret == 0 {
            return Err(SoundTouchError::ProcessingFailed("soundtouch_flush failed"));
        }
        Ok(())
    }

    fn drain_available(
        &mut self,
        out: &mut Vec<f32>,
        max_frames_per_read: usize,
    ) -> Result<(), SoundTouchError> {
        let mut temp = vec![0.0f32; max_frames_per_read.max(1) * self.channels];
        loop {
            let available = unsafe { soundtouch_numSamples(self.handle) as usize };
            if available == 0 {
                break;
            }
            let want_frames = available.min(max_frames_per_read.max(1));
            let got = unsafe {
                soundtouch_receiveSamples(self.handle, temp.as_mut_ptr(), want_frames as c_uint)
                    as usize
            };
            if got == 0 {
                break;
            }
            out.extend_from_slice(&temp[..got * self.channels]);
        }
        Ok(())
    }
}

impl Drop for SoundTouchState {
    fn drop(&mut self) {
        if self.handle.is_null() {
            return;
        }
        unsafe { soundtouch_destroyInstance(self.handle) };
        self.handle = std::ptr::null_mut();
    }
}

pub struct RealtimeStretcher {
    inner: SoundTouchState,
    channels: usize,
    out_buffer: Vec<f32>,
    /// Read cursor into `out_buffer`; draining from the front of a large
    /// buffer with `Vec::drain` is O(remaining) per chunk, so we advance a
    /// cursor and only compact when the buffer is fully consumed.
    out_read_pos: usize,
}

unsafe impl Send for RealtimeStretcher {}

impl RealtimeStretcher {
    pub fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, String> {
        let inner =
            SoundTouchState::new(sample_rate, channels, time_ratio).map_err(|e| e.to_string())?;
        if std::env::var("HIFISHIFTER_DEBUG_COMMANDS").ok().as_deref() == Some("1") {
            if let Some(version) = version_string() {
                eprintln!(
                    "[SoundTouch] Created realtime stretcher: sr={} ch={} ratio={:.6} version={}",
                    sample_rate, channels, time_ratio, version
                );
            }
        }
        Ok(Self {
            inner,
            channels,
            out_buffer: Vec::with_capacity(4096),
            out_read_pos: 0,
        })
    }

    #[allow(dead_code)]
    pub fn reset(&mut self, time_ratio: f64) -> Result<(), String> {
        self.out_buffer.clear();
        self.out_read_pos = 0;
        self.inner.reset(time_ratio).map_err(|e| e.to_string())
    }

    pub fn process_interleaved(
        &mut self,
        input_interleaved: &[f32],
        final_chunk: bool,
    ) -> Result<(), String> {
        if input_interleaved.is_empty() {
            if final_chunk {
                self.inner.flush().map_err(|e| e.to_string())?;
                self.inner
                    .drain_available(&mut self.out_buffer, 1024)
                    .map_err(|e| e.to_string())?;
            }
            return Ok(());
        }
        self.inner
            .put_samples(input_interleaved)
            .map_err(|e| e.to_string())?;
        self.inner
            .drain_available(&mut self.out_buffer, 1024)
            .map_err(|e| e.to_string())?;
        if final_chunk {
            self.inner.flush().map_err(|e| e.to_string())?;
            self.inner
                .drain_available(&mut self.out_buffer, 1024)
                .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn retrieve_interleaved_into(
        &mut self,
        out_interleaved: &mut Vec<f32>,
        max_frames: usize,
    ) -> Result<usize, String> {
        let remaining = self.out_buffer.len() - self.out_read_pos;
        if remaining == 0 || max_frames == 0 {
            return Ok(0);
        }
        let avail_frames = remaining / self.channels.max(1);
        let take_frames = avail_frames.min(max_frames);
        let take_samples = take_frames * self.channels;
        let end = self.out_read_pos + take_samples;
        out_interleaved.extend_from_slice(&self.out_buffer[self.out_read_pos..end]);
        self.out_read_pos = end;
        if self.out_read_pos == self.out_buffer.len() {
            self.out_buffer.clear();
            self.out_read_pos = 0;
        }
        Ok(take_frames)
    }
}

pub fn try_time_stretch_interleaved_offline(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    if input_interleaved.is_empty() || channels == 0 {
        return Ok(vec![]);
    }
    let in_frames = input_interleaved.len() / channels;
    if in_frames < 2 {
        return Ok(normalize_output_len(
            input_interleaved.to_vec(),
            channels,
            out_frames_hint.max(in_frames),
        ));
    }

    let mut state =
        SoundTouchState::new(sample_rate, channels, time_ratio).map_err(|e| e.to_string())?;
    state
        .put_samples(input_interleaved)
        .map_err(|e| e.to_string())?;
    state.flush().map_err(|e| e.to_string())?;

    let mut out = Vec::with_capacity(out_frames_hint.max(1) * channels);
    state
        .drain_available(&mut out, 4096)
        .map_err(|e| e.to_string())?;

    let target_frames = if out_frames_hint > 0 {
        out_frames_hint
    } else {
        ((in_frames as f64) * time_ratio).round().max(1.0) as usize
    };
    Ok(normalize_output_len(out, channels, target_frames))
}

pub fn try_time_stretch_interleaved_realtime(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    let mut stretcher = RealtimeStretcher::new(sample_rate, channels, time_ratio)?;
    stretcher.process_interleaved(input_interleaved, true)?;
    let mut out = Vec::with_capacity(out_frames_hint.max(1) * channels);
    loop {
        let got = stretcher.retrieve_interleaved_into(&mut out, 4096)?;
        if got == 0 {
            break;
        }
    }
    Ok(normalize_output_len(out, channels, out_frames_hint))
}

#[cfg(test)]
mod tests {
    use super::{normalize_output_len, runtime_library_name, SoundTouchError};

    #[test]
    fn normalize_output_len_pads_short_output() {
        let out = normalize_output_len(vec![1.0, 2.0, 3.0, 4.0], 2, 4);
        assert_eq!(out.len(), 8);
        assert_eq!(&out[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&out[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalize_output_len_truncates_long_output() {
        let out = normalize_output_len(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 2);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn missing_runtime_error_is_clear() {
        let err = SoundTouchError::RuntimeUnavailable(runtime_library_name().to_string());
        assert!(err.to_string().contains(runtime_library_name()));
    }

    #[test]
    fn runtime_library_name_is_pinned() {
        let name = runtime_library_name();
        if cfg!(target_os = "windows") {
            assert_eq!(name, "SoundTouchDLL.dll");
        } else if cfg!(target_os = "macos") {
            assert_eq!(name, "libSoundTouchDLL.dylib");
        } else {
            assert_eq!(name, "libSoundTouchDLL.so");
        }
    }

    #[test]
    fn availability_does_not_depend_on_current_working_directory_dll_probe() {
        assert!(super::is_available());
    }
}
