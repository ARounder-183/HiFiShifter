//! Signalsmith Stretch FFI wrapper.
//!
//! Batch stretching uses Signalsmith's `exact()` path (via `outputSeek()`),
//! which aligns output sample 0 with input sample 0. This is important:
//! the previous implementation fed the whole input followed by tail silence
//! and then discarded `outputLatency` samples from the start, which made
//! short clips silent and cut the leading ~0.06-0.12 s off longer clips.

use std::ffi::c_int;

// FFI declarations matching sstretch-c.h.
type SStretchState = *mut std::ffi::c_void;

extern "C" {
    fn sstretch_new(sample_rate: u32, channels: u32) -> SStretchState;
    fn sstretch_delete(state: SStretchState);
    fn sstretch_reset(state: SStretchState);
    fn sstretch_set_transpose_semitones(state: SStretchState, semitones: f64);
    fn sstretch_input_latency(state: SStretchState) -> c_int;
    fn sstretch_output_latency(state: SStretchState) -> c_int;

    fn sstretch_process_interleaved(
        state: SStretchState,
        input_interleaved: *const f32,
        in_frames: u32,
        output_interleaved: *mut f32,
        out_frames: u32,
    ) -> c_int;

    fn sstretch_exact(
        state: SStretchState,
        input_interleaved: *const f32,
        in_frames: u32,
        output_interleaved: *mut f32,
        out_frames: u32,
    ) -> c_int;
}

/// Signalsmith Stretch is statically linked, so it is always available.
pub fn is_available() -> bool {
    true
}

/// Streaming-style stretcher used by `stretch_stream`-compatible callers.
///
/// Signalsmith's `process()` consumes input and produces output in the same
/// call. This wrapper keeps the produced output in an internal buffer and
/// exposes it through `retrieve_interleaved_into()`.
pub struct SignalsmithRealtimeStretcher {
    state: SStretchState,
    #[allow(dead_code)]
    channels: usize,
    sample_rate: u32,
    #[allow(dead_code)]
    time_ratio: f64,

    #[allow(dead_code)]
    out_buffer: Vec<f32>,
    #[allow(dead_code)]
    temp_out: Vec<f32>,
}

unsafe impl Send for SignalsmithRealtimeStretcher {}

#[allow(dead_code)]
impl SignalsmithRealtimeStretcher {
    pub fn new(sample_rate: u32, channels: usize, time_ratio: f64) -> Result<Self, String> {
        if channels == 0 {
            return Err("signalsmith stretch: channels == 0".to_string());
        }
        if channels > 2 {
            return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
        }

        let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
            time_ratio
        } else {
            1.0
        };

        let state = unsafe { sstretch_new(sample_rate.max(1), channels as u32) };
        if state.is_null() {
            return Err("sstretch_new returned null".to_string());
        }

        unsafe {
            sstretch_set_transpose_semitones(state, 0.0);
        }

        Ok(Self {
            state,
            channels,
            sample_rate: sample_rate.max(1),
            time_ratio,
            out_buffer: Vec::with_capacity(4096),
            temp_out: Vec::with_capacity(4096),
        })
    }

    pub fn reset(&mut self, time_ratio: f64) -> Result<(), String> {
        let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
            time_ratio
        } else {
            1.0
        };
        self.time_ratio = time_ratio;
        self.out_buffer.clear();
        unsafe {
            sstretch_reset(self.state);
            sstretch_set_transpose_semitones(self.state, 0.0);
        }
        Ok(())
    }

    pub fn process_interleaved(
        &mut self,
        input_interleaved: &[f32],
        _final: bool,
    ) -> Result<(), String> {
        if input_interleaved.is_empty() {
            return Ok(());
        }
        let in_frames = input_interleaved.len() / self.channels.max(1);
        if in_frames == 0 {
            return Ok(());
        }

        let out_frames = ((in_frames as f64) * self.time_ratio).ceil() as usize;
        if out_frames == 0 {
            return Ok(());
        }

        self.temp_out.resize(out_frames * self.channels, 0.0);
        let ret = unsafe {
            sstretch_process_interleaved(
                self.state,
                input_interleaved.as_ptr(),
                in_frames as u32,
                self.temp_out.as_mut_ptr(),
                out_frames as u32,
            )
        };
        if ret < 0 {
            return Err("sstretch_process_interleaved failed".to_string());
        }

        self.out_buffer.extend_from_slice(&self.temp_out);
        Ok(())
    }

    pub fn retrieve_interleaved_into(
        &mut self,
        out_interleaved: &mut Vec<f32>,
        max_frames: usize,
    ) -> Result<usize, String> {
        if self.out_buffer.is_empty() || max_frames == 0 {
            return Ok(0);
        }

        let avail_samples = self.out_buffer.len();
        let avail_frames = avail_samples / self.channels.max(1);
        let take_frames = avail_frames.min(max_frames);
        let take_samples = take_frames * self.channels;
        if take_samples == 0 {
            return Ok(0);
        }

        out_interleaved.extend_from_slice(&self.out_buffer[..take_samples]);
        self.out_buffer.drain(..take_samples);
        Ok(take_frames)
    }

    #[allow(dead_code)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl Drop for SignalsmithRealtimeStretcher {
    fn drop(&mut self) {
        if self.state.is_null() {
            return;
        }
        unsafe {
            sstretch_delete(self.state);
        }
        self.state = std::ptr::null_mut();
    }
}

/// Run Signalsmith's `exact()` path with an automatic leading-zero pad for
/// very short clips.
///
/// `exact()` internally uses `outputSeek()`, which aligns output sample 0 to
/// input sample 0 instead of leaving an output-latency-sized gap at the start.
/// When the source is shorter than the library's seek length, we prepend
/// silence so the whole source can still be processed without producing a
/// silent buffer.
fn stretch_exact_interleaved(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames: usize,
) -> Result<Vec<f32>, String> {
    if input_interleaved.is_empty() || channels == 0 {
        return Ok(vec![]);
    }
    if channels > 2 {
        return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
    }

    let in_frames = input_interleaved.len() / channels;
    if in_frames == 0 || out_frames == 0 {
        return Ok(vec![]);
    }

    let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
        time_ratio
    } else {
        1.0
    };

    unsafe {
        let state = sstretch_new(sample_rate.max(1), channels as u32);
        if state.is_null() {
            return Err("sstretch_new returned null".to_string());
        }

        sstretch_set_transpose_semitones(state, 0.0);

        let input_latency = sstretch_input_latency(state).max(0) as usize;
        let output_latency = sstretch_output_latency(state).max(0) as usize;

        // Signalsmith's seek length is inputLatency + playbackRate * outputLatency,
        // where playbackRate = in_frames / out_frames.
        let playback_rate = in_frames as f64 / out_frames.max(1) as f64;
        let seek_length = input_latency as f64 + playback_rate * output_latency as f64;

        // Pad the front only when the source is too short for exact alignment.
        let pad_in = if (in_frames as f64) >= seek_length {
            0
        } else {
            (seek_length - in_frames as f64).ceil().max(0.0) as usize
        };
        let pad_out = (pad_in as f64 * time_ratio).round() as usize;

        let padded_in_frames = in_frames + pad_in;
        let padded_out_frames = out_frames + pad_out;

        let mut padded_input = vec![0.0f32; padded_in_frames * channels];
        padded_input[pad_in * channels..(pad_in + in_frames) * channels]
            .copy_from_slice(input_interleaved);

        let mut padded_output = vec![0.0f32; padded_out_frames * channels];
        let ret = sstretch_exact(
            state,
            padded_input.as_ptr(),
            padded_in_frames as u32,
            padded_output.as_mut_ptr(),
            padded_out_frames as u32,
        );

        sstretch_delete(state);

        if ret < 0 {
            return Err("sstretch_exact failed".to_string());
        }
        if ret == 0 {
            // Should not happen because we padded past the seek length.
            return Err("sstretch_exact reported input too short".to_string());
        }

        let start = pad_out * channels;
        let end = start + out_frames * channels;
        Ok(padded_output[start..end].to_vec())
    }
}

/// Offline whole-buffer time stretch, latency-aligned from sample 0.
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
    if channels > 2 {
        return Err("signalsmith stretch: channels > 2 not supported yet".to_string());
    }

    let in_frames = input_interleaved.len() / channels;
    if in_frames < 2 {
        return Ok(input_interleaved.to_vec());
    }

    let time_ratio = if time_ratio.is_finite() && time_ratio > 1e-6 {
        time_ratio
    } else {
        1.0
    };
    let out_frames = if out_frames_hint > 0 {
        out_frames_hint
    } else {
        ((in_frames as f64) * time_ratio).ceil() as usize
    };

    stretch_exact_interleaved(
        input_interleaved,
        channels,
        sample_rate,
        time_ratio,
        out_frames,
    )
}

/// Batch "realtime" entry point. This is also a whole-buffer call, so it uses
/// the same latency-aligned exact path as the offline entry point.
pub fn try_time_stretch_interleaved_realtime(
    input_interleaved: &[f32],
    channels: usize,
    sample_rate: u32,
    time_ratio: f64,
    out_frames_hint: usize,
) -> Result<Vec<f32>, String> {
    try_time_stretch_interleaved_offline(
        input_interleaved,
        channels,
        sample_rate,
        time_ratio,
        out_frames_hint,
    )
}

#[cfg(test)]
mod tests {
    use super::try_time_stretch_interleaved_realtime;

    fn peak_frame(samples: &[f32], channels: usize) -> (usize, f32) {
        let mut best = (0usize, 0.0f32);
        for (frame, chunk) in samples.chunks_exact(channels).enumerate() {
            let peak = chunk.iter().copied().fold(0.0f32, f32::max);
            if peak > best.1 {
                best = (frame, peak);
            }
        }
        best
    }

    #[test]
    fn short_clip_is_not_silent_and_keeps_its_leading_audio() {
        // 0.05s mono source at 44.1 kHz, well below Signalsmith's ~0.12s seek length.
        let sample_rate = 44_100;
        let in_frames = 2_205;
        let mut input = vec![0.0f32; in_frames];
        input[0] = 1.0;
        input[100] = 0.5;

        let out = try_time_stretch_interleaved_realtime(&input, 1, sample_rate, 1.0, in_frames)
            .expect("signalsmith stretch should succeed");
        assert_eq!(out.len(), in_frames);

        let (peak, value) = peak_frame(&out, 1);
        assert!(
            value > 0.5,
            "short clip became (nearly) silent: peak={value}"
        );
        assert!(
            peak <= 2,
            "short clip lost its leading audio: peak at frame {peak}"
        );
    }

    #[test]
    fn long_clip_keeps_audio_at_the_very_start() {
        let sample_rate = 44_100;
        let in_frames = 8_820; // 0.2s
        let mut input = vec![0.0f32; in_frames];
        input[0] = 1.0;

        for (out_frames, time_ratio) in
            [(in_frames, 1.0), (in_frames * 2, 2.0), (in_frames / 2, 0.5)]
        {
            let out = try_time_stretch_interleaved_realtime(
                &input,
                1,
                sample_rate,
                time_ratio,
                out_frames,
            )
            .expect("signalsmith stretch should succeed");
            assert_eq!(out.len(), out_frames);

            let (peak, value) = peak_frame(&out, 1);
            assert!(
                value > 0.5,
                "stretched clip lost its start (ratio={time_ratio}): peak={value}"
            );
            assert!(
                peak <= 2,
                "stretched clip lost its leading audio (ratio={time_ratio}): peak at frame {peak}"
            );
        }
    }
}
