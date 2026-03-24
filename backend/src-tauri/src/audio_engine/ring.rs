use std::sync::atomic::AtomicU32 as AtomicU32Cell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

#[derive(Debug)]
pub(crate) struct StreamRingStereo {
    pub(crate) cap_frames: u64,
    // Interleaved stereo stored as atomic bits so the audio callback can read lock-free.
    pub(crate) buf: Vec<AtomicU32Cell>,
    // If enabled, the callback may choose to output silence and not advance
    // position until the requested window is covered by [base_frame, write_frame).
    pub(crate) hard_start_enabled: AtomicBool,
    pub(crate) base_frame: AtomicU64,
    pub(crate) write_frame: AtomicU64,
}

impl StreamRingStereo {
    pub(crate) fn new(cap_frames: u64) -> Self {
        // 强制容量为 2 的幂，使得后续可以安全使用位掩码代替取模
        let cap_frames = cap_frames.max(256).next_power_of_two();
        let len = (cap_frames as usize) * 2;
        let mut buf = Vec::with_capacity(len);
        buf.resize_with(len, || AtomicU32Cell::new(0));
        Self {
            cap_frames,
            buf,
            hard_start_enabled: AtomicBool::new(false),
            base_frame: AtomicU64::new(0),
            write_frame: AtomicU64::new(0),
        }
    }

    pub(crate) fn set_hard_start_enabled(&self, enabled: bool) {
        self.hard_start_enabled.store(enabled, Ordering::Release);
    }

    pub(crate) fn is_hard_start_enabled(&self) -> bool {
        self.hard_start_enabled.load(Ordering::Acquire)
    }

    pub(crate) fn reset(&self, start_frame: u64) {
        self.base_frame.store(start_frame, Ordering::Release);
        self.write_frame.store(start_frame, Ordering::Release);
    }

    pub(crate) fn write_interleaved(&self, start_frame: u64, pcm: &[f32]) {
        let frames = pcm.len() / 2;
        if frames == 0 {
            return;
        }

        // Ensure the window never exceeds capacity.
        let mut base = self.base_frame.load(Ordering::Acquire);
        let end_frame = start_frame.saturating_add(frames as u64);
        if end_frame.saturating_sub(base) > self.cap_frames {
            base = end_frame.saturating_sub(self.cap_frames);
            self.base_frame.store(base, Ordering::Release);
        }

        // 定义掩码
        let mask = self.cap_frames - 1;

        // 提前计算跳过的帧，消除循环里的 if
        let skip = if start_frame < base {
            (base - start_frame) as usize
        } else {
            0
        };

        for i in skip..frames {
            let f = start_frame + (i as u64);
            let idx = ((f & mask) as usize) * 2; // 使用掩码替代 %
            self.buf[idx].store(pcm[i * 2].to_bits(), Ordering::Relaxed);
            self.buf[idx + 1].store(pcm[i * 2 + 1].to_bits(), Ordering::Relaxed);
        }

        let prev = self.write_frame.load(Ordering::Acquire);
        if end_frame > prev {
            self.write_frame.store(end_frame, Ordering::Release);
        }
    }

    pub(crate) fn read_frame(&self, frame: u64) -> Option<(f32, f32)> {
        let base = self.base_frame.load(Ordering::Acquire);
        let write = self.write_frame.load(Ordering::Acquire);
        if frame < base || frame >= write {
            return None;
        }
        let mask = self.cap_frames - 1; // 定义掩码
        let idx = ((frame & mask) as usize) * 2; // 使用掩码替代 %
        let l = f32::from_bits(self.buf[idx].load(Ordering::Relaxed));
        let r = f32::from_bits(self.buf[idx + 1].load(Ordering::Relaxed));
        Some((l, r))
    }

    pub(crate) fn is_window_covered(&self, start_frame: u64, frames: u64) -> bool {
        if frames == 0 {
            return true;
        }
        let end = start_frame.saturating_add(frames);
        let base = self.base_frame.load(Ordering::Acquire);
        let write = self.write_frame.load(Ordering::Acquire);
        start_frame >= base && end <= write
    }

    // Reads a contiguous interleaved stereo window. Returns false if not fully covered.
    pub(crate) fn read_interleaved_into(&self, start_frame: u64, out: &mut [f32]) -> bool {
        if out.len() % 2 != 0 {
            return false;
        }
        let frames = (out.len() / 2) as u64;
        if !self.is_window_covered(start_frame, frames) {
            return false;
        }
        let mask = self.cap_frames - 1; // 定义掩码
        for i in 0..(frames as usize) {
            let f = start_frame + (i as u64);
            let idx = ((f & mask) as usize) * 2; // 使用掩码替代 %
            out[i * 2] = f32::from_bits(self.buf[idx].load(Ordering::Relaxed));
            out[i * 2 + 1] = f32::from_bits(self.buf[idx + 1].load(Ordering::Relaxed));
        }
        true
    }
}
