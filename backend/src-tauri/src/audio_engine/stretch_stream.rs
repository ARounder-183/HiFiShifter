use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;

use super::ring::StreamRingStereo;
use super::types::ResampledStereo;

/// 启动 stretch_stream 后台 worker。
///
/// Worker 使用 Signalsmith Stretch 实时拉伸器将 `src` 中 `[src_start, src_end)` 范围的 PCM
/// 以 `playback_rate` 速率写入 `ring`，供音频回调低延迟读取。
///
/// # 参数
/// - `ring`: 写入目标 ring buffer
/// - `src`: 源 PCM（已重采样到引擎采样率）
/// - `src_start`/`src_end`: 源帧范围（exclusive end）
/// - `playback_rate`: 播放速率（> 1 加速，< 1 减速）
/// - `start_frame`: clip 在 timeline 上的起始帧（用于 local → abs 转换）
/// - `length_frames`: clip 在 timeline 上的总帧数
/// - `repeat`: 是否循环
/// - `silence_frames`: leading silence 帧数（slip-edit 产生）
/// - `out_rate`: 引擎输出采样率
/// - `position_frames`: 当前播放头（绝对帧）
/// - `is_playing`: 播放状态
/// - `epoch`: per-clip epoch，用于 cancel 检测
/// - `my_epoch`: 本 worker 启动时的 epoch 值
pub(crate) fn spawn_stretch_stream(
    ring: Arc<StreamRingStereo>,
    src: ResampledStereo,
    src_start: u64,
    src_end: u64,
    playback_rate: f64,
    start_frame: u64,
    length_frames: u64,
    repeat: bool,
    silence_frames: u64,
    out_rate: u32,
    position_frames: Arc<AtomicU64>,
    is_playing: Arc<AtomicBool>,
    epoch: Arc<AtomicU64>,
    my_epoch: u64,
) {
    let ring_for_thread = ring;
    let local0 = position_frames
        .load(Ordering::Relaxed)
        .saturating_sub(start_frame);

    thread::spawn(move || {
        let pr = playback_rate;
        let time_ratio = 1.0 / pr.max(1e-6);
        eprintln!(
            "[StretchStream] Starting worker: playback_rate={:.3}, time_ratio={:.6}",
            pr, time_ratio
        );
        eprintln!(
            "[StretchStream] Interpretation: playback_rate={:.3}x means audio plays {:.3}x faster",
            pr, pr
        );
        eprintln!("[StretchStream] Signalsmith time_ratio={:.6} means stretched duration is {:.6}x original", time_ratio, time_ratio);
        let mut rb =
            match crate::sstretch::SignalsmithRealtimeStretcher::new(out_rate, 2, time_ratio) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!(
                        "[StretchStream ERROR] Failed to create SignalsmithStretch: {}",
                        e
                    );
                    return;
                }
            };

        let src_pcm = src.pcm.as_slice();
        let src_total = src.frames as u64;

        let mut out_cursor: u64 = local0;
        let mut in_cursor: u64 = src_start;

        let mut in_block: Vec<f32> = vec![0.0; 1024 * 2];
        let mut out_block: Vec<f32> = Vec::with_capacity(2048 * 2);

        loop {
            if epoch.load(Ordering::Relaxed) != my_epoch {
                break;
            }
            if !is_playing.load(Ordering::Relaxed) {
                std::thread::sleep(std::time::Duration::from_millis(8));
                continue;
            }

            let now_abs = position_frames.load(Ordering::Relaxed);
            let local = now_abs.saturating_sub(start_frame);
            if local >= length_frames {
                std::thread::sleep(std::time::Duration::from_millis(8));
                continue;
            }

            // Leading silence region (slip-edit past the source start).
            if local < silence_frames {
                std::thread::sleep(std::time::Duration::from_millis(4));
                continue;
            }

            let local_audio = local.saturating_sub(silence_frames);

            // Reset on large jumps (seek).
            let base = ring_for_thread.base_frame.load(Ordering::Acquire);
            let write = ring_for_thread.write_frame.load(Ordering::Acquire);
            if local < base || local > write.saturating_add(4096) {
                let _ = rb.reset(time_ratio);
                ring_for_thread.reset(local);
                out_cursor = local;

                let start_in = (local_audio as f64 * pr).floor().max(0.0) as u64;
                if repeat {
                    let loop_len = src_end.saturating_sub(src_start).max(1);
                    in_cursor = src_start + (start_in % loop_len);
                } else {
                    in_cursor = (src_start + start_in).min(src_end);
                }
            }

            // Maintain some lookahead.
            let ahead = write.saturating_sub(local);
            if ahead >= 4096 {
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }

            // Fill an input block from the source window.
            let mut want_in = 1024usize;

            if !repeat {
                // 非循环模式
                if in_cursor >= src_end || in_cursor >= src_total {
                    std::thread::sleep(std::time::Duration::from_millis(4));
                    continue;
                }
                let remain = src_end
                    .saturating_sub(in_cursor)
                    .min(src_total.saturating_sub(in_cursor)) as usize;
                want_in = want_in.min(remain.max(1));

                // 直接使用底层内存拷贝 (memcpy)，完成数据填充
                let start_idx = (in_cursor as usize) * 2;
                let end_idx = start_idx + want_in * 2;
                in_block[..want_in * 2].copy_from_slice(&src_pcm[start_idx..end_idx]);
            } else {
                // 循环模式
                // 将循环不变量全部提到 for 循环外部，避免重复计算
                let loop_len = src_end.saturating_sub(src_start).max(1);
                let cursor_offset = in_cursor.saturating_sub(src_start);
                let max_safe_idx = src_total.saturating_sub(1);

                for i in 0..want_in {
                    let within = (cursor_offset + i as u64) % loop_len;
                    let src_f = (src_start + within).min(max_safe_idx);
                    let si = (src_f as usize) * 2;

                    // 上面已经用 max_safe_idx 锁死了上限，这里不可能越界。
                    // 放弃 .get().copied().unwrap_or(0.0)，直接裸索引
                    in_block[i * 2] = src_pcm[si];
                    in_block[i * 2 + 1] = src_pcm[si + 1];
                }
            }

            let _ = rb.process_interleaved(&in_block[..want_in * 2], false);
            in_cursor = in_cursor.saturating_add(want_in as u64);
            if repeat {
                let loop_len = src_end.saturating_sub(src_start).max(1);
                if in_cursor >= src_end {
                    in_cursor = src_start + ((in_cursor - src_start) % loop_len);
                }
            }

            out_block.clear();
            for _ in 0..4 {
                let got = rb
                    .retrieve_interleaved_into(&mut out_block, 1024)
                    .unwrap_or_default();
                if got == 0 {
                    break;
                }
            }

            if !out_block.is_empty() {
                ring_for_thread.write_interleaved(out_cursor, out_block.as_slice());
                out_cursor = out_cursor.saturating_add((out_block.len() / 2) as u64);
            }
        }
    });
}
