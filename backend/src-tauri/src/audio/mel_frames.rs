/*
 * audio/mel_frames.rs - mel 帧数的解析计算。
 *
 * 主要内容：
 * - `predicted_mel_frames`：由（已重采样到模型采样率的）音频长度推算 mel 帧数。
 *
 * 与其他模块的关系：
 * - `vocoder/nsf_hifigan_onnx.rs` 的 `mel_from_audio_fast` 与本模块共用这一
 *   个公式（单一实现），并据此在**付出 STFT + mel 矩阵乘法之前**预判分块
 *   边界，从而支持"全部分块命中缓存时跳过特征提取"的快路径（见 P1-5）。
 * - 本模块刻意**不**放在 `vocoder/` 下（那里被 `feature = "onnx"` 门控），
 *   因此其公式在无 onnx 的构建中也能编译并被单元测试覆盖 —— 这个公式一旦
 *   与真实 mel 提取不一致，"快路径"就会产出错误长度的音频，是本项目里
 *   少数必须被测试锁死的数值契约之一。
 */

/// 由音频长度推算 mel 帧数。
///
/// 与 `mel_from_audio_fast` 完全一致：
/// - 左右按 `(win_size - hop) / 2`（右偏 1）做 reflect padding；
/// - `y_len < win_size` 时仍产出 1 帧（全零经 log 压缩后为极小值）；
/// - 否则 `1 + (y_len - win_size) / hop`。
///
/// 参数：
/// - `audio_len`：**未加 padding** 的音频样本数。
/// - `win_size` / `hop`：STFT 窗长与步长（样本）。
#[allow(dead_code)]
pub(crate) fn predicted_mel_frames(audio_len: usize, win_size: usize, hop: usize) -> usize {
    if win_size == 0 || hop == 0 {
        return 0;
    }
    let pad_left = ((win_size as isize - hop as isize) / 2).max(0) as usize;
    let pad_right = ((win_size as isize - hop as isize + 1) / 2).max(0) as usize;
    let y_len = audio_len
        .saturating_add(pad_left)
        .saturating_add(pad_right);
    if y_len < win_size {
        1
    } else {
        1 + (y_len - win_size) / hop
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PC-NSF-HiFiGAN 的实际配置。
    const WIN: usize = 2048;
    const HOP: usize = 512;

    /// 与"逐帧滑窗覆盖"的朴素参考实现对照。
    fn reference_frames(audio_len: usize, win: usize, hop: usize) -> usize {
        let pad_left = (win - hop) / 2;
        let pad_right = (win - hop + 1) / 2;
        let y = audio_len + pad_left + pad_right;
        if y < win {
            return 1;
        }
        1 + (y - win) / hop
    }

    #[test]
    fn matches_reference_for_a_range_of_lengths() {
        for len in [0usize, 1, 2, 511, 512, 1023, 1536, 2047, 2048, 2049, 4096, 48_000, 1_000_003] {
            assert_eq!(
                predicted_mel_frames(len, WIN, HOP),
                reference_frames(len, WIN, HOP),
                "mismatch at audio_len={len}"
            );
        }
    }

    #[test]
    fn degenerate_config_returns_zero() {
        assert_eq!(predicted_mel_frames(1000, 0, HOP), 0);
        assert_eq!(predicted_mel_frames(1000, WIN, 0), 0);
    }

    #[test]
    fn very_short_audio_still_yields_one_frame() {
        assert_eq!(predicted_mel_frames(0, WIN, HOP), 1);
        assert_eq!(predicted_mel_frames(10, WIN, HOP), 1);
    }

    #[test]
    fn padded_length_at_or_above_window_yields_first_extra_frame() {
        // padding 后恰好等于 win_size → 1 帧；再多 1 个样本仍不足一个 hop → 1 帧。
        let pad = (WIN - HOP) / 2 + (WIN - HOP + 1) / 2; // 1536
        assert_eq!(predicted_mel_frames(WIN - pad, WIN, HOP), 1);
        assert_eq!(predicted_mel_frames(WIN - pad + HOP, WIN, HOP), 2);
    }

    #[test]
    fn frame_count_is_monotonic_and_grows_one_per_hop() {
        let base = 40_000usize;
        let a = predicted_mel_frames(base, WIN, HOP);
        assert_eq!(predicted_mel_frames(base + HOP, WIN, HOP), a + 1);
        assert!(predicted_mel_frames(base + HOP - 1, WIN, HOP) <= a + 1);
    }

    #[test]
    fn saturating_on_absurd_lengths_does_not_panic() {
        // 仅要求不 panic / 不溢出（usize::MAX 也不应回绕）。
        let v = predicted_mel_frames(usize::MAX, WIN, HOP);
        assert!(v >= 1);
    }
}
