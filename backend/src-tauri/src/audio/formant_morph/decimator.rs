/*
 * formant_morph/decimator.rs - 共振峰分析用的抗混叠降采样器。
 *
 * 主要内容：
 * - `Decimator`：FIR 低通 + 整数抽取，把任意采样率的 mono PCM 降到共振峰
 *   分析域（目标 ~11 kHz）。分析速率取 `in_rate / D`（D 为整数抽取因子），
 *   因此调用方必须用 `analysis_rate()` 的返回值做频率换算，而不是假设
 *   恰好等于目标速率（例如 48k → 12k，44.1k → 11.025k）。
 *
 * 与其他模块的关系：
 * - 仅被 `formant_morph/mod.rs` 与 `formant_morph/analysis.rs` 使用。
 * - 刻意不复用 `mixdown::linear_resample_interleaved`（线性插值无抗混叠，
 *   48k→11k 会产生混叠并污染 LPC 极点估计）。
 *
 * 设计要点：
 * - Blackman 窗 sinc 低通（N=129）：阻带 ≥ -58 dB，过渡带 ≈ 5.5·in_rate/N，
 *   通带上沿 ≈ 0.45·in_rate/D − trans/2，保证 F2（≤3.2 kHz）无衰减。
 * - D = 1 时严格恒等直通（逐样本相等），不做滤波。
 * - 滤波器有 (N-1)/2 样本的群时延，输出按整体时延对齐（调用方按缓冲处理，
 *   不做样本级对齐；帧分析对固定时延不敏感）。
 *
 * 维护说明：
 * - 严禁在 D > 1 时跳过低通直接抽取（会混叠）。
 * - taps 归一化到 DC 增益为 1，保证通带响度不变。
 */

/// FIR 阶数（抽头数-1）。129 抽头在 48k 下过渡带约 ±2 kHz。
const FIR_TAPS: usize = 129;

/// 降采样器：FIR 抗混叠 + 整数抽取。
pub struct Decimator {
    /// 归一化 FIR 低通抽头（DC 增益 = 1）。
    taps: Vec<f32>,
    /// 输入历史（环形延迟线），长度 = taps 长度。
    delay: Vec<f32>,
    /// 延迟线写指针（指向最旧样本位置）。
    write_pos: usize,
    /// 整数抽取因子。
    decim: usize,
    /// 抽取相位计数。
    phase: usize,
    /// 实际分析采样率（= in_rate / decim）。
    analysis_rate: f32,
}

impl Decimator {
    /// 创建降采样器。
    ///
    /// 流程：
    /// 1. `decim = round(in_rate / target_rate)`，至少为 1。
    /// 2. `decim == 1` → 恒等直通（不建滤波器）。
    /// 3. 否则生成 Blackman 窗 sinc 低通，截止 = 0.45 × in_rate / decim，
    ///    抽头归一化到 DC 增益 1。
    ///
    /// 参数：
    /// - `in_rate`：输入采样率。
    /// - `target_rate`：目标分析速率（近似值，实际为 in_rate/decim）。
    ///
    /// 返回：`None` 表示参数非法（采样率为 0）。
    pub fn new(in_rate: u32, target_rate: u32) -> Option<Decimator> {
        if in_rate == 0 || target_rate == 0 {
            return None;
        }
        let decim = ((in_rate as f32 / target_rate as f32).round() as usize).max(1);
        let analysis_rate = in_rate as f32 / decim as f32;
        if decim == 1 {
            return Some(Decimator {
                taps: Vec::new(),
                delay: Vec::new(),
                write_pos: 0,
                decim: 1,
                phase: 0,
                analysis_rate,
            });
        }

        let taps = design_lowpass(in_rate as f32, decim as f32, FIR_TAPS);
        Some(Decimator {
            delay: vec![0.0; taps.len()],
            write_pos: 0,
            decim,
            phase: 0,
            analysis_rate,
            taps,
        })
    }

    /// 实际分析采样率（in_rate / decim）。
    pub fn analysis_rate(&self) -> f32 {
        self.analysis_rate
    }

    /// 处理一段 mono 输入，返回抽取后的样本。
    ///
    /// 规则：
    /// - `decim == 1`：原样拷贝（严格恒等）。
    /// - 其余：FIR 卷积后每 `decim` 个输入样本输出 1 个。
    ///   流式安全：内部保留延迟线，可分块调用。
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        if self.decim == 1 {
            return input.to_vec();
        }
        let mut out = Vec::with_capacity(input.len() / self.decim + 1);
        for &sample in input {
            self.delay[self.write_pos] = sample;
            self.write_pos = (self.write_pos + 1) % self.delay.len();
            self.phase += 1;
            if self.phase >= self.decim {
                self.phase = 0;
                out.push(self.convolve());
            }
        }
        out
    }

    /// 用当前延迟线计算一个滤波输出样本。
    fn convolve(&self) -> f32 {
        // delay 为环形缓冲：write_pos 指向最旧样本的写入位（下一帧覆盖它），
        // 因此最新样本在 (write_pos - 1 + len) % len，按 tap 序从新到旧累加。
        let n = self.delay.len();
        let mut acc = 0.0_f32;
        for (k, &tap) in self.taps.iter().enumerate() {
            let idx = (self.write_pos + n - 1 - k) % n;
            acc += tap * self.delay[idx];
        }
        acc
    }
}

/// 设计 Blackman 窗 sinc 低通（线性相位，DC 增益 1）。
///
/// 参数：`in_rate` 输入采样率，`decim` 抽取因子，`taps` 抽头数。
/// 截止频率取 0.45 × in_rate / decim（新奈奎斯特的 90%）。
fn design_lowpass(in_rate: f32, decim: f32, taps: usize) -> Vec<f32> {
    let cutoff = 0.45 * in_rate / decim;
    let omega = 2.0 * std::f32::consts::PI * cutoff / in_rate;
    let center = (taps - 1) as f32 / 2.0;
    let mut out = vec![0.0_f32; taps];
    for (i, tap) in out.iter_mut().enumerate() {
        let m = i as f32 - center;
        // Blackman 窗
        let w = 0.42
            - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos()
            + 0.08 * (4.0 * std::f32::consts::PI * i as f32 / (taps - 1) as f32).cos();
        let sinc = if m.abs() < 1.0e-9 {
            omega / std::f32::consts::PI
        } else {
            (omega * m).sin() / (std::f32::consts::PI * m)
        };
        *tap = sinc * w;
    }
    let sum: f32 = out.iter().sum();
    if sum.abs() > 1.0e-12 {
        for tap in out.iter_mut() {
            *tap /= sum;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN_RATE: u32 = 48_000;

    /// 生成正弦测试信号。
    fn sine(freq_hz: f32, sr: u32, secs: f32, amp: f32) -> Vec<f32> {
        (0..(sr as f32 * secs) as usize)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        if x.is_empty() {
            return 0.0;
        }
        (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn identity_when_no_decimation_needed() {
        // 16k / 11.025k → decim = round(1.45) = 1 → 恒等直通
        let mut d = Decimator::new(16_000, 11_025).unwrap();
        assert_eq!(d.decim, 1);
        let input = sine(440.0, 16_000, 0.05, 0.5);
        let out = d.process(&input);
        assert_eq!(out, input, "decim=1 must be byte-identical passthrough");
    }

    #[test]
    fn rejects_zero_rates() {
        assert!(Decimator::new(0, 11_025).is_none());
        assert!(Decimator::new(48_000, 0).is_none());
    }

    #[test]
    fn decimation_factor_and_rate() {
        let d = Decimator::new(48_000, 11_025).unwrap();
        assert_eq!(d.decim, 4);
        assert!((d.analysis_rate() - 12_000.0).abs() < 1.0);
        let d = Decimator::new(44_100, 11_025).unwrap();
        assert_eq!(d.decim, 4);
        assert!((d.analysis_rate() - 11_025.0).abs() < 0.5);
    }

    #[test]
    fn passband_content_is_preserved() {
        // 220 Hz 在通带内：降采样后 RMS 变化 < 0.2 dB
        let mut d = Decimator::new(IN_RATE, 11_025).unwrap();
        let input = sine(220.0, IN_RATE, 0.25, 0.5);
        let out = d.process(&input);
        let ratio = rms(&out) / rms(&input);
        let db = 20.0 * ratio.log10();
        assert!(
            db.abs() < 0.2,
            "passband gain must stay within 0.2 dB, got {db:.3} dB"
        );
    }

    #[test]
    fn out_of_band_content_is_suppressed() {
        // 13 kHz 超出 12k 输出的奈奎斯特（6 kHz）：不滤波会混叠到 1 kHz。
        // 要求抑制 ≥ 40 dB（输出 RMS < 输入的 1%）。
        let mut d = Decimator::new(IN_RATE, 11_025).unwrap();
        let input = sine(13_000.0, IN_RATE, 0.25, 0.5);
        let out = d.process(&input);
        let ratio = rms(&out) / rms(&input);
        assert!(
            ratio < 0.01,
            "out-of-band suppression must be >= 40 dB, got ratio={ratio:.5}"
        );
    }

    #[test]
    fn streaming_equals_bulk() {
        // 分块调用与一次性调用结果一致（延迟线状态正确）。
        let input = sine(220.0, IN_RATE, 0.2, 0.5);
        let mut bulk = Decimator::new(IN_RATE, 11_025).unwrap();
        let whole = bulk.process(&input);

        let mut streamed = Decimator::new(IN_RATE, 11_025).unwrap();
        let mut pieces = Vec::new();
        for chunk in input.chunks(997) {
            pieces.extend(streamed.process(chunk));
        }
        assert_eq!(pieces.len(), whole.len());
        for (a, b) in pieces.iter().zip(whole.iter()) {
            assert!((a - b).abs() < 1.0e-6, "streamed output must match bulk");
        }
    }
}
