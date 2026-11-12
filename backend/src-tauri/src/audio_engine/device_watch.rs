//! 默认输出设备 watcher。
//!
//! 【已知限制（Linux）】cpal 0.18 的 Linux 后端只有 ALSA（没有 pulse / pipewire
//! feature），`default_output_device()` 指向 ALSA 的 `default` 设备。多数
//! PipeWire / PulseAudio 桌面会把它桥接到用户选择的默认 sink，但在手工配置 ALSA
//! 或桥接缺失的机器上，"在系统里换默认输出"不一定跟随 —— 需要在系统层面把 ALSA
//! 的 default 指向目标 sink。这是**平台能力限制**，不是本模块的判据问题：判据
//! （身份比对 + 去抖）在两个平台上完全一致。
//!
//! 每 500ms 查询一次系统默认输出；新身份连续两次一致才通知 worker，
//! 避免 USB / HDMI 枚举瞬态。worker 收到命令后自己重新查询 cpal，命令
//! 通道里不传平台句柄。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};

use hifishifter_kernel::engine_command::EngineCommand;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OutputDeviceIdentity {
    pub id: String,
    pub name: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: String,
}

impl OutputDeviceIdentity {
    pub(crate) fn from_device(
        device: &cpal::Device,
        config: &cpal::SupportedStreamConfig,
    ) -> Self {
        let id = device.id().map(|value| value.to_string()).unwrap_or_default();
        let name = device
            .description()
            .map(|description| description.name().to_string())
            .unwrap_or_else(|_| device.to_string());
        Self {
            id,
            name,
            sample_rate: config.sample_rate(),
            channels: config.channels(),
            sample_format: format!("{:?}", config.sample_format()),
        }
    }
}

pub(crate) fn output_device_changed(
    previous: Option<&OutputDeviceIdentity>,
    current: Option<&OutputDeviceIdentity>,
) -> bool {
    match (previous, current) {
        (None, None) => false,
        (None, Some(_)) | (Some(_), None) => true,
        (Some(a), Some(b)) => {
            if !a.id.is_empty() && !b.id.is_empty() {
                a.id != b.id
                    || a.sample_rate != b.sample_rate
                    || a.channels != b.channels
                    || a.sample_format != b.sample_format
            } else {
                a != b
            }
        }
    }
}

/// 按新旧采样率折算"当前位置"，返回新采样率域下的 `(base, position)`。
///
/// 【为什么要一个函数】设备切换要按新 sr 重算帧数；这段算术如果就地写在
/// 切换逻辑里，就只能靠"拔设备"来验证。抽出来之后，44.1k↔48k、0 位置、
/// 溢出三种情形都有逐值断言。
///
/// 时间线播放的 `base` 恒为 0，因此返回值里的 `base` 也是 0；保留两个分量
/// 是为了与引擎的 `(base + position)` 口径完全一致，避免出现第三套算法。
pub(crate) fn rescale_position_frames(
    base: u64,
    position: u64,
    old_sr: u32,
    new_sr: u32,
) -> (u64, u64) {
    let old_sr = u64::from(old_sr.max(1));
    let new_sr = u64::from(new_sr.max(1));
    let total = u128::from(base) + u128::from(position);
    let scaled = total.saturating_mul(u128::from(new_sr)) / u128::from(old_sr);
    (0, scaled.min(u128::from(u64::MAX)) as u64)
}

/// 去抖状态机：**纯逻辑**，不碰线程与设备，因此可以逐状态单测。
///
/// 规则与旧实现一致（连续两次观测一致才通知），但基线改为**外部注入**：
/// 旧实现让 watcher 自己把首轮观测当作基线，而 worker 可能恰好用了另一个
/// 设备建流 —— 两者基线一旦不同，真正的切换会被判成"没变"而永远漏掉。
pub(crate) struct DeviceDebounce {
    last: Option<OutputDeviceIdentity>,
    pending: Option<OutputDeviceIdentity>,
    pending_hits: u8,
    baseline_initialized: bool,
}

impl DeviceDebounce {
    /// `initial` 必须是 worker **真正建流用的那个设备**身份。
    pub(crate) fn new(initial: Option<OutputDeviceIdentity>) -> Self {
        let baseline_initialized = initial.is_some();
        Self {
            last: initial.clone(),
            pending: initial,
            pending_hits: 0,
            baseline_initialized,
        }
    }

    /// 喂一次观测；返回 `Some(identity)` 表示应当通知 worker 重建。
    pub(crate) fn observe(
        &mut self,
        current: Option<OutputDeviceIdentity>,
    ) -> Option<OutputDeviceIdentity> {
        let same_as_pending = current == self.pending && current.is_some();
        if same_as_pending {
            self.pending_hits = self.pending_hits.saturating_add(1);
        } else {
            self.pending = current.clone();
            self.pending_hits = 1;
        }
        if !self.baseline_initialized {
            // 还没有基线：这一轮只记录，不通知（worker 的流已经建好了）。
            self.last = current;
            self.baseline_initialized = true;
            return None;
        }
        if self.pending_hits >= 2 && output_device_changed(self.last.as_ref(), current.as_ref()) {
            self.last = current.clone();
            return current;
        }
        if current.is_none() {
            // 设备消失：立刻把基线清空，使"同一台设备重新插回"也算变化。
            // 但**不通知** —— 没有设备可建流，通知只会让 worker 白跑一趟。
            self.last = None;
        }
        None
    }
}

pub(crate) fn spawn_default_output_watcher(
    tx: Sender<EngineCommand>,
    shutdown: Arc<AtomicBool>,
    initial_identity: Option<OutputDeviceIdentity>,
) {
    thread::spawn(move || {
        let host = cpal::default_host();
        let mut debounce = DeviceDebounce::new(initial_identity);
        while !shutdown.load(Ordering::Relaxed) {
            let current = host.default_output_device().and_then(|device| {
                device
                    .default_output_config()
                    .ok()
                    .map(|config| OutputDeviceIdentity::from_device(&device, &config))
            });
            if debounce.observe(current).is_some()
                && tx
                    .send(EngineCommand::DefaultOutputDeviceChanged { force: false })
                    .is_err()
            {
                break;
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(id: &str, name: &str, rate: u32) -> OutputDeviceIdentity {
        OutputDeviceIdentity {
            id: id.to_string(),
            name: name.to_string(),
            sample_rate: rate,
            channels: 2,
            sample_format: "F32".to_string(),
        }
    }

    #[test]
    fn debounce_notifies_once_per_real_change() {
        let mut d = DeviceDebounce::new(Some(identity("a", "A", 44100)));
        // 同一设备重复观测：永不通知。
        assert_eq!(d.observe(Some(identity("a", "A", 44100))), None);
        assert_eq!(d.observe(Some(identity("a", "A", 44100))), None);

        // 新设备：第一次观测只是 pending，第二次才通知（去抖）。
        assert_eq!(d.observe(Some(identity("b", "B", 48000))), None);
        assert_eq!(
            d.observe(Some(identity("b", "B", 48000))),
            Some(identity("b", "B", 48000))
        );
        // 通知之后基线已推进，再观测不重复通知。
        assert_eq!(d.observe(Some(identity("b", "B", 48000))), None);
    }

    /// 采样率变化也必须算"变化"：它决定快照要不要按新 sr 重建。
    #[test]
    fn debounce_treats_a_sample_rate_change_as_a_change() {
        let mut d = DeviceDebounce::new(Some(identity("a", "A", 44100)));
        assert_eq!(d.observe(Some(identity("a", "A", 48000))), None);
        assert_eq!(
            d.observe(Some(identity("a", "A", 48000))),
            Some(identity("a", "A", 48000))
        );
    }

    /// 设备消失 → 重新出现（同一台）：必须再通知一次。
    ///
    /// 【为什么】USB 拔出再插回是同一个 id/name/format；若基线在消失期间不
    /// 复位，重插那次会被判成"没变"，用户看到的是"插回去了但还是没声"。
    #[test]
    fn debounce_notifies_when_the_same_device_comes_back() {
        let mut d = DeviceDebounce::new(Some(identity("a", "A", 44100)));
        assert_eq!(d.observe(None), None);
        assert_eq!(d.observe(None), None);
        assert_eq!(d.observe(Some(identity("a", "A", 44100))), None);
        assert_eq!(
            d.observe(Some(identity("a", "A", 44100))),
            Some(identity("a", "A", 44100))
        );
    }

    /// 没有初始基线（worker 还没建流）时，首轮观测只建立基线。
    #[test]
    fn debounce_uses_the_first_observation_as_the_baseline() {
        let mut d = DeviceDebounce::new(None);
        assert_eq!(d.observe(Some(identity("a", "A", 44100))), None);
        assert_eq!(d.observe(Some(identity("a", "A", 44100))), None);
    }

    /// 位置帧换算：设备切换时按新旧采样率折算，保证"听感位置不跳"。
    #[test]
    fn rescale_position_frames_follows_the_new_sample_rate() {
        // 44.1k 的 44100 帧 = 1s → 48k 的 48000 帧。
        assert_eq!(rescale_position_frames(0, 44_100, 44_100, 48_000), (0, 48_000));
        // 0 位置恒为 0。
        assert_eq!(rescale_position_frames(0, 0, 44_100, 48_000), (0, 0));
        // 1 采样率的退化输入不得除零。
        assert_eq!(rescale_position_frames(0, 10, 0, 0), (0, 10));
        // 溢出钳制而不是回绕。
        assert_eq!(
            rescale_position_frames(u64::MAX, 0, 1, u32::MAX).1,
            u64::MAX
        );
    }

    #[test]
    fn detects_none_some_and_format_changes() {
        assert!(!output_device_changed(None, None));
        assert!(output_device_changed(None, Some(&identity("a", "A", 44100))));
        assert!(output_device_changed(Some(&identity("a", "A", 44100)), None));
        assert!(!output_device_changed(
            Some(&identity("a", "A", 44100)),
            Some(&identity("a", "A", 44100))
        ));
        assert!(output_device_changed(
            Some(&identity("a", "A", 44100)),
            Some(&identity("a", "A", 48000))
        ));
        assert!(output_device_changed(
            Some(&identity("", "A", 44100)),
            Some(&identity("", "B", 44100))
        ));
    }
}
