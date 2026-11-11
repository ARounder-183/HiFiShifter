//! 默认输出设备 watcher。
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

pub(crate) fn spawn_default_output_watcher(
    tx: Sender<EngineCommand>,
    shutdown: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let host = cpal::default_host();
        let mut last: Option<OutputDeviceIdentity> = None;
        let mut pending: Option<OutputDeviceIdentity> = None;
        let mut pending_hits = 0u8;
        let mut first = true;
        while !shutdown.load(Ordering::Relaxed) {
            let current = host.default_output_device().and_then(|device| {
                device
                    .default_output_config()
                    .ok()
                    .map(|config| OutputDeviceIdentity::from_device(&device, &config))
            });
            if current == pending && current.is_some() {
                pending_hits = pending_hits.saturating_add(1);
            } else {
                pending = current.clone();
                pending_hits = 1;
            }
            if first {
                // worker 启动时自己会建流；首轮只记录基线，避免重复通知。
                last = current;
                first = false;
            } else if pending_hits >= 2 && output_device_changed(last.as_ref(), current.as_ref()) {
                last = current;
                if tx.send(EngineCommand::DefaultOutputDeviceChanged).is_err() {
                    break;
                }
            } else if current.is_none() {
                last = None;
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
