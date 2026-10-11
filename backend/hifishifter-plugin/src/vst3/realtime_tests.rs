//! `vst3.rs` 的实时路径测试。从内联模块纯搬到这里。

use super::*;
use std::cell::Cell;

std::thread_local! {
    static IN_REALTIME_CALLBACK: Cell<bool> = const { Cell::new(false) };
    static REALTIME_LOGS: Cell<usize> = const { Cell::new(0) };
}

struct ObservingLogger;

impl log::Log for ObservingLogger {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, _: &log::Record<'_>) {
        if IN_REALTIME_CALLBACK.with(Cell::get) {
            REALTIME_LOGS.with(|count| count.set(count.get() + 1));
        }
    }
    fn flush(&self) {}
}

/// setProcessing 和 process 都可在音频线程运行，不能触发同步文件日志。
#[test]
fn realtime_callbacks_do_not_enter_the_file_logger() {
    static LOGGER: ObservingLogger = ObservingLogger;
    log::set_logger(&LOGGER).unwrap();
    log::set_max_level(log::LevelFilter::Info);
    let mut processor = Processor::create().expect("processor creation");
    // SAFETY: 取真实音频处理器子对象地址，而非组件基址。
    let this = unsafe { audio_ptr(&mut *processor) };
    let mut left = [1.0_f32; 2];
    let mut right = [1.0_f32; 2];
    let mut planes = [left.as_mut_ptr(), right.as_mut_ptr()];
    let mut bus = crate::audio_abi::AudioBusBuffers {
        num_channels: 2,
        silence_flags: 0,
        channel_buffers: planes.as_mut_ptr(),
    };
    let mut data = crate::audio_abi::ProcessData {
        num_samples: 2,
        num_outputs: 1,
        outputs: &raw mut bus,
        ..Default::default()
    };
    IN_REALTIME_CALLBACK.with(|flag| flag.set(true));
    // SAFETY: 处理器在两次状态回调期间存活。
    unsafe {
        assert_eq!(audio_set_processing(this, 1), K_RESULT_OK);
        assert_eq!(
            (AUDIO_VTBL.process)(this, (&raw mut data).cast()),
            K_RESULT_OK
        );
        assert_eq!(audio_set_processing(this, 0), K_RESULT_OK);
    }
    IN_REALTIME_CALLBACK.with(|flag| flag.set(false));
    assert_eq!(REALTIME_LOGS.with(Cell::get), 0);
    assert_eq!(left, [0.0; 2]);
    assert_eq!(right, [0.0; 2]);
}
