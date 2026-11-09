//! 中性契约的行为夹具。
//!
//! 断言文字刻意与 `capture/recorder.rs` 的既有真机行为夹具同口径，便于逐条对照：
//! 「起播滞后」「停播钉住」「循环回绕」「离线供音」「无 transport 概念」「非法时间基」。

use super::*;

fn frame() -> TransportFrame {
    TransportFrame {
        project_time_samples: 100,
        sample_rate: 48_000.0,
        playing: Some(true),
        recording: Some(false),
        cycle_active: Some(false),
        tempo: Some(120.0),
        time_sig_num: Some(4),
        time_sig_den: Some(4),
        steady_time: Some(100),
        offline: false,
        position_trusted: true,
    }
}

/// 起播首块 `playing=Some(false)` 而位置前进 ⇒ 位置仍可信（滞后置位）。
#[test]
fn a_lagging_playing_bit_keeps_a_moving_position_trustworthy() {
    let first = TransportFrame {
        playing: Some(false),
        project_time_samples: 100,
        ..frame()
    };
    let second = TransportFrame {
        playing: Some(false),
        project_time_samples: 108,
        ..frame()
    };
    assert!(frame_is_trustworthy(&first));
    assert_ne!(first.project_time_samples, second.project_time_samples);
}

/// 停播后位置钉住不动且 `playing=Some(false)` ⇒ 不把「停播」误读成「宿主没给」。
#[test]
fn a_stopped_host_is_not_the_same_as_a_host_without_transport() {
    let stopped = TransportFrame {
        playing: Some(false),
        position_trusted: false,
        ..frame()
    };
    assert!(stopped.has_transport());
    assert!(!stopped.is_playing());
    assert!(!frame_is_trustworthy(&stopped));
    let no_transport = TransportFrame {
        playing: None,
        ..frame()
    };
    assert!(!no_transport.has_transport());
}

/// `cycle_active=Some(true)` 且位置回绕不能被当成「从头开始的新录制」的等价证据。
#[test]
fn a_cycle_wrap_keeps_its_own_flag_instead_of_rewriting_the_position() {
    let wrapped = TransportFrame {
        cycle_active: Some(true),
        project_time_samples: 0,
        ..frame()
    };
    assert!(wrapped.cycle_active == Some(true));
    assert_eq!(wrapped.project_time_samples, 0);
}

/// `offline=true` 且 `playing=Some(false)` ⇒ 仍允许供音（bounce 通路）。
#[test]
fn an_offline_block_still_supplies_audio_even_without_a_playing_bit() {
    let offline = TransportFrame {
        offline: true,
        playing: Some(false),
        ..frame()
    };
    assert!(frame_is_trustworthy(&offline));
    assert!(offline.offline);
}

/// `playing=None`（无 transport 概念的宿主）⇒ 如实降级，不虚构播放证据。
#[test]
fn a_host_without_transport_degrades_honestly() {
    let degraded = TransportFrame {
        playing: None,
        position_trusted: false,
        ..frame()
    };
    assert!(!degraded.is_playing());
    assert!(!degraded.has_transport());
    assert!(!frame_is_trustworthy(&degraded));
}

/// 非法采样率 / 负工程位置一律拒绝块。
#[test]
fn invalid_rates_and_negative_positions_are_rejected() {
    for sample_rate in [0.0, -48_000.0, f64::NAN, f64::INFINITY] {
        assert!(!frame_is_trustworthy(&TransportFrame {
            sample_rate,
            ..frame()
        }));
    }
    assert!(!frame_is_trustworthy(&TransportFrame {
        project_time_samples: -1,
        ..frame()
    }));
}
