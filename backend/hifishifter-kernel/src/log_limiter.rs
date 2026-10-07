//! 限流日志原语。
//!
//! 【为什么在内核里】内核模块（渲染器、音高分析、缓存）大量使用
//! [`crate::log_warn_limited`] / [`crate::log_error_limited`]：这些调用点位于循环 /
//! 回调 / 逐轮询路径上，不设限会刷屏挤占日志轮转额度。限流本身与宿主无关 ——
//! 它只用到 `log` crate 与一个进程级哈希表，所以属于内核。
//!
//! app 侧的文件日志 / stderr tee 仍在 `backend/src-tauri/src/logging.rs` 里；
//! 那个模块的 `emit_limited` 现在委托到这里，**两边共用同一个限流状态**。

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 限流窗口：同一调用点在该窗口内最多输出一条。
const RATE_LIMIT_WINDOW: Duration = Duration::from_secs(10);

struct RateLimitState {
    last_emit: Option<Instant>,
    suppressed: u64,
}

enum EmitDecision {
    Emit { previously_suppressed: u64 },
    Suppress,
}

type RateLimitMap = HashMap<(&'static str, u32), RateLimitState>;

static RATE_LIMITER: OnceLock<Mutex<RateLimitMap>> = OnceLock::new();

fn rate_limiter() -> &'static Mutex<RateLimitMap> {
    RATE_LIMITER.get_or_init(|| Mutex::new(HashMap::new()))
}

/// 限流判定的纯逻辑（便于单元测试）：窗口外首条放行并结算上一窗口的
/// 抑制计数；窗口内后续条一律抑制。
fn limited_decision(state: &mut RateLimitState, now: Instant, window: Duration) -> EmitDecision {
    match state.last_emit {
        Some(last) if now.duration_since(last) < window => {
            state.suppressed += 1;
            EmitDecision::Suppress
        }
        _ => {
            let previously_suppressed = std::mem::take(&mut state.suppressed);
            state.last_emit = Some(now);
            EmitDecision::Emit {
                previously_suppressed,
            }
        }
    }
}

fn short_source(file: &str) -> &str {
    file.rsplit(['/', '\\']).next().unwrap_or(file)
}

fn level_filter_of(level: log::Level) -> log::LevelFilter {
    match level {
        log::Level::Error => log::LevelFilter::Error,
        log::Level::Warn => log::LevelFilter::Warn,
        log::Level::Info => log::LevelFilter::Info,
        log::Level::Debug => log::LevelFilter::Debug,
        log::Level::Trace => log::LevelFilter::Trace,
    }
}

/// 限流输出入口。级别被过滤时不做任何计数（没有真正被"抑制"的内容）。
pub fn emit_limited(
    level: log::Level,
    file: &'static str,
    line: u32,
    args: std::fmt::Arguments<'_>,
) {
    let filter = level_filter_of(level);
    if filter > log::STATIC_MAX_LEVEL || filter > log::max_level() {
        return;
    }

    let decision = {
        let mut map = rate_limiter().lock().unwrap_or_else(|e| e.into_inner());
        let state = map.entry((file, line)).or_insert_with(|| RateLimitState {
            last_emit: None,
            suppressed: 0,
        });
        limited_decision(state, Instant::now(), RATE_LIMIT_WINDOW)
    };

    if let EmitDecision::Emit {
        previously_suppressed,
    } = decision
    {
        if previously_suppressed > 0 {
            log::log!(
                level,
                "[throttled] {}:{line} — {previously_suppressed} message(s) suppressed",
                short_source(file),
            );
        }
        log::log!(level, "{args}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limited_decision_first_call_emits() {
        let mut state = RateLimitState {
            last_emit: None,
            suppressed: 0,
        };
        let now = Instant::now();
        match limited_decision(&mut state, now, RATE_LIMIT_WINDOW) {
            EmitDecision::Emit {
                previously_suppressed,
            } => {
                assert_eq!(previously_suppressed, 0);
            }
            EmitDecision::Suppress => panic!("first call must emit"),
        }
    }

    #[test]
    fn limited_decision_suppresses_within_window() {
        let mut state = RateLimitState {
            last_emit: None,
            suppressed: 0,
        };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        for i in 1..=5u64 {
            let at = now + Duration::from_millis(100 * i);
            assert!(matches!(
                limited_decision(&mut state, at, RATE_LIMIT_WINDOW),
                EmitDecision::Suppress
            ));
        }
        assert_eq!(state.suppressed, 5);
    }

    #[test]
    fn limited_decision_flushes_suppressed_count_after_window() {
        let mut state = RateLimitState {
            last_emit: None,
            suppressed: 0,
        };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        for i in 1..=3u64 {
            let at = now + Duration::from_millis(100 * i);
            assert!(matches!(
                limited_decision(&mut state, at, RATE_LIMIT_WINDOW),
                EmitDecision::Suppress
            ));
        }
        // 窗口过期后的下一条：放行，并携带上一窗口累计的抑制条数。
        let after_window = now + RATE_LIMIT_WINDOW + Duration::from_secs(1);
        match limited_decision(&mut state, after_window, RATE_LIMIT_WINDOW) {
            EmitDecision::Emit {
                previously_suppressed,
            } => {
                assert_eq!(previously_suppressed, 3);
            }
            EmitDecision::Suppress => panic!("call after window must emit"),
        }
        assert_eq!(state.suppressed, 0);
    }

    #[test]
    fn limited_decision_re_arms_window_after_flush() {
        let mut state = RateLimitState {
            last_emit: None,
            suppressed: 0,
        };
        let now = Instant::now();
        assert!(matches!(
            limited_decision(&mut state, now, RATE_LIMIT_WINDOW),
            EmitDecision::Emit { .. }
        ));
        // 放行后重新进入窗口期：紧随其后的调用再次被抑制。
        assert!(matches!(
            limited_decision(
                &mut state,
                now + Duration::from_millis(1),
                RATE_LIMIT_WINDOW
            ),
            EmitDecision::Suppress
        ));
    }

    #[test]
    fn short_source_strips_paths() {
        assert_eq!(short_source("src\\commands\\playback.rs"), "playback.rs");
        assert_eq!(
            short_source("src/renderer/vslib_processor.rs"),
            "vslib_processor.rs"
        );
        assert_eq!(short_source("plain.rs"), "plain.rs");
    }
}
