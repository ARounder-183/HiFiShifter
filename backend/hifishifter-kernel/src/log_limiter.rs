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

/// 动态键限流器同时跟踪的键上限。
///
/// 【为什么必须有上限】键来自运行期数据（方法名 + 错误摘要），一个持续报错的
/// 宿主可以在几分钟里造出成千上万个互不相同的键。没有上限就等于把一个无界
/// 哈希表挂在进程上 —— 那正是本模块要防的"日志刷屏"换了个形式。
/// 32 个足以覆盖"同时有好几类错误在刷"的真实情形；超出后按最久未见淘汰。
const DYNAMIC_KEY_CAPACITY: usize = 32;

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

/// 动态字符串键的限流表（与站点键分开，因为键的生命周期与容量策略不同）。
static DYNAMIC_RATE_LIMITER: OnceLock<Mutex<HashMap<String, RateLimitState>>> = OnceLock::new();

fn rate_limiter() -> &'static Mutex<RateLimitMap> {
    RATE_LIMITER.get_or_init(|| Mutex::new(HashMap::new()))
}

fn dynamic_rate_limiter() -> &'static Mutex<HashMap<String, RateLimitState>> {
    DYNAMIC_RATE_LIMITER.get_or_init(|| Mutex::new(HashMap::new()))
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

/// 按**动态字符串键**限流的输出入口。
///
/// 【为什么需要它，而不是继续用 `(file, line)`】`emit_limited` 的键是编译期固定的
/// 调用点。当一个调用点会被**运行期数据**驱动时（例如"前端上报的 invoke 失败"，
/// 每个方法名都是一类新错误），所有内容都落进同一个桶 —— 桶太粗会把真实的新错误
/// 一起吞掉，桶太细（每行一个 `line!`）又无法覆盖动态方法名。
/// 这里用调用方给的键（如 `"invoke_failed:plugin_get_apply_state"`）分桶，
/// 每类错误各自限流，互不干扰。
///
/// 【为什么键集合有上限】见 [`DYNAMIC_KEY_CAPACITY`]：超出后淘汰最久未见者，
/// 保证"一直有新错误"不会把这张表变成内存泄漏。
pub fn emit_limited_key(level: log::Level, key: &str, args: std::fmt::Arguments<'_>) {
    let filter = level_filter_of(level);
    if filter > log::STATIC_MAX_LEVEL || filter > log::max_level() {
        return;
    }

    let now = Instant::now();
    let decision = {
        let mut map = dynamic_rate_limiter()
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if !map.contains_key(key) && map.len() >= DYNAMIC_KEY_CAPACITY {
            evict_stalest(&mut map);
        }
        let state = map.entry(key.to_owned()).or_insert_with(|| RateLimitState {
            last_emit: None,
            suppressed: 0,
        });
        limited_decision(state, now, RATE_LIMIT_WINDOW)
    };

    if let EmitDecision::Emit {
        previously_suppressed,
    } = decision
    {
        if previously_suppressed > 0 {
            log::log!(
                level,
                "[throttled] {key} — {previously_suppressed} message(s) suppressed",
            );
        }
        log::log!(level, "{args}");
    }
}

/// 淘汰最久未输出的一条动态键。仅在表满时调用。
fn evict_stalest(map: &mut HashMap<String, RateLimitState>) {
    let Some(key) = map
        .iter()
        .min_by_key(|(_, state)| state.last_emit)
        .map(|(key, _)| key.clone())
    else {
        return;
    };
    map.remove(&key);
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

    /// 【为什么钉这条】动态键限流是"每个错误原因各自限流"的实现基础：两个不同的键
    /// 必须各自计数，否则方法名不同的两类失败会互相把对方挤掉。
    #[test]
    fn dynamic_keys_are_limited_independently() {
        let mut map: HashMap<String, RateLimitState> = HashMap::new();
        let now = Instant::now();
        let decide = |map: &mut HashMap<String, RateLimitState>, key: &str, at: Instant| {
            let state = map.entry(key.to_owned()).or_insert_with(|| RateLimitState {
                last_emit: None,
                suppressed: 0,
            });
            limited_decision(state, at, RATE_LIMIT_WINDOW)
        };
        // 键 A 首条放行，同窗口内第二条抑制。
        assert!(matches!(
            decide(&mut map, "a", now),
            EmitDecision::Emit { .. }
        ));
        assert!(matches!(
            decide(&mut map, "a", now + Duration::from_millis(10)),
            EmitDecision::Suppress
        ));
        // 键 B 完全不受 A 的影响：它自己的第一条必须放行。
        assert!(matches!(
            decide(&mut map, "b", now + Duration::from_millis(10)),
            EmitDecision::Emit { .. }
        ));
    }

    /// 【为什么钉这条】表必须**有界**：一个持续报错的宿主会造出成千上万个不同键，
    /// 无界表就是把"日志刷屏"换成了"内存泄漏"。淘汰策略必须真的回收条目。
    #[test]
    fn dynamic_key_table_evicts_the_stalest_when_full() {
        let mut map: HashMap<String, RateLimitState> = HashMap::new();
        let base = Instant::now();
        // 填满到上限，键 i 的 last_emit 递增 ⇒ 键 0 最旧。
        for index in 0..DYNAMIC_KEY_CAPACITY {
            map.insert(
                format!("key-{index}"),
                RateLimitState {
                    last_emit: Some(base + Duration::from_millis(index as u64)),
                    suppressed: 0,
                },
            );
        }
        assert_eq!(map.len(), DYNAMIC_KEY_CAPACITY);
        evict_stalest(&mut map);
        assert_eq!(map.len(), DYNAMIC_KEY_CAPACITY - 1, "必须真的回收一条");
        assert!(!map.contains_key("key-0"), "最久未见的键应被淘汰");
        assert!(map.contains_key(&format!("key-{}", DYNAMIC_KEY_CAPACITY - 1)));
    }
}
