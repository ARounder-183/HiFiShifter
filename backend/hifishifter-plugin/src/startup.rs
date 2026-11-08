//! End-to-end instance timing and slow preparation spans. No PCM or user paths.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub(crate) enum Stage {
    Component,
    AraBind,
    ViewRequested,
    Attached,
    Navigation,
    Bridge,
    React,
    Metadata,
}

pub(crate) struct StartupTrace {
    id: u64,
    started: Instant,
    stages: AtomicU64,
}
impl Default for StartupTrace {
    fn default() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            started: Instant::now(),
            stages: AtomicU64::new(0),
        }
    }
}
impl StartupTrace {
    pub(crate) fn mark(&self, stage: Stage, document: u64) {
        let bit = 1u64 << stage as u8;
        if self.stages.fetch_or(bit, Ordering::Relaxed) & bit != 0 {
            return;
        }
        log::warn!(
            "[startup] pid={} startup={} doc={} stage={stage:?} elapsed_ms={} thread={:?}",
            std::process::id(),
            self.id,
            document,
            self.started.elapsed().as_millis(),
            std::thread::current().id()
        );
    }
}

pub(crate) struct Span {
    name: &'static str,
    document: u64,
    units: u64,
    start: Instant,
}
impl Span {
    pub(crate) fn new(name: &'static str, document: u64, units: u64) -> Self {
        Self {
            name,
            document,
            units,
            start: Instant::now(),
        }
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        if self.start.elapsed() < Duration::from_millis(16) {
            return;
        }
        hifishifter_kernel::log_limiter::emit_limited_key(
            log::Level::Warn,
            &format!("slow.{}", self.name),
            format_args!(
                "[startup-work] pid={} doc={} phase={} units={} elapsed_ms={} thread={:?}",
                std::process::id(),
                self.document,
                self.name,
                self.units,
                self.start.elapsed().as_millis(),
                std::thread::current().id()
            ),
        );
    }
}
