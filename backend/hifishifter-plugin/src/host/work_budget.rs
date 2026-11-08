//! Cooperative callback budget. A running host API call cannot be preempted;
//! callers must split large jobs and check this between calls.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

pub(crate) struct WorkBudget {
    start: Instant,
    duration: Duration,
    remaining: usize,
}

pub(crate) fn round_robin<T>(
    items: &[T],
    cursor: &AtomicUsize,
    budget: &mut WorkBudget,
    mut visit: impl FnMut(&T),
) {
    if items.is_empty() {
        return;
    }
    let start = cursor.load(Ordering::Relaxed) % items.len();
    let mut done = 0;
    while done < items.len() && budget.take() {
        visit(&items[(start + done) % items.len()]);
        done += 1;
    }
    cursor.store((start + done) % items.len(), Ordering::Relaxed);
}
impl WorkBudget {
    pub(crate) fn new(duration: Duration, jobs: usize) -> Self {
        Self {
            start: Instant::now(),
            duration,
            remaining: jobs,
        }
    }
    pub(crate) fn available(&self) -> bool {
        self.remaining > 0 && self.start.elapsed() < self.duration
    }
    pub(crate) fn take(&mut self) -> bool {
        if !self.available() {
            return false;
        }
        self.remaining -= 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn callback_budget_yields_at_count_and_deadline() {
        let mut count = WorkBudget::new(Duration::from_secs(1), 2);
        assert!(count.take());
        assert!(count.take());
        assert!(!count.take());
        assert!(!WorkBudget::new(Duration::ZERO, 10).take());
    }
    #[test]
    fn limited_ticks_eventually_visit_every_owner() {
        let cursor = AtomicUsize::new(0);
        let mut seen = Vec::new();
        for _ in 0..3 {
            round_robin(
                &[0, 1, 2, 3, 4],
                &cursor,
                &mut WorkBudget::new(Duration::from_secs(1), 2),
                |item| seen.push(*item),
            );
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4, 0]);
    }
}
