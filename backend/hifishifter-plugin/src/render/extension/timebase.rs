//! 宿主时基（HostTimebase）：把 ProcessContext 的原始位置映射到一条
//! 插件本地时间轴。
//!
//! VEGAS 的 Event 级 FX 会在一次连续播放中切换坐标域：起播窗口先报事件在工程里的
//! 绝对位置，随后改报事件内相对位置。音频是连续的，两个读数却不是同一把尺子。
//! 把二者直接相减会得到一次巨大的“倒退”，下游于是切段、留洞、或把首段留在工程
//! 绝对位置上。这里把“域”提升成显式状态：只有分类器说这是域切换，才平移整条 take；
//! 否则位置变化一律按真正的 seek 处理。
//!
//! 【为什么是纯函数】分类器不碰宿主、不取锁、不分配，因此可以在音频线程调用，也可以
//! 脱离宿主单测。音频线程需要的是确定性，不是“再猜一次”。

/// 当前生效的宿主坐标域。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum TimebaseKind {
    /// 还没有足够证据；位置按宿主原始读数使用。
    #[default]
    Unknown,
    /// Track FX / 普通插入：工程绝对位置。
    ProjectAbsolute,
    /// Event FX：事件内相对位置。offset 满足 local = raw + offset。
    EventRelative,
    /// 离线 pass 自己的本地游标；宿主原始位置完全忽略。
    OfflineLocal,
}

impl TimebaseKind {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::ProjectAbsolute => 1,
            Self::EventRelative => 2,
            Self::OfflineLocal => 3,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Unknown),
            1 => Some(Self::ProjectAbsolute),
            2 => Some(Self::EventRelative),
            3 => Some(Self::OfflineLocal),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ProjectAbsolute => "project",
            Self::EventRelative => "event-relative",
            Self::OfflineLocal => "offline-local",
        }
    }
}

/// 一次坐标域切换的结论。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DomainSwitch {
    /// 绝对位置 -> 事件内相对位置。new_local 是切换后本块应落到的本地位置。
    ToEventRelative { new_local: i64 },
    /// 事件内相对位置 -> 绝对位置。
    ToProjectAbsolute { new_local: i64 },
}

impl DomainSwitch {
    pub(crate) fn new_local(self) -> i64 {
        match self {
            Self::ToEventRelative { new_local } | Self::ToProjectAbsolute { new_local } => {
                new_local
            }
        }
    }

    pub(crate) fn kind(self) -> TimebaseKind {
        match self {
            Self::ToEventRelative { .. } => TimebaseKind::EventRelative,
            Self::ToProjectAbsolute { .. } => TimebaseKind::ProjectAbsolute,
        }
    }

    /// 为了让“已经写下的音频”接上 new_local，整条 take 需要移动的帧数。
    pub(crate) fn shift(self, previous_local_end: i64) -> i64 {
        self.new_local().saturating_sub(previous_local_end)
    }
}

/// 域切换的观测窗口：只在一次传输会话的最初 2 秒内猜测。
///
/// VEGAS 现场实测切换发生在起播后约 0.6 s；2 s 给了三倍余量，同时把
/// “用户播放两秒后按 Home 回到 0”这种真正的 seek 排除在候选之外。
pub(crate) const DOMAIN_SWITCH_WINDOW_SEC: f64 = 2.0;

/// 域切换的最小幅度：小于半秒的差值属于块边界/记账误差，不是换了一把尺子。
pub(crate) const DOMAIN_SWITCH_MIN_SEC: f64 = 0.5;

/// 事件内相对位置合理的量级上限（秒）。
///
/// Event FX 的 projectTimeSamples 在相对域里是“事件内偏移”；一个事件通常远短于
/// 5 秒的 1000 倍。这个界只用于分类，不用于截断任何真实位置。
pub(crate) const EVENT_RELATIVE_PLAUSIBLE_SEC: f64 = 5.0;

/// 分类一次域切换。返回 None 表示“这不是域切换，按真 seek 处理”。
///
/// 判据刻意同时要求：
/// 1. 同一进程流（prefetch/realtime 的节奏不同，不能互相作证）；
/// 2. 不是 kPlaying 上升沿（上升沿是新的传输会话，不是域切换）；
/// 3. 仍在会话起点窗口内；
/// 4. 位置差至少半秒；
/// 5. 新读数落在“事件内偏移”的量级，或旧本地位置落在该量级（反向切换）。
#[allow(clippy::too_many_arguments)]
pub(crate) fn classify_domain_switch(
    kind: TimebaseKind,
    session_frames: u64,
    rate: f64,
    raw_start: i64,
    previous_local_end: i64,
    same_stream: bool,
    playing_edge: bool,
) -> Option<DomainSwitch> {
    if !same_stream || playing_edge || kind == TimebaseKind::OfflineLocal {
        return None;
    }
    let rate = if rate.is_finite() && rate > 0.0 {
        rate
    } else {
        48_000.0
    };
    let window = (rate * DOMAIN_SWITCH_WINDOW_SEC).round().max(1.0) as u64;
    if session_frames == 0 || session_frames > window {
        return None;
    }
    let delta = raw_start.saturating_sub(previous_local_end);
    let magnitude = delta.unsigned_abs();
    if magnitude < (rate * DOMAIN_SWITCH_MIN_SEC).round().max(1.0) as u64 {
        return None;
    }
    let event_scale = (rate * EVENT_RELATIVE_PLAUSIBLE_SEC).round() as i64;
    // 只有“旧的本地位置已经在事件级尺度上”时才可能是绝对 -> 相对。
    // 短循环/短回退（例如 0.2 s 回到 0.2 s）必须仍按真 seek 处理。
    if delta < 0
        && (0..event_scale).contains(&raw_start)
        && previous_local_end >= (rate * 2.0).round() as i64
    {
        return Some(DomainSwitch::ToEventRelative {
            new_local: raw_start,
        });
    }
    // 相对 -> 绝对只可能发生在“已经确认过 EventRelative”的实例上。否则
    // 一次普通的向前 seek（0 -> 10 s）会被误判成域切换。
    if kind == TimebaseKind::EventRelative
        && delta > 0
        && (0..event_scale).contains(&previous_local_end)
    {
        return Some(DomainSwitch::ToProjectAbsolute {
            new_local: previous_local_end,
        });
    }
    None
}

/// 把一个原始位置映射到本地域。
#[allow(dead_code)]
pub(crate) fn local_from_raw(kind: TimebaseKind, offset: i64, raw_start: i64) -> i64 {
    match kind {
        TimebaseKind::OfflineLocal => 0,
        TimebaseKind::Unknown | TimebaseKind::ProjectAbsolute | TimebaseKind::EventRelative => {
            raw_start.saturating_add(offset)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: f64 = 48_000.0;

    #[test]
    fn event_relative_switch_is_recognised_inside_the_start_window() {
        // 事件绝对位置 31.091 s，已经写了 0.6 s；随后 raw 回到 0.6 s。
        let previous = 31_091 * 48 + 28_800;
        let switch = classify_domain_switch(
            TimebaseKind::Unknown,
            28_800,
            RATE,
            28_800,
            previous,
            true,
            false,
        );
        assert_eq!(
            switch,
            Some(DomainSwitch::ToEventRelative { new_local: 28_800 })
        );
        assert_eq!(switch.unwrap().shift(previous), -31_091 * 48);
    }

    #[test]
    fn a_late_seek_is_not_a_domain_switch() {
        // 播放 3 秒后按 Home：超出窗口 ⇒ 真 seek。
        assert_eq!(
            classify_domain_switch(
                TimebaseKind::ProjectAbsolute,
                3 * 48_000,
                RATE,
                0,
                3 * 48_000,
                true,
                false,
            ),
            None
        );
    }

    #[test]
    fn a_playing_edge_is_a_new_session_not_a_domain_switch() {
        assert_eq!(
            classify_domain_switch(
                TimebaseKind::Unknown,
                16_000,
                RATE,
                0,
                31_091 * 48,
                true,
                true,
            ),
            None
        );
    }

    #[test]
    fn a_small_delta_is_block_quantisation() {
        assert_eq!(
            classify_domain_switch(
                TimebaseKind::ProjectAbsolute,
                16_000,
                RATE,
                16_000 + 200,
                16_000,
                true,
                false,
            ),
            None
        );
    }

    #[test]
    fn reverse_switch_keeps_audio_continuity() {
        let switch = classify_domain_switch(
            TimebaseKind::EventRelative,
            24_000,
            RATE,
            31_091 * 48 + 24_000,
            24_000,
            true,
            false,
        );
        assert_eq!(
            switch,
            Some(DomainSwitch::ToProjectAbsolute { new_local: 24_000 })
        );
        assert_eq!(switch.unwrap().shift(24_000), 0);
    }
}
