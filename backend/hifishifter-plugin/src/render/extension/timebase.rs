//! 宿主时基的**诊断编码**。
//!
//! 真正的域判定已经搬到 `host_clock.rs`（HostClock）。这里只保留一个稳定编码，
//! 供日志、块账本与前端读数使用 —— 编码值只增不改，日志里出现过的数字必须保持含义。
//!
//! 【为什么不再保留旧分类器】`classify_domain_switch` 把“域切换”写成“2 秒内、
//! 5 秒量级、kPlaying 上升沿除外”的启发式；而 VEGAS Event FX 的切换恰好发生在
//! kPlaying 上升沿，于是它被永久跳过。保留一个已被证明会漏判的分类器只会让下一位
//! 读者以为它仍然可用。

/// 当前生效的宿主时基（诊断编码）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub(crate) enum TimebaseKind {
    /// 还没有足够证据。
    #[default]
    Unknown,
    /// Track FX / 普通插入：工程绝对位置。
    ProjectAbsolute,
    /// Event FX：事件内相对位置。offset 满足 local = raw + offset。
    EventRelative,
    /// Event FX：宿主仍在报工程绝对位置（已知事件起点时使用）。
    EventAbsolute,
    /// 离线 pass 自己的本地游标；宿主原始位置完全忽略。
    OfflineLocal,
    /// 冻结预览 pass 的本地游标。
    PreviewLocal,
}

impl TimebaseKind {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::ProjectAbsolute => 1,
            Self::EventRelative => 2,
            Self::OfflineLocal => 3,
            Self::EventAbsolute => 4,
            Self::PreviewLocal => 5,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(Self::Unknown),
            1 => Some(Self::ProjectAbsolute),
            2 => Some(Self::EventRelative),
            3 => Some(Self::OfflineLocal),
            4 => Some(Self::EventAbsolute),
            5 => Some(Self::PreviewLocal),
            _ => None,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::ProjectAbsolute => "project",
            Self::EventRelative => "event-relative",
            Self::EventAbsolute => "event-absolute",
            Self::OfflineLocal => "offline-local",
            Self::PreviewLocal => "preview-local",
        }
    }
}
