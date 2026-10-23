//! 持久化状态恢复的进度账：`RESTORE_*` 读数与 `RestoreProgress`。
/// 持久化状态从未尝试恢复。
pub(crate) const RESTORE_NEVER: u8 = 0;
/// 持久化状态已成功合并（`0` = 从未尝试）。
pub(crate) const RESTORE_APPLIED: u8 = 1;
/// 持久化状态已到达，但文档还没就绪 / 仍在重试（不是故障）。
pub(crate) const RESTORE_PENDING: u8 = 2;
/// 持久化状态合并失败，且没有可重试的路径（真故障）。
pub(crate) const RESTORE_FAILED: u8 = 3;
/// 宿主图**一次都没变**的时长上界：超过它就改报"没等到"并转入慢重试。
///
/// 【为什么必须有上界】"身份对不上 ⇒ 继续等"如果没有截止时刻，载荷永远挂着、界面
/// 永远说"正在等待"，而 `encode_archive` 在此期间会把这份**没合入**的状态当成权威
/// 写回磁盘 —— 用户看到的是"参数没了"，日志里一行错都没有。
///
/// 【为什么是 10 秒，且必须按"宿主没动"计】重开工程时宿主交付几何的典型跨度是
/// 几百毫秒到几秒。旧实现按**我们自己 poll 的次数**计时（`RESTORE_GIVE_UP_ATTEMPTS`），
/// 于是界面刷新得越快、预算烧得越快：10 秒的窗口能打印出 38.2 秒，那个数字既不是
/// "等了多久"也不是"宿主多久没动"。现在只有一个起点 —— 图签名最后一次变化的时刻。
pub(crate) const RESTORE_WAIT_WINDOW: std::time::Duration = std::time::Duration::from_secs(10);
/// 报过"没等到"之后的重试间隔。
///
/// 【为什么"放弃"还要重试】放弃的只是**界面读数**（从"等待"改成"没等到"），载荷本身
/// 必须留着：宿主随时可能补发几何，而那时合入一次就能把用户参数全部找回来。
pub(crate) const RESTORE_RETRY_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// 一份恢复载荷的等待账：**只由宿主图的变化推进**。
///
/// 【为什么是一个结构而不是三个原子】"等了多久 / 变过没有 / 报过没有"必须一起读、
/// 一起写，否则又会出现上一版那种"三个计数器凑出一个数字"的状态。
#[derive(Default)]
pub(super) struct RestoreProgress {
    /// 图签名最后一次变化的时刻（`None` = 还没有载荷）。
    pub(super) since: Option<std::time::Instant>,
    /// 上一次看到的图签名。
    pub(super) signature: u64,
    /// 这一次等待是否已经报过"没等到"（避免每轮重报）。
    pub(super) reported: bool,
}
