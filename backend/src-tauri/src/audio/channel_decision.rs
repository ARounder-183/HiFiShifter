//! Take 级「声道判定档案」：折叠决策的来源、结论与有效性。
//!
//! 背景：v4→v5 的假立体声折叠曾以**工程文件版本号**作为「该 Take 有没有权威
//! 声道来源」的代理判据（`project_file_version < 5`）。这个代理有两处硬伤：
//!
//! 1. 它是一次性的 —— 打开瞬间没判成的 Take 停在 `channel_mode = 0`，工程一
//!    保存成 v5，从此再没有第二次机会，而 0 会被当作「用户显式决定」永久尊重；
//! 2. 它无法区分「判过了，结论是真立体声」与「压根没读到这个文件」。
//!
//! 本模块用一份显式持久化的档案取代该代理：它同时回答「谁定的 / 定的什么 /
//! 结论还有效吗」。有了它，折叠就从「一次性迁移」变成**幂等、可恢复、可重跑**
//! 的扫描 —— 读不到的 Take 保持 [`VERDICT_PENDING`]，下次打开自动重试；用户
//! 显式设置的 Take 带 [`ORIGIN_USER`]，自动扫描永不触碰。
//!
//! 判定语义（怎么比 L/R）仍然唯一收敛在 [`crate::stereo_detect`]；本模块只
//! 承载「这个结论还作不作数」的记账。

use serde::{Deserialize, Serialize};

// ─── 判定来源 ────────────────────────────────────────────────────────────────

/// 自动扫描（导入策略 / 旧工程升级 / 手动重扫）得出的结论。
pub const ORIGIN_AUTO: u8 = 0;
/// 用户显式设置的声道模式；自动扫描**永不改写**。
///
/// REAPER 导入的 `CHANMODE` 也归此类：它是素材自带的权威字段，不是我们的推断。
pub const ORIGIN_USER: u8 = 1;

// ─── 自动结论 ────────────────────────────────────────────────────────────────

/// 源是单声道（`channels < 2`），无需折叠。
pub const VERDICT_MONO: u8 = 0;
/// 真立体声，保持原样。
pub const VERDICT_TRUE_STEREO: u8 = 1;
/// 假立体声（L/R 在容差内一致），已折叠或应折叠为单声道。
pub const VERDICT_FAKE_STEREO: u8 = 2;
/// 本次读不到（源缺失 / 不可解码 / 探测不出声道数）—— **下次打开重试**。
///
/// 与「结论是单声道」严格区分：把「读不了」记成「单声道」会让漏判永久静默
/// （扫描报告里显示成「无事可做」而不是「我没读到」）。
pub const VERDICT_PENDING: u8 = 3;
/// 由用户决定，没有自动结论（`origin` 必为 [`ORIGIN_USER`]）。
pub const VERDICT_USER: u8 = 4;
/// 策略为"全部转换为单声道"：不判定内容，直接折叠。
///
/// 与 [`VERDICT_FAKE_STEREO`] 分开记录：前者是"我们看了内容，L/R 一致"，
/// 后者是"用户要求无条件折叠"。混为一谈会让扫描报告谎报检测结果。
pub const VERDICT_FORCED_MONO: u8 = 5;

/// 一次声道判定的完整上下文。
///
/// `fingerprint` / `policy_sig` / `region_q` 三者共同构成「结论是否仍然适用」
/// 的判据：源文件被换掉、用户改了抽样策略、或 Take 的消费区间被 trim 改变，
/// 旧结论都必须失效重判。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelDecisionRecord {
    /// [`ORIGIN_AUTO`] / [`ORIGIN_USER`]。
    #[serde(default)]
    pub origin: u8,
    /// 自动结论之一（见本模块 `VERDICT_*` 常量）。
    #[serde(default)]
    pub verdict: u8,
    /// 判定时源文件的内容指纹。
    ///
    /// `None` 表示当时读不出指纹 —— 无法证明文件没被替换，故结论一律视为
    /// 不权威（见 [`Self::is_authoritative_for`]）。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fingerprint: Option<u64>,
    /// 判定时生效的策略签名（[`crate::stereo_detect::DetectOptions::signature`]）。
    #[serde(default)]
    pub policy_sig: u64,
    /// 判定的源域区间（毫秒量化，与判定缓存同口径）；`None` = 整个文件。
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_q: Option<(i64, i64)>,
}

impl ChannelDecisionRecord {
    /// 一条自动扫描结论。
    pub fn auto(
        verdict: u8,
        fingerprint: Option<u64>,
        policy_sig: u64,
        region_q: Option<(i64, i64)>,
    ) -> Self {
        Self {
            origin: ORIGIN_AUTO,
            verdict,
            fingerprint,
            policy_sig,
            region_q,
        }
    }

    /// 一条「本次读不到，下次重试」的占位结论。
    pub fn pending(fingerprint: Option<u64>, policy_sig: u64, region_q: Option<(i64, i64)>) -> Self {
        Self::auto(VERDICT_PENDING, fingerprint, policy_sig, region_q)
    }

    /// 用户显式决定的封印档案。
    pub fn user() -> Self {
        Self {
            origin: ORIGIN_USER,
            verdict: VERDICT_USER,
            fingerprint: None,
            policy_sig: 0,
            region_q: None,
        }
    }

    /// 是否由用户显式决定（自动扫描不得改写）。
    pub fn is_user(self) -> bool {
        self.origin == ORIGIN_USER
    }

    /// 是否「本次读不到，下次重试」。
    pub fn is_pending(self) -> bool {
        self.origin == ORIGIN_AUTO && self.verdict == VERDICT_PENDING
    }

    /// 该结论对给定的上下文是否**仍然有效**（可直接采用，零解码）。
    ///
    /// 用户档案永远不「权威」：它不该被自动流程消费，只该被尊重。
    pub fn is_authoritative_for(
        self,
        fingerprint: Option<u64>,
        policy_sig: u64,
        region_q: Option<(i64, i64)>,
    ) -> bool {
        if self.is_user() || self.is_pending() {
            return false;
        }
        // 指纹缺位 ⇒ 无法证明源文件未被替换 ⇒ 结论不可复用。
        if self.fingerprint.is_none() {
            return false;
        }
        self.fingerprint == fingerprint
            && self.policy_sig == policy_sig
            && self.region_q == region_q
    }
}

/// 该 Take 是否应被**自动扫描**纳入候选。
///
/// 覆盖三种情况：从未判过、判过但上下文已变（指纹/策略/区间）、以及上次
/// 「读不到」需要重试。用户封印的 Take 一律排除。
///
/// 这是「漏判不再永久化」的落点：v4 旧工程里任何一次没读到的 Take，都会在
/// 后续每次打开时重新进入候选，直到真正得出结论为止。
pub fn needs_auto_scan(
    record: Option<ChannelDecisionRecord>,
    fingerprint: Option<u64>,
    policy_sig: u64,
    region_q: Option<(i64, i64)>,
) -> bool {
    match record {
        None => true,
        Some(record) => {
            if record.is_user() {
                return false;
            }
            !record.is_authoritative_for(fingerprint, policy_sig, region_q)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIG: u64 = 42;
    const FP: u64 = 0xDEAD_BEEF;
    const REGION: Option<(i64, i64)> = Some((0, 10_000));

    #[test]
    fn missing_record_needs_scan() {
        assert!(needs_auto_scan(None, Some(FP), SIG, REGION));
    }

    #[test]
    fn settled_auto_record_needs_no_scan() {
        let record = ChannelDecisionRecord::auto(VERDICT_TRUE_STEREO, Some(FP), SIG, REGION);
        assert!(!needs_auto_scan(Some(record), Some(FP), SIG, REGION));
    }

    #[test]
    fn every_context_change_invalidates_the_record() {
        let record = ChannelDecisionRecord::auto(VERDICT_FAKE_STEREO, Some(FP), SIG, REGION);
        // 源文件被替换
        assert!(needs_auto_scan(Some(record), Some(FP + 1), SIG, REGION));
        // 抽样策略变化
        assert!(needs_auto_scan(Some(record), Some(FP), SIG + 1, REGION));
        // 消费区间变化（trim / 拆分）
        assert!(needs_auto_scan(Some(record), Some(FP), SIG, Some((0, 9_000))));
    }

    #[test]
    fn pending_record_is_always_retried() {
        let record = ChannelDecisionRecord::pending(None, SIG, REGION);
        assert!(record.is_pending());
        assert!(needs_auto_scan(Some(record), None, SIG, REGION));
        // 上下文一字未变也照样重试 —— 这正是「读不到」与「已定论」的区别。
        assert!(needs_auto_scan(Some(record), Some(FP), SIG, REGION));
    }

    #[test]
    fn user_record_is_never_scanned() {
        let record = ChannelDecisionRecord::user();
        assert!(record.is_user());
        assert!(!record.is_authoritative_for(None, SIG, REGION));
        assert!(!needs_auto_scan(Some(record), Some(FP), SIG, REGION));
        // 连上下文变化也不该把它拉回候选：用户的选择是终局的。
        assert!(!needs_auto_scan(Some(record), Some(FP + 1), SIG + 1, None));
    }

    #[test]
    fn record_without_fingerprint_is_never_authoritative() {
        let record = ChannelDecisionRecord::auto(VERDICT_MONO, None, SIG, REGION);
        assert!(!record.is_authoritative_for(None, SIG, REGION));
        assert!(needs_auto_scan(Some(record), None, SIG, REGION));
    }

    #[test]
    fn record_round_trips_through_serde() {
        let record = ChannelDecisionRecord::auto(VERDICT_FAKE_STEREO, Some(FP), SIG, REGION);
        let json = serde_json::to_string(&record).expect("serialize");
        let back: ChannelDecisionRecord = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(record, back);

        // 缺字段的旧档案必须能读出来（前向兼容：手改 / 未来加字段）。
        let sparse: ChannelDecisionRecord =
            serde_json::from_str(r#"{"origin":1,"verdict":4}"#).expect("sparse");
        assert_eq!(sparse.origin, ORIGIN_USER);
        assert_eq!(sparse.verdict, VERDICT_USER);
        assert_eq!(sparse.fingerprint, None);
    }
}
