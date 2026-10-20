//! 片段媒体身份的**唯一**判据：这一轮该把哪一份采样挂到这条 take 上。
//!
//! 【为什么要有独立模块】"这条 clip 为什么没有音频"此前由六处各自的 `or_else` 链回答，
//! 于是同一个片段在不同路径上会得到不同的结论 —— 而用户只看到一句"等待宿主提供音频"。
//! 这里把判据收敛成一个纯函数，让"哪一环赢了"可断言、可单测、可进诊断。
//!
//! 【不变量 I2：身份单调】任何一轮展示都不得**降低**某个片段的媒体身份：
//! 只有"本来就不知道"才可以是不知道。实现上就是 `choose_link` 的返回序 ——
//! 只要有任一环给出证据，就绝不返回 `Link::None`。

/// 媒体身份的证据来源，按**可信度**从高到低。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Provenance {
    /// 宿主**这一次**的观察与 item 对得上（`authorized_takes` / `authorized_sources`）。
    Authorized,
    /// 分割谱系：父段已被授权，右半段与它是同一份素材。
    Inherited,
    /// 本实例**手里有 PCM**（`edit_sources`），且活着的 region 指向它。
    Pcm,
    /// 上一轮已经挂在这条 clip 上的身份。只在前三环都缺席时使用。
    LastKnown,
}

/// 这一轮该挂哪一环。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Link {
    Authorized,
    Inherited,
    Pcm,
    LastKnown,
    /// 不挂：按无源占位显示。
    None,
}

impl Link {
    /// 语言无关的名字（进诊断与日志）。
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Link::Authorized => "authorized",
            Link::Inherited => "inherited",
            Link::Pcm => "pcm",
            Link::LastKnown => "last_known",
            Link::None => "none",
        }
    }
}

/// 决定本轮的媒体身份来源。
///
/// 【为什么 `host_switched_take` 是**最高优先级**】用户把 active take 换成另一个文件之后，
/// ARA 还没重新认领；此时任何一环给出的都是**上一个** take 的采样。挂上去等于让用户听到
/// 一段不是他选中的内容 —— 宁可显示占位，也不能挂错。
///
/// 【为什么 `Pcm` 排在 `Authorized` 之后】PCM 这一环回答的是"我现在能不能渲染它"，
/// 而不是"这是不是用户选中的那一份"。两者一致时前者更可靠（它不依赖任何一次宿主观察），
/// 不一致时以宿主的观察为准 —— 后者才是"用户想要哪一段"的权威。
///
/// 【为什么 `Pcm` 排在 `LastKnown` 之前】`LastKnown` 只是"上一轮的记忆"，而 `Pcm`
/// 每一轮都能**重新求值**：哪怕 clip 被整份重建（`ara-region-*` 升级成 `ara-item-*`）也
/// 能把它找回来。记忆会断，重新求值不会。
pub(crate) fn choose_link(
    host_switched_take: bool,
    has_authorized: bool,
    has_inherited: bool,
    has_pcm: bool,
    has_last_known: bool,
) -> Link {
    if host_switched_take {
        return Link::None;
    }
    if has_authorized {
        return Link::Authorized;
    }
    if has_inherited {
        return Link::Inherited;
    }
    if has_pcm {
        return Link::Pcm;
    }
    if has_last_known {
        return Link::LastKnown;
    }
    Link::None
}

/// 本实例手里那份 PCM 的身份（`Pcm` 环的输入）。
///
/// 【为什么带几何而不只带 id】单独一个 `source_path` 不足以让下游画出波形或算出时长；
/// 而"这份 PCM 有多长、什么率"恰好是我们**已经知道**的事实，重算一次比从宿主元数据
/// 再猜一次更可靠。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PcmFacts {
    pub source_id: String,
    pub sample_rate: u32,
    pub channels: u16,
    pub frames: usize,
}

impl PcmFacts {
    /// 把这份 PCM 的媒体身份写进一条 take。
    ///
    /// 【为什么只写这几项】源窗口 / 倍率 / 名字 / 声道模式都是**宿主几何**，由清单或
    /// ARA 映射提供；这里只补"这份采样是哪一个文件"这一族（与 `apply_take_media` 同
    /// 一条边界）。
    pub(crate) fn apply(&self, take: &mut hifishifter_kernel::state::ClipTake) {
        take.source_path = Some(self.source_id.clone());
        take.source_path_relative = None;
        take.source_sample_rate = Some(self.sample_rate);
        take.duration_frames = Some(self.frames as u64);
        take.duration_sec = Some(if self.sample_rate == 0 {
            0.0
        } else {
            self.frames as f64 / f64::from(self.sample_rate)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **I2 的可执行形式：只要有任一环有证据，就绝不返回 `None`。**
    #[test]
    fn a_known_identity_never_regresses_to_unknown() {
        for (authorized, inherited, pcm, last_known) in [
            (false, false, false, true),
            (false, false, true, false),
            (false, true, false, false),
            (true, false, false, false),
        ] {
            assert_ne!(
                choose_link(false, authorized, inherited, pcm, last_known),
                Link::None,
                "有证据却报未知：身份回退"
            );
        }
        assert_eq!(choose_link(false, false, false, false, false), Link::None);
    }

    /// 宿主换了 take ⇒ 一个都不挂（宁可占位，也不挂错的采样）。
    #[test]
    fn a_switched_take_is_never_papered_over() {
        assert_eq!(choose_link(true, true, true, true, true), Link::None);
    }

    /// 优先级：宿主观察 > 分割谱系 > 手里的 PCM > 上一轮的记忆。
    #[test]
    fn the_link_order_is_authorized_inherited_pcm_last_known() {
        assert_eq!(choose_link(false, true, true, true, true), Link::Authorized);
        assert_eq!(choose_link(false, false, true, true, true), Link::Inherited);
        assert_eq!(choose_link(false, false, false, true, true), Link::Pcm);
        assert_eq!(choose_link(false, false, false, false, true), Link::LastKnown);
    }

    /// `PcmFacts::apply` 只写媒体身份，不碰宿主几何。
    #[test]
    fn pcm_facts_write_media_identity_only() {
        let facts = PcmFacts {
            source_id: "s1".into(),
            sample_rate: 48_000,
            channels: 2,
            frames: 96_000,
        };
        let mut take: hifishifter_kernel::state::ClipTake = serde_json::from_value(
            serde_json::json!({"id": "t", "source_start_sec": 1.5, "playback_rate": 2.0}),
        )
        .unwrap();
        facts.apply(&mut take);
        assert_eq!(take.source_path.as_deref(), Some("s1"));
        assert_eq!(take.source_sample_rate, Some(48_000));
        assert_eq!(take.duration_frames, Some(96_000_u64));
        assert_eq!(take.duration_sec, Some(2.0));
        assert_eq!(take.source_start_sec, 1.5, "宿主几何不得被覆盖");
        assert_eq!(take.playback_rate, 2.0);
    }
}