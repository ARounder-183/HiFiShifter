//! REAPER take 几何 → 内核 Clip 源窗口的**唯一**解算。
//!
//! ## 为什么需要它（一次真实事故）
//!
//! 两个前端都要回答同一个问题："这个 take 实际消费源媒体的哪一段？"
//!
//! - 独立 App 从 `.rpp` 文本读 `SOFFS` / `LENGTH` / `PLAYRATE` / `SOURCE SECTION`；
//! - ARA 插件从宿主实时 API 读 `D_STARTOFFS` / `D_LENGTH` / `D_PLAYRATE`。
//!
//! 此前两侧各写了一份：App 的那份经过实测打磨（`probe/ara/REVERSE-FINDINGS.md`），
//! 插件的那份是"按现象凑出来"的近似（只处理无 SECTION 的倒放，且方向判定与 App
//! 不一致）。于是**同一个工程在两个前端听感不同** —— 用户报障"倒放的 Take，其
//! Offset 的逻辑不对，方向与 REAPER 完全相反"。
//!
//! 现在只有这一份：两侧各自把宿主读数装进 [`TakeGeometry`]，调用
//! [`TakeGeometry::forward_source_window`]，结果必然一致。
//!
//! ## 坐标约定（这是全部混乱的源头，先钉死）
//!
//! 内核的倒放消费模型（`state::model::clip_playback_window_sec`）是：
//!
//! ```text
//! 正放 win = [source_start, source_start + length×rate)
//! 倒放 win = [source_end   − length×rate, source_end)
//! ```
//!
//! 也就是说内核**只**认一个**正向**窗口，倒放时用它的**终点**反向取窗。而宿主报的
//! `SOFFS` 在倒放 take 上是**镜像**量（`D_STARTOFFS = 媒体长 − 正向起点 − 跨度`，
//! 见 `REVERSE-FINDINGS.md` 的实测：媒体 2.0 s、正向窗口 `[0.25, 1.25]` ⇒
//! `D_STARTOFFS = 0.75`）。因此本模块的职责就是**把镜像翻回正向**，产出内核要的窗口。
//!
//! ## 镜像轴
//!
//! 实测（`probe/ara/captures/reverse-script.log`）：REAPER 为倒放 take 建的 SECTION
//! 覆盖**整个媒体**（`PCM_Source_GetSectionInfo` 返回 `offset=0, length=媒体长`），
//! 所以"段末端"与"媒体末端"在该情形下重合。本模块按 App 的既有口径取
//! `min(段末端, 媒体长)` 为轴 —— 对 REAPER 生成的倒放段二者相等；对用户显式建过
//! SECTION 边界的工程则尊重段边界。

/// 一个 SOURCE SECTION 的边界。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSection {
    /// 段起点（秒）。
    pub start_sec: f64,
    /// 段长度（秒）；`None` 或非正表示"到媒体末尾"。
    pub length_sec: Option<f64>,
}

/// 一个 take 的几何输入（宿主无关）。
///
/// 【为什么是纯数据而不是引用宿主结构】两侧的宿主类型完全不同（RPP 的
/// `ReaperTake` 与插件的实时 getter），但它们都能拆成这几个 `f64`。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakeGeometry {
    /// `SOFFS` / `D_STARTOFFS`。**倒放时是镜像量**，见模块文档。
    pub start_offset_sec: f64,
    /// item 长度（`LENGTH` / `D_LENGTH`），秒。
    pub item_length_sec: f64,
    /// 播放速率（`PLAYRATE[0]` / `D_PLAYRATE`），**保留符号**。
    ///
    /// 【为什么保留符号】App 用 `play_rate < 0` 作为倒放的**两个来源之一**。
    /// 插件曾把它 `.abs()` 掉、只认方向位，于是"只用负速率表达倒放"的 take 被
    /// 静默按正放渲染。
    pub play_rate_signed: f64,
    /// 媒体总时长（秒）；`None` = 未知（媒体缺失 / 尚未解码）。
    pub media_duration_sec: Option<f64>,
    /// SECTION 边界（RPP 的 `STARTPOS`/`LENGTH`，或实时 `PCM_Source_GetSectionInfo`）。
    pub section: Option<SourceSection>,
    /// 方向：`play_rate < 0` **或** `SECTION MODE > 0`（App），或宿主方向位（插件）。
    pub reversed: bool,
    /// 循环源（REAPER "Loop source" / `B_LOOPSRC`）。
    pub loop_enabled: bool,
}

impl TakeGeometry {
    /// App 口径的方向判定：负速率 **或** 反向 SECTION。
    ///
    /// 【为什么两个来源都要】REAPER 有两种表达倒放的方式，用哪一种取决于用户
    /// 怎么操作（"Toggle take reverse" 建 SECTION，手改速率则没有 SECTION）。
    /// 只看一个会漏掉另一半。
    pub fn reversed_from_host(play_rate_signed: f64, section_mode: i32) -> bool {
        play_rate_signed < 0.0 || section_mode > 0
    }

    /// 速率幅值，下限 0.01（与 App 的 `raw_play_rate.abs().max(0.01)` 一致）。
    pub fn play_rate_magnitude(&self) -> f64 {
        if self.play_rate_signed.is_finite() {
            self.play_rate_signed.abs().max(0.01)
        } else {
            1.0
        }
    }

    /// 该 take 在源域**消耗**的秒数：`item 长度 × |速率|`。
    pub fn consumed_sec(&self) -> f64 {
        (self.item_length_sec.max(0.0)) * self.play_rate_magnitude()
    }

    /// 可寻址的源区间 `(min_bound, max_bound)` 与"是否存在 SECTION"。
    fn source_bounds(&self) -> (f64, f64, bool) {
        let mut min_bound = 0.0;
        let mut max_bound = f64::INFINITY;
        let has_section = self.section.is_some();
        if let Some(section) = self.section {
            min_bound = section.start_sec.max(0.0);
            if let Some(length) = section.length_sec.filter(|len| len.is_finite() && *len > 0.0) {
                max_bound = (section.start_sec + length).max(min_bound);
            }
        }
        if let Some(total) = self
            .media_duration_sec
            .filter(|value| value.is_finite() && *value > 0.0)
        {
            max_bound = max_bound.min(total);
        }
        (min_bound, max_bound, has_section)
    }

    /// 入口锚点（源域秒）：正放是窗口**起点**，倒放是窗口**终点**。
    fn source_anchor(&self, min_bound: f64, max_bound: f64, has_section: bool) -> f64 {
        let section_start = self.section.map(|section| section.start_sec).unwrap_or(0.0);
        // SOFFS 的负值只在"正放 + 无 SECTION"时是合法的（REAPER 左延伸 item 的
        // 前导静音）；其余情形按 0 处理，与 App 一致。
        let soffs_nonneg = self.start_offset_sec.max(0.0);

        let primary = if has_section {
            if self.reversed {
                max_bound - soffs_nonneg
            } else {
                section_start + soffs_nonneg
            }
        } else if self.reversed {
            if max_bound.is_finite() {
                max_bound - soffs_nonneg
            } else {
                // 媒体时长未知 ⇒ 无法镜像，退回原始读数（不编造）。
                self.start_offset_sec
            }
        } else {
            // 兼容 REAPER 左延伸 item（负 SOFFS = 前导静音）：保留原始符号，
            // 仅在媒体时长已知时对上界钳制。
            self.start_offset_sec.clamp(-1_000_000.0, max_bound)
        };

        let mut anchor = if has_section || self.reversed {
            primary.clamp(min_bound, max_bound)
        } else if primary.is_finite() {
            primary
        } else {
            0.0
        };

        if has_section {
            // 兼容部分工程把 SOFFS 写成**绝对源坐标**而非段相对坐标的写法：
            // 取"能消费更长一段"的那个解释。
            let alt = soffs_nonneg.clamp(min_bound, max_bound);
            let primary_span = if self.reversed {
                anchor - min_bound
            } else {
                max_bound - anchor
            };
            let alt_span = if self.reversed {
                alt - min_bound
            } else {
                max_bound - alt
            };
            if alt_span > primary_span {
                anchor = alt;
            }
        }
        anchor
    }

    /// 该 take 消费的**正向**源窗口 `(start, end)`，恒有 `start ≤ end`。
    ///
    /// 这是交给内核的唯一窗口：倒放时内核用它的 `end` 反向取窗，**不要再翻一次**。
    pub fn forward_source_window(&self) -> (f64, f64) {
        let (min_bound, max_bound, has_section) = self.source_bounds();
        let consumed = self.consumed_sec();
        let anchor = self.source_anchor(min_bound, max_bound, has_section);

        if self.loop_enabled {
            // Loop：源窗口取"锚点 → 可用边界"的完整区间，**不**按 item 长度钳制 ——
            // 超出窗口的播放时间由引擎按该窗口周期回绕产生循环内容。
            if self.reversed {
                return (min_bound.min(anchor), anchor);
            }
            let end = if max_bound.is_finite() {
                max_bound.max(anchor)
            } else {
                // 媒体时长未知时退化为按消耗量截取（无回绕素材可用）。
                anchor + consumed
            };
            return (anchor, end);
        }

        if self.reversed {
            let end = anchor;
            let start = (end - consumed).max(min_bound).min(end);
            (start, end)
        } else {
            // 非 Loop 正放：派生窗口 —— 终点 = 起点 + 消耗量，**不**按媒体时长钳制。
            // REAPER 中 LENGTH 大于可用源的 item 其超出部分为静音尾巴，导入必须保真
            // （派生模型下 source_end 与 length 保持一致，Slip/拖边的语义才成立）。
            let start = anchor;
            (start, start + consumed)
        }
    }

    /// 正向窗口的**终点** —— 内核倒放消费数学唯一用到的量。
    pub fn forward_window_end_sec(&self) -> f64 {
        self.forward_source_window().1
    }
}

/// 正向窗口终点 → 宿主 `SOFFS`（镜像域）。[`TakeGeometry::forward_source_window`]
/// 的**逆运算**：写回宿主时必须再翻回镜像域，否则倒放 take 会被整体平移
/// `媒体长 − 正向终点`。
///
/// `axis_sec` 是镜像轴 —— 与读取侧同源，取 `min(段末端, 媒体长)`。
/// 轴不可用（媒体时长未知且无 SECTION）时返回 `None`：宁可不写，也不写一个错的起点。
pub fn host_start_offset(forward_end_sec: f64, axis_sec: Option<f64>) -> Option<f64> {
    let axis = axis_sec.filter(|value| value.is_finite() && *value > 0.0)?;
    if !forward_end_sec.is_finite() {
        return None;
    }
    Some(axis - forward_end_sec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> TakeGeometry {
        TakeGeometry {
            start_offset_sec: 0.0,
            item_length_sec: 1.0,
            play_rate_signed: 1.0,
            media_duration_sec: Some(2.0),
            section: None,
            reversed: false,
            loop_enabled: false,
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// 【为什么钉这条】这是 `probe/ara/REVERSE-FINDINGS.md` 的实测夹具：媒体 2.0 s、
    /// 正向窗口 `[0.25, 1.25]`、倒放 take 的 `D_STARTOFFS = 0.75`。
    /// 复原点错一位，用户就会看到"方向与 REAPER 完全相反"。
    #[test]
    fn reversed_take_un_mirrors_to_the_forward_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            reversed: true,
            ..base()
        };
        let (start, end) = geometry.forward_source_window();
        assert!(close(start, 0.25), "got {start}");
        assert!(close(end, 1.25), "got {end}");
    }

    /// 同一个夹具，但 SECTION 覆盖整个媒体（REAPER 实测给的就是这个形状）：
    /// 段末端与媒体末端重合，结果必须与上一条相同。
    #[test]
    fn a_full_media_reverse_section_matches_the_sectionless_case() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            section: Some(SourceSection {
                start_sec: 0.0,
                length_sec: Some(2.0),
            }),
            reversed: true,
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), (0.25, 1.25));
    }

    /// 正放 take 的窗口就是"锚点 → 锚点 + 消耗量"，且**不**按媒体时长钳制
    /// （超出部分是静音尾巴，REAPER 的 Slip/拖边语义依赖它）。
    #[test]
    fn forward_window_is_derived_and_not_clamped_to_the_media() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.25,
            item_length_sec: 3.0,
            media_duration_sec: Some(2.0),
            ..base()
        };
        let (start, end) = geometry.forward_source_window();
        assert!(close(start, 0.25));
        assert!(close(end, 3.25), "must extend past the media as a silence tail");
    }

    /// 负 SOFFS（REAPER 左延伸 item 的前导静音）只在"正放 + 无 SECTION"时保留符号。
    /// 这条一旦被"统一钳到 [0, D]"改掉，前导静音就会整段错位。
    #[test]
    fn negative_soffs_survives_for_a_forward_take_without_a_section() {
        let geometry = TakeGeometry {
            start_offset_sec: -0.5,
            item_length_sec: 1.0,
            ..base()
        };
        let (start, end) = geometry.forward_source_window();
        assert!(close(start, -0.5), "got {start}");
        assert!(close(end, 0.5));
    }

    /// 负速率是倒放的**另一个**来源，必须与 SECTION MODE 1 等价。
    #[test]
    fn negative_play_rate_is_reversed_like_a_section_mode() {
        assert!(TakeGeometry::reversed_from_host(-1.0, 0));
        assert!(TakeGeometry::reversed_from_host(1.0, 1));
        assert!(!TakeGeometry::reversed_from_host(1.0, 0));
        // 幅值用于消耗量，符号不参与窗口宽度。
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            play_rate_signed: -1.0,
            reversed: TakeGeometry::reversed_from_host(-1.0, 0),
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), (0.25, 1.25));
    }

    /// 倒放 Loop 走"锚点 → 区间起点"的完整区间，不被 item 长度钳制。
    #[test]
    fn reversed_loop_takes_the_whole_interval_from_the_anchor() {
        let geometry = TakeGeometry {
            start_offset_sec: 1.0,
            item_length_sec: 0.5,
            media_duration_sec: Some(4.0),
            reversed: true,
            loop_enabled: true,
            ..base()
        };
        // 锚点 = 4.0 − 1.0 = 3.0；回绕区间 = [0, 3.0]。
        assert_eq!(geometry.forward_source_window(), (0.0, 3.0));
    }

    /// 正放 Loop：区间到媒体末端。
    #[test]
    fn forward_loop_extends_to_the_media_end() {
        let geometry = TakeGeometry {
            start_offset_sec: 1.0,
            item_length_sec: 0.5,
            media_duration_sec: Some(4.0),
            loop_enabled: true,
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), (1.0, 4.0));
    }

    /// 媒体时长未知时**不编造**镜像：退回原始读数，宁可偏移也不静默给错值。
    #[test]
    fn an_unknown_media_duration_does_not_invent_a_mirror() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            media_duration_sec: None,
            reversed: true,
            ..base()
        };
        let (start, end) = geometry.forward_source_window();
        assert!(close(end, 0.75), "anchor falls back to the raw offset");
        assert!(close(start, 0.0), "clamped to the lower bound");
    }

    /// 写回是读取的逆运算：`host_start_offset(正向终点)` 必须还原出宿主读数。
    #[test]
    fn host_start_offset_inverts_the_forward_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            reversed: true,
            ..base()
        };
        let (_, end) = geometry.forward_source_window();
        let mirrored = host_start_offset(end, Some(2.0)).expect("axis known");
        assert!(close(mirrored, 0.75), "got {mirrored}");
    }

    /// 轴不可用时宁可拒绝写回，也不写出一个错的起点。
    #[test]
    fn host_start_offset_refuses_without_an_axis() {
        assert_eq!(host_start_offset(1.25, None), None);
        assert_eq!(host_start_offset(1.25, Some(0.0)), None);
        assert_eq!(host_start_offset(f64::NAN, Some(2.0)), None);
    }

    /// 无 SECTION、有媒体时长：SOFFS 即锚点，窗口是派生区间。
    #[test]
    fn a_plain_trim_maps_directly() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.25,
            item_length_sec: 1.0,
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), (0.25, 1.25));
    }

    /// 【为什么钉住"退化"而不是"修好"】某些输入会给出零长窗口：镜像锚点被钳到下界
    /// 时该 take 没有可反向消费的源内容，正放 Loop 的锚点被钳到段末端时回绕区间为空。
    ///
    /// 这两条同时说明 App 侧那段"零长度窗口回退到基于锚点的正向区间"的兜底是**死
    /// 代码**：它算出的区间与主路径逐位相同（倒放分支的公式一模一样；正放分支只是
    /// 多钳一次上界，而退化时本来就没超上界）。本次删除它而不是搬进内核 —— 保留
    /// 一个永远不改结果的函数只会让后来者以为它在保护什么。
    #[test]
    fn degenerate_windows_are_pinned_as_known_boundaries() {
        // 倒放、无 SECTION、SOFFS 到达媒体长 ⇒ 镜像锚点钳到下界 0。
        let beyond_media = TakeGeometry {
            start_offset_sec: 2.0,
            media_duration_sec: Some(2.0),
            reversed: true,
            ..base()
        };
        assert_eq!(beyond_media.forward_source_window(), (0.0, 0.0));

        // 正放 Loop、锚点被钳到段末端 ⇒ 回绕区间为空。
        let clamped_loop = TakeGeometry {
            start_offset_sec: 1.5,
            item_length_sec: 1.0,
            media_duration_sec: Some(2.0),
            section: Some(SourceSection {
                start_sec: 0.25,
                length_sec: Some(1.0),
            }),
            loop_enabled: true,
            ..base()
        };
        assert_eq!(clamped_loop.forward_source_window(), (1.25, 1.25));
    }

    /// 内容存在的区间里（`SOFFS` 落在段内、且窗口能装下消耗量）窗口严格非零 ——
    /// 这是那段兜底**本来想保护**的范围，也是实际工程里唯一会出现的范围。
    #[test]
    fn a_content_bearing_window_is_never_degenerate() {
        for start_offset in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for reversed in [false, true] {
                let geometry = TakeGeometry {
                    start_offset_sec: start_offset,
                    item_length_sec: 1.0,
                    media_duration_sec: Some(2.0),
                    reversed,
                    ..base()
                };
                let (start, end) = geometry.forward_source_window();
                assert!(
                    end > start,
                    "degenerate window for offset={start_offset} reversed={reversed}: {start}..{end}"
                );
            }
        }
    }
}
