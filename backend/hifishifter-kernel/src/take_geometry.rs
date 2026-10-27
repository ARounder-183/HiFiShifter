//! REAPER take 几何 → 内核 Clip 源窗口的**唯一**解算。
//!
//! ## 为什么需要它（一次真实事故）
//!
//! 两个前端都要回答同一个问题："这个 take 实际消费源媒体的哪一段？"
//!
//! - 独立 App 从 .rpp 文本读 SOFFS / LENGTH / PLAYRATE / SOURCE SECTION；
//! - ARA 插件从宿主实时 API 读 D_STARTOFFS / D_LENGTH / D_PLAYRATE。
//!
//! 此前两侧各写了一份：App 的那份经过实测打磨，插件的那份是"按现象凑出来"的近似。
//! 于是**同一个工程在两个前端听感不同** —— 用户报障"倒放的 Take，其 Offset 的
//! 逻辑不对，方向与 REAPER 完全相反"。
//!
//! 现在只有这一份：两侧各自把宿主读数装进 [TakeGeometry]，调用
//! [TakeGeometry::forward_source_window]，结果必然一致。
//!
//! ## 坐标约定（这是全部混乱的源头，先钉死）
//!
//! 内核的倒放消费模型（state::model::clip_playback_window_sec）是：
//!
//! 正放 win = [source_start, source_start + length*rate)
//!
//! 倒放 win = [source_end - length*rate, source_end)
//!
//!
//! 也就是说内核**只**认一个**正向**窗口，倒放时用它的**终点**反向取窗。而宿主报的
//! SOFFS 在倒放 take 上是**镜像**量（D_STARTOFFS = 媒体长 - 正向起点 - 跨度，
//! 见模块夹具：媒体 2.0 s、正向窗口 [0.25, 1.25] ⇒ D_STARTOFFS = 0.75）。因此本模块
//! 的职责就是**把镜像翻回正向**，产出内核要的窗口。
//!
//! ## 坐标域：两个类型，一个入口
//!
//! 镜像量与正向窗口都是 f64。把它们都写成裸 f64，就迟早会有人把一边当另一边用 ——
//! 那会让整段内容平移 2*soffs - D 秒，而且"看起来只是偏了一点"。所以：
//!
//! - [ForwardWindow] 只由本模块产出；
//! - [MirroredOffset] 只由 [host_start_offset]（读取的逆运算）产出；
//! - 两者都不实现 Deref，想取值必须显式调 .start() / .end() / .value()。

/// 一个 SOURCE SECTION 的边界。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceSection {
    /// 段起点（秒）。
    pub start_sec: f64,
    /// 段长度（秒）；None 或非正表示"到媒体末尾"。
    pub length_sec: Option<f64>,
}

/// 内核正向域的消费窗口（恒 start <= end）。
///
/// 【为什么必须是 newtype】同一份几何在宿主侧是**镜像域**（D - 正向终点），
/// 在内核侧是**正向域**。两者都是 f64，靠命名区分不了 —— 而把镜像值当正向值用的
/// 后果是整段内容平移 2*soffs - D 秒：静音与错音都出现过（见模块文档）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForwardWindow {
    start: f64,
    end: f64,
}

impl ForwardWindow {
    /// 有限且 start <= end 才构成一个窗口；否则返回 None（调用方按"没有内容"处理）。
    pub fn new(start: f64, end: f64) -> Option<Self> {
        (start.is_finite() && end.is_finite() && start <= end).then_some(Self { start, end })
    }
    pub fn start(self) -> f64 {
        self.start
    }
    pub fn end(self) -> f64 {
        self.end
    }
    pub fn span(self) -> f64 {
        self.end - self.start
    }
}

/// 宿主镜像域的取起点（D_STARTOFFS）。
///
/// 【为什么也要 newtype】它是 [ForwardWindow] 的镜像，只在写给宿主时出现。
/// 单独一个类型，才能让"这段代码在处理镜像量"在签名上可见。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MirroredOffset(f64);

impl MirroredOffset {
    pub fn value(self) -> f64 {
        self.0
    }
}

/// 一个 take 的几何输入（宿主无关）。
///
/// 【为什么是纯数据而不是引用宿主结构】两侧的宿主类型完全不同（RPP 的
/// ReaperTake 与插件的实时 getter），但它们都能拆成这几个 f64。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TakeGeometry {
    /// SOFFS / D_STARTOFFS。**倒放时是镜像量**，见模块文档。
    pub start_offset_sec: f64,
    /// item 长度（LENGTH / D_LENGTH），秒。
    pub item_length_sec: f64,
    /// 播放速率（PLAYRATE[0] / D_PLAYRATE），**保留符号**。
    ///
    /// 【为什么保留符号】App 用 play_rate < 0 作为倒放的**两个来源之一**。
    /// 插件曾把它 .abs() 掉、只认方向位，于是"只用负速率表达倒放"的 take 被
    /// 静默按正放渲染。
    pub play_rate_signed: f64,
    /// 媒体总时长（秒）；None = 未知（媒体缺失 / 尚未解码）。
    pub media_duration_sec: Option<f64>,
    /// SECTION 边界（RPP 的 STARTPOS/LENGTH，或实时 PCM_Source_GetSectionInfo）。
    pub section: Option<SourceSection>,
    /// 方向：`play_rate < 0` **或** SECTION MODE > 0（App），或宿主方向位（插件）。
    ///
    /// 【为什么是三态】REAPER 的 "Toggle take reverse" 是"正 PLAYRATE + SECTION MODE 1"，
    /// 方向**唯一**来源就是那个方向位。读不到它时旧实现按正放处理 —— 而"读不到"与
    /// "确实是正放"是两件完全不同的事：前者会把倒放片段按正放渲染（方向与 REAPER
    /// 完全相反、偏移量差 `2·soffs + consumed − D`），后者是正常。
    ///
    /// `None` = 读不到。此时 [`TakeGeometry::forward_source_window`] 返回 `None`
    /// （调用方保留宿主镜像坐标并留读数），**不允许**静默按正放渲染。
    pub reversed: Option<bool>,
    /// 循环源（REAPER "Loop source" / B_LOOPSRC）。
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

    /// 速率幅值，下限 0.01（与 App 的 raw_play_rate.abs().max(0.01) 一致）。
    pub fn play_rate_magnitude(&self) -> f64 {
        if self.play_rate_signed.is_finite() {
            self.play_rate_signed.abs().max(0.01)
        } else {
            1.0
        }
    }

    /// 该 take 在源域**消耗**的秒数：item 长度 × |速率|。
    pub fn consumed_sec(&self) -> f64 {
        (self.item_length_sec.max(0.0)) * self.play_rate_magnitude()
    }

    /// 可寻址的源区间 (min_bound, max_bound) 与"是否存在 SECTION"。
    fn source_bounds(&self) -> (f64, f64, bool) {
        let min_bound = self
            .section
            .map(|section| section.start_sec.max(0.0))
            .unwrap_or(0.0);
        // 【轴只有一个来源】`take_axis_sec` 与写回侧调的是**同一个函数** —— 读用一份
        // 轴、写用另一份，每编辑一次就漂 `Δaxis`。这里不再就地重算一遍。
        let max_bound =
            take_axis_sec(self.section, self.media_duration_sec, None).unwrap_or(f64::INFINITY);
        (min_bound, max_bound, self.section.is_some())
    }


    /// 方向位是否**读得到**。
    pub fn direction_known(&self) -> bool {
        self.reversed.is_some()
    }

    /// 有效方向（`None` 按正放参与数学，但调用方必须先问 `direction_known`）。
    pub fn is_reversed(&self) -> bool {
        self.reversed.unwrap_or(false)
    }

    /// 镜像轴；倒放且轴未知时为 `None`（见 [`take_axis_sec`]）。
    pub fn axis_sec(&self) -> Option<f64> {
        let (_, max_bound, _) = self.source_bounds();
        max_bound.is_finite().then_some(max_bound)
    }

    /// 入口锚点（源域秒）：正放是窗口**起点**，倒放是窗口**终点**。
    ///
    /// 【为什么返回 `Option`】倒放的锚点是 `轴 − soffs`，轴未知时这个式子没有值。
    /// 旧实现在那种情形下"退回原始读数"—— 那是把"不知道"伪装成"知道"：镜像量被当成
    /// 正向起点用，整段内容偏移 `2·soffs − D`（F-4 夹具：D=2、soffs=0.75 ⇒ 得
    /// `[0, 0.75]`，正确是 `[0.25, 1.25]`）。宁可不给窗口，也不给一个错的。
    fn source_anchor(&self, min_bound: f64, max_bound: f64, has_section: bool) -> Option<f64> {
        // 【方向读不到 ⇒ 没有锚点】"不知道是不是倒放"与"确实是正放"必须分开：
        // 前者的正确处置是"保留宿主镜像坐标 + 留读数"，不是按正放解算。
        let reversed = self.reversed?;
        let section_start = self.section.map(|section| section.start_sec).unwrap_or(0.0);
        // SOFFS 的负值只在"正放 + 无 SECTION"时是合法的（REAPER 左延伸 item 的
        // 前导静音）；其余情形按 0 处理，与 App 一致。
        let soffs_nonneg = self.start_offset_sec.max(0.0);

        let primary = if has_section {
            if reversed {
                max_bound - soffs_nonneg
            } else {
                section_start + soffs_nonneg
            }
        } else if reversed {
            // 轴未知 ⇒ 无法镜像（不编造）。
            if !max_bound.is_finite() {
                return None;
            }
            max_bound - soffs_nonneg
        } else {
            // 兼容 REAPER 左延伸 item（负 SOFFS = 前导静音）：保留原始符号，
            // 仅在媒体时长已知时对上界钳制。
            self.start_offset_sec.clamp(-1_000_000.0, max_bound)
        };

        let mut anchor = if has_section || reversed {
            primary.clamp(min_bound, max_bound)
        } else if primary.is_finite() {
            primary
        } else {
            0.0
        };

        // 【这条"兼容"为什么只对正放成立】它想说"SOFFS 也可能是绝对源坐标，取能
        // 消费更长一段的那个解释"。对正放，这是一个**常量选择**：SECTION 起点 > 0 时
        // max - alt > max - (section + soffs) 恒成立，于是它总选 alt —— 行为可预期、
        // 可测试，且既有工程依赖它。对倒放，同一个判据变成 soffs > max/2 的
        // **数据相关翻转**：同一个 take 只要 offset 跨过区间中点，解释就换一种。
        // 而倒放 take 的 SOFFS 是**镜像量**（见模块文档的实测夹具），它只有一个正确
        // 解释 max_bound - soffs；再让"更长跨度"去决定用哪种，就是把解释权交给被解释
        // 的数。宿主分割右半段恰好把 offset 推过中点 —— 那正是"分割后右段偏移完全
        // 错误"的机理。
        if has_section && !reversed {
            let alt = soffs_nonneg.clamp(min_bound, max_bound);
            let primary_span = max_bound - anchor;
            let alt_span = max_bound - alt;
            if alt_span > primary_span {
                anchor = alt;
            }
        }
        Some(anchor)
    }

    /// 该 take 消费的**正向**源窗口，恒有 start <= end。
    ///
    /// 这是交给内核的唯一窗口：倒放时内核用它的 end 反向取窗，**不要再翻一次**。
    ///
    /// 【为什么返回 `Option`】三种"不知道"必须能被调用方看见，而不是变成一个看起来
    /// 合法的窗口：① 方向位读不到；② 倒放但没有镜像轴；③ 几何退化（NaN / 负长度）。
    /// 调用方拿到 `None` 时的正确处置是**保留宿主镜像坐标并留读数**，不是渲染。
    pub fn forward_source_window(&self) -> Option<ForwardWindow> {
        let (min_bound, max_bound, has_section) = self.source_bounds();
        let consumed = self.consumed_sec();
        let reversed = self.reversed?;
        let anchor = self.source_anchor(min_bound, max_bound, has_section)?;

        let (start, end) = if self.loop_enabled {
            // Loop：源窗口取"锚点 → 可用边界"的完整区间，**不**按 item 长度钳制 ——
            // 超出窗口的播放时间由引擎按该窗口周期回绕产生循环内容。
            if reversed {
                (min_bound.min(anchor), anchor)
            } else {
                let end = if max_bound.is_finite() {
                    max_bound.max(anchor)
                } else {
                    // 媒体时长未知时退化为按消耗量截取（无回绕素材可用）。
                    anchor + consumed
                };
                (anchor, end)
            }
        } else if reversed {
            let end = anchor;
            ((end - consumed).max(min_bound).min(end), end)
        } else {
            // 非 Loop 正放：派生窗口 —— 终点 = 起点 + 消耗量，**不**按媒体时长钳制。
            // REAPER 中 LENGTH 大于可用源的 item 其超出部分为静音尾巴，导入必须保真。
            (anchor, anchor + consumed)
        };
        ForwardWindow::new(start, end)
    }

    /// 正向窗口的**终点** —— 内核倒放消费数学唯一用到的量。
    pub fn forward_window_end_sec(&self) -> Option<f64> {
        self.forward_source_window().map(ForwardWindow::end)
    }
}


/// 倒放 take 的**镜像轴**（秒）—— 读写两侧唯一的来源。
///
/// 【为什么必须只有一个来源】镜像关系是 `soffs = axis − 正向终点`。读的时候用一个轴、
/// 写的时候用另一个，每编辑一次就漂 `Δaxis`。而轴上可用的候选有三个，优先级不同：
///   1. **SECTION 末端**（`PCM_Source_GetSectionInfo` 的 `offset + length`）——
///      宿主自己说"这一段到这儿为止"，是最强的证据；
///   2. **ARA 源时长**（`AudioSource.durationSeconds`）—— 与授权无关，映射阶段就有；
///   3. **解码时长**（本地文件）—— 容器头差一点就会差几毫秒。
///
/// 取 `min(1, 2|3)`：宿主把段设在媒体中间时，可寻址的上界就是段末端；否则就是媒体全长。
/// 三者都没有 ⇒ `None`（**不猜**）。
///
/// 【为什么这里不接收 `soffs`】轴是**媒体**的属性，`soffs` 是**宿主读数**。让轴依赖
/// 被解释的那个数，就再也无法用它去校验那个数 —— 而往返自校验正是本模块现在唯一的
/// 真机取证手段。
pub fn take_axis_sec(
    section: Option<SourceSection>,
    ara_duration: Option<f64>,
    decoded_duration: Option<f64>,
) -> Option<f64> {
    let mut axis = f64::INFINITY;
    if let Some(section) = section {
        let start = section.start_sec.max(0.0);
        if let Some(length) = section.length_sec.filter(|len| len.is_finite() && *len > 0.0) {
            axis = axis.min((start + length).max(start));
        }
    }
    // 【为什么 ARA 优先于解码时长】两者的差别通常只有一个容器头（几毫秒），但 ARA 的
    // 那个数是宿主与插件**共享**的读数，而解码时长是本地文件的事实。写回宿主时用前者，
    // 宿主读回时才会逐位相等 —— 往返不变量靠的就是这一点。
    for candidate in [ara_duration, decoded_duration].into_iter().flatten() {
        if candidate.is_finite() && candidate > 0.0 {
            axis = axis.min(candidate);
            break;
        }
    }
    axis.is_finite().then_some(axis)
}

/// **读 → 写 → 读**的残差：把正向窗口按 `axis` 折回宿主读数，与宿主原本给的
/// `soffs` 相减。
///
/// 【为什么这是一条不变量而不是一个测试】镜像约定（`soffs = axis − 正向终点` 还是
/// `soffs = 正向起点`）属于**宿主**，而仓库里没有任何一处能证明 REAPER 用的是哪一条。
/// 把"读出来的窗口再写回去必须回到原值"变成一行运行时读数，真机日志就能直接回答这个
/// 问题：残差为 0 ⇒ 约定正确；残差是 `2·soffs − axis` 之类 ⇒ 约定反了。
pub fn roundtrip_residual(
    forward: ForwardWindow,
    axis_sec: Option<f64>,
    host_offset_sec: f64,
) -> Option<f64> {
    let axis = axis_sec.filter(|value| value.is_finite() && *value > 0.0)?;
    if !host_offset_sec.is_finite() {
        return None;
    }
    Some((axis - forward.end()) - host_offset_sec)
}

/// 把父段的**正向**窗口按"子段在播放时间上的偏移 / 长度"切成子段窗口。
///
/// 【为什么从这里推导而不是重读宿主】父段窗口是**分割前真的在响的那一段**，子段必须
/// 划分它 —— 这是分割的定义，与宿主怎么填 D_STARTOFFS 无关。独立 App 的分割
/// （state::model::split_clip_take_window）用的就是这条算式；宿主读数只作校验。
///
/// `offset_sec` 是子段起点相对左段起点的**播放时间**偏移，`length_sec` 是它的播放长度。
/// 方向决定消费轴：正放沿 +rate 推进，倒放沿 −rate 推进。
pub fn split_child_window(
    left: ForwardWindow,
    rate: f64,
    offset_sec: f64,
    length_sec: f64,
    reversed: bool,
) -> Option<ForwardWindow> {
    let rate = if rate.is_finite() && rate > 1e-6 {
        rate
    } else {
        1.0
    };
    let offset = if offset_sec.is_finite() {
        offset_sec.max(0.0)
    } else {
        0.0
    };
    let length = if length_sec.is_finite() {
        length_sec.max(0.0)
    } else {
        0.0
    };
    if reversed {
        let end = left.end() - offset * rate;
        ForwardWindow::new(end - length * rate, end)
    } else {
        let start = left.start() + offset * rate;
        ForwardWindow::new(start, start + length * rate)
    }
}

/// 正向窗口终点 → 宿主 SOFFS（镜像域）。[TakeGeometry::forward_source_window]
/// 的**逆运算**：写回宿主时必须再翻回镜像域，否则倒放 take 会被整体平移
/// 媒体长 − 正向终点。
///
/// axis_sec 是镜像轴 —— 与读取侧同源，取 min(段末端, 媒体长)。
/// 轴不可用（媒体时长未知且无 SECTION）时返回 None：宁可不写，也不写一个错的起点。
pub fn host_start_offset(forward: ForwardWindow, axis_sec: Option<f64>) -> Option<MirroredOffset> {
    let axis = axis_sec.filter(|value| value.is_finite() && *value > 0.0)?;
    if !forward.end().is_finite() {
        return None;
    }
    Some(MirroredOffset(axis - forward.end()))
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
            reversed: Some(false),
            loop_enabled: false,
        }
    }

    fn win(start: f64, end: f64) -> ForwardWindow {
        ForwardWindow::new(start, end).unwrap()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// 【为什么钉这条】这是探针实测夹具：媒体 2.0 s、正向窗口 [0.25, 1.25]、
    /// 倒放 take 的 D_STARTOFFS = 0.75。复原点错一位，用户就会看到"方向与 REAPER
    /// 完全相反"。
    #[test]
    fn reversed_take_un_mirrors_to_the_forward_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            reversed: Some(true),
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), Some(win(0.25, 1.25)));
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
            reversed: Some(true),
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), Some(win(0.25, 1.25)));
    }

    /// **倒放 take 的正向窗口落在素材前半段时，SOFFS 仍然是镜像量。**
    ///
    /// 【为什么这条是"分割后右段偏移错误"的回归】媒体 2.0、正向窗口 [0.1, 0.9) ⇒
    /// 镜像后的 D_STARTOFFS = 1.1 > D/2。旧实现里的"绝对坐标"第二解释会在这里
    /// 胜出（它比较的是"哪种解释能消费更长一段"），把 anchor 换成 1.1，窗口变成
    /// [0.3, 1.1) —— 整体偏移 0.2 s。宿主分割右半段恰好把 soffs 推过 D/2，
    /// 于是"一分就错"。倒放分支只能有一个解释：max - soffs。
    #[test]
    fn a_reversed_take_in_the_first_half_keeps_its_mirrored_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 1.1,
            item_length_sec: 0.8,
            section: Some(SourceSection {
                start_sec: 0.0,
                length_sec: Some(2.0),
            }),
            reversed: Some(true),
            ..base()
        };
        let window = geometry.forward_source_window().expect("window known");
        assert!(close(window.start(), 0.1), "got {window:?}");
        assert!(close(window.end(), 0.9), "got {window:?}");
    }

    /// 同一个夹具把 soffs 扫过中点：倒放分支的窗口必须**连续**，不能在中点换解释。
    #[test]
    fn a_reversed_window_never_switches_interpretation_at_the_midpoint() {
        for soffs in [0.1, 0.4, 0.9, 1.1, 1.6] {
            let geometry = TakeGeometry {
                start_offset_sec: soffs,
                item_length_sec: 0.5,
                section: Some(SourceSection {
                    start_sec: 0.0,
                    length_sec: Some(2.0),
                }),
                reversed: Some(true),
                ..base()
            };
            let window = geometry.forward_source_window().expect("window known");
            assert!(close(window.end(), 2.0 - soffs), "soffs={soffs}");
            assert!(
                close(window.start(), (2.0 - soffs - 0.5).max(0.0)),
                "soffs={soffs}"
            );
        }
    }

    /// 正放 + SECTION 的既有行为逐位不变：SECTION 起点 > 0 时仍按绝对坐标读。
    #[test]
    fn a_forward_take_with_a_section_still_reads_the_absolute_offset() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.5,
            item_length_sec: 0.5,
            section: Some(SourceSection {
                start_sec: 0.25,
                length_sec: Some(1.0),
            }),
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), Some(win(0.5, 1.0)));
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
        assert_eq!(geometry.forward_source_window(), Some(win(0.25, 3.25)));
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
        assert_eq!(geometry.forward_source_window(), Some(win(-0.5, 0.5)));
    }

    /// 负速率是倒放的**另一个**来源，必须与 SECTION MODE 1 等价。
    #[test]
    fn negative_play_rate_is_reversed_like_a_section_mode() {
        assert!(TakeGeometry::reversed_from_host(-1.0, 0));
        assert!(TakeGeometry::reversed_from_host(1.0, 1));
        assert!(!TakeGeometry::reversed_from_host(1.0, 0));
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            play_rate_signed: -1.0,
            reversed: Some(TakeGeometry::reversed_from_host(-1.0, 0)),
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), Some(win(0.25, 1.25)));
    }

    /// 倒放 Loop 走"锚点 → 区间起点"的完整区间，不被 item 长度钳制。
    #[test]
    fn reversed_loop_takes_the_whole_interval_from_the_anchor() {
        let geometry = TakeGeometry {
            start_offset_sec: 1.0,
            item_length_sec: 0.5,
            media_duration_sec: Some(4.0),
            reversed: Some(true),
            loop_enabled: true,
            ..base()
        };
        // 锚点 = 4.0 − 1.0 = 3.0；回绕区间 = [0, 3.0]。
        assert_eq!(geometry.forward_source_window(), Some(win(0.0, 3.0)));
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
        assert_eq!(geometry.forward_source_window(), Some(win(1.0, 4.0)));
    }

    /// **轴未知时不给窗口** —— 旧实现"退回原始读数"，把镜像量当成正向起点用。
    ///
    /// 【为什么这是本模块唯一正确的处置】`soffs` 在倒放 take 上是镜像量
    /// （`axis − 正向终点`）。轴不知道时，`soffs` 与"正向起点"之间**没有**任何可用的
    /// 等式 —— 拿它当起点，整段内容就偏移 `2·soffs − D`（F-4 夹具：D=2、soffs=0.75
    /// ⇒ 得 `[0, 0.75]`，正确是 `[0.25, 1.25]`）。宁可交 `None` 让调用方保留宿主
    /// 镜像坐标并留读数，也不交一个看起来合法的错窗口。
    #[test]
    fn an_unknown_axis_yields_no_window_instead_of_a_raw_offset() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            media_duration_sec: None,
            reversed: Some(true),
            ..base()
        };
        assert_eq!(
            geometry.forward_source_window(),
            None,
            "轴未知时不得把镜像量当成正向起点"
        );
        // 正放不受影响：它的窗口不需要轴。
        let forward = TakeGeometry {
            start_offset_sec: 0.75,
            media_duration_sec: None,
            reversed: Some(false),
            ..base()
        };
        assert_eq!(forward.forward_source_window(), Some(win(0.75, 1.75)));
    }

    /// **方向位读不到时不给窗口** —— "不知道是不是倒放"与"确实是正放"必须分开。
    #[test]
    fn an_unknown_direction_yields_no_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            reversed: None,
            ..base()
        };
        assert!(!geometry.direction_known());
        assert_eq!(geometry.forward_source_window(), None);
    }


    /// 写回是读取的逆运算：host_start_offset(正向终点) 必须还原出宿主读数。
    #[test]
    fn host_start_offset_inverts_the_forward_window() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.75,
            reversed: Some(true),
            ..base()
        };
        let window = geometry.forward_source_window().expect("window known");
        let mirrored = host_start_offset(window, Some(2.0)).expect("axis known");
        assert!(close(mirrored.value(), 0.75), "got {mirrored:?}");
    }

    /// 轴不可用时宁可拒绝写回，也不写出一个错的起点。
    #[test]
    fn host_start_offset_refuses_without_an_axis() {
        let window = win(0.25, 1.25);
        assert_eq!(host_start_offset(window, None), None);
        assert_eq!(host_start_offset(window, Some(0.0)), None);
        // 窗口本身不可能非有限（[ForwardWindow::new] 已经拒绝），这里只钉"轴"这一侧。
    }

    /// 无 SECTION、有媒体时长：SOFFS 即锚点，窗口是派生区间。
    #[test]
    fn a_plain_trim_maps_directly() {
        let geometry = TakeGeometry {
            start_offset_sec: 0.25,
            item_length_sec: 1.0,
            ..base()
        };
        assert_eq!(geometry.forward_source_window(), Some(win(0.25, 1.25)));
    }

    /// 【为什么钉住"退化"而不是"修好"】某些输入会给出零长窗口：镜像锚点被钳到下界
    /// 时该 take 没有可反向消费的源内容，正放 Loop 的锚点被钳到段末端时回绕区间为空。
    #[test]
    fn degenerate_windows_are_pinned_as_known_boundaries() {
        let beyond_media = TakeGeometry {
            start_offset_sec: 2.0,
            media_duration_sec: Some(2.0),
            reversed: Some(true),
            ..base()
        };
        assert_eq!(beyond_media.forward_source_window(), Some(win(0.0, 0.0)));

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
        assert_eq!(clamped_loop.forward_source_window(), Some(win(1.25, 1.25)));
    }

    /// 内容存在的区间里（SOFFS 落在段内、且窗口能装下消耗量）窗口严格非零 ——
    /// 这是那段兜底**本来想保护**的范围，也是实际工程里唯一会出现的范围。
    #[test]
    fn a_content_bearing_window_is_never_degenerate() {
        for start_offset in [0.0, 0.25, 0.5, 0.75, 1.0] {
            for reversed in [false, true] {
                let geometry = TakeGeometry {
                    start_offset_sec: start_offset,
                    item_length_sec: 1.0,
                    media_duration_sec: Some(2.0),
                    reversed: Some(reversed),
                    ..base()
                };
                let window = geometry.forward_source_window().expect("window known");
                assert!(
                    window.end() > window.start(),
                    "degenerate window for offset={start_offset} reversed={reversed}: {window:?}"
                );
            }
        }
    }

    /// 分割切分：正放从起点向右切，倒放从终点向左切；两半必须划分父段。
    #[test]
    fn a_split_child_partitions_the_parent_window_from_both_sides() {
        let parent = win(0.25, 1.25);
        // 正放：左 [0.25, 0.75)，右 [0.75, 1.25)。
        assert_eq!(
            split_child_window(parent, 1.0, 0.0, 0.5, false),
            Some(win(0.25, 0.75))
        );
        assert_eq!(
            split_child_window(parent, 1.0, 0.5, 0.5, false),
            Some(win(0.75, 1.25))
        );
        // 倒放：左 [0.75, 1.25)，右 [0.25, 0.75) —— 从终点向左切。
        assert_eq!(
            split_child_window(parent, 1.0, 0.0, 0.5, true),
            Some(win(0.75, 1.25))
        );
        assert_eq!(
            split_child_window(parent, 1.0, 0.5, 0.5, true),
            Some(win(0.25, 0.75))
        );
    }

    /// 拉伸过的父段：切分按 |rate| 折算，位置与宽度同时缩放。
    #[test]
    fn a_split_child_scales_with_the_play_rate() {
        let parent = win(0.0, 2.0);
        assert_eq!(
            split_child_window(parent, 2.0, 0.25, 0.25, false),
            Some(win(0.5, 1.0))
        );
        assert_eq!(
            split_child_window(parent, 2.0, 0.25, 0.25, true),
            Some(win(1.0, 1.5))
        );
    }

    /// 类型层拒绝：起点晚于终点、或任一端非有限，都不是窗口。
    #[test]
    fn a_forward_window_rejects_inverted_or_nonfinite_input() {
        assert!(ForwardWindow::new(1.0, 0.0).is_none());
        assert!(ForwardWindow::new(f64::NAN, 1.0).is_none());
        assert!(ForwardWindow::new(0.0, f64::INFINITY).is_none());
        assert!(ForwardWindow::new(0.5, 0.5).is_some());
    }
}
