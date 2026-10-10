//! 真实take的只读宿主元数据；原生marker单位/坡度未验收，不作为kernel渲染坐标。

/// 两个宿主/ARA 浮点量是否相容。
///
/// 【为什么全局只有这一份】此前绑定用 `1e-7 + 8ε·max`（`render/extension.rs`），
/// 而分割规划、写回前置检查、剪贴板各写各的 `1e-6`。两者不对称会造出"规划接受、
/// 绑定拒绝"（或反过来）的窗口 —— 用户表现为"分割/拖动偶尔报 host clip changed"。
/// 现在绑定、规划、写回、剪贴板共用这一个判据。
///
/// 【为什么是相对容差而不是纯绝对】`Clip::playback_rate` 在 kernel 里是 f32，写回
/// 前置检查拿它与 REAPER 的 f64 `D_PLAYRATE` 比 —— f32 往返本身就有 ~1e-7 相对误差。
/// 纯绝对阈值会把这种正常舍入判成"宿主变了"。相对项吸收它，绝对项兜住接近 0 的量。
pub(crate) fn host_value_compatible(a: f64, b: f64) -> bool {
    a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-6 + 1e-9 * a.abs().max(b.abs())
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostStretchMarker {
    pub item_position_raw: f64,
    pub source_position_raw: f64,
    pub slope_raw: f64,
}

/// 普通fade仍由宿主负责；legacy shape/dir与7.81两个新参数分别忠实保留。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HostClipGeometry {
    pub item_id: String,
    pub take_id: String,
    pub start_sec: f64,
    pub source_start_sec: f64,
    pub duration_sec: f64,
    /// item秒域吸附偏移，仅用于原GUI几何回流，不参与源PCM和音频渲染。
    pub snap_offset_sec: f64,
    pub playback_rate: f64,
    pub preserve_pitch: bool,
    pub channel_mode: i32,
    pub take_pitch: f64,
    pub item_timebase: i32,
    pub auto_stretch: bool,
    /// 宿主B_MUTE包含item solo覆盖；用于有效播放状态，不等同原始mute开关。
    pub muted: bool,
    /// 宿主item音量仅用于GUI/原生写口；避免在ARA源PCM上再烘焙一次宿主增益。
    pub item_gain: f64,
    /// REAPER item I_GROUPID，0表示未编组；不等同HFS私有groupId字符串。
    pub group_id: i32,
    /// 原样保留take音量及负号极性，GUI改item音量不改此值。
    pub take_gain: f64,
    pub markers: Vec<HostStretchMarker>,
    pub fade_in_sec: f64,
    pub fade_out_sec: f64,
    pub fade_in_shape: f64,
    pub fade_out_shape: f64,
    pub fade_in_dir: f64,
    pub fade_out_dir: f64,
    pub fade_in_dir_new: f64,
    pub fade_out_dir_new: f64,
    pub fade_in_dir2_new: f64,
    pub fade_out_dir2_new: f64,
    /// 官方版本决定轴语义；版本无法读取时不把旧shape假报成7.81曲线。
    pub fade_axes_new: Option<bool>,
    pub auto_fade_in_sec: f64,
    pub auto_fade_out_sec: f64,
    /// 宿主 item 的 `B_LOOPSRC`（"循环源"）。
    ///
    /// 【为什么是 item 级】REAPER 的循环源是 **item** 属性（官方头文件里
    /// `B_LOOPSRC` 列在 `GetMediaItemInfo_Value` 的属性表中），与 RPP 的 `LOOP` 行同源 ——
    /// 不是 take 属性。内核的 `Clip.loop_enabled` 语义（对**整份媒体**取模回绕）与它一致，
    /// 而插件物化的 PCM 是完整源，所以这条读出来可以直接交给内核，无需改渲染路径。
    ///
    /// 【为什么读不出来时给 false 而不是 Option】与 `reversed` 同一纪律：读不到就是
    /// 安全默认（不循环 / 正放）。回绕是加法性的，关掉它只是少绕一圈，不会把内容指向
    /// 别处；方向读不到时按正放渲染，与 `loop_enabled` 一样是"不阻断渲染"的降级。
    pub loop_source: bool,
    /// take 源文件路径（`GetMediaSourceFileName`）；读不出来为 `None`。
    ///
    /// 【用途】媒体嫁接的回退身份：用户把 active take 换成**同一个文件**的另一个 take
    /// （复制 take、切换 active take）时，GUID 对不上但文件相同 —— 同一份已授权 PCM，
    /// 可以安全地把授权媒体挂上去，而不是让明明有音频的片段显示占位。
    pub source_file_name: Option<String>,
}

/// 此key来自唯一真实assignment；GUID不是用来搜索/猜测ARA对应关系的。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundHostGeometry {
    pub region_key: u64,
    pub geometry: HostClipGeometry,
}

/// ARA **不提供**、但会改变渲染结果的宿主事实快照（`ara::mapping::LOST_FIELDS`
/// 的渲染相关子集）。
///
/// 【为什么单独一层，而不是塞进 `HostClipGeometry`】几何是**逐 region 绑定**的、
/// 偏 UI 的；这一组是**逐 item** 的、纯渲染输入。混在一起会让"UI 几何变了"与
/// "渲染输入变了"两个语义互相污染 —— 而"宿主改了方向，插件不重渲染"这个报障，
/// 根因正是 `render_epoch` 只跟着**淡变长度**走，方向/循环源/声道模式没有写者。
///
/// 【为什么按 item GUID】与 `project_host_take_facts_locked` 同一身份口径：clip id
/// 形如 `ara-item-{guid}`，item 是宿主清单里的最小单位，且对尚未被 ARA 认领的 item
/// 也成立。
///
/// 【用途】任何一项变化都推进 `DocumentSession::render_epoch`，于是
/// `PreparedVersion`（渲染缓存键）与工作区投影指纹同时失效 —— 这是"宿主改了、
/// 插件立刻重渲染"的**唯一**通路，不再依赖某一处记得手动 bump。
#[derive(Clone, Default, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct RenderFacts {
    pub items: std::collections::BTreeMap<String, RenderItemFacts>,
}

/// 单个 item 的渲染相关宿主事实。
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub(crate) struct RenderItemFacts {
    /// 方向（`PCM_Source_GetSectionInfo` 的 revOut）；读不到即正放。
    pub reversed: bool,
    /// 循环源（`B_LOOPSRC`，item 级）。
    pub loop_enabled: bool,
    /// 声道模式（`I_CHANMODE`）。
    pub channel_mode: i32,
}

impl RenderFacts {
    /// 从宿主清单（`UiTrack` 集）采样。取 **active take** 的事实：扁平投影描述的就是
    /// active take，`normalize_takes()` 之后 clip 的字段来自它。
    ///
    /// 【为什么不用 `impl Iterator<Item = &UiTrack>`】`UiTrack` 是 `host::reaper` 的
    /// 私有子模块类型，这里不引入它 —— 调用点就地展开成 `(item_id, facts)` 迭代即可。
    pub(crate) fn from_items<I>(items: I) -> Self
    where
        I: IntoIterator<Item = (String, RenderItemFacts)>,
    {
        Self {
            items: items.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::host_value_compatible;

    #[test]
    fn identical_values_are_compatible() {
        assert!(host_value_compatible(0.0, 0.0));
        assert!(host_value_compatible(4.0, 4.0));
        assert!(host_value_compatible(-1.25, -1.25));
    }

    #[test]
    fn f32_round_trip_of_playback_rate_stays_compatible() {
        // `Clip::playback_rate` 是 f32，写回前置检查拿它与 REAPER 的 f64 `D_PLAYRATE`
        // 比 —— f32 往返本身就有 ~1e-7 相对误差。这一档**必须**判为相容，否则用户
        // 每改一次速率都会撞 "host clip changed before GUI commit"。
        for rate in [0.25_f32, 0.5, 1.0, 1.5, 2.0, 4.0] {
            let projected = rate as f64;
            let host = (rate as f64) * (1.0 + 1e-7);
            assert!(
                host_value_compatible(projected, host),
                "f32 round-trip of {rate} must stay compatible"
            );
        }
    }

    #[test]
    fn a_real_geometry_change_is_rejected() {
        // 1e-4 秒在 44.1k 上约 4 个采样点，是真实的移动，不是舍入。
        assert!(!host_value_compatible(0.0, 1e-4));
        assert!(!host_value_compatible(1.0, 1.001));
        assert!(!host_value_compatible(48000.0, 48000.5));
    }

    #[test]
    fn absolute_floor_covers_values_near_zero() {
        // 接近 0 的量没有相对项可用，由 1e-6 的绝对项兜底。
        assert!(host_value_compatible(0.0, 5e-7));
        assert!(!host_value_compatible(0.0, 2e-6));
    }

    #[test]
    fn non_finite_values_are_never_compatible() {
        assert!(!host_value_compatible(f64::NAN, f64::NAN));
        assert!(!host_value_compatible(f64::INFINITY, f64::INFINITY));
        assert!(!host_value_compatible(0.0, f64::NAN));
        assert!(!host_value_compatible(f64::INFINITY, 1.0));
    }
}
