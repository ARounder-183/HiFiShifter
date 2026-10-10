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
    /// 【为什么读不出来时给 false 而不是 Option】与 `reversed` 不同：方向读不出来时
    /// 说"没倒放"是**错的结论**（会漏掉一个真实状态），而循环源读不出来时说"不循环"
    /// 是安全默认 —— 回绕是加法性的，关掉它只是少绕一圈，不会把内容指向别处。
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
