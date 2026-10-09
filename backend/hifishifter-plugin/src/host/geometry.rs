//! 真实take的只读宿主元数据；原生marker单位/坡度未验收，不作为kernel渲染坐标。

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
}

/// 此key来自唯一真实assignment；GUID不是用来搜索/猜测ARA对应关系的。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundHostGeometry {
    pub region_key: u64,
    pub geometry: HostClipGeometry,
}
