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
    pub playback_rate: f64,
    pub preserve_pitch: bool,
    pub channel_mode: i32,
    pub take_pitch: f64,
    pub item_timebase: i32,
    pub auto_stretch: bool,
    /// 宿主B_MUTE包含item solo覆盖；用于有效播放状态，不等同原始mute开关。
    pub muted: bool,
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
}

/// 此key来自唯一真实assignment；GUID不是用来搜索/猜测ARA对应关系的。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BoundHostGeometry {
    pub region_key: u64,
    pub geometry: HostClipGeometry,
}
