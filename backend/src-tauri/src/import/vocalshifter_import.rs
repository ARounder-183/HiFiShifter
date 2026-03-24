// VocalShifter 工程文件 (.vshp / .vsp) 解析与转换模块
//
// 文件格式：二进制、小端序，由多个数据块组成。
// 支持的块类型：PRJP, TRKP, ITMP, Itmp, Ctrp, Time
//
// 参考规范：用户需求文档§2

use crate::audio_utils::try_read_wav_info;
use crate::models::PitchRange;
use crate::state::{Clip, PitchAnalysisAlgo, TimelineState, Track, TrackParamsState};
use std::collections::BTreeMap;
use std::path::Path;

// ─── 块标识 (8 bytes each) ───

const TAG_PRJP: [u8; 8] = [0x50, 0x52, 0x4A, 0x50, 0x00, 0x01, 0x00, 0x00];
const TAG_TRKP: [u8; 8] = [0x54, 0x52, 0x4B, 0x50, 0x00, 0x01, 0x00, 0x00];
const TAG_ITMP: [u8; 8] = [0x49, 0x54, 0x4D, 0x50, 0x00, 0x02, 0x00, 0x00];
const TAG_ITMP_EXT: [u8; 8] = [0x49, 0x74, 0x6D, 0x70, 0x00, 0x01, 0x00, 0x00]; // Itmp
const TAG_CTRP: [u8; 8] = [0x43, 0x74, 0x72, 0x70, 0x60, 0x00, 0x00, 0x00];
const TAG_TIME: [u8; 8] = [0x54, 0x69, 0x6D, 0x65, 0x10, 0x00, 0x00, 0x00];

const PRJP_DATA_SIZE: usize = 0x100;
const TRKP_DATA_SIZE: usize = 0x100;
const ITMP_DATA_SIZE: usize = 0x200;
const ITMP_EXT_DATA_SIZE: usize = 0x100;
const CTRP_DATA_SIZE: usize = 0x60;
const TIME_DATA_SIZE: usize = 0x10;

/// VocalShifter pitch 值 0 = C-1 (MIDI 0), 6000 = C4 (MIDI 60)
/// 换算公式: midi_note = vsp_pitch / 100.0
const VSP_PITCH_TO_MIDI: f64 = 1.0 / 100.0;

/// 每个 Ctrp 调音点固定间隔（秒）
const CTRP_FRAME_PERIOD: f64 = 0.005;

/// Time 标记分段平滑重叠上限（秒）
const SEGMENT_OVERLAP_MAX_SEC: f64 = 0.1;

/// 相邻分段过渡长度：取两段中较短者的 50%，并限制在上限内。
fn segment_overlap_sec(left_timeline_sec: f64, right_timeline_sec: f64) -> f64 {
    (left_timeline_sec.max(0.0) * 0.5)
        .min(right_timeline_sec.max(0.0) * 0.5)
        .min(SEGMENT_OVERLAP_MAX_SEC * 0.5)
}

/// HiFiShifter 支持的音频格式扩展名
const SUPPORTED_AUDIO_EXTS: &[&str] = &["wav", "flac", "mp3", "ogg", "m4a"];

const FILE_HEADER_SIZE: usize = 16;
const MAGIC: [u8; 4] = [0x56, 0x53, 0x50, 0x44]; // "VSPD"

// ─── 解析后的中间数据结构 ───

#[derive(Debug, Clone)]
struct VspProject {
    sample_rate: u32,
    time_sig_num: i32,
    time_sig_den: i32,
    bpm: f64,
}

#[derive(Debug, Clone)]
struct VspTrack {
    name: String,
    volume: f64,
    pan: f64,
    muted: bool,
    solo: bool,
    selected: bool,
    _inverted: bool,
}

#[derive(Debug, Clone)]
struct VspItemBase {
    audio_path: String,
    track_index: i32,
    selected: bool,
    start_sample: f64,
}

#[derive(Debug, Clone)]
struct VspItemExt {
    algo_type: i16,
    pitch_points: Vec<VspPitchPoint>,
    time_markers: Vec<VspTimeMarker>,
}

#[derive(Debug, Clone, Copy)]
struct VspPitchPoint {
    disabled: bool,
    original_pitch: i16, // *PIT (offset 20)
    pitch: i16,          // PIT (offset 22)
    formant: i16,        // FRM (offset 24)
    bre: i16,            // BRE (offset 26)
    eq1: i16,            // EQ1 (offset 28)
    eq2: i16,            // EQ2 (offset 30)
    dyn_orig: f64,       // *DYN (offset 32)
    dyn_edit: f64,       // DYN (offset 40)
    vol: f64,            // VOL (offset 48)
    pan: f64,            // PAN (offset 56)
    heq_or_mrp: i16,     // HEQ/MRP (offset 82)
}

#[derive(Debug, Clone, Copy)]
struct VspTimeMarker {
    original_pos: f64,
    new_pos: f64,
}

// ─── 导入结果 ───

pub struct VspImportResult {
    pub timeline: TimelineState,
    pub skipped_files: Vec<String>,
    pub beats_per_bar: u32,
}

#[derive(Default, Clone, Copy)]
struct PitchFrameAccumulator {
    sum: f64,
    weight: f64,
    disabled_weight: f64,
}

/// 累积器：用于收集 vslib 额外曲线帧数据（加权平均）。
#[derive(Default, Clone, Copy)]
struct ExtraCurveAccumulator {
    formant_shift_sum: f64,
    vol_sum: f64,
    dyn_orig_sum: f64,
    dyn_edit_sum: f64,
    pan_sum: f64,
    breathiness_sum: f64,
    weight: f64,
    /// 单独用于共振峰（formant）求平均的权重，只计入非 disabled 的点
    formant_weight: f64,
}

// ─── 系统编码检测 ───

/// 将系统本地编码的字节串解码为 UTF-8。
/// VocalShifter 使用系统 ANSI 编码（Windows 上常为 GBK 或 Shift-JIS）。
fn decode_local_string(bytes: &[u8]) -> String {
    // 找到 null 终止符
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let raw = &bytes[..end];

    // 使用系统编码
    let encoding = get_system_encoding();
    let (result, _, _) = encoding.decode(raw);
    result.to_string()
}

#[cfg(windows)]
fn get_system_encoding() -> &'static encoding_rs::Encoding {
    extern "system" {
        fn GetACP() -> u32;
    }
    let cp = unsafe { GetACP() };
    match cp {
        936 | 54936 => encoding_rs::GBK,
        932 => encoding_rs::SHIFT_JIS,
        950 => encoding_rs::BIG5,
        949 => encoding_rs::EUC_KR,
        1252 => encoding_rs::WINDOWS_1252,
        _ => encoding_rs::SHIFT_JIS, // VocalShifter 默认 Shift-JIS
    }
}

#[cfg(not(windows))]
fn get_system_encoding() -> &'static encoding_rs::Encoding {
    encoding_rs::SHIFT_JIS
}

// ─── 二进制读取辅助 ───

struct BinReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BinReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn peek_bytes(&self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n <= self.data.len() {
            Some(&self.data[self.pos..self.pos + n])
        } else {
            None
        }
    }

    fn skip(&mut self, n: usize) {
        self.pos = (self.pos + n).min(self.data.len());
    }

    fn read_bytes(&mut self, n: usize) -> Option<&'a [u8]> {
        if self.pos + n <= self.data.len() {
            let slice = &self.data[self.pos..self.pos + n];
            self.pos += n;
            Some(slice)
        } else {
            None
        }
    }

    fn read_i16_le(&mut self) -> Option<i16> {
        let b = self.read_bytes(2)?;
        Some(i16::from_le_bytes([b[0], b[1]]))
    }

    fn read_i32_le(&mut self) -> Option<i32> {
        let b = self.read_bytes(4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn read_f64_le(&mut self) -> Option<f64> {
        let b = self.read_bytes(8)?;
        Some(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }
}

// ─── 从数据块中的偏移位置读取值（不移动主游标） ───

fn read_i16_at(data: &[u8], offset: usize) -> Option<i16> {
    if offset + 2 > data.len() {
        return None;
    }
    Some(i16::from_le_bytes([data[offset], data[offset + 1]]))
}

fn read_i32_at(data: &[u8], offset: usize) -> Option<i32> {
    if offset + 4 > data.len() {
        return None;
    }
    Some(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

fn read_f64_at(data: &[u8], offset: usize) -> Option<f64> {
    if offset + 8 > data.len() {
        return None;
    }
    Some(f64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

// ─── 文件解析 ───

/// 解析 VocalShifter 工程文件 (.vshp / .vsp)。
/// 返回解析后的中间结构或错误信息。
fn parse_vsp_file(
    data: &[u8],
) -> Result<(VspProject, Vec<VspTrack>, Vec<VspItemBase>, Vec<VspItemExt>), String> {
    // §2.1 文件头校验
    if data.len() < FILE_HEADER_SIZE {
        return Err("File too small to be a valid VocalShifter project".into());
    }
    if data[0..4] != MAGIC {
        return Err("Invalid file header: expected VSPD magic bytes".into());
    }
    // 文件总大小（来自头部后 12 字节中的最后 4 字节）
    let _file_size = i32::from_le_bytes([data[12], data[13], data[14], data[15]]);

    let mut project: Option<VspProject> = None;
    let mut tracks: Vec<VspTrack> = Vec::new();
    let mut item_bases: Vec<VspItemBase> = Vec::new();
    let mut item_exts: Vec<VspItemExt> = Vec::new();

    let mut reader = BinReader::new(data);
    reader.skip(FILE_HEADER_SIZE);

    while reader.remaining() >= 8 {
        let tag = match reader.peek_bytes(8) {
            Some(t) => {
                let mut arr = [0u8; 8];
                arr.copy_from_slice(t);
                arr
            }
            None => break,
        };

        if tag == TAG_PRJP {
            reader.skip(8);
            if let Some(block_data) = reader.read_bytes(PRJP_DATA_SIZE) {
                project = Some(parse_prjp(block_data));
            }
        } else if tag == TAG_TRKP {
            reader.skip(8);
            if let Some(block_data) = reader.read_bytes(TRKP_DATA_SIZE) {
                tracks.push(parse_trkp(block_data));
            }
        } else if tag == TAG_ITMP {
            reader.skip(8);
            if let Some(block_data) = reader.read_bytes(ITMP_DATA_SIZE) {
                item_bases.push(parse_itmp(block_data));
            }
        } else if tag == TAG_ITMP_EXT {
            reader.skip(8);
            if let Some(block_data) = reader.read_bytes(ITMP_EXT_DATA_SIZE) {
                let mut ext = parse_itmp_ext(block_data);

                // 读取后续的 Ctrp 和 Time 块
                loop {
                    if reader.remaining() < 8 {
                        break;
                    }
                    let sub_tag = match reader.peek_bytes(8) {
                        Some(t) => {
                            let mut arr = [0u8; 8];
                            arr.copy_from_slice(t);
                            arr
                        }
                        None => break,
                    };

                    if sub_tag == TAG_CTRP {
                        reader.skip(8);
                        if let Some(ctrp_data) = reader.read_bytes(CTRP_DATA_SIZE) {
                            ext.pitch_points.push(parse_ctrp(ctrp_data));
                        }
                    } else if sub_tag == TAG_TIME {
                        reader.skip(8);
                        if let Some(time_data) = reader.read_bytes(TIME_DATA_SIZE) {
                            ext.time_markers.push(parse_time_marker(time_data));
                        }
                    } else {
                        break;
                    }
                }

                item_exts.push(ext);
            }
        } else {
            // 未知块：跳过 8 字节继续
            reader.skip(8);
        }
    }

    let project = project.ok_or("Missing PRJP block: no project information found")?;
    Ok((project, tracks, item_bases, item_exts))
}

fn parse_prjp(data: &[u8]) -> VspProject {
    VspProject {
        sample_rate: read_i32_at(data, 16).unwrap_or(44100) as u32,
        time_sig_num: read_i32_at(data, 20).unwrap_or(4),
        time_sig_den: read_i32_at(data, 24).unwrap_or(4),
        bpm: read_f64_at(data, 32).unwrap_or(120.0),
    }
}

fn parse_trkp(data: &[u8]) -> VspTrack {
    let name = decode_local_string(&data[0..64.min(data.len())]);
    VspTrack {
        name,
        volume: read_f64_at(data, 64).unwrap_or(1.0),
        pan: read_f64_at(data, 72).unwrap_or(0.0),
        muted: read_i32_at(data, 80).unwrap_or(0) != 0,
        solo: read_i32_at(data, 84).unwrap_or(0) != 0,
        selected: read_i32_at(data, 0x58).unwrap_or(0) == 1,
        _inverted: read_i32_at(data, 96).unwrap_or(0) != 0,
    }
}

fn parse_itmp(data: &[u8]) -> VspItemBase {
    // 偏移 0: 变长字符串到 null 终止
    let path_end = data
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(0x108.min(data.len()));
    let audio_path = decode_local_string(&data[0..path_end]);

    VspItemBase {
        audio_path,
        track_index: read_i32_at(data, 0x108).unwrap_or(0),
        selected: read_i32_at(data, 0x104).unwrap_or(0) == 1,
        start_sample: read_f64_at(data, 0x110).unwrap_or(0.0),
    }
}

fn parse_itmp_ext(data: &[u8]) -> VspItemExt {
    VspItemExt {
        algo_type: read_i16_at(data, 0x30).unwrap_or(0),
        pitch_points: Vec::new(),
        time_markers: Vec::new(),
    }
}

fn parse_ctrp(data: &[u8]) -> VspPitchPoint {
    VspPitchPoint {
        disabled: read_i16_at(data, 18).unwrap_or(0) != 0,
        original_pitch: read_i16_at(data, 20).unwrap_or(0),
        pitch: read_i16_at(data, 22).unwrap_or(0),
        formant: read_i16_at(data, 24).unwrap_or(0),
        bre: read_i16_at(data, 26).unwrap_or(0),
        eq1: read_i16_at(data, 28).unwrap_or(0),
        eq2: read_i16_at(data, 30).unwrap_or(0),
        dyn_orig: read_f64_at(data, 32).unwrap_or(1.0),
        dyn_edit: read_f64_at(data, 40).unwrap_or(1.0),
        vol: read_f64_at(data, 48).unwrap_or(1.0),
        pan: read_f64_at(data, 56).unwrap_or(0.0),
        heq_or_mrp: read_i16_at(data, 82).unwrap_or(0),
    }
}

fn parse_time_marker(data: &[u8]) -> VspTimeMarker {
    VspTimeMarker {
        original_pos: read_f64_at(data, 0).unwrap_or(0.0),
        new_pos: read_f64_at(data, 8).unwrap_or(0.0),
    }
}

// ─── 算法映射 ───

/// 将 VocalShifter algo_type 映射为 (is_world, synth_mode)。
///
/// - M=0 → vslib SYNTHMODE_M (单音)
/// - V=1 → vslib SYNTHMODE_MF (单音+共振峰补正)
/// - P=2 → vslib SYNTHMODE_P (和音)
/// - R=4 → vslib SYNTHMODE_M (打击乐→暂无对应，回退到单音)
/// - World=8 → WorldDll
/// - 其他 → V 算法 → vslib SYNTHMODE_MF
fn algo_type_to_hs(algo_type: i16) -> (bool, i32) {
    match algo_type {
        0 => (false, 0), // M → SYNTHMODE_M
        1 => (false, 1), // V → SYNTHMODE_MF
        2 => (false, 2), // P → SYNTHMODE_P
        4 => (false, 0), // R → SYNTHMODE_M（暂无对应）
        8 => (true, 0),  // World
        _ => (false, 1), // 默认 V → SYNTHMODE_MF
    }
}

/// 判断 algo_type 是否为 World 算法。
fn is_world_algo(algo_type: i16) -> bool {
    algo_type_to_hs(algo_type).0
}

/// 获取 algo_type 对应的 vslib synth_mode。
fn algo_synth_mode(algo_type: i16) -> i32 {
    algo_type_to_hs(algo_type).1
}

/// 根据 (is_world, synth_mode) 对返回轨道名称后缀。
fn algo_pair_suffix(is_world: bool, synth_mode: i32) -> &'static str {
    if is_world {
        " (World)"
    } else {
        match synth_mode {
            0 => " (VsLib-M)",
            2 => " (VsLib-P)",
            _ => " (VsLib-V)",
        }
    }
}

// ─── 转换为 HiFiShifter 工程 ───

fn new_track_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

fn new_clip_id() -> String {
    format!("clip_{}", uuid::Uuid::new_v4())
}

/// 判断音频文件扩展名是否被 HiFiShifter 支持。
fn is_audio_supported(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            SUPPORTED_AUDIO_EXTS
                .iter()
                .any(|&ext| ext.eq_ignore_ascii_case(e))
        })
        .unwrap_or(false)
}

/// 将 VocalShifter 音量倍率（1.0 = 0 dB）转换为 HiFiShifter 的 0.0–1.0 音量范围。
/// HiFiShifter 默认音量为 0.9，VocalShifter 1.0 对应全音量。
fn convert_volume(vs_volume: f64) -> f32 {
    (vs_volume as f32).clamp(0.0, 1.0)
}

/// 轨道颜色调色板（与 state.rs 中一致）
const TRACK_COLORS: &[&str] = &[
    "#4f8ef7", "#a78bfa", "#34d399", "#fb923c", "#f472b6", "#38bdf8", "#facc15", "#f87171",
];

fn clip_color() -> String {
    "#4fc3f7".to_string()
}

/// 将解析后的 VocalShifter 数据转换为 HiFiShifter TimelineState。
pub fn import_vsp(data: &[u8], vsp_file_dir: &Path) -> Result<VspImportResult, String> {
    let (project, vsp_tracks, item_bases, item_exts) = parse_vsp_file(data)?;

    let sample_rate = project.sample_rate.max(1) as f64;
    let bpm = if project.bpm > 0.0 {
        project.bpm
    } else {
        120.0
    };

    let mut skipped_files: Vec<String> = Vec::new();

    // ─── 第一步：创建轨道映射 ───
    // 检测每个原始轨道内是否存在混合算法，需要拆分
    // key: (original_track_index, is_world, synth_mode) → new_track_id
    let mut track_algo_map: std::collections::HashMap<(i32, bool, i32), String> =
        std::collections::HashMap::new();
    let mut hs_tracks: Vec<Track> = Vec::new();
    let mut track_order: i32 = 0;

    // 统计每个原始轨道内使用的算法（按 (is_world, synth_mode) 对区分）
    let mut track_algos: std::collections::HashMap<i32, std::collections::HashSet<(bool, i32)>> =
        std::collections::HashMap::new();
    for (i, base) in item_bases.iter().enumerate() {
        let pair = item_exts
            .get(i)
            .map(|e| algo_type_to_hs(e.algo_type))
            .unwrap_or((false, 1));
        track_algos
            .entry(base.track_index)
            .or_default()
            .insert(pair);
    }

    // 统计每个 (原始轨道索引, algo_pair) 的最早出现时间（用于对拆分后的同源轨道进行排序）
    let mut earliest_start_by_algo: std::collections::HashMap<(i32, bool, i32), f64> =
        std::collections::HashMap::new();
    for (i, base) in item_bases.iter().enumerate() {
        let pair = item_exts
            .get(i)
            .map(|e| algo_type_to_hs(e.algo_type))
            .unwrap_or((false, 1));
        let start_sec = base.start_sample / sample_rate;
        let key = (base.track_index, pair.0, pair.1);
        let entry = earliest_start_by_algo.entry(key).or_insert(f64::INFINITY);
        if start_sec.is_finite() && start_sec < *entry {
            *entry = start_sec;
        }
    }

    // 为每个 VspTrack 创建 HiFiShifter 轨道
    for (vsp_idx, vsp_track) in vsp_tracks.iter().enumerate() {
        let idx = vsp_idx as i32;
        let algos = track_algos.get(&idx);
        let has_mixed = algos.map(|s| s.len() > 1).unwrap_or(false);

        if has_mixed {
            // 需要拆分：为每个不同的 (is_world, synth_mode) 各建一条轨道
            // 按该算法在原轨道中音频块首次出现的时间排序
            let mut sorted: Vec<(bool, i32)> = algos
                .map(|s| s.iter().copied().collect())
                .unwrap_or_default();
            sorted.sort_by(|a, b| {
                let ka = (idx, a.0, a.1);
                let kb = (idx, b.0, b.1);
                let ta = earliest_start_by_algo
                    .get(&ka)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let tb = earliest_start_by_algo
                    .get(&kb)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (is_world, synth_mode) in sorted {
                let id = new_track_id();
                let algo = if is_world {
                    PitchAnalysisAlgo::WorldDll
                } else {
                    PitchAnalysisAlgo::VocalShifterVslib
                };
                hs_tracks.push(Track {
                    id: id.clone(),
                    name: vsp_track.name.clone(),
                    parent_id: None,
                    order: track_order,
                    muted: vsp_track.muted,
                    solo: vsp_track.solo,
                    volume: convert_volume(vsp_track.volume),
                    compose_enabled: false,
                    pitch_analysis_algo: algo,
                    color: TRACK_COLORS[hs_tracks.len() % TRACK_COLORS.len()].to_string(),
                });
                track_algo_map.insert((idx, is_world, synth_mode), id);
                track_order += 1;
            }
        } else {
            // 单一算法或无音频项
            let (is_world, synth_mode) = algos
                .and_then(|s| s.iter().next().copied())
                .unwrap_or((false, 1));
            let algo = if is_world {
                PitchAnalysisAlgo::WorldDll
            } else {
                PitchAnalysisAlgo::VocalShifterVslib
            };
            let id = new_track_id();
            hs_tracks.push(Track {
                id: id.clone(),
                name: vsp_track.name.clone(),
                parent_id: None,
                order: track_order,
                muted: vsp_track.muted,
                solo: vsp_track.solo,
                volume: convert_volume(vsp_track.volume),
                compose_enabled: false,
                pitch_analysis_algo: algo,
                color: TRACK_COLORS[hs_tracks.len() % TRACK_COLORS.len()].to_string(),
            });
            // 单一算法，映射精确的 (is_world, synth_mode) 对
            track_algo_map.insert((idx, is_world, synth_mode), id);
            track_order += 1;
        }
    }

    // 如果某些 item 引用了超出 vsp_tracks 范围的轨道索引，为其创建轨道
    for base in &item_bases {
        if (base.track_index as usize) >= vsp_tracks.len() {
            let idx = base.track_index;
            if !track_algo_map.keys().any(|&(i, _, _)| i == idx) {
                let id = new_track_id();
                hs_tracks.push(Track {
                    id: id.clone(),
                    name: format!("Track {}", idx + 1),
                    parent_id: None,
                    order: track_order,
                    muted: false,
                    solo: false,
                    volume: 0.9,
                    compose_enabled: false,
                    pitch_analysis_algo: PitchAnalysisAlgo::default(),
                    color: TRACK_COLORS[hs_tracks.len() % TRACK_COLORS.len()].to_string(),
                });
                track_algo_map.insert((idx, false, 1), id);
                track_order += 1;
            }
        }
    }

    // ─── 第二步：创建剪辑 ───
    let mut hs_clips: Vec<Clip> = Vec::new();
    // 用于收集每个轨道的 pitch 数据：track_id → frame_idx → 累积加权值
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();
    let mut extra_curve_data_by_track: std::collections::HashMap<
        String,
        Vec<ExtraCurveAccumulator>,
    > = std::collections::HashMap::new();
    // 记录每个轨道的 synth_mode（仅 vslib 轨道有效）
    let mut synth_mode_by_track: std::collections::HashMap<String, i32> =
        std::collections::HashMap::new();

    for (i, base) in item_bases.iter().enumerate() {
        let ext = item_exts.get(i);

        // 解析音频路径
        let audio_path = resolve_audio_path(&base.audio_path, vsp_file_dir);

        // 检查格式支持
        if !is_audio_supported(&audio_path) {
            skipped_files.push(base.audio_path.clone());
            continue;
        }

        // 检查文件是否存在
        if !Path::new(&audio_path).exists() {
            skipped_files.push(base.audio_path.clone());
            continue;
        }

        // 确定目标轨道
        let (is_world, synth_mode) = ext
            .map(|e| algo_type_to_hs(e.algo_type))
            .unwrap_or((false, 1));
        let track_id = track_algo_map
            .get(&(base.track_index, is_world, synth_mode))
            .or_else(|| {
                // 回退：查找同一原始轨道索引下的任意映射
                track_algo_map
                    .iter()
                    .find(|(&(i, _, _), _)| i == base.track_index)
                    .map(|(_, id)| id)
            })
            .cloned()
            .unwrap_or_else(|| hs_tracks.first().map(|t| t.id.clone()).unwrap_or_default());

        // 记录 vslib 轨道的 synth_mode
        // 记录该轨道对应的 synth_mode（不再因为 World 而忽略）
        if let Some(e) = ext {
            synth_mode_by_track
                .entry(track_id.clone())
                .or_insert_with(|| algo_synth_mode(e.algo_type));
        }

        let item_start_sec = base.start_sample / sample_rate;

        // 读取音频文件信息
        let audio_info = try_read_wav_info(Path::new(&audio_path), 4096);
        let (duration_sec, duration_frames, source_sr, waveform_preview) = match &audio_info {
            Some(info) => (
                Some(info.duration_sec),
                Some(info.total_frames),
                Some(info.sample_rate),
                Some(info.waveform_preview.clone()),
            ),
            None => (None, None, None, None),
        };

        let source_duration_sec = duration_sec.unwrap_or(0.0);

        // 处理时间拉伸标记
        let time_markers = ext.map(|e| &e.time_markers[..]).unwrap_or(&[]);
        let pitch_points = ext.map(|e| &e.pitch_points[..]).unwrap_or(&[]);

        if time_markers.len() >= 3 {
            // 非线性拉伸：拆分为多个子剪辑
            let seg_count = time_markers.len() - 1;
            let mut segment_clip_indices: Vec<usize> = Vec::with_capacity(seg_count);
            let mut segment_actual_pre_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let mut segment_actual_post_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let seg_timeline_durations: Vec<f64> = (0..seg_count)
                .map(|i| {
                    let m_start = &time_markers[i];
                    let m_end = &time_markers[i + 1];
                    ((m_end.new_pos - m_start.new_pos) / sample_rate).max(0.001)
                })
                .collect();
            for seg_idx in 0..seg_count {
                let m_start = &time_markers[seg_idx];
                let m_end = &time_markers[seg_idx + 1];

                let src_start = m_start.original_pos / sample_rate;
                let src_end = m_end.original_pos / sample_rate;
                let src_dur = (src_end - src_start).max(0.001);

                let new_start = m_start.new_pos / sample_rate;
                let _new_end = m_end.new_pos / sample_rate;
                let new_dur = seg_timeline_durations[seg_idx];

                let rate = (src_dur / new_dur) as f32;

                let want_pre_tl = if seg_idx > 0 {
                    segment_overlap_sec(seg_timeline_durations[seg_idx - 1], new_dur)
                } else {
                    0.0
                };
                let want_post_tl = if seg_idx + 1 < seg_count {
                    segment_overlap_sec(new_dur, seg_timeline_durations[seg_idx + 1])
                } else {
                    0.0
                };
                let rate64 = (rate as f64).max(0.0001);
                let want_pre_src = want_pre_tl * rate64;
                let want_post_src = want_post_tl * rate64;

                let seg_src_start = (src_start - want_pre_src).max(0.0);
                let seg_src_end = (src_end + want_post_src).min(source_duration_sec.max(src_end));
                let actual_pre_tl = (src_start - seg_src_start) / rate64;
                let actual_post_tl = (seg_src_end - src_end).max(0.0) / rate64;

                let clip_start = item_start_sec + new_start - actual_pre_tl;
                let clip_length = (new_dur + actual_pre_tl + actual_post_tl).max(0.001);
                let clip_id = new_clip_id();
                let clip_name = Path::new(&audio_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Audio")
                    .to_string();
                let clip_index = hs_clips.len();

                hs_clips.push(Clip {
                    id: clip_id.clone(),
                    track_id: track_id.clone(),
                    name: format!("{} ({})", clip_name, seg_idx + 1),
                    start_sec: clip_start,
                    length_sec: clip_length,
                    color: clip_color(),
                    source_path: Some(audio_path.clone()),
                    duration_sec,
                    duration_frames,
                    source_sample_rate: source_sr,
                    waveform_preview: waveform_preview.clone(),
                    pitch_range: Some(PitchRange {
                        min: -24.0,
                        max: 24.0,
                    }),
                    gain: 1.0,
                    muted: false,
                    source_start_sec: seg_src_start,
                    source_end_sec: seg_src_end,
                    playback_rate: rate.clamp(0.1, 10.0),
                    fade_in_sec: 0.0,
                    fade_out_sec: 0.0,
                    fade_in_curve: String::new(),
                    fade_out_curve: String::new(),
                    extra_curves: None,
                    extra_params: None,
                });
                segment_clip_indices.push(clip_index);
                segment_actual_pre_tl.push(actual_pre_tl);
                segment_actual_post_tl.push(actual_post_tl);

                // 写入 pitch 数据（源时间范围内的 Ctrp 点）
                write_pitch_data_for_segment(
                    &track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut pitch_data_by_track,
                );

                // 写入额外曲线数据（不再因为 World 而跳过）
                write_extra_curves_for_segment(
                    &track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut extra_curve_data_by_track,
                );
            }

            for seg_idx in 0..seg_count {
                let clip_idx = segment_clip_indices[seg_idx];
                let Some(clip) = hs_clips.get_mut(clip_idx) else {
                    continue;
                };
                let fade_in = if seg_idx > 0 {
                    (segment_actual_pre_tl[seg_idx] + segment_actual_post_tl[seg_idx - 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                let fade_out = if seg_idx + 1 < seg_count {
                    (segment_actual_post_tl[seg_idx] + segment_actual_pre_tl[seg_idx + 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                clip.fade_in_sec = fade_in;
                clip.fade_out_sec = fade_out;
            }
        } else {
            // 线性拉伸或无拉伸
            let (rate, clip_length) = if time_markers.len() == 2 {
                let m0 = &time_markers[0];
                let m1 = &time_markers[1];
                let src_dur = ((m1.original_pos - m0.original_pos) / sample_rate).max(0.001);
                let new_dur = ((m1.new_pos - m0.new_pos) / sample_rate).max(0.001);
                let r = src_dur / new_dur;
                (r as f32, new_dur)
            } else {
                (1.0f32, source_duration_sec)
            };

            let clip_id = new_clip_id();
            let clip_name = Path::new(&audio_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Audio")
                .to_string();

            hs_clips.push(Clip {
                id: clip_id.clone(),
                track_id: track_id.clone(),
                name: clip_name,
                start_sec: item_start_sec,
                length_sec: clip_length,
                color: clip_color(),
                source_path: Some(audio_path.clone()),
                duration_sec,
                duration_frames,
                source_sample_rate: source_sr,
                waveform_preview,
                pitch_range: Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                }),
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: source_duration_sec,
                playback_rate: rate.clamp(0.1, 10.0),
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                extra_curves: None,
                extra_params: None,
            });

            // 写入 pitch 数据
            write_pitch_data_for_segment(
                &track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut pitch_data_by_track,
            );

            // 写入额外曲线数据（不再因为 World 而跳过）
            write_extra_curves_for_segment(
                &track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut extra_curve_data_by_track,
            );
        }
    }

    // ─── 第三步：计算工程时长 ───
    let project_end = hs_clips
        .iter()
        .map(|c| c.start_sec + c.length_sec)
        .fold(32.0_f64, f64::max);

    // ─── 第四步：构建 pitch 参数 ───
    let mut params_by_root_track: BTreeMap<String, TrackParamsState> = BTreeMap::new();
    let frame_period_ms = CTRP_FRAME_PERIOD * 1000.0; // 5.0ms

    for track in &hs_tracks {
        if let Some(points) = pitch_data_by_track.get(&track.id) {
            if points.is_empty() {
                continue;
            }
            let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);
            let mut pitch_edit = vec![0.0f32; total_frames];

            for (frame_idx, acc) in points.iter().enumerate() {
                if frame_idx < total_frames {
                    if acc.weight > 0.0 {
                        pitch_edit[frame_idx] = (acc.sum / acc.weight) as f32;
                    } else if acc.disabled_weight > 0.0 {
                        pitch_edit[frame_idx] = 0.0;
                    }
                }
            }

            // 构建 vslib 额外曲线和参数
            let extra_curves = extra_curve_data_by_track
                .get(&track.id)
                .map(|ecm| build_extra_curves_from_accumulators(ecm, total_frames))
                .unwrap_or_default();

            let mut extra_params: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            if let Some(&sm) = synth_mode_by_track.get(&track.id) {
                extra_params.insert("synth_mode".to_string(), sm as f64);
            }

            params_by_root_track.insert(
                track.id.clone(),
                TrackParamsState {
                    frame_period_ms,
                    pitch_orig: pitch_edit.clone(),
                    pitch_edit,
                    pitch_edit_user_modified: true,
                    tension_orig: Vec::new(),
                    tension_edit: Vec::new(),
                    pitch_orig_key: None,
                    pending_pitch_offset: None,
                    extra_curves,
                    extra_params,
                },
            );
        }
    }

    // ─── 第四步(b)：为有 pitch 数据的轨道开启合成 ───
    for track in &mut hs_tracks {
        if params_by_root_track.contains_key(&track.id) {
            track.compose_enabled = true;
        }
    }

    // ─── 第五步：组装 TimelineState ───
    let timeline = TimelineState {
        tracks: hs_tracks,
        clips: hs_clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm,
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        next_track_order: track_order,
    };

    Ok(VspImportResult {
        timeline,
        skipped_files,
        beats_per_bar: project.time_sig_num.clamp(1, 32) as u32,
    })
}

/// 将 Ctrp 调音点写入指定轨道的 pitch 数据。
fn write_pitch_data_for_segment(
    track_id: &str,
    pitch_points: &[VspPitchPoint],
    src_start_sec: f64,
    src_end_sec: f64,
    clip_start_sec: f64,
    playback_rate: f64,
    fade_in_sec: f64,
    fade_out_sec: f64,
    pitch_data: &mut std::collections::HashMap<String, Vec<PitchFrameAccumulator>>,
) {
    if pitch_points.is_empty() {
        return;
    }

    let rate = playback_rate.max(0.0001);
    let clip_end_sec = clip_start_sec + (src_end_sec - src_start_sec).max(0.0) / rate;
    if clip_end_sec <= clip_start_sec {
        return;
    }

    let start_frame = (clip_start_sec / CTRP_FRAME_PERIOD).floor().max(0.0) as usize;
    let end_frame = (clip_end_sec / CTRP_FRAME_PERIOD).ceil().max(0.0) as usize;

    let entry = pitch_data.entry(track_id.to_string()).or_default();

    // 按目标时间线逐帧采样，避免拉伸后 round 投影造成的“漏帧锯齿”。
    for frame_idx in start_frame..=end_frame {
        let timeline_time = frame_idx as f64 * CTRP_FRAME_PERIOD;
        if timeline_time < clip_start_sec || timeline_time > clip_end_sec {
            continue;
        }

        let rel_t = timeline_time - clip_start_sec;
        let src_time = src_start_sec + rel_t * rate;
        if src_time < src_start_sec || src_time > src_end_sec {
            continue;
        }

        let src_idx = (src_time / CTRP_FRAME_PERIOD).round().max(0.0) as usize;
        let Some(point) = pitch_points.get(src_idx) else {
            continue;
        };

        let mut weight = 1.0;
        if fade_in_sec > 0.0 {
            let fi_end = clip_start_sec + fade_in_sec;
            if timeline_time <= fi_end {
                let k = ((timeline_time - clip_start_sec) / fade_in_sec).clamp(0.0, 1.0);
                weight *= k;
            }
        }
        if fade_out_sec > 0.0 {
            let fo_start = (clip_end_sec - fade_out_sec).max(clip_start_sec);
            if timeline_time >= fo_start {
                let k = ((clip_end_sec - timeline_time) / fade_out_sec).clamp(0.0, 1.0);
                weight *= k;
            }
        }
        if weight <= 0.0 {
            continue;
        }

        if frame_idx >= entry.len() {
            entry.resize(frame_idx + 1, PitchFrameAccumulator::default());
        }
        let acc = &mut entry[frame_idx];

        if point.disabled {
            acc.disabled_weight += weight;
            continue;
        }

        let midi_val = point.pitch as f64 * VSP_PITCH_TO_MIDI;
        if midi_val > 0.0 {
            acc.sum += midi_val * weight;
            acc.weight += weight;
        }
    }
}

/// 将 Ctrp 调音点的额外曲线数据（FRM, VOL, DYN, *DYN, PAN, BRE）写入累积器。
/// 仅用于 vslib 算法轨道。
fn write_extra_curves_for_segment(
    track_id: &str,
    pitch_points: &[VspPitchPoint],
    src_start_sec: f64,
    src_end_sec: f64,
    clip_start_sec: f64,
    playback_rate: f64,
    fade_in_sec: f64,
    fade_out_sec: f64,
    curve_data: &mut std::collections::HashMap<String, Vec<ExtraCurveAccumulator>>,
) {
    if pitch_points.is_empty() {
        return;
    }

    let rate = playback_rate.max(0.0001);
    let clip_end_sec = clip_start_sec + (src_end_sec - src_start_sec).max(0.0) / rate;
    if clip_end_sec <= clip_start_sec {
        return;
    }

    let start_frame = (clip_start_sec / CTRP_FRAME_PERIOD).floor().max(0.0) as usize;
    let end_frame = (clip_end_sec / CTRP_FRAME_PERIOD).ceil().max(0.0) as usize;

    let entry = curve_data.entry(track_id.to_string()).or_default();

    for frame_idx in start_frame..=end_frame {
        let timeline_time = frame_idx as f64 * CTRP_FRAME_PERIOD;
        if timeline_time < clip_start_sec || timeline_time > clip_end_sec {
            continue;
        }

        let rel_t = timeline_time - clip_start_sec;
        let src_time = src_start_sec + rel_t * rate;
        if src_time < src_start_sec || src_time > src_end_sec {
            continue;
        }

        let src_idx = (src_time / CTRP_FRAME_PERIOD).round().max(0.0) as usize;
        let Some(point) = pitch_points.get(src_idx) else {
            continue;
        };

        let mut weight = 1.0;
        if fade_in_sec > 0.0 {
            let fi_end = clip_start_sec + fade_in_sec;
            if timeline_time <= fi_end {
                let k = ((timeline_time - clip_start_sec) / fade_in_sec).clamp(0.0, 1.0);
                weight *= k;
            }
        }
        if fade_out_sec > 0.0 {
            let fo_start = (clip_end_sec - fade_out_sec).max(clip_start_sec);
            if timeline_time >= fo_start {
                let k = ((clip_end_sec - timeline_time) / fade_out_sec).clamp(0.0, 1.0);
                weight *= k;
            }
        }
        if weight <= 0.0 {
            continue;
        }
        // 共振峰偏移：仅当点未被 disabled 时贡献到 formant 统计；其他参数始终贡献
        let formant_shift = point.formant as f64;

        if frame_idx >= entry.len() {
            entry.resize(frame_idx + 1, ExtraCurveAccumulator::default());
        }
        let acc = &mut entry[frame_idx];

        // 如果点没有被 disabled，则将其计入 formant 的权重和和
        if !point.disabled {
            acc.formant_shift_sum += formant_shift * weight;
            acc.formant_weight += weight;
        }
        // 其他参数（volume, dyn, pan, breathiness）无论 disabled 与否都应贡献
        acc.vol_sum += point.vol * weight;
        acc.dyn_orig_sum += point.dyn_orig * weight;
        acc.dyn_edit_sum += point.dyn_edit * weight;
        acc.pan_sum += point.pan * weight;
        acc.breathiness_sum += point.bre as f64 * weight;
        acc.weight += weight;
    }
}

/// 从 ExtraCurveAccumulator 映射构建 extra_curves HashMap。
fn build_extra_curves_from_accumulators(
    acc_map: &[ExtraCurveAccumulator],
    total_frames: usize,
) -> std::collections::HashMap<String, Vec<f32>> {
    let mut formant_shift = vec![0.0f32; total_frames];
    let mut volume = vec![1.0f32; total_frames];
    let mut dyn_orig = vec![1.0f32; total_frames];
    let mut dyn_edit = vec![1.0f32; total_frames];
    let mut pan = vec![0.0f32; total_frames];
    let mut breathiness = vec![0.0f32; total_frames];

    for (frame_idx, acc) in acc_map.iter().enumerate() {
        if frame_idx < total_frames && acc.weight > 0.0 {
            let w = acc.weight;
            // 对于 formant，只使用非 disabled 点的权重进行平均
            if acc.formant_weight > 0.0 {
                formant_shift[frame_idx] = (acc.formant_shift_sum / acc.formant_weight) as f32;
            } else {
                formant_shift[frame_idx] = 0.0;
            }

            // 计算 DYN 合并：avg_dyn_edit / avg_dyn_orig，为 0 除法的情况回退为 1
            let avg_vol = (acc.vol_sum / w) as f64;
            let avg_dyn_orig = (acc.dyn_orig_sum / w) as f64;
            let avg_dyn_edit = (acc.dyn_edit_sum / w) as f64;
            let multiplier = if avg_dyn_orig.abs() < 1e-12 {
                1.0f64
            } else {
                avg_dyn_edit / avg_dyn_orig
            };
            let merged_vol = (avg_vol * multiplier) as f32;

            volume[frame_idx] = merged_vol;
            dyn_orig[frame_idx] = (avg_dyn_orig) as f32;
            dyn_edit[frame_idx] = (avg_dyn_edit) as f32;
            pan[frame_idx] = (acc.pan_sum / w) as f32;
            breathiness[frame_idx] = (acc.breathiness_sum / w) as f32;
        }
    }

    let mut curves = std::collections::HashMap::new();
    curves.insert("formant_shift_cents".to_string(), formant_shift);
    curves.insert("volume".to_string(), volume);
    curves.insert("dyn_orig".to_string(), dyn_orig);
    curves.insert("dyn_edit".to_string(), dyn_edit);
    curves.insert("pan".to_string(), pan);
    curves.insert("breathiness".to_string(), breathiness);
    curves
}

/// 从 VocalShifter 剪贴板工程文件 (.clb.vshp / .clb.vsp) 导入选中的 Item。
///
/// - 只导入被标记为 `selected` 的 Item
/// - 所有 Item 中最早的起始位置对齐到 `playhead_sec`
/// - 轨道映射参考 Reaper 剪贴板逻辑：从 `selected_track_idx` 开始，按原轨道偏移分配
pub fn import_vsp_clipboard(
    data: &[u8],
    vsp_file_dir: &Path,
    playhead_sec: f64,
    selected_track_idx: usize,
    ordered_track_ids: &[String],
) -> Result<VspImportResult, String> {
    let (project, vsp_tracks, item_bases, item_exts) = parse_vsp_file(data)?;

    let sample_rate = project.sample_rate.max(1) as f64;

    // 筛选被选中的 Item 索引
    let selected_indices: Vec<usize> = item_bases
        .iter()
        .enumerate()
        .filter(|(_, base)| base.selected)
        .map(|(i, _)| i)
        .collect();

    // 若没有选中的 Item，则回退到导入选中的轨道
    if selected_indices.is_empty() {
        return import_vsp_clipboard_selected_tracks(
            &project,
            &vsp_tracks,
            &item_bases,
            &item_exts,
            vsp_file_dir,
            ordered_track_ids,
        );
    }

    // 计算选中 Item 中最小的起始时间，用于对齐到 playhead
    let min_start_sec = selected_indices
        .iter()
        .map(|&i| item_bases[i].start_sample / sample_rate)
        .fold(f64::MAX, f64::min);
    let time_offset = if min_start_sec.is_finite() {
        playhead_sec - min_start_sec
    } else {
        0.0
    };

    // 收集选中 Item 使用的原始轨道索引（去重、排序）
    let mut unique_track_indices: Vec<i32> = selected_indices
        .iter()
        .map(|&i| item_bases[i].track_index)
        .collect();
    unique_track_indices.sort();
    unique_track_indices.dedup();

    // 原始轨道索引 → 目标轨道偏移（从 0 开始递增）
    let track_idx_to_offset: std::collections::HashMap<i32, usize> = unique_track_indices
        .iter()
        .enumerate()
        .map(|(offset, &idx)| (idx, offset))
        .collect();

    let mut skipped_files: Vec<String> = Vec::new();
    let mut hs_clips: Vec<Clip> = Vec::new();
    let mut new_tracks: Vec<Track> = Vec::new();
    let mut created_track_ids: std::collections::HashMap<usize, String> =
        std::collections::HashMap::new();
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();
    let mut extra_curve_data_by_track: std::collections::HashMap<
        String,
        Vec<ExtraCurveAccumulator>,
    > = std::collections::HashMap::new();
    let mut synth_mode_by_track: std::collections::HashMap<String, i32> =
        std::collections::HashMap::new();

    let mut next_order = ordered_track_ids.len() as i32;

    for &item_idx in &selected_indices {
        let base = &item_bases[item_idx];
        let ext = item_exts.get(item_idx);

        // 确定目标轨道
        let track_offset = track_idx_to_offset
            .get(&base.track_index)
            .copied()
            .unwrap_or(0);
        let target_track_idx = selected_track_idx + track_offset;
        let target_track_id = if target_track_idx < ordered_track_ids.len() {
            ordered_track_ids[target_track_idx].clone()
        } else if let Some(id) = created_track_ids.get(&target_track_idx) {
            id.clone()
        } else {
            // 需要创建新轨道
            let tid = new_track_id();
            let is_world = ext.map(|e| is_world_algo(e.algo_type)).unwrap_or(false);
            let algo = if is_world {
                PitchAnalysisAlgo::WorldDll
            } else {
                PitchAnalysisAlgo::VocalShifterVslib
            };
            let vsp_track = vsp_tracks.get(base.track_index as usize);
            let track_name = vsp_track
                .map(|t| t.name.clone())
                .unwrap_or_else(|| format!("Track {}", next_order + 1));
            let color_idx = (ordered_track_ids.len() + new_tracks.len()) % TRACK_COLORS.len();
            new_tracks.push(Track {
                id: tid.clone(),
                name: track_name,
                parent_id: None,
                order: next_order,
                muted: false,
                solo: false,
                volume: 0.9,
                compose_enabled: false,
                pitch_analysis_algo: algo,
                color: TRACK_COLORS[color_idx].to_string(),
            });
            created_track_ids.insert(target_track_idx, tid.clone());
            next_order += 1;
            tid
        };

        // 记录轨道的 synth_mode（不再因为 World 而忽略）
        if let Some(e) = ext {
            synth_mode_by_track
                .entry(target_track_id.clone())
                .or_insert_with(|| algo_synth_mode(e.algo_type));
        }

        // 解析音频路径
        let audio_path = resolve_audio_path(&base.audio_path, vsp_file_dir);

        if !is_audio_supported(&audio_path) {
            skipped_files.push(base.audio_path.clone());
            continue;
        }
        if !Path::new(&audio_path).exists() {
            skipped_files.push(base.audio_path.clone());
            continue;
        }

        let item_start_sec = base.start_sample / sample_rate + time_offset;

        // 读取音频文件信息
        let audio_info = try_read_wav_info(Path::new(&audio_path), 4096);
        let (duration_sec, duration_frames, source_sr, waveform_preview) = match &audio_info {
            Some(info) => (
                Some(info.duration_sec),
                Some(info.total_frames),
                Some(info.sample_rate),
                Some(info.waveform_preview.clone()),
            ),
            None => (None, None, None, None),
        };

        let source_duration_sec = duration_sec.unwrap_or(0.0);
        let time_markers = ext.map(|e| &e.time_markers[..]).unwrap_or(&[]);
        let pitch_points = ext.map(|e| &e.pitch_points[..]).unwrap_or(&[]);

        if time_markers.len() >= 3 {
            // 非线性拉伸：拆分为多个子剪辑
            let seg_count = time_markers.len() - 1;
            let mut segment_clip_indices: Vec<usize> = Vec::with_capacity(seg_count);
            let mut segment_actual_pre_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let mut segment_actual_post_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let seg_timeline_durations: Vec<f64> = (0..seg_count)
                .map(|i| {
                    let m_start = &time_markers[i];
                    let m_end = &time_markers[i + 1];
                    ((m_end.new_pos - m_start.new_pos) / sample_rate).max(0.001)
                })
                .collect();
            for seg_idx in 0..seg_count {
                let m_start = &time_markers[seg_idx];
                let m_end = &time_markers[seg_idx + 1];

                let src_start = m_start.original_pos / sample_rate;
                let src_end = m_end.original_pos / sample_rate;
                let src_dur = (src_end - src_start).max(0.001);

                let new_start = m_start.new_pos / sample_rate;
                let _new_end = m_end.new_pos / sample_rate;
                let new_dur = seg_timeline_durations[seg_idx];

                let rate = (src_dur / new_dur) as f32;

                let want_pre_tl = if seg_idx > 0 {
                    segment_overlap_sec(seg_timeline_durations[seg_idx - 1], new_dur)
                } else {
                    0.0
                };
                let want_post_tl = if seg_idx + 1 < seg_count {
                    segment_overlap_sec(new_dur, seg_timeline_durations[seg_idx + 1])
                } else {
                    0.0
                };
                let rate64 = (rate as f64).max(0.0001);
                let want_pre_src = want_pre_tl * rate64;
                let want_post_src = want_post_tl * rate64;

                let seg_src_start = (src_start - want_pre_src).max(0.0);
                let seg_src_end = (src_end + want_post_src).min(source_duration_sec.max(src_end));
                let actual_pre_tl = (src_start - seg_src_start) / rate64;
                let actual_post_tl = (seg_src_end - src_end).max(0.0) / rate64;

                let clip_start = item_start_sec + new_start - actual_pre_tl;
                let clip_length = (new_dur + actual_pre_tl + actual_post_tl).max(0.001);
                let clip_id = new_clip_id();
                let clip_name = Path::new(&audio_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Audio")
                    .to_string();
                let clip_index = hs_clips.len();

                hs_clips.push(Clip {
                    id: clip_id.clone(),
                    track_id: target_track_id.clone(),
                    name: format!("{} ({})", clip_name, seg_idx + 1),
                    start_sec: clip_start,
                    length_sec: clip_length,
                    color: clip_color(),
                    source_path: Some(audio_path.clone()),
                    duration_sec,
                    duration_frames,
                    source_sample_rate: source_sr,
                    waveform_preview: waveform_preview.clone(),
                    pitch_range: Some(PitchRange {
                        min: -24.0,
                        max: 24.0,
                    }),
                    gain: 1.0,
                    muted: false,
                    source_start_sec: seg_src_start,
                    source_end_sec: seg_src_end,
                    playback_rate: rate.clamp(0.1, 10.0),
                    fade_in_sec: 0.0,
                    fade_out_sec: 0.0,
                    fade_in_curve: String::new(),
                    fade_out_curve: String::new(),
                    extra_curves: None,
                    extra_params: None,
                });
                segment_clip_indices.push(clip_index);
                segment_actual_pre_tl.push(actual_pre_tl);
                segment_actual_post_tl.push(actual_post_tl);

                write_pitch_data_for_segment(
                    &target_track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut pitch_data_by_track,
                );

                // 始终写入额外曲线数据（包括 World）
                write_extra_curves_for_segment(
                    &target_track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut extra_curve_data_by_track,
                );
            }

            for seg_idx in 0..seg_count {
                let clip_idx = segment_clip_indices[seg_idx];
                let Some(clip) = hs_clips.get_mut(clip_idx) else {
                    continue;
                };
                let fade_in = if seg_idx > 0 {
                    (segment_actual_pre_tl[seg_idx] + segment_actual_post_tl[seg_idx - 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                let fade_out = if seg_idx + 1 < seg_count {
                    (segment_actual_post_tl[seg_idx] + segment_actual_pre_tl[seg_idx + 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                clip.fade_in_sec = fade_in;
                clip.fade_out_sec = fade_out;
            }
        } else {
            // 线性拉伸或无拉伸
            let (rate, clip_length) = if time_markers.len() == 2 {
                let m0 = &time_markers[0];
                let m1 = &time_markers[1];
                let src_dur = ((m1.original_pos - m0.original_pos) / sample_rate).max(0.001);
                let new_dur = ((m1.new_pos - m0.new_pos) / sample_rate).max(0.001);
                let r = src_dur / new_dur;
                (r as f32, new_dur)
            } else {
                (1.0f32, source_duration_sec)
            };

            let clip_id = new_clip_id();
            let clip_name = Path::new(&audio_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Audio")
                .to_string();

            hs_clips.push(Clip {
                id: clip_id.clone(),
                track_id: target_track_id.clone(),
                name: clip_name,
                start_sec: item_start_sec,
                length_sec: clip_length,
                color: clip_color(),
                source_path: Some(audio_path.clone()),
                duration_sec,
                duration_frames,
                source_sample_rate: source_sr,
                waveform_preview,
                pitch_range: Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                }),
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: source_duration_sec,
                playback_rate: rate.clamp(0.1, 10.0),
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                extra_curves: None,
                extra_params: None,
            });

            write_pitch_data_for_segment(
                &target_track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut pitch_data_by_track,
            );

            // 始终写入额外曲线数据（包括 World）
            write_extra_curves_for_segment(
                &target_track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut extra_curve_data_by_track,
            );
        }
    }

    // 构建 pitch 参数
    let project_end = hs_clips
        .iter()
        .map(|c| c.start_sec + c.length_sec)
        .fold(32.0_f64, f64::max);
    let frame_period_ms = CTRP_FRAME_PERIOD * 1000.0;

    let mut params_by_root_track: BTreeMap<String, TrackParamsState> = BTreeMap::new();

    for track_id in created_track_ids.values().chain(ordered_track_ids.iter()) {
        if let Some(points) = pitch_data_by_track.get(track_id) {
            if points.is_empty() {
                continue;
            }
            let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);
            let mut pitch_edit = vec![0.0f32; total_frames];

            for (frame_idx, acc) in points.iter().enumerate() {
                if frame_idx < total_frames {
                    if acc.weight > 0.0 {
                        pitch_edit[frame_idx] = (acc.sum / acc.weight) as f32;
                    }
                }
            }

            // 构建 vslib 额外曲线和参数
            let extra_curves = extra_curve_data_by_track
                .get(track_id)
                .map(|ecm| build_extra_curves_from_accumulators(ecm, total_frames))
                .unwrap_or_default();

            let mut extra_params: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            if let Some(&sm) = synth_mode_by_track.get(track_id) {
                extra_params.insert("synth_mode".to_string(), sm as f64);
            }

            params_by_root_track.insert(
                track_id.clone(),
                TrackParamsState {
                    frame_period_ms,
                    pitch_orig: pitch_edit.clone(),
                    pitch_edit,
                    pitch_edit_user_modified: true,
                    tension_orig: Vec::new(),
                    tension_edit: Vec::new(),
                    pitch_orig_key: None,
                    pending_pitch_offset: None,
                    extra_curves,
                    extra_params,
                },
            );
        }
    }

    // 为有 pitch 数据的新轨道开启合成
    for track in &mut new_tracks {
        if params_by_root_track.contains_key(&track.id) {
            track.compose_enabled = true;
        }
    }

    let timeline = TimelineState {
        tracks: new_tracks,
        clips: hs_clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: project.bpm.max(1.0),
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        next_track_order: next_order,
    };

    Ok(VspImportResult {
        timeline,
        skipped_files,
        beats_per_bar: project.time_sig_num.clamp(1, 32) as u32,
    })
}

/// 剪贴板工程文件中没有选中的 Item 时，导入被标记为选中的轨道。
/// 导入的目标轨道始终是新建的轨道，起始位置保持原样（不对齐光标）。
fn import_vsp_clipboard_selected_tracks(
    project: &VspProject,
    vsp_tracks: &[VspTrack],
    item_bases: &[VspItemBase],
    item_exts: &[VspItemExt],
    vsp_file_dir: &Path,
    ordered_track_ids: &[String],
) -> Result<VspImportResult, String> {
    let sample_rate = project.sample_rate.max(1) as f64;

    // 找出被选中的轨道索引
    let selected_track_indices: Vec<usize> = vsp_tracks
        .iter()
        .enumerate()
        .filter(|(_, t)| t.selected)
        .map(|(i, _)| i)
        .collect();

    if selected_track_indices.is_empty() {
        return Err("no_selected_items".into());
    }

    let selected_set: std::collections::HashSet<usize> =
        selected_track_indices.iter().copied().collect();

    let mut skipped_files: Vec<String> = Vec::new();
    let mut hs_tracks: Vec<Track> = Vec::new();
    let mut hs_clips: Vec<Clip> = Vec::new();
    let mut pitch_data_by_track: std::collections::HashMap<String, Vec<PitchFrameAccumulator>> =
        std::collections::HashMap::new();
    let mut extra_curve_data_by_track: std::collections::HashMap<
        String,
        Vec<ExtraCurveAccumulator>,
    > = std::collections::HashMap::new();
    let mut synth_mode_by_track: std::collections::HashMap<String, i32> =
        std::collections::HashMap::new();

    let mut track_order = ordered_track_ids.len() as i32;

    // 原始轨道索引 + (is_world, synth_mode) → 新建的 HiFiShifter 轨道 ID
    let mut vsp_to_hs_track: std::collections::HashMap<(i32, bool, i32), String> =
        std::collections::HashMap::new();

    // 对被选中轨道内不同算法的首次出现时间进行统计（用于排序拆分后的轨道）
    let mut earliest_start_by_algo: std::collections::HashMap<(i32, bool, i32), f64> =
        std::collections::HashMap::new();
    for (i, base) in item_bases.iter().enumerate() {
        if !selected_set.contains(&(base.track_index as usize)) {
            continue;
        }
        let pair = item_exts
            .get(i)
            .map(|e| algo_type_to_hs(e.algo_type))
            .unwrap_or((false, 1));
        let start_sec = base.start_sample / sample_rate;
        let key = (base.track_index, pair.0, pair.1);
        let entry = earliest_start_by_algo.entry(key).or_insert(f64::INFINITY);
        if start_sec.is_finite() && start_sec < *entry {
            *entry = start_sec;
        }
    }

    // 为每个选中的轨道创建新轨道（按不同算法对拆分）
    for &vsp_idx in &selected_track_indices {
        let vsp_track = &vsp_tracks[vsp_idx];

        // 收集该轨道内所有不同的 (is_world, synth_mode) 对
        let mut algo_pairs: std::collections::HashSet<(bool, i32)> =
            std::collections::HashSet::new();
        for (i, base) in item_bases.iter().enumerate() {
            if base.track_index as usize == vsp_idx {
                let pair = item_exts
                    .get(i)
                    .map(|e| algo_type_to_hs(e.algo_type))
                    .unwrap_or((false, 1));
                algo_pairs.insert(pair);
            }
        }

        let has_mixed = algo_pairs.len() > 1;

        if has_mixed {
            let mut sorted: Vec<(bool, i32)> = algo_pairs.into_iter().collect();
            // 按首次出现时间排序同源拆分轨道
            sorted.sort_by(|a, b| {
                let ka = (vsp_idx as i32, a.0, a.1);
                let kb = (vsp_idx as i32, b.0, b.1);
                let ta = earliest_start_by_algo
                    .get(&ka)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                let tb = earliest_start_by_algo
                    .get(&kb)
                    .copied()
                    .unwrap_or(f64::INFINITY);
                ta.partial_cmp(&tb).unwrap_or(std::cmp::Ordering::Equal)
            });
            for (is_world, synth_mode) in sorted {
                let algo = if is_world {
                    PitchAnalysisAlgo::WorldDll
                } else {
                    PitchAnalysisAlgo::VocalShifterVslib
                };
                let id = new_track_id();
                let color_idx = (ordered_track_ids.len() + hs_tracks.len()) % TRACK_COLORS.len();
                hs_tracks.push(Track {
                    id: id.clone(),
                    name: vsp_track.name.clone(),
                    parent_id: None,
                    order: track_order,
                    muted: vsp_track.muted,
                    solo: vsp_track.solo,
                    volume: convert_volume(vsp_track.volume),
                    compose_enabled: false,
                    pitch_analysis_algo: algo,
                    color: TRACK_COLORS[color_idx].to_string(),
                });
                vsp_to_hs_track.insert((vsp_idx as i32, is_world, synth_mode), id);
                track_order += 1;
            }
        } else {
            let (is_world, _synth_mode) = algo_pairs.into_iter().next().unwrap_or((false, 1));
            let algo = if is_world {
                PitchAnalysisAlgo::WorldDll
            } else {
                PitchAnalysisAlgo::VocalShifterVslib
            };
            let id = new_track_id();
            let color_idx = (ordered_track_ids.len() + hs_tracks.len()) % TRACK_COLORS.len();
            hs_tracks.push(Track {
                id: id.clone(),
                name: vsp_track.name.clone(),
                parent_id: None,
                order: track_order,
                muted: vsp_track.muted,
                solo: vsp_track.solo,
                volume: convert_volume(vsp_track.volume),
                compose_enabled: false,
                pitch_analysis_algo: algo,
                color: TRACK_COLORS[color_idx].to_string(),
            });
            vsp_to_hs_track.insert((vsp_idx as i32, is_world, _synth_mode), id);
            track_order += 1;
        }
    }

    // 导入选中轨道上的所有 Item（保持原始起始位置）
    for (i, base) in item_bases.iter().enumerate() {
        if !selected_set.contains(&(base.track_index as usize)) {
            continue;
        }

        let ext = item_exts.get(i);

        let (is_world, synth_mode) = ext
            .map(|e| algo_type_to_hs(e.algo_type))
            .unwrap_or((false, 1));
        let Some(track_id) = vsp_to_hs_track
            .get(&(base.track_index, is_world, synth_mode))
            .or_else(|| {
                // 回退：查找同一原始轨道索引下的任意映射
                vsp_to_hs_track
                    .iter()
                    .find(|(&(i, _, _), _)| i == base.track_index)
                    .map(|(_, id)| id)
            })
            .cloned()
        else {
            continue;
        };

        // 记录轨道的 synth_mode（不再因为 World 而忽略）
        if let Some(e) = ext {
            let sm = algo_synth_mode(e.algo_type);
            synth_mode_by_track.entry(track_id.clone()).or_insert(sm);
        }

        let audio_path = resolve_audio_path(&base.audio_path, vsp_file_dir);

        if !is_audio_supported(&audio_path) {
            skipped_files.push(base.audio_path.clone());
            continue;
        }
        if !Path::new(&audio_path).exists() {
            skipped_files.push(base.audio_path.clone());
            continue;
        }

        let item_start_sec = base.start_sample / sample_rate;

        let audio_info = try_read_wav_info(Path::new(&audio_path), 4096);
        let (duration_sec, duration_frames, source_sr, waveform_preview) = match &audio_info {
            Some(info) => (
                Some(info.duration_sec),
                Some(info.total_frames),
                Some(info.sample_rate),
                Some(info.waveform_preview.clone()),
            ),
            None => (None, None, None, None),
        };

        let source_duration_sec = duration_sec.unwrap_or(0.0);
        let time_markers = ext.map(|e| &e.time_markers[..]).unwrap_or(&[]);
        let pitch_points = ext.map(|e| &e.pitch_points[..]).unwrap_or(&[]);

        if time_markers.len() >= 3 {
            let seg_count = time_markers.len() - 1;
            let mut segment_clip_indices: Vec<usize> = Vec::with_capacity(seg_count);
            let mut segment_actual_pre_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let mut segment_actual_post_tl: Vec<f64> = Vec::with_capacity(seg_count);
            let seg_timeline_durations: Vec<f64> = (0..seg_count)
                .map(|i| {
                    let m_start = &time_markers[i];
                    let m_end = &time_markers[i + 1];
                    ((m_end.new_pos - m_start.new_pos) / sample_rate).max(0.001)
                })
                .collect();
            for seg_idx in 0..seg_count {
                let m_start = &time_markers[seg_idx];
                let m_end = &time_markers[seg_idx + 1];

                let src_start = m_start.original_pos / sample_rate;
                let src_end = m_end.original_pos / sample_rate;
                let src_dur = (src_end - src_start).max(0.001);

                let new_start = m_start.new_pos / sample_rate;
                let _new_end = m_end.new_pos / sample_rate;
                let new_dur = seg_timeline_durations[seg_idx];

                let rate = (src_dur / new_dur) as f32;
                let rate64 = (rate as f64).max(0.0001);

                let want_pre_tl = if seg_idx > 0 {
                    segment_overlap_sec(seg_timeline_durations[seg_idx - 1], new_dur)
                } else {
                    0.0
                };
                let want_post_tl = if seg_idx + 1 < seg_count {
                    segment_overlap_sec(new_dur, seg_timeline_durations[seg_idx + 1])
                } else {
                    0.0
                };
                let want_pre_src = want_pre_tl * rate64;
                let want_post_src = want_post_tl * rate64;

                let seg_src_start = (src_start - want_pre_src).max(0.0);
                let seg_src_end = (src_end + want_post_src).min(source_duration_sec.max(src_end));
                let actual_pre_tl = (src_start - seg_src_start) / rate64;
                let actual_post_tl = (seg_src_end - src_end).max(0.0) / rate64;

                let clip_start = item_start_sec + new_start - actual_pre_tl;
                let clip_length = (new_dur + actual_pre_tl + actual_post_tl).max(0.001);
                let clip_id = new_clip_id();
                let clip_name = Path::new(&audio_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Audio")
                    .to_string();
                let clip_index = hs_clips.len();

                hs_clips.push(Clip {
                    id: clip_id.clone(),
                    track_id: track_id.clone(),
                    name: format!("{} ({})", clip_name, seg_idx + 1),
                    start_sec: clip_start,
                    length_sec: clip_length,
                    color: clip_color(),
                    source_path: Some(audio_path.clone()),
                    duration_sec,
                    duration_frames,
                    source_sample_rate: source_sr,
                    waveform_preview: waveform_preview.clone(),
                    pitch_range: Some(PitchRange {
                        min: -24.0,
                        max: 24.0,
                    }),
                    gain: 1.0,
                    muted: false,
                    source_start_sec: seg_src_start,
                    source_end_sec: seg_src_end,
                    playback_rate: rate.clamp(0.1, 10.0),
                    fade_in_sec: 0.0,
                    fade_out_sec: 0.0,
                    fade_in_curve: String::new(),
                    fade_out_curve: String::new(),
                    extra_curves: None,
                    extra_params: None,
                });
                segment_clip_indices.push(clip_index);
                segment_actual_pre_tl.push(actual_pre_tl);
                segment_actual_post_tl.push(actual_post_tl);

                write_pitch_data_for_segment(
                    &track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut pitch_data_by_track,
                );
                // 始终写入额外曲线数据（包括 World）
                write_extra_curves_for_segment(
                    &track_id,
                    pitch_points,
                    seg_src_start,
                    seg_src_end,
                    clip_start,
                    rate64,
                    0.0,
                    0.0,
                    &mut extra_curve_data_by_track,
                );
            }

            for seg_idx in 0..seg_count {
                let clip_idx = segment_clip_indices[seg_idx];
                let Some(clip) = hs_clips.get_mut(clip_idx) else {
                    continue;
                };
                let fade_in = if seg_idx > 0 {
                    (segment_actual_pre_tl[seg_idx] + segment_actual_post_tl[seg_idx - 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                let fade_out = if seg_idx + 1 < seg_count {
                    (segment_actual_post_tl[seg_idx] + segment_actual_pre_tl[seg_idx + 1])
                        .min(clip.length_sec.max(0.0))
                } else {
                    0.0
                };
                clip.fade_in_sec = fade_in;
                clip.fade_out_sec = fade_out;
            }
        } else {
            let (rate, clip_length) = if time_markers.len() == 2 {
                let m0 = &time_markers[0];
                let m1 = &time_markers[1];
                let src_dur = ((m1.original_pos - m0.original_pos) / sample_rate).max(0.001);
                let new_dur = ((m1.new_pos - m0.new_pos) / sample_rate).max(0.001);
                let r = src_dur / new_dur;
                (r as f32, new_dur)
            } else {
                (1.0f32, source_duration_sec)
            };

            let clip_id = new_clip_id();
            let clip_name = Path::new(&audio_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Audio")
                .to_string();

            hs_clips.push(Clip {
                id: clip_id.clone(),
                track_id: track_id.clone(),
                name: clip_name,
                start_sec: item_start_sec,
                length_sec: clip_length,
                color: clip_color(),
                source_path: Some(audio_path.clone()),
                duration_sec,
                duration_frames,
                source_sample_rate: source_sr,
                waveform_preview,
                pitch_range: Some(PitchRange {
                    min: -24.0,
                    max: 24.0,
                }),
                gain: 1.0,
                muted: false,
                source_start_sec: 0.0,
                source_end_sec: source_duration_sec,
                playback_rate: rate.clamp(0.1, 10.0),
                fade_in_sec: 0.0,
                fade_out_sec: 0.0,
                fade_in_curve: String::new(),
                fade_out_curve: String::new(),
                extra_curves: None,
                extra_params: None,
            });

            write_pitch_data_for_segment(
                &track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut pitch_data_by_track,
            );
            // 始终写入额外曲线数据（包括 World）
            write_extra_curves_for_segment(
                &track_id,
                pitch_points,
                0.0,
                source_duration_sec,
                item_start_sec,
                rate as f64,
                0.0,
                0.0,
                &mut extra_curve_data_by_track,
            );
        }
    }

    // 构建 pitch 参数
    let project_end = hs_clips
        .iter()
        .map(|c| c.start_sec + c.length_sec)
        .fold(32.0_f64, f64::max);
    let frame_period_ms = CTRP_FRAME_PERIOD * 1000.0;

    let mut params_by_root_track: BTreeMap<String, TrackParamsState> = BTreeMap::new();

    for track in &hs_tracks {
        if let Some(points) = pitch_data_by_track.get(&track.id) {
            if points.is_empty() {
                continue;
            }
            let total_frames = ((project_end * 1000.0 / frame_period_ms).ceil() as usize).max(1);
            let mut pitch_edit = vec![0.0f32; total_frames];

            for (frame_idx, acc) in points.iter().enumerate() {
                if frame_idx < total_frames {
                    pitch_edit[frame_idx] = (acc.sum / acc.weight) as f32;
                }
            }

            // 构建 vslib 额外曲线和参数
            let extra_curves = extra_curve_data_by_track
                .get(&track.id)
                .map(|ecm| build_extra_curves_from_accumulators(ecm, total_frames))
                .unwrap_or_default();

            let mut extra_params: std::collections::HashMap<String, f64> =
                std::collections::HashMap::new();
            if let Some(&sm) = synth_mode_by_track.get(&track.id) {
                extra_params.insert("synth_mode".to_string(), sm as f64);
            }

            params_by_root_track.insert(
                track.id.clone(),
                TrackParamsState {
                    frame_period_ms,
                    pitch_orig: pitch_edit.clone(),
                    pitch_edit,
                    pitch_edit_user_modified: true,
                    tension_orig: Vec::new(),
                    tension_edit: Vec::new(),
                    pitch_orig_key: None,
                    pending_pitch_offset: None,
                    extra_curves,
                    extra_params,
                },
            );
        }
    }

    // 为有 pitch 数据的轨道开启合成
    for track in &mut hs_tracks {
        if params_by_root_track.contains_key(&track.id) {
            track.compose_enabled = true;
        }
    }

    let timeline = TimelineState {
        tracks: hs_tracks,
        clips: hs_clips,
        selected_track_id: None,
        selected_clip_id: None,
        bpm: project.bpm.max(1.0),
        playhead_sec: 0.0,
        project_sec: project_end,
        params_by_root_track,
        next_track_order: track_order,
    };

    Ok(VspImportResult {
        timeline,
        skipped_files,
        beats_per_bar: project.time_sig_num.clamp(1, 32) as u32,
    })
}

/// 将相对路径解析为绝对路径。
fn resolve_audio_path(raw_path: &str, vsp_dir: &Path) -> String {
    let p = Path::new(raw_path);
    if p.is_absolute() {
        // 规范化路径分隔符
        return raw_path.to_string();
    }
    // 相对路径：基于 .vshp/.vsp 所在目录拼接
    let resolved = vsp_dir.join(p);
    resolved.to_string_lossy().to_string()
}
