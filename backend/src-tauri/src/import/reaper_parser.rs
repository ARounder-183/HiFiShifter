// Reaper 工程文件 / 剪贴板数据解析模块
//
// 将 Reaper RPP 文本格式解析为中间数据结构。

use std::path::Path;

// ─── 数据结构 ───

#[derive(Debug, Clone, Default)]
pub struct ReaperData {
    pub tracks: Vec<ReaperTrack>,
    pub is_track_data: bool,
    pub tempo_envelope: Option<ReaperTempoEnvelope>,
    /// 工程 BPM 与拍号信息（从 TEMPO 行解析）。
    pub tempo: Option<ReaperTempo>,
    /// 每个 track 相对于首个 track 的轨道偏移量（由 TRACKSKIP 累计得出）。
    /// 与 tracks 等长，tracks[0] 的 offset 始终为 0。
    pub track_offsets: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ReaperTempo {
    /// 工程 BPM 值
    pub bpm: f64,
    /// 每小节拍数（拍号分子）
    pub beats_per_bar: u32,
    /// 基准音符（4 = 四分音符，8 = 八分音符等）
    #[allow(dead_code)]
    pub beat_note: u32,
}

#[derive(Debug, Clone)]
pub struct ReaperTrack {
    pub items: Vec<ReaperItem>,
    pub name: String,
    pub vol_pan: Vec<f64>,   // [vol, pan, ...]
    pub mute_solo: Vec<i32>, // [mute, solo, ...]
    pub iphase: bool,
    pub envelopes: Vec<ReaperEnvelope>,
    /// ISBUS 参数：[type, delta]，delta 决定下一条轨道的层级变化量。
    /// 例如 ISBUS 1 1 表示下一条轨道深度 +1（成为子轨道），ISBUS 2 -1 表示 -1。
    pub isbus: Vec<i32>,
}

impl Default for ReaperTrack {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            name: String::new(),
            vol_pan: vec![1.0, 0.0, -1.0, -1.0, 1.0],
            mute_solo: vec![0, 0, 0],
            iphase: false,
            envelopes: Vec::new(),
            isbus: vec![0, 0],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReaperItem {
    pub position: f64,
    pub snap_offs: f64,
    pub length: f64,
    pub is_loop: bool,
    /// ITEM 块是否显式写出 `LOOP n` 行。
    ///
    /// REAPER 保存的 ITEM 几乎总是携带 LOOP 行；缺失时（极老版本/第三方生成器）
    /// 导入方应以"为新的音频块启用循环"设置作为默认值，而不是硬编码 false。
    pub has_loop_token: bool,
    pub all_takes: bool,
    pub fade_in: Vec<f64>,
    pub fade_out: Vec<f64>,
    pub mute: Vec<i32>,
    pub selected: bool,
    pub envelopes: Vec<ReaperEnvelope>,
    pub takes: Vec<ReaperTake>,
    // 首个 take 的属性（item 自身也是一个隐式 take）
    pub default_take: ReaperTake,
    pub stretch_markers: Vec<ReaperStretchMarker>,
    pub group_id: Option<i32>,
}

impl Default for ReaperItem {
    fn default() -> Self {
        Self {
            position: 0.0,
            snap_offs: 0.0,
            length: 0.0,
            is_loop: false,
            has_loop_token: false,
            all_takes: false,
            fade_in: vec![0.0; 7],
            fade_out: vec![0.0; 7],
            mute: vec![0, 0],
            selected: false,
            envelopes: Vec::new(),
            takes: Vec::new(),
            default_take: ReaperTake::default(),
            stretch_markers: Vec::new(),
            group_id: None,
        }
    }
}

impl ReaperItem {
    /// 返回当前活跃的 take。
    /// 如果没有显式 take，返回 item 的默认 take（隐式首 take）。
    /// 如果有显式 take，先检查被标记 selected 的；否则优先返回
    /// 有 source 的默认 take，再回退到第一个显式 take。
    pub fn active_take(&self) -> &ReaperTake {
        for take in &self.takes {
            if take.selected {
                return take;
            }
        }
        if self.default_take.source.is_some() {
            return &self.default_take;
        }
        if let Some(first_take) = self.takes.first() {
            return first_take;
        }
        &self.default_take
    }
}

#[derive(Debug, Clone)]
pub struct ReaperTake {
    pub selected: bool,
    pub name: String,
    pub vol_pan: Vec<f64>, // [vol, pan, gainTrim, ...]
    pub fade_in: Vec<f64>,
    pub fade_out: Vec<f64>,
    pub s_offs: f64,
    pub play_rate: Vec<f64>, // [rate, preserve, pitch, method, ...]
    pub chan_mode: i32,
    pub source: Option<ReaperSource>,
}

impl Default for ReaperTake {
    fn default() -> Self {
        Self {
            selected: false,
            name: String::new(),
            vol_pan: vec![1.0, 0.0, 1.0, -1.0],
            fade_in: vec![0.0; 7],
            fade_out: vec![0.0; 7],
            s_offs: 0.0,
            play_rate: vec![1.0, 1.0, 0.0, -1.0, 0.0, 0.0025],
            chan_mode: 0,
            source: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReaperSource {
    pub source_type: String,
    pub file_path: String,
    /// Reaper SECTION SOURCE 的 MODE 值。
    /// 当 MODE > 0 时表示该 SECTION 以反向方式读取。
    pub section_mode: i32,
    /// Reaper SECTION SOURCE 的起点（秒）。
    pub section_start_sec: Option<f64>,
    /// Reaper SECTION SOURCE 的长度（秒）。
    pub section_length_sec: Option<f64>,
    file_path_full: Option<String>,
    /// MIDI 源数据（仅当 source_type == "MIDI" 时填充）
    pub midi_source: Option<ReaperMidiSourceData>,
}

impl ReaperSource {
    pub fn new() -> Self {
        Self {
            source_type: String::new(),
            file_path: String::new(),
            section_mode: 0,
            section_start_sec: None,
            section_length_sec: None,
            file_path_full: None,
            midi_source: None,
        }
    }

    pub fn resolved_path(&self) -> &str {
        if let Some(ref full) = self.file_path_full {
            if Path::new(full).exists() {
                return full;
            }
        }
        &self.file_path
    }

    pub fn update_full_path(&mut self, folder: &Path) {
        if self.file_path.is_empty() {
            return;
        }
        let joined = folder.join(&self.file_path);
        if joined.exists() {
            self.file_path_full = Some(joined.to_string_lossy().to_string());
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReaperMidiEvent {
    pub tick_offset: u64,
    pub status: u8,
    pub data1: u8,
    pub data2: u8,
}

#[derive(Debug, Clone)]
pub struct ReaperIgnTempo {
    /// true = 使用自身 BPM 而非工程 BPM
    pub ignore_project: bool,
    pub tempo: f64,
    #[allow(dead_code)]
    pub beats: u32,
    #[allow(dead_code)]
    pub beat_note: u32,
}

#[derive(Debug, Clone)]
pub struct ReaperMidiSourceData {
    pub ticks_per_qn: u32,
    pub events: Vec<ReaperMidiEvent>,
    pub igntempo: Option<ReaperIgnTempo>,
}

#[derive(Debug, Clone)]
pub struct ReaperStretchMarker {
    pub offset: f64,
    pub position: f64,
    pub velocity_change: f64,
}

#[derive(Debug, Clone)]
pub struct ReaperStretchSegment {
    /// 段起点，take 媒体时间（item 时间 × take 播放速率，item 起点为 0）。
    pub offset_start: f64,
    /// 段终点，take 媒体时间。
    pub offset_end: f64,
    /// 段起点对应的源媒体位置（原始媒体坐标，start ≤ end；倒放 take 已做
    /// 镜像换算，时间线推进时源位置自 end 向 start 递减）。
    pub src_start: f64,
    /// 段终点对应的源媒体位置（原始媒体坐标）。
    pub src_end: f64,
    pub velocity_start: f64,
    pub velocity_end: f64,
}

impl ReaperStretchSegment {
    pub fn offset_length(&self) -> f64 {
        self.offset_end - self.offset_start
    }

    pub fn velocity_average(&self) -> f64 {
        (self.velocity_start + self.velocity_end) / 2.0
    }
}

/// 将 REAPER 拉伸标记展开为覆盖整个 item 窗口的分段源映射。
///
/// 拉伸标记是 REAPER Take 的属性：它直接锚定"take 媒体时间 → 源媒体位置"
/// 的分段线性映射，且**先于 Item 的裁断存在**。因此：
/// - 所有标记（无论是否落在 item 窗口内）都是映射锚点，不得因窗口裁断
///   被丢弃或钳位——末标记超出 item 末尾时，其所在段的速率仍治理 item
///   尾部区域；窗口起点在首标记之前时，同理由既有锚点段穿越决定。
/// - 标记的 `offset` 是 take 媒体时间（item 时间 × playrate，item 起点为
///   原点）；`position` 是该 take 源坐标系下的绝对位置（随 u 单调递增）。
///   倒放 take 的该坐标系被 REAPER 预先翻转为"媒体全长 − 原始位置"
///   （实测 SM_Test_2：镜像轴 = 媒体文件全长，而非 SECTION 长度），本函数
///   在 `reversed` 时用 `media_length_sec` 把各段源窗口镜像回原始媒体坐标。
/// - item 起点的隐式锚点为 (0, SOFFS)，仅当首标记 offset > 0 时插入；
///   存在 offset ≤ 0 的标记时，窗口起点由其所在段穿越决定（REAPER 的
///   裁断/倒放会同步平移标记，两种表达保持一致）。
/// - 最后一个锚点之后、窗口终点之前按基准速率（斜率 1）外推。
///
/// `velocity_*` 为该段源消耗斜率（Δ源 / Δtake媒体，恒为正），段起点锚点
/// 上的速率变化系数展开为段内线性速率坡（倒放 take 的该系数已被 REAPER 反号，
/// 直接套用即为原始坐标下的坡向）。
/// 窗口外的锚点只参与几何计算，不产生额外段。无标记、item 长度非正、
/// 或倒放 take 缺少媒体时长（无法镜像坐标系）时返回空表，调用方回退到
/// 单 clip 路径。
pub fn stretch_segments_full_cover(
    markers: &[ReaperStretchMarker],
    s_offs: f64,
    play_rate: f64,
    item_length: f64,
    media_length_sec: Option<f64>,
    reversed: bool,
) -> Vec<ReaperStretchSegment> {
    if markers.is_empty() {
        return Vec::new();
    }
    let u_end = item_length * play_rate;
    if !u_end.is_finite() || u_end <= 1e-9 {
        return Vec::new();
    }

    // 倒放 take 的源窗口镜像（s → 媒体全长 − s）必须知道媒体全长。
    let media_length_sec = if reversed {
        match media_length_sec {
            Some(len) if len.is_finite() && len > 0.0 => Some(len),
            _ => return Vec::new(),
        }
    } else {
        None
    };

    // 锚点 (take媒体时间, take源坐标位置, 速率变化系数)，按时间排序；
    // 窗口外的标记保留（take 属性，先于裁断存在）。take 源坐标系随 u
    // 单调递增，几何全部在该坐标系内计算，仅在输出时镜像。
    let mut ordered: Vec<&ReaperStretchMarker> = markers.iter().collect();
    ordered.sort_by(|a, b| a.offset.partial_cmp(&b.offset).unwrap_or(std::cmp::Ordering::Equal));
    let mut anchors: Vec<(f64, f64, f64)> = Vec::with_capacity(ordered.len() + 1);
    for marker in ordered {
        if !marker.offset.is_finite() || !marker.position.is_finite() {
            continue;
        }
        if anchors
            .last()
            .is_some_and(|last: &(f64, f64, f64)| (last.0 - marker.offset).abs() <= 1e-9)
        {
            continue;
        }
        anchors.push((marker.offset, marker.position, marker.velocity_change));
    }
    if anchors.is_empty() {
        return Vec::new();
    }

    // item 起点隐式锚点：仅当首标记在窗口起点之后。
    if anchors[0].0 > 1e-9 {
        anchors.insert(0, (0.0, s_offs, 0.0));
    }

    // 输出：正放直接取存储坐标；倒放把段源窗口镜像回原始媒体坐标
    // （时间线推进时源位置自 end 向 start 递减）。速率坡为坐标无关量，
    // 两种情况同形。
    let emit =
        |offset_start: f64,
         offset_end: f64,
         s_lo: f64,
         s_hi: f64,
         velocity_start: f64,
         velocity_end: f64| {
            if !reversed {
                ReaperStretchSegment {
                    offset_start,
                    offset_end,
                    src_start: s_lo,
                    src_end: s_hi,
                    velocity_start,
                    velocity_end,
                }
            } else {
                let len = media_length_sec.unwrap_or(0.0);
                ReaperStretchSegment {
                    offset_start,
                    offset_end,
                    src_start: len - s_hi,
                    src_end: len - s_lo,
                    velocity_start,
                    velocity_end,
                }
            }
        };

    let mut segments = Vec::with_capacity(anchors.len());
    for pair in anchors.windows(2) {
        let (u_a, s_a, vc_a) = pair[0];
        let (u_b, s_b, _) = pair[1];
        let du = u_b - u_a;
        if du <= 1e-9 {
            continue;
        }
        let lo = u_a.max(0.0);
        let hi = u_b.min(u_end);
        if hi - lo <= 1e-9 {
            continue;
        }
        let rate = (s_b - s_a) / du;
        if !rate.is_finite() {
            continue;
        }
        let s_lo = s_a + rate * (lo - u_a);
        let s_hi = s_a + rate * (hi - u_a);
        segments.push(emit(
            lo,
            hi,
            s_lo,
            s_hi,
            rate * (1.0 - vc_a),
            rate * (1.0 + vc_a),
        ));
    }

    // 末锚点之后、窗口终点之前：基准速率（斜率 1）外推。
    let last = anchors[anchors.len() - 1];
    if u_end - last.0 > 1e-9 {
        let lo = last.0.max(0.0);
        let s_lo = last.1 + (lo - last.0);
        let s_hi = last.1 + (u_end - last.0);
        segments.push(emit(lo, u_end, s_lo, s_hi, 1.0, 1.0));
    }

    segments
}

#[derive(Debug, Clone)]
pub struct ReaperEnvelope {
    pub env_type: String,
    pub act: Vec<i32>,
    pub seg_range: Option<Vec<f64>>,
    pub points: Vec<Vec<f64>>,
}

impl Default for ReaperEnvelope {
    fn default() -> Self {
        Self {
            env_type: String::new(),
            act: vec![1, -1],
            seg_range: None,
            points: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ReaperTempoEnvelopePoint {
    /// 位置（秒，时间锚定）。
    pub position_sec: f64,
    /// BPM。
    pub bpm: f64,
    /// 变化形状：1 = 阶梯（square），0 = 线性渐变到下一个点。
    pub shape: i32,
    /// 拍号分子（有拍号信息时为 Some）。
    pub numerator: Option<u32>,
    /// 拍号分母（有拍号信息时为 Some）。
    pub denominator: Option<u32>,
}

#[derive(Debug, Clone)]
pub struct ReaperTempoEnvelope {
    pub points: Vec<ReaperTempoEnvelopePoint>,
}

/// 解析 TEMPOENVEX `PT` 行第 4 个值（slowcurv）中打包的拍号。
///
/// REAPER 将拍号编码为 `denom_base * 1000 + (num_base + num - 1)`：
/// - 分母基值：65 = 全音符(1)，131 = 2 分音符(2)，262 = 4 分音符(4)，
///   524 = 8 分音符(8)，1048 = 16 分音符(16)，2097 = 32 分音符(32)，
///   4194 = 64 分音符(64)，8388 = 128 分音符(128)；
/// - 分子基值按分母依次为 537 / 73 / 145 / 289 / 577 / 153 / 305 / 609。
/// 参见 https://wiki.cockos.com/wiki/index.php/RPR_GetSetEnvelopeState
fn parse_tempo_env_time_signature(slow_curv: f64) -> Option<(u32, u32)> {
    let value = slow_curv.round() as i64;
    if value <= 0 {
        return None;
    }
    const DENOM_BASES: &[(i64, u32)] = &[
        (8388, 128),
        (4194, 64),
        (2097, 32),
        (1048, 16),
        (524, 8),
        (262, 4),
        (131, 2),
        (65, 1),
    ];
    const NUM_BASES: &[(i64, i64)] = &[
        (8388, 609),
        (4194, 305),
        (2097, 153),
        (1048, 577),
        (524, 289),
        (262, 145),
        (131, 73),
        (65, 537),
    ];
    for &(denom_base, denominator) in DENOM_BASES {
        if value >= denom_base * 1000 && value < (denom_base + 1) * 1000 {
            let num_part = value - denom_base * 1000;
            let num_base = NUM_BASES
                .iter()
                .find(|(base, _)| *base == denom_base)
                .map(|(_, nb)| *nb)
                .unwrap_or(0);
            let numerator = (num_part - num_base + 1).clamp(1, 32) as u32;
            // 仅接受 HiFiShifter 支持的分母集合。
            if matches!(denominator, 1 | 2 | 4 | 8 | 16 | 32) {
                return Some((numerator, denominator));
            }
        }
    }
    None
}

// ─── 块解析器 ───

const ENVELOPE_TYPES: &[&str] = &[
    "ENVSEG",
    "VOLENV",
    "VOLENV2",
    "PANENV",
    "PANENV2",
    "MUTEENV",
    "TEMPOENVEX",
    "PITCHENV",
];

#[derive(Debug)]
struct Block {
    lines: Vec<String>,
    children: Vec<Block>,
}

impl Block {
    fn block_type(&self) -> Option<String> {
        let first = self.lines.first()?;
        let trimmed = first.trim();
        if !trimmed.starts_with('<') {
            return None;
        }
        let after = &trimmed[1..]; // skip '<'
        let end = after
            .find(|c: char| c == ' ' || c == '\t')
            .unwrap_or(after.len());
        Some(after[..end].to_uppercase())
    }
}

/// 从原始文本行构建嵌套块结构（对应 C# ReaperBlock 构造函数）。
fn parse_blocks(lines: &[String]) -> Block {
    let root = Block {
        lines: Vec::new(),
        children: Vec::new(),
    };
    let mut stack: Vec<Block> = vec![Block {
        lines: Vec::new(),
        children: Vec::new(),
    }];

    static SKIP_DIRECTIVES: &[&str] = &["TRACKSKIP"];

    for raw_line in lines {
        let line = raw_line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        let directive = tokens.first().unwrap_or(&"");

        if SKIP_DIRECTIVES
            .iter()
            .any(|&d| d.eq_ignore_ascii_case(directive))
        {
            let child = Block {
                lines: vec![line],
                children: Vec::new(),
            };
            stack.last_mut().unwrap().children.push(child);
            continue;
        }

        let first_char = line.chars().next().unwrap_or(' ');

        if first_char == '<' {
            // 开始新块
            let new_block = Block {
                lines: vec![line],
                children: Vec::new(),
            };
            // push onto stack
            stack.push(new_block);
        } else if first_char == '>' {
            // 关闭当前块
            if stack.len() > 1 {
                let mut finished = stack.pop().unwrap();
                finished.lines.push(line);
                stack.last_mut().unwrap().children.push(finished);
            }
        } else {
            // 普通行，添加到当前块
            stack.last_mut().unwrap().lines.push(line);
        }
    }

    // 收集所有剩余未关闭的块
    while stack.len() > 1 {
        let finished = stack.pop().unwrap();
        stack.last_mut().unwrap().children.push(finished);
    }

    stack.pop().unwrap_or(root)
}

// ─── 文本分割 ───

/// Reaper 使用两种分隔符：\r\n（.rpp 文件）和 \0（剪贴板数据）。
fn split_lines(data: &[u8]) -> Vec<String> {
    let mut lines = Vec::with_capacity(data.len() / 40);
    let mut start = 0;
    let mut i = 0;
    while i < data.len() {
        if data[i] == 0x00 {
            if i > start {
                if let Ok(s) = std::str::from_utf8(&data[start..i]) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.to_string());
                    }
                }
            }
            start = i + 1;
        } else if data[i] == 0x0D && i + 1 < data.len() && data[i + 1] == 0x0A {
            if i > start {
                if let Ok(s) = std::str::from_utf8(&data[start..i]) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.to_string());
                    }
                }
            }
            start = i + 2;
            i += 1; // 跳过 \n
        } else if data[i] == 0x0A {
            // 单独的 \n
            if i > start {
                if let Ok(s) = std::str::from_utf8(&data[start..i]) {
                    let trimmed = s.trim();
                    if !trimmed.is_empty() {
                        lines.push(trimmed.to_string());
                    }
                }
            }
            start = i + 1;
        }
        i += 1;
    }
    if start < data.len() {
        if let Ok(s) = std::str::from_utf8(&data[start..]) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                lines.push(trimmed.to_string());
            }
        }
    }
    lines
}

// ─── Token 解析辅助 ───

fn split_tokens(line: &str) -> Vec<&str> {
    line.split(|c: char| c == ' ' || c == '\t')
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_double(s: &str) -> f64 {
    s.parse::<f64>()
        .ok()
        // 拒绝非有限值（"inf"/"NaN"/"-inf"）：畸形 RPP 里的 POSITION/LENGTH/
        // SOFFS 若携带 inf/NaN 会静默污染下游几何运算（缓存 key、环绕数学、
        // 序列化往返），统一回退 0.0。
        .filter(|v| v.is_finite())
        .unwrap_or(0.0)
}

fn parse_int(s: &str) -> i32 {
    s.parse::<i32>().unwrap_or(0)
}

fn parse_bool(s: &str) -> bool {
    parse_int(s) != 0
}

fn parse_double_array(tokens: &[&str]) -> Vec<f64> {
    tokens[1..].iter().map(|s| parse_double(s)).collect()
}

fn parse_int_array(tokens: &[&str]) -> Vec<i32> {
    tokens[1..].iter().map(|s| parse_int(s)).collect()
}

fn parse_hex_byte(s: &str) -> u8 {
    u8::from_str_radix(s, 16).unwrap_or(0)
}

/// 解析 FADEIN/FADEOUT 参数（不做任何长度归一化）。
///
/// REAPER 的 fade 数组中同时携带「手动淡化长度」与「自动交叉淡化长度」：
/// - 索引 1：手动淡化的长度（始终存在，是用户手动设置/持久保存的值）；
/// - 索引 2：自动交叉淡化的长度（仅在自动标记开启时有意义，通常等于与相邻
///   item 的重叠量）；
/// - 倒数第 3 个参数（0 基索引 len-3）为 1 时表示「自动交叉淡化」生效
///   （见 reaper_fade_is_auto）。
///
/// ⚠️ 本函数不再把“有效长度”覆写到索引 1（旧实现会据此丢失手动长度）。
/// 要取手动 / 自动 / 有效长度，请分别使用：
/// reaper_fade_manual_length_sec / reaper_fade_auto_length_sec /
/// reaper_fade_effective_length_sec。
fn parse_fade_array(tokens: &[&str]) -> Vec<f64> {
    parse_double_array(tokens)
}

/// 判断某条 FADEIN/FADEOUT 是否被 REAPER 标记为「自动（交叉淡化）淡化」。
///
/// REAPER 在 fade 数组“倒数第 3 个参数”（0 基索引 len-3，与 parse_fade_array
/// 使用同一 selector）写入标记：为 1 时该淡化由 REAPER 自动生成并跟踪与相邻
/// item 的重叠量（如拖动重叠形成的自动交叉淡化），其真实长度位于第 3 个参数
/// （索引 2）；为 0 时是普通手动淡化（长度位于第 2 个参数，索引 1）。
pub fn reaper_fade_is_auto(values: &[f64]) -> bool {
    if values.len() < 4 {
        return false;
    }
    let selector_idx = values.len().saturating_sub(3);
    values.get(selector_idx).copied().unwrap_or(0.0).round() as i32 == 1
}

/// 读取 fade 数组中的“手动淡化长度”（索引 1）。
///
/// REAPER 的 fade 数组中索引 1 始终保存用户手动设置/持久保存的淡化长度；
/// 即使当前有自动交叉淡化生效，这个手动值也会被保留，供自动淡化移除后恢复。
pub fn reaper_fade_manual_length_sec(values: &[f64]) -> f64 {
    if values.len() >= 2 {
        values[1]
    } else {
        values.first().copied().unwrap_or(0.0)
    }
    .max(0.0)
}

/// 读取 fade 数组中的“自动交叉淡化长度”（索引 2；仅自动标记开启时有意义）。
///
/// 非自动淡化返回 0；自动淡化时索引 2 通常是该 item 与相邻 item 的重叠量。
pub fn reaper_fade_auto_length_sec(values: &[f64]) -> f64 {
    if reaper_fade_is_auto(values) {
        values.get(2).copied().unwrap_or(0.0).max(0.0)
    } else {
        0.0
    }
}

/// 计算 fade 数组的“有效淡化长度”。
///
/// 与应用模型一致：自动交叉淡化生效（>0）时取自动长度，否则取手动长度。
/// 用于淡化的存在性判断（take vs item 优先级、音量包络兜底等）。
pub fn reaper_fade_effective_length_sec(values: &[f64]) -> f64 {
    let auto = reaper_fade_auto_length_sec(values);
    if auto > 1e-9 {
        auto
    } else {
        reaper_fade_manual_length_sec(values)
    }
}

/// 读取 fade 数组的 REAPER 浮点形状 id 与曲率（D_FADEINDIR，[-1,1]）。
///
/// - 索引 0：形状（浮点；整数 0..6 为标准七预设，小数变体如 1.1/5.1
///   是官方开发者帖确认的扩展编码 —— 1.1→内部形状 7（等功率）、
///   5.1→内部形状 6（锐利 S 曲线）），原样透传；
/// - 索引 1：手动淡化长度；索引 2：自动交叉淡化长度（见上方辅助函数）；
/// - 索引 3：形状镜像值（与索引 0 同步写出，如 `FADEIN 5.1 … 0 5.1 …`），
///   解析时仅作索引 0 缺失时的回退；
/// - 倒数第 3 位：自动交叉淡化 selector（见 reaper_fade_is_auto）；
/// - 索引 5：曲率 D_FADEINDIR ∈ [-1, 1]（用户示例工程里淡入 0.25 /
///   淡出 0.35 落在此槽）；
/// - 索引 6：第二曲率参数 D_FADEINDIR2（语义未公开，暂不消费）。
pub fn reaper_fade_shape_dir(values: &[f64]) -> (f64, f64) {
    let mut shape = values.first().copied().unwrap_or(0.0);
    if values.len() >= 4 && (shape == 0.0 || !shape.is_finite()) {
        // 索引 0 缺失或异常时回退到镜像槽。
        shape = values[3];
    }
    if !shape.is_finite() {
        shape = 0.0;
    }
    let dir = values.get(5).copied().unwrap_or(0.0);
    (shape, dir.clamp(-1.0, 1.0))
}

/// 解析可能带引号的路径字符串
fn parse_path_string(tokens: &[&str]) -> String {
    if tokens.len() < 2 {
        return String::new();
    }
    let mut result = String::new();
    for i in 1..tokens.len() {
        if !result.is_empty() {
            result.push(' ');
        }
        result.push_str(tokens[i]);
        if tokens[i].ends_with('"') {
            break;
        }
    }
    result.trim().trim_matches('"').to_string()
}

/// 解析 SM 行中以 "+" 分隔的 stretch marker 数组
fn parse_stretch_markers(tokens: &[&str]) -> Vec<ReaperStretchMarker> {
    let mut markers = Vec::new();
    let mut buffer: Vec<f64> = Vec::new();

    for i in 1..tokens.len() {
        if tokens[i] == "+" {
            if buffer.len() >= 2 {
                markers.push(ReaperStretchMarker {
                    offset: buffer[0],
                    position: buffer[1],
                    velocity_change: if buffer.len() > 2 { buffer[2] } else { 0.0 },
                });
            }
            buffer.clear();
        } else {
            buffer.push(parse_double(tokens[i]));
        }
    }
    if buffer.len() >= 2 {
        markers.push(ReaperStretchMarker {
            offset: buffer[0],
            position: buffer[1],
            velocity_change: if buffer.len() > 2 { buffer[2] } else { 0.0 },
        });
    }

    markers
}

// ─── 公开解析 API ───

/// 解析 Reaper 工程文件（.rpp）
pub fn parse_rpp_file(path: &Path) -> Result<ReaperData, String> {
    let data = std::fs::read(path).map_err(|e| format!("Failed to read file: {}", e))?;
    let mut result = parse_bytes(&data)?;

    // 更新文件路径（将相对路径拼接为绝对路径）
    if let Some(folder) = path.parent() {
        update_source_paths(&mut result, folder);
    }

    Ok(result)
}

/// 解析 Reaper 剪贴板数据（字节数组，使用 \0 分隔）
pub fn parse_clipboard_bytes(data: &[u8]) -> Result<ReaperData, String> {
    parse_bytes(data)
}

/// 通用解析函数
fn parse_bytes(data: &[u8]) -> Result<ReaperData, String> {
    let lines = split_lines(data);
    if lines.is_empty() {
        return Err("Empty data".into());
    }
    let root_block = parse_blocks(&lines);
    Ok(parse_data_block(&root_block))
}

fn update_source_paths(data: &mut ReaperData, folder: &Path) {
    for track in &mut data.tracks {
        for item in &mut track.items {
            if let Some(ref mut src) = item.default_take.source {
                src.update_full_path(folder);
            }
            for take in &mut item.takes {
                if let Some(ref mut src) = take.source {
                    src.update_full_path(folder);
                }
            }
        }
    }
}

// ─── 块到数据结构的转换 ───

fn parse_data_block(block: &Block) -> ReaperData {
    let mut data = ReaperData::default();
    let mut current_track: Option<ReaperTrack> = None;
    let mut current_track_has_content = false;
    let mut cumulative_track_offset: usize = 0;
    let mut pending_offset: usize = 0;

    // 扫描当前块的直接行，提取 TEMPO
    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        if tokens[0].to_uppercase() == "TEMPO" && tokens.len() >= 4 {
            data.tempo = Some(ReaperTempo {
                bpm: parse_double(&tokens[1]),
                beats_per_bar: tokens[2].parse::<u32>().unwrap_or(4),
                beat_note: tokens[3].parse::<u32>().unwrap_or(4),
            });
        }
    }

    for child in &block.children {
        let block_type = child.block_type();

        if block_type.as_deref() == Some("TRACK") {
            if let Some(t) = current_track.take() {
                if current_track_has_content {
                    data.track_offsets.push(pending_offset);
                    data.tracks.push(t);
                }
            }
            let track = parse_track_block(child);
            data.is_track_data = true;
            data.track_offsets.push(cumulative_track_offset);
            data.tracks.push(track);
            cumulative_track_offset += 1;
            current_track = None;
            current_track_has_content = false;
            continue;
        }

        if block_type.as_deref() == Some("ITEM") {
            let item = parse_item_block(child);
            if current_track.is_none() {
                pending_offset = cumulative_track_offset;
                current_track = Some(ReaperTrack::default());
            }
            current_track.as_mut().unwrap().items.push(item);
            current_track_has_content = true;
            continue;
        }

        if block_type.as_deref() == Some("TEMPOENVEX") {
            data.tempo_envelope = Some(parse_tempo_envelope_block(child));
            continue;
        }

        if let Some(ref bt) = block_type {
            if is_envelope_type(bt) {
                let env = parse_envelope_block(child);
                if current_track.is_none() {
                    pending_offset = cumulative_track_offset;
                    current_track = Some(ReaperTrack::default());
                }
                current_track.as_mut().unwrap().envelopes.push(env);
                current_track_has_content = true;
                continue;
            }
        }

        // TRACKSKIP
        if child
            .lines
            .first()
            .map(|l| l.starts_with("TRACKSKIP"))
            .unwrap_or(false)
        {
            if let Some(t) = current_track.take() {
                if current_track_has_content {
                    data.track_offsets.push(pending_offset);
                    data.tracks.push(t);
                }
            }
            // 解析跳过的轨道数（TRACKSKIP N ...）
            let skip_n = child
                .lines
                .first()
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(1);
            cumulative_track_offset += skip_n;
            pending_offset = cumulative_track_offset;
            current_track = Some(ReaperTrack::default());
            current_track_has_content = false;
        }
    }

    if let Some(t) = current_track {
        if current_track_has_content {
            data.track_offsets.push(pending_offset);
            data.tracks.push(t);
        }
    }

    // 如果顶层没有 track/item，尝试递归查找
    if data.tracks.is_empty() {
        for child in &block.children {
            let nested = parse_data_block(child);
            if !nested.tracks.is_empty() {
                return nested;
            }
        }
    }

    data
}

fn parse_track_block(block: &Block) -> ReaperTrack {
    let mut track = ReaperTrack::default();

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].to_uppercase().as_str() {
            "NAME" => track.name = parse_path_string(&tokens),
            "VOLPAN" => track.vol_pan = parse_double_array(&tokens),
            "MUTESOLO" => track.mute_solo = parse_int_array(&tokens),
            "IPHASE" if tokens.len() >= 2 => track.iphase = parse_double(&tokens[1]) != 0.0,
            "ISBUS" => track.isbus = parse_int_array(&tokens),
            _ => {}
        }
    }

    for child in &block.children {
        let block_type = child.block_type();
        if block_type.as_deref() == Some("ITEM") {
            track.items.push(parse_item_block(child));
        } else if let Some(ref bt) = block_type {
            if is_envelope_type(bt) {
                track.envelopes.push(parse_envelope_block(child));
            }
        }
    }

    track
}

fn parse_item_block(block: &Block) -> ReaperItem {
    let mut item = ReaperItem::default();
    let mut raw_markers: Vec<ReaperStretchMarker> = Vec::new();
    let mut current_take_is_default = true;
    let has_take_blocks = block
        .children
        .iter()
        .any(|child| child.block_type().as_deref() == Some("TAKE"));

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        let key = tokens[0].to_uppercase();
        match key.as_str() {
            "POSITION" if tokens.len() >= 2 => item.position = parse_double(&tokens[1]),
            "SNAPOFFS" if tokens.len() >= 2 => item.snap_offs = parse_double(&tokens[1]),
            "LENGTH" if tokens.len() >= 2 => item.length = parse_double(&tokens[1]),
            "LOOP" if tokens.len() >= 2 => {
                item.is_loop = parse_bool(&tokens[1]);
                item.has_loop_token = true;
            }
            "ALLTAKES" if tokens.len() >= 2 => item.all_takes = parse_bool(&tokens[1]),
            "FADEIN" => item.fade_in = parse_fade_array(&tokens),
            "FADEOUT" => item.fade_out = parse_fade_array(&tokens),
            "MUTE" => item.mute = parse_int_array(&tokens),
            "SEL" if tokens.len() >= 2 => item.selected = parse_bool(&tokens[1]),
            "SM" => {
                raw_markers.extend(parse_stretch_markers(&tokens));
            }
            "TAKE" => {
                current_take_is_default = false;
                if !has_take_blocks {
                    let sel = tokens.len() > 1 && tokens[1].eq_ignore_ascii_case("SEL");
                    item.takes.push(ReaperTake {
                        selected: sel,
                        ..ReaperTake::default()
                    });
                }
            }
            "NAME" => {
                let name = parse_path_string(&tokens);
                if let Some(take) = current_take_mut(&mut item, current_take_is_default) {
                    take.name = name;
                }
            }
            "VOLPAN" | "TAKEVOLPAN" => {
                let arr = parse_double_array(&tokens);
                if let Some(take) = current_take_mut(&mut item, current_take_is_default) {
                    take.vol_pan = arr;
                }
            }
            "SOFFS" if tokens.len() >= 2 => {
                let v = parse_double(&tokens[1]);
                if let Some(take) = current_take_mut(&mut item, current_take_is_default) {
                    take.s_offs = v;
                }
            }
            "PLAYRATE" => {
                let arr = parse_double_array(&tokens);
                if let Some(take) = current_take_mut(&mut item, current_take_is_default) {
                    take.play_rate = arr;
                }
            }
            "CHANMODE" if tokens.len() >= 2 => {
                let v = parse_int(&tokens[1]);
                if let Some(take) = current_take_mut(&mut item, current_take_is_default) {
                    take.chan_mode = v;
                }
            }
            "GROUP" if tokens.len() >= 2 => {
                let gid = parse_int(&tokens[1]);
                if gid > 0 {
                    item.group_id = Some(gid);
                }
            }
            _ => {}
        }
    }

    // 将 stretch markers 转换为 stretch segments（存储在 item 上）
    item.stretch_markers = raw_markers;

    // 处理 SOURCE 子块：按顺序分配给 default_take 和各 take
    let mut source_idx: isize = -1;
    let mut take_envelopes: Vec<Vec<ReaperEnvelope>> = Vec::new();
    for child in &block.children {
        let block_type = child.block_type();
        if block_type.as_deref() == Some("TAKE") {
            let (take, take_envs) = parse_take_block(child);
            item.takes.push(take);
            take_envelopes.push(take_envs);
        } else if block_type.as_deref() == Some("SOURCE") {
            let source = parse_source_block(child);
            source_idx += 1;
            if source_idx == 0 {
                item.default_take.source = Some(source);
            } else {
                let take_idx = (source_idx - 1) as usize;
                if take_idx < item.takes.len() {
                    item.takes[take_idx].source = Some(source);
                }
            }
        } else if let Some(ref bt) = block_type {
            if is_envelope_type(bt) {
                item.envelopes.push(parse_envelope_block(child));
            }
        }
    }

    if !take_envelopes.is_empty() {
        let active_take_idx = item
            .takes
            .iter()
            .position(|take| take.selected)
            .unwrap_or(0);
        if let Some(envs) = take_envelopes.get(active_take_idx) {
            item.envelopes.extend(envs.iter().cloned());
        }
    }

    item
}

fn current_take_mut<'a>(item: &'a mut ReaperItem, is_default: bool) -> Option<&'a mut ReaperTake> {
    if is_default {
        Some(&mut item.default_take)
    } else {
        item.takes.last_mut()
    }
}

fn parse_take_block(block: &Block) -> (ReaperTake, Vec<ReaperEnvelope>) {
    let mut take = ReaperTake::default();
    let mut envelopes: Vec<ReaperEnvelope> = Vec::new();

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].to_uppercase().as_str() {
            "<TAKE" => {
                take.selected = tokens
                    .iter()
                    .skip(1)
                    .any(|tok| tok.eq_ignore_ascii_case("SEL"));
            }
            "SEL" if tokens.len() >= 2 => {
                take.selected = parse_bool(tokens[1]);
            }
            "NAME" => {
                take.name = parse_path_string(&tokens);
            }
            "VOLPAN" | "TAKEVOLPAN" => {
                take.vol_pan = parse_double_array(&tokens);
            }
            "FADEIN" => {
                take.fade_in = parse_fade_array(&tokens);
            }
            "FADEOUT" => {
                take.fade_out = parse_fade_array(&tokens);
            }
            "SOFFS" if tokens.len() >= 2 => {
                take.s_offs = parse_double(tokens[1]);
            }
            "PLAYRATE" => {
                take.play_rate = parse_double_array(&tokens);
            }
            "CHANMODE" if tokens.len() >= 2 => {
                take.chan_mode = parse_int(tokens[1]);
            }
            _ => {}
        }
    }

    for child in &block.children {
        let block_type = child.block_type();
        if block_type.as_deref() == Some("SOURCE") {
            take.source = Some(parse_source_block(child));
        } else if let Some(ref bt) = block_type {
            if is_envelope_type(bt) {
                envelopes.push(parse_envelope_block(child));
            }
        }
    }

    (take, envelopes)
}

fn parse_source_block(block: &Block) -> ReaperSource {
    let mut source = ReaperSource::new();
    let mut midi_events: Vec<ReaperMidiEvent> = Vec::new();
    let mut midi_ticks_per_qn: u32 = 960;
    let mut midi_igntempo: Option<ReaperIgnTempo> = None;

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].to_uppercase().as_str() {
            "<SOURCE" if tokens.len() > 1 => {
                source.source_type = tokens[1].to_string();
            }
            "FILE" => {
                source.file_path = parse_path_string(&tokens);
            }
            "MODE" if tokens.len() >= 2 => {
                source.section_mode = parse_int(tokens[1]);
            }
            "STARTPOS" if tokens.len() >= 2 => {
                source.section_start_sec = Some(parse_double(tokens[1]));
            }
            "LENGTH" if tokens.len() >= 2 => {
                source.section_length_sec = Some(parse_double(tokens[1]));
            }
            // ─── MIDI 源 ───
            "HASDATA" if tokens.len() >= 3 => {
                midi_ticks_per_qn = tokens[2].parse::<u32>().unwrap_or(960);
            }
            "IGNTEMPO" if tokens.len() >= 5 => {
                midi_igntempo = Some(ReaperIgnTempo {
                    ignore_project: parse_bool(tokens[1]),
                    tempo: parse_double(tokens[2]),
                    beats: tokens[3].parse::<u32>().unwrap_or(4),
                    beat_note: tokens[4].parse::<u32>().unwrap_or(4),
                });
            }
            "E" | "e" if tokens.len() >= 4 => {
                let tick_offset = tokens[1].parse::<u64>().unwrap_or(0);
                midi_events.push(ReaperMidiEvent {
                    tick_offset,
                    status: parse_hex_byte(tokens[2]),
                    data1: parse_hex_byte(tokens[3]),
                    data2: if tokens.len() >= 5 {
                        parse_hex_byte(tokens[4])
                    } else {
                        0
                    },
                });
            }
            "X" | "x" if tokens.len() >= 6 => {
                let hi = tokens[1].parse::<u64>().unwrap_or(0);
                let lo = tokens[2].parse::<u64>().unwrap_or(0);
                let tick_offset = (hi << 32) | lo;
                midi_events.push(ReaperMidiEvent {
                    tick_offset,
                    status: parse_hex_byte(tokens[3]),
                    data1: parse_hex_byte(tokens[4]),
                    data2: parse_hex_byte(tokens[5]),
                });
            }
            _ => {}
        }
    }

    // 组装 MIDI 源数据
    if source.source_type.eq_ignore_ascii_case("MIDI") && !midi_events.is_empty() {
        source.midi_source = Some(ReaperMidiSourceData {
            ticks_per_qn: midi_ticks_per_qn,
            events: midi_events,
            igntempo: midi_igntempo,
        });
    }

    // 处理 SECTION 类型的嵌套 SOURCE
    if source.source_type.eq_ignore_ascii_case("SECTION") {
        for child in &block.children {
            if child.block_type().as_deref() == Some("SOURCE") {
                let inner = parse_source_block(child);
                source.file_path = inner.file_path;
                // MODE 信息来自外层 SECTION；仅补齐内部 SOURCE 的其它字段。
                if source.section_start_sec.is_none() {
                    source.section_start_sec = inner.section_start_sec;
                }
                if source.section_length_sec.is_none() {
                    source.section_length_sec = inner.section_length_sec;
                }
                break;
            }
        }
    }

    source
}

fn parse_envelope_block(block: &Block) -> ReaperEnvelope {
    let mut env = ReaperEnvelope::default();
    env.env_type = block.block_type().unwrap_or_default();

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        match tokens[0].to_uppercase().as_str() {
            "<ENVSEG" if tokens.len() > 1 => env.env_type = tokens[1].to_string(),
            "ACT" => env.act = parse_int_array(&tokens),
            "SEG_RANGE" => env.seg_range = Some(parse_double_array(&tokens)),
            "PT" => env.points.push(parse_double_array(&tokens)),
            _ => {}
        }
    }

    env
}

fn parse_tempo_envelope_block(block: &Block) -> ReaperTempoEnvelope {
    let mut env = ReaperTempoEnvelope { points: Vec::new() };

    for line in &block.lines {
        let tokens = split_tokens(line);
        if tokens.is_empty() {
            continue;
        }
        if tokens[0].to_uppercase() != "PT" || tokens.len() < 4 {
            continue;
        }
        let position_sec = parse_double(&tokens[1]).max(0.0);
        let bpm = parse_double(&tokens[2]);
        let shape = parse_int(&tokens[3]);
        // 拍号打包在第 4 个值（slowcurv）中；无第 4 个值时继承前一点的拍号。
        let (numerator, denominator) = if tokens.len() >= 5 {
            match parse_tempo_env_time_signature(parse_double(&tokens[4])) {
                Some(ts) => (Some(ts.0), Some(ts.1)),
                None => (None, None),
            }
        } else {
            (None, None)
        };
        env.points.push(ReaperTempoEnvelopePoint {
            position_sec,
            bpm,
            shape,
            numerator,
            denominator,
        });
    }

    env
}

fn is_envelope_type(s: &str) -> bool {
    ENVELOPE_TYPES.iter().any(|&e| e.eq_ignore_ascii_case(s))
}

// ---------------------------------------------------------------------------
// REAPER clipboard serialization
// ---------------------------------------------------------------------------

fn format_reaper_f64(value: f64) -> String {
    if !value.is_finite() {
        return "0".to_string();
    }
    if value == 0.0 {
        return "0".to_string();
    }
    let mut text = format!("{:.12}", value);
    while text.contains('.') && text.ends_with('0') {
        text.pop();
    }
    if text.ends_with('.') {
        text.pop();
    }
    text
}

fn quote_reaper_string(value: &str) -> String {
    // Double quotes inside a path are sanitized because REAPERMedia has no
    // reliable escape sequence for them. Backslashes are preserved.
    let escaped = value.replace('"', "'");
    format!("\"{}\"", escaped)
}

fn push_reaper_token(out: &mut Vec<u8>, token: String) {
    out.extend_from_slice(token.as_bytes());
    out.push(0);
}

fn push_reaper_array(out: &mut Vec<u8>, key: &str, values: &[f64]) {
    if values.is_empty() {
        return;
    }
    let text = values
        .iter()
        .map(|value| format_reaper_f64(*value))
        .collect::<Vec<_>>()
        .join(" ");
    push_reaper_token(out, format!("{key} {text}"));
}

fn push_reaper_int_array(out: &mut Vec<u8>, key: &str, values: &[i32]) {
    if values.is_empty() {
        return;
    }
    let text = values
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    push_reaper_token(out, format!("{key} {text}"));
}

fn push_reaper_source(out: &mut Vec<u8>, source: &ReaperSource) {
    let source_type = source.source_type.to_uppercase();

    if source_type == "SECTION" {
        push_reaper_token(out, "<SOURCE SECTION".to_string());
        if let Some(length) = source.section_length_sec {
            push_reaper_token(out, format!("LENGTH {}", format_reaper_f64(length)));
        }
        push_reaper_token(out, format!("MODE {}", source.section_mode));
        if let Some(start) = source.section_start_sec {
            push_reaper_token(out, format!("STARTPOS {}", format_reaper_f64(start)));
        }
        push_reaper_token(out, "OVERLAP 0.01".to_string());

        let mut inner = source.clone();
        inner.source_type = "WAVE".to_string();
        inner.section_mode = 0;
        inner.section_start_sec = None;
        inner.section_length_sec = None;
        inner.midi_source = None;
        push_reaper_source(out, &inner);
        push_reaper_token(out, ">".to_string());
        return;
    }

    if source_type == "MIDI" {
        push_reaper_token(out, "<SOURCE MIDI".to_string());
        let ppq = source
            .midi_source
            .as_ref()
            .map(|midi| midi.ticks_per_qn.max(1))
            .unwrap_or(960);
        push_reaper_token(out, format!("HASDATA 1 {ppq} QN"));
        push_reaper_token(out, "CCINTERP 32".to_string());
        if let Some(igntempo) = source
            .midi_source
            .as_ref()
            .and_then(|midi| midi.igntempo.as_ref())
        {
            push_reaper_token(
                out,
                format!(
                    "IGNTEMPO {} {} {} {}",
                    if igntempo.ignore_project { 1 } else { 0 },
                    format_reaper_f64(igntempo.tempo),
                    igntempo.beats,
                    igntempo.beat_note,
                ),
            );
        } else {
            push_reaper_token(out, "IGNTEMPO 0 120 4 4".to_string());
        }
        if let Some(midi) = source.midi_source.as_ref() {
            for event in &midi.events {
                push_reaper_token(
                    out,
                    format!(
                        "E {} {:02X} {:02X} {:02X}",
                        event.tick_offset, event.status, event.data1, event.data2
                    ),
                );
            }
        }
        push_reaper_token(out, ">".to_string());
        return;
    }

    push_reaper_token(
        out,
        format!(
            "<SOURCE {}",
            if source_type.is_empty() {
                "WAVE"
            } else {
                &source_type
            }
        ),
    );
    push_reaper_token(
        out,
        format!("FILE {}", quote_reaper_string(&source.file_path)),
    );
    push_reaper_token(out, ">".to_string());
}

fn push_reaper_take(out: &mut Vec<u8>, take: &ReaperTake, is_item_default: bool) {
    if !is_item_default {
        push_reaper_token(
            out,
            if take.selected {
                "TAKE SEL".to_string()
            } else {
                "TAKE".to_string()
            },
        );
    }
    push_reaper_token(out, format!("NAME {}", quote_reaper_string(&take.name)));
    push_reaper_array(
        out,
        if is_item_default {
            "VOLPAN"
        } else {
            "TAKEVOLPAN"
        },
        &take.vol_pan,
    );
    if !is_item_default {
        push_reaper_array(out, "FADEIN", &take.fade_in);
        push_reaper_array(out, "FADEOUT", &take.fade_out);
    }
    push_reaper_token(out, format!("SOFFS {}", format_reaper_f64(take.s_offs)));
    push_reaper_array(out, "PLAYRATE", &take.play_rate);
    push_reaper_token(out, format!("CHANMODE {}", take.chan_mode));
    if let Some(source) = take.source.as_ref() {
        push_reaper_source(out, source);
    }
}

fn push_reaper_item(out: &mut Vec<u8>, item: &ReaperItem) {
    push_reaper_token(out, "<ITEM".to_string());
    push_reaper_token(
        out,
        format!("POSITION {}", format_reaper_f64(item.position)),
    );
    push_reaper_token(
        out,
        format!("SNAPOFFS {}", format_reaper_f64(item.snap_offs)),
    );
    push_reaper_token(out, format!("LENGTH {}", format_reaper_f64(item.length)));
    push_reaper_token(out, format!("LOOP {}", if item.is_loop { 1 } else { 0 }));
    push_reaper_token(
        out,
        format!("ALLTAKES {}", if item.all_takes { 1 } else { 0 }),
    );
    push_reaper_array(out, "FADEIN", &item.fade_in);
    push_reaper_array(out, "FADEOUT", &item.fade_out);
    push_reaper_int_array(out, "MUTE", &item.mute);
    push_reaper_token(out, format!("SEL {}", if item.selected { 1 } else { 0 }));
    push_reaper_take(out, &item.default_take, true);
    for take in &item.takes {
        push_reaper_take(out, take, false);
    }
    for envelope in &item.envelopes {
        push_reaper_token(out, format!("<{}", envelope.env_type));
        if let Some(range) = envelope.seg_range.as_ref() {
            push_reaper_array(out, "SEG_RANGE", range);
        }
        for point in &envelope.points {
            push_reaper_array(out, "PT", point);
        }
        push_reaper_token(out, ">".to_string());
    }
    push_reaper_token(out, ">".to_string());
}

/// Serialize parsed REAPER data into the REAPERMedia clipboard byte format.
/// `as_track_data = false` writes ITEM + TRACKSKIP blocks; `true` writes
/// `<TRACK ...>` wrapped track data.
pub fn serialize_reaper_clipboard(data: &ReaperData, as_track_data: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(4096);

    let offsets = if data.track_offsets.len() == data.tracks.len() {
        data.track_offsets.clone()
    } else {
        (0..data.tracks.len()).collect::<Vec<_>>()
    };

    for (index, track) in data.tracks.iter().enumerate() {
        if as_track_data {
            push_reaper_token(&mut out, "<TRACK".to_string());
            push_reaper_token(
                &mut out,
                format!("NAME {}", quote_reaper_string(&track.name)),
            );
            push_reaper_array(&mut out, "VOLPAN", &track.vol_pan);
            push_reaper_int_array(&mut out, "MUTESOLO", &track.mute_solo);
            push_reaper_token(
                &mut out,
                format!("IPHASE {}", if track.iphase { 1 } else { 0 }),
            );
        }

        for item in &track.items {
            push_reaper_item(&mut out, item);
        }
        for envelope in &track.envelopes {
            push_reaper_token(&mut out, format!("<{}", envelope.env_type));
            for point in &envelope.points {
                push_reaper_array(&mut out, "PT", point);
            }
            push_reaper_token(&mut out, ">".to_string());
        }

        if as_track_data {
            push_reaper_token(&mut out, ">".to_string());
        } else {
            let current = offsets.get(index).copied().unwrap_or(0);
            let next = offsets
                .get(index + 1)
                .copied()
                .unwrap_or(current.saturating_add(1));
            let skip = next.saturating_sub(current).max(1);
            push_reaper_token(&mut out, format!("TRACKSKIP {skip} 1"));
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_tempo_env_time_signature() {
        // 4/4 → 262 * 1000 + (145 + 4 - 1) = 262148
        assert_eq!(parse_tempo_env_time_signature(262148.0), Some((4, 4)));
        // 3/4 → 262 * 1000 + (145 + 3 - 1) = 262147
        assert_eq!(parse_tempo_env_time_signature(262147.0), Some((3, 4)));
        // 6/8 → 524 * 1000 + (289 + 6 - 1) = 524294
        assert_eq!(parse_tempo_env_time_signature(524294.0), Some((6, 8)));
        // 非法值
        assert_eq!(parse_tempo_env_time_signature(1.0), None);
    }

    #[test]
    fn parses_tempo_envelope_block_with_time_signatures() {
        // TEMPOENVEX 的 PT 行：`PT 位置 BPM 形状 [slowcurv(打包拍号) …]`，
        // 拍号打包值 262148 = 4/4、262147 = 3/4；块内其余行仅用于确认被忽略。
        let block_text = "<TEMPOENVEX\n\
EGUID {2A001931-7D16-4259-A7F4-7ED6A60C53C6}\n\
ACT 1 -1\n\
VIS 1 0 1\n\
LANEHEIGHT 0 0\n\
ARM 1\n\
DEFSHAPE 1 -1 -1\n\
PT 0 120 1 262148 0 1 0 \"\" 0 169 0 ABBB\n\
PT 0.5 140 1\n\
PT 1 160 1\n\
PT 4 100 1 262147 0 1 0 \"\" 0 41 0 ABB\n\
>";
        let lines: Vec<String> = block_text.lines().map(|s| s.to_string()).collect();
        let root = parse_blocks(&lines);
        // parse_blocks 返回根块；TEMPOENVEX 是其子块。
        let tempo_block = root
            .children
            .first()
            .expect("TEMPOENVEX should parse as a child block");
        assert_eq!(tempo_block.block_type().as_deref(), Some("TEMPOENVEX"));
        let envelope = parse_tempo_envelope_block(tempo_block);

        assert_eq!(envelope.points.len(), 4);
        // 首点：120 BPM、4/4。
        let first = &envelope.points[0];
        assert_eq!(first.position_sec, 0.0);
        assert_eq!(first.bpm, 120.0);
        assert_eq!(first.shape, 1);
        assert_eq!(first.numerator, Some(4));
        assert_eq!(first.denominator, Some(4));
        // 普通点：无拍号信息（继承前一点）。
        let second = &envelope.points[1];
        assert_eq!(second.bpm, 140.0);
        assert_eq!(second.numerator, None);
        // 末点：100 BPM、3/4（ABB 拍型）。
        let last = &envelope.points[3];
        assert_eq!(last.position_sec, 4.0);
        assert_eq!(last.bpm, 100.0);
        assert_eq!(last.numerator, Some(3));
        assert_eq!(last.denominator, Some(4));
    }

    #[test]
    fn distinguishes_auto_crossfade_fades_from_manual_fades() {
        // REAPER 的 fade 数组：索引 1 = 手动长度，索引 2 = 自动交叉淡化长度，
        // 倒数第 3 个值为自动 selector。
        // - 手动淡化：selector（索引 4）为 0，长度在手动的索引 1、自动为 0。
        let manual_in = parse_fade_array(&["FADEIN", "1", "0.01", "0", "1", "0", "0", "0"]);
        assert!(!reaper_fade_is_auto(&manual_in));
        assert_eq!(reaper_fade_manual_length_sec(&manual_in), 0.01);
        assert_eq!(reaper_fade_auto_length_sec(&manual_in), 0.0);

        let manual_long = parse_fade_array(&["FADEIN", "1", "0.67", "0", "1", "0", "0", "0"]);
        assert!(!reaper_fade_is_auto(&manual_long));
        assert_eq!(reaper_fade_manual_length_sec(&manual_long), 0.67);
        assert_eq!(
            reaper_fade_effective_length_sec(&manual_long),
            reaper_fade_manual_length_sec(&manual_long),
        );

        // - 自动交叉淡化：selector（索引 4）为 1；索引 1 = 手动长度（保留），
        //   索引 2 = 自动长度（= 重叠量）。有效长度取自动。
        let auto_out = parse_fade_array(&["FADEOUT", "1.1", "0.01", "0.02", "1", "1", "0", "0"]);
        assert!(reaper_fade_is_auto(&auto_out));
        assert_eq!(auto_out[1], 0.01); // 手动长度不被自动值覆盖
        assert_eq!(reaper_fade_manual_length_sec(&auto_out), 0.01);
        assert_eq!(reaper_fade_auto_length_sec(&auto_out), 0.02);
        assert_eq!(reaper_fade_effective_length_sec(&auto_out), 0.02);

        let auto_in = parse_fade_array(&["FADEIN", "1.1", "0.01", "0.02", "1", "1", "0", "0"]);
        assert!(reaper_fade_is_auto(&auto_in));
        assert_eq!(reaper_fade_manual_length_sec(&auto_in), 0.01);
        assert_eq!(reaper_fade_auto_length_sec(&auto_in), 0.02);

        // 重叠场景：Item A 手动淡出 0.67s → 重叠后自动交叉淡化 0.18s
        // （自动 = 重叠量），手动淡化被保留在索引 1。
        let a_fadeout = parse_fade_array(&["FADEOUT", "1.1", "0.67", "0.18", "1", "1", "0", "0"]);
        assert!(reaper_fade_is_auto(&a_fadeout));
        assert_eq!(reaper_fade_manual_length_sec(&a_fadeout), 0.67);
        assert_eq!(reaper_fade_auto_length_sec(&a_fadeout), 0.18);
        assert_eq!(reaper_fade_effective_length_sec(&a_fadeout), 0.18);

        let b_fadein = parse_fade_array(&["FADEIN", "1.1", "0", "0.18", "1", "1", "0", "0"]);
        assert!(reaper_fade_is_auto(&b_fadein));
        assert_eq!(reaper_fade_manual_length_sec(&b_fadein), 0.0);
        assert_eq!(reaper_fade_auto_length_sec(&b_fadein), 0.18);

        // 过短数组（<4 个值）不能把 shape 误当 selector / 自动标记。
        assert!(!reaper_fade_is_auto(&[1.0, 0.1]));
        assert!(!reaper_fade_is_auto(&[1.0, 0.1, 0.0]));
    }

    #[test]
    fn decodes_reaper7_shape_and_curvature_fields() {
        // 淡入曲率 0.25、淡出曲率 0.35，形状槽为整数 0。
        let fade_in = parse_fade_array(&["FADEIN", "0", "0.9", "0", "0", "0", "0.25", "0"]);
        let fade_out = parse_fade_array(&["FADEOUT", "0", "0.61", "0", "0", "0", "0.35", "0"]);
        assert_eq!(fade_in.len(), 7);
        assert!(!reaper_fade_is_auto(&fade_in));
        assert_eq!(reaper_fade_manual_length_sec(&fade_in), 0.9);
        let (in_shape, in_dir) = reaper_fade_shape_dir(&fade_in);
        assert_eq!(in_shape, 0.0);
        assert_eq!(in_dir, 0.25);
        let (out_shape, out_dir) = reaper_fade_shape_dir(&fade_out);
        assert_eq!(out_shape, 0.0);
        assert_eq!(out_dir, 0.35);

        // 小数变体形状 `5.1` 在索引 0 与镜像槽同步出现；
        // 镜像槽（索引 3）存的是取整后的基础形状（官方自动淡化样本中
        // 形状 1.1 对应镜像槽写 1），仅在索引 0 缺失时作为回退。
        let fractional = parse_fade_array(&["FADEIN", "5.1", "0.4", "0", "5.1", "0", "0", "0"]);
        let (shape, dir) = reaper_fade_shape_dir(&fractional);
        assert_eq!(shape, 5.1, "fractional shape passthrough");
        assert_eq!(dir, 0.0);

        let auto = parse_fade_array(&["FADEOUT", "1.1", "0.01", "0.02", "1", "1", "0", "0"]);
        let (auto_shape, auto_dir) = reaper_fade_shape_dir(&auto);
        assert_eq!(auto_shape, 1.1);
        assert_eq!(auto_dir, 0.0);

        // 索引 5 的曲率越界时夹紧到 [-1, 1]；短数组的曲率回退 0。
        let clamped = reaper_fade_shape_dir(&parse_fade_array(&[
            "FADEIN", "2", "0.01", "0", "2", "0", "3.7", "0",
        ]));
        assert_eq!(clamped.1, 1.0);
        let (short_shape, short_dir) =
            reaper_fade_shape_dir(&parse_fade_array(&["FADEIN", "4", "0.02"]));
        assert_eq!(short_shape, 4.0);
        assert_eq!(short_dir, 0.0);
    }

    #[test]
    fn exported_fade_values_are_understood_by_reaper_layout() {
        // 导出 → 解析往返：shape/dir/手动与自动长度各归其位。
        let values = super::super::reaper_export::fade_values(3.0, 0.42, 0.1, 0.25);
        assert_eq!(values.len(), 7);
        assert!(reaper_fade_is_auto(&values));
        assert!((reaper_fade_manual_length_sec(&values) - 0.1).abs() < 1e-12);
        assert!((reaper_fade_auto_length_sec(&values) - 0.25).abs() < 1e-12);
        let (shape, dir) = reaper_fade_shape_dir(&values);
        assert_eq!(shape, 3.0);
        assert!((dir - 0.42).abs() < 1e-12);

        // 手动淡化的 selector 为 0，形状小数变体原样写出、镜像槽取整。
        let manual = super::super::reaper_export::fade_values(5.1, -0.75, 0.33, 0.0);
        assert!(!reaper_fade_is_auto(&manual));
        assert_eq!(manual[3], 5.0, "mirror slot stores truncated base shape");
        let (shape, dir) = reaper_fade_shape_dir(&manual);
        assert!((shape - 5.1).abs() < 1e-12);
        assert!((dir - (-0.75)).abs() < 1e-12);
    }

    // 5 个标记：0/2 号带速率变化系数，其余缺省（段内线性）；首标记不在
    // item 起点（有前导段），末标记在窗口终点之前（有尾段）。
    // 段 1 速率 2 × (1 − vc 0.5) = 1，与前导段的基准速率衔接。
    const SM_TEST_TOKENS: &[&str] = &[
        "SM",
        "1", "1", "0.5", "+",
        "2", "3", "+",
        "4", "5", "0.25", "+",
        "6", "11", "+",
        "10", "12",
    ];

    #[test]
    fn parses_sm_groups_with_optional_rate_handles() {
        let markers = parse_stretch_markers(SM_TEST_TOKENS);
        assert_eq!(markers.len(), 5);
        assert_eq!(markers[0].offset, 1.0);
        assert_eq!(markers[0].position, 1.0);
        assert_eq!(markers[0].velocity_change, 0.5);
        // 无速率手柄的标记缺省 0（段内线性）。
        assert_eq!(markers[1].velocity_change, 0.0);
        assert_eq!(markers[2].offset, 4.0);
        assert_eq!(markers[2].position, 5.0);
        assert_eq!(markers[2].velocity_change, 0.25);
    }

    #[test]
    fn stretch_markers_not_at_item_start_cover_leading_and_trailing() {
        let markers = parse_stretch_markers(SM_TEST_TOKENS);
        // item：POSITION 0 / LENGTH 20 / PLAYRATE 1 / SOFFS 0。
        let segments = stretch_segments_full_cover(&markers, 0.0, 1.0, 20.0, None, false);
        assert_eq!(segments.len(), 6, "前导段 + 4 个标记段 + 尾段");

        // 前导段：item 开头到首标记，基准速率 1。
        let lead = &segments[0];
        assert_eq!(lead.offset_start, 0.0);
        assert_eq!(lead.offset_end, 1.0);
        assert_eq!(lead.velocity_average(), 1.0);

        // 标记段速率 = Δ源媒体 / Δtake媒体；段 1 源 [1,3] 摊在 [1,2]，速率 2。
        let second = &segments[1];
        assert_eq!(second.velocity_average(), 2.0);
        assert_eq!(second.offset_start, 1.0);
        // 速率变化系数展开为段内坡：2 × (1 − 0.5) = 1，与前导段速率衔接。
        assert_eq!(second.velocity_start, 1.0);

        // 尾段：末标记源位置沿基准速率外推到 item 末端。
        let tail = segments.last().unwrap();
        assert_eq!(tail.offset_start, 10.0);
        assert_eq!(tail.offset_end, 20.0);
        assert_eq!(tail.velocity_average(), 1.0);

        // 段链连续，且铺满整个 item 的 take 媒体时间。
        for pair in segments.windows(2) {
            assert_eq!(pair[1].offset_start, pair[0].offset_end);
        }
        let covered: f64 = segments.iter().map(|s| s.offset_length()).sum();
        assert_eq!(covered, 20.0);
    }

    #[test]
    fn stretch_marker_at_item_start_merges_with_base_anchor() {
        // 首标记钉在 item 开头（源位置 = SOFFS）：不产生前导段。
        let markers = vec![
            ReaperStretchMarker { offset: 0.0, position: 5.0, velocity_change: 0.0 },
            ReaperStretchMarker { offset: 4.0, position: 7.0, velocity_change: 0.0 },
        ];
        let segments = stretch_segments_full_cover(&markers, 5.0, 1.0, 10.0, None, false);
        assert_eq!(segments.len(), 2);
        // [0,4] 消耗源 [5,7]；尾段 [4,10] 基准速率外推到源 12。
        assert!((segments[0].offset_end - 4.0).abs() < 1e-12);
        assert!((segments[0].velocity_average() - 0.5).abs() < 1e-12);
        assert!((segments[1].velocity_average() - 1.0).abs() < 1e-12);
        assert!((segments[1].offset_end - 10.0).abs() < 1e-12);
    }

    #[test]
    fn single_stretch_marker_still_covers_full_item() {
        // 旧实现 markers.len() < 2 直接返回空表 → 单标记 item 完全忽略拉伸。
        let markers = vec![ReaperStretchMarker {
            offset: 3.0,
            position: 1.0,
            velocity_change: 0.0,
        }];
        let segments = stretch_segments_full_cover(&markers, 0.0, 1.0, 10.0, None, false);
        assert_eq!(segments.len(), 2);
        // 前导段：源 [0,1] 摊在 [0,3]，斜率 1/3（拖动首标记 = 拉伸前导区）。
        assert!((segments[0].velocity_average() - 1.0 / 3.0).abs() < 1e-12);
        assert!((segments[1].velocity_average() - 1.0).abs() < 1e-12);
        let covered: f64 = segments.iter().map(|s| s.offset_length()).sum();
        assert!((covered - 10.0).abs() < 1e-9);
    }

    #[test]
    fn stretch_full_cover_without_markers_or_length_is_empty() {
        assert!(stretch_segments_full_cover(&[], 0.0, 1.0, 10.0, None, false).is_empty());
        assert!(stretch_segments_full_cover(
            &[ReaperStretchMarker { offset: 1.0, position: 1.0, velocity_change: 0.0 }],
            0.0,
            1.0,
            0.0,
            None,
            false,
        )
        .is_empty());
    }

    #[test]
    fn stretch_full_cover_offsets_scale_with_play_rate() {
        // take 媒体时间 = item 时间 × play_rate：item 4s、rate 2 → u_end = 8。
        // 标记在 item 1s（u=2）钉住源 2s 处：前导段斜率 1，尾段斜率 1。
        let markers = vec![ReaperStretchMarker {
            offset: 2.0,
            position: 2.0,
            velocity_change: 0.0,
        }];
        let segments = stretch_segments_full_cover(&markers, 0.0, 2.0, 4.0, None, false);
        assert_eq!(segments.len(), 2);
        assert!((segments[0].offset_end - 2.0).abs() < 1e-12);
        assert!((segments[0].velocity_average() - 1.0).abs() < 1e-12);
        assert!((segments[1].offset_end - 8.0).abs() < 1e-12);
        assert!((segments[1].velocity_average() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn stretch_full_cover_reversed_mirrors_source_windows() {
        // 倒放 take 的标记 position 处于 REAPER 预翻转的源坐标系
        // （s = 媒体全长 − 原始位置）；输出时用媒体全长把各段源窗口
        // 镜像回原始坐标（start ≤ end），速率坡保持不变。
        let markers = vec![
            ReaperStretchMarker { offset: 0.0, position: 5.0, velocity_change: 0.5 },
            ReaperStretchMarker { offset: 4.0, position: 9.0, velocity_change: 0.0 },
        ];
        let segments = stretch_segments_full_cover(&markers, 5.0, 1.0, 10.0, Some(20.0), true);
        assert_eq!(segments.len(), 2);
        // 段 1：take 源坐标 [5,9] → 原始坐标 [11,15]（20−9=11、20−5=15）。
        assert!((segments[0].offset_start - 0.0).abs() < 1e-12);
        assert!((segments[0].offset_end - 4.0).abs() < 1e-12);
        assert!((segments[0].src_start - 11.0).abs() < 1e-12);
        assert!((segments[0].src_end - 15.0).abs() < 1e-12);
        // 速率坡与坐标系无关：vc=0.5 展开为 [0.5, 1.5]。
        assert!((segments[0].velocity_start - 0.5).abs() < 1e-12);
        assert!((segments[0].velocity_end - 1.5).abs() < 1e-12);
        // 尾段：基准速率外推，镜像后 src [5,11]，与段 1 衔接
        // （倒放推进时源位置自 end 向 start 递减）。
        assert!((segments[1].offset_start - 4.0).abs() < 1e-12);
        assert!((segments[1].offset_end - 10.0).abs() < 1e-12);
        assert!((segments[1].src_start - 5.0).abs() < 1e-12);
        assert!((segments[1].src_end - 11.0).abs() < 1e-12);
        assert!((segments[1].velocity_average() - 1.0).abs() < 1e-12);

        // 倒放但缺少有效媒体全长时无法镜像 → 返回空表（调用方回退单 clip 路径）。
        assert!(stretch_segments_full_cover(&markers, 5.0, 1.0, 10.0, None, true).is_empty());
        assert!(stretch_segments_full_cover(&markers, 5.0, 1.0, 10.0, Some(0.0), true).is_empty());
    }
}
