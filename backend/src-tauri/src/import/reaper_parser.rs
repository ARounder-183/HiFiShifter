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
}

impl Default for ReaperItem {
    fn default() -> Self {
        Self {
            position: 0.0,
            snap_offs: 0.0,
            length: 0.0,
            is_loop: false,
            all_takes: false,
            fade_in: vec![0.0; 7],
            fade_out: vec![0.0; 7],
            mute: vec![0, 0],
            selected: false,
            envelopes: Vec::new(),
            takes: Vec::new(),
            default_take: ReaperTake::default(),
            stretch_markers: Vec::new(),
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
pub struct ReaperStretchMarker {
    pub offset: f64,
    pub position: f64,
    pub velocity_change: f64,
}

#[derive(Debug, Clone)]
pub struct ReaperStretchSegment {
    pub offset_start: f64,
    pub offset_end: f64,
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

pub fn stretch_segments_from_markers(markers: &[ReaperStretchMarker]) -> Vec<ReaperStretchSegment> {
    let mut segments = Vec::new();
    if markers.len() < 2 {
        return segments;
    }
    let mut current_start: Option<(f64, &ReaperStretchMarker)> = None;

    for marker in markers {
        if let Some((start_offset, last_marker)) = current_start {
            let offset_length = marker.offset - start_offset;
            if offset_length.abs() > 1e-12 {
                let velocity_average = (marker.position - last_marker.position) / offset_length;
                let velocity_half = last_marker.velocity_change * velocity_average;
                segments.push(ReaperStretchSegment {
                    offset_start: start_offset,
                    offset_end: marker.offset,
                    velocity_start: velocity_average - velocity_half,
                    velocity_end: velocity_average + velocity_half,
                });
            }
        }
        current_start = Some((marker.offset, marker));
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
pub struct ReaperTempoEnvelope {
    pub points: Vec<Vec<f64>>,
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
    s.parse::<f64>().unwrap_or(0.0)
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

/// 解析 FADEIN/FADEOUT 参数，并将有效淡入淡出长度标准化到索引 1。
///
/// Reaper 规则：倒数第 3 个参数为 1 时，长度取第 3 个参数；否则取第 2 个参数。
fn parse_fade_array(tokens: &[&str]) -> Vec<f64> {
    let mut values = parse_double_array(tokens);
    if values.len() < 2 {
        return values;
    }

    let selector_idx = values.len().saturating_sub(3);
    let selector = values.get(selector_idx).copied().unwrap_or(0.0).round() as i32;
    let effective = if selector == 1 {
        values.get(2).copied().unwrap_or(values[1])
    } else {
        values[1]
    };

    values[1] = effective;
    values
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
                data.track_offsets.push(pending_offset);
                data.tracks.push(t);
            }
            let track = parse_track_block(child);
            data.is_track_data = true;
            data.track_offsets.push(cumulative_track_offset);
            data.tracks.push(track);
            cumulative_track_offset += 1;
            current_track = None;
            continue;
        }

        if block_type.as_deref() == Some("ITEM") {
            let item = parse_item_block(child);
            if current_track.is_none() {
                pending_offset = cumulative_track_offset;
                current_track = Some(ReaperTrack::default());
            }
            current_track.as_mut().unwrap().items.push(item);
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
                data.track_offsets.push(pending_offset);
                data.tracks.push(t);
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
        }
    }

    if let Some(t) = current_track {
        data.track_offsets.push(pending_offset);
        data.tracks.push(t);
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
            "LOOP" if tokens.len() >= 2 => item.is_loop = parse_bool(&tokens[1]),
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
        let active_take_idx = item.takes.iter().position(|take| take.selected).unwrap_or(0);
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
            _ => {}
        }
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
        if tokens[0].to_uppercase() == "PT" {
            env.points.push(parse_double_array(&tokens));
        }
    }

    env
}

fn is_envelope_type(s: &str) -> bool {
    ENVELOPE_TYPES.iter().any(|&e| e.eq_ignore_ascii_case(s))
}
