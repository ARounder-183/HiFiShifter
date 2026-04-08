// VocalShifter 剪贴板文件解析模块。
//
// 临时目录 %TEMP%/vocalshifter_tmp/ 下的候选文件（按修改时间取最晚的）：
// - vocalshifter_tr.clb.vshp   (工程文件)
// - vocalshifter_le_tr.clb.vshp (工程文件)
// - vocalshifter_tr.clb.vsp    (工程文件)
// - vocalshifter_le_tr.clb.vsp (工程文件)
// - vocalshifter_id.clb        (音高线数据)
// - vocalshifter_le_id.clb     (音高线数据)
//
// .clb 记录格式：每条 0x80 字节（16 个 little-endian f64）：
// - [0]  offset  0: time_sec
// - [1]  offset  8: disabled (1.0 disabled / 0.0 enabled)
// - [2]  offset 16: pitch_cents (0 = C-1, 6000 = C4)
// - [3]  offset 24: formant_cents (FRM)
// - [4]  offset 32: volume (VOL, double multiplier)
// - [5]  offset 40: pan (PAN, -1.0..1.0)
// - [6]  offset 48: dyn_edit (DYN, double multiplier)
// - [7]  offset 56: original_pitch_cents (*PIT)
// - [8]  offset 64: breathiness (BRE, raw)
// - [9]  offset 72: eq1 (EQ1, raw)
// - [10] offset 80: eq2 (EQ2, raw)
// - [11] offset 88: heq_mrp (HEQ/MRP, raw)

use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const RECORD_SIZE: usize = 0x80;
const CENTS_TO_MIDI: f64 = 1.0 / 100.0;

#[derive(Debug, Clone, Copy)]
pub struct ClipboardPitchPoint {
    pub time_sec: f64,
    pub disabled: bool,
    pub midi_pitch: f32,
    /// FRM (formant shift in cents, relative to original formant; 0.0 = no shift)
    pub formant_cents: f64,
    /// VOL (volume multiplier, 1.0 = 0dB)
    pub volume: f64,
    /// PAN (-1.0 = left, 0.0 = center, 1.0 = right)
    pub pan: f64,
    /// DYN (dynamics edit, multiplier)
    pub dyn_edit: f64,
    /// *PIT (original pitch cents before edit)
    #[allow(dead_code)]
    pub original_pitch_cents: f64,
    /// BRE (breathiness, raw value -10000..10000)
    pub breathiness: f64,
    /// EQ1 (raw value)
    #[allow(dead_code)]
    pub eq1: f64,
    /// EQ2 (raw value)
    #[allow(dead_code)]
    pub eq2: f64,
    /// HEQ/MRP (raw value)
    #[allow(dead_code)]
    pub heq_mrp: f64,
}

/// 剪贴板文件类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipboardFileKind {
    /// .clb 文件 — 纯音高线数据
    PitchData,
    /// .clb.vshp / .clb.vsp 文件 — VocalShifter 工程文件
    Project,
}

/// 查找最新的 VocalShifter 剪贴板文件，返回路径及其类型。
pub fn find_latest_clipboard_file() -> Option<(PathBuf, ClipboardFileKind)> {
    let base = std::env::temp_dir().join("vocalshifter_tmp");
    // 将堆分配延迟到循环内，消除未命中时的无意义内存开销
    let candidates = [
        ("vocalshifter_tr.clb.vshp", ClipboardFileKind::Project),
        ("vocalshifter_le_tr.clb.vshp", ClipboardFileKind::Project),
        ("vocalshifter_tr.clb.vsp", ClipboardFileKind::Project),
        ("vocalshifter_le_tr.clb.vsp", ClipboardFileKind::Project),
        ("vocalshifter_id.clb", ClipboardFileKind::PitchData),
        ("vocalshifter_le_id.clb", ClipboardFileKind::PitchData),
    ];

    let mut best: Option<(PathBuf, ClipboardFileKind, SystemTime)> = None;
    for (name, kind) in candidates {
        let path = base.join(name);
        if let Ok(meta) = fs::metadata(&path) {
            if meta.is_file() {
                let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                match &best {
                    None => best = Some((path, kind, modified)),
                    Some((_, _, t)) if modified > *t => best = Some((path, kind, modified)),
                    _ => {}
                }
            }
        }
    }

    best.map(|(p, k, _)| (p, k))
}

pub fn parse_clipboard_file(path: &Path) -> Result<Vec<ClipboardPitchPoint>, String> {
    let data = fs::read(path).map_err(|e| format!("io_error: {}", e))?;

    if data.len() % RECORD_SIZE != 0 {
        return Err(format!(
            "invalid_format: file size {} is not multiple of {}",
            data.len(),
            RECORD_SIZE
        ));
    }

    let read_f64 = |rec: &[u8], offset: usize| -> f64 {
        f64::from_le_bytes(rec[offset..offset + 8].try_into().unwrap_or([0u8; 8]))
    };

    let mut out = Vec::with_capacity(data.len() / RECORD_SIZE);
    for rec in data.chunks_exact(RECORD_SIZE) {
        let time_sec = read_f64(rec, 0);
        let disabled_raw = read_f64(rec, 8);
        let pitch_cents = read_f64(rec, 16);
        let formant_cents = read_f64(rec, 24);
        let volume = read_f64(rec, 32);
        let pan = read_f64(rec, 40);
        let dyn_edit = read_f64(rec, 48);
        let original_pitch_cents = read_f64(rec, 56);
        let breathiness = read_f64(rec, 64);
        let eq1 = read_f64(rec, 72);
        let eq2 = read_f64(rec, 80);
        let heq_mrp = read_f64(rec, 88);

        out.push(ClipboardPitchPoint {
            time_sec,
            disabled: (disabled_raw - 1.0).abs() < 1e-9,
            midi_pitch: (pitch_cents * CENTS_TO_MIDI) as f32,
            formant_cents,
            volume,
            pan,
            dyn_edit,
            original_pitch_cents,
            breathiness,
            eq1,
            eq2,
            heq_mrp,
        });
    }

    Ok(out)
}
