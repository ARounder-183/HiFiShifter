//! REAPER 原生 MIDI item/take 读写。
//!
//! ARA 2.0 没有 MIDI 对象；MIDI 只能走宿主原生 API。本模块把官方
//! reaper_plugin_functions.h 中稳定的 MIDI_* / CreateNewMIDIItemInProj
//! 一束函数收在一起，并用与现有 take 枚举完全相同的授权/线程纪律调用。
use super::{checked, NativeHost};
use crate::host::geometry::HostClipGeometry;
use hifishifter_kernel::midi_import::MidiNoteEvent;
use hifishifter_kernel::midi_take::{
    MidiExpressionEvent, MidiExpressionKind, MidiTakeData, MidiTakeRole,
};
use std::ffi::{c_char, c_void};
use std::sync::Arc;

pub(super) type ActiveTake = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
pub(super) type TakeIsMidi = unsafe extern "C" fn(*mut c_void) -> bool;
pub(super) type MidiCountEvts =
    unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32, *mut i32) -> bool;
pub(super) type MidiGetNote = unsafe extern "C" fn(
    *mut c_void,
    i32,
    *mut bool,
    *mut bool,
    *mut f64,
    *mut f64,
    *mut i32,
    *mut i32,
    *mut i32,
) -> bool;
pub(super) type MidiGetAllEvts = unsafe extern "C" fn(*mut c_void, *mut c_char, *mut i32) -> bool;
pub(super) type MidiSetAllEvts = unsafe extern "C" fn(*mut c_void, *const c_char, i32) -> bool;
pub(super) type MidiGetHash = unsafe extern "C" fn(*mut c_void, bool, *mut c_char, i32) -> bool;
pub(super) type MidiProjTimeFromPpq = unsafe extern "C" fn(*mut c_void, f64) -> f64;
pub(super) type MidiPpqFromProjTime = unsafe extern "C" fn(*mut c_void, f64) -> f64;
pub(super) type MidiSort = unsafe extern "C" fn(*mut c_void);
pub(super) type CreateMidiItem =
    unsafe extern "C" fn(*mut c_void, f64, f64, *const bool) -> *mut c_void;
pub(super) type TrackFxInstrument = unsafe extern "C" fn(*mut c_void) -> i32;

pub(crate) struct MidiTakeSnapshot {
    pub hash: String,
    pub data: MidiTakeData,
}

pub(super) struct MidiApi {
    pub active_take: Option<ActiveTake>,
    pub take_is_midi: TakeIsMidi,
    pub count: MidiCountEvts,
    pub get_note: MidiGetNote,
    pub get_all: MidiGetAllEvts,
    pub set_all: Option<MidiSetAllEvts>,
    pub hash: Option<MidiGetHash>,
    pub proj_time_from_ppq: MidiProjTimeFromPpq,
    pub ppq_from_proj_time: Option<MidiPpqFromProjTime>,
    pub sort: Option<MidiSort>,
    pub create_item: Option<CreateMidiItem>,
    pub track_fx_instrument: Option<TrackFxInstrument>,
}

impl MidiApi {
    pub(super) fn has_read(&self) -> bool {
        true
    }
    pub(super) fn has_write(&self) -> bool {
        self.set_all.is_some() && self.sort.is_some() && self.ppq_from_proj_time.is_some()
    }
    pub(super) fn has_create(&self) -> bool {
        self.create_item.is_some()
            && self.set_all.is_some()
            && self.sort.is_some()
            && self.active_take.is_some()
            && self.ppq_from_proj_time.is_some()
    }
}

impl NativeHost {
    /// 新建一个宿主 MIDI item，并把 Take 数据写进去。返回 item GUID。
    pub(super) fn create_midi_item(
        self: &Arc<Self>,
        track: *mut c_void,
        start_sec: f64,
        length_sec: f64,
        data: &MidiTakeData,
        name: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<String, String> {
        if std::thread::current().id() != self.thread {
            return self.on_host(|| {
                self.create_midi_item(track, start_sec, length_sec, data, name, authorized)
            })?;
        }
        let api = self.midi.as_ref().ok_or("host MIDI API unavailable")?;
        let create = api
            .create_item
            .ok_or("host MIDI item creation unavailable")?;
        let set_all = api.set_all.ok_or("host MIDI write unavailable")?;
        let sort = api.sort.ok_or("host MIDI sort unavailable")?;
        if !start_sec.is_finite() || !length_sec.is_finite() || length_sec <= 0.0 {
            return Err("invalid MIDI item geometry".into());
        }
        let item = checked(authorized, || unsafe {
            create(track, start_sec, start_sec + length_sec, std::ptr::null())
        })?;
        if item.is_null() {
            return Err("REAPER rejected MIDI item creation".into());
        }
        let active_take = api.active_take.ok_or("host active-take API unavailable")?;
        let take = checked(authorized, || unsafe { active_take(item) })?;
        if take.is_null() {
            return Err("new MIDI item has no active take".into());
        }
        let events = encode_events(data, start_sec, api, take, authorized)?;
        if !checked(authorized, || unsafe {
            set_all(take, events.as_ptr().cast(), events.len() as i32)
        })? {
            return Err("REAPER rejected MIDI event write".into());
        }
        checked(authorized, || unsafe { sort(take) })?;
        let write = self
            .write
            .as_ref()
            .ok_or("host item write API unavailable")?;
        let _ = checked(authorized, || unsafe {
            (write.set_item)(item, c"D_POSITION".as_ptr(), start_sec)
        });
        let _ = checked(authorized, || unsafe {
            (write.set_item)(item, c"D_LENGTH".as_ptr(), length_sec)
        });
        if !name.is_empty() {
            let mut bytes = name.as_bytes().to_vec();
            bytes.push(0);
            let _ = checked(authorized, || unsafe {
                (write.set_take_string)(take, c"P_NAME".as_ptr(), bytes.as_mut_ptr().cast(), true)
            });
        }
        item_guid(self, item, authorized)
    }

    /// 覆盖一条已有 take 的 MIDI 数据；调用方负责 undo block 与后续清单刷新。
    #[allow(dead_code)]
    pub(super) fn write_midi_take(
        &self,
        take: *mut c_void,
        geometry: &HostClipGeometry,
        data: &MidiTakeData,
        authorized: &impl Fn() -> bool,
    ) -> Result<(), String> {
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.write_midi_take(take, geometry, data, authorized))?;
        }
        let api = self.midi.as_ref().ok_or("host MIDI API unavailable")?;
        let set_all = api.set_all.ok_or("host MIDI write unavailable")?;
        let sort = api.sort.ok_or("host MIDI sort unavailable")?;
        let events = encode_events_for_take(data, geometry, api, take, authorized)?;
        if !checked(authorized, || unsafe {
            set_all(take, events.as_ptr().cast(), events.len() as i32)
        })? {
            return Err("REAPER rejected MIDI event write".into());
        }
        checked(authorized, || unsafe { sort(take) })?;
        Ok(())
    }
}

fn encode_events(
    data: &MidiTakeData,
    item_start_sec: f64,
    api: &MidiApi,
    take: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Result<Vec<u8>, String> {
    let ppq = api
        .ppq_from_proj_time
        .ok_or("host MIDI time conversion unavailable")?;
    let mut events: Vec<(f64, Vec<u8>)> = Vec::new();
    for note in &data.notes {
        if note.muted || note.velocity == 0 || note.end_sec <= note.start_sec {
            continue;
        }
        let channel = note.channel.min(15);
        let key = note.note.round().clamp(0.0, 127.0) as u8;
        let on = vec![0x90 | channel, key, note.velocity.min(127)];
        let off = vec![0x80 | channel, key, note.release_velocity.min(127)];
        events.push((note.start_sec, on));
        events.push((note.end_sec, off));
    }
    for event in &data.expression {
        let message = match &event.kind {
            MidiExpressionKind::ControlChange {
                channel,
                controller,
                value,
            } => Some(vec![
                0xB0 | channel.min(&15),
                *controller & 0x7F,
                *value & 0x7F,
            ]),
            MidiExpressionKind::PitchBend { channel, value } => {
                let raw = (*value as i32 + 8192).clamp(0, 16383) as u16;
                Some(vec![
                    0xE0 | channel.min(&15),
                    (raw & 0x7F) as u8,
                    ((raw >> 7) & 0x7F) as u8,
                ])
            }
            MidiExpressionKind::ProgramChange { channel, program } => {
                Some(vec![0xC0 | channel.min(&15), *program & 0x7F])
            }
            MidiExpressionKind::ChannelAftertouch { channel, value } => {
                Some(vec![0xD0 | channel.min(&15), *value & 0x7F])
            }
            MidiExpressionKind::PolyAftertouch {
                channel,
                key,
                value,
            } => Some(vec![0xA0 | channel.min(&15), *key & 0x7F, *value & 0x7F]),
            _ => None,
        };
        if let Some(message) = message {
            events.push((event.time_sec, message));
        }
    }
    if events.is_empty() {
        return Ok(Vec::new());
    }
    events.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = Vec::new();
    let mut previous_ticks = 0i64;
    for (time_sec, message) in events {
        let project = item_start_sec + time_sec;
        let absolute = checked(authorized, || unsafe { ppq(take, project) }).unwrap_or(0.0);
        let ticks = (absolute * 960.0).round().max(0.0) as i64;
        let delta = (ticks - previous_ticks).max(0) as i32;
        previous_ticks = ticks;
        out.extend_from_slice(&delta.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&(message.len() as i32).to_le_bytes());
        out.extend_from_slice(&message);
    }
    // 补一条 all-notes-off，避免宿主把最后一个音符挂死。
    out.extend_from_slice(&0i32.to_le_bytes());
    out.push(0);
    out.extend_from_slice(&3i32.to_le_bytes());
    out.extend_from_slice(&[0xB0, 123, 0]);
    Ok(out)
}

fn encode_events_for_take(
    data: &MidiTakeData,
    geometry: &HostClipGeometry,
    api: &MidiApi,
    take: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Result<Vec<u8>, String> {
    // 已有 take 的 tick 是 take 源域；source_sec 先回到工程时间。
    let rate = geometry.play_rate_signed.abs().max(1e-6);
    let project_start = geometry.start_sec
        + (data.notes.iter().map(|n| n.start_sec).fold(0.0, f64::min) - geometry.source_start_sec)
            / rate;
    encode_events(data, project_start, api, take, authorized)
}

fn item_guid(
    host: &NativeHost,
    item: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Result<String, String> {
    let geometry = host.geometry.as_ref().ok_or("host geometry unavailable")?;
    let mut bytes = [0u8; 64];
    if !checked(authorized, || unsafe {
        (geometry.item_guid)(item, c"GUID".as_ptr(), bytes.as_mut_ptr().cast(), false)
    })? {
        return Err("REAPER item GUID unavailable".into());
    }
    if bytes[38] != 0 || bytes[0] != b'{' || bytes[37] != b'}' {
        return Err("invalid REAPER item GUID".into());
    }
    Ok(String::from_utf8_lossy(&bytes[..38]).into_owned())
}

impl NativeHost {
    /// 读取一条 take 的 MIDI 内容。非 MIDI take 返回 None；API 缺失时也返回 None。
    pub(super) fn read_midi_take(
        &self,
        take: *mut c_void,
        geometry: &HostClipGeometry,
        authorized: &impl Fn() -> bool,
    ) -> Option<Arc<MidiTakeSnapshot>> {
        if std::thread::current().id() != self.thread {
            return self
                .on_host(|| self.read_midi_take(take, geometry, authorized))
                .ok()
                .flatten();
        }
        let api = self.midi.as_ref()?;
        if take.is_null()
            || !checked(authorized, || unsafe { (api.take_is_midi)(take) }).unwrap_or(false)
        {
            return None;
        }
        let mut notes_budget = 0i32;
        let mut cc_budget = 0i32;
        let mut text_budget = 0i32;
        if !checked(authorized, || unsafe {
            (api.count)(take, &mut notes_budget, &mut cc_budget, &mut text_budget)
        })
        .unwrap_or(false)
        {
            return None;
        }
        if !(0..=500_000).contains(&notes_budget) {
            return None;
        }
        let mut notes = Vec::with_capacity(notes_budget as usize);
        let mut diagnostics = Vec::new();
        for index in 0..notes_budget {
            let mut selected = false;
            let mut muted = false;
            let mut start_ppq = 0.0;
            let mut end_ppq = 0.0;
            let mut channel = 0i32;
            let mut pitch = 0i32;
            let mut velocity = 0i32;
            let ok = checked(authorized, || unsafe {
                (api.get_note)(
                    take,
                    index,
                    &mut selected,
                    &mut muted,
                    &mut start_ppq,
                    &mut end_ppq,
                    &mut channel,
                    &mut pitch,
                    &mut velocity,
                )
            })
            .unwrap_or(false);
            if !ok {
                diagnostics.push(format!("note {index} unreadable"));
                continue;
            }
            let project_start = checked(authorized, || unsafe {
                (api.proj_time_from_ppq)(take, start_ppq)
            })
            .unwrap_or(0.0);
            let project_end = checked(authorized, || unsafe {
                (api.proj_time_from_ppq)(take, end_ppq)
            })
            .unwrap_or(project_start);
            let start_sec = source_sec(geometry, project_start);
            let end_sec = source_sec(geometry, project_end);
            if end_sec <= start_sec || !start_sec.is_finite() || !end_sec.is_finite() {
                continue;
            }
            notes.push(MidiNoteEvent {
                start_sec,
                end_sec,
                note: pitch.clamp(0, 127) as f32,
                velocity: velocity.clamp(0, 127) as u8,
                channel: channel.clamp(0, 15) as u8,
                muted,
                release_velocity: 0,
            });
        }
        let mut expression = Vec::new();
        if let Some(buf) = read_all_events(api, take, authorized) {
            expression = parse_expression(&buf, geometry, api, take, authorized);
        }
        // 【空 MIDI take 也是 MIDI take】此前这里把"没有音符"当成"不是 MIDI"，
        // 于是用户刚建的空 MIDI item 拿不到快照 ⇒ clip 上没有 MIDI 内容 ⇒ 界面按
        // **音频占位**渲染，永远挂着"正在等待宿主完成音频分配"。宿主已经明确回答
        // `TakeIsMidi` 了，空内容只是一条还没有音符的 MIDI take。
        let hash = read_hash(api, take, authorized)
            .unwrap_or_else(|| format!("notes:{}:{}", notes.len(), expression.len()));
        if !diagnostics.is_empty() {
            log::debug!("[midi] host take diagnostics: {:?}", diagnostics);
        }
        let mut data = MidiTakeData {
            role: MidiTakeRole::MidiTake,
            notes,
            expression,
            // 【源总长可能是 0】REAPER 对 MIDI source 常常报 0（它不是采样文件）。
            // 0 会让"内容长度"退化成 0 ⇒ Loop 周期未知、裁剪窗口无意义。此时用
            // item 消费的源域长度（时长 × 速率）兜底 —— 那是这段内容真实覆盖的范围。
            length_sec: geometry
                .source_length_sec
                .filter(|length| length.is_finite() && *length > 0.0)
                .unwrap_or_else(|| {
                    geometry.duration_sec * geometry.play_rate_signed.abs().max(1e-6)
                })
                .max(0.0),
            source_bpm: 0.0,
            ticks_per_beat: 0,
            pitch_bend_ranges: hifishifter_kernel::midi_take::default_pitch_bend_ranges(),
            tempo_map: Vec::new(),
            time_signatures: Vec::new(),
            key_signatures: Vec::new(),
            name: String::new(),
            pitch_reference: Default::default(),
            content_hash: hash.clone(),
            preview: Default::default(),
        };
        data.normalize();
        Some(Arc::new(MidiTakeSnapshot { hash, data }))
    }

    /// 是否有 MIDI 读取能力。
    pub(super) fn can_read_midi(&self) -> bool {
        self.midi.as_ref().is_some_and(MidiApi::has_read)
    }

    pub(super) fn can_create_midi(&self) -> bool {
        self.midi.as_ref().is_some_and(MidiApi::has_create)
    }

    pub(super) fn can_write_midi(&self) -> bool {
        self.midi.as_ref().is_some_and(MidiApi::has_write)
    }

    /// 轨道是否有 VSTi；用于决定新建 MIDI Take 的默认试听开关。
    #[allow(dead_code)]
    pub(super) fn track_has_instrument(
        &self,
        track: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> bool {
        let Some(api) = self.midi.as_ref() else {
            return false;
        };
        let Some(fx) = api.track_fx_instrument else {
            return false;
        };
        checked(authorized, || unsafe { fx(track) >= 0 }).unwrap_or(false)
    }
}

fn source_sec(geometry: &HostClipGeometry, project_sec: f64) -> f64 {
    let local = project_sec - geometry.start_sec;
    let rate = if geometry.play_rate_signed.is_finite() {
        geometry.play_rate_signed
    } else {
        1.0
    };
    local * rate + geometry.source_start_sec
}

fn read_hash(api: &MidiApi, take: *mut c_void, authorized: &impl Fn() -> bool) -> Option<String> {
    let hash = api.hash?;
    let mut buf = vec![0u8; 256];
    let ok = checked(authorized, || unsafe {
        hash(take, true, buf.as_mut_ptr().cast(), buf.len() as i32)
    })
    .unwrap_or(false);
    if !ok {
        return None;
    }
    let end = buf.iter().position(|byte| *byte == 0).unwrap_or(buf.len());
    Some(String::from_utf8_lossy(&buf[..end]).into_owned())
}

fn read_all_events(
    api: &MidiApi,
    take: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Option<Vec<u8>> {
    let get = api.get_all;
    let mut size = 0i32;
    let _ = checked(authorized, || unsafe {
        get(take, std::ptr::null_mut(), &mut size)
    });
    if !(0..=16 * 1024 * 1024).contains(&size) {
        return None;
    }
    let mut buf = vec![0u8; size.max(1) as usize];
    let ok = checked(authorized, || unsafe {
        get(take, buf.as_mut_ptr().cast(), &mut size)
    })
    .unwrap_or(false);
    if !ok {
        return None;
    }
    buf.truncate(size.max(0) as usize);
    Some(buf)
}

fn parse_expression(
    buf: &[u8],
    geometry: &HostClipGeometry,
    api: &MidiApi,
    take: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Vec<MidiExpressionEvent> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    let mut abs_tick = 0i64;
    while offset + 9 <= buf.len() {
        let delta = i32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap_or([0; 4]));
        let _flags = buf[offset + 4];
        let len = i32::from_le_bytes(buf[offset + 5..offset + 9].try_into().unwrap_or([0; 4]));
        if len < 0 || offset + 9 + len as usize > buf.len() {
            break;
        }
        abs_tick += i64::from(delta.max(0));
        let msg = &buf[offset + 9..offset + 9 + len as usize];
        offset += 9 + len as usize;
        if msg.is_empty() {
            continue;
        }
        let status = msg[0] & 0xF0;
        let channel = msg[0] & 0x0F;
        if matches!(status, 0x80 | 0x90 | 0xA0 | 0xB0 | 0xE0) && msg.len() >= 3 {
            let data1 = msg[1] & 0x7F;
            let data2 = msg[2] & 0x7F;
            // 用 PPQ 的秒换算；GetAllEvts 的 offset 是 tick，但 REAPER 的
            // MIDI_GetProjTimeFromPPQPos 需要绝对 PPQ。这里按 960 PPQ 的
            // 常见缺省无法保证正确，因此只在有 PPQ 转换时使用事件；
            // 读不到时返回空，不伪造时间。
            let ppq = abs_tick as f64 / 960.0;
            let project = checked(authorized, || unsafe {
                (api.proj_time_from_ppq)(take, ppq)
            })
            .unwrap_or(f64::NAN);
            if !project.is_finite() {
                continue;
            }
            let time_sec = source_sec(geometry, project);
            let kind = match status {
                0xB0 => Some(MidiExpressionKind::ControlChange {
                    channel,
                    controller: data1,
                    value: data2,
                }),
                0xE0 => Some(MidiExpressionKind::PitchBend {
                    channel,
                    value: (((data2 as i16) << 7) | data1 as i16) - 8192,
                }),
                0xA0 => Some(MidiExpressionKind::PolyAftertouch {
                    channel,
                    key: data1,
                    value: data2,
                }),
                0xD0 => Some(MidiExpressionKind::ChannelAftertouch {
                    channel,
                    value: data1,
                }),
                0xC0 => Some(MidiExpressionKind::ProgramChange {
                    channel,
                    program: data1,
                }),
                _ => None,
            };
            if let Some(kind) = kind {
                out.push(MidiExpressionEvent { time_sec, kind });
            }
        }
    }
    out
}
