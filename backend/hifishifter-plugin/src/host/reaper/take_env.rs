//! take 的 **Pitch 包络对象**：用 Envelope API 读写，而不是重写 item chunk。
//!
//! # 为什么必须有这条路（真机实测，REAPER 7.85）
//!
//! `SetItemStateChunk` 对**挂着本插件（ARA）的 take FX 的那个 item** 一律返回 `false`
//! —— 连"原样写回"都拒绝（`isundo`、undo 块、改一个字节，三种写法全 false）。同一份
//! chunk 在**没有**本插件 take FX 的 item 上写回成功；把本插件设成 offline 之后也成功。
//! 结论：宿主不允许在 ARA 实例**在线**时重建这个 item 的 FX 链。
//!
//! take FX 放置下，"要改包络的那个 item"**恰好就是**挂着本插件的那个 item —— 于是
//! 任何走 chunk 的写回（take 音高、倒放）在 take FX 下都必然失败。真机现场就是
//! `[pitch] auto take-pitch write failed: REAPER rejected the item state`。
//!
//! 而 **Envelope API 不受这条限制**（同一台机器实测）：`GetTakeEnvelopeByName` 拿到的
//! 包络可以 `SetEnvelopeStateChunk` 整块替换，本插件在线时照样成功，写进去的点立即可读，
//! 空块会让宿主删掉整条包络。所以：
//!
//! * **读**：有包络就读它自己的 chunk（比 item chunk 小几个数量级）；
//! * **写**：有包络就写它；没有包络时才回到 item chunk（轨道 FX 下的历史路径）。
//!
//! # 创建包络为什么借宿主的动作
//!
//! ARA 2.0 与 REAPER 的扩展 API 都**没有**"新建一条 take 包络"的入口：
//! `GetTakeEnvelopeByName` 在没有包络时返回空指针，而写 chunk 正是上面被拒绝的那条路。
//! 宿主自己的 `Take: Toggle take pitch envelope` 动作是唯一能创建它的入口，
//! 于是这里**按名字**（而不是硬编码 id）解析该命令，并用"只影响本 item"的选择态调用它。
//!
//! 【为什么要临时改选择又改回来】动作作用于**所有选中的 item**。不改选择就等于给用户
//! 选中的每一条都加一条音高包络 —— 那是数据损坏。这里先记下全部 item 的选择位、
//! 只选中目标 item、跑动作、再逐条恢复；任何一步失败都先恢复再报错。

use super::{checked, HostVtbl};
use std::ffi::{c_char, c_void, CStr};

/// 宿主动作列表里那条命令的**名字**（本地化宿主里找不到时会退化成一条可读错误）。
const PITCH_ENVELOPE_ACTION: &CStr = c"Take: Toggle take pitch envelope";

/// item 选择位的上限：超过这个规模就**不动**选择，宁可报错也不做 O(n) 的全局改写。
const MAX_SELECTION_ITEMS: usize = 20_000;

type GetEnvelopeByName = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type GetEnvelopeChunk = unsafe extern "C" fn(*mut c_void, *mut c_char, i32, bool) -> bool;
type SetEnvelopeChunk = unsafe extern "C" fn(*mut c_void, *const c_char, bool) -> bool;
type EnumerateActions = unsafe extern "C" fn(i32, i32, *mut c_char, i32) -> i32;
type RunCommand = unsafe extern "C" fn(i32, i32);
type CountTracks = unsafe extern "C" fn(*mut c_void) -> i32;
type GetTrack = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
type CountItems = unsafe extern "C" fn(*mut c_void) -> i32;
type GetItem = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
type ItemValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;
type SetItemValue = unsafe extern "C" fn(*mut c_void, *const c_char, f64) -> bool;

impl super::NativeHost {
    /// 取一个宿主原生函数入口（与 `ui_track_at` 同一条纪律：用到才取，取不到就报错）。
    fn native_api<T: Copy>(
        &self,
        name: &CStr,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<T, String> {
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let function = checked(authorized, || unsafe {
            (table.api)(pointer, name.as_ptr())
        })?;
        if function.is_null() {
            return Err(format!(
                "REAPER API unavailable: {}",
                name.to_string_lossy()
            ));
        }
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&function) })
    }

    /// 这条 take 的 Pitch 包络；没有就 `None`。
    pub(super) fn take_pitch_envelope(
        &self,
        take: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<Option<*mut c_void>, String> {
        if !self.on_host_thread() {
            // 裸指针不是 `Send`：过边界前换成 `usize`（与 `NativeHost::project` 同一条纪律）。
            return self
                .on_host(|| {
                    self.take_pitch_envelope(take, authorized)
                        .map(|envelope| envelope.map(|pointer| pointer as usize))
                })?
                .map(|envelope| envelope.map(|pointer| pointer as *mut c_void));
        }
        // 【宿主没有这一族读数 ⇒ 当作"没有包络对象"】老宿主、以及没有包络 API 的适配层
        // 必须继续走 item chunk 那条历史路径；把"API 缺失"报成错误会让一次本来能成的
        // 音高写回直接失败。
        let Ok(by_name) =
            self.native_api::<GetEnvelopeByName>(c"GetTakeEnvelopeByName", authorized)
        else {
            return Ok(None);
        };
        let envelope = checked(authorized, || unsafe { by_name(take, c"Pitch".as_ptr()) })?;
        Ok(if envelope.is_null() {
            None
        } else {
            Some(envelope)
        })
    }

    /// 一条包络自己的 state chunk（就是那份 `<PITCHENV>` 块）。
    pub(super) fn envelope_state_chunk(
        &self,
        envelope: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<String, String> {
        if !self.on_host_thread() {
            return self.on_host(|| self.envelope_state_chunk(envelope, authorized))?;
        }
        let get: GetEnvelopeChunk = self.native_api(c"GetEnvelopeStateChunk", authorized)?;
        // 包络比 item chunk 小几个数量级（点数 × 约 30 字节），8 KiB 起步够用；
        // 真放不下时宿主返回 false 或写满缓冲区，两种都靠"再加倍"兜住。
        let mut capacity = 8 * 1024_usize;
        loop {
            let mut buffer = vec![0_u8; capacity];
            let ok = checked(authorized, || unsafe {
                get(envelope, buffer.as_mut_ptr().cast(), capacity as i32, false)
            })?;
            if ok {
                if let Some(end) = buffer.iter().position(|byte| *byte == 0) {
                    if end > 0 && end + 1 < capacity {
                        return String::from_utf8(buffer[..end].to_vec())
                            .map_err(|_| "host envelope state is not valid UTF-8".to_string());
                    }
                }
            }
            if capacity >= 4 * 1024 * 1024 {
                return Err("host envelope state exceeds the supported budget".into());
            }
            capacity *= 2;
        }
    }

    /// 整块替换一条包络（空点集由宿主删掉整条包络 —— 真机行为，见模块文档）。
    pub(super) fn set_envelope_state_chunk(
        &self,
        envelope: *mut c_void,
        text: &str,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        if !self.on_host_thread() {
            return self.on_host(|| self.set_envelope_state_chunk(envelope, text, authorized))?;
        }
        if text.is_empty() || text.contains('\0') {
            return Err("invalid envelope state to write".into());
        }
        let set: SetEnvelopeChunk = self.native_api(c"SetEnvelopeStateChunk", authorized)?;
        let c_text = std::ffi::CString::new(text).map_err(|_| "envelope state contains NUL")?;
        if !checked(authorized, || unsafe {
            set(envelope, c_text.as_ptr(), false)
        })? {
            return Err("REAPER rejected the envelope state".into());
        }
        Ok(())
    }

    /// 建一条 take 的 Pitch 包络并返回它（宿主没有创建 API，只能借它自己的动作）。
    pub(super) fn create_take_pitch_envelope(
        &self,
        item: *mut c_void,
        take: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<*mut c_void, String> {
        if !self.on_host_thread() {
            return self
                .on_host(|| {
                    self.create_take_pitch_envelope(item, take, authorized)
                        .map(|envelope| envelope as usize)
                })?
                .map(|envelope| envelope as *mut c_void);
        }
        if item.is_null() || take.is_null() {
            return Err("the host item/take is gone".into());
        }
        let command = self.pitch_envelope_action(authorized)?;
        let selection = self.item_selection(authorized)?;
        if selection.len() > MAX_SELECTION_ITEMS {
            return Err("the project has too many items to safely change the selection".into());
        }
        let result = (|| -> Result<(), String> {
            for (pointer, _) in &selection {
                self.set_item_selected(*pointer, false, authorized)?;
            }
            self.set_item_selected(item, true, authorized)?;
            let run: RunCommand = self.native_api(c"Main_OnCommand", authorized)?;
            checked(authorized, || unsafe { run(command, 0) })?;
            Ok(())
        })();
        // 【无论成败都先恢复选择】动作只该影响目标 item；让用户的选区停在我们改过的
        // 状态上，等于"点了一次转换，REAPER 里的选中全变了"。
        let mut restore_error = None;
        for (pointer, selected) in selection {
            if let Err(error) = self.set_item_selected(pointer, selected, authorized) {
                restore_error.get_or_insert(error);
            }
        }
        result?;
        if let Some(error) = restore_error {
            return Err(format!("item selection could not be restored: {error}"));
        }
        self.take_pitch_envelope(take, authorized)?
            .ok_or_else(|| "the host did not create a take pitch envelope".to_string())
    }

    /// 宿主动作表里 `Take: Toggle take pitch envelope` 的命令号。
    fn pitch_envelope_action(
        &self,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<i32, String> {
        let enumerate: EnumerateActions = self.native_api(c"kbd_enumerateActions", authorized)?;
        let mut buffer = [0_i8; 256];
        // 上限是**动作表**的规模（REAPER 自带约 2600 条，上限翻倍留余量）。
        for index in 0..8192 {
            buffer.fill(0);
            let command = checked(authorized, || unsafe {
                enumerate(0, index, buffer.as_mut_ptr().cast(), 256)
            })?;
            if command == 0 {
                break;
            }
            let name = unsafe { CStr::from_ptr(buffer.as_ptr()) };
            if name == PITCH_ENVELOPE_ACTION {
                return Ok(command);
            }
        }
        Err(format!(
            "the host action {:?} was not found (localized REAPER?); enable the take pitch \
             envelope once by hand and retry",
            PITCH_ENVELOPE_ACTION.to_string_lossy()
        ))
    }

    /// 工程里每一项 item 的选择位（指针, 是否选中）。
    fn item_selection(
        &self,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<Vec<(*mut c_void, bool)>, String> {
        let count_tracks: CountTracks = self.native_api(c"CountTracks", authorized)?;
        let get_track: GetTrack = self.native_api(c"GetTrack", authorized)?;
        let count_items: CountItems = self.native_api(c"CountTrackMediaItems", authorized)?;
        let get_item: GetItem = self.native_api(c"GetTrackMediaItem", authorized)?;
        let read: ItemValue = self.native_api(c"GetMediaItemInfo_Value", authorized)?;
        let project = self.project(&|| authorized())?;
        let tracks = checked(authorized, || unsafe { count_tracks(project) })?;
        if !(0..=4096).contains(&tracks) {
            return Err("the host reported an implausible track count".into());
        }
        let mut items = Vec::new();
        for index in 0..tracks {
            let track = checked(authorized, || unsafe { get_track(project, index) })?;
            if track.is_null() {
                continue;
            }
            let count = checked(authorized, || unsafe { count_items(track) })?;
            if !(0..=MAX_SELECTION_ITEMS as i32).contains(&count) {
                return Err("the host reported an implausible item count".into());
            }
            for slot in 0..count {
                let item = checked(authorized, || unsafe { get_item(track, slot) })?;
                if item.is_null() {
                    continue;
                }
                let selected =
                    checked(authorized, || unsafe { read(item, c"B_UISEL".as_ptr()) })? != 0.0;
                items.push((item, selected));
            }
        }
        Ok(items)
    }

    fn set_item_selected(
        &self,
        item: *mut c_void,
        selected: bool,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        let write: SetItemValue = self.native_api(c"SetMediaItemInfo_Value", authorized)?;
        let value = if selected { 1.0 } else { 0.0 };
        if !checked(authorized, || unsafe {
            write(item, c"B_UISEL".as_ptr(), value)
        })? {
            return Err("REAPER rejected the item selection state".into());
        }
        Ok(())
    }
}
