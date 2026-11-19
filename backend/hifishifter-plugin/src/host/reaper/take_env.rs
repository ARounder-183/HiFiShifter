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
//! 任何走 chunk 的写回在 take FX 下都必然失败。真机现场就是
//! `[pitch] auto take-pitch write failed: REAPER rejected the item state`。
//!
//! 而 **Envelope API 不受这条限制**（同一台机器实测）：`GetTakeEnvelopeByName` 拿到的
//! 包络可以 `SetEnvelopeStateChunk` 整块替换，本插件在线时照样成功，写进去的点立即可读，
//! 空块会让宿主删掉整条包络。所以：
//!
//! * **读**：有包络就读它自己的 chunk（比 item chunk 小几个数量级）；
//! * **写**：有包络就写它；没有包络时先借宿主的动作建一条（[`super::host_action`]），
//!   只有那条路也不可用时才回到 item chunk 的逐字节路径。
//!
//! # 为什么创建要借宿主的动作
//!
//! ARA 2.0 与 REAPER 的扩展 API 都**没有**"新建一条 take 包络"的入口：
//! `GetTakeEnvelopeByName` 在没有包络时返回空指针，而写 chunk 正是上面被拒绝的那条路。
//! 宿主自己的 `Take: Toggle take pitch envelope` 是唯一能创建它的入口 —— 因此本模块的
//! 创建走 [`super::host_action`]，那里负责"按名字解析动作 + 只影响这一条 item/take"。

use super::checked;
use std::ffi::{c_char, c_void};

type GetEnvelopeByName = unsafe extern "C" fn(*mut c_void, *const c_char) -> *mut c_void;
type GetEnvelopeChunk = unsafe extern "C" fn(*mut c_void, *mut c_char, i32, bool) -> bool;
type SetEnvelopeChunk = unsafe extern "C" fn(*mut c_void, *const c_char, bool) -> bool;

impl super::NativeHost {
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

    /// 建一条 take 的 Pitch 包络并返回它。
    ///
    /// 【为什么只借宿主动作】见模块文档：这是唯一能在"ARA 实例在线"时创建包络的入口。
    /// 动作是 toggle —— 只在"确实没有包络"时才走到这里，所以不会把已有的包络删掉。
    pub(super) fn create_take_pitch_envelope(
        &self,
        item: *mut c_void,
        take: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<*mut c_void, String> {
        if !self.on_host_thread() {
            let (item, take) = (item as usize, take as usize);
            return self
                .on_host(|| {
                    self.create_take_pitch_envelope(
                        item as *mut c_void,
                        take as *mut c_void,
                        authorized,
                    )
                    .map(|envelope| envelope as usize)
                })?
                .map(|envelope| envelope as *mut c_void);
        }
        self.run_item_action(
            item,
            take,
            &super::host_action::TAKE_PITCH_ENVELOPE,
            authorized,
        )?;
        self.take_pitch_envelope(take, authorized)?
            .ok_or_else(|| "the host did not create a take pitch envelope".to_string())
    }
}
