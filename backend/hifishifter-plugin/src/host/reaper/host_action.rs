//! 宿主**自己的动作**（main section 命令）的调用入口。
//!
//! # 为什么需要它
//!
//! REAPER 的扩展 API 里有两件事**没有**任何 setter：
//!
//! * 给一条 take **新建**音高包络 —— `GetTakeEnvelopeByName` 在没有包络时返回空指针，
//!   而"新建一条"没有独立入口；
//! * 改 take 的**播放方向** —— 倒放是 `<SOURCE SECTION MODE 3>` 这层**结构**，不是字段
//!   （见 [`super::source_direction`]）。
//!
//! 两条路原本都靠重写 item state chunk 完成，而真机实测（REAPER 7.85）：**一个挂着
//! 在线 ARA FX 的 item，`SetItemStateChunk` 一律返回 `false`** —— 连"原样写回"都拒绝
//! （取证见 [`super::take_env`] 的模块文档）。take FX 放置下，"要改的那个 item"恰好就是
//! 挂着本插件的那个 item，于是这两件事在 take FX 下**必然**失败。
//!
//! 宿主自己当然做得到，它只是不把入口开给扩展。唯一能把"宿主自己的做法"借出来的通道是
//! 它的**动作表**：`Take: Toggle take pitch envelope` 建包络、
//! `Item properties: Toggle take reverse` 换方向。两条都在真机上验证过 —— ARA 实例在线时
//! 照样成功，而且写出的结构与宿主自己 toggle 出来的逐字节一致（`MODE 3` +
//! `SOFFS = D − 正向终点`）。
//!
//! # 三条必须守住的纪律
//!
//! 1. **按名字解析，不硬编码命令号**：动作表属于宿主，名字才是它公开的标识。命令号只在
//!    "名字没对上"时作为退路，而且仍要求宿主认得它（`kbd_getTextFromCmd` 非空）。
//! 2. **动作只作用于选中项**：不改选区就等于把用户整条选区都加上包络 / 翻了方向 ——
//!    那是数据损坏。这里先快照**原来选中的那几条**（`CountSelectedMediaItems` +
//!    `GetSelectedMediaItem`），用一次 `SelectAllMediaItems(false)` 清空、只选中目标，
//!    跑完原样回填。代价与工程规模无关 —— 早先"整工程逐条改选择位"的写法在真机上让
//!    宿主主线程卡了 2 s。
//! 3. **active take 也要对齐**：动作作用于 item 的 *active* take。目标 take 不是 active
//!    时先切过去（`I_CURTAKE`），跑完切回来 —— 否则"给这条 take 建包络"会落到兄弟 take 上。
//!
//! # 线程
//!
//! 与其它宿主入口同一条纪律：不在宿主线程就投递过去并原地等（裸指针过边界换成 `usize`）。
//!
//! # 缓存
//!
//! 动作表是静态的（只有宿主换版本/换语言才变），而逐条枚举 4400 条动作每次都要几千次
//! 宿主调用。解析结果按"退路命令号"缓存在 [`super::NativeHost`] 上，一次解析终身有效。

use super::checked;
use crate::sync::MutexExt;
use std::ffi::{c_char, c_void, CStr};

/// 主动作段的 unique id（`SectionFromUniqueID(0)`）。
///
/// 【为什么是 0】REAPER 的 `kbd_*` 一族要一个 `KbdSectionInfo*`，而 main section 就是
/// unique id 0 那一段。真机实测：`SectionFromUniqueID(0)` 返回非空指针，
/// `kbd_enumerateActions` 用它能枚举到 4390 条动作（含本模块要的两条）。
const MAIN_SECTION: i32 = 0;

/// 动作表的规模上限（REAPER 7.85 自带约 4400 条，翻倍留余量）。
const MAX_ACTIONS: i32 = 20_000;

/// 一次动作前后要恢复的**选中** item 数上限：超过就不动选区，宁可报错。
///
/// 【为什么按"选中"而不是"工程里有多少 item"定界】恢复只碰原本选中的那几条（见
/// `run_item_action`），与工程规模无关 —— 上千条 item 的工程里通常只选了一两条。
const MAX_SELECTION_ITEMS: usize = 20_000;

/// `int kbd_enumerateActions(KbdSectionInfo*, int idx, const char** nameOut)`。
///
/// 【签名逐字取自 reaper.exe 的 API 注册表】`KbdSectionInfo*,int,const char**`：出参是
/// **指向名字的指针**（`const char**`），不是调用方提供的字符缓冲区。早先按
/// "缓冲区 + 长度"实现过一次，于是宿主写回来的指针被当成字符串读，名字永远匹配不上。
type EnumerateActions = unsafe extern "C" fn(*mut c_void, i32, *mut *const c_char) -> i32;
/// `KbdSectionInfo* SectionFromUniqueID(int uniqueID)`。
type SectionFromUniqueId = unsafe extern "C" fn(i32) -> *mut c_void;
/// `const char* kbd_getTextFromCmd(int cmd, KbdSectionInfo* section)`。
type ActionText = unsafe extern "C" fn(i32, *mut c_void) -> *const c_char;
/// `void Main_OnCommandEx(int command, int flag, ReaProject* proj)`。
type RunCommandEx = unsafe extern "C" fn(i32, i32, *mut c_void);
/// `void Main_OnCommand(int command, int flag)`（老宿主）。
type RunCommand = unsafe extern "C" fn(i32, i32);
/// `int CountSelectedMediaItems(ReaProject*)` / `MediaItem* GetSelectedMediaItem(ReaProject*, int)`。
type CountSelected = unsafe extern "C" fn(*mut c_void) -> i32;
type GetSelected = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
/// `void SelectAllMediaItems(ReaProject*, bool)`。
type SelectAll = unsafe extern "C" fn(*mut c_void, bool);
/// `void SetMediaItemSelected(MediaItem*, bool)` / `bool IsMediaItemSelected(MediaItem*)`。
type SetSelected = unsafe extern "C" fn(*mut c_void, bool);
type IsSelected = unsafe extern "C" fn(*mut c_void) -> bool;
type CountTakes = unsafe extern "C" fn(*mut c_void) -> i32;
type GetTake = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
type ItemValue = unsafe extern "C" fn(*mut c_void, *const c_char) -> f64;
type ItemWrite = unsafe extern "C" fn(*mut c_void, *const c_char, f64) -> bool;

/// 一条宿主动作的**身份**：公开名字 + 名字解析不到时的退路命令号。
pub(super) struct HostAction {
    /// 动作表里的名字（宿主自己的标识，英文原文）。
    pub name: &'static CStr,
    /// 名字没对上（本地化/改造过的动作表）时用的命令号；REAPER 7.85 实测值。
    pub fallback_id: i32,
}

/// `Take: Toggle take pitch envelope` —— 建/删一条 take 的 Pitch 包络。
pub(super) const TAKE_PITCH_ENVELOPE: HostAction = HostAction {
    name: c"Take: Toggle take pitch envelope",
    fallback_id: 41_612,
};

/// `Item properties: Toggle take reverse` —— 换 take 的播放方向（SECTION MODE 3）。
pub(super) const TAKE_REVERSE: HostAction = HostAction {
    name: c"Item properties: Toggle take reverse",
    fallback_id: 41_051,
};

/// `Item: Build peaks for selected items` —— 让宿主为选中的 item 建峰值。
///
/// 【为什么需要它】`PCM_Source_BuildPeaks` 在部分宿主版本上缺席；那条路要求"先选中、
/// 再跑动作"。走这里的代价是临时改一次选区，所以它只是**退路**，首选仍是源级入口
/// （见 `super::media::HostTrackTarget::kick_peaks`）。
pub(super) const BUILD_PEAKS: HostAction = HostAction {
    name: c"Item: Build peaks for selected items",
    fallback_id: 40_047,
};

/// 动作表里的名字是否就是 `wanted`。
///
/// 【为什么要处理分隔符】REAPER 的动作描述可能是 `本地化文本 |||||| English`（真机上
/// MIDI/OSC 那一批就是这样）。分隔符之后的那一段才是语言无关的标识，所以只比它；
/// 没有分隔符时整串参与比较。
fn action_matches(text: &CStr, wanted: &CStr) -> bool {
    let text = text.to_string_lossy();
    let text = text.rsplit("||||||").next().unwrap_or(&text).trim();
    text.eq_ignore_ascii_case(wanted.to_string_lossy().trim())
}

impl super::NativeHost {
    /// 在"只有这一条 item、且 active take 就是这一条 take"的临时状态下跑一条宿主动作。
    ///
    /// 成败都会先恢复选区与 active take（模块文档的纪律 2/3）。调用方负责核对**效果**
    /// （包络出现了吗？方向翻过来了吗？）—— 动作号是不可信输入，效果才是判据。
    pub(super) fn run_item_action(
        &self,
        item: *mut c_void,
        take: *mut c_void,
        action: &HostAction,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        if !self.on_host_thread() {
            // 裸指针不是 `Send`：过边界前换成 `usize`（与 `NativeHost::project` 同一条纪律）。
            let (item, take) = (item as usize, take as usize);
            return self.on_host(|| {
                self.run_item_action(item as *mut c_void, take as *mut c_void, action, authorized)
            })?;
        }
        if item.is_null() || take.is_null() {
            return Err("the host item/take is gone".into());
        }
        let command = self.action_command(action, authorized)?;
        // 【只搬"原来选中的那几条"】整工程逐条改选择位是 O(全部 item) 的写操作 —— 真机
        // 现场一次转换让宿主主线程卡了 2 s。这里改成：快照选中的几条（`CountSelectedMediaItems`
        // + `GetSelectedMediaItem`）、一次 `SelectAllMediaItems(false)` 清空、只选中目标，
        // 事后同样一次性回填。代价与工程规模无关。
        let selection = self.selected_items(authorized)?;
        if selection.len() > MAX_SELECTION_ITEMS {
            return Err("too many selected items to safely change the selection".into());
        }
        let target_selected = self.item_selected(item, authorized)?;
        let active = self.active_take_index(item, take, authorized)?;
        let result = (|| -> Result<(), String> {
            self.clear_selection(authorized)?;
            self.set_item_selected(item, true, authorized)?;
            if let Some((index, current)) = active {
                if index != current {
                    self.set_item_f64(item, c"I_CURTAKE", index as f64, authorized)?;
                }
            }
            self.run_command(command, authorized)
        })();
        // 【无论成败都先恢复】让用户的选区/active take 停在动作结束时的状态上，等于
        // "点了一次转换，REAPER 里的选中全变了"。
        let mut restore_error = None;
        if let Some((_, current)) = active {
            if let Err(error) = self.set_item_f64(item, c"I_CURTAKE", current as f64, authorized) {
                restore_error.get_or_insert(error);
            }
        }
        let restore = (|| -> Result<(), String> {
            self.clear_selection(authorized)?;
            if target_selected {
                self.set_item_selected(item, true, authorized)?;
            }
            for pointer in &selection {
                if *pointer != item {
                    self.set_item_selected(*pointer, true, authorized)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = restore {
            restore_error.get_or_insert(error);
        }
        result?;
        if let Some(error) = restore_error {
            return Err(format!("the item selection could not be restored: {error}"));
        }
        Ok(())
    }

    /// 一条宿主动作的命令号（首次解析后缓存）。
    fn action_command(
        &self,
        action: &HostAction,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<i32, String> {
        if let Some(command) = self
            .action_ids
            .lock_or_recover()
            .get(&action.fallback_id)
            .copied()
        {
            return Ok(command);
        }
        let command = self.resolve_action(action, authorized)?;
        self.action_ids
            .lock_or_recover()
            .insert(action.fallback_id, command);
        Ok(command)
    }

    /// 按名字在主动作段里找命令号；找不到才退回登记过的那一个。
    fn resolve_action(
        &self,
        action: &HostAction,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<i32, String> {
        let section = self.main_section(authorized)?;
        let mut scanned = 0;
        if let Ok(enumerate) =
            self.native_api::<EnumerateActions>(c"kbd_enumerateActions", authorized)
        {
            for index in 0..MAX_ACTIONS {
                scanned = index;
                let mut name: *const c_char = std::ptr::null();
                let command = checked(authorized, || unsafe {
                    enumerate(section, index, &mut name)
                })?;
                if command == 0 {
                    break;
                }
                if !name.is_null() && action_matches(unsafe { CStr::from_ptr(name) }, action.name) {
                    // 【为什么要打一行】"借宿主动作"这条路出问题时，报障现场唯一能分辨
                    // "名字没匹配上"与"动作跑了但没效果"的就是这一行。
                    log::warn!(
                        "[ara] host action resolved by name: {:?} -> command {command} \
                         ({} actions scanned)",
                        action.name.to_string_lossy(),
                        scanned + 1
                    );
                    return Ok(command);
                }
            }
        }
        // 【退路】名字没对上（本地化 / 改造过的动作表）时用登记过的命令号，但要求宿主
        // 自己认得它 —— 认不出就如实报错，绝不盲跑一个"可能是别的动作"的号码。
        if self
            .action_text(action.fallback_id, section, authorized)?
            .is_some()
        {
            log::warn!(
                "[ara] host action {:?} is not in the main action table; \
                 using the documented command {}",
                action.name.to_string_lossy(),
                action.fallback_id
            );
            return Ok(action.fallback_id);
        }
        log::warn!(
            "[ara] host action {:?} is not available ({} actions scanned; \
             no documented command {})",
            action.name.to_string_lossy(),
            scanned + 1,
            action.fallback_id
        );
        Err(format!(
            "the host does not offer the action {:?}; enable it once by hand and retry",
            action.name.to_string_lossy()
        ))
    }

    /// 主动作段的段指针。
    fn main_section(
        &self,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<*mut c_void, String> {
        let section: SectionFromUniqueId = self.native_api(c"SectionFromUniqueID", authorized)?;
        let pointer = checked(authorized, || unsafe { section(MAIN_SECTION) })?;
        if pointer.is_null() {
            return Err("the host has no main action section".into());
        }
        Ok(pointer)
    }

    /// 一条命令在主动作段里的显示文本；宿主不认识它时 `None`。
    fn action_text(
        &self,
        command: i32,
        section: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<Option<String>, String> {
        let text: ActionText = self.native_api(c"kbd_getTextFromCmd", authorized)?;
        let pointer = checked(authorized, || unsafe { text(command, section) })?;
        if pointer.is_null() {
            return Ok(None);
        }
        Ok(Some(
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned(),
        ))
    }

    /// 跑一条主动作段的命令（优先按项目定位，老宿主退回全局入口）。
    fn run_command(
        &self,
        command: i32,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        let project = self.project(&|| authorized())?;
        if let Ok(run) = self.native_api::<RunCommandEx>(c"Main_OnCommandEx", authorized) {
            return checked(authorized, || unsafe { run(command, 0, project) });
        }
        let run: RunCommand = self.native_api(c"Main_OnCommand", authorized)?;
        checked(authorized, || unsafe { run(command, 0) })
    }

    /// 工程里此刻**被选中**的那些 item。
    ///
    /// 【为什么用 `CountSelectedMediaItems` + `GetSelectedMediaItem`】它只列选中的，
    /// 代价与选中数成正比。宿主文档提醒这对函数在"两次调用之间增删/重排 item"时不安全 ——
    /// 这里两次调用紧挨着、中间没有任何改动，且结果只用于事后回填选择位。
    fn selected_items(
        &self,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<Vec<*mut c_void>, String> {
        let count: CountSelected = self.native_api(c"CountSelectedMediaItems", authorized)?;
        let at: GetSelected = self.native_api(c"GetSelectedMediaItem", authorized)?;
        let project = self.project(&|| authorized())?;
        let total = checked(authorized, || unsafe { count(project) })?;
        if !(0..=MAX_SELECTION_ITEMS as i32).contains(&total) {
            return Err("the host reported an implausible selection size".into());
        }
        let mut items = Vec::with_capacity(total as usize);
        for index in 0..total {
            let item = checked(authorized, || unsafe { at(project, index) })?;
            if !item.is_null() {
                items.push(item);
            }
        }
        Ok(items)
    }

    /// 这条 item 此刻选没选中。
    fn item_selected(
        &self,
        item: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<bool, String> {
        let is_selected: IsSelected = self.native_api(c"IsMediaItemSelected", authorized)?;
        checked(authorized, || unsafe { is_selected(item) })
    }

    /// 清空整个选择（一条宿主调用，与工程规模无关）。
    fn clear_selection(&self, authorized: &(impl Fn() -> bool + ?Sized)) -> Result<(), String> {
        let select_all: SelectAll = self.native_api(c"SelectAllMediaItems", authorized)?;
        let project = self.project(&|| authorized())?;
        checked(authorized, || unsafe { select_all(project, false) })
    }

    /// 把一条 item 的选中位写到宿主。
    fn set_item_selected(
        &self,
        item: *mut c_void,
        selected: bool,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        let write: SetSelected = self.native_api(c"SetMediaItemSelected", authorized)?;
        checked(authorized, || unsafe { write(item, selected) })
    }

    /// 目标 take 在 item 上的序号，以及宿主此刻报告的 active 序号。
    ///
    /// 读不到多 take 枚举入口时返回 `None` —— 单 take 的常态下不需要切 take。
    fn active_take_index(
        &self,
        item: *mut c_void,
        take: *mut c_void,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<Option<(i32, i32)>, String> {
        let Ok(count) = self.native_api::<CountTakes>(c"GetMediaItemNumTakes", authorized) else {
            return Ok(None);
        };
        let Ok(at) = self.native_api::<GetTake>(c"GetMediaItemTake", authorized) else {
            return Ok(None);
        };
        let total = checked(authorized, || unsafe { count(item) })?;
        if !(0..=64).contains(&total) {
            return Err("the host reported an implausible take count".into());
        }
        let mut index = None;
        for slot in 0..total {
            let pointer = checked(authorized, || unsafe { at(item, slot) })?;
            if pointer == take {
                index = Some(slot);
                break;
            }
        }
        // 【为什么找不到就是错】API 齐全却枚举不到绑定指针 = take 已被删除/重建，
        // 绑定指针失效。此时若照旧返回 `None`，下面的 toggle 动作（倒放/建音高包络，
        // 都落在 item **当前**的 active take 上）会翻错 take —— 倒放是宿主侧的破坏性
        // 变更，编辑失败后的回滚只认绑定 take（它本来就没被动过），翻错的兄弟 take
        // 只能靠宿主自己的 undo。如实报错，让动作根本不跑。
        // `None` 只保留给"宿主根本没有 take 枚举 API"的情形：那种宿主无从切换
        // active take，单 take item（常态）上动作恰好落在绑定 take，是仅有的退路。
        let Some(index) = index else {
            return Err("the bound take is no longer one of the item's takes".into());
        };
        let current = self.item_f64(item, c"I_CURTAKE", authorized)?.round() as i32;
        Ok(Some((index, current)))
    }

    /// 读一个 item 数值字段（`GetMediaItemInfo_Value`）。
    fn item_f64(
        &self,
        item: *mut c_void,
        name: &CStr,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<f64, String> {
        let read: ItemValue = self.native_api(c"GetMediaItemInfo_Value", authorized)?;
        checked(authorized, || unsafe { read(item, name.as_ptr()) })
    }

    /// 写一个 item 数值字段（`SetMediaItemInfo_Value`）。
    fn set_item_f64(
        &self,
        item: *mut c_void,
        name: &CStr,
        value: f64,
        authorized: &(impl Fn() -> bool + ?Sized),
    ) -> Result<(), String> {
        let write: ItemWrite = self.native_api(c"SetMediaItemInfo_Value", authorized)?;
        if !checked(authorized, || unsafe { write(item, name.as_ptr(), value) })? {
            return Err(format!(
                "REAPER rejected the item field {}",
                name.to_string_lossy()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    fn cs(value: &str) -> CString {
        CString::new(value).unwrap()
    }

    /// 名字逐字相同即命中（英文宿主）。
    #[test]
    fn an_exact_name_matches() {
        assert!(action_matches(
            &cs("Take: Toggle take pitch envelope"),
            TAKE_PITCH_ENVELOPE.name
        ));
    }

    /// `本地化 |||||| English` 形态只比分隔符之后那一段。
    ///
    /// 【为什么这条必须有】真机上 MIDI/OSC 那批动作就长这样；按整串比较会把本地化宿主
    /// 上的每一条动作都判成"不存在"。
    #[test]
    fn a_localized_name_matches_on_its_english_tail() {
        let localized = cs("项目：切换 take 音高包络 |||||| Take: Toggle take pitch envelope");
        assert!(action_matches(&localized, TAKE_PITCH_ENVELOPE.name));
        assert!(!action_matches(&localized, TAKE_REVERSE.name));
    }

    /// 完全不同的名字不许命中 —— 拿错动作号就是跑错动作。
    #[test]
    fn a_different_name_never_matches() {
        assert!(!action_matches(
            &cs("Take: Toggle take volume envelope"),
            TAKE_PITCH_ENVELOPE.name
        ));
    }
}
