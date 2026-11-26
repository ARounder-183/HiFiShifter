//! 官方REAPER媒体创建接口：宿主创建item/take/source，PCM与region仍等真实ARA回流授权。
use super::{checked, Guid, HostPlacement, HostVtbl, NativeHost};
use std::ffi::{c_char, c_void, CString};
use std::sync::Arc;

pub(super) type CreateSource = unsafe extern "C" fn(*const c_char, bool) -> *mut c_void;
pub(super) type DestroySource = unsafe extern "C" fn(*mut c_void);
pub(super) type SourceLength = unsafe extern "C" fn(*mut c_void, *mut bool) -> f64;
pub(super) type CreateItem = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
pub(super) type CreateTake = unsafe extern "C" fn(*mut c_void) -> *mut c_void;
pub(super) type TakeInfo =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void) -> *mut c_void;
pub(super) type SetActiveTake = unsafe extern "C" fn(*mut c_void);
pub(super) type CountTakes = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type TakeString =
    unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char, bool) -> bool;
pub(super) type DeleteItem = unsafe extern "C" fn(*mut c_void, *mut c_void) -> bool;
/// `int PCM_Source_BuildPeaks(PCM_source* src, int mode)`：对**源**直接构建峰值。
///
/// 【为什么是源而不是 item】峰值文件属于源；item 只是它的一个使用者。按 item 那条路
/// （`Main_OnCommand(40047)`）要求先选中它，而"改用户的选区再改回来"是在用户脸上动手 ——
/// 有源级入口时就该走源级入口。
pub(super) type BuildPeaks = unsafe extern "C" fn(*mut c_void, i32) -> i32;
pub(super) type FilePicker =
    unsafe extern "C" fn(*mut c_char, *const c_char, *const c_char) -> bool;
pub(super) type MultiPicker = unsafe extern "C" fn(
    i32,
    *const c_char,
    *const c_char,
    *const c_char,
    *mut c_char,
    i32,
) -> bool;
pub(super) type InsertTrack = unsafe extern "C" fn(*mut c_void, i32, i32);
pub(super) type CountTracks = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type GetTrack = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;
pub(super) type AddFx = unsafe extern "C" fn(*mut c_void, *const c_char, bool, i32) -> i32;
pub(super) type DeleteTrack = unsafe extern "C" fn(*mut c_void);
pub(super) type TrackCount = unsafe extern "C" fn(*mut c_void) -> i32;
pub(super) type FxGuid = unsafe extern "C" fn(*mut c_void, i32) -> *const u8;
pub(super) struct NewTrackApi {
    pub insert: InsertTrack,
    pub count: CountTracks,
    pub get: GetTrack,
    pub add_fx: AddFx,
    pub delete: DeleteTrack,
    pub item_count: TrackCount,
    pub fx_count: TrackCount,
    pub fx_guid: FxGuid,
}
pub(super) struct MediaApi {
    pub create_source: CreateSource,
    pub destroy_source: DestroySource,
    pub length: SourceLength,
    pub create_item: CreateItem,
    pub create_take: CreateTake,
    pub take_info: TakeInfo,
    pub delete_item: DeleteItem,
    pub track_guid: Guid,
    pub picker: FilePicker,
    /// `SetActiveTake`：把某个 take 设为 item 的**活动** take。
    ///
    /// 【为什么是 Option】缺了它整条媒体创建链仍要能用（旧宿主 / 未来改名），
    /// 代价只是导入后的 take 可能不是活动 take —— 那一点由下面的自检日志显式说出来，
    /// 不静默降级。
    pub set_active_take: Option<SetActiveTake>,
    /// `CountTakes`：只服务 `[import]` 诊断行（"这个 item 到底有几个 take"）。
    pub count_takes: Option<CountTakes>,
    /// `GetSetMediaItemTakeInfo_String`：给 take 起名（`P_NAME`）。
    pub take_string: Option<TakeString>,
    /// `PCM_Source_BuildPeaks`：让宿主为某个源建峰值（见 [`HostTrackTarget::kick_peaks`]）。
    ///
    /// 【为什么是 Option】缺了它导入照常成立，只是波形要等宿主自己高兴才出现 ——
    /// 那正是本轮要修的现场。缺省时退到 `Item: Build peaks for selected items` 那条
    /// 宿主动作（见 `host_action`），两条都没有才明确降级。
    pub build_peaks: Option<BuildPeaks>,
}
pub(super) struct ExtendedMedia {
    pub picker: Option<MultiPicker>,
    pub tracks: Option<NewTrackApi>,
}

/// 目标轨道由直接parent或已绑定item的真实parent证明，不接受JS地址或活动工程编号。
#[derive(Clone)]
pub(crate) struct HostTrackTarget {
    host: Arc<NativeHost>,
    project: usize,
    track: usize,
    guid: String,
}
/// Source_Create返回的未附着源由调用者拥有；附着后立即撤销此析构责任。
struct OwnedSource {
    pointer: *mut c_void,
    destroy: DestroySource,
}
impl Drop for OwnedSource {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            unsafe { (self.destroy)(self.pointer) };
        }
    }
}
/// 只回滚自己新建且GUID仍相同的item；地址重用或项目销毁时不误删用户数据。
struct CreatedItem {
    target: HostTrackTarget,
    item: usize,
    guid: String,
    state_guid: Option<String>,
    committed: bool,
}
/// 失败时只有仍为空且仅含自己新增FX的轨道才回滚，不能删除重入添加的用户内容。
pub(crate) struct CreatedTrack {
    pub target: HostTrackTarget,
    fx: Option<[u8; 16]>,
    committed: bool,
}
impl CreatedTrack {
    pub(crate) fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for CreatedTrack {
    fn drop(&mut self) {
        if self.committed || self.target.verify(&|| true).is_err() {
            return;
        }
        let Some(api) = self
            .target
            .host
            .extended_media
            .as_ref()
            .and_then(|m| m.tracks.as_ref())
        else {
            return;
        };
        let track = self.target.track as *mut c_void;
        if unsafe { (api.item_count)(track) } != 0 {
            return;
        }
        let count = unsafe { (api.fx_count)(track) };
        let expected = if let Some(guid) = self.fx {
            count == 1 && fx_guid(api, track, 0).is_some_and(|current| current == guid)
        } else {
            count == 0
        };
        if expected {
            unsafe { (api.delete)(track) };
        }
    }
}
fn fx_guid(api: &NewTrackApi, track: *mut c_void, index: i32) -> Option<[u8; 16]> {
    let pointer = unsafe { (api.fx_guid)(track, index) };
    if pointer.is_null() {
        return None;
    }
    let mut value = [0_u8; 16];
    unsafe { std::ptr::copy_nonoverlapping(pointer, value.as_mut_ptr(), 16) };
    Some(value)
}
impl CreatedItem {
    fn verify(&self, authorized: &impl Fn() -> bool) -> Result<(), String> {
        self.target.verify(authorized)?;
        let api = self
            .target
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        if !checked(authorized, || unsafe {
            (api.validate)(
                self.target.project as *mut c_void,
                self.item as *mut c_void,
                c"MediaItem*".as_ptr(),
            )
        })? {
            return Err("created item no longer valid".into());
        }
        let guid = read_guid(api.item_guid, self.item as *mut c_void, authorized)?;
        let write = self
            .target
            .host
            .write
            .as_ref()
            .ok_or("REAPER write API unavailable")?;
        if (guid != self.guid && self.state_guid.as_ref() != Some(&guid))
            || checked(authorized, || unsafe {
                (write.item_track)(self.item as *mut c_void)
            })? as usize
                != self.target.track
        {
            return Err("created item identity/parent changed during import".into());
        }
        Ok(())
    }
}
impl Drop for CreatedItem {
    fn drop(&mut self) {
        if self.committed || std::thread::current().id() != self.target.host.thread {
            return;
        }
        let Some(media) = &self.target.host.media else {
            return;
        };
        let Some(api) = &self.target.host.geometry else {
            return;
        };
        let project = self.target.project as *mut c_void;
        let item = self.item as *mut c_void;
        if self.verify(&|| true).is_ok()
            && unsafe { (api.validate)(project, item, c"MediaItem*".as_ptr()) }
        {
            unsafe { (media.delete_item)(self.target.track as *mut c_void, item) };
        }
    }
}

fn read_guid(
    getter: Guid,
    object: *mut c_void,
    authorized: &impl Fn() -> bool,
) -> Result<String, String> {
    let mut bytes = [0_u8; 64];
    if !checked(authorized, || unsafe {
        getter(object, c"GUID".as_ptr(), bytes.as_mut_ptr().cast(), false)
    })? || bytes[0] != b'{'
        || bytes[37] != b'}'
        || bytes[38] != 0
    {
        return Err("invalid host import GUID".into());
    }
    if !(1..37).all(|i| {
        if [9, 14, 19, 24].contains(&i) {
            bytes[i] == b'-'
        } else {
            bytes[i].is_ascii_hexdigit()
        }
    }) {
        return Err("invalid host import GUID".into());
    }
    String::from_utf8(bytes[..38].to_vec()).map_err(|_| "invalid GUID encoding".into())
}

impl NativeHost {
    /// 媒体剪贴板要求完整创建/删除/历史和状态API；不借clipEditing宣称全部命令可用。
    pub(crate) fn can_clipboard_items(&self) -> bool {
        self.can_import_audio() && self.item_state.is_some() && self.split.is_some()
    }
    /// 只生成宿主GUID和解析状态，不创建对象；整批可在Undo/首个setter之前完成预检。
    pub(crate) fn prepare_copied_item(
        &self,
        chunk: &str,
        position: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<super::RewrittenItem, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.prepare_copied_item(chunk, position, authorized))?;
        }
        self.project(authorized)?;
        let api = self
            .item_state
            .as_ref()
            .ok_or("host item state API unavailable")?;
        super::item_chunk::rewrite_item(chunk, position, || {
            let mut guid = super::item_chunk::NativeGuid::default();
            checked(authorized, || unsafe { (api.generate)(&mut guid) })?;
            let mut bytes = [0_u8; 64];
            checked(authorized, || unsafe {
                (api.stringify)(&guid, bytes.as_mut_ptr().cast())
            })?;
            let end = bytes
                .iter()
                .position(|value| *value == 0)
                .ok_or("unterminated generated GUID")?;
            String::from_utf8(bytes[..end].to_vec())
                .map_err(|_| "generated GUID is not UTF-8".into())
        })
    }
    pub(crate) fn can_import_audio(&self) -> bool {
        self.media.is_some() && self.has_project_history()
    }
    pub(crate) fn can_create_audio_track(&self) -> bool {
        self.can_import_audio()
            && self
                .extended_media
                .as_ref()
                .is_some_and(|m| m.tracks.is_some())
    }
    /// 现代宿主选择器支持多选，返回完整UTF-8路径列表；旧宿主仍回退单文件。
    pub(crate) fn pick_audio_paths(
        &self,
        multiple: bool,
        authorized: &impl Fn() -> bool,
    ) -> Result<Vec<String>, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.pick_audio_paths(multiple, authorized))?;
        }
        self.project(authorized)?;
        // 【为什么选文件也要包住】宿主的选择器是**模态**的：它跑在自己的消息循环里，
        // 期间可能把挂起的事件泵进来（包括 ARA 回调）。守卫让那些回调知道自己在一次
        // 宿主调用里 —— 否则它们会在模态循环内部去等准备 worker，把选择器一起拖死。
        let _call = crate::host::HostCallGuard::enter("host.pick_audio");
        if let Some(picker) = self.extended_media.as_ref().and_then(|m| m.picker) {
            let mut bytes = vec![0_u8; 65536];
            if !checked(authorized, || unsafe {
                picker(
                    if multiple { 2 } else { 1 },
                    c"Import audio into HiFiShifter".as_ptr(),
                    c".wav".as_ptr(),
                    c"Audio files|*.wav;*.flac;*.mp3;*.aiff;*.aif;*.ogg;*.opus;*.m4a|All files|*.*"
                        .as_ptr(),
                    bytes.as_mut_ptr().cast(),
                    bytes.len() as i32,
                )
            })? {
                return Ok(Vec::new());
            }
            let end = bytes
                .iter()
                .position(|b| *b == 0)
                .ok_or("host picker path budget exceeded")?;
            let text =
                std::str::from_utf8(&bytes[..end]).map_err(|_| "host picker path is not UTF-8")?;
            let paths = text
                .split('|')
                .filter(|path| !path.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            if paths.len() > 512
                || paths
                    .iter()
                    .any(|path| !std::path::Path::new(path).is_absolute())
            {
                return Err("host picker returned invalid path batch".into());
            }
            return Ok(paths);
        }
        Ok(self.pick_audio(authorized)?.into_iter().collect())
    }
    /// 新轨身份取所属project创建前后的唯一GUID差集；重入额外增删时失败，不按位置猜新轨。
    ///
    /// `after_track_order` = 插入锚点的 `IP_TRACKNUMBER`（1-based）：新轨道落在它
    /// **之后**。锚点缺失/越界一律退回工程末尾 —— 用户点"添加轨道"却失败是最差的
    /// 体验，而"插在末尾"始终是一个合法且可理解的结果。
    pub(crate) fn create_audio_track(
        self: &Arc<Self>,
        name: &str,
        after_track_order: Option<i32>,
        authorized: &impl Fn() -> bool,
    ) -> Result<CreatedTrack, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self
                .on_host(|| self.create_audio_track(name, after_track_order, authorized))?;
        }
        if !self.can_create_audio_track() {
            return Err("REAPER new-track import API unavailable".into());
        }
        let api = self
            .extended_media
            .as_ref()
            .and_then(|media| media.tracks.as_ref())
            .ok_or("REAPER new-track API unavailable")?;
        let project = self.project(authorized)?;
        let enumerate = || -> Result<Vec<(String, usize)>, String> {
            let token = self.geometry_revision(authorized)?;
            let count = checked(authorized, || unsafe { (api.count)(project) })?;
            if !(0..=10000).contains(&count) {
                return Err("host track count budget exceeded".into());
            }
            let mut result = Vec::new();
            for index in 0..count {
                let track = checked(authorized, || unsafe { (api.get)(project, index) })?;
                if track.is_null() {
                    return Err("host track enumeration changed".into());
                }
                let target = self.track_target(project as usize, track as usize, authorized)?;
                result.push((target.guid, target.track));
            }
            if token != self.geometry_revision(authorized)? {
                return Err("host tracks changed during enumeration".into());
            }
            Ok(result)
        };
        let before = enumerate()?;
        // 【为什么要包住插入】插入轨道会改变工程的轨道集合，宿主的 ARA 文档随之重建
        // （region 重新分配 / `beginEditing`）。这次重入比一次字段写入重得多。
        let _call = crate::host::HostCallGuard::enter("track.create");
        // `IP_TRACKNUMBER` 是 1-based；`InsertTrackInProject` 的 idx 是 0-based，且
        // 插入后新轨道占据 idx。因此"插在 IP_TRACKNUMBER=N 的轨道之后" = `idx = N`。
        let insert_at = match after_track_order {
            Some(order) if order >= 1 && (order as usize) <= before.len() => order as usize,
            _ => before.len(),
        };
        checked(authorized, || unsafe {
            (api.insert)(project, insert_at as i32, 0)
        })?;
        let after = enumerate()?;
        let old = before
            .iter()
            .map(|(guid, _)| guid)
            .collect::<std::collections::HashSet<_>>();
        let added = after
            .iter()
            .filter(|(guid, _)| !old.contains(guid))
            .collect::<Vec<_>>();
        if after.len() != before.len() + 1
            || added.len() != 1
            || before
                .iter()
                .any(|(guid, _)| !after.iter().any(|(known, _)| known == guid))
        {
            return Err("new track creation reentered; undo the host operation to cancel".into());
        }
        let target = self.track_target(project as usize, added[0].1, authorized)?;
        let mut created = CreatedTrack {
            target,
            fx: None,
            committed: false,
        };
        let mut name = CString::new(name)
            .map_err(|_| "track name contains NUL")?
            .into_bytes_with_nul();
        created.target.verify(authorized)?;
        let media = self.media.as_ref().ok_or("REAPER media API unavailable")?;
        if !checked(authorized, || unsafe {
            (media.track_guid)(
                created.target.track as *mut c_void,
                c"P_NAME".as_ptr(),
                name.as_mut_ptr().cast(),
                true,
            )
        })? {
            return Err("host track naming failed".into());
        }
        let index = checked(authorized, || unsafe {
            (api.add_fx)(
                created.target.track as *mut c_void,
                c"VST3:HiFiShifter".as_ptr(),
                false,
                -1000,
            )
        })?;
        if index != 0 {
            return Err("new audio track could not attach HiFiShifter as first FX".into());
        }
        created.fx = fx_guid(api, created.target.track as *mut c_void, index);
        if created.fx.is_none() {
            return Err("new HiFiShifter FX identity unavailable".into());
        }
        created.target.verify(authorized)?;
        Ok(created)
    }
    /// 原生文件选择器仅打开媒体文件；取消不创建item或Undo块，UTF-8路径不经过ANSI转换。
    pub(crate) fn pick_audio(
        &self,
        authorized: &impl Fn() -> bool,
    ) -> Result<Option<String>, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.pick_audio(authorized))?;
        }
        self.project(authorized)?;
        let media = self.media.as_ref().ok_or("REAPER media API unavailable")?;
        let mut bytes = vec![0_u8; 4096];
        if !checked(authorized, || unsafe {
            (media.picker)(
                bytes.as_mut_ptr().cast(),
                c"Import audio into HiFiShifter".as_ptr(),
                c"wav".as_ptr(),
            )
        })? {
            return Ok(None);
        }
        let end = bytes
            .iter()
            .position(|b| *b == 0)
            .ok_or("host file picker path exceeds buffer")?;
        String::from_utf8(bytes[..end].to_vec())
            .map(Some)
            .map_err(|_| "host file path is not UTF-8".into())
    }
    /// 本实例所属的宿主轨道：take FX 下沿 take → item → track，其余沿直接 parent(1)。
    ///
    /// 【为什么 take FX 必须换一条边】take FX 的 `parent(1)` 不保证是 item 的宿主轨
    /// （也可能直接是 `null`，见方案 E0）。而清单作用域、folder 深度这些读数都要求
    /// 一个**真实**的轨道。轨道的唯一可靠来源是 `GetMediaItemTake_Item` →
    /// `GetMediaItem_Track` 这条边。轨道 FX 下保持 `parent(1)` 不变：那是本实例的
    /// 直接父对象，比任何推导都强。
    pub(crate) fn direct_track_target(
        self: &Arc<Self>,
        authorized: &impl Fn() -> bool,
    ) -> Result<HostTrackTarget, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.direct_track_target(authorized))?;
        }
        let project = self.project(authorized)?;
        let pointer = self._interface.0 as *mut c_void;
        let table = unsafe { &**pointer.cast::<*const HostVtbl>() };
        let track = match self.placement() {
            HostPlacement::TakeFx { track, .. } => track as *mut c_void,
            _ => checked(authorized, || unsafe { (table.parent)(pointer, 1) })?,
        };
        if track.is_null() {
            return Err("REAPER instance has no host track".into());
        }
        self.track_target(project as usize, track as usize, authorized)
    }
    pub(super) fn track_target(
        self: &Arc<Self>,
        project: usize,
        track: usize,
        authorized: &impl Fn() -> bool,
    ) -> Result<HostTrackTarget, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.thread {
            return self.on_host(|| self.track_target(project, track, authorized))?;
        }
        if !self.can_import_audio() {
            return Err("REAPER audio import API unavailable".into());
        }
        let api = self
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        if project == 0
            || track == 0
            || !checked(authorized, || unsafe {
                (api.validate)(
                    project as *mut c_void,
                    track as *mut c_void,
                    c"MediaTrack*".as_ptr(),
                )
            })?
        {
            return Err("invalid import target track".into());
        }
        let media = self.media.as_ref().ok_or("REAPER media API unavailable")?;
        let guid = read_guid(media.track_guid, track as *mut c_void, authorized)?;
        let target = HostTrackTarget {
            host: self.clone(),
            project,
            track,
            guid,
        };
        target.verify(authorized)?;
        Ok(target)
    }
}
impl HostTrackTarget {
    /// 本轨道的 `I_FOLDERDEPTH`（`>= 1` 即 folder 父轨）。
    ///
    /// 读不出来时返回 `Err`，调用方据此保持"不知道" —— **不得**把它当作 `0`：
    /// 那会给用户一句可能是错的提示（"你没挂错轨道"），比不提示更坏。
    pub(crate) fn folder_depth(&self, authorized: &impl Fn() -> bool) -> Result<i32, String> {
        self.host
            .track_folder_depth(self.track as *mut c_void, authorized)
    }

    /// 从已重建身份的真实item状态创建；SOURCE/take/FX所有权由宿主state loader管理。
    pub(crate) fn create_copied_item(
        &self,
        state: &super::RewrittenItem,
        authorized: &impl Fn() -> bool,
    ) -> Result<crate::host::geometry::HostClipGeometry, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.create_copied_item(state, authorized))?;
        }
        self.verify(authorized)?;
        if !self.host.can_clipboard_items() {
            return Err("host clip clipboard capability unavailable".into());
        }
        let bytes = CString::new(state.text.as_str()).map_err(|_| "item state contains NUL")?;
        let media = self
            .host
            .media
            .as_ref()
            .ok_or("REAPER media API unavailable")?;
        let api = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let state_api = self
            .host
            .item_state
            .as_ref()
            .ok_or("REAPER item state API unavailable")?;
        let write = self
            .host
            .write
            .as_ref()
            .ok_or("REAPER write API unavailable")?;
        let item = checked(authorized, || unsafe {
            (media.create_item)(self.track as *mut c_void)
        })?;
        if item.is_null() {
            return Err("host copied item creation failed".into());
        }
        let guid = read_guid(api.item_guid, item, authorized)?;
        let mut created = CreatedItem {
            target: self.clone(),
            item: item as usize,
            guid,
            state_guid: Some(state.item_guid.clone()),
            committed: false,
        };
        created.verify(authorized)?;
        if !checked(authorized, || unsafe {
            (state_api.set)(item, bytes.as_ptr(), false)
        })? {
            return Err("host rejected copied item state".into());
        }
        created.verify(authorized)?;
        if read_guid(api.item_guid, item, authorized)? != state.item_guid {
            return Err("copied item identity differs from prepared GUID".into());
        }
        let split = self
            .host
            .split
            .as_ref()
            .ok_or("REAPER split API unavailable")?;
        let take = checked(authorized, || unsafe { (split.active_take)(item) })?;
        if take.is_null() {
            return Err("copied item has no active take".into());
        }
        let geometry = self.host.geometry_for_take(take, authorized)?;
        if geometry.item_id != state.item_guid || !state.take_guids.contains(&geometry.take_id) {
            return Err("copied active take identity differs from prepared state".into());
        }
        created.verify(authorized)?;
        let _call = crate::host::HostCallGuard::enter("item.paste");
        checked(authorized, || unsafe { (write.update)(item) })?;
        checked(authorized, || unsafe { (write.arrange)() })?;
        created.committed = true;
        Ok(geometry)
    }
    /// parent轨道的GUID用于GUI去重，不用名称或轨道序号猜对象。
    pub(crate) fn inventory_guid(&self) -> &str {
        &self.guid
    }
    fn verify(&self, authorized: &impl Fn() -> bool) -> Result<(), String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self.host.on_host(|| self.verify(authorized))?;
        }
        if std::thread::current().id() != self.host.thread {
            return Err("host audio import off UI thread".into());
        }
        let api = self
            .host
            .geometry
            .as_ref()
            .ok_or("host geometry API missing")?;
        for (pointer, kind) in [(self.project, c"ReaProject*"), (self.track, c"MediaTrack*")] {
            if !checked(authorized, || unsafe {
                (api.validate)(
                    self.project as *mut c_void,
                    pointer as *mut c_void,
                    kind.as_ptr(),
                )
            })? {
                return Err("import project/track no longer valid".into());
            }
        }
        let media = self
            .host
            .media
            .as_ref()
            .ok_or("REAPER media API unavailable")?;
        if read_guid(media.track_guid, self.track as *mut c_void, authorized)? != self.guid {
            return Err("import track identity changed".into());
        }
        Ok(())
    }
    /// 新source尚未被任何项目引用；用GetSetMediaItemTakeInfo转移所有权并按官方约定销毁旧源。
    pub(crate) fn import_audio(
        &self,
        path: &str,
        start_sec: f64,
        authorized: &impl Fn() -> bool,
    ) -> Result<String, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.import_audio(path, start_sec, authorized))?;
        }
        self.verify(authorized)?;
        // 【为什么整笔导入都要包住】它在宿主里连续做四件会触发重入的事：建 source、
        // 建 item、建 take、`UpdateItem`+`UpdateArrange`。现场就是在这里挂起的 ——
        // 最后的 `UpdateArrange` 同步触发 ARA `endEditing`，而那次回调去等了 2 s 的
        // 准备 worker。守卫的覆盖范围必须包含**整笔**，否则前面几步触发的回调照样会等。
        let _call = crate::host::HostCallGuard::enter("import_audio");
        if !std::path::Path::new(path).is_absolute()
            || path.len() > 32767
            || !start_sec.is_finite()
            || start_sec < 0.
        {
            return Err("absolute audio path and nonnegative finite start required".into());
        }
        // take 的名字（`P_NAME`）。在 path 被移进 CString 之前取出来：现场看到
        // "take 是空的"，而一个没有名字的 take 在宿主的 take 列表里就是一行空白，
        // 与"音频没挂上"完全无法区分。
        let file_stem = std::path::Path::new(path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| String::from("imported audio"));
        let path = CString::new(path).map_err(|_| "audio path contains NUL")?;
        let media = self
            .host
            .media
            .as_ref()
            .ok_or("REAPER media API unavailable")?;
        let api = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let write = self
            .host
            .write
            .as_ref()
            .ok_or("REAPER write API unavailable")?;
        if !authorized() {
            return Err("import authorization revoked".into());
        }
        let pointer = unsafe { (media.create_source)(path.as_ptr(), true) };
        if pointer.is_null() {
            return Err("REAPER could not open audio source".into());
        }
        let mut source = OwnedSource {
            pointer,
            destroy: media.destroy_source,
        };
        self.verify(authorized)?;
        let mut quarter_notes = false;
        let length = checked(authorized, || unsafe {
            (media.length)(source.pointer, &mut quarter_notes)
        })?;
        if quarter_notes || !length.is_finite() || length <= 0. {
            return Err("source must be nonempty time-based audio, not MIDI".into());
        }
        let item = checked(authorized, || unsafe {
            (media.create_item)(self.track as *mut c_void)
        })?;
        if item.is_null() {
            return Err("host item creation failed".into());
        }
        let guid = read_guid(api.item_guid, item, authorized)?;
        let mut created = CreatedItem {
            target: self.clone(),
            item: item as usize,
            guid: guid.clone(),
            state_guid: None,
            committed: false,
        };
        created.verify(authorized)?;
        let take = checked(authorized, || unsafe { (media.create_take)(item) })?;
        if take.is_null() {
            return Err("host take creation failed".into());
        }
        let take_guid = read_guid(api.take_guid, take, authorized)?;
        let verify = || -> Result<(), String> {
            created.verify(authorized)?;
            if !checked(authorized, || unsafe {
                (api.validate)(
                    self.project as *mut c_void,
                    take,
                    c"MediaItem_Take*".as_ptr(),
                )
            })? || checked(authorized, || unsafe { (api.item)(take) })? != item
                || read_guid(api.take_guid, take, authorized)? != take_guid
            {
                return Err("created take identity/parent changed during import".into());
            }
            Ok(())
        };
        for (name, value) in [
            (c"D_POSITION", start_sec),
            (c"D_LENGTH", length),
            (c"B_LOOPSRC", 0.),
        ] {
            verify()?;
            if !checked(authorized, || unsafe {
                (write.set_item)(item, name.as_ptr(), value)
            })? {
                return Err("host rejected imported item geometry".into());
            }
        }
        let old = checked(authorized, || unsafe {
            (media.take_info)(take, c"P_SOURCE".as_ptr(), std::ptr::null_mut())
        })?;
        verify()?;
        // 此setter无bool失败返回；调用发生即把新源交给宿主，随后租约失效也不能双重释放。
        // 交接出去的指针值留一份：下面的自检要证明"活动 take 的 source 就是它"。
        let handed_over = source.pointer;
        unsafe { (media.take_info)(take, c"P_SOURCE".as_ptr(), source.pointer) };
        source.pointer = std::ptr::null_mut();
        if !old.is_null() {
            unsafe { (media.destroy_source)(old) };
        }
        verify()?;
        // ── take 的写后自检 ────────────────────────────────────────────────
        //
        // 【为什么必须做】现场观察到"item 建出来了、take 却是空的"。此前的日志只能证明
        // **item** 的 GUID —— 而音频真正挂上去的地方是 **take**，它的状态从来没有被读过。
        // 于是"导入失败"与"take 没填"只能靠猜。这三步把 take 变成**可读的事实**：
        // 活动 take 是谁、名字是什么、P_SOURCE 是不是我们刚交出去的那个指针。
        //
        // 【为什么还要 SetActiveTake】新建 item 只有一个 take，但"存在"不等于"活动"：
        // 宿主侧读 take 的路径（渲染、状态块、拷贝）取的是 active take，不设的话它可能
        // 指向别处（或为空），表现为"item 在、音频不在"。
        if let Some(set_active_take) = media.set_active_take {
            checked(authorized, || unsafe { set_active_take(take) })?;
        } else {
            log::warn!(
                "[import] SetActiveTake is unavailable; the created take may not be the active one"
            );
        }
        if let Some(set_string) = media.take_string {
            if let Ok(name) = CString::new(file_stem.clone()) {
                verify()?;
                checked(authorized, || unsafe {
                    set_string(take, c"P_NAME".as_ptr(), name.as_ptr() as *mut c_char, true)
                })?;
            }
        }
        verify()?;
        // 活动 take 优先按宿主自己的答案取（GetActiveTake 已在 SplitApi 里）；
        // 没有这个 API 时退回刚创建的那个 take —— 那正是我们唯一建过的 take。
        let active = match self.host.split.as_ref() {
            Some(split) => checked(authorized, || unsafe { (split.active_take)(item) })?,
            None => take,
        };
        let active_source = if active.is_null() {
            std::ptr::null_mut()
        } else {
            checked(authorized, || unsafe {
                (media.take_info)(active, c"P_SOURCE".as_ptr(), std::ptr::null_mut())
            })?
        };
        let takes = match media.count_takes {
            Some(count) => checked(authorized, || unsafe { count(item) })?,
            None => 1,
        };
        let active_guid = if active.is_null() {
            String::from("-")
        } else {
            read_guid(api.take_guid, active, authorized).unwrap_or_else(|_| String::from("-"))
        };
        let source_state = if active_source.is_null() {
            "none"
        } else if active_source == handed_over {
            "match"
        } else {
            "mismatch"
        };
        if source_state != "match" {
            // 自检不过 = 这一笔导入在宿主侧是坏的。删掉它（CreatedItem 的 Drop 会做），
            // 不留一个"看起来有 item、实际没有音频"的残骸 —— 那正是用户看到的现场。
            log::warn!(
                "[import] item={guid} takes={takes} active={active_guid} source={source_state} len={length:.6} name={file_stem} verdict=rolled-back"
            );
            return Err("imported take did not keep the audio source".into());
        }
        // 一行日志回答"item / take / source 各是什么状态"。区域是否绑定由插件的回执
        // 泵另行记录（那一层才知道 ARA 图），这里只报宿主侧的事实。
        log::info!(
            "[import] item={guid} takes={takes} active={active_guid} source={source_state} len={length:.6} name={file_stem} verdict=ok"
        );
        checked(authorized, || unsafe { (write.update)(item) })?;
        checked(authorized, || unsafe { (write.arrange)() })?;
        created.committed = true;
        Ok(guid)
    }
}

/// `CountTrackMediaItems(MediaTrack*)` / `GetTrackMediaItem(MediaTrack*, int)`。
type CountItems = unsafe extern "C" fn(*mut c_void) -> i32;
type GetItemAt = unsafe extern "C" fn(*mut c_void, i32) -> *mut c_void;

/// 一条轨道上一次导入能枚举的 item 数上限（与清单枚举同一道预算）。
const MAX_ITEMS: i32 = 10_000;

/// `PCM_Source_BuildPeaks` 的 `mode`：**缺才建**。官方取值 0；1 是强制重建。
///
/// 【为什么是 0 而不是 1】导入的源刚被宿主接手，缓存里通常没有它的峰值；0 于是等价于
/// "建"，而"已经有了"时它什么都不做 —— 同一次导入被重试两次也不会白重算一份几十 MB 的
/// `.reapeaks`。
const BUILD_PEAKS_IF_MISSING: i32 = 0;

/// 一次峰值踢的结论。只服务 `[import] peaks` 那一行与用例断言 —— 它不改导入的成败。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PeakKick {
    /// 宿主直接回答了构建进度（非零 = 这一轮已完成，0 = 还有活）。
    Built(i32),
    /// 走了宿主动作那条路（动作没有返回值）。
    Action,
    /// item 已经不在了：用户撤回了这次导入，正常结束。
    Gone,
    /// 没有踢，原因在这里（`host-call` / `repeat` / `no-source` / `no-api`）。
    Skipped(&'static str),
}

impl HostTrackTarget {
    /// 让宿主为**刚导入的 item** 建一次峰值。
    ///
    /// # 三条必须守住的纪律
    ///
    /// 1. **绝不在另一次宿主调用里踢**。REAPER 的峰值作业由它自己的峰值工作者在
    ///    **空闲期**执行；在一个宿主回调里请求它，等于把这一轮空闲用掉了 —— 那正是
    ///    "item 建出来了、波形却是空白，要切窗口才出现"的成因。所以这里第一件事就是
    ///    读 `in_host_call()`，为真直接拒绝：这个不变量的**执行者**是函数本身，不是
    ///    调用方的记性。
    /// 2. **一条 item 只踢一次**（`NativeHost::peaks_kicked` 记最近若干条）。重复请求
    ///    本身无害，但"每次导入一行 `[import] peaks`"这条契约会因为重试而失真。
    /// 3. **找不到 item 不算错误**。用户撤销 / 删掉那条 item 是正常操作，安静返回即可；
    ///    只有"API 在、调用失败"才值得报错。
    pub(crate) fn kick_peaks(
        &self,
        item_guid: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<PeakKick, String> {
        // 宿主调用必须在宿主线程上执行；不在就投递过去并原地等它完成（见 `host::marshal`）。
        if std::thread::current().id() != self.host.thread {
            return self
                .host
                .on_host(|| self.kick_peaks(item_guid, authorized))?;
        }
        if crate::host::in_host_call() {
            return Ok(PeakKick::Skipped("host-call"));
        }
        if self.host.peaks_kicked(item_guid) {
            return Ok(PeakKick::Skipped("repeat"));
        }
        let media = self
            .host
            .media
            .as_ref()
            .ok_or("REAPER media API unavailable")?;
        let Some(item) = self.imported_item(item_guid, authorized)? else {
            self.report_peaks(item_guid, "absent", PeakKick::Gone);
            return Ok(PeakKick::Gone);
        };
        let take = self.imported_take(item, authorized)?;
        let source = match take {
            None => None,
            Some(take) => Some(checked(authorized, || unsafe {
                (media.take_info)(take, c"P_SOURCE".as_ptr(), std::ptr::null_mut())
            })?),
        }
        .filter(|pointer| !pointer.is_null());
        let source_state = if source.is_some() { "ok" } else { "absent" };
        let kick = match (source, take, media.build_peaks) {
            (None, _, _) => PeakKick::Skipped("no-source"),
            (Some(source), _, Some(build)) => {
                // 【为什么这一次要包守卫】它是**新的一次**宿主调用（导入那一次早已结束），
                // 守卫在这里只负责计时与停滞登记 —— 峰值构建是宿主内部排一个作业，很短。
                let _call = crate::host::HostCallGuard::enter("import.peaks");
                PeakKick::Built(checked(authorized, || unsafe {
                    build(source, BUILD_PEAKS_IF_MISSING)
                })?)
            }
            (Some(_), Some(take), None) => {
                let _call = crate::host::HostCallGuard::enter("import.peaks");
                self.host.run_item_action(
                    item,
                    take,
                    &super::host_action::BUILD_PEAKS,
                    authorized,
                )?;
                PeakKick::Action
            }
            // 源在、却没有两个入口中的任何一个：明确降级，不静默。
            (Some(_), None, None) => PeakKick::Skipped("no-api"),
        };
        self.report_peaks(item_guid, source_state, kick);
        if matches!(kick, PeakKick::Built(_) | PeakKick::Action) {
            self.host.remember_peaks_kick(item_guid);
        }
        Ok(kick)
    }

    /// 这条 item 上"峰值该建在哪条 take 的源上"。
    ///
    /// 【为什么不是直接用 active take】`GetActiveTake` 只随 `SplitApi` 一起装（那一束是
    /// "能写才齐备"的能力束），而峰值与写回无关：读清单时宿主只要给出 take 就够了。
    /// 于是这里按"活动 take → 第一条 take"两级取：新建 item 只有一个 take，第一条就是它。
    fn imported_take(
        &self,
        item: *mut c_void,
        authorized: &impl Fn() -> bool,
    ) -> Result<Option<*mut c_void>, String> {
        if let Some(split) = self.host.split.as_ref() {
            let take = checked(authorized, || unsafe { (split.active_take)(item) })?;
            if !take.is_null() {
                return Ok(Some(take));
            }
        }
        let Some(api) = self.host.take_enum.as_ref() else {
            return Ok(None);
        };
        let total = checked(authorized, || unsafe { (api.count)(item) })?;
        if total <= 0 {
            return Ok(None);
        }
        let take = checked(authorized, || unsafe { (api.at)(item, 0) })?;
        Ok((!take.is_null()).then_some(take))
    }

    /// 本轨上 GUID 等于 `item_guid` 的那条 item；没有枚举接口或它已经不在时为 `None`。
    ///
    /// 【为什么按 GUID 而不是"刚创建的那个指针"】`import_audio` 返回之后，item 的所有权
    /// 已经在宿主手里；跨越"返回"再去撞一个裸指针，等于假设宿主没有重建过它。GUID 是
    /// 宿主自己发的身份，按它找回对象是唯一不依赖时间的做法。
    fn imported_item(
        &self,
        item_guid: &str,
        authorized: &impl Fn() -> bool,
    ) -> Result<Option<*mut c_void>, String> {
        let Ok(count) = self
            .host
            .native_api::<CountItems>(c"CountTrackMediaItems", authorized)
        else {
            return Ok(None);
        };
        let Ok(at) = self
            .host
            .native_api::<GetItemAt>(c"GetTrackMediaItem", authorized)
        else {
            return Ok(None);
        };
        let geometry = self
            .host
            .geometry
            .as_ref()
            .ok_or("REAPER geometry API unavailable")?;
        let track = self.track as *mut c_void;
        let total = checked(authorized, || unsafe { count(track) })?;
        if !(0..=MAX_ITEMS).contains(&total) {
            return Err("host track item count budget exceeded".into());
        }
        for index in 0..total {
            let item = checked(authorized, || unsafe { at(track, index) })?;
            if item.is_null() {
                continue;
            }
            if read_guid(geometry.item_guid, item, authorized)
                .ok()
                .as_deref()
                == Some(item_guid)
            {
                return Ok(Some(item));
            }
        }
        Ok(None)
    }

    /// 一行 `[import] peaks` 把三位读数写出来：哪条 item、源在不在、踢的结果是什么。
    ///
    /// 【为什么 result 必须落日志】这条能力是"看不见的"：峰值建没建，界面在下一帧才说
    /// 话，而下一帧可能永远不来（宿主没排作业）。没有这一行，"导入后没有波形"就只能靠
    /// 猜是缓存目录、源坏了、还是根本没请求过。
    fn report_peaks(&self, item_guid: &str, source_state: &str, kick: PeakKick) {
        match kick {
            PeakKick::Built(result) => {
                log::info!("[import] peaks guid={item_guid} source={source_state} result={result}")
            }
            PeakKick::Action => {
                log::info!("[import] peaks guid={item_guid} source={source_state} result=action")
            }
            PeakKick::Gone => {
                log::info!("[import] peaks guid={item_guid} source={source_state} result=gone")
            }
            PeakKick::Skipped(reason) => log::warn!(
                "[import] peaks guid={item_guid} source={source_state} result=skipped:{reason}"
            ),
        }
    }
}
